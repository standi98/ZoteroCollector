import sqlite3
import re
import shutil
import argparse
import yaml
import os
import pandas as pd
from pathlib import Path
from html.parser import HTMLParser
from dotenv import load_dotenv


# ─── CONFIG — loaded from config.yaml and .env ────────────────────────────────────────────────


load_dotenv()  # Load environment variables from .env
ZOTERO_DB = Path(os.getenv("ZOTERO_DB_PATH", "").strip())
assert ZOTERO_DB.exists(), "ZOTERO_DB_PATH environment variable not set. Please set it in your .env file, or ensure it points to zotero.sqlite"


with open("config.yaml") as f:
    config = yaml.safe_load(f)
# File information
OUTPUT_CSV   = Path(config["paths"]["output_csv"])
COLLECTION   = config["zotero_information"]["collection_name"]

# Schema: {fieldName: type}  e.g. {"Accuracy": "nested", "Year": "flat"}
SCHEMA: dict[str, str] = config["extraction_schema"]


# ─────────────────────────────────────────────────────────────────────────────


class HTMLTextExtractor(HTMLParser):
    """
    Converts Zotero note HTML to indented plain text.
    Handles Zotero's habit of wrapping list item text in <p> tags
    when the item contains children.
    """
    def __init__(self):
        super().__init__()
        self._lines   = []
        self._current = ""
        self._depth   = 0
        self._in_li   = False

    def handle_starttag(self, tag, attrs):
        if tag == "ul":
            self._depth += 1
        elif tag == "li":
            # Flush previous item if any
            if self._current.strip():
                self._lines.append(self._current)
            self._current = "  " * self._depth + "- "
            self._in_li   = True
        # <p> inside <li> is just inline text — don't flush, don't indent

    def handle_endtag(self, tag):
        if tag == "ul":
            self._depth -= 1
        elif tag == "li":
            # Flush the completed item
            if self._current.strip():
                self._lines.append(self._current)
            self._current = ""
            self._in_li   = False
        # Ignore </p> — content already accumulated into _current

    def handle_data(self, data):
        text = data.strip()
        if text:
            self._current += text

    def get_text(self):
        if self._current.strip():
            self._lines.append(self._current)
        return "\n".join(self._lines)

def html_to_text(html: str) -> str:
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ─── Field parsers ────────────────────────────────────────────────────────────

def parse_flat(lines: list[str], field: str) -> str:
    """
    Extracts a single value from a line like:
        - **Year:** 2024
    Returns the value as a string, or "" if not found.
    """
    pattern = re.compile(rf"\*{{0,2}}{re.escape(field)}\*{{0,2}}\s*:\s*(.+)", re.IGNORECASE)
    for line in lines:
        match = pattern.search(line)
        if match:
            return match.group(1).strip()
    return ""


def parse_list(lines: list[str], field: str) -> str:
    """
    Extracts a comma-separated list from a line like:
        - **Features:** Speed, Depth, Wind
    Returns the raw comma-separated string, or "" if not found.
    """
    # List fields are stored the same way as flat — just return as-is
    return parse_flat(lines, field)


def parse_nested(lines: list[str], field: str) -> dict[str, str]:
    """
    Recursively extracts nested sub-key/value pairs.
    A line with no value after the colon is treated as a parent node.
    Returns a flat dict with keys joined by '_':
        {"Neural_NN": "50", "Neural_CNN": "60", "Boosting_XGBoost": "78"}
    """
    header_pattern = re.compile(rf"\*{{0,2}}{re.escape(field)}\*{{0,2}}\s*:\s*$", re.IGNORECASE)

    # Find the header line and its indentation depth
    header_idx   = None
    header_depth = 0
    for i, line in enumerate(lines):
        if header_pattern.search(line.strip()):
            header_idx   = i
            header_depth = len(line) - len(line.lstrip())
            break

    if header_idx is None:
        return {}

    # Collect all lines that belong to this block (deeper indentation)
    block = []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        depth = len(line) - len(line.lstrip())
        if depth <= header_depth:
            break   # back to same or higher level — block is done
        block.append(line)

    return _parse_block(block)


def _parse_block(lines: list[str], prefix: str = "") -> dict[str, str]:
    result     = {}
    i          = 0
    leaf_pat   = re.compile(r"[-*]\s+([^:]+):\s+(.+)")
    parent_pat = re.compile(r"[-*]\s+([^:]+):\s*$")

    while i < len(lines):
        line     = lines[i]
        stripped = line.strip()
        depth    = len(line) - len(line.lstrip())

        leaf = leaf_pat.match(stripped)
        if leaf:
            subkey = leaf.group(1).strip()
            key    = f"{prefix}{subkey}" if prefix else subkey
            result[key] = leaf.group(2).strip()
            i += 1
            continue

        parent = parent_pat.match(stripped)
        if parent:
            subkey   = parent.group(1).strip()
            new_prefix = f"{prefix}{subkey}_" if prefix else f"{subkey}_"

            children = []
            i += 1
            while i < len(lines):
                child_depth = len(lines[i]) - len(lines[i].lstrip())
                if child_depth <= depth:
                    break
                children.append(lines[i])
                i += 1

            result.update(_parse_block(children, prefix=new_prefix))
            continue

        i += 1

    return result


def extract_fields(note_html: str, schema: dict[str, str]) -> dict[str, any]: # type: ignore
    """
    Given raw note HTML and a schema dict, extract all fields.
    Returns a flat dict ready for a DataFrame row, with nested fields
    expanded to  FieldName_SubKey  columns.
    """
    text  = html_to_text(note_html)
    lines = text.splitlines()
    result = {}

    for field, ftype in schema.items():
        if ftype == "flat":
            result[field] = parse_flat(lines, field)

        elif ftype == "list":
            result[field] = parse_list(lines, field)

        elif ftype == "nested":
            sub = parse_nested(lines, field)
            for subkey, val in sub.items():
                result[f"{field}_{subkey}"] = val
            # If nothing found, leave no columns (they'll be NaN after pd.DataFrame)

        else:
            raise ValueError(f"Unknown field type '{ftype}' for field '{field}' in config.yaml")

    return result


# ─── Zotero helpers ───────────────────────────────────────────────────────────

def get_citekey(extra: str, creators: list, year: str) -> str:
    """
    Tries to get the Better BibTeX citekey from the Extra field.
    Falls back to LastName + Year if not found.
    """
    if extra:
        match = re.search(r"(?:citation key|citekey)\s*:\s*(\S+)", extra, re.IGNORECASE)
        if match:
            return match.group(1)
    last_name = creators[0] if creators else "Unknown"
    return f"{last_name}{year}"


# ─── Main extraction ──────────────────────────────────────────────────────────

def extract_zotero_data(
    db_path: Path = ZOTERO_DB,
    schema: dict[str, str] = SCHEMA,
    collection: str | None = COLLECTION
) -> pd.DataFrame:
    """
    Connects to Zotero SQLite (via a safe read-only copy), finds all Data Entry
    notes, parses fields according to the schema, and returns a DataFrame.

    Args:
        db_path:    Path to zotero.sqlite
        schema:     Dict of {fieldName: type} — see config.yaml
        collection: Collection name to filter by (case-insensitive). None = all items.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Zotero database not found at: {db_path}")

    # Work on a safe copy
    tmp_db = Path("/tmp/zotero_readonly_copy.sqlite")
    shutil.copy2(db_path, tmp_db)
    print(f"Working on a copy of: {db_path}")
    del db_path

    conn = sqlite3.connect(tmp_db)
    conn.row_factory = sqlite3.Row

    # ── SQL query ─────────────────────────────────────────────────────────────
    query = """
        SELECT
            i.itemID,
            i.key           AS zotero_key,
            idv_title.value AS title,
            idv_year.value  AS year,
            idv_extra.value AS extra,
            n.note          AS note_html
        FROM items i
        LEFT JOIN itemData id_title
            ON id_title.itemID = i.itemID
            AND id_title.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
        LEFT JOIN itemDataValues idv_title
            ON idv_title.valueID = id_title.valueID
        LEFT JOIN itemData id_year
            ON id_year.itemID = i.itemID
            AND id_year.fieldID = (
                SELECT fieldID FROM fields
                WHERE fieldName IN ('date', 'year')
                LIMIT 1
            )
        LEFT JOIN itemDataValues idv_year
            ON idv_year.valueID = id_year.valueID
        LEFT JOIN itemData id_extra
            ON id_extra.itemID = i.itemID
            AND id_extra.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'extra')
        LEFT JOIN itemDataValues idv_extra
            ON idv_extra.valueID = id_extra.valueID
        INNER JOIN itemNotes n
            ON n.parentItemID = i.itemID
            AND n.itemID NOT IN (SELECT itemID FROM deletedItems)
        WHERE 1=1
    """

    if collection:
        collection_id_row = conn.execute(
            "SELECT collectionID FROM collections WHERE LOWER(collectionName) = LOWER(?)",
            (collection,)
        ).fetchone()

        if not collection_id_row:
            available = [r[0] for r in conn.execute("SELECT collectionName FROM collections").fetchall()]
            raise ValueError(
                f"Collection '{collection}' not found.\n"
                f"Available collections:\n  " + "\n  ".join(sorted(available))
            )

        rows = conn.execute(query + """
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            AND i.itemID IN (
                SELECT itemID FROM collectionItems WHERE collectionID = ?
            )
        """, (collection_id_row["collectionID"],)).fetchall()
    else:
        rows = conn.execute(query + """
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        """).fetchall()

    # Pull creators separately (one-to-many)
    creator_rows = conn.execute("""
        SELECT ic.itemID, c.lastName, ic.orderIndex
        FROM itemCreators ic
        JOIN creators c ON c.creatorID = ic.creatorID
        ORDER BY ic.itemID, ic.orderIndex
    """).fetchall()
    conn.close()

    # ── Process in Python ─────────────────────────────────────────────────────

    creators_by_item: dict[int, list[str]] = {}
    for cr in creator_rows:
        creators_by_item.setdefault(cr["itemID"], []).append(cr["lastName"])

    field_names = list(schema.keys())
    records = []
    seen_items = set()

    for row in rows:
        item_id   = row["itemID"]
        note_html = row["note_html"] or ""

        # Only process notes that look like Data Entry notes
        if not any(f.lower() in note_html.lower() for f in field_names):
            continue

        if item_id in seen_items:
            continue
        seen_items.add(item_id)

        extracted = extract_fields(note_html, schema)

        if not any(v for v in extracted.values()):
            continue

        creators = creators_by_item.get(item_id, [])
        year     = (row["year"] or "")[:4]
        citekey  = get_citekey(row["extra"], creators, year)

        record = {
            "citekey":    citekey,
            "title":      row["title"] or "",
            "year":       year,
            "authors":    "; ".join(creators),
            "zotero_key": row["zotero_key"],
        }
        record.update(extracted)
        records.append(record)

    df = pd.DataFrame(records)

    # Reorder: metadata first, then extracted columns
    meta_cols  = ["citekey", "title", "year", "authors", "zotero_key"]
    extra_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + extra_cols]

    return df



# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Better Notes data from Zotero SQLite into a CSV."
    )
    parser.add_argument("--collection", "-c",
        type=str, default=COLLECTION,
        help=f"Zotero collection to filter by. Default: {COLLECTION!r}"
    )
    parser.add_argument("--db", "-d",
        type=str, default=str(ZOTERO_DB),
        help=f"Path to zotero.sqlite. Default: {ZOTERO_DB}"
    )
    parser.add_argument("--output", "-o",
        type=str, default=str(OUTPUT_CSV),
        help=f"Output CSV path. Default: {OUTPUT_CSV}"
    )
    args = parser.parse_args()

    df = extract_zotero_data(
        collection=args.collection,
        db_path=Path(args.db),
    )

    label = f"collection '{args.collection}'" if args.collection else "all collections"
    print(f"\n✓ Extracted {len(df)} papers from {label}\n")
    print(df.to_string(index=False))

    df.to_csv(args.output, index=False)
    print(f"\n✓ Saved to {args.output}")