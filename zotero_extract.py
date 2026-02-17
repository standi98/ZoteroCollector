"""
zotero_extract.py
-----------------
Reads your Zotero SQLite database directly and extracts structured data
from Better Notes extraction notes into a pandas DataFrame.

Fields extracted per paper: Age, Name, Hobbies
(edit EXTRACT_FIELDS below to match your actual field names)

Usage:
    # All items with extraction notes:
    python zotero_extract.py

    # Filter to a specific collection (case-insensitive):
    python zotero_extract.py --collection "My Collection"

    # Or import and use in a notebook:
    from zotero_extract import load_zotero_data
    df = load_zotero_data(collection="My Collection")
"""

import sqlite3
import re
import shutil
import argparse
import os
import pandas as pd
from pathlib import Path
from html.parser import HTMLParser
from dotenv import load_dotenv

# ─── CONFIG — loaded from .env ────────────────────────────────────────────────

load_dotenv()

_db_path = os.getenv("ZOTERO_DB_PATH", "").strip()
ZOTERO_DB = Path(_db_path) if _db_path else Path.home() / "Zotero" / "zotero.sqlite"

_fields = os.getenv("EXTRACT_FIELDS", "Age,Name,Hobbies").strip()
EXTRACT_FIELDS = [f.strip() for f in _fields.split(",") if f.strip()]

_collection = os.getenv("DEFAULT_COLLECTION", "").strip()
DEFAULT_COLLECTION = _collection if _collection else None

OUTPUT_CSV = Path(os.getenv("OUTPUT_CSV", "zotero_extracted.csv").strip())

# ─────────────────────────────────────────────────────────────────────────────


class HTMLTextExtractor(HTMLParser):
    """Strips HTML tags and returns plain text, preserving list item boundaries."""
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def handle_starttag(self, tag, attrs):
        # Add a space before block-level tags so fields don't run together
        if tag in ("li", "p", "br", "div", "h1", "h2", "h3", "tr", "td"):
            self._parts.append(" ")

    def get_text(self):
        return " ".join(self._parts)


def html_to_text(html: str) -> str:
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def extract_fields(note_html: str, fields: list[str]) -> dict:
    """
    Given raw HTML from a Zotero note, extract the value for each field.
    Handles both:
      - Bullet list format:  * Age: 25
      - Bold format:         <b>Age:</b> 25
    Returns a dict {field: value} with empty string for missing fields.
    """
    text = html_to_text(note_html)
    result = {}
    for field in fields:
        # Match "FieldName: value" — greedy until next field or end of meaningful content
        pattern = re.compile(
            rf"{re.escape(field)}\s*:\s*([^\n:]+?)(?=\s+\w+\s*:|$)",
            re.IGNORECASE
        )
        match = pattern.search(text)
        result[field] = match.group(1).strip() if match else ""
    return result


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


def get_collection_item_ids(conn: sqlite3.Connection, collection_name: str) -> set[int]:
    """
    Returns the set of itemIDs belonging to the named collection (top-level only).
    Raises ValueError if the collection name is not found.
    Prints all available collection names if there's a mismatch.
    """
    # Find the collection — case-insensitive match
    rows = conn.execute(
        "SELECT collectionID, collectionName FROM collections"
    ).fetchall()

    matched = [r for r in rows if r["collectionName"].lower() == collection_name.lower()]

    if not matched:
        available = sorted(r["collectionName"] for r in rows)
        raise ValueError(
            f"Collection '{collection_name}' not found.\n"
            f"Available collections:\n  " + "\n  ".join(available)
        )

    collection_id = matched[0]["collectionID"]

    item_rows = conn.execute(
        "SELECT itemID FROM collectionItems WHERE collectionID = ?",
        (collection_id,)
    ).fetchall()

    return {r["itemID"] for r in item_rows}


def load_zotero_data(
    db_path: Path = ZOTERO_DB,
    fields: list[str] = EXTRACT_FIELDS,
    collection: str | None = DEFAULT_COLLECTION
) -> pd.DataFrame:
    """
    Main function. Connects to Zotero SQLite (via a safe read-only copy),
    joins items + notes + creators, parses extracted fields, returns DataFrame.

    Args:
        db_path:    Path to zotero.sqlite
        fields:     List of field names to extract from notes
        collection: Optional collection name to filter by (case-insensitive).
                    Pass None to extract all items.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Zotero database not found at: {db_path}\n"
            "Edit the ZOTERO_DB path in this script."
        )

    # Work on a copy — never modify the live Zotero database
    tmp_db = Path("/tmp/zotero_readonly_copy.sqlite")
    shutil.copy2(db_path, tmp_db)
    print(f"Working on a copy of: {db_path}")

    conn = sqlite3.connect(tmp_db)
    conn.row_factory = sqlite3.Row

    # ── Resolve collection filter ─────────────────────────────────────────────
    collection_item_ids: set[int] | None = None
    if collection:
        collection_item_ids = get_collection_item_ids(conn, collection)
        print(f"Filtering to collection '{collection}' — {len(collection_item_ids)} items found")

    # ── Pull all parent items with their notes ────────────────────────────────
    query = """
        SELECT
            i.itemID,
            i.key           AS zotero_key,
            idv_title.value AS title,
            idv_year.value  AS year,
            idv_extra.value AS extra,
            n.note          AS note_html
        FROM items i
        -- Title
        LEFT JOIN itemData id_title
            ON id_title.itemID = i.itemID
            AND id_title.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
        LEFT JOIN itemDataValues idv_title
            ON idv_title.valueID = id_title.valueID
        -- Year
        LEFT JOIN itemData id_year
            ON id_year.itemID = i.itemID
            AND id_year.fieldID = (
                SELECT fieldID FROM fields
                WHERE fieldName IN ('date', 'year')
                LIMIT 1
            )
        LEFT JOIN itemDataValues idv_year
            ON idv_year.valueID = id_year.valueID
        -- Extra (for citekey)
        LEFT JOIN itemData id_extra
            ON id_extra.itemID = i.itemID
            AND id_extra.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'extra')
        LEFT JOIN itemDataValues idv_extra
            ON idv_extra.valueID = id_extra.valueID
        -- Child notes
        INNER JOIN itemNotes n
            ON n.parentItemID = i.itemID
        -- Exclude deleted items
        WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
    """

    rows = conn.execute(query).fetchall()

    # ── Pull creators separately (one-to-many) ────────────────────────────────
    creators_query = """
        SELECT
            ic.itemID,
            c.lastName,
            c.firstName,
            ic.orderIndex
        FROM itemCreators ic
        JOIN creators c ON c.creatorID = ic.creatorID
        ORDER BY ic.itemID, ic.orderIndex
    """
    creator_rows = conn.execute(creators_query).fetchall()
    conn.close()

    # Build a lookup: itemID → [lastName, ...]
    creators_by_item: dict[int, list[str]] = {}
    for cr in creator_rows:
        creators_by_item.setdefault(cr["itemID"], []).append(cr["lastName"])

    # ── Parse each row ────────────────────────────────────────────────────────
    records = []
    seen_items = set()  # In case an item has multiple notes, use the first match

    for row in rows:
        item_id = row["itemID"]

        # Apply collection filter if specified
        if collection_item_ids is not None and item_id not in collection_item_ids:
            continue

        note_html = row["note_html"] or ""

        # Only process notes that look like our extraction notes
        if not any(f.lower() in note_html.lower() for f in fields):
            continue

        if item_id in seen_items:
            continue
        seen_items.add(item_id)

        extracted = extract_fields(note_html, fields)

        # Only include rows where at least one field was found
        if not any(extracted.values()):
            continue

        creators = creators_by_item.get(item_id, [])
        year = (row["year"] or "")[:4]  # Zotero sometimes stores full date
        citekey = get_citekey(row["extra"], creators, year)

        record = {
            "citekey":     citekey,
            "title":       row["title"] or "",
            "year":        year,
            "zotero_key":  row["zotero_key"],
            "authors":     "; ".join(creators),
        }
        record.update(extracted)
        records.append(record)

    df = pd.DataFrame(records)

    # Reorder columns: metadata first, then extracted fields
    meta_cols = ["citekey", "title", "year", "authors", "zotero_key"]
    field_cols = [f for f in fields if f in df.columns]
    df = df[meta_cols + field_cols]

    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Better Notes data from Zotero SQLite into a CSV."
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=DEFAULT_COLLECTION,
        help=(
            "Name of the Zotero collection to filter by (case-insensitive). "
            "Omit to extract all items. "
            f"Default: {DEFAULT_COLLECTION!r}"
        )
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="Print all available collection names and exit."
    )
    args = parser.parse_args()

    # ── List collections mode ─────────────────────────────────────────────────
    if args.list_collections:
        tmp_db = Path("/tmp/zotero_readonly_copy.sqlite")
        shutil.copy2(ZOTERO_DB, tmp_db)
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT collectionName FROM collections ORDER BY collectionName"
        ).fetchall()
        conn.close()
        print("Available collections:")
        for r in rows:
            print(f"  {r['collectionName']}")
        raise SystemExit(0)

    # ── Normal extraction ─────────────────────────────────────────────────────
    df = load_zotero_data(collection=args.collection)

    label = f"collection '{args.collection}'" if args.collection else "all collections"
    print(f"\n✓ Extracted {len(df)} papers from {label}\n")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved to {OUTPUT_CSV}")