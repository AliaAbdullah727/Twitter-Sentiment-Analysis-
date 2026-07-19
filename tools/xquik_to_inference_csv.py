from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


TEXT_FIELDS = (
    "text",
    "full_text",
    "tweetText",
    "tweet_text",
    "content",
    "body",
)
STATUS_FIELDS = ("status", "review_status", "approval_status")
APPROVED_STATUSES = {"approved", "published", "ready", "reviewed", "selected"}
BLOCKED_STATUSES = {
    "draft",
    "needs_review",
    "not_approved",
    "not_reviewed",
    "pending",
    "rejected",
    "unreviewed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert reviewed Xquik or TweetClaw exports into notebook inference input."
    )
    parser.add_argument("--input", required=True, help="Path to a CSV, JSON, or JSONL export.")
    parser.add_argument("--output", required=True, help="Path to the output CSV with a text column.")
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include rows without an explicit approved/reviewed status.",
    )
    return parser.parse_args()


def normalize_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def row_is_reviewed(row: dict[str, Any], include_unreviewed: bool) -> bool:
    statuses = [normalize_status(row.get(field)) for field in STATUS_FIELDS]
    statuses = [status for status in statuses if status]
    if not statuses:
        return include_unreviewed
    return any(status in APPROVED_STATUSES for status in statuses) and not any(
        status in BLOCKED_STATUSES for status in statuses
    )


def first_string(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def flatten_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("items", "tweets", "results", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def read_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        return rows
    return flatten_items(json.loads(text))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        return read_json(path)
    return read_csv(path)


def convert(input_path: Path, output_path: Path, include_unreviewed: bool) -> int:
    rows = read_rows(input_path)
    texts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row_is_reviewed(row, include_unreviewed):
            continue
        text = first_string(row, TEXT_FIELDS)
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text"])
        writer.writeheader()
        for text in texts:
            writer.writerow({"text": text})
    return len(texts)


def main() -> None:
    args = parse_args()
    count = convert(Path(args.input), Path(args.output), args.include_unreviewed)
    print(f"Wrote {count} reviewed rows to {args.output}")


if __name__ == "__main__":
    main()
