#!/usr/bin/env python3
"""
Find parent ID(s) for a given child ID from a JSON mapping:
{
  "4699511": ["4699513", "4699515", ...],
  "4699520": ["4699521", ...],
  ...
}
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def build_reverse_index(parent_to_children: dict[str, list[str]]) -> dict[str, set[str]]:
    """Build child -> {parent,...} reverse mapping."""
    rev = defaultdict(set)
    for parent, children in parent_to_children.items():
        # normalize string IDs
        p = str(parent)
        if not isinstance(children, (list, tuple)):
            continue
        for child in children:
            rev[str(child)].add(p)
    return rev


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve parent ID(s) for a given child ID from a JSON file."
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to JSON file with mapping {parent_id: [child_ids...]}"
    )
    parser.add_argument(
        "--id",
        required=True,
        help="Child ID to look up (string or number)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON instead of plain text."
    )
    args = parser.parse_args()

    json_path = Path(args.file)
    if not json_path.exists():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(data, dict):
        print("Error: JSON root must be an object {parent: [children...]}", file=sys.stderr)
        sys.exit(2)

    rev = build_reverse_index(data)

    child_id = str(args.id)
    parents = sorted(rev.get(child_id, []), key=str)

    if args.json:
        # Emit a tiny JSON payload for easy scripting
        out = {"child": child_id, "parents": parents}
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        if not parents:
            print(f"(no parent found for child {child_id})")
        elif len(parents) == 1:
            print(parents[0])
        else:
            # If multiple parents exist, print all on one line (space-separated)
            print(" ".join(parents))


if __name__ == "__main__":
    main()
