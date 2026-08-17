#!/usr/bin/env python3
"""Read-only Gamma keyset walk probe.

This is a durability check for the measurement that retired the old "Gamma offset hides the PM
universe" product hook. It records lower bounds honestly: if `--max-pages` is reached with a
cursor still outstanding, the result is incomplete by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import venues  # noqa: E402


def walk(limit: int, max_pages: int | None, progress_every: int) -> dict:
    started = time.time()
    cursor = None
    pages = 0
    events = 0
    seen = set()
    while True:
        data = venues.get(venues.pm_keyset_url(limit=limit, after_cursor=cursor))
        batch = data.get("events") or []
        pages += 1
        events += len(batch)
        seen.update(str(e.get("id")) for e in batch if e.get("id") is not None)
        cursor = data.get("next_cursor")
        elapsed = time.time() - started
        if progress_every and pages % progress_every == 0:
            print(f"{pages:4d} pages, {events} events, {len(seen)} unique, {elapsed:.0f}s",
                  flush=True)
        if not cursor or not batch:
            return {
                "pages": pages,
                "events": events,
                "unique": len(seen),
                "cursor_outstanding": False,
                "elapsed_sec": elapsed,
            }
        if max_pages is not None and pages >= max_pages:
            return {
                "pages": pages,
                "events": events,
                "unique": len(seen),
                "cursor_outstanding": True,
                "elapsed_sec": elapsed,
            }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=400,
                    help="Page cap. Use --uncapped to walk until Gamma exhausts the cursor.")
    ap.add_argument("--uncapped", action="store_true")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--out", help="Optional JSON output path for the summary only.")
    args = ap.parse_args()

    result = walk(
        limit=args.limit,
        max_pages=None if args.uncapped else args.max_pages,
        progress_every=args.progress_every,
    )
    state = "OUTSTANDING" if result["cursor_outstanding"] else "EXHAUSTED"
    print(
        f"RESULT: {result['events']} events over {result['pages']} pages | "
        f"{result['unique']} unique | cursor {state} | {result['elapsed_sec']:.0f}s"
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
