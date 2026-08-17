#!/usr/bin/env python3
"""Read-only intra-Polymarket binary complement check.

A Yes/No binary market is exhaustive by construction: buying one YES share and one NO share
redeems for exactly $1. So a live CLOB state where YES ask + NO ask, size-walked and net of
taker fees, costs less than $1 is either a crossed-book anomaly or a scanner/book-parsing bug.

This is deliberately narrower than the group-coherence work. There is no dropped-leg or
exhaustiveness judgment: a single binary's two CLOB tokens are the whole universe.

Uses `data/universe_snapshot.json` from the PM x Kalshi pass for market/token discovery, then
fetches live CLOB books for prices. Gamma prices are never used for pricing.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probe_overlap as P  # noqa: E402
import venues  # noqa: E402
from walk_hedge import pm_ask_ladder, walk  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SNAPSHOT = os.path.join(DATA, "universe_snapshot.json")


@dataclass(frozen=True)
class Market:
    market_id: str
    slug: str
    question: str
    yes_token: str
    no_token: str
    fee_rate: float
    gamma_ask_sum: float
    end_date: str | None = None


def load_markets(path: str = SNAPSHOT) -> list[Market]:
    """Load PM Yes/No binaries from the cached universe snapshot.

    Pricing still comes from CLOB. `gamma_ask_sum` is only a deterministic priority key so the
    bounded run looks first where a complement anomaly would be most likely if Gamma's stale
    marks were pointing at anything real.
    """
    with open(path, encoding="utf-8") as fh:
        snap = json.load(fh)
    rows = []
    now = dt.datetime.now(dt.timezone.utc)
    for r in P.pm_rows(snap):
        if not (r.get("yes_token") and r.get("no_token")):
            continue
        end = P.iso(r.get("end_date"))
        if end and end < now:
            continue
        rows.append(Market(
            market_id=str(r["id"]),
            slug=str(r["slug"]),
            question=str(r["question"]),
            yes_token=str(r["yes_token"]),
            no_token=str(r["no_token"]),
            fee_rate=float(r["fee_rate"]),
            gamma_ask_sum=float(r.get("best_ask") or 9.0) + (1.0 - float(r.get("best_bid") or 0.0)),
            end_date=r.get("end_date"),
        ))
    return sorted(rows, key=lambda m: (m.gamma_ask_sum, m.slug))


def score_market(m: Market, contracts: float) -> dict:
    """Fetch live books and walk YES+NO asks to `contracts` shares."""
    try:
        yes_ladder = pm_ask_ladder(m.yes_token)
        no_ladder = pm_ask_ladder(m.no_token)
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            return {"market_id": m.market_id, "slug": m.slug, "status": "no_fill",
                    "yes_fill": 0.0, "no_fill": 0.0, "note": "CLOB 404"}
        raise
    yes_cost, yes_fill = walk(yes_ladder, contracts)
    no_cost, no_fill = walk(no_ladder, contracts)
    fill = min(yes_fill, no_fill)
    if fill <= 0:
        return {"market_id": m.market_id, "slug": m.slug, "status": "no_fill",
                "yes_fill": yes_fill, "no_fill": no_fill}
    yes_cost, _ = walk(yes_ladder, fill)
    no_cost, _ = walk(no_ladder, fill)
    yes_px = yes_cost / fill
    no_px = no_cost / fill
    fee = venues.pm_taker_fee(yes_px, fill, m.fee_rate) + venues.pm_taker_fee(no_px, fill, m.fee_rate)
    gross = fill - (yes_cost + no_cost)
    net = gross - fee
    return {
        "market_id": m.market_id,
        "slug": m.slug,
        "question": m.question,
        "status": "ok",
        "fill": fill,
        "yes_px": yes_px,
        "no_px": no_px,
        "ask_sum": yes_px + no_px,
        "fee_rate": m.fee_rate,
        "fees": fee,
        "gross": gross,
        "net": net,
        "net_per_pair": net / fill,
    }


def _clob_book(token_id: str, timeout: float = 8.0) -> dict:
    """Fast public CLOB book fetch for a large scan.

    `venues.get` is intentionally conservative for catalogue enumeration, but a full complement
    scan can hit thousands of stale token ids. Waiting 60s x retries on each one turns dead
    books into the runtime. CLOB 404 is an unevaluable market here, not a transport mystery.
    """
    req = urllib.request.Request(
        f"{venues.CLOB}/book?token_id={token_id}",
        headers={"Accept": "application/json", "User-Agent": venues.USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError("CLOB 404") from exc
        raise


def _book_sides(token_id: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    book = _clob_book(token_id)
    asks = sorted(((float(a["price"]), float(a["size"])) for a in (book.get("asks") or [])),
                  key=lambda r: r[0])
    bids = sorted(((float(b["price"]), float(b["size"])) for b in (book.get("bids") or [])),
                  key=lambda r: -r[0])
    return asks, bids


def score_market_reflected(m: Market, contracts: float) -> dict:
    """One-book version of the same check, deriving NO asks from YES bids.

    In a normal PM binary book, a YES bid at p is a NO ask at 1-p with the same size. This is the
    algebraic complement check: YES ask + reflected NO ask < 1 is the same as a crossed YES book.
    """
    try:
        yes_ladder, yes_bids = _book_sides(m.yes_token)
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            return {"market_id": m.market_id, "slug": m.slug, "status": "no_fill",
                    "yes_fill": 0.0, "no_fill": 0.0, "note": "CLOB 404"}
        raise
    no_ladder = sorted(((1.0 - p, s) for p, s in yes_bids), key=lambda r: r[0])
    yes_cost, yes_fill = walk(yes_ladder, contracts)
    no_cost, no_fill = walk(no_ladder, contracts)
    fill = min(yes_fill, no_fill)
    if fill <= 0:
        return {"market_id": m.market_id, "slug": m.slug, "status": "no_fill",
                "yes_fill": yes_fill, "no_fill": no_fill}
    yes_cost, _ = walk(yes_ladder, fill)
    no_cost, _ = walk(no_ladder, fill)
    yes_px = yes_cost / fill
    no_px = no_cost / fill
    fee = venues.pm_taker_fee(yes_px, fill, m.fee_rate) + venues.pm_taker_fee(no_px, fill, m.fee_rate)
    gross = fill - (yes_cost + no_cost)
    net = gross - fee
    return {
        "market_id": m.market_id,
        "slug": m.slug,
        "question": m.question,
        "status": "ok",
        "mode": "reflected",
        "fill": fill,
        "yes_px": yes_px,
        "no_px": no_px,
        "ask_sum": yes_px + no_px,
        "fee_rate": m.fee_rate,
        "fees": fee,
        "gross": gross,
        "net": net,
        "net_per_pair": net / fill,
    }


def run_streaming(markets: list[Market], contracts: float, workers: int, use_reflection: bool,
                  out_path: str) -> list[dict]:
    """Run the scan and write rows as they complete.

    The returned list is still used for the summary, but the JSONL exists during the run so a
    long scan can be inspected and a crash does not erase every completed market.
    """
    rows = []
    scorer = score_market_reflected if use_reflection else score_market
    with open(out_path, "w", encoding="utf-8") as fh:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(scorer, m, contracts): m for m in markets}
            for fut in cf.as_completed(futs):
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001 - this is a live public endpoint scan
                    m = futs[fut]
                    row = {"market_id": m.market_id, "slug": m.slug, "status": "fetch_error",
                           "error": f"{type(exc).__name__}: {exc}"}
                rows.append(row)
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contracts", type=float, default=100.0,
                    help="contracts per leg to walk")
    ap.add_argument("--limit", type=int, default=500,
                    help="max markets to scan after Gamma-priority ordering; 0 means all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--use-reflection", action="store_true",
                    help="fetch only YES book and derive NO asks from YES bids")
    ap.add_argument("--out", default=os.path.join(DATA, "pm_complement_scan.jsonl"))
    args = ap.parse_args()

    markets = load_markets()
    chosen = markets if args.limit <= 0 else markets[:args.limit]
    start = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = run_streaming(chosen, args.contracts, args.workers, args.use_reflection, args.out)

    ok = [r for r in rows if r.get("status") == "ok"]
    positive = sorted((r for r in ok if r["net"] > 0.0), key=lambda r: -r["net"])
    top = sorted(ok, key=lambda r: r["net"], reverse=True)[:10]
    no_fill = sum(1 for r in rows if r.get("status") == "no_fill")
    errors = sum(1 for r in rows if r.get("status") == "fetch_error")

    print(f"loaded {len(markets):,} PM Yes/No binaries with token ids")
    books_per = 1 if args.use_reflection else 2
    print(f"scanned {len(chosen):,} markets x {books_per} live CLOB book(s), {args.contracts:g} contracts/leg, "
          f"{args.workers} workers in {time.time() - start:.1f}s")
    print(f"ok {len(ok):,}; no_fill {no_fill:,}; fetch_error {errors:,}; positive {len(positive):,}")
    print(f"wrote {args.out}")
    print("\nTop net results (read-only, live CLOB, net of PM taker fee):")
    for r in top:
        flag = " POSITIVE" if r["net"] > 0 else ""
        print(f"  {r['net']:>8.2f}  ask_sum={r['ask_sum']:.4f}  fees={r['fees']:.2f}  "
              f"fill={r['fill']:.0f}  {r['slug'][:78]}{flag}")
    if positive:
        print("\nPositive rows:")
        for r in positive[:25]:
            print(f"  {r['net']:>8.2f}  net/pair={r['net_per_pair']:.4f}  "
                  f"YES {r['yes_px']:.4f} + NO {r['no_px']:.4f}  {r['slug']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
