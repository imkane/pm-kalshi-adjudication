#!/usr/bin/env python3
"""Walk a fixed notional through both venues' live books for adjudicated pairs and report the
hedge cost after published taker fees. READ-ONLY: fetches two order books and prints. No keys,
no account, no orders, no order-shaped data structures.

RUNS ONLY ON PAIRS THAT SURVIVED TEXT ADJUDICATION
---------------------------------------------------
It reads `data/text_verdicts.json` and refuses anything not marked `same_contract` or
`same_contract_inverted`. A number computed across two contracts that settle differently is not
a hedge cost, it is a directional bet with a misleading label, and the whole reason the
adjudication step exists is that title similarity produces exactly that number very convincingly.

POLARITY
--------
`same_contract`          Kalshi YES == PM YES. The flat book is YES on one venue, NO on the other.
`same_contract_inverted` Kalshi YES == PM NO.  The flat book is the SAME side on both venues.

Getting this backwards doubles the position while reporting it as flat, so the direction is read
from the verdict file rather than inferred, and an unrecognised verdict is refused rather than
defaulted.

KALSHI BOOKS ARE BID-ONLY
--------------------------
`/markets/{ticker}/orderbook` returns `yes_dollars` and `no_dollars`, and BOTH are bid ladders.
There is no ask side to read. To buy YES you lift the NO bids: a resting NO bid at q fills a YES
buy at 1-q, and the size available at that YES price is the size resting at that NO bid. So the
YES ask ladder is `[(1-q, size) for q, size in no_bids]`, cheapest first meaning HIGHEST q first.
Reading `yes_dollars` as an ask ladder inverts the book and reports the best bid as the best
offer, which makes every market look crossed with itself.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
No capital-logistics model. Kalshi settles in KYC'd USD and Polymarket in on-chain USDC, and
they cannot be rebalanced against each other quickly, so any return has to be measured against
capital PARKED on both venues rather than against trade notional. That number is out of scope
here unless recurring post-fee gaps are first demonstrated.

    python walk_hedge.py               # $100 through every adjudicated pair
    python walk_hedge.py --notional 500
"""

from __future__ import annotations

import argparse
import json
import os

import venues

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SNAPSHOT = os.path.join(DATA, "universe_snapshot.json")
VERDICTS = os.path.join(DATA, "text_verdicts.json")

USABLE = {"same_contract", "same_contract_inverted"}


def derive_ask_ladder(opposite_bids) -> list[tuple[float, float]]:
    """Turn a Kalshi opposite-side BID ladder into the ask ladder for the side you want to buy.

    A resting NO bid at q fills a YES buy at 1-q with the size resting at q, so the cheapest YES
    offer comes from the HIGHEST NO bid. Kept pure and separate from the fetch so the inversion
    is unit-testable; getting it wrong reports the best bid as the best offer and makes every
    market look crossed with itself.
    """
    return sorted(((1.0 - float(p), float(sz)) for p, sz in (opposite_bids or [])),
                  key=lambda r: r[0])


def kalshi_ask_ladder(ticker: str, side: str) -> list[tuple[float, float]]:
    """Ask ladder for buying `side` on Kalshi, cheapest first, derived from the opposite bids.

    Returns [(price_dollars, contracts)]. Empty when the opposite side has no resting bids,
    which is a real and common state -- Kalshi books go one-sided routinely -- and is reported
    as "cannot be crossed" rather than silently priced at 1.00.
    """
    ob = (venues.get(f"{venues.KALSHI}/markets/{ticker}/orderbook") or {}).get("orderbook_fp") or {}
    return derive_ask_ladder(ob.get("no_dollars" if side == "yes" else "yes_dollars"))


def pm_ask_ladder(token_id: str) -> list[tuple[float, float]]:
    """Ask ladder for a PM CLOB token, cheapest first. Returns [(price, shares)]."""
    book = venues.get(f"{venues.CLOB}/book?token_id={token_id}")
    asks = [(float(a["price"]), float(a["size"])) for a in (book.get("asks") or [])]
    return sorted(asks, key=lambda r: r[0])


def walk(ladder: list[tuple[float, float]], want: float) -> tuple[float, float]:
    """Buy `want` contracts through `ladder`. Returns (cost_dollars, contracts_filled).

    Partial fills are returned as partials, not scaled up. A pair that can only fill 30 of 100
    contracts is a different fact from one that fills 100, and averaging the two away is how a
    thin book comes to look like a tradeable one.
    """
    cost = filled = 0.0
    for price, size in ladder:
        if filled >= want:
            break
        take = min(size, want - filled)
        cost += take * price
        filled += take
    return cost, filled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notional", type=float, default=100.0,
                    help="target contracts per leg (one pair pays $1 at settlement)")
    args = ap.parse_args()

    with open(SNAPSHOT) as fh:
        snap = json.load(fh)
    with open(VERDICTS) as fh:
        verdicts = json.load(fh)

    import probe_overlap as P
    krows = {r["ticker"]: r for r in P.kalshi_rows(snap, None)}
    prows = {str(r["id"]): r for r in P.pm_rows(snap)}

    pairs = [(k, v) for k, v in verdicts.items()
             if isinstance(v, dict) and v.get("verdict") in USABLE]
    print(f"{len(pairs)} adjudicated pair(s) marked usable; walking {args.notional:g} contracts "
          f"per leg through live books\n")

    for key, v in pairs:
        ticker, pm_id = key.split("|", 1)
        k, p = krows.get(ticker), prows.get(pm_id)
        if not k or not p:
            print(f"  {key}: no longer in the snapshot universe (closed or delisted) -- skipped\n")
            continue
        inverted = v["verdict"] == "same_contract_inverted"
        print(f"  {ticker}  x  PM {p['slug'][:60]}")
        print(f"    {k['title'][:80]} [{k['sub'][:28]}]")
        print(f"    polarity: Kalshi YES == PM {'NO' if inverted else 'YES'}")

        # Two ways to hold the pair. Under normal polarity they are PM-YES/Kalshi-NO and
        # PM-NO/Kalshi-YES; under inverted polarity both legs sit on the same nominal side.
        legs = [("PM YES", p["yes_token"], "no" if not inverted else "yes"),
                ("PM NO", p["no_token"], "yes" if not inverted else "no")]
        for label, token, kside in legs:
            if not token:
                print(f"    {label:<7} + Kalshi {kside.upper():<3}: PM token id missing -- skipped")
                continue
            pm_l = pm_ask_ladder(token)
            k_l = kalshi_ask_ladder(ticker, kside)
            pm_cost, pm_fill = walk(pm_l, args.notional)
            k_cost, k_fill = walk(k_l, args.notional)
            fill = min(pm_fill, k_fill)
            if fill <= 0:
                print(f"    {label:<7} + Kalshi {kside.upper():<3}: one side has no offers -- "
                      f"cannot be crossed (PM {pm_fill:g}, Kalshi {k_fill:g})")
                continue
            # Re-walk to the achievable size so cost and fill describe the same trade.
            pm_cost, _ = walk(pm_l, fill)
            k_cost, _ = walk(k_l, fill)
            pm_px, k_px = pm_cost / fill, k_cost / fill
            fee = (venues.pm_taker_fee(pm_px, fill, p["fee_rate"])
                   + venues.kalshi_taker_fee(k_px, fill, k["fee_multiplier"]))
            gross = fill * 1.0 - (pm_cost + k_cost)
            net = gross - fee
            flag = "  <-- POSITIVE" if net > 0 else ""
            print(f"    {label:<7} + Kalshi {kside.upper():<3}: fill {fill:>6.0f}  "
                  f"PM {pm_px:.4f} + K {k_px:.4f} = {pm_px + k_px:.4f}  "
                  f"fees ${fee:>6.2f}  net ${net:>8.2f}{flag}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
