#!/usr/bin/env python3
"""Stage 1 of the PM x Kalshi feasibility measurement: how many contracts even exist on both
venues, with a two-sided book, that a hedge could theoretically use. READ-ONLY. No account.

This is a MATCHED-UNIVERSE AND DEPTH probe, not an arbitrage scanner. It deliberately stops
before computing a single hedge cost, because three things can kill the idea outright and all
three are cheaper to test than a book walk:

    1. the fee floor    -- combined taker fees exceed any plausible gap
    2. the universe     -- too few contracts exist on both venues at all
    3. the text         -- the ones that do exist do not ask the same question

The output is three buckets -- no match / different-or-unreadable text / book gap after
fees -- except that this stage refuses to fill the third bucket. A pair reaches
"unreviewed candidate" and stops there. Nothing here promotes a pair to "matched" on
title similarity, because "same headline" is exactly the failure mode that makes a cross-venue
number look real when the two contracts settle on different sources, deadlines, agencies or
recount rules. Adjudicated matches are read from `text_verdicts.json`, which this probe never
writes.

WHAT THE STOP RULE IS
---------------------
Committed before the run: fewer than 20 clean text matches with real two-sided depth
-> stop. Matches but no post-fee gaps -> stop. Only recurring post-fee gaps justify spending
time on capital logistics. This script reports against the first of those; it cannot report
against the second, by construction.

WHY TITLE MATCHING IS A CANDIDATE GENERATOR AND NOTHING MORE
-------------------------------------------------------------
The scoring below is IDF-weighted token overlap. It is tuned for RECALL -- it should surface
pairs that are not really the same contract, and the review step should throw them out. A
matcher tuned the other way would quietly shrink the universe and the shrinkage would be
invisible, which is the same shape of error as trap 1 and trap 4 in `venues.py`: a plausible,
non-empty, wrong answer.

    python probe_overlap.py --refresh          # re-fetch both universes (~6 min), then report
    python probe_overlap.py                    # report from the cached snapshot

A missing snapshot refreshes implicitly, so the first run costs the same six minutes with or
without --refresh. The Polymarket leg dominates: keyset pagination walks the whole not-closed
universe every time, because the end-date horizon is applied client-side. Cached reports avoid
that network walk but still recompute candidate matching locally.

    python probe_overlap.py --candidates 200   # write the review file for the top N pairs
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import re

import venues

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SNAPSHOT = os.path.join(DATA, "universe_snapshot.json")
CANDIDATES = os.path.join(DATA, "candidates.json")
VERDICTS = os.path.join(DATA, "text_verdicts.json")
PM_HORIZON_DAYS = 900
PM_LOOKBACK_DAYS = 30

# The verdict vocabulary this report knows how to name. `text_mismatch` is the retired spelling
# of `different_contract` and is kept so older verdict files still count. Anything not in here
# is surfaced explicitly rather than absorbed into a bucket it does not belong to.
KNOWN_VERDICTS = {
    "unreviewed", "different_contract", "text_mismatch", "unreadable",
    "same_contract", "same_contract_inverted",
}

# Kalshi categories whose contracts could plausibly have a Polymarket twin asking the identical
# question. Sports is excluded at this stage for a reason worth stating: PM's sports books are
# game-level and short-lived, an earlier cross-venue study of sportsbooks already died on PM
# sports depth, and including several thousand sports events would swamp the candidate list with
# the one region already measured and rejected. It can be switched back on with --all-categories.
DEFAULT_CATEGORIES = {
    "Politics", "Elections", "Economics", "Financials", "Crypto", "Companies",
    "Climate and Weather", "Science and Technology", "World", "Health", "Commodities",
}

STOP = set("""a an the is are was were be been being will would shall should can could may might
do does did done have has had of in on at to for from by with without within into onto over under
above below before after during until till than then this that these those there here it its and
or not no nor if as vs v via per about between among across against up down out off""".split())

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(*parts: str) -> set[str]:
    out = set()
    for p in parts:
        if p:
            out |= {t for t in TOKEN.findall(p.lower()) if t not in STOP and len(t) > 1}
    return out


def iso(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- snapshot

def refresh() -> dict:
    """Fetch both live universes and cache them with the timestamp they were taken at.

    The timestamp is not decoration. A book gap measured against a universe fetched an hour ago
    is measuring a market that may have closed, and every downstream count in this file is only
    a claim about the moment in `taken_at`.
    """
    os.makedirs(DATA, exist_ok=True)
    series = venues.fetch_kalshi_series()
    k_events = venues.fetch_kalshi_events()
    p_events = venues.fetch_pm_events(
        horizon_days=PM_HORIZON_DAYS,
        lookback_days=PM_LOOKBACK_DAYS,
    )
    snap = {
        "taken_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kalshi_series": series,
        "kalshi_events": k_events,
        "pm_events": p_events,
        "pm_query": {
            "predicate": "closed=false",
            "horizon_days": PM_HORIZON_DAYS,
            "lookback_days": PM_LOOKBACK_DAYS,
        },
    }
    with open(SNAPSHOT, "w") as fh:
        json.dump(snap, fh)
    return snap


def load(refresh_first: bool) -> dict:
    if refresh_first or not os.path.exists(SNAPSHOT):
        return refresh()
    with open(SNAPSHOT) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- flatten

def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def kalshi_rows(snap: dict, categories: set[str] | None) -> list[dict]:
    """One row per open Kalshi binary market, with series category and fee terms joined on.

    The join is client-side because Kalshi has no server-side category filter (venues.py trap 3),
    and the fee multiplier is a SERIES field -- reading it per market is the only way to notice
    the 32 series that charge half or nothing.
    """
    series = snap["kalshi_series"]
    rows = []
    for ev in snap["kalshi_events"]:
        st = ev.get("series_ticker") or ""
        meta = series.get(st) or {}
        cat = meta.get("category") or "?"
        if categories and cat not in categories:
            continue
        for m in ev.get("markets") or []:
            bid, ask = num(m.get("yes_bid_dollars")), num(m.get("yes_ask_dollars"))
            rows.append({
                "ticker": m.get("ticker"),
                "event_ticker": ev.get("event_ticker"),
                "series_ticker": st,
                "category": cat,
                "title": ev.get("title") or m.get("title") or "",
                "sub": m.get("yes_sub_title") or ev.get("sub_title") or "",
                "rules_primary": m.get("rules_primary") or "",
                "rules_secondary": m.get("rules_secondary") or "",
                "settlement_sources": meta.get("settlement_sources") or ev.get("settlement_sources"),
                "close_time": m.get("close_time"),
                "yes_bid": bid,
                "yes_ask": ask,
                "yes_bid_size": num(m.get("yes_bid_size_fp")),
                "yes_ask_size": num(m.get("yes_ask_size_fp")),
                "two_sided": bid > 0 and ask > 0,
                "fee_multiplier": venues.kalshi_fee_multiplier(meta),
            })
    return rows


def pm_rows(snap: dict) -> list[dict]:
    """One row per PM market that is a genuine Yes/No binary with a live order book.

    The polarity filter is not cosmetic. Some PM markets carry outcome pairs like Odd/Even,
    Over/Under or two team names. Those can still be binaries, but mapping their YES side onto
    a Kalshi YES side requires reading which label means what, and a probe that guesses would
    hedge the wrong direction -- doubling exposure while reporting it as flat.
    """
    rows = []
    for ev in snap["pm_events"]:
        for m in ev.get("markets") or []:
            if not (m.get("active") and not m.get("archived") and m.get("enableOrderBook")):
                continue
            try:
                outcomes = json.loads(m.get("outcomes") or "[]")
                token_ids = json.loads(m.get("clobTokenIds") or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            if [str(o).strip().lower() for o in outcomes] != ["yes", "no"]:
                continue
            bid, ask = num(m.get("bestBid")), num(m.get("bestAsk"))
            rows.append({
                "id": m.get("id"),
                "condition_id": m.get("conditionId"),
                "slug": m.get("slug"),
                "event_slug": ev.get("slug"),
                "question": m.get("question") or "",
                "description": m.get("description") or "",
                "resolution_source": m.get("resolutionSource") or "",
                "end_date": m.get("endDate"),
                "yes_token": token_ids[0] if len(token_ids) == 2 else None,
                "no_token": token_ids[1] if len(token_ids) == 2 else None,
                "best_bid": bid,
                "best_ask": ask,
                "two_sided": bid > 0 and ask > 0,
                "neg_risk": bool(m.get("negRisk")),
                "tick": num(m.get("orderPriceMinTickSize")),
                "fee_rate": venues.pm_fee_rate(m),
            })
    return rows


# --------------------------------------------------------------------------- matching

def idf(docs: list[set[str]]) -> dict[str, float]:
    """Inverse document frequency over the Kalshi side, used to weight token overlap.

    Without it, "will" and "2026" dominate every pairing and the top candidates are whatever two
    venues happen to share the most boilerplate. With it, the signal is carried by the rare
    tokens -- names, tickers, agencies -- which is where a real match lives.
    """
    df = collections.Counter()
    for d in docs:
        df.update(d)
    n = max(1, len(docs))
    return {t: math.log(n / (1 + c)) for t, c in df.items()}


def candidates(krows: list[dict], prows: list[dict], min_score: float,
               deadline_days: float) -> list[dict]:
    """Generate candidate pairs, tuned for recall. Every returned pair is UNADJUDICATED.

    Two gates before scoring, both structural rather than lexical:

      * deadline proximity -- Kalshi `close_time` against PM `endDate`. Two contracts on the
        same question with deadlines a month apart are different contracts, and a hedge across
        them is a naked directional bet with extra steps. The window is generous because PM's
        `endDate` is frequently a placeholder far past the real resolution.
      * two-sided books on BOTH venues. A pair where either side has no bid or no ask cannot be
        crossed at any price, so it is not a depth candidate no matter how well the text matches.
    """
    ktok = [tokens(r["title"], r["sub"]) for r in krows]
    weights = idf(ktok)
    # Invert the Kalshi side once: token -> row indices. Scoring every PM row against every
    # Kalshi row is 18k x 25k pairs; the inverted index only visits pairs sharing a token.
    index = collections.defaultdict(list)
    for i, toks in enumerate(ktok):
        for t in toks:
            if weights.get(t, 0.0) > 0.5:  # skip boilerplate as a blocking key
                index[t].append(i)

    out = []
    for p in prows:
        if not p["two_sided"]:
            continue
        p_end = iso(p["end_date"])
        ptok = tokens(p["question"])
        hits = collections.Counter()
        for t in sorted(ptok):
            for i in index.get(t, ()):
                hits[i] += 1
        best = None
        # Deterministic truncation. `Counter.most_common` breaks ties by insertion order, and
        # insertion order here follows set iteration, which Python randomises per process via
        # PYTHONHASHSEED. Two runs over the SAME cached snapshot produced 5,867 and 5,900
        # candidates, and adjudicated pairs silently fell out of the pool between them -- so a
        # verdict written yesterday stopped being counted today. Tie-break on the ticker.
        ranked = sorted(hits.items(), key=lambda kv: (-kv[1], krows[kv[0]]["ticker"] or ""))
        for i, _ in ranked[:40]:
            k = krows[i]
            if not k["two_sided"]:
                continue
            k_end = iso(k["close_time"])
            if p_end and k_end and abs((p_end - k_end).total_seconds()) > deadline_days * 86400:
                continue
            shared = ktok[i] & ptok
            union = ktok[i] | ptok
            if not union:
                continue
            score = sum(weights.get(t, 0.0) for t in shared) / sum(
                weights.get(t, 0.0) for t in union)
            if score >= min_score and (best is None or score > best["score"]
                                       or (score == best["score"]
                                           and (k["ticker"] or "") < (best["kalshi"]["ticker"] or ""))):
                best = {"score": round(score, 4), "kalshi": k, "pm": p,
                        "shared": sorted(shared, key=lambda t: (-weights.get(t, 0.0), t))[:12]}
        if best:
            out.append(best)
    return sorted(out, key=lambda c: (-c["score"], c["kalshi"]["ticker"] or "", str(c["pm"]["id"])))


# --------------------------------------------------------------------------- report

def report(snap: dict, args) -> int:
    cats = None if args.all_categories else DEFAULT_CATEGORIES
    krows = kalshi_rows(snap, cats)
    prows = pm_rows(snap)
    taken = snap["taken_at"]

    print(f"snapshot taken {taken}")
    print(f"\nKALSHI live universe")
    print(f"  events                        {len(snap['kalshi_events']):>8,}")
    print(f"  binary markets (all cats)     {sum(len(e.get('markets') or []) for e in snap['kalshi_events']):>8,}")
    label = "all categories" if cats is None else f"{len(cats)} non-sport categories"
    print(f"  in scope ({label})".ljust(34) + f"{len(krows):>8,}")
    print(f"  of those, two-sided book      {sum(1 for r in krows if r['two_sided']):>8,}")
    free = sum(1 for r in krows if r["fee_multiplier"] == 0)
    print(f"  of those, zero taker fee      {free:>8,}")

    pm_query = snap.get("pm_query") or {
        "predicate": "legacy date-bisection closed=false",
        "horizon_days": 900,
        "lookback_days": 30,
    }
    print(f"\nPOLYMARKET not-closed universe")
    print(f"  predicate                     {pm_query['predicate']}")
    print(f"  endDate window                -{pm_query['lookback_days']}d to +{pm_query['horizon_days']}d")
    pm_all = sum(len(e.get('markets') or []) for e in snap['pm_events'])
    print(f"  events                        {len(snap['pm_events']):>8,}")
    print(f"  markets                       {pm_all:>8,}")
    print(f"  Yes/No binaries with a book   {len(prows):>8,}")
    print(f"  of those, two-sided book      {sum(1 for r in prows if r['two_sided']):>8,}")
    print(f"  of those, zero taker fee      {sum(1 for r in prows if r['fee_rate'] == 0):>8,}")

    print(f"\nFEE FLOOR -- combined taker cost of one hedged pair, cents per $1 pair")
    print("  Both venues charge on P(1-P), so there is no cheap leg to route into. A gap that")
    print("  does not clear this column is dead before spreads, depth or matching are considered.")
    print(f"\n  {'P':>6}{'Kalshi 0.07':>13}{'PM 0.04':>10}{'PM 0.07':>10}"
          f"{'floor@PM.04':>13}{'floor@PM.07':>13}")
    for p in (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        k = venues.kalshi_taker_fee(p, 100.0, 1.0) / 100.0 * 100
        a = venues.pm_taker_fee(p, 1.0, 0.04) * 100
        b = venues.pm_taker_fee(p, 1.0, 0.07) * 100
        print(f"  {p:>6.2f}{k:>13.2f}{a:>10.2f}{b:>10.2f}{k + a:>13.2f}{k + b:>13.2f}")

    cands = candidates(krows, prows, args.min_score, args.deadline_days)
    verdicts = {}
    if os.path.exists(VERDICTS):
        with open(VERDICTS) as fh:
            verdicts = json.load(fh)

    print(f"\nMATCHED UNIVERSE (candidate generation only -- nothing here is an adjudicated match)")
    print(f"  PM two-sided binaries considered    {sum(1 for r in prows if r['two_sided']):>8,}")
    print(f"  candidate pairs at score >= {args.min_score:<4}    {len(cands):>8,}")
    for lo in (0.5, 0.4, 0.3, 0.2):
        print(f"    of which score >= {lo:<4}              {sum(1 for c in cands if c['score'] >= lo):>8,}")

    buckets = collections.Counter()
    strata = collections.defaultdict(collections.Counter)
    for c in cands:
        key = f"{c['kalshi']['ticker']}|{c['pm']['id']}"
        v = verdicts.get(key, {})
        verdict = v.get("verdict", "unreviewed")
        buckets[verdict] += 1
        if v:
            strata[v.get("stratum", "?")][verdict] += 1
    matched = buckets.get("same_contract", 0) + buckets.get("same_contract_inverted", 0)
    print(f"\n  BUCKETS")
    print(f"    no match          {sum(1 for r in prows if r['two_sided']) - len(cands):>8,}"
          f"   (two-sided PM binary with no Kalshi candidate)")
    print(f"    unreviewed        {buckets.get('unreviewed', 0):>8,}   (candidate, resolution text not yet adjudicated)")
    different = buckets.get("different_contract", 0) + buckets.get("text_mismatch", 0)
    print(f"    different         {different:>8,}   (adjudicated: different question)")
    print(f"    unreadable        {buckets.get('unreadable', 0):>8,}   (published rule text incomplete or templated)")
    print(f"    same contract     {buckets.get('same_contract', 0):>8,}   (adjudicated: same question, same settlement)")
    print(f"    same, INVERTED    {buckets.get('same_contract_inverted', 0):>8,}   (complementary framing -- hedge is the SAME side on both venues)")

    # The five lines above enumerate a closed vocabulary, so a verdict string outside it would
    # be counted and never printed -- the breakdown would silently under-sum against the
    # candidate total. That matters because the same typo is invisible downstream too:
    # walk_hedge.py only walks `same_contract` and `same_contract_inverted`, so a pair the
    # reviewer believes they adjudicated as usable would vanish from both the report and the
    # walk with nothing anywhere saying so. Name it instead.
    for verdict, n in sorted(buckets.items()):
        if verdict not in KNOWN_VERDICTS:
            print(f"    {'! ' + verdict:<18}{n:>8,}   (unrecognised verdict -- typo? not usable by walk_hedge.py)")

    # The adjudicated pairs are a stratified random sample, so the whole pool can be estimated
    # from them rather than guessed at. Reported as a point estimate with the sample size that
    # produced it attached, because 3 hits out of 30 carries an interval wide enough that only
    # the order of magnitude is being claimed.
    if strata:
        print(f"\n  SAMPLE-BASED ESTIMATE OF THE FULL POOL")
        est = 0.0
        for name, lo, hi in (("lo", 0.20, 0.35), ("mid", 0.35, 0.55), ("hi", 0.55, 1.01)):
            pool = sum(1 for c in cands if lo <= c["score"] < hi)
            seen = sum(strata[name].values())
            hits = strata[name]["same_contract"] + strata[name]["same_contract_inverted"]
            share = hits / seen if seen else 0.0
            est += pool * share
            print(f"    score [{lo:.2f},{hi:.2f}): pool {pool:>6,}   adjudicated {seen:>3}   "
                  f"same-contract {hits}   -> est {pool * share:>7,.0f}")
        print(f"    estimated same-contract pairs in the full candidate pool: ~{est:,.0f}")
        print(f"    This is an extrapolation from {sum(sum(s.values()) for s in strata.values())} "
              f"hand-read pairs. Treat it as an order of magnitude, not a count.")

    print(f"\n  STOP RULE: fewer than 20 adjudicated same-contract pairs with two-sided depth -> stop.")
    if matched < 20:
        print(f"  Adjudicated so far: {matched}. NOT CLEARED on adjudicated pairs. The estimate above")
        print(f"  is not a substitute -- the rule was written against pairs actually read.")

    if args.candidates:
        top = cands[:args.candidates]
        with open(CANDIDATES, "w") as fh:
            json.dump({"taken_at": taken, "pairs": top}, fh, indent=1)
        print(f"\n  wrote {len(top)} candidate pairs with full resolution text from both venues to")
        print(f"  {CANDIDATES}")
        print(f"  Adjudicate into {os.path.basename(VERDICTS)} as "
              f'{{"<kalshi_ticker>|<pm_id>": {{"verdict": "same_contract"|"same_contract_inverted"|"different_contract"|"unreadable", "why": "..."}}}}')

    print(f"\n  top candidates by score (UNADJUDICATED -- read the rules text before believing any):")
    for c in cands[:args.show]:
        print(f"\n    {c['score']:.3f}  K {c['kalshi']['ticker']}  [{c['kalshi']['category']}]")
        print(f"           {c['kalshi']['title'][:96]}  ({c['kalshi']['sub'][:36]})")
        print(f"           PM {c['pm']['slug'][:80]}")
        print(f"           {c['pm']['question'][:96]}")
        print(f"           K yes {c['kalshi']['yes_bid']:.2f}/{c['kalshi']['yes_ask']:.2f}   "
              f"PM yes {c['pm']['best_bid']:.3f}/{c['pm']['best_ask']:.3f}   "
              f"fees K x{c['kalshi']['fee_multiplier']:g} PM {c['pm']['fee_rate']:.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="re-fetch both universes")
    ap.add_argument("--all-categories", action="store_true",
                    help="include Sports and Entertainment (excluded by default; see module docstring)")
    ap.add_argument("--min-score", type=float, default=0.20)
    ap.add_argument("--deadline-days", type=float, default=7.0)
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--candidates", type=int, default=0,
                    help="write the top N candidate pairs to data/candidates.json for review")
    args = ap.parse_args()
    return report(load(args.refresh), args)


if __name__ == "__main__":
    raise SystemExit(main())
