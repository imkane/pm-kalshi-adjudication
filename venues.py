"""Read-only enumeration of the Polymarket and Kalshi live/not-closed universes, plus the fee
rules that decide whether any cross-venue gap is worth anything. No keys, no orders, no account.

Nothing in this module places, prices, or simulates an order. It fetches public catalogue data
and computes published fee formulas. That is the whole surface, deliberately: a feasibility
measurement should be runnable by someone with no account on either venue, and reviewable by
someone who wants to confirm it never had one.

API TRAPS, ALL OF WHICH PRODUCED WRONG ANSWERS BEFORE THEY WERE FOUND
--------------------------------------------------------------------------
These are recorded because each one returns a plausible, non-empty, wrong result rather than an
error. A silent 1,000-row slice looks exactly like a small universe.

  1. Kalshi's pagination key is `cursor`, NOT `next_cursor`. `/incentive_programs` answers
     `next_cursor` and the incentives scanner reads that correctly; `/markets` and `/events`
     answer `cursor`. Reading `next_cursor` against `/markets` terminates after ONE page, and
     the first page happens to be 100% auto-generated parlay markets. That is precisely the
     "two series, 1,000 markets" sample that made an earlier overlap probe unusable.

  2. Kalshi's `/markets?status=open` universe is ~400,000 rows and 99.5% of it is `Exotics` --
     the KXMVE* multi-leg combination markets, machine-generated in bulk. Paginating it is
     40 minutes of traffic to find ~2,000 real books. `/events?status=open&with_nested_markets=true`
     returns 9,619 events / 84,111 markets in ~26 seconds with the cursor exhausted, and the
     MVE combinations do not appear in it. Enumerate events, never markets.

  3. Kalshi silently IGNORES `?category=`. `/markets?status=open&category=Politics` answers
     parlay markets. There is no server-side category filter; category lives on the SERIES, so
     it has to be joined client-side from `/series` (which itself ignores `limit` and returns
     all 13,029 rows in one response).

  4. Polymarket's legacy Gamma offset pagination hard-fails with HTTP 422 at `offset >= 2100`,
     on every ordering and on both `/markets` and `/events`. It is a legacy-endpoint trap, not
     the current recommended path. Gamma keyset pagination reached 40,000 unique events over
     400 capped pages in a broad walk on 2026-08-17; the raw `closed=false` query this module
     uses then exhausted cleanly at 21,075 unique events over 211 pages. `fetch_pm_events`
     therefore uses `/events/keyset`, then applies the same 900-day horizon used by the original
     feasibility snapshot; the old date-bisection workaround is intentionally retired.

KALSHI HISTORICAL BOUNDARY
--------------------------
This module intentionally enumerates the live Kalshi universe only. Kalshi's historical
catalogue is split onto separate `/historical/*` endpoints with a server-reported cutoff; on
2026-08-17 `/historical/cutoff` returned `2026-06-18T00:00:00Z` for markets, trades, orders and
market positions. Any claim about historical Kalshi coverage must use those endpoints explicitly,
not `fetch_kalshi_events`.

FEES: BOTH VENUES NOW CHARGE, AND BOTH PEAK AT p=0.5
-----------------------------------------------------
This is the first-order gate on the whole idea, so the formulas live here rather than in the
scanner, where they could be quietly relaxed.

  Kalshi taker fee = ceil_cents(0.07 x multiplier x C x P x (1-P))
      `fee_type` and `fee_multiplier` are per-SERIES fields on `/series`. 12,867 of 13,029
      series are ('quadratic', 1); 18 are half-rate, and 14 charge nothing at all. 129 are
      'quadratic_with_maker_fees', which adds a maker fee we never pay because this design
      crosses the spread on both legs.

  Polymarket taker fee = shares x rate x P x (1-P),  rate from the market's own `feeSchedule`
      Ranges 0.04 (politics, tech, finance, mentions) to 0.07 (crypto), with 0.05 for sports,
      economics, culture, weather and general. `takerOnly: true` on every schedule observed.
      130 markets had `feesEnabled: false` and pay nothing -- but read that denominator
      carefully: it was 130 of the 2,100 markets reachable under legacy offset pagination,
      and 2,100 is the offset ceiling in trap 4 above, not a universe size. It is a rate over
      whatever the broken pager happened to return, which is the exact error this file exists
      to warn about. Re-measure it over a keyset walk before quoting it as a share.

      Do NOT use CLOB `/fee-rate` for this. It answers `{"base_fee": 1000}` for a 0.07 crypto
      market AND a 0.05 general market -- it is an on/off flag with a fixed magnitude, not a
      rate. An earlier study of PM's crypto up/down markets established that the Gamma
      `feeSchedule` is the authoritative read, and that `shares x rate x P x (1-P)` reproduces
      both worked examples in Polymarket's own documentation.

The two formulas have the same shape. A hedge that takes on both venues pays the sum, and the
sum is maximal at P=0.5 -- 1.00c on politics plus 1.75c on standard Kalshi, 2.75c on a $1 pair
before either spread is crossed. That fee floor is what killed the crypto up/down study, and it
is why the cheap screen below runs before any book is fetched.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import datetime as dt

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# Identify the tool to venue operators so traffic is attributable and contactable.
USER_AGENT = "pm-kalshi-adjudication/1.0 (+https://github.com/imkane/pm-kalshi-adjudication)"

KALSHI_BASE_RATE = 0.07
GAMMA_OFFSET_CEILING = 2100  # Legacy offset pagination: HTTP 422 at and beyond this; see trap 4


def get(url: str, tries: int = 4, timeout: int = 60):
    """GET JSON with backoff. Raises rather than returning empty.

    An empty list from a catalogue endpoint reads as "the venue has nothing", which is the one
    false negative a universe probe cannot afford -- it would report zero overlap and look like
    a finding. HTTP 422 is re-raised unwrapped so legacy-offset probes can recognise Gamma's
    offset ceiling instead of treating it as a transport failure.
    """
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 422:
                raise
            last = f"HTTP {exc.code}"
            time.sleep(0.5 * (i + 1))
        except Exception as exc:  # noqa: BLE001 - transport variety is wide and all of it retries
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (i + 1))
    raise RuntimeError(f"GET {url} failed after {tries}: {last}")


# --------------------------------------------------------------------------- Kalshi

def fetch_kalshi_series() -> dict[str, dict]:
    """Ticker -> series metadata. `/series` ignores `limit` and returns all 13,029 in one shot.

    Carries the two fields nothing else exposes: `category` (there is no server-side category
    filter, trap 3) and the fee terms (`fee_type`, `fee_multiplier`).
    """
    return {s["ticker"]: s for s in get(f"{KALSHI}/series")["series"]}


def fetch_kalshi_events(limit: int = 200, max_pages: int = 400) -> list[dict]:
    """Open events with their markets nested. Cursor key is `cursor` -- see trap 1.

    Raises if the page cap is hit with a cursor still outstanding. A truncated universe that
    reports itself as complete is the exact failure this probe exists to avoid, so it fails
    loudly rather than under-counting the overlap.
    """
    out, cursor, pages = [], None, 0
    while pages < max_pages:
        q = f"{KALSHI}/events?status=open&limit={limit}&with_nested_markets=true"
        if cursor:
            q += f"&cursor={cursor}"
        d = get(q)
        batch = d.get("events") or []
        out += batch
        pages += 1
        cursor = d.get("cursor")
        if not cursor or not batch:
            return out
    raise RuntimeError(
        f"Kalshi event enumeration hit the {max_pages}-page cap with cursor {cursor!r} still "
        f"outstanding: {len(out)} events collected but the universe is NOT complete.")


def kalshi_taker_fee(price: float, contracts: float, multiplier: float = 1.0) -> float:
    """Published Kalshi taker fee in dollars: ceil to the cent of 0.07 x m x C x P x (1-P).

    Rounding is UP to the next whole cent per Kalshi's schedule, which matters at the small
    sizes this probe walks -- at 100 contracts and P=0.02 the unrounded fee is $0.0137 and the
    charged fee is $0.01, a 27% difference on a number that decides pass/fail.
    """
    raw = KALSHI_BASE_RATE * multiplier * contracts * price * (1.0 - price)
    # Round before the ceiling. P(1-P) is symmetric in exact arithmetic but not in binary
    # floating point -- 0.9*0.1 evaluates a hair above 0.1*0.9 -- and a ceiling amplifies that
    # into a whole cent, so the fee at P and at 1-P would differ. Ten places is far below the
    # cent this then rounds to and far above the error being suppressed.
    return math.ceil(round(raw, 10) * 100.0) / 100.0


# --------------------------------------------------------------------------- Polymarket

def pm_keyset_url(limit: int = 100, after_cursor: str | None = None) -> str:
    """Gamma keyset URL for not-closed PM events (`closed=false`).

    The cursor can contain punctuation, so build the query with `urlencode` rather than string
    concatenation. A bad cursor encoding is the keyset version of the old offset trap: it can
    turn a complete walk into a plausible first page.
    """
    params = {"closed": "false", "limit": str(limit)}
    if after_cursor:
        params["after_cursor"] = after_cursor
    return f"{GAMMA}/events/keyset?{urllib.parse.urlencode(params)}"


def _gamma_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _within_pm_horizon(event: dict, now: dt.datetime, horizon_days: int | None,
                       lookback_days: int) -> bool:
    """Preserve the old feasibility population: endDate in [-30d, +horizon].

    Missing end dates are kept rather than silently dropped, because a parsing gap should not
    shrink the universe invisibly. Known long-dated events are excluded when `horizon_days` is
    not None.
    """
    if horizon_days is None:
        return True
    end = _gamma_time(event.get("endDate"))
    if end is None:
        return True
    return now - dt.timedelta(days=lookback_days) <= end <= now + dt.timedelta(days=horizon_days)


def fetch_pm_events(horizon_days: int | None = 900, *, limit: int = 100,
                    max_pages: int | None = None, lookback_days: int = 30) -> list[dict]:
    """Not-closed PM events with markets nested, using Gamma keyset pagination.

    A 2026-08-17 uncapped walk of the raw `closed=false` query exhausted at 21,075 unique events
    over 211 pages, while legacy offset pagination still fails at offset 2100. Keyset is therefore
    the canonical enumerator. The default `horizon_days=900` preserves the date-bounded population
    used by the original PM x Kalshi feasibility snapshot; pass `None` to keep every not-closed
    event. If `max_pages` is supplied and the cursor is still outstanding at that cap, this raises
    rather than reporting a lower bound as a complete universe.
    """
    out, cursor, pages = [], None, 0
    now = dt.datetime.now(dt.timezone.utc)
    while True:
        d = get(pm_keyset_url(limit=limit, after_cursor=cursor))
        batch = d.get("events") or []
        out += [e for e in batch if _within_pm_horizon(e, now, horizon_days, lookback_days)]
        pages += 1
        cursor = d.get("next_cursor")
        if not cursor or not batch:
            return list({e["id"]: e for e in out}.values())
        if max_pages is not None and pages >= max_pages:
            raise RuntimeError(
                f"PM keyset enumeration hit the {max_pages}-page cap with cursor still "
                f"outstanding: {len(out)} events collected but the universe is NOT complete.")


def pm_taker_fee(price: float, shares: float, rate: float) -> float:
    """Polymarket taker fee in dollars: shares x rate x P x (1-P). Not rounded.

    `rate` must come from the market's own Gamma `feeSchedule`. See the module docstring for why
    CLOB `/fee-rate` cannot supply it.
    """
    return shares * rate * price * (1.0 - price)


def pm_fee_rate(market: dict) -> float:
    """Taker rate for one PM market, 0.0 when fees are off for it.

    Fails closed the other way from most of this codebase: an unrecognised or missing schedule
    on a fees-enabled market returns the WORST rate seen in the wild (0.07) rather than zero,
    so a parsing gap cannot manufacture an opportunity.
    """
    if not market.get("feesEnabled"):
        return 0.0
    sched = market.get("feeSchedule") or {}
    rate = sched.get("rate")
    if not isinstance(rate, (int, float)):
        return 0.07
    if sched.get("exponent") not in (1, 1.0, None):
        return 0.07  # a different exponent is a different formula; do not extrapolate
    return float(rate)


# --------------------------------------------------------------------------- the cheap screen

def hedge_fee_floor(price: float, contracts: float, kalshi_multiplier: float,
                    pm_rate: float) -> float:
    """Combined taker fee in dollars for one hedged pair: buy YES on one venue, NO on the other.

    The two legs sit at P and 1-P, and P(1-P) is symmetric, so BOTH venues charge against the
    same factor -- there is no cheap leg to route into. The floor is therefore

        contracts x P x (1-P) x (0.07 x kalshi_multiplier + pm_rate)

    modulo Kalshi's round-up to the cent, which this applies. Independent of depth, spread and
    matching: it is the floor under everything else, which is why the scanner evaluates it
    before fetching a single order book.
    """
    return (kalshi_taker_fee(price, contracts, kalshi_multiplier)
            + pm_taker_fee(price, contracts, pm_rate))


def kalshi_fee_multiplier(series: dict | None) -> float:
    """Fee multiplier for a Kalshi series, defaulting to full rate when unknown.

    Fails closed: an unrecognised `fee_type` returns 1.0 rather than 0.0, so a series whose
    terms we cannot read is never mistaken for a free one. 14 series genuinely charge nothing
    and they say so explicitly with `fee_multiplier: 0`.
    """
    if not series:
        return 1.0
    if str(series.get("fee_type", "")).startswith("quadratic"):
        m = series.get("fee_multiplier")
        return float(m) if isinstance(m, (int, float)) else 1.0
    return 1.0
