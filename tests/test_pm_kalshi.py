"""Unit tests for the arithmetic the PM x Kalshi feasibility pass turns on.

Only the pure functions are tested. Everything that touches a venue is left to the live probes,
because a mocked order book proves the mock is consistent and nothing else -- and the failure
mode this whole directory exists to guard against is a plausible number derived from a wrong
assumption about a real API.
"""

import os
import sys
import datetime as dt

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import venues  # noqa: E402
from walk_hedge import derive_ask_ladder, walk  # noqa: E402
import scan_pm_complements as comp  # noqa: E402


# --------------------------------------------------------------------------- fees

def test_kalshi_fee_is_quadratic_and_peaks_at_half():
    """The whole idea's first-order gate: the fee is maximal exactly where the depth is."""
    fees = [venues.kalshi_taker_fee(p, 10_000.0) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert fees[2] == max(fees)
    assert venues.kalshi_taker_fee(0.1, 10_000.0) == pytest.approx(
        venues.kalshi_taker_fee(0.9, 10_000.0))


def test_kalshi_fee_rounds_up_to_the_cent():
    """0.07 x 100 x 0.02 x 0.98 = $0.1372 -> $0.14, not $0.13 and not $0.1372.

    At probe sizes the round-up is a material fraction of the fee, and rounding the wrong way
    manufactures headroom that does not exist.
    """
    assert venues.kalshi_taker_fee(0.02, 100.0) == 0.14


def test_kalshi_multiplier_scales_and_zero_means_free():
    assert venues.kalshi_taker_fee(0.5, 100.0, 0.0) == 0.0
    assert venues.kalshi_taker_fee(0.5, 10_000.0, 0.5) == pytest.approx(
        venues.kalshi_taker_fee(0.5, 10_000.0, 1.0) / 2, abs=0.01)


def test_pm_fee_matches_the_documented_shape():
    assert venues.pm_taker_fee(0.5, 100.0, 0.04) == pytest.approx(1.0)
    assert venues.pm_taker_fee(0.1, 100.0, 0.04) == pytest.approx(0.36)


def test_pm_fee_rate_fails_closed_on_an_unreadable_schedule():
    """An unparseable schedule must return the worst rate seen in the wild, never zero.

    Failing closed the usual way -- returning 0 -- would let a parsing gap manufacture an
    opportunity, which is precisely the error class this directory is built around.
    """
    assert venues.pm_fee_rate({"feesEnabled": True, "feeSchedule": {"exponent": 2, "rate": 0.01}}) == 0.07
    assert venues.pm_fee_rate({"feesEnabled": True, "feeSchedule": {}}) == 0.07
    assert venues.pm_fee_rate({"feesEnabled": True}) == 0.07
    assert venues.pm_fee_rate({"feesEnabled": False, "feeSchedule": {"rate": 0.07}}) == 0.0
    assert venues.pm_fee_rate({"feesEnabled": True,
                               "feeSchedule": {"exponent": 1, "rate": 0.04}}) == 0.04


def test_kalshi_fee_multiplier_fails_closed_on_an_unknown_fee_type():
    assert venues.kalshi_fee_multiplier(None) == 1.0
    assert venues.kalshi_fee_multiplier({"fee_type": "something_new", "fee_multiplier": 0}) == 1.0
    assert venues.kalshi_fee_multiplier({"fee_type": "quadratic", "fee_multiplier": 0}) == 0.0
    assert venues.kalshi_fee_multiplier(
        {"fee_type": "quadratic_with_maker_fees", "fee_multiplier": 0.5}) == 0.5


def test_hedge_floor_is_the_sum_of_both_legs_at_the_same_price_factor():
    """P(1-P) is symmetric, so there is no cheap leg to route into."""
    floor = venues.hedge_fee_floor(0.5, 100.0, 1.0, 0.04)
    assert floor == pytest.approx(venues.kalshi_taker_fee(0.5, 100.0) + 1.0)
    assert venues.hedge_fee_floor(0.5, 100.0, 1.0, 0.04) > venues.hedge_fee_floor(0.1, 100.0, 1.0, 0.04)


# --------------------------------------------------------------------------- universe enumeration

def test_pm_keyset_url_encodes_the_cursor():
    url = venues.pm_keyset_url(limit=25, after_cursor="abc+/=")
    assert "events/keyset?" in url
    assert "closed=false" in url
    assert "limit=25" in url
    assert "after_cursor=abc%2B%2F%3D" in url


def test_pm_keyset_fetch_dedupes_across_pages(monkeypatch):
    pages = [
        {"events": [{"id": "1"}, {"id": "2"}], "next_cursor": "c1"},
        {"events": [{"id": "2"}, {"id": "3"}], "next_cursor": ""},
    ]

    monkeypatch.setattr(venues, "get", lambda _url: pages.pop(0))
    events = venues.fetch_pm_events(limit=2)
    assert [e["id"] for e in events] == ["1", "2", "3"]


def test_pm_keyset_fetch_preserves_the_default_horizon(monkeypatch):
    now = dt.datetime.now(dt.timezone.utc)
    near = (now + dt.timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    far = (now + dt.timedelta(days=1200)).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(
        venues,
        "get",
        lambda _url: {
            "events": [
                {"id": "near", "endDate": near},
                {"id": "far", "endDate": far},
                {"id": "missing"},
            ],
            "next_cursor": "",
        },
    )
    assert [e["id"] for e in venues.fetch_pm_events(limit=3)] == ["near", "missing"]
    assert [e["id"] for e in venues.fetch_pm_events(None, limit=3)] == ["near", "far", "missing"]


def test_pm_keyset_fetch_refuses_a_capped_lower_bound(monkeypatch):
    monkeypatch.setattr(
        venues,
        "get",
        lambda _url: {"events": [{"id": "1"}], "next_cursor": "still-going"},
    )
    with pytest.raises(RuntimeError, match="universe is NOT complete"):
        venues.fetch_pm_events(limit=1, max_pages=1)


# --------------------------------------------------------------------------- books

def test_ask_ladder_inverts_the_opposite_bids_cheapest_first():
    """Kalshi publishes bids on both sides and no asks at all. The cheapest YES offer comes from
    the HIGHEST NO bid, so the ladder must come back ascending in YES price."""
    ladder = derive_ask_ladder([["0.0100", "275"], ["0.0600", "73"], ["0.0300", "40"]])
    assert [round(p, 4) for p, _ in ladder] == [0.94, 0.97, 0.99]
    assert ladder[0][1] == 73.0


def test_ask_ladder_of_an_empty_side_is_empty_not_priced_at_one():
    assert derive_ask_ladder([]) == []
    assert derive_ask_ladder(None) == []


def test_walk_returns_partial_fills_as_partials():
    """A book that fills 30 of 100 is a different fact from one that fills 100. Scaling the
    partial up to the target is how a thin book comes to look tradeable."""
    cost, filled = walk([(0.10, 20.0), (0.12, 10.0)], 100.0)
    assert filled == 30.0
    assert cost == pytest.approx(20 * 0.10 + 10 * 0.12)


def test_walk_consumes_the_cheapest_rungs_first_and_stops_at_the_target():
    cost, filled = walk([(0.10, 50.0), (0.20, 50.0), (0.30, 50.0)], 75.0)
    assert filled == 75.0
    assert cost == pytest.approx(50 * 0.10 + 25 * 0.20)


def test_walk_of_an_empty_ladder_fills_nothing():
    assert walk([], 100.0) == (0.0, 0.0)


# --------------------------------------------------------------------------- PM complement scan

def test_complement_scan_scores_size_walked_yes_and_no(monkeypatch):
    """The intra-PM check must re-walk both ladders to the common fill size."""
    market = comp.Market(
        market_id="m1",
        slug="test",
        question="Test?",
        yes_token="yes",
        no_token="no",
        fee_rate=0.04,
        gamma_ask_sum=0.0,
        end_date=None,
    )

    def fake_ladder(token):
        return {
            "yes": [(0.40, 50.0), (0.41, 50.0)],
            "no": [(0.58, 60.0), (0.59, 20.0)],
        }[token]

    monkeypatch.setattr(comp, "pm_ask_ladder", fake_ladder)
    r = comp.score_market(market, 100.0)
    assert r["status"] == "ok"
    assert r["fill"] == 80.0
    assert r["yes_px"] == pytest.approx((50 * 0.40 + 30 * 0.41) / 80)
    assert r["no_px"] == pytest.approx((60 * 0.58 + 20 * 0.59) / 80)
    assert r["net"] < 0


def test_complement_scan_reports_no_fill_when_one_side_is_empty(monkeypatch):
    market = comp.Market("m1", "test", "Test?", "yes", "no", 0.04, 0.0, None)
    monkeypatch.setattr(comp, "pm_ask_ladder",
                        lambda token: [(0.50, 100.0)] if token == "yes" else [])
    r = comp.score_market(market, 100.0)
    assert r["status"] == "no_fill"
    assert r["yes_fill"] == 100.0
    assert r["no_fill"] == 0.0


def test_reflected_complement_scan_derives_no_ask_from_yes_bid(monkeypatch):
    market = comp.Market("m1", "test", "Test?", "yes", "no", 0.0, 0.0, None)
    monkeypatch.setattr(
        comp,
        "_book_sides",
        lambda token: ([(0.40, 100.0)], [(0.39, 100.0)]),
    )
    r = comp.score_market_reflected(market, 100.0)
    assert r["status"] == "ok"
    assert r["yes_px"] == pytest.approx(0.40)
    assert r["no_px"] == pytest.approx(0.61)
    assert r["ask_sum"] == pytest.approx(1.01)
