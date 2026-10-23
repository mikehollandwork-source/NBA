"""Market check for picks (user rule, 2026-10-23): a 70%+ lean is published as a ✅ PICK only if
the same line, or a better one, is offered on Kalshi or DraftKings at a price our probability beats
(positive expected return after Kalshi's fee; DraftKings' price already includes its cut).

Prices are used ONLY to decide which picks are published and to grade them; they are never model
inputs. Our probability is the calibrated lean probability at the player's own-average line.
A better line keeps that probability as a floor (over: market line ≤ ours; under: market line ≥ ours).

Kalshi: free; every rung from the live ladder (PTS/REB/AST/PRA only), every run.
Polymarket: free, no fees on these markets; player Points/Rebounds/Assists O/U inside each game event
(gamma-api slug nba-<away>-<home>-<date>), every run. Over = YES at best ask, under = NO at 1 - best bid.
DraftKings: The Odds API (needs ODDS_API_KEY). Costs credits, so it is pulled only at a game's final
lock (1 hour before tip), only for the stats our picks need, and skipped when credits run low.
"""
import os

import numpy as np
import pandas as pd
import requests

from nba.kalshi import key as name_key

ODDS = "https://api.the-odds-api.com/v4/sports/basketball_nba"
ODDS_KEY = (os.environ.get("ODDS_API_KEY") or "").strip() or None
CREDIT_RESERVE = 25
MARKET = {"pts": "player_points", "reb": "player_rebounds", "ast": "player_assists",
          "pra": "player_points_rebounds_assists", "pr": "player_points_rebounds",
          "pa": "player_points_assists", "ra": "player_rebounds_assists"}
STAT_OF = {v: k for k, v in MARKET.items()}
TEAM_NAME = {  # Odds API full names -> nba.com codes
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"}
credits_left = None


def american_to_cost(odds: float) -> float:
    """Break-even probability of an American price (e.g. -130 -> 0.565, +120 -> 0.455)."""
    odds = float(odds)
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def american_text(odds: float) -> str:
    return f"{int(odds):+d}"


# ---------------------------------------------------------------- offers
def kalshi_offers(kb: pd.DataFrame) -> pd.DataFrame:
    """Every Kalshi rung as two offers: over (k-0.5) at ask+fee, under (k-0.5) at 1-bid+fee."""
    if kb is None or kb.empty:
        return pd.DataFrame()
    k = kb[kb.status.astype(str).eq("active")] if "status" in kb else kb
    over = pd.DataFrame({"pid": k.pid.astype(str), "stat": k.stat, "side": "over", "line": k.k - 0.5,
                         "cost": k.cost_yes, "price_txt": [f"YES {s.upper()} {int(x)}+ @ {a * 100:.0f}¢"
                                                          for s, x, a in zip(k.stat, k.k, k.yes_ask)],
                         "volume": k.volume, "ok": k.yes_ask > 0})
    under = pd.DataFrame({"pid": k.pid.astype(str), "stat": k.stat, "side": "under", "line": k.k - 0.5,
                          "cost": k.cost_no, "price_txt": [f"NO {s.upper()} {int(x)}+ @ {(1 - b) * 100:.0f}¢"
                                                           for s, x, b in zip(k.stat, k.k, k.yes_bid)],
                          "volume": k.volume, "ok": k.yes_bid > 0})
    o = pd.concat([over, under], ignore_index=True)
    o = o[o.ok].drop(columns="ok")
    o["venue"] = "Kalshi"
    o["payout"] = (1 - o.cost) / o.cost      # profit per 1u staked if it wins
    return o


PM = "https://gamma-api.polymarket.com/events"
PM_STAT = {"points": "pts", "rebounds": "reb", "assists": "ast"}


def pm_offers(game, roster: pd.DataFrame) -> pd.DataFrame:
    """Polymarket player props for one game (free; zero-fee markets)."""
    day = pd.Timestamp(game.tip_utc).tz_convert("America/New_York").date()
    try:
        ev = requests.get(PM, params={"slug": f"nba-{game.away.lower()}-{game.home.lower()}-{day}"}, timeout=30).json()
    except Exception as ex:
        print("Polymarket unavailable:", ex, flush=True)
        return pd.DataFrame()
    if not ev:
        return pd.DataFrame()
    by_name = dict(zip(roster.player.map(name_key), roster.pid.astype(str)))
    rows = []
    for m in ev[0].get("markets", []):
        stat = PM_STAT.get(m.get("sportsMarketType"))
        q = str(m.get("question", ""))
        if stat is None or ":" not in q or m.get("line") is None or not m.get("acceptingOrders") or m.get("closed"):
            continue
        pid = by_name.get(name_key(q.split(":")[0]))
        if pid is None:
            continue
        fee = float((m.get("feeSchedule") or {}).get("rate", 0) or 0)  # 0 on these markets
        line = float(m["line"])
        ask, bid = m.get("bestAsk"), m.get("bestBid")
        if ask and 0 < float(ask) < 1:
            c = float(ask) + fee * float(ask) * (1 - float(ask))
            rows.append({"pid": pid, "stat": stat, "side": "over", "line": line, "cost": c,
                         "price_txt": f"PM o{line:g} @ {float(ask) * 100:.0f}¢", "liquidity": m.get("liquidityNum")})
        if bid and 0 < float(bid) < 1:
            c = (1 - float(bid)) + fee * float(bid) * (1 - float(bid))
            rows.append({"pid": pid, "stat": stat, "side": "under", "line": line, "cost": c,
                         "price_txt": f"PM u{line:g} @ {(1 - float(bid)) * 100:.0f}¢", "liquidity": m.get("liquidityNum")})
    o = pd.DataFrame(rows)
    if len(o):
        o["venue"] = "Polymarket"
        o["payout"] = (1 - o.cost) / o.cost
        o["volume"] = np.nan
    return o


def _odds_get(path, **params):
    global credits_left
    r = requests.get(f"{ODDS}{path}", params={"apiKey": ODDS_KEY, **params}, timeout=30)
    if "x-requests-remaining" in r.headers:
        try:
            credits_left = float(r.headers["x-requests-remaining"])
        except ValueError:
            pass
    r.raise_for_status()
    return r.json()


def dk_events() -> pd.DataFrame:
    """Upcoming NBA events (this endpoint costs no credits)."""
    ev = _odds_get("/events")
    return pd.DataFrame([{"odds_id": e["id"], "home": TEAM_NAME.get(e["home_team"]), "away": TEAM_NAME.get(e["away_team"]),
                          "commence": e["commence_time"]} for e in ev])


def parse_dk(payload: dict, roster: pd.DataFrame) -> pd.DataFrame:
    """Odds API event-odds payload -> offers (pid, stat, side, line, cost, price_txt, payout)."""
    by_name = dict(zip(roster.player.map(name_key), roster.pid.astype(str)))
    rows = []
    for b in payload.get("bookmakers", []):
        if b.get("key") != "draftkings":
            continue
        for mk in b.get("markets", []):
            stat = STAT_OF.get(mk.get("key"))
            for o in mk.get("outcomes", []):
                pid = by_name.get(name_key(o.get("description", "")))
                side = str(o.get("name", "")).lower()
                if stat is None or pid is None or side not in ("over", "under") or o.get("point") is None:
                    continue
                cost = american_to_cost(o["price"])
                rows.append({"pid": pid, "stat": stat, "side": side, "line": float(o["point"]), "cost": cost,
                             "price_txt": f"DK {side[0]}{o['point']:g} {american_text(o['price'])}",
                             "payout": (1 - cost) / cost, "venue": "DraftKings", "volume": np.nan})
    return pd.DataFrame(rows)


def dk_offers(game, stats: set, roster: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    if not ODDS_KEY or not stats or events is None or events.empty:
        return pd.DataFrame()
    if credits_left is not None and credits_left < CREDIT_RESERVE + len(stats):
        print(f"Odds API credits low ({credits_left:.0f}); DraftKings skipped", flush=True)
        return pd.DataFrame()
    e = events[(events.home == game.home) & (events.away == game.away)]
    if e.empty:
        return pd.DataFrame()
    try:
        payload = _odds_get(f"/events/{e.odds_id.iloc[0]}/odds", regions="us", bookmakers="draftkings",
                            markets=",".join(MARKET[s] for s in sorted(stats)), oddsFormat="american")
    except Exception as ex:
        print("DraftKings unavailable:", ex, flush=True)
        return pd.DataFrame()
    return parse_dk(payload, roster)


# ---------------------------------------------------------------- matching
def best_offer(q, offers: pd.DataFrame):
    """Best-EV offer at the same or a better line for this pick, or None."""
    if offers is None or offers.empty:
        return None
    o = offers[(offers.pid == str(q.pid)) & (offers.stat == q.lean_stat) & (offers.side == q.lean_side)]
    o = o[o.line <= q.lean_ref] if q.lean_side == "over" else o[o.line >= q.lean_ref]
    if o.empty:
        return None
    o = o.assign(ev=q.lean_prob_cal / o.cost - 1)
    return o.sort_values(["ev", "line"], ascending=[False, q.lean_side == "over"]).iloc[0]


PUB_COLS = ["pub", "pub_venue", "pub_line", "pub_cost", "pub_payout", "pub_ev", "pub_price", "pub_checked"]


def check(r: pd.DataFrame, G: pd.DataFrame, kb: pd.DataFrame, writable: set, now, strong_min: float,
          lock_lead_min: int) -> pd.DataFrame:
    """Add pub_* columns: which 70%+ leans are published picks (market available, price beats us)."""
    r = r.copy()
    for c in ("pub_line", "pub_cost", "pub_payout", "pub_ev"):
        r[c] = np.nan
    for c in ("pub_venue", "pub_price", "pub_checked"):
        r[c] = pd.Series([None] * len(r), index=r.index, dtype=object)
    r["pub"] = False
    strong = (r.lean_prob_cal.round(2) >= strong_min) & (r.proj_min >= 10) & \
        r.projected.astype(str).str.lower().eq("true") & r.lean_stat.apply(lambda v: isinstance(v, str))
    ko = kalshi_offers(kb)
    events = None
    for _, g in G[G.event_id.isin(writable)].iterrows():
        rows = r[(r.gameId == int(g.event_id)) & strong]
        if rows.empty:
            continue
        final = now >= pd.Timestamp(g.tip_utc) - pd.Timedelta(minutes=lock_lead_min)
        dk = pd.DataFrame()
        if final and ODDS_KEY:
            if events is None:
                try:
                    events = dk_events()
                except Exception as ex:
                    print("Odds API events unavailable:", ex, flush=True)
                    events = pd.DataFrame()
            dk = dk_offers(g, set(rows.lean_stat), r[r.gameId == int(g.event_id)], events)
        pm = pm_offers(g, r[r.gameId == int(g.event_id)])
        offers = pd.concat([x for x in (ko, pm, dk) if len(x)], ignore_index=True) if (len(ko) or len(pm) or len(dk)) \
            else pd.DataFrame()
        for i, q in rows.iterrows():
            b = best_offer(q, offers)
            r.loc[i, "pub_checked"] = "Kalshi+Polymarket+DK" if (final and ODDS_KEY) else "Kalshi+Polymarket"
            if b is not None and b.ev > 0:
                r.loc[i, "pub"] = True
                r.loc[i, "pub_venue"], r.loc[i, "pub_price"] = b.venue, b.price_txt
                r.loc[i, ["pub_line", "pub_cost", "pub_payout", "pub_ev"]] = [b.line, b.cost, b.payout, b.ev]
    return r
