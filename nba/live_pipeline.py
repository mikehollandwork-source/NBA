"""Live game-day projections with the full backtested model.

Steps for a date D (season = LIVE_SEASON):
  1. ESPN box scores for every completed game this season before D
  2. D's slate, current rosters and ESPN injury report (team codes mapped to nba.com)
  3. "pending" rows for tonight: rostered players not listed OUT
  4. preseason prior for this season (preseason model retrained on all past seasons)
  5. the same pregame features as the backtests (prior games only)
  6. the game-day ensemble (minutes x rate, direct, residual), trained once on all
     past seasons and cached in data/models/
  7. calibrated 10/50/90% ranges (negative binomial + within-game correlation)
Lines/odds are never used.
"""
import json
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from nba.config import PROC, ROOT, LIVE_SEASON, SEASON_WINDOWS
from nba.ingest import espn, espn_athletes
from nba.analysis import features, drivers, season_sim as ss
from nba.analysis.groups import all_features

B = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
TEAM = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
MODELS = ROOT / "data" / "models"
LIVE = ROOT / "data" / "live"
COMP = ["pts", "reb", "ast", "tov"]
COMBOS = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
EXTRA = ["fg3m", "stl", "blk"]  # direct model only (least predictable stats)
STAT_COLS = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "pf"]
tm = lambda x: TEAM.get(x, x)


# ---------------------------------------------------------------- data
def completed_games(season: int, before: date) -> pd.DataFrame:
    start, _ = SEASON_WINDOWS[season]
    end = (before - timedelta(days=1)).isoformat()
    if end < start:
        return pd.DataFrame()
    ev = espn.regular_season_event_ids(start, end, season=season)
    rows = [r for e in ev for r in espn.parse_box(espn.fetch_summary(e["event_id"], season))]
    if not rows:
        return pd.DataFrame()
    b = pd.DataFrame(rows)
    b = b[b.dnp == 0].copy()
    b["team"], b["opp"] = b.team.map(tm), b.opp.map(tm)
    b = b[~b.team.isin(["EAST", "WEST", "STARS", "WORLD", "STRIPES", "USA", "LEB", "STE"])]
    b["gameId"] = b.event_id.astype("int64")
    b["season"], b["nba_id"], b["source"] = season, np.nan, "espn_live"
    return b


def slate(d: date):
    sb = requests.get(f"{B}/scoreboard", params={"dates": d.strftime("%Y%m%d")}, timeout=30).json()
    games, teams = [], {}
    for e in sb.get("events", []):
        if e["season"]["type"] != 2:
            continue
        c = e["competitions"][0]["competitors"]
        h = next(x for x in c if x["homeAway"] == "home"); a = next(x for x in c if x["homeAway"] == "away")
        games.append({"event_id": e["id"], "tip_utc": e["date"], "home": tm(h["team"]["abbreviation"]),
                      "away": tm(a["team"]["abbreviation"]), "status": e["status"]["type"]["description"]})
        for x, o, hm in ((h, a, 1), (a, h, 0)):
            teams[x["team"]["id"]] = (tm(x["team"]["abbreviation"]), tm(o["team"]["abbreviation"]), hm, e["id"], e["date"][:10])
    rows = []
    for tid, (ab, opp, hm, eid, gd) in teams.items():
        for a in requests.get(f"{B}/teams/{tid}/roster", timeout=30).json().get("athletes", []):
            inj = a.get("injuries") or []
            rows.append({"team": ab, "opp": opp, "home": hm, "event_id": eid, "game_date": gd, "pid": str(a["id"]),
                         "player": a["displayName"], "pos": (a.get("position") or {}).get("abbreviation", ""),
                         "height_in": a.get("height"), "weight_lb": a.get("weight"), "dob": (a.get("dateOfBirth") or "")[:10],
                         "status": inj[0].get("status", "") if inj else "",
                         "injury": (inj[0].get("details") or {}).get("type", "") if inj else ""})
    R = pd.DataFrame(rows)
    inj = requests.get(f"{B}/injuries", timeout=30).json()
    st = {}
    for t in inj.get("injuries", []):
        for i in t.get("injuries", []):
            a = i.get("athlete", {})
            pid = str(a.get("id", "")) or next((l["href"].split("/id/")[1].split("/")[0] for l in a.get("links", [])
                                                 if "/id/" in l.get("href", "")), "")
            st[pid] = (i.get("status", ""), i.get("shortComment", ""))
    R["status"] = [st.get(p, (s, ""))[0] or s for p, s in zip(R.pid, R.status)]
    R["note"] = [st.get(p, ("", ""))[1] for p in R.pid]
    R["status"] = R.status.replace("", "Active")
    return pd.DataFrame(games), R


def pending_rows(R: pd.DataFrame, season: int) -> pd.DataFrame:
    p = R[R.status != "Out"].copy()
    p["gameId"] = p.event_id.astype("int64")
    p["espn_id"] = p.pid
    p["season"], p["nba_id"], p["source"], p["starter"], p["min"], p["plus_minus"] = season, np.nan, "pending", 0, 1.0, 0
    for c in STAT_COLS:
        p[c] = 0
    return p


# ---------------------------------------------------------------- preseason prior
def preseason_rows(R: pd.DataFrame, season: int) -> pd.DataFrame:
    from nba import live
    f, fh = live.season_features(R.assign(inj_status=R.status), season)
    o = live.preseason_projection(f, fh)
    out = o[["pid"]].assign(season=season)
    for s in ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]:
        out[f"pre_{s}"] = o[f"proj_{s}"].to_numpy()
    out["pre_avail"] = o.proj_avail.to_numpy()
    for k, cols in COMBOS.items():
        out[f"pre_{k}"] = sum(out[f"pre_{c}"] for c in cols)
    pf = pd.read_parquet(PROC / "preseason_final.parquet")
    pf = pd.concat([pf[pf.season != season], out.drop_duplicates("pid")], ignore_index=True)
    pf.to_parquet(PROC / "preseason_final.parquet", index=False)
    return out


# ---------------------------------------------------------------- models
def _fit(tr, feats, y, weight=None, name=None):
    path = MODELS / f"{name}.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path))
    m = lgb.LGBMRegressor(**drivers.PARAMS).fit(tr[feats], tr[y], sample_weight=None if weight is None else tr[weight])
    MODELS.mkdir(parents=True, exist_ok=True)
    m.booster_.save_model(str(path))
    json.dump(feats, open(MODELS / f"{name}.features.json", "w"))
    return m.booster_


def train_models(retrain: bool = False) -> dict:
    if retrain and MODELS.exists():
        for f in MODELS.glob("*"):
            f.unlink()
    tr = drivers.load(0, True)
    M = {"min": (_fit(tr, all_features("min"), "min", name="min"), all_features("min"))}
    trr = tr[tr["min"] >= 3].copy()
    for s in COMP + list(COMBOS):
        feats = all_features(s)
        if s in COMP:
            trr[f"{s}_rate"] = trr[s] / trr["min"]
            M[f"{s}_rate"] = (_fit(trr, feats, f"{s}_rate", weight="min", name=f"{s}_rate"), feats)
        M[f"{s}_direct"] = (_fit(tr, feats, s, name=f"{s}_direct"), feats)
        trd = tr.assign(_dev=tr[s] - tr[f"{s}_szn"]).dropna(subset=["_dev"])
        M[f"{s}_resid"] = (_fit(trd, feats, "_dev", name=f"{s}_resid"), feats)
    for s in EXTRA:
        M[f"{s}_direct"] = (_fit(tr, all_features(s), s, name=f"{s}_direct"), all_features(s))
    return M


def predict(M: dict, te: pd.DataFrame) -> pd.DataFrame:
    P = lambda k: M[k][0].predict(te.reindex(columns=M[k][1]))
    r = te[["season", "gameId", "game_date", "pid", "player", "team", "opp", "home", "gp"]].copy()
    r["proj_min"] = np.clip(P("min"), 0, 48)
    for s in COMP + list(COMBOS):
        two = r.proj_min * np.clip(P(f"{s}_rate"), 0, None) if s in COMP else sum(r[f"{c}_two_stage"] for c in COMBOS[s])
        r[f"{s}_two_stage"] = two
        r[f"{s}_direct"] = P(f"{s}_direct")
        r[f"{s}_residual"] = te[f"{s}_szn"].to_numpy() + P(f"{s}_resid")
        r[f"proj_{s}"] = r[[f"{s}_two_stage", f"{s}_direct", f"{s}_residual"]].mean(axis=1).clip(lower=0)
    for s in EXTRA:
        r[f"proj_{s}"] = np.clip(P(f"{s}_direct"), 0, None)
    return r


NO_HISTORY_MIN = 12.6   # avg minutes (DNPs as 0) of players with no prior NBA season, team games 1-5, 2017-2025
TEAM_MIN, MAX_MIN, ROTATION = 240.0, 40.0, 8


def normalize_minutes(r: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """The model was trained on players who ended up playing, so it gives every
    rostered player regular-player minutes. Fix: players with no NBA history get the
    historical early-season average; then each team is scaled to 240 minutes (cap 40)
    and every counting stat moves in proportion to minutes."""
    r = r.copy()
    nohist = (te.prev_mpg.isna() & (te.gp.fillna(0) == 0)).to_numpy()
    r["no_nba_history"] = nohist
    raw = np.where(nohist, NO_HISTORY_MIN, r.proj_min)
    r["raw_min"] = raw
    new = np.zeros(len(r))
    for t, idx in r.groupby("team").indices.items():
        # coaches cut the end of the bench first: keep the top-8 rotation's model
        # minutes, squeeze players 9+ into what is left (heuristic, not backtested)
        order = idx[np.argsort(-raw[idx])]
        top, rest = order[:ROTATION], order[ROTATION:]
        top_sum = np.minimum(raw[top], MAX_MIN).sum()
        if top_sum >= TEAM_MIN or len(rest) == 0:
            new[order] = np.minimum(raw[order], MAX_MIN) * TEAM_MIN / np.minimum(raw[order], MAX_MIN).sum()
        else:
            new[top] = np.minimum(raw[top], MAX_MIN)
            left = TEAM_MIN - top_sum
            new[rest] = raw[rest] * min(1.0, left / raw[rest].sum())
            if raw[rest].sum() < left:  # short bench: give the remainder back to the top 8
                new[top] *= (TEAM_MIN - new[rest].sum()) / new[top].sum()
    ratio = new / r.proj_min.clip(lower=0.5).to_numpy()
    for s in COMP + list(COMBOS) + EXTRA:
        r[f"proj_{s}"] = r[f"proj_{s}"] * ratio
    r["proj_min"] = new
    return r


def ranges(r: pd.DataFrame, n_sims=4000) -> pd.DataFrame:
    Pz = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    var, C = ss.fit_params(Pz, -1)
    mu = r[["proj_pts", "proj_reb", "proj_ast"]].to_numpy()
    sims = ss.draw(np.clip(mu, 0.05, None), var, C, n_sims=n_sims)
    for j, s in enumerate(["pts", "reb", "ast", "pra"]):
        x = sims[:, :, j] if s != "pra" else sims.sum(2)
        q = np.percentile(x, [10, 50, 90], axis=0)
        r[f"{s}_p10"], r[f"{s}_p50"], r[f"{s}_p90"] = q
    return r


# ---------------------------------------------------------------- run
def run(d: date, season: int = LIVE_SEASON, retrain: bool = False) -> pd.DataFrame:
    LIVE.mkdir(parents=True, exist_ok=True)
    done = completed_games(season, d)
    G, R = slate(d)
    if G.empty:
        raise SystemExit(f"no regular-season games on {d}")
    espn_athletes.build(R.pid.unique())
    pend = pending_rows(R, season)
    cols = ["season", "gameId", "game_date", "espn_id", "nba_id", "player", "pos", "team", "opp", "home", "starter",
            "min", "plus_minus", "source"] + STAT_COLS
    pg = pd.concat([done.reindex(columns=cols), pend.reindex(columns=cols)], ignore_index=True)
    pg.to_parquet(PROC / f"player_games_{season}.parquet", index=False)
    preseason_rows(R, season)
    f = features.build([season - 1, season], out_path=PROC / "features_live.parquet")
    te = f[(f.season == season) & f.gameId.isin(pend.gameId) & f.pid.isin(pend.pid)].copy()
    M = train_models(retrain)
    r = ranges(normalize_minutes(predict(M, te).reset_index(drop=True), te.reset_index(drop=True)))
    r = r.merge(R[["pid", "status", "note", "injury"]], on="pid", how="left")
    r.to_parquet(LIVE / f"projections_{d}.parquet", index=False)
    write_sheet(r, G, R, d)
    return r


def write_sheet(r, G, R, d):
    out_dir = ROOT / "reports" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = r[r.proj_min >= 10].sort_values(["gameId", "team", "proj_min"], ascending=[True, True, False])
    cols = ["team", "opp", "player", "status", "proj_min", "proj_pts", "pts_p10", "pts_p90", "proj_reb", "reb_p10", "reb_p90",
            "proj_ast", "ast_p10", "ast_p90", "proj_pra", "pra_p10", "pra_p90", "proj_fg3m", "proj_stl", "proj_blk", "proj_tov"]
    keep[cols].round(1).to_csv(out_dir / f"game_day_{d}.csv", index=False)
    L = [f"# Game-day projections — {d:%a %b %-d, %Y}", "",
         "Full game-day model (same as the backtests): no betting lines used. 80% range = 10th–90th percentile.",
         "Teammates-out adjusts automatically once a team has played 3+ games this season.",
         "Team minutes = 240: top-8 keep model minutes, bench squeezed (rule, not backtested); * = no NBA history (rookie), given the early-season average.", ""]
    for _, g in G.sort_values("tip_utc").iterrows():
        tip = pd.Timestamp(g.tip_utc).tz_convert("America/New_York").strftime("%-I:%M %p ET")
        L.append(f"## {g.away} @ {g.home} — {tip}")
        for t in (g.away, g.home):
            x = keep[keep.team == t]
            L += [f"**{t}** | MIN | PTS (80%) | REB | AST | PRA (80%) | 3PM", "|---|---|---|---|---|---|---|"]
            for _, q in x.iterrows():
                nm = q.player + (" (DTD)" if q.status == "Day-To-Day" else "") + (" *" if q.no_nba_history else "")
                L.append(f"{nm} | {q.proj_min:.0f} | {q.proj_pts:.1f} ({q.pts_p10:.0f}–{q.pts_p90:.0f}) | {q.proj_reb:.1f} | "
                         f"{q.proj_ast:.1f} | {q.proj_pra:.1f} ({q.pra_p10:.0f}–{q.pra_p90:.0f}) | {q.proj_fg3m:.1f}")
            o = R[(R.team == t) & (R.status == "Out")]
            if len(o):
                L.append("OUT: " + ", ".join(o.player))
            L.append("")
    (out_dir / f"game_day_{d}.md").write_text("\n".join(L))
