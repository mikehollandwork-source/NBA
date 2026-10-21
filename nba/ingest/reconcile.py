"""Per-season outcome table.

* Official game list = nba.com shot-chart game IDs (regular season only).
* ESPN official box scores = truth; mapped to nba.com game IDs by teams + date,
  tie-broken by team field goals made (teams can meet on back-to-back days).
* Seasons with nba.com CDN play-by-play (2020-21+): audit every stat, add exact
  minutes, fill players ESPN dropped, and attach nba.com player IDs.
* 2017-18 to 2019-20: nba.com player IDs attached via the defender-matchup file.
"""
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from nba.config import PROC, RAW

ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
STAT_COLS = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "pf"]


def norm_last(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().replace(".", "")
    toks = [t for t in s.replace("-", " ").split() if t not in SUFFIXES]
    return toks[-1] if toks else ""


def official_games(season: int) -> pd.DataFrame:
    s = pd.read_csv(RAW / f"bulk/shotdetail_{season}/shotdetail_{season}.csv",
                    usecols=["GAME_ID", "GAME_DATE", "HTM", "VTM", "TEAM_NAME", "SHOT_MADE_FLAG", "TEAM_ID"])
    g = s.groupby("GAME_ID").agg(date=("GAME_DATE", "first"), htm=("HTM", "first"), vtm=("VTM", "first"),
                                  fgm=("SHOT_MADE_FLAG", "sum")).reset_index()
    g["date"] = pd.to_datetime(g.date.astype(str))
    g["teams"] = [tuple(sorted(x)) for x in zip(g.htm, g.vtm)]
    return g.rename(columns={"GAME_ID": "gameId"})


def _match_games(e: pd.DataFrame, og: pd.DataFrame) -> pd.DataFrame:
    """Optimal one-to-one matching within each team pair. ESPN dates are UTC, so
    ESPN date - local date is 0 or 1; ties (back-to-back meetings) broken by FGM."""
    eg = e.groupby("event_id").agg(d=("game_date", "first"), teams=("team", lambda x: tuple(sorted(set(x)))),
                                   fgm=("fgm", "sum")).reset_index()
    eg["d"] = pd.to_datetime(eg.d)
    out = []
    ogg = {k: d for k, d in og.groupby("teams")}
    for k, de in eg.groupby("teams"):
        do = ogg.get(k)
        if do is None:
            continue
        delta = (de.d.to_numpy()[:, None] - do.date.to_numpy()[None, :]) / np.timedelta64(1, "D")
        cost = (~np.isin(delta, [0, 1])) * 1e6 + np.abs(de.fgm.to_numpy()[:, None] - do.fgm.to_numpy()[None, :])
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < 1e6
        out.append(pd.DataFrame({"event_id": de.event_id.to_numpy()[r[ok]], "gameId": do.gameId.to_numpy()[c[ok]]}))
    return pd.concat(out, ignore_index=True)


def _assign(e: pd.DataFrame, p: pd.DataFrame, key_min: str, key_pts: str | None) -> pd.DataFrame:
    """Per team-game optimal assignment on last name (+ minutes/points), then a
    season-wide majority vote so each ESPN id maps to one nba.com id."""
    pairs = []
    pg = {k: d for k, d in p.groupby(["gameId", "team"])}
    for k, de in e.groupby(["gameId", "team"]):
        dp = pg.get(k)
        if dp is None:
            continue
        cost = ((de.ln.to_numpy()[:, None] != dp.ln.to_numpy()[None, :]) * 100
                + np.abs(de["min"].to_numpy()[:, None] - dp[key_min].to_numpy()[None, :]))
        if key_pts:
            cost = cost + np.abs(de.pts.to_numpy()[:, None] - dp[key_pts].to_numpy()[None, :])
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < 100
        pairs.append(pd.DataFrame({"espn_id": de.espn_id.to_numpy()[r[ok]], "nba_id": dp.nba_id.to_numpy()[c[ok]]}))
    votes = pd.concat(pairs).value_counts().rename("n").reset_index()
    return votes.drop_duplicates("espn_id").drop_duplicates("nba_id")[["espn_id", "nba_id"]]


def _load_espn(season: int) -> pd.DataFrame:
    e = pd.read_parquet(PROC / f"espn_box_{season}.parquet")
    e["team"] = e.team.replace(ESPN_TO_NBA)
    e["opp"] = e.opp.replace(ESPN_TO_NBA)
    e["ln"] = e.player.map(norm_last)
    return e


def audit(m: pd.DataFrame) -> pd.DataFrame:
    both = m[m._merge == "both"]
    rows = []
    for s in STAT_COLS:
        diff = both[s] - both[f"{s}_pbp"]
        rows.append({"stat": s, "exact_match_pct": round(100 * (diff == 0).mean(), 2), "rows_off": int((diff != 0).sum())})
    d = both["min"] - both["min_pbp"]
    rows.append({"stat": "min(±1)", "exact_match_pct": round(100 * (d.abs() <= 1).mean(), 2), "rows_off": int((d.abs() > 1).sum())})
    return pd.DataFrame(rows)


def _full_key(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalpha())


def _patch_blank_games(e: pd.DataFrame, og: pd.DataFrame, season: int) -> tuple[pd.DataFrame, int]:
    """Replace ESPN games whose box score is blank with Basketball-Reference."""
    from nba.ingest import bbref
    blank = e.groupby("gameId").dnp.mean()
    blank = blank[blank == 1].index
    if len(blank) == 0:
        return e, 0
    ids = e[e.dnp == 0].assign(k=e.player.map(_full_key)).groupby(["team", "k"]).agg(
        espn_id=("espn_id", "first"), pos=("pos", lambda x: x.mode().iloc[0] if len(x) else ""))
    patched = []
    for gid in blank:
        g = og[og.gameId == gid].iloc[0]
        b = bbref.box(g.date, g.htm, g.vtm)
        b["k"] = b.player.map(_full_key)
        b = b.join(ids, on=["team", "k"])
        b["espn_id"] = b.espn_id.fillna("bbref_" + b.k)
        meta = e[e.gameId == gid].iloc[0]
        b = b.assign(gameId=gid, event_id=meta.event_id, game_date=meta.game_date, dnp=0, dnp_reason="",
                     season=season, ln=b.player.map(norm_last), pos=b.pos.fillna(""), source="bbref")
        patched.append(b.drop(columns="k"))
    e = pd.concat([e[~e.gameId.isin(blank)], pd.concat(patched)], ignore_index=True)
    return e, len(blank)


def build(season: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    e = _load_espn(season)
    og = official_games(season)
    e = e.merge(_match_games(e, og), on="event_id", how="inner")
    e, n_patched = _patch_blank_games(e, og, season)
    played = e[e.dnp == 0].copy()
    report = {"season": season, "official_games": len(og), "espn_games_matched": e.gameId.nunique(),
              "games_from_bbref": n_patched}
    aud = None
    cdn = RAW / f"bulk/cdnnba_{season}/cdnnba_{season}.csv"
    if cdn.exists():
        from nba.ingest import pbp_box
        pp = PROC / f"pbp_box_{season}.parquet"
        p = pd.read_parquet(pp) if pp.exists() else pbp_box.build(cdn, season)
        names = pd.read_csv(cdn, usecols=["gameId", "personId", "playerName"], low_memory=False).dropna() \
            .drop_duplicates(["gameId", "personId"]).rename(columns={"personId": "nba_id"})
        p = p.merge(names, on=["gameId", "nba_id"], how="left")
        p["ln"] = p.playerName.map(norm_last)
        played = played.merge(_assign(played, p, "min_pbp", "pts"), on="espn_id", how="left")
        m = played.merge(p, on=["gameId", "nba_id"], how="outer", suffixes=("", "_pbp"), indicator=True)
        aud = audit(m).assign(season=season)
        both = m[m._merge == "both"].copy(); both["source"] = both.get("source", "espn")
        both["source"] = both.source.fillna("espn") if "source" in both else "espn"
        fill = m[(m._merge == "right_only") & (m.min_pbp > 0) & m.playerName.notna()].copy()
        for s in STAT_COLS:
            fill[s] = fill[f"{s}_pbp"]
        fill["team"] = fill.team_pbp
        fill["min"] = fill.min_pbp.round()
        fill["player"] = fill.playerName
        fill["espn_id"] = "nba_" + fill.nba_id.astype("int64").astype(str)
        fill["source"] = "pbp_fill"
        gi = e.drop_duplicates(["gameId", "team"]).set_index(["gameId", "team"])[["opp", "home", "game_date"]]
        fill = fill.drop(columns=["opp", "home", "game_date"]).join(gi, on=["gameId", "team"])
        t = pd.concat([both, fill], ignore_index=True)
        t["min_exact"] = t.min_pbp
        report["pbp_filled_rows"] = len(fill)
    else:
        t = played.copy(); t["source"] = t.get("source", pd.Series("espn", index=t.index)).fillna("espn")
        t["min_exact"] = np.nan
        mp = RAW / f"bulk/matchups_{season}/matchups_{season}.csv"
        if mp.exists():
            mu = pd.read_csv(mp, usecols=["game_id", "person_id", "family_name", "team_tricode", "matchup_minutes"])
            mm = mu.matchup_minutes.str.split(":", expand=True).astype(float)
            mu["mm"] = mm[0] + mm[1] / 60
            p = mu.groupby(["game_id", "person_id", "team_tricode", "family_name"]).mm.sum().reset_index()
            p.columns = ["gameId", "nba_id", "team", "family_name", "min_m"]
            p["ln"] = p.family_name.map(norm_last)
            t = t.merge(_assign(t, p, "min_m", None), on="espn_id", how="left")
        else:
            t["nba_id"] = np.nan
    t["season"] = season
    t["min"] = t.min_exact.fillna(t["min"])
    cols = ["season", "gameId", "game_date", "espn_id", "nba_id", "player", "pos", "team", "opp", "home", "starter",
            "min", "plus_minus", "source"] + STAT_COLS
    t = t.reindex(columns=cols)
    t = t[t.team.notna() & t.opp.notna()]
    t.to_parquet(PROC / f"player_games_{season}.parquet", index=False)
    report.update(rows=len(t), nba_id_mapped_pct=round(100 * t.nba_id.notna().mean(), 1))
    return t, aud, report
