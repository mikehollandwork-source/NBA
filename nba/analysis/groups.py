"""Feature groups shared by Test B, the projection models and the disagreement analysis."""

RATE = {"pts", "reb", "ast", "fg3m", "stl", "blk", "tov"}
OPP = {"pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra"}


def groups(s: str, lineup_known: bool = False) -> dict[str, list[str]]:
    rate = [f"{s}_p36_szn"] if s in RATE else []
    g = {
        "season_avg": [f"{s}_szn", "gp"] + rate,
        "last_season": ["prev_mpg", "prev_gp", "prev_start_rate"] + ([f"prev_{s}_p36"] if s in RATE else []),
        "recent_form": [f"{s}_l5", f"{s}_l10"] + ([f"{s}_p36_l10"] if s in RATE else []),
        "minutes_role": ["min_szn", "min_l5", "min_l10", "min_l3", "min_sd", "starter_last", "start_rate_l5", "start_rate_szn"],
        "usage_profile": ["usg_szn", "usg_l10", "usg_share", "fga_szn", "fg3a_szn", "fta_szn"],
        "return_from_absence": ["missed_before", "first_game_back"],
        "teammates_out": ["out_min", "out_usg", "out_ast", "out_reb", "out_starters", "out_min_same_pos",
                          "out_usg_share", "usg_x_out", "n_out"],
        "teammates_back": ["back_min", "back_usg"],
        "opponent_defense": ([f"oppallow_{s}", f"oppallow_{s}_mypos"] if s in OPP else []) + ["opp_net_prior"],
        "opp_absences": ["opp_out_min", "opp_out_usg"],
        "matchup_coverage": ["match_diff", "opp_switch_proxy", "opp_cross_share", "opp_allow_rim", "opp_allow_paint",
                             "opp_allow_mid", "opp_allow_c3", "opp_allow_ab3"],
        "pace": ["exp_pace", "pace_prior", "opp_pace_prior"],
        "game_script": ["net_prior", "net_diff", "abs_net_diff"],
        "rest_home": ["days_rest", "b2b", "opp_b2b", "home"],
        "season_timing": ["team_gnum"],
    }
    if s == "min":
        g["season_avg"] = ["min_szn", "gp"]
        g["recent_form"] = ["min_l5", "min_l10", "min_l3"]
        g["minutes_role"] = ["min_sd", "starter_last", "start_rate_l5", "start_rate_szn"]
    if s in ("pra", "pr", "pa", "ra"):
        g["season_avg"] += ["pts_p36_szn", "reb_p36_szn", "ast_p36_szn"]
    if lineup_known:
        g["starter_tonight"] = ["starter"]
    return g


def all_features(s: str, lineup_known: bool = False) -> list[str]:
    return sorted({c for v in groups(s, lineup_known).values() for c in v})
