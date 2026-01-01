"""NBA model service for outcome and score predictions."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.pipeline import Pipeline

from app.core.nba_config import nba_settings

# Robust pathing: Root of repo is where `app` and `data` reside.
# This file is in app/routes/NBA/nba_service.py
# .parents[0] = app/routes/NBA
# .parents[1] = app/routes
# .parents[2] = app
# .parents[3] = <root>
BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "app" / "models" / "NBA"
DATA_DIR = BASE_DIR / "data" / "NBA"


def get_goalserve_url() -> str:
    """Construct the Goalserve NBA schedule API URL."""
    api_key = nba_settings.goalserve_api_key
    base_url = nba_settings.goalserve_base_url
    if not api_key:
        raise ValueError("GOALSERVE_API_KEY not configured. Please set it in .env file.")
    return f"{base_url}{api_key}/bsktbl/nba-shedule?json=1"


def fetch_goalserve_schedule(timeout: int = 20) -> Dict:
    """Fetch raw schedule JSON from Goalserve. Returns mock data on failure or if configured."""
    if nba_settings.use_mock_data:
        print("Using Mock Data (Configured)")
        return get_mock_schedule()

    url = get_goalserve_url()
    try:
        # Enforce JSON expectation with header
        headers = {"Accept": "application/json"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data
    except (requests.RequestException, ValueError) as e:
        print(f"GoalServe API Error: {e}. Falling back to mock data.")
        return get_mock_schedule()


def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


@lru_cache(maxsize=1)
def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw datasets. Cached in memory to avoid reading from disk on every request."""
    return (
        pd.read_csv(DATA_DIR / "nba_games_dataset.csv", low_memory=False),
        pd.read_csv(DATA_DIR / "nba_player_performances_dataset.csv", low_memory=False),
        pd.read_csv(DATA_DIR / "injured_data.csv", low_memory=False),
    )


def clean_games(g: pd.DataFrame) -> pd.DataFrame:
    g = g[~g["status"].str.contains("pre", case=False, na=False)].copy()
    non_numeric = {
        "date",
        "formatted_date",
        "status",
        "timezone",
        "time",
        "timer",
        "venue_name",
        "home_name",
        "away_name",
    }
    for col in g.columns:
        if col not in non_numeric:
            g[col] = pd.to_numeric(g[col], errors="coerce")
    g["game_date"] = pd.to_datetime(g["date"], format="%d.%m.%Y", errors="coerce")
    g = g.dropna(subset=["game_date"]).copy()
    g["season"] = g["game_date"].dt.year
    g["home_win"] = (g["home_totalscore"] > g["away_totalscore"]).astype(int)
    g["score_diff"] = g["home_totalscore"] - g["away_totalscore"]
    g["total_points"] = g["home_totalscore"] + g["away_totalscore"]
    return g


def clean_players(p: pd.DataFrame) -> pd.DataFrame:
    p["starter"] = p["starter"].replace({"False": False, "True": True}).astype(bool)
    p["is_home"] = p["is_home"].replace({"False": False, "True": True}).astype(bool)
    num_cols = [
        "minutes",
        "fg_made",
        "fg_attempts",
        "3pt_made",
        "3pt_attempts",
        "ft_made",
        "ft_attempts",
        "oreb",
        "dreb",
        "reb",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "personal_fouls",
        "plus_minus",
        "points",
    ]
    for col in num_cols:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    return p


def aggregate_player_team(p: pd.DataFrame) -> pd.DataFrame:
    def _agg(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "bench_points": df.loc[~df["starter"], "points"].sum(),
                "bench_minutes": df.loc[~df["starter"], "minutes"].sum(),
                "top5_minutes_avg": df.nlargest(5, "minutes")["minutes"].mean(),
                "players_used": df.loc[df["minutes"] > 0, "player_id"].nunique(),
            }
        )

    return p.groupby(["match_id", "team_id", "is_home"]).apply(_agg).reset_index()


def build_team_games(g: pd.DataFrame, player_team_game: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in g.iterrows():
        home_poss = r["home_fg_attempts"] - r["home_oreb"] + r["home_turnovers"] + 0.44 * r["home_ft_attempts"]
        away_poss = r["away_fg_attempts"] - r["away_oreb"] + r["away_turnovers"] + 0.44 * r["away_ft_attempts"]
        pace = (home_poss + away_poss) / 2
        home = {
            "match_id": r["match_id"],
            "team_id": r["home_id"],
            "opponent_id": r["away_id"],
            "is_home": True,
            "game_date": r["game_date"],
            "season": r["season"],
            "points": r["home_totalscore"],
            "opp_points": r["away_totalscore"],
            "fg_pct": r["home_fg_pct"],
            "3pt_pct": r["home_3pt_pct"],
            "ft_pct": r["home_ft_pct"],
            "fg_made": r["home_fg_made"],
            "fg_attempts": r["home_fg_attempts"],
            "3pt_made": r["home_3pt_made"],
            "3pt_attempts": r["home_3pt_attempts"],
            "ft_made": r["home_ft_made"],
            "ft_attempts": r["home_ft_attempts"],
            "oreb": r["home_oreb"],
            "dreb": r["home_dreb"],
            "opp_oreb": r["away_oreb"],
            "opp_dreb": r["away_dreb"],
            "assists": r["home_assists"],
            "steals": r["home_steals"],
            "blocks": r["home_blocks"],
            "turnovers": r["home_turnovers"],
            "possessions": home_poss,
            "opp_possessions": away_poss,
            "pace": pace,
        }
        away = {
            "match_id": r["match_id"],
            "team_id": r["away_id"],
            "opponent_id": r["home_id"],
            "is_home": False,
            "game_date": r["game_date"],
            "season": r["season"],
            "points": r["away_totalscore"],
            "opp_points": r["home_totalscore"],
            "fg_pct": r["away_fg_pct"],
            "3pt_pct": r["away_3pt_pct"],
            "ft_pct": r["away_ft_pct"],
            "fg_made": r["away_fg_made"],
            "fg_attempts": r["away_fg_attempts"],
            "3pt_made": r["away_3pt_made"],
            "3pt_attempts": r["away_3pt_attempts"],
            "ft_made": r["away_ft_made"],
            "ft_attempts": r["away_ft_attempts"],
            "oreb": r["away_oreb"],
            "dreb": r["away_dreb"],
            "opp_oreb": r["home_oreb"],
            "opp_dreb": r["home_dreb"],
            "assists": r["away_assists"],
            "steals": r["away_steals"],
            "blocks": r["away_blocks"],
            "turnovers": r["away_turnovers"],
            "possessions": away_poss,
            "opp_possessions": home_poss,
            "pace": pace,
        }
        rows.extend([home, away])
    t = pd.DataFrame(rows)
    t["efg_pct"] = safe_div(t["fg_made"] + 0.5 * t["3pt_made"], t["fg_attempts"])
    t["ortg"] = safe_div(t["points"], t["possessions"]) * 100
    t["drtg"] = safe_div(t["opp_points"], t["opp_possessions"]) * 100
    t["net_rating"] = t["ortg"] - t["drtg"]
    t["tov_pct"] = safe_div(t["turnovers"], t["possessions"]) * 100
    t["orb_pct"] = safe_div(t["oreb"], t["oreb"] + t["opp_dreb"])
    t["drb_pct"] = safe_div(t["dreb"], t["dreb"] + t["opp_oreb"])
    t["win"] = (t["points"] > t["opp_points"]).astype(int)
    t = t.merge(player_team_game, on=["match_id", "team_id", "is_home"], how="left")
    t[["bench_points", "bench_minutes", "top5_minutes_avg", "players_used"]] = t[
        ["bench_points", "bench_minutes", "top5_minutes_avg", "players_used"]
    ].fillna(0)
    return t


def add_rest_and_elo(tg: pd.DataFrame, k: float = 20.0) -> pd.DataFrame:
    tg = tg.sort_values(["game_date", "match_id", "team_id"]).copy()
    tg["rest_days"] = tg.groupby("team_id")["game_date"].diff().dt.days
    tg["is_b2b"] = (tg["rest_days"] <= 1).astype(int)
    ratings: Dict[int, float] = {}
    elo_pre = []
    for _, row in tg.iterrows():
        team, opp = row["team_id"], row["opponent_id"]
        r_team = ratings.get(team, 1500)
        r_opp = ratings.get(opp, 1500)
        elo_pre.append(r_team)
        expected = 1 / (1 + 10 ** ((r_opp - r_team) / 400))
        ratings[team] = r_team + k * (row["win"] - expected)
    tg["elo_pre"] = elo_pre
    return tg


def prepare_injuries(inj: pd.DataFrame) -> pd.DataFrame:
    inj = inj.copy()
    inj["return_date"] = pd.to_datetime(inj["return_date"], format="%d.%m.%Y", errors="coerce")
    return inj


def merge_injuries(tg: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    tg = tg.copy()
    inj = prepare_injuries(inj)
    counts = []
    names = []
    for _, row in tg.iterrows():
        team_id = row["team_id"]
        gd = row["game_date"]
        active = inj[(inj["team_id"] == team_id) & ((inj["return_date"].isna()) | (inj["return_date"] >= gd))]
        counts.append(len(active))
        names.append(", ".join(active["player_name"].head(3).tolist()))
    tg["injury_count"] = counts
    tg["injury_top3"] = names
    return tg


def rolling_features(tg: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    tg = tg.sort_values(["team_id", "season", "game_date"]).copy()
    grp = tg.groupby(["team_id", "season"], group_keys=False)

    def _esm(series: pd.Series) -> pd.Series:
        return series.shift().expanding().mean()

    metrics = [
        "points",
        "opp_points",
        "net_rating",
        "fg_pct",
        "3pt_pct",
        "ft_pct",
        "efg_pct",
        "ortg",
        "drtg",
        "pace",
        "tov_pct",
        "orb_pct",
        "drb_pct",
        "bench_points",
        "bench_minutes",
        "top5_minutes_avg",
        "players_used",
    ]
    for col in metrics:
        tg[f"prev_{col}"] = grp[col].transform(_esm)
    tg["prev_win_pct"] = grp["win"].transform(_esm)
    tg["recent_ortg_5"] = grp["ortg"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    tg["recent_drtg_5"] = grp["drtg"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    tg["recent_net_5"] = tg["recent_ortg_5"] - tg["recent_drtg_5"]
    tg["games_played"] = grp.cumcount()
    base_feats = [c for c in tg.columns if c.startswith("prev_") or c.startswith("recent_")]
    extra = ["rest_days", "is_b2b", "elo_pre", "injury_count"]
    tg_ready = tg.dropna(subset=base_feats + ["games_played"]).copy()
    return tg_ready, base_feats, extra


def apply_bounds(features: pd.DataFrame, bounds: Dict[str, Tuple[float, float]], cols: List[str]) -> pd.DataFrame:
    clipped = features.copy()
    for col in cols:
        if col in bounds:
            lo, hi = bounds[col]
            clipped[col] = clipped[col].clip(lo, hi)
    return clipped


def round_or_none(val: float, decimals: int = 2):
    if pd.isna(val):
        return None
    return round(float(val), decimals)


def build_prediction_payload(row: pd.Series, home_win_prob: float, total_pred: float, diff_pred: float) -> Dict:
    home_score_pred = (total_pred + diff_pred) / 2
    away_score_pred = total_pred - home_score_pred
    return {
        "game_overview": {
            "matchup": f"{row['home_name']} vs {row['away_name']}",
            "date": row["game_date"].date().isoformat(),
            "venue": row.get("venue_name", ""),
            "home_team": row["home_name"],
            "away_team": row["away_name"],
        },
        "model_probabilities": {
            "home_win_prob_pct": round_or_none(home_win_prob * 100),
            "away_win_prob_pct": round_or_none((1 - home_win_prob) * 100) if home_win_prob is not None else None,
            "model_confidence_0_100": round_or_none(home_win_prob * 100),
            "ai_confidence_pct": round_or_none(home_win_prob * 100),
        },
        "core_home": {
            "season_win_pct": round_or_none(row.get("home_prev_win_pct", 0) * 100),
            "ppg": round_or_none(row.get("home_prev_points", 0)),
            "opp_ppg": round_or_none(row.get("home_prev_opp_points", 0)),
            "net_rating": round_or_none(row.get("home_prev_net_rating", 0)),
            "fg_pct": round_or_none(row.get("home_prev_fg_pct", 0)),
            "three_pt_pct": round_or_none(row.get("home_prev_3pt_pct", 0)),
            "ft_pct": round_or_none(row.get("home_prev_ft_pct", 0)),
            "efg_pct": round_or_none(row.get("home_prev_efg_pct", 0)),
        },
        "core_away": {
            "season_win_pct": round_or_none(row.get("away_prev_win_pct", 0) * 100),
            "ppg": round_or_none(row.get("away_prev_points", 0)),
            "opp_ppg": round_or_none(row.get("away_prev_opp_points", 0)),
            "net_rating": round_or_none(row.get("away_prev_net_rating", 0)),
            "fg_pct": round_or_none(row.get("away_prev_fg_pct", 0)),
            "three_pt_pct": round_or_none(row.get("away_prev_3pt_pct", 0)),
            "ft_pct": round_or_none(row.get("away_prev_ft_pct", 0)),
            "efg_pct": round_or_none(row.get("away_prev_efg_pct", 0)),
        },
        "efficiency_home": {
            "ortg": round_or_none(row.get("home_prev_ortg", 0)),
            "drtg": round_or_none(row.get("home_prev_drtg", 0)),
            "pace": round_or_none(row.get("home_prev_pace", 0)),
            "tov_pct": round_or_none(row.get("home_prev_tov_pct", 0)),
            "orb_pct": round_or_none(row.get("home_prev_orb_pct", 0)),
            "drb_pct": round_or_none(row.get("home_prev_drb_pct", 0)),
            "recent_ortg_5": round_or_none(row.get("home_recent_ortg_5", 0)),
            "recent_drtg_5": round_or_none(row.get("home_recent_drtg_5", 0)),
            "recent_net_5": round_or_none(row.get("home_recent_net_5", 0)),
        },
        "efficiency_away": {
            "ortg": round_or_none(row.get("away_prev_ortg", 0)),
            "drtg": round_or_none(row.get("away_prev_drtg", 0)),
            "pace": round_or_none(row.get("away_prev_pace", 0)),
            "tov_pct": round_or_none(row.get("away_prev_tov_pct", 0)),
            "orb_pct": round_or_none(row.get("away_prev_orb_pct", 0)),
            "drb_pct": round_or_none(row.get("away_prev_drb_pct", 0)),
            "recent_ortg_5": round_or_none(row.get("away_recent_ortg_5", 0)),
            "recent_drtg_5": round_or_none(row.get("away_recent_drtg_5", 0)),
            "recent_net_5": round_or_none(row.get("away_recent_net_5", 0)),
        },
        "lineup_bench": {
            "home_bench_ppg": round_or_none(row.get("home_prev_bench_points", 0)),
            "away_bench_ppg": round_or_none(row.get("away_prev_bench_points", 0)),
            "home_rotation_size": round_or_none(row.get("home_prev_players_used", 0)),
            "away_rotation_size": round_or_none(row.get("away_prev_players_used", 0)),
            "home_top5_minutes_avg": round_or_none(row.get("home_prev_top5_minutes_avg", 0)),
            "away_top5_minutes_avg": round_or_none(row.get("away_prev_top5_minutes_avg", 0)),
        },
        "prediction": {
            "predicted_winner": row["home_name"] if home_win_prob >= 0.5 else row["away_name"],
            "predicted_score": {"home": int(round(home_score_pred)), "away": int(round(away_score_pred))},
            "predicted_total_points": round_or_none(total_pred),
            "predicted_home_point_diff": round_or_none(diff_pred),
        },
    }


@lru_cache(maxsize=1)
def load_artifacts() -> Tuple[Pipeline, Dict[str, Pipeline], Dict]:
    with open(MODEL_DIR / "win_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open(MODEL_DIR / "score_regressors.pkl", "rb") as f:
        regressors = pickle.load(f)
    with open(MODEL_DIR / "model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return classifier, regressors, metadata


@lru_cache(maxsize=1)
def prepare_historical_data() -> Tuple[pd.DataFrame, Dict[str, float], List[str], List[str], List[str], Dict[str, Tuple[float, float]]]:
    """
    Loads raw data, processes it into features, and returns the latest team state and metadata.
    This is cached so it only runs once per server verification/lifetime.
    """
    _, __, metadata = load_artifacts()
    g_raw, p_raw, inj_raw = load_raw_data()
    
    g_data = clean_games(g_raw)
    p_data = clean_players(p_raw)
    player_team_game = aggregate_player_team(p_data)
    team_games = build_team_games(g_data, player_team_game)
    team_games = add_rest_and_elo(team_games)
    team_games = merge_injuries(team_games, inj_raw)
    team_games_ready, base_feats, extra_feats = rolling_features(team_games)
    
    feature_columns = metadata["feature_columns"]
    bounds = metadata["bounds"]
    latest, medians = latest_team_rows(team_games_ready)
    
    return latest, medians, base_feats, extra_feats, feature_columns, bounds


@lru_cache(maxsize=1)
def get_mock_schedule() -> Dict:
    """Return a static mock schedule for testing/fallback."""
    return {
        "shedules": {
            "matches": [
                {
                    "date": "01.01.2025",
                    "match": [
                        {
                            "id": "1",
                            "hometeam": {"id": "1610612737", "name": "Atlanta Hawks"},
                            "awayteam": {"id": "1610612738", "name": "Boston Celtics"},
                            "formatted_date": "01.01.2025",
                            "status": "19:00",
                            "venue_name": "State Farm Arena"
                        },
                        {
                            "id": "2",
                            "hometeam": {"id": "1610612739", "name": "Cleveland Cavaliers"},
                            "awayteam": {"id": "1610612740", "name": "New Orleans Pelicans"},
                            "formatted_date": "01.01.2025",
                            "status": "20:00",
                            "venue_name": "Rocket Mortgage FieldHouse"
                        }
                    ]
                }
            ]
        }
    }


def fetch_goalserve_schedule(timeout: int = 20) -> Dict:
    """Fetch raw schedule JSON from Goalserve. Returns mock data on failure or if configured."""
    if nba_settings.use_mock_data:
        print("Using Mock Data (Configured)")
        return get_mock_schedule()

    url = get_goalserve_url()
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data
    except (requests.RequestException, ValueError) as e:
        print(f"GoalServe API Error: {e}. Falling back to mock data.")
        return get_mock_schedule()


def parse_goalserve_matches(raw: Dict) -> List[Dict]:
    """Extract flat match entries from Goalserve response."""
    matches = raw.get("shedules", {}).get("matches", [])
    rows = []
    for day in matches:
        if not isinstance(day, dict):
            continue
        day_date = day.get("formatted_date") or day.get("date")
        match_entries = day.get("match", [])
        if isinstance(match_entries, dict):
            match_entries = [match_entries]
        for m in match_entries:
            if not isinstance(m, dict):
                continue
            raw_match_id = m.get("id")
            try:
                match_id = str(int(raw_match_id)) if raw_match_id is not None else None
            except (TypeError, ValueError):
                match_id = str(raw_match_id) if raw_match_id is not None else None
            rows.append(
                {
                    "match_id": match_id,
                    "date": m.get("formatted_date") or day_date,
                    "status": m.get("status", ""),
                    "venue_name": m.get("venue_name", ""),
                    "home_id": int(m.get("hometeam", {}).get("id", -1)) if m.get("hometeam") else None,
                    "away_id": int(m.get("awayteam", {}).get("id", -1)) if m.get("awayteam") else None,
                    "home_name": m.get("hometeam", {}).get("name", ""),
                    "away_name": m.get("awayteam", {}).get("name", ""),
                }
            )
    return rows


def parse_match_date(date_str: Optional[str]):
    """Attempt to parse a date string into a date object."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except (TypeError, ValueError):
            continue
    parsed = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def find_match_details(match_date: Optional[str], home_team_id: int, away_team_id: int) -> Dict:
    """Locate a specific match in the Goalserve schedule using date + team IDs."""
    sched_raw = fetch_goalserve_schedule()
    rows = parse_goalserve_matches(sched_raw)
    target_date = parse_match_date(match_date)

    for r in rows:
        r_date = parse_match_date(r.get("date"))
        if target_date and r_date and r_date != target_date:
            continue
        if r.get("home_id") and int(r["home_id"]) != int(home_team_id):
            continue
        if r.get("away_id") and int(r["away_id"]) != int(away_team_id):
            continue
        return r
    return {}


def latest_team_rows(tg_ready: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return latest row per team and median fallback values."""
    latest = (
        tg_ready.sort_values("game_date")
        .groupby("team_id", as_index=False)
        .tail(1)
        .set_index("team_id")
    )
    numeric_cols = tg_ready.select_dtypes(include=["number", "bool"]).columns
    medians = {col: tg_ready[col].median() for col in numeric_cols}
    return latest, medians


def build_live_feature_row(
    home_id: int,
    away_id: int,
    latest: pd.DataFrame,
    medians: Dict[str, float],
    base_feats: List[str],
    extra: List[str],
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """Create a single feature row using latest team history, with median fallbacks."""
    def pick(team_id: int, col: str) -> float:
        if team_id in latest.index and col in latest.columns:
            val = latest.loc[team_id, col]
            if pd.notna(val):
                return float(val)
        return float(medians.get(col, 0.0))

    feat = {}
    context = {}
    for col in base_feats + extra:
        feat[f"home_{col}"] = pick(home_id, col)
        feat[f"away_{col}"] = pick(away_id, col)
        context[f"home_{col}"] = feat[f"home_{col}"]
        context[f"away_{col}"] = feat[f"away_{col}"]
    for col in base_feats + extra:
        feat[f"delta_{col}"] = feat[f"home_{col}"] - feat[f"away_{col}"]
    feature_df = pd.DataFrame([feat])[feature_columns]
    return feature_df, context


def predict_upcoming_from_goalserve(limit: int = 10) -> List[Dict]:
    """Fetch schedule, build features from latest team history, and predict upcoming games."""
    classifier, regressors, _ = load_artifacts()
    latest, medians, base_feats, extra_feats, feature_columns, bounds = prepare_historical_data()

    sched_raw = fetch_goalserve_schedule()
    rows = parse_goalserve_matches(sched_raw)
    upcoming = [r for r in rows if "final" not in r.get("status", "").lower()][:limit]
    predictions = []
    for r in upcoming:
        home_id = r.get("home_id")
        away_id = r.get("away_id")
        if home_id is None or away_id is None:
            continue
        feat_df, ctx = build_live_feature_row(home_id, away_id, latest, medians, base_feats, extra_feats, feature_columns)
        feat_df = apply_bounds(feat_df, bounds, feature_columns)
        proba = classifier.predict_proba(feat_df)[:, 1][0]
        diff_pred = regressors["diff"].predict(feat_df)[0]
        total_pred = regressors["total"].predict(feat_df)[0]
        # Build synthetic row for payload using latest stats where available
        payload_row = pd.Series(
            {
                "home_name": r.get("home_name", ""),
                "away_name": r.get("away_name", ""),
                "venue_name": r.get("venue_name", ""),
                "game_date": pd.to_datetime(r.get("date"), format="%d.%m.%Y", errors="coerce"),
                **{k: v for k, v in ctx.items()},
            }
        )
        predictions.append(build_prediction_payload(payload_row, proba, total_pred, diff_pred))
    return predictions


def predict_specific_game(match_date: str, home_team_id: int, away_team_id: int) -> Dict:
    """Predict a single upcoming game using identifiers provided by the client."""
    classifier, regressors, _ = load_artifacts()
    latest, medians, base_feats, extra_feats, feature_columns, bounds = prepare_historical_data()

    feat_df, ctx = build_live_feature_row(home_team_id, away_team_id, latest, medians, base_feats, extra_feats, feature_columns)
    feat_df = apply_bounds(feat_df, bounds, feature_columns)
    proba = classifier.predict_proba(feat_df)[:, 1][0]
    diff_pred = regressors["diff"].predict(feat_df)[0]
    total_pred = regressors["total"].predict(feat_df)[0]

    match_info = find_match_details(match_date, home_team_id, away_team_id)
    if not match_info:
        raise ValueError("Requested match not found in schedule feed")
    parsed_date = parse_match_date(match_date) or parse_match_date(match_info.get("date"))
    game_date = pd.to_datetime(parsed_date) if parsed_date else pd.to_datetime(
        match_info.get("date"), format="%d.%m.%Y", errors="coerce"
    )
    if pd.isna(game_date):
        game_date = pd.Timestamp.utcnow().normalize()

    payload_row = pd.Series(
        {
            "home_name": match_info.get("home_name", "") or str(home_team_id),
            "away_name": match_info.get("away_name", "") or str(away_team_id),
            "venue_name": match_info.get("venue_name", ""),
            "game_date": game_date,
            **{k: v for k, v in ctx.items()},
        }
    )
    return build_prediction_payload(payload_row, proba, total_pred, diff_pred)
