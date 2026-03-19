"""
=============================================================================
PREMIER LEAGUE AI PREDICTOR
Phase 1 — Data Pipeline & Feature Store
=============================================================================
Data sources (all free, no paid tier needed):
  - football-data.co.uk  : historical results + odds CSVs (no key required)
  - football-data.org    : current season only (free key, optional)
  - Understat            : xG / xGA per match (scraped, no key)
  - The Odds API         : live upcoming odds (free key, optional)

Install:
  pip install requests pandas pyarrow aiohttp understat tqdm python-dotenv nest_asyncio

Optional API keys (only needed for live/current data):
  football-data.org → https://www.football-data.org/client/register
  The Odds API      → https://the-odds-api.com/#get-access

  Create a .env file:
      FOOTBALL_DATA_KEY=your_key_here
      ODDS_API_KEY=your_key_here
=============================================================================
"""

import os
import time
import sqlite3
import asyncio
import warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


# =============================================================================
# CONFIG
# =============================================================================

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
ODDS_API_KEY      = os.getenv("ODDS_API_KEY", "")

SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
DB_PATH  = DATA_DIR / "pl_predictor.db"


# =============================================================================
# SECTION 0 — Setup
# =============================================================================

def setup_dirs():
    for d in [DATA_DIR, RAW_DIR, FEAT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Directories ready. DB will be at: {DB_PATH.resolve()}")


# =============================================================================
# SECTION 1 — Historical results from football-data.co.uk
#
# Why this instead of football-data.org?
#   football-data.org free tier only returns the CURRENT season.
#   football-data.co.uk provides free CSV downloads for every PL season
#   back to 1993, including Bet365 and Pinnacle odds — no API key needed.
# =============================================================================

FDCO_BASE = "https://www.football-data.co.uk/mmz4281"

FDCO_SEASON_CODES = {
    "2014-15": "1415", "2015-16": "1516", "2016-17": "1617",
    "2017-18": "1718", "2018-19": "1819", "2019-20": "1920",
    "2020-21": "2021", "2021-22": "2122", "2022-23": "2223",
    "2023-24": "2324",
}

FDCO_NAME_MAP = {
    "Man United":       "Manchester United",
    "Man City":         "Manchester City",
    "Spurs":            "Tottenham",
    "Wolves":           "Wolves",
    "Nott'm Forest":    "Nottingham Forest",
    "West Ham":         "West Ham",
    "West Brom":        "West Brom",
    "Newcastle":        "Newcastle",
    "QPR":              "QPR",
    "Sheffield United": "Sheffield United",
    "Brighton":         "Brighton",
    "Bournemouth":      "Bournemouth",
    "Leicester":        "Leicester",
    "Brentford":        "Brentford",
    "Luton":            "Luton",
    "Leeds":            "Leeds",
    "Fulham":           "Fulham",
    "Middlesbrough":    "Middlesbrough",
    "Huddersfield":     "Huddersfield",
    "Cardiff":          "Cardiff",
    "Norwich":          "Norwich",
    "Watford":          "Watford",
    "Burnley":          "Burnley",
    "Southampton":      "Southampton",
    "Sunderland":       "Sunderland",
    "Swansea":          "Swansea",
    "Stoke":            "Stoke",
    "Hull":             "Hull",
    "Crystal Palace":   "Crystal Palace",
    "Everton":          "Everton",
    "Chelsea":          "Chelsea",
    "Arsenal":          "Arsenal",
    "Liverpool":        "Liverpool",
    "Aston Villa":      "Aston Villa",
    "Ipswich":          "Ipswich",
}


def fetch_fdco_season(season_str: str) -> pd.DataFrame:
    code = FDCO_SEASON_CODES[season_str]
    url  = f"{FDCO_BASE}/{code}/E0.csv"
    df   = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
    df["season"] = season_str
    return df


def parse_fdco_season(df_raw: pd.DataFrame) -> tuple:
    df_raw = df_raw.copy()
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], dayfirst=True, errors="coerce")
    df_raw = df_raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    # Results
    results = pd.DataFrame({
        "season":        df_raw["season"],
        "date":          df_raw["Date"],
        "home_team":     df_raw["HomeTeam"],
        "away_team":     df_raw["AwayTeam"],
        "home_goals_ft": df_raw["FTHG"].astype(int),
        "away_goals_ft": df_raw["FTAG"].astype(int),
        "home_goals_ht": pd.to_numeric(df_raw.get("HTHG", pd.Series(dtype=float)), errors="coerce"),
        "away_goals_ht": pd.to_numeric(df_raw.get("HTAG", pd.Series(dtype=float)), errors="coerce"),
        "referee":       df_raw.get("Referee", pd.Series(dtype=str)),
    })

    results["home_points"] = results.apply(
        lambda r: 3 if r.home_goals_ft > r.away_goals_ft else
                  (1 if r.home_goals_ft == r.away_goals_ft else 0), axis=1)
    results["away_points"] = results.apply(
        lambda r: 3 if r.away_goals_ft > r.home_goals_ft else
                  (1 if r.away_goals_ft == r.home_goals_ft else 0), axis=1)
    results["goal_diff_home"] = results["home_goals_ft"] - results["away_goals_ft"]
    results["result"] = results.apply(
        lambda r: "H" if r.home_goals_ft > r.away_goals_ft else
                  ("A" if r.home_goals_ft < r.away_goals_ft else "D"), axis=1)

    # Odds
    odds = pd.DataFrame({
        "season":    df_raw["season"],
        "date":      df_raw["Date"],
        "home_team": df_raw["HomeTeam"],
        "away_team": df_raw["AwayTeam"],
    })

    for h_col, d_col, a_col in [("B365H","B365D","B365A"), ("PSH","PSD","PSA")]:
        if all(c in df_raw.columns for c in [h_col, d_col, a_col]):
            pfx  = h_col[:3].lower()
            rh   = pd.to_numeric(df_raw[h_col], errors="coerce")
            rd   = pd.to_numeric(df_raw[d_col], errors="coerce")
            ra   = pd.to_numeric(df_raw[a_col], errors="coerce")
            tot  = (1/rh) + (1/rd) + (1/ra)
            odds[f"{pfx}_prob_h"]    = (1/rh) / tot
            odds[f"{pfx}_prob_d"]    = (1/rd) / tot
            odds[f"{pfx}_prob_a"]    = (1/ra) / tot
            odds[f"{pfx}_overround"] = tot

    return results, odds


def pull_all_results() -> tuple:
    print("\n── STEP 1: Pulling results from football-data.co.uk ────────────")
    print("  Free CSVs — no API key required, all 10 seasons available\n")

    all_results, all_odds = [], []

    for season in tqdm(SEASONS, desc="Seasons"):
        print(f"  {season}...", end=" ", flush=True)
        try:
            raw          = fetch_fdco_season(season)
            results, odds = parse_fdco_season(raw)
            all_results.append(results)
            all_odds.append(odds)
            print(f"{len(results)} matches")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.5)

    df_results = pd.concat(all_results, ignore_index=True)
    df_odds    = pd.concat(all_odds,    ignore_index=True)

    df_results["match_id"] = range(len(df_results))

    df_results["home_team"] = df_results["home_team"].replace(FDCO_NAME_MAP)
    df_results["away_team"] = df_results["away_team"].replace(FDCO_NAME_MAP)
    df_odds["home_team"]    = df_odds["home_team"].replace(FDCO_NAME_MAP)
    df_odds["away_team"]    = df_odds["away_team"].replace(FDCO_NAME_MAP)

    df_results["season_year"] = df_results["date"].dt.year.where(
        df_results["date"].dt.month >= 8, df_results["date"].dt.year - 1
    )
    df_results["matchday"] = (df_results.groupby("season")["date"]
                              .rank(method="dense").astype(int))

    df_results = df_results.sort_values("date").reset_index(drop=True)

    df_results.to_parquet(RAW_DIR / "results_raw.parquet", index=False)
    df_odds.to_parquet(RAW_DIR / "odds_raw.parquet", index=False)

    print(f"\n  ✅ {len(df_results)} matches | "
          f"{df_results.season.nunique()} seasons | "
          f"{df_results.home_team.nunique()} unique teams")
    return df_results, df_odds


# =============================================================================
# SECTION 2 — xG from Understat
#
# Fix: newer understat library (>=0.3) requires an aiohttp.ClientSession
# passed explicitly rather than using the context manager.
# We try the new API first, fall back to old API if it fails.
# =============================================================================

US_NAME_MAP = {
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion":    "West Brom",
    "Queens Park Rangers":     "QPR",
    "Newcastle United":        "Newcastle",
    "Nottingham Forest":       "Nottingham Forest",
    "Leeds United":            "Leeds",
    "Sheffield United":        "Sheffield United",
}


def _parse_us_results(results: list, season_year: int) -> pd.DataFrame:
    rows = []
    for m in results:
        h, a = m["h"], m["a"]
        ppda_h = h.get("ppda") or {}
        ppda_a = a.get("ppda") or {}
        rows.append({
            "season_year":  season_year,
            "date":         m.get("datetime", "")[:10],
            "home_team_us": h.get("title", ""),
            "away_team_us": a.get("title", ""),
            "home_xg":      float(h.get("xG", 0.0)),
            "away_xg":      float(a.get("xG", 0.0)),
            "home_xga":     float(a.get("xG", 0.0)),
            "away_xga":     float(h.get("xG", 0.0)),
            "home_deep":    int(h.get("deep", 0)),
            "away_deep":    int(a.get("deep", 0)),
            "home_ppda":    float(ppda_h.get("att", 0)) / max(float(ppda_h.get("def", 1)), 1),
            "away_ppda":    float(ppda_a.get("att", 0)) / max(float(ppda_a.get("def", 1)), 1),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


async def _fetch_all_us() -> pd.DataFrame:
    import aiohttp
    from understat import Understat

    season_years = list(range(2014, 2024))
    dfs = []

    async with aiohttp.ClientSession() as session:
        for yr in tqdm(season_years, desc="Understat seasons"):
            print(f"  {yr}/{yr+1}...", end=" ", flush=True)
            try:
                # New API style: pass session explicitly
                us      = Understat(session)
                results = await us.get_league_results("EPL", yr)
                df_s    = _parse_us_results(results, yr)
                dfs.append(df_s)
                print(f"{len(df_s)} matches")
            except TypeError:
                # Old API style: no session argument
                try:
                    from understat import Understat as US2
                    async with US2() as us2:
                        results = await us2.get_league_results("EPL", yr)
                    df_s = _parse_us_results(results, yr)
                    dfs.append(df_s)
                    print(f"{len(df_s)} matches (old API)")
                except Exception as e2:
                    print(f"ERROR: {e2}")
            except Exception as e:
                print(f"ERROR: {e}")
            await asyncio.sleep(1.5)

    if not dfs:
        raise RuntimeError("No Understat data fetched — check your internet connection.")
    return pd.concat(dfs, ignore_index=True)


def pull_understat_xg() -> pd.DataFrame:
    print("\n── STEP 2: Scraping xG from Understat ─────────────────────────")

    df = asyncio.run(_fetch_all_us())

    df["home_team_us"] = df["home_team_us"].replace(US_NAME_MAP)
    df["away_team_us"] = df["away_team_us"].replace(US_NAME_MAP)

    df.to_parquet(RAW_DIR / "xg_raw.parquet", index=False)
    print(f"\n  ✅ {len(df)} xG rows from Understat")
    return df


# =============================================================================
# SECTION 3 — Merge results + xG
# =============================================================================

def merge_results_xg(df_results: pd.DataFrame, df_xg: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 3: Merging results + xG ────────────────────────────────")

    xg_slim = df_xg[[
        "date","home_team_us","away_team_us",
        "home_xg","away_xg","home_xga","away_xga",
        "home_deep","away_deep","home_ppda","away_ppda",
    ]].rename(columns={"home_team_us":"home_team","away_team_us":"away_team"})

    df = df_results.merge(xg_slim, on=["date","home_team","away_team"], how="left")

    n, n_xg = len(df), df["home_xg"].notna().sum()
    print(f"  Matched xG: {n_xg}/{n} ({100*n_xg/n:.1f}%)")

    if n_xg < n * 0.8:
        print("  ⚠️  Low match rate — showing unmatched rows:")
        print(df[df["home_xg"].isna()][["season","date","home_team","away_team"]].head(10).to_string(index=False))

    print(f"  ✅ Master shape: {df.shape}")
    return df


# =============================================================================
# SECTION 4 — Rolling features (shift(1) = no leakage)
# =============================================================================

def build_rolling_features(df: pd.DataFrame) -> tuple:
    print("\n── STEP 4: Computing rolling features ──────────────────────────")

    home = df[["match_id","date","season","matchday",
               "home_team","away_team","home_goals_ft","away_goals_ft",
               "home_xg","away_xg","home_points"]].copy()
    home.columns = ["match_id","date","season","matchday",
                    "team","opponent","gf","ga","xg","xga","pts"]
    home["home"] = 1

    away = df[["match_id","date","season","matchday",
               "away_team","home_team","away_goals_ft","home_goals_ft",
               "away_xg","home_xg","away_points"]].copy()
    away.columns = ["match_id","date","season","matchday",
                    "team","opponent","gf","ga","xg","xga","pts"]
    away["home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team","date"]).reset_index(drop=True)

    long["xg"]  = long["xg"].fillna(0.0)
    long["xga"] = long["xga"].fillna(0.0)

    def rm(s, n):
        return s.shift(1).rolling(n, min_periods=1).mean()

    def rs(s, n):
        return s.shift(1).rolling(n, min_periods=1).sum()

    grp = long.groupby("team")

    for n in [5, 10]:
        long[f"xg_{n}"]  = grp["xg"].transform(lambda s: rm(s, n))
        long[f"xga_{n}"] = grp["xga"].transform(lambda s: rm(s, n))
        long[f"gf_{n}"]  = grp["gf"].transform(lambda s: rm(s, n))
        long[f"ga_{n}"]  = grp["ga"].transform(lambda s: rm(s, n))
        long[f"pts_{n}"] = grp["pts"].transform(lambda s: rs(s, n))

    long["win"]  = (long["pts"] == 3).astype(int)
    long["draw"] = (long["pts"] == 1).astype(int)

    long["win_rate_5"]  = grp["win"].transform(lambda s: rm(s, 5))
    long["draw_rate_5"] = grp["draw"].transform(lambda s: rm(s, 5))

    # rolling().apply() must return a float — build form string manually instead
    def compute_form_str(group: pd.Series) -> pd.Series:
        shifted = group.shift(1).values
        out = []
        for i in range(len(shifted)):
            window = [v for v in shifted[max(0, i-4):i+1] if not np.isnan(v)]
            out.append("".join([{3:"W",1:"D",0:"L"}.get(int(v),"?") for v in window]))
        return pd.Series(out, index=group.index)

    long["form_str_5"]    = grp["pts"].transform(compute_form_str)
    long["xg_momentum_5"] = long["xg_5"] - long["xga_5"]

    feat_cols = ["match_id","xg_5","xg_10","xga_5","xga_10","gf_5","ga_5",
                 "pts_5","pts_10","win_rate_5","draw_rate_5","xg_momentum_5","form_str_5"]

    home_f = long[long["home"] == 1][feat_cols].copy()
    home_f.columns = ["match_id"] + [f"home_{c}" for c in feat_cols[1:]]

    away_f = long[long["home"] == 0][feat_cols].copy()
    away_f.columns = ["match_id"] + [f"away_{c}" for c in feat_cols[1:]]

    df = df.merge(home_f, on="match_id", how="left")
    df = df.merge(away_f, on="match_id", how="left")

    df["xg_diff_5"]  = df["home_xg_5"]  - df["away_xg_5"]
    df["xga_diff_5"] = df["home_xga_5"] - df["away_xga_5"]
    df["pts_diff_5"] = df["home_pts_5"] - df["away_pts_5"]

    print(f"  ✅ Rolling features added. Shape: {df.shape}")
    return df, long


# =============================================================================
# SECTION 5 — ELO ratings
# =============================================================================

class EloSystem:
    """Standard ELO + goal-adjusted variant. K=20 is standard for club football."""

    def __init__(self, k: float = 20.0, start: float = 1000.0):
        self.k = k
        self.start = start
        self.ratings: dict = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.start)

    def expected(self, home: str, away: str) -> float:
        return 1.0 / (1.0 + 10 ** ((self.get(away) - self.get(home)) / 400.0))

    def update(self, home: str, away: str, score_h: float, score_a: float):
        rh, ra   = self.get(home), self.get(away)
        total    = score_h + score_a
        actual_h = score_h / total if total > 0 else 0.5
        exp_h    = self.expected(home, away)
        self.ratings[home] = rh + self.k * (actual_h - exp_h)
        self.ratings[away] = ra + self.k * ((1 - actual_h) - (1 - exp_h))
        return rh, ra  # pre-match, no leakage


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 5: Computing ELO ratings ───────────────────────────────")

    elo_std  = EloSystem(k=20.0)
    elo_goal = EloSystem(k=20.0)
    rows     = []

    for _, r in df.sort_values("date").iterrows():
        h, a   = r["home_team"], r["away_team"]
        hg, ag = float(r["home_goals_ft"]), float(r["away_goals_ft"])
        s_h    = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        s_a    = 1.0 - s_h if hg != ag else 0.5

        rh,   ra   = elo_std.update(h, a, s_h, s_a)
        rh_g, ra_g = elo_goal.update(h, a, hg, ag)

        rows.append({"match_id": r["match_id"],
                     "home_elo": rh, "away_elo": ra,
                     "home_elo_goal": rh_g, "away_elo_goal": ra_g})

    df_elo = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True).merge(df_elo, on="match_id", how="left")
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    final = {}
    for _, r in df[df["season"] == df["season"].max()].sort_values("date").iterrows():
        final[r["home_team"]] = r["home_elo"]
        final[r["away_team"]] = r["away_elo"]
    top = pd.Series(final).sort_values(ascending=False).head(10).round(1)
    print(f"  ELO top 10 (end of {df['season'].max()}):\n{top.to_string()}")
    print("  ✅ ELO computed")
    return df


# =============================================================================
# SECTION 6 — Pi-ratings (Constantinou & Fenton, 2013)
# =============================================================================

class PiRatingSystem:
    """
    Per team: home_att, home_def, away_att, away_def.
    c=3.0 scales goal counts to probabilities.
    lam=0.035 is the learning rate from the original paper.
    """

    def __init__(self, c: float = 3.0, lam: float = 0.035):
        self.c = c
        self.lam = lam
        self.r: dict = {}

    def _init(self, team):
        if team not in self.r:
            self.r[team] = {"ha": 0.0, "hd": 0.0, "aa": 0.0, "ad": 0.0}

    def expected_goals(self, home: str, away: str):
        self._init(home); self._init(away)
        rh, ra = self.r[home], self.r[away]
        diff   = (rh["ha"] - ra["ad"] + ra["aa"] - rh["hd"]) / 2
        return max(0.1, np.exp(diff / self.c)), max(0.1, np.exp(-diff / self.c))

    def update(self, home: str, away: str, hg: int, ag: int):
        self._init(home); self._init(away)
        rh_pre, ra_pre = {**self.r[home]}, {**self.r[away]}
        exp_h, exp_a   = self.expected_goals(home, away)
        err_h, err_a   = hg - exp_h, ag - exp_a
        self.r[home]["ha"] += self.lam * err_h
        self.r[home]["hd"] -= self.lam * err_a
        self.r[away]["aa"] += self.lam * err_a
        self.r[away]["ad"] -= self.lam * err_h
        return rh_pre, ra_pre


def compute_pi_ratings(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 6: Computing Pi-ratings ────────────────────────────────")

    pi   = PiRatingSystem()
    rows = []

    for _, r in df.sort_values("date").iterrows():
        h  = r["home_team"]
        a  = r["away_team"]
        hg = int(r["home_goals_ft"]) if pd.notna(r["home_goals_ft"]) else 0
        ag = int(r["away_goals_ft"]) if pd.notna(r["away_goals_ft"]) else 0

        rh, ra = pi.update(h, a, hg, ag)
        rows.append({
            "match_id":         r["match_id"],
            "home_pi_ha":       rh["ha"],
            "home_pi_hd":       rh["hd"],
            "away_pi_aa":       ra["aa"],
            "away_pi_ad":       ra["ad"],
            "pi_strength_diff": (rh["ha"] - ra["ad"]) - (ra["aa"] - rh["hd"]),
        })

    df_pi = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True).merge(df_pi, on="match_id", how="left")
    print("  ✅ Pi-ratings computed")
    return df


# =============================================================================
# SECTION 7 — Live odds via The Odds API (optional)
# =============================================================================

def fetch_live_odds() -> pd.DataFrame:
    print("\n── STEP 7: Live odds from The Odds API ─────────────────────────")

    if not ODDS_API_KEY:
        print("  Skipped — ODDS_API_KEY not set.")
        print("  Get a free key at https://the-odds-api.com/#get-access")
        return pd.DataFrame()

    url    = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "uk",
              "markets": "h2h", "oddsFormat": "decimal"}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()

    rows = []
    for game in data:
        home = game["home_team"]
        away = game["away_team"]
        odds_h, odds_d, odds_a = [], [], []
        for bookie in game.get("bookmakers", []):
            for mkt in bookie.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        if   o["name"] == home: odds_h.append(o["price"])
                        elif o["name"] == away: odds_a.append(o["price"])
                        else:                   odds_d.append(o["price"])

        if odds_h and odds_d and odds_a:
            avg_h, avg_d, avg_a = np.mean(odds_h), np.mean(odds_d), np.mean(odds_a)
            total = 1/avg_h + 1/avg_d + 1/avg_a
            rows.append({
                "date":         game["commence_time"][:10],
                "home_team":    home,
                "away_team":    away,
                "live_prob_h":  (1/avg_h) / total,
                "live_prob_d":  (1/avg_d) / total,
                "live_prob_a":  (1/avg_a) / total,
                "n_bookmakers": len(game.get("bookmakers", [])),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(RAW_DIR / "live_odds.parquet", index=False)
        print(f"  ✅ {len(df)} upcoming fixtures with live odds")
        print(df[["date","home_team","away_team","live_prob_h","live_prob_d","live_prob_a"]].to_string(index=False))
    return df


# =============================================================================
# SECTION 8 — Assemble feature table
# =============================================================================

def build_feature_table(df_master: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 8: Assembling full feature table ───────────────────────")

    odds_prob_cols = [c for c in df_odds.columns if "prob" in c or "overround" in c]
    merge_cols     = ["date","home_team","away_team","season"] + odds_prob_cols

    df = df_master.merge(
        df_odds[[c for c in merge_cols if c in df_odds.columns]],
        on=["date","home_team","away_team","season"],
        how="left"
    )

    df["result_code"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    print(f"  ✅ Feature table: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# =============================================================================
# SECTION 9 — Write to SQLite + Parquet
# =============================================================================

def write_to_db(df_feat: pd.DataFrame, df_long: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 9: Writing to SQLite + Parquet ─────────────────────────")

    conn = sqlite3.connect(DB_PATH)

    match_cols = ["match_id","season","matchday","date","home_team","away_team",
                  "home_goals_ft","away_goals_ft","home_goals_ht","away_goals_ht",
                  "referee","result","home_points","away_points","goal_diff_home"]
    (df_feat[[c for c in match_cols if c in df_feat.columns]]
     .assign(date=lambda x: x["date"].astype(str))
     .to_sql("matches", conn, if_exists="replace", index=False))

    df_feat.copy().assign(date=lambda x: x["date"].astype(str)).to_sql(
        "features", conn, if_exists="replace", index=False
    )

    latest_h = (df_feat.sort_values("date").groupby("home_team").last()
                [["home_elo","home_elo_goal","home_pi_ha","home_pi_hd"]]
                .reset_index()
                .rename(columns={"home_team":"team","home_elo":"elo",
                                 "home_elo_goal":"elo_goal","home_pi_ha":"pi_ha","home_pi_hd":"pi_hd"}))
    latest_a = (df_feat.sort_values("date").groupby("away_team").last()
                [["away_elo","away_elo_goal","away_pi_aa","away_pi_ad"]]
                .reset_index()
                .rename(columns={"away_team":"team","away_elo":"elo_away",
                                 "away_elo_goal":"elo_goal_away","away_pi_aa":"pi_aa","away_pi_ad":"pi_ad"}))
    df_ratings = (latest_h.merge(latest_a, on="team", how="outer")
                  .assign(elo_combined=lambda x: x[["elo","elo_away"]].mean(axis=1))
                  .sort_values("elo_combined", ascending=False)
                  .reset_index(drop=True))
    df_ratings.to_sql("team_ratings", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_date   ON features (date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_teams  ON features (home_team, away_team)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_season ON features (season)")
    conn.commit()
    conn.close()

    df_feat.to_parquet(FEAT_DIR / "features.parquet",       index=False)
    df_ratings.to_parquet(FEAT_DIR / "team_ratings.parquet", index=False)

    kb = DB_PATH.stat().st_size / 1024
    print(f"  SQLite  : {DB_PATH}  ({kb:.0f} KB)")
    print(f"  Parquet : {FEAT_DIR / 'features.parquet'}")
    print(f"  Parquet : {FEAT_DIR / 'team_ratings.parquet'}")
    return df_ratings


# =============================================================================
# SECTION 10 — Verification report
# =============================================================================

def verification_report(df_ratings: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  PHASE 1 — VERIFICATION REPORT")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    for table in ["matches", "features", "team_ratings"]:
        n = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn).iloc[0, 0]
        print(f"  [{table}]: {n} rows")

    print("\n  Feature coverage (first 2000 rows):")
    df_check = pd.read_sql("SELECT * FROM features LIMIT 2000", conn)
    for col in ["home_xg","away_xg","home_xg_5","home_elo","home_pi_ha","b36_prob_h"]:
        if col in df_check.columns:
            pct  = df_check[col].notna().mean() * 100
            icon = "✅" if pct > 80 else ("⚠️" if pct > 40 else "❌")
            print(f"    {icon}  {col:<25} {pct:.1f}% non-null")

    print("\n  Season coverage:")
    seasons = pd.read_sql(
        "SELECT season, COUNT(*) as matches FROM matches GROUP BY season ORDER BY season", conn
    )
    print(seasons.to_string(index=False))

    print("\n  ELO top 10:")
    print(df_ratings[["team","elo_combined"]].head(10)
          .assign(elo_combined=lambda x: x["elo_combined"].round(1))
          .to_string(index=False))

    conn.close()

    print("\n" + "=" * 60)
    print("  ✅ Phase 1 complete — ready for Phase 2")
    print("=" * 60)
    print("""
  Output files:
    data/pl_predictor.db
      ├── matches       (raw results)
      ├── features      (ML training table)
      └── team_ratings  (current ELO + pi per team)
    data/raw/
      ├── results_raw.parquet
      ├── odds_raw.parquet
      └── xg_raw.parquet
    data/features/
      ├── features.parquet       ← main Phase 2 input
      └── team_ratings.parquet
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
=============================================================================
  Premier League AI Predictor — Phase 1: Data Pipeline & Feature Store
=============================================================================
""")
    setup_dirs()

    df_results, df_odds = pull_all_results()
    df_xg               = pull_understat_xg()
    df_master           = merge_results_xg(df_results, df_xg)
    df_master, df_long  = build_rolling_features(df_master)
    df_master           = compute_elo(df_master)
    df_master           = compute_pi_ratings(df_master)
    fetch_live_odds()
    df_feat             = build_feature_table(df_master, df_odds)
    df_ratings          = write_to_db(df_feat, df_long)
    verification_report(df_ratings)


if __name__ == "__main__":
    main()