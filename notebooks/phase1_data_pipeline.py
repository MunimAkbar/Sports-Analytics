"""
=============================================================================
PREMIER LEAGUE AI PREDICTOR
Phase 1 — Data Pipeline & Feature Store
=============================================================================
Steps:
  1. Pull 10 seasons of PL results via football-data.org
  2. Scrape xG / xGA / PPDA from Understat
  3. Compute rolling features (5 & 10 match windows)
  4. ELO ratings (standard + goal-adjusted)
  5. Pi-ratings (home/away attack & defence splits)
  6. Historical odds from football-data.co.uk
  7. Live odds from The Odds API
  8. Merge everything → SQLite + Parquet feature store

Setup:
  pip install requests pandas pyarrow aiohttp understat tqdm python-dotenv

API Keys (free):
  football-data.org  → https://www.football-data.org/client/register
  The Odds API       → https://the-odds-api.com/#get-access

  Either set them in a .env file:
      FOOTBALL_DATA_KEY=your_key_here
      ODDS_API_KEY=your_key_here

  Or paste directly into the CONFIG section below.
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

# Try loading .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONFIG — edit this section
# =============================================================================

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "YOUR_KEY_HERE")
ODDS_API_KEY      = os.getenv("ODDS_API_KEY",      "YOUR_KEY_HERE")

# 10 seasons of Premier League data
SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

BASE_DIR  = Path(".")
DATA_DIR  = BASE_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
FEAT_DIR  = DATA_DIR / "features"
DB_PATH   = DATA_DIR / "pl_predictor.db"


# =============================================================================
# SECTION 0 — Directory setup
# =============================================================================

def setup_dirs():
    for d in [DATA_DIR, RAW_DIR, FEAT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Directories ready. DB will be at: {DB_PATH.resolve()}")


# =============================================================================
# SECTION 1 — Pull results from football-data.org
# =============================================================================

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
COMPETITION        = "PL"

# Maps football-data.org long names → canonical short names
# Understat and football-data.co.uk both use the short versions
TEAM_NAME_MAP = {
    "Arsenal FC":                    "Arsenal",
    "Aston Villa FC":                "Aston Villa",
    "Brentford FC":                  "Brentford",
    "Brighton & Hove Albion FC":     "Brighton",
    "Burnley FC":                    "Burnley",
    "Chelsea FC":                    "Chelsea",
    "Crystal Palace FC":             "Crystal Palace",
    "Everton FC":                    "Everton",
    "Fulham FC":                     "Fulham",
    "Ipswich Town FC":               "Ipswich",
    "Leeds United FC":               "Leeds",
    "Leicester City FC":             "Leicester",
    "Liverpool FC":                  "Liverpool",
    "Luton Town FC":                 "Luton",
    "Manchester City FC":            "Manchester City",
    "Manchester United FC":          "Manchester United",
    "Middlesbrough FC":              "Middlesbrough",
    "Newcastle United FC":           "Newcastle",
    "Norwich City FC":               "Norwich",
    "Nottingham Forest FC":          "Nottingham Forest",
    "Queens Park Rangers FC":        "QPR",
    "Sheffield United FC":           "Sheffield United",
    "Southampton FC":                "Southampton",
    "Stoke City FC":                 "Stoke",
    "Sunderland AFC":                "Sunderland",
    "Swansea City AFC":              "Swansea",
    "Tottenham Hotspur FC":          "Tottenham",
    "Watford FC":                    "Watford",
    "West Bromwich Albion FC":       "West Brom",
    "West Ham United FC":            "West Ham",
    "Wolverhampton Wanderers FC":    "Wolves",
    "Hull City AFC":                 "Hull",
    "AFC Bournemouth":               "Bournemouth",
}


def fetch_season_matches(season_str: str) -> pd.DataFrame:
    """
    Fetch all finished PL matches for one season from football-data.org.
    season_str: e.g. '2023-24'
    """
    start_year = int(season_str.split("-")[0])
    url    = f"{FOOTBALL_DATA_BASE}/competitions/{COMPETITION}/matches"
    params = {"season": start_year}
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}

    resp = requests.get(url, headers=headers, params=params, timeout=15)

    if resp.status_code == 429:
        print("  Rate limited — waiting 65s...")
        time.sleep(65)
        resp = requests.get(url, headers=headers, params=params, timeout=15)

    resp.raise_for_status()
    matches = resp.json().get("matches", [])

    rows = []
    for m in matches:
        ft = m.get("score", {}).get("fullTime", {})
        ht = m.get("score", {}).get("halfTime", {})
        rows.append({
            "match_id":       m["id"],
            "season":         season_str,
            "matchday":       m.get("matchday"),
            "date":           m.get("utcDate", "")[:10],
            "status":         m.get("status"),
            "home_team":      m["homeTeam"]["name"],
            "away_team":      m["awayTeam"]["name"],
            "home_goals_ft":  ft.get("home"),
            "away_goals_ft":  ft.get("away"),
            "home_goals_ht":  ht.get("home"),
            "away_goals_ht":  ht.get("away"),
            "winner":         m.get("score", {}).get("winner"),
            "referee":        m.get("referees", [{}])[0].get("name") if m.get("referees") else None,
            "venue":          m.get("venue"),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["status"] == "FINISHED"].copy()


def pull_all_results() -> pd.DataFrame:
    print("\n── STEP 1: Pulling results from football-data.org ──────────────")
    print("Free tier = 10 calls/min → sleeping 7s between seasons")

    all_dfs = []
    for season in tqdm(SEASONS, desc="Seasons"):
        print(f"  {season}...", end=" ", flush=True)
        try:
            df_s = fetch_season_matches(season)
            all_dfs.append(df_s)
            print(f"{len(df_s)} matches")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(7)

    df = pd.concat(all_dfs, ignore_index=True)
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)

    # Season year for joining with Understat
    df["season_year"] = df["date"].dt.year.where(
        df["date"].dt.month >= 8, df["date"].dt.year - 1
    )

    # Points and goal diff
    df["home_points"] = df.apply(
        lambda r: 3 if r.home_goals_ft > r.away_goals_ft else
                  (1 if r.home_goals_ft == r.away_goals_ft else 0), axis=1
    )
    df["away_points"] = df.apply(
        lambda r: 3 if r.away_goals_ft > r.home_goals_ft else
                  (1 if r.away_goals_ft == r.home_goals_ft else 0), axis=1
    )
    df["goal_diff_home"] = df["home_goals_ft"] - df["away_goals_ft"]
    df["result"] = df.apply(
        lambda r: "H" if r.home_goals_ft > r.away_goals_ft else
                  ("A" if r.home_goals_ft < r.away_goals_ft else "D"), axis=1
    )

    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(RAW_DIR / "results_raw.parquet", index=False)

    print(f"\n  ✅ {len(df)} finished matches across {df.season.nunique()} seasons")
    return df


# =============================================================================
# SECTION 2 — Scrape xG from Understat
# =============================================================================

# Understat → canonical name corrections
US_NAME_MAP = {
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion":    "West Brom",
    "Queens Park Rangers":     "QPR",
    "Newcastle United":        "Newcastle",
    "Nottingham Forest":       "Nottingham Forest",
}


async def _fetch_us_season(season_year: int) -> pd.DataFrame:
    from understat import Understat
    async with Understat() as us:
        results = await us.get_league_results("EPL", season_year)

    rows = []
    for m in results:
        h, a = m["h"], m["a"]
        rows.append({
            "understat_id":    m.get("id"),
            "season_year":     season_year,
            "date":            m.get("datetime", "")[:10],
            "home_team_us":    h.get("title"),
            "away_team_us":    a.get("title"),
            "home_xg":         float(h.get("xG", 0.0)),
            "away_xg":         float(a.get("xG", 0.0)),
            "home_xga":        float(a.get("xG", 0.0)),  # opponent xG = your xGA
            "away_xga":        float(h.get("xG", 0.0)),
            "home_deep":       int(h.get("deep", 0)),
            "away_deep":       int(a.get("deep", 0)),
            "home_ppda":       (float(h.get("ppda", {}).get("att", 0)) /
                                max(float(h.get("ppda", {}).get("def", 1)), 1)),
            "away_ppda":       (float(a.get("ppda", {}).get("att", 0)) /
                                max(float(a.get("ppda", {}).get("def", 1)), 1)),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


async def _fetch_all_us() -> pd.DataFrame:
    season_years = list(range(2014, 2024))
    dfs = []
    for yr in tqdm(season_years, desc="Understat seasons"):
        print(f"  {yr}/{yr+1}...", end=" ", flush=True)
        try:
            df_s = await _fetch_us_season(yr)
            dfs.append(df_s)
            print(f"{len(df_s)} matches")
        except Exception as e:
            print(f"ERROR: {e}")
        await asyncio.sleep(1.5)
    return pd.concat(dfs, ignore_index=True)


def pull_understat_xg() -> pd.DataFrame:
    print("\n── STEP 2: Scraping xG from Understat ─────────────────────────")

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    df = asyncio.run(_fetch_all_us())

    df["home_team_us"] = df["home_team_us"].replace(US_NAME_MAP)
    df["away_team_us"] = df["away_team_us"].replace(US_NAME_MAP)

    df.to_parquet(RAW_DIR / "xg_raw.parquet", index=False)
    print(f"\n  ✅ {len(df)} xG rows pulled from Understat")
    return df


def merge_results_xg(df_results: pd.DataFrame, df_xg: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 3: Merging results + xG ────────────────────────────────")

    df_xg_slim = df_xg[[
        "date", "home_team_us", "away_team_us",
        "home_xg", "away_xg", "home_xga", "away_xga",
        "home_deep", "away_deep", "home_ppda", "away_ppda",
    ]].rename(columns={"home_team_us": "home_team", "away_team_us": "away_team"})

    df = df_results.merge(df_xg_slim, on=["date", "home_team", "away_team"], how="left")

    n_total   = len(df)
    n_with_xg = df["home_xg"].notna().sum()
    print(f"  Matches with xG: {n_with_xg}/{n_total} ({100*n_with_xg/n_total:.1f}%)")

    if n_with_xg < n_total:
        unmatched = df[df["home_xg"].isna()][["season","date","home_team","away_team"]]
        print("  Unmatched (check name map):")
        print(unmatched.head(10).to_string(index=False))

    print(f"  ✅ Master table: {df.shape}")
    return df


# =============================================================================
# SECTION 3 — Rolling features
# =============================================================================

def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unstack fixture format → long format (one row per team per match),
    compute rolling features with shift(1) to prevent leakage,
    then pivot back to fixture-level for home and away.
    """
    print("\n── STEP 4: Computing rolling features ──────────────────────────")

    # Build long format
    home = df[["match_id","date","season","matchday","home_team","away_team",
               "home_goals_ft","away_goals_ft","home_xg","away_xg","home_points"]].copy()
    home.columns = ["match_id","date","season","matchday","team","opponent",
                    "gf","ga","xg","xga","pts"]
    home["home"] = 1

    away = df[["match_id","date","season","matchday","away_team","home_team",
               "away_goals_ft","home_goals_ft","away_xg","home_xg","away_points"]].copy()
    away.columns = ["match_id","date","season","matchday","team","opponent",
                    "gf","ga","xg","xga","pts"]
    away["home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    def rm(s, n):   # rolling mean, shift 1
        return s.shift(1).rolling(n, min_periods=1).mean()
    def rs(s, n):   # rolling sum, shift 1
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

    def form_str(s):
        return s.shift(1).rolling(5, min_periods=1).apply(
            lambda x: "".join([{3:"W",1:"D",0:"L"}.get(int(v),"?") for v in x]),
            raw=True
        )
    long["form_str_5"] = grp["pts"].transform(form_str)
    long["xg_momentum_5"] = long["xg_5"] - long["xga_5"]

    # Pivot back to fixture level
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
# SECTION 4 — ELO Ratings
# =============================================================================

class EloSystem:
    """
    Standard ELO + goal-adjusted variant.
    Pre-match ratings are stored — no leakage.
    K=20 is the accepted standard for club football.
    """
    def __init__(self, k: float = 20.0, start: float = 1000.0):
        self.k = k
        self.start = start
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.start)

    def expected(self, home: str, away: str) -> float:
        return 1.0 / (1.0 + 10 ** ((self.get(away) - self.get(home)) / 400.0))

    def update(self, home: str, away: str, score_h: float, score_a: float):
        rh, ra = self.get(home), self.get(away)
        total    = score_h + score_a
        actual_h = score_h / total if total > 0 else 0.5
        exp_h    = self.expected(home, away)
        self.ratings[home] = rh + self.k * (actual_h - exp_h)
        self.ratings[away] = ra + self.k * ((1 - actual_h) - (1 - exp_h))
        return rh, ra  # pre-match (no leakage)


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 5: Computing ELO ratings ───────────────────────────────")

    elo_std  = EloSystem(k=20.0)
    elo_goal = EloSystem(k=20.0)  # uses goal counts as score

    home_elo, away_elo, home_elo_g, away_elo_g = [], [], [], []

    for _, row in df.sort_values("date").iterrows():
        h, a  = row["home_team"], row["away_team"]
        hg, ag = float(row["home_goals_ft"]), float(row["away_goals_ft"])

        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        s_a = 1.0 - s_h if hg != ag else 0.5

        rh, ra = elo_std.update(h, a, s_h, s_a)
        home_elo.append(rh); away_elo.append(ra)

        rh_g, ra_g = elo_goal.update(h, a, hg, ag)
        home_elo_g.append(rh_g); away_elo_g.append(ra_g)

    df = df.sort_values("date").reset_index(drop=True)
    df["home_elo"]      = home_elo
    df["away_elo"]      = away_elo
    df["home_elo_goal"] = home_elo_g
    df["away_elo_goal"] = away_elo_g
    df["elo_diff"]      = df["home_elo"] - df["away_elo"]

    # Print current top 10
    final = {}
    for _, r in df[df["season"] == df["season"].max()].sort_values("date").iterrows():
        final[r["home_team"]] = r["home_elo"]
        final[r["away_team"]] = r["away_elo"]
    top10 = pd.Series(final).sort_values(ascending=False).head(10).round(1)
    print(f"  Current ELO top 10:\n{top10.to_string()}")
    print(f"  ✅ ELO computed")
    return df


# =============================================================================
# SECTION 5 — Pi-Ratings
# =============================================================================

class PiRatingSystem:
    """
    Simplified Constantinou & Fenton (2013) pi-ratings.
    Per team: home_att, home_def, away_att, away_def.
    Learning rate lam=0.035, scaling constant c=3.0.
    """
    def __init__(self, c: float = 3.0, lam: float = 0.035):
        self.c   = c
        self.lam = lam
        self.r: dict[str, dict] = {}

    def _init(self, team: str):
        if team not in self.r:
            self.r[team] = {"ha": 0.0, "hd": 0.0, "aa": 0.0, "ad": 0.0}

    def expected_goals(self, home: str, away: str):
        self._init(home); self._init(away)
        rh, ra = self.r[home], self.r[away]
        diff = (rh["ha"] - ra["ad"] + ra["aa"] - rh["hd"]) / 2
        return max(0.1, np.exp(diff / self.c)), max(0.1, np.exp(-diff / self.c))

    def update(self, home: str, away: str, hg: int, ag: int):
        self._init(home); self._init(away)
        rh_pre, ra_pre = {**self.r[home]}, {**self.r[away]}

        exp_h, exp_a = self.expected_goals(home, away)
        err_h, err_a = hg - exp_h, ag - exp_a

        self.r[home]["ha"] += self.lam * err_h
        self.r[home]["hd"] -= self.lam * err_a
        self.r[away]["aa"] += self.lam * err_a
        self.r[away]["ad"] -= self.lam * err_h

        return rh_pre, ra_pre


def compute_pi_ratings(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 6: Computing Pi-ratings ────────────────────────────────")

    pi = PiRatingSystem()
    rows = []

    for _, row in df.sort_values("date").iterrows():
        h  = row["home_team"]
        a  = row["away_team"]
        hg = int(row["home_goals_ft"]) if pd.notna(row["home_goals_ft"]) else 0
        ag = int(row["away_goals_ft"]) if pd.notna(row["away_goals_ft"]) else 0

        rh, ra = pi.update(h, a, hg, ag)
        rows.append({
            "match_id":          row["match_id"],
            "home_pi_ha":        rh["ha"],
            "home_pi_hd":        rh["hd"],
            "away_pi_aa":        ra["aa"],
            "away_pi_ad":        ra["ad"],
            "pi_strength_diff":  (rh["ha"] - ra["ad"]) - (ra["aa"] - rh["hd"]),
        })

    df_pi = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True).merge(df_pi, on="match_id", how="left")
    print(f"  ✅ Pi-ratings computed")
    return df


# =============================================================================
# SECTION 6 — Historical odds (football-data.co.uk CSVs)
# =============================================================================

FDCO_BASE = "https://www.football-data.co.uk/mmz4281"
FDCO_SEASONS = {
    "2014-15": "1415", "2015-16": "1516", "2016-17": "1617",
    "2017-18": "1718", "2018-19": "1819", "2019-20": "1920",
    "2020-21": "2021", "2021-22": "2122", "2022-23": "2223",
    "2023-24": "2324",
}

FDCO_NAME_MAP = {
    "Man United":    "Manchester United",
    "Man City":      "Manchester City",
    "Spurs":         "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves":        "Wolves",
    "West Ham":      "West Ham",
    "West Brom":     "West Brom",
    "Newcastle":     "Newcastle",
    "QPR":           "QPR",
    "Sheffield United": "Sheffield United",
    "Brighton":      "Brighton",
    "Bournemouth":   "Bournemouth",
    "Leicester":     "Leicester",
    "Brentford":     "Brentford",
    "Luton":         "Luton",
    "Leeds":         "Leeds",
    "Fulham":        "Fulham",
}


def fetch_fdco_odds(season_str: str) -> pd.DataFrame:
    code = FDCO_SEASONS[season_str]
    url  = f"{FDCO_BASE}/{code}/E0.csv"
    df   = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
    df["season"] = season_str
    return df


def clean_fdco(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["Date", "HomeTeam", "AwayTeam", "season"]
    for c in ["B365H","B365D","B365A","PSH","PSD","PSA","BWH","BWD","BWA"]:
        if c in df_raw.columns:
            cols.append(c)

    df = df_raw[cols].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam"])
    df = df.rename(columns={"Date":"date","HomeTeam":"home_team","AwayTeam":"away_team"})

    for h_col, d_col, a_col in [("B365H","B365D","B365A"), ("PSH","PSD","PSA")]:
        if all(c in df.columns for c in [h_col, d_col, a_col]):
            pfx = h_col[:3].lower()
            raw_h, raw_d, raw_a = 1/df[h_col], 1/df[d_col], 1/df[a_col]
            total = raw_h + raw_d + raw_a
            df[f"{pfx}_prob_h"]     = raw_h / total
            df[f"{pfx}_prob_d"]     = raw_d / total
            df[f"{pfx}_prob_a"]     = raw_a / total
            df[f"{pfx}_overround"]  = total

    return df


def pull_historical_odds() -> pd.DataFrame:
    print("\n── STEP 7: Pulling historical odds (football-data.co.uk) ───────")
    dfs = []
    for season in tqdm(SEASONS, desc="Odds CSVs"):
        print(f"  {season}...", end=" ", flush=True)
        try:
            raw     = fetch_fdco_odds(season)
            cleaned = clean_fdco(raw)
            dfs.append(cleaned)
            print(f"{len(cleaned)} rows")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.5)

    df = pd.concat(dfs, ignore_index=True)
    df["home_team"] = df["home_team"].replace(FDCO_NAME_MAP)
    df["away_team"] = df["away_team"].replace(FDCO_NAME_MAP)

    df.to_parquet(RAW_DIR / "odds_raw.parquet", index=False)
    print(f"  ✅ {len(df)} odds rows across {df.season.nunique()} seasons")
    return df


# =============================================================================
# SECTION 7 — Live odds via The Odds API
# =============================================================================

def fetch_live_odds() -> pd.DataFrame:
    print("\n── STEP 8: Fetching live odds from The Odds API ────────────────")

    if ODDS_API_KEY == "YOUR_KEY_HERE":
        print("  ⚠️  ODDS_API_KEY not set — skipping.")
        print("     Set it in .env or at the top of this file.")
        return pd.DataFrame()

    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
    params = {
        "apiKey":      ODDS_API_KEY,
        "regions":     "uk",
        "markets":     "h2h",
        "oddsFormat":  "decimal",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for game in data:
        home = game["home_team"]
        away = game["away_team"]
        date = game["commence_time"][:10]

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
                "date":          date,
                "home_team":     home,
                "away_team":     away,
                "live_prob_h":   (1/avg_h) / total,
                "live_prob_d":   (1/avg_d) / total,
                "live_prob_a":   (1/avg_a) / total,
                "n_bookmakers":  len(game.get("bookmakers", [])),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(RAW_DIR / "live_odds.parquet", index=False)
        print(f"  ✅ {len(df)} upcoming fixtures with live odds")
        print(df.to_string(index=False))
    return df


# =============================================================================
# SECTION 8 — Assemble feature table + write to SQLite & Parquet
# =============================================================================

def build_feature_table(df_master: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 9: Assembling full feature table ───────────────────────")

    # Merge historical odds
    odds_cols = ["date","home_team","away_team","season"] + [
        c for c in df_odds.columns if c.startswith(("b36","ps_","bw_"))
    ]
    df = df_master.merge(
        df_odds[[c for c in odds_cols if c in df_odds.columns]],
        on=["date","home_team","away_team","season"],
        how="left"
    )

    df["result_code"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    print(f"  ✅ Feature table: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def write_to_db(df_feat: pd.DataFrame, df_long: pd.DataFrame):
    print("\n── STEP 10: Writing to SQLite + Parquet ────────────────────────")

    conn = sqlite3.connect(DB_PATH)

    # Raw matches
    df_feat[["match_id","season","matchday","date","home_team","away_team",
             "home_goals_ft","away_goals_ft","home_goals_ht","away_goals_ht",
             "winner","referee","result","home_points","away_points"]
    ].assign(date=lambda x: x["date"].astype(str)
    ).to_sql("matches", conn, if_exists="replace", index=False)

    # Full feature table
    df_feat_sql = df_feat.copy()
    df_feat_sql["date"] = df_feat_sql["date"].astype(str)
    df_feat_sql.to_sql("features", conn, if_exists="replace", index=False)

    # Team ratings snapshot (latest per team)
    latest_h = (df_feat.sort_values("date").groupby("home_team").last()
                [["home_elo","home_elo_goal","home_pi_ha","home_pi_hd"]]
                .reset_index().rename(columns={"home_team":"team","home_elo":"elo",
                                                "home_elo_goal":"elo_goal","home_pi_ha":"pi_ha","home_pi_hd":"pi_hd"}))
    latest_a = (df_feat.sort_values("date").groupby("away_team").last()
                [["away_elo","away_elo_goal","away_pi_aa","away_pi_ad"]]
                .reset_index().rename(columns={"away_team":"team","away_elo":"elo_away",
                                                "away_elo_goal":"elo_goal_away","away_pi_aa":"pi_aa","away_pi_ad":"pi_ad"}))
    df_ratings = (latest_h.merge(latest_a, on="team", how="outer")
                  .assign(elo_combined=lambda x: x[["elo","elo_away"]].mean(axis=1))
                  .sort_values("elo_combined", ascending=False).reset_index(drop=True))
    df_ratings.to_sql("team_ratings", conn, if_exists="replace", index=False)

    # Indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_date   ON features (date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_teams  ON features (home_team, away_team)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_season ON features (season)")
    conn.commit()
    conn.close()

    # Parquet
    df_feat.to_parquet(FEAT_DIR / "features.parquet", index=False)
    df_ratings.to_parquet(FEAT_DIR / "team_ratings.parquet", index=False)

    size_kb = DB_PATH.stat().st_size / 1024
    print(f"  SQLite: {DB_PATH}  ({size_kb:.0f} KB)")
    print(f"  Parquet: {FEAT_DIR / 'features.parquet'}")
    print(f"  Parquet: {FEAT_DIR / 'team_ratings.parquet'}")
    return df_ratings


# =============================================================================
# SECTION 9 — Verification report
# =============================================================================

def verification_report(df_ratings: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  PHASE 1 — VERIFICATION REPORT")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    for table in ["matches", "features", "team_ratings"]:
        n = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn).iloc[0, 0]
        print(f"\n  [{table}]: {n} rows")

    print("\n  Feature coverage (sample of 1000 rows):")
    df_check = pd.read_sql("SELECT * FROM features LIMIT 1000", conn)
    check_cols = ["home_xg","away_xg","home_xg_5","home_elo","home_pi_ha","b36_prob_h"]
    for col in check_cols:
        if col in df_check.columns:
            pct = df_check[col].notna().mean() * 100
            icon = "✅" if pct > 80 else ("⚠️ " if pct > 50 else "❌")
            print(f"    {icon}  {col:<25} {pct:.1f}%")

    print("\n  Season coverage:")
    df_s = pd.read_sql(
        "SELECT season, COUNT(*) as matches FROM matches GROUP BY season ORDER BY season",
        conn
    )
    print(df_s.to_string(index=False))

    print("\n  Current team ELO top 10:")
    print(df_ratings[["team","elo_combined"]].head(10).to_string(index=False))

    conn.close()

    print("\n" + "=" * 60)
    print("  ✅ Phase 1 Complete — feature store ready for Phase 2")
    print("=" * 60)

    print("""
  Output files:
    data/pl_predictor.db           ← SQLite (matches, features, team_ratings)
    data/raw/results_raw.parquet   ← football-data.org raw results
    data/raw/xg_raw.parquet        ← Understat xG data
    data/raw/odds_raw.parquet      ← football-data.co.uk historical odds
    data/features/features.parquet ← ML training table  ← main Phase 2 input
    data/features/team_ratings.parquet
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

    if FOOTBALL_DATA_KEY == "YOUR_KEY_HERE":
        print("⚠️  WARNING: FOOTBALL_DATA_KEY not set.")
        print("   Get a free key at https://www.football-data.org/client/register")
        print("   Then set it in .env or at the top of this file.\n")

    setup_dirs()

    # 1. Results
    df_results = pull_all_results()

    # 2. xG
    df_xg = pull_understat_xg()

    # 3. Merge
    df_master = merge_results_xg(df_results, df_xg)

    # 4. Rolling features
    df_master, df_long = build_rolling_features(df_master)

    # 5. ELO
    df_master = compute_elo(df_master)

    # 6. Pi-ratings
    df_master = compute_pi_ratings(df_master)

    # 7. Historical odds
    df_odds = pull_historical_odds()

    # 8. Live odds (skipped if key not set)
    fetch_live_odds()

    # 9. Assemble
    df_feat = build_feature_table(df_master, df_odds)

    # 10. Write DB
    df_ratings = write_to_db(df_feat, df_long)

    # 11. Report
    verification_report(df_ratings)


if __name__ == "__main__":
    main()
