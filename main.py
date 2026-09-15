"""
Dixon-Coles Soccer Match Prediction API
========================================

Implements the Dixon & Coles (1997) bivariate-Poisson model with the
Dixon-Coles low-score correction and exponential time-decay weighting
(as recommended in the original paper and used in nearly all published
follow-up work — see README for citations). This is the model that
academic comparisons consistently treat as the credible statistical
baseline for football score prediction; it is not "the world's most
accurate model" in any absolute sense (no such single model exists —
see README), but it is the most defensible, well-documented, reproducible
approach you can stand up yourself, rather than a black box.

Data source: football-data.co.uk (free historical CSVs, no key required).

Deploy: Render (see README / render.yaml). Runs fine on a free instance.
"""

import os
import io
import time
import math
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dixon-coles-api")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# football-data.co.uk league codes:
# E0 = English Premier League, E1 = Championship, SP1 = La Liga,
# I1 = Serie A, D1 = Bundesliga, F1 = Ligue 1, ... etc.
LEAGUE = os.environ.get("LEAGUE", "E0")

# Seasons to pull, most recent first, format "2425" = 2024/25.
# football-data.co.uk keeps ~ the last 25 years available like this.
SEASONS = os.environ.get("SEASONS", "2526,2425,2324").split(",")

DATA_URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Time-decay factor (xi) from Dixon & Coles (1997). Larger = forgets
# older matches faster. 0.0018 corresponds to a "half-life" of roughly
# one season, which is the commonly used default in the literature.
XI = float(os.environ.get("XI", "0.0018"))

MAX_GOALS = 10  # score matrix truncation

# Allow the deployed frontend (Vercel) to call this API.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def fetch_league_data(league: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        season = season.strip()
        url = DATA_URL_TEMPLATE.format(season=season, league=league)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df = df[[c for c in REQUIRED_COLS if c in df.columns]].dropna()
            if len(df) == 0 or not set(REQUIRED_COLS).issubset(df.columns):
                log.warning("Season %s for %s missing required columns, skipping", season, league)
                continue
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Date"])
            frames.append(df)
            log.info("Loaded %d matches from %s season %s", len(df), league, season)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load %s season %s: %s", league, season, e)

    if not frames:
        raise RuntimeError(
            f"Could not load any data for league={league}. "
            "Check the LEAGUE / SEASONS environment variables."
        )

    data = pd.concat(frames, ignore_index=True)
    data["FTHG"] = data["FTHG"].astype(int)
    data["FTAG"] = data["FTAG"].astype(int)
    data = data.sort_values("Date").reset_index(drop=True)
    return data


# --------------------------------------------------------------------------
# Dixon-Coles model
# --------------------------------------------------------------------------

def rho_correction(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score dependence correction, tau(x,y) from Dixon & Coles (1997)."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


class DixonColesModel:
    def __init__(self):
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self.fitted_at: Optional[float] = None
        self.n_matches: int = 0
        self.league: str = LEAGUE

    def fit(self, df: pd.DataFrame, xi: float = XI):
        teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        max_date = df["Date"].max()
        days_ago = (max_date - df["Date"]).dt.days.values
        weights = np.exp(-xi * days_ago)

        home_idx = df["HomeTeam"].map(idx).values
        away_idx = df["AwayTeam"].map(idx).values
        fthg = df["FTHG"].values
        ftag = df["FTAG"].values

        def unpack(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]
            return attack, defense, home_adv, rho

        def neg_log_likelihood(params):
            attack, defense, home_adv, rho = unpack(params)
            lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] + defense[home_idx])

            log_lik = poisson.logpmf(fthg, lam) + poisson.logpmf(ftag, mu)

            # Vectorised low-score correction (only affects 0/1 scorelines)
            tau = np.ones_like(lam)
            m00 = (fthg == 0) & (ftag == 0)
            m01 = (fthg == 0) & (ftag == 1)
            m10 = (fthg == 1) & (ftag == 0)
            m11 = (fthg == 1) & (ftag == 1)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.clip(tau, 1e-6, None)

            log_lik = log_lik + np.log(tau)
            return -np.sum(weights * log_lik)

        # Identifiability constraint: mean attack strength pinned to 0
        # via a soft penalty (keeps optimizer well-behaved without
        # needing constrained optimization).
        def objective(params):
            attack, defense, home_adv, rho = unpack(params)
            penalty = 1000.0 * (np.mean(attack)) ** 2
            return neg_log_likelihood(params) + penalty

        init = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [0.0]])
        bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-2, 2)] + [(-1, 1)]

        result = minimize(
            objective, init, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success:
            log.warning("Optimizer did not fully converge: %s", result.message)

        attack, defense, home_adv, rho = unpack(result.x)

        self.teams = teams
        self.attack = dict(zip(teams, attack))
        self.defense = dict(zip(teams, defense))
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.fitted_at = time.time()
        self.n_matches = len(df)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        if home not in self.attack or away not in self.attack:
            raise KeyError("Unknown team")
        lam = math.exp(self.attack[home] + self.defense[away] + self.home_adv)
        mu = math.exp(self.attack[away] + self.defense[home])
        return lam, mu

    def score_matrix(self, home: str, away: str, max_goals: int = MAX_GOALS) -> np.ndarray:
        lam, mu = self.expected_goals(home, away)
        home_probs = poisson.pmf(np.arange(max_goals + 1), lam)
        away_probs = poisson.pmf(np.arange(max_goals + 1), mu)
        matrix = np.outer(home_probs, away_probs)

        for x in range(2):
            for y in range(2):
                matrix[x, y] *= rho_correction(x, y, lam, mu, self.rho)

        matrix = matrix / matrix.sum()  # renormalize after correction
        return matrix

    def predict(self, home: str, away: str) -> dict:
        matrix = self.score_matrix(home, away)
        lam, mu = self.expected_goals(home, away)

        p_home = float(np.tril(matrix, -1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, 1).sum())

        goals = np.arange(matrix.shape[0])
        total_goals_grid = goals[:, None] + goals[None, :]
        p_over25 = float(matrix[total_goals_grid > 2.5].sum())
        p_under25 = 1 - p_over25

        p_btts_yes = float(matrix[1:, 1:].sum())
        p_btts_no = 1 - p_btts_yes

        flat_idx = np.dstack(np.unravel_index(np.argsort(-matrix.ravel())[:5], matrix.shape))[0]
        top_scores = [
            {"home_goals": int(i), "away_goals": int(j), "probability": float(matrix[i, j])}
            for i, j in flat_idx
        ]

        return {
            "home_team": home,
            "away_team": away,
            "expected_goals": {"home": round(lam, 3), "away": round(mu, 3)},
            "result_probabilities": {
                "home_win": round(p_home, 4),
                "draw": round(p_draw, 4),
                "away_win": round(p_away, 4),
            },
            "over_under_2_5": {
                "over": round(p_over25, 4),
                "under": round(p_under25, 4),
            },
            "both_teams_to_score": {
                "yes": round(p_btts_yes, 4),
                "no": round(p_btts_no, 4),
            },
            "most_likely_scorelines": top_scores,
            "model_meta": {
                "league": self.league,
                "n_matches_used": self.n_matches,
                "home_advantage_param": round(self.home_adv, 4),
                "rho_param": round(self.rho, 4),
            },
        }


model = DixonColesModel()

# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(
    title="Dixon-Coles Soccer Prediction API",
    description="Match outcome probabilities from a fitted Dixon-Coles bivariate Poisson model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RefreshResponse(BaseModel):
    status: str
    league: str
    n_matches: int
    n_teams: int


@app.on_event("startup")
def startup_event():
    try:
        df = fetch_league_data(LEAGUE, SEASONS)
        model.fit(df)
        log.info(
            "Model fitted: %d matches, %d teams, home_adv=%.3f, rho=%.3f",
            model.n_matches, len(model.teams), model.home_adv, model.rho,
        )
    except Exception as e:  # noqa: BLE001
        # Don't crash the whole app if the data source is briefly down;
        # /predict will just report the model as not-ready until /refresh.
        log.error("Startup fit failed: %s", e)


@app.get("/health")
def health():
    return {
        "status": "ok" if model.fitted_at else "model_not_fitted",
        "league": model.league,
        "n_matches": model.n_matches,
        "n_teams": len(model.teams),
    }


@app.get("/teams")
def teams():
    if not model.teams:
        raise HTTPException(status_code=503, detail="Model not fitted yet. Try /refresh.")
    return {"league": model.league, "teams": model.teams}


@app.get("/predict")
def predict(
    home: str = Query(..., description="Home team name, exactly as in /teams"),
    away: str = Query(..., description="Away team name, exactly as in /teams"),
):
    if not model.teams:
        raise HTTPException(status_code=503, detail="Model not fitted yet. Try /refresh.")
    if home == away:
        raise HTTPException(status_code=400, detail="Home and away team must differ.")
    try:
        return model.predict(home, away)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team. Call /teams for valid names in league {model.league}.",
        )


@app.post("/refresh", response_model=RefreshResponse)
def refresh(league: Optional[str] = None, seasons: Optional[str] = None):
    """Re-fetch data and re-fit the model. Call this periodically (e.g. weekly)
    via a cron job / Render scheduled job, since match data changes every round."""
    global model
    use_league = league or LEAGUE
    use_seasons = seasons.split(",") if seasons else SEASONS
    df = fetch_league_data(use_league, use_seasons)
    new_model = DixonColesModel()
    new_model.league = use_league
    new_model.fit(df)
    model = new_model
    return RefreshResponse(
        status="refreshed",
        league=model.league,
        n_matches=model.n_matches,
        n_teams=len(model.teams),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
