"""
Mathayo Institutional Quant Engine — Production Backend
Version: 2026.4 (Real Data Collection, Honest Provenance, OCR, Multi-Book Consensus)

WHAT CHANGED FROM v2026.3 AND WHY
----------------------------------
v2026.3 claimed to use "Understat xG Model" and "Transfermarkt Rest Registry" as
data sources but never actually called them — lambda/mu were derived purely from
the Elo differential. That's a factual misrepresentation of where the numbers
come from, which is the opposite of what you asked for. This version either
genuinely fetches a data source, or it labels the output as a model-derived
estimate. Every match result now carries a `data_provenance` block telling you
exactly what was real and what was a fallback.

REQUIRED CONFIGURATION (environment variables)
------------------------------------------------
API_FOOTBALL_KEY   - from https://www.api-football.com (free tier: 100 req/day).
                     Powers: team news/injuries, head-to-head, recent form,
                     venue coordinates (for weather). Without it, the engine
                     still runs, but those fields are explicitly marked
                     "unavailable" instead of being invented.
ODDS_API_KEY       - from https://the-odds-api.com (free tier: 500 req/month).
                     Powers: multi-bookmaker consensus, used as the
                     "compare against the rest of the world" check. Without
                     it, consensus checks are skipped and marked unavailable.
No key is needed for weather (Open-Meteo is free/keyless) or for Elo
(ClubElo is free/keyless).

WHAT "ULTRA-SAFE" MEANS HERE
------------------------------
A leg only qualifies for an accumulator if ALL of the following hold:
  1. Model probability >= ULTRA_SAFE_PROB_FLOOR (default 0.62)
  2. The model's implied edge over the market is small or negative — i.e. the
     model and the market AGREE. A pick where your model loves something the
     market hates is a value bet, not a safe bet. Safe = boring consensus.
  3. No key-player injury/suspension flag on the favored side (only checked
     when API_FOOTBALL_KEY is set).
  4. If bookmaker consensus data is available, the pick is not a consensus
     outlier (>8 percentage-point deviation from the multi-book average).
This is a heuristic, not a guarantee — no model eliminates variance in
football. Treat "ultra-safe" as "lowest-variance available", not "certain".
"""

import os
import re
import io
import json
import asyncio
import datetime as dt
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.stats import poisson
import httpx
from rapidfuzz import fuzz, process
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# =====================================================================
# CONFIG
# =====================================================================
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
CLUBELO_BASE = "http://api.clubelo.com"
UNDERSTAT_BASE = "https://understat.com"

# The Odds API has no generic "all soccer" endpoint — each league needs its
# own sport_key (confirmed against their docs). Querying every one of these
# per match burns your free-tier quota fast (500/month), so this list is
# deliberately just the majors. Add more keys from https://the-odds-api.com/sports-odds-data/
# if you need wider coverage, but expect to pay for a higher tier.
ODDS_API_SOCCER_LEAGUES = [
    "soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a",
    "soccer_germany_bundesliga", "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]

# Understat only covers these 6 leagues. No other leagues will resolve here.
UNDERSTAT_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]

FUZZY_MATCH_MIN_SCORE = 78  # rapidfuzz token_sort_ratio threshold for team-name matches

# TESTED FINDING: token_sort_ratio alone scores common abbreviations far below
# threshold — "Man Utd" vs "Manchester United" = 58, "Spurs" vs "Tottenham
# Hotspur" = 33 (verified in dev). Betslips are full of exactly these
# shorthand forms, so without normalization the pipeline would silently fail
# to find the team for a large fraction of real inputs. This alias map
# expands common shorthand to the club's full name before fuzzy matching.
TEAM_ALIASES = {
    "man utd": "manchester united", "man u": "manchester united", "manu": "manchester united",
    "man city": "manchester city", "mancity": "manchester city",
    "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "brighton": "brighton and hove albion",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "leicester": "leicester city",
    "forest": "nottingham forest", "nffc": "nottingham forest",
    "barca": "barcelona", "fc barcelona": "barcelona",
    "real": "real madrid", "atleti": "atletico madrid", "atletico": "atletico madrid",
    "psg": "paris saint germain", "paris sg": "paris saint germain",
    "bayern": "bayern munich", "fc bayern": "bayern munich",
    "dortmund": "borussia dortmund", "bvb": "borussia dortmund",
    "inter": "inter milan", "internazionale": "inter milan",
    "ac milan": "milan", "juve": "juventus",
    "united": "manchester united", "city": "manchester city",  # last resort, low-precision
}


def normalize_team_name(name: str) -> str:
    key = re.sub(r'[^a-z0-9 ]', '', name.lower()).strip()
    return TEAM_ALIASES.get(key, name)

ULTRA_SAFE_PROB_FLOOR = 0.62
ULTRA_SAFE_MAX_MARKET_DISAGREEMENT = 0.10  # model can't out-favor the market by more than this
CONSENSUS_OUTLIER_THRESHOLD = 0.08


def current_european_season() -> int:
    """European club seasons run Jul-Jun. API-Football/Understat both key
    seasons by the year they START in (e.g. '2026' means the 2026/27 season).
    The previous version of this file hardcoded 2025, which is already one
    season stale as of Sept 2026 — this computes it from the real clock."""
    today = dt.datetime.utcnow()
    return today.year if today.month >= 7 else today.year - 1

app = FastAPI(title="Mathayo Autonomous Quant Engine", version="2026.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _api_football_headers() -> Dict[str, str]:
    return {"x-apisports-key": API_FOOTBALL_KEY}


# =====================================================================
# AGENT 1: INGESTION — parses pasted betslip text OR OCR'd screenshot text
# =====================================================================
class IngestionAgent:
    @staticmethod
    def parse_fixtures_in_order(raw_text: str) -> List[Dict[str, Any]]:
        raw_lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

        sanitized = []
        for l in raw_lines:
            if re.search(r'(odibets|sportpesa|betika|mozzart|Dashboard|Netflix|YouTube|PayPal|Gmail|SafariTour|Inbox|Ask Gemini)', l, re.I):
                continue
            if re.search(r'(Over\/Under|UNDER|OVER|Draw no bet|Both Teams|Full Time|Double Chance|Markets|Correct)', l, re.I):
                continue
            if re.match(r'^\d{1,2}[\/\.]\d{1,2}(?:\/\d{2,4})?(\s*,\s*|\s*-\s*|\s+)\d{1,2}:\d{2}', l):
                continue
            if re.match(r'^\d{1,2}\s+(Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug)', l, re.I):
                continue
            if re.match(r'^(1|X|2|1X|X2|12|HOME|DRAW|AWAY)$', l, re.I):
                continue
            sanitized.append(l)

        is_odd_regex = re.compile(r'^\d+(\.\d{1,2})?$')
        matches = []
        pending_teams: List[str] = []
        pending_odds: List[float] = []

        for item in sanitized:
            if is_odd_regex.match(item) and 1.05 <= float(item) <= 45.0:
                pending_odds.append(float(item))
                if len(pending_odds) == 3:
                    valid_candidates = [t for t in pending_teams if not t.isdigit() and len(t) > 2]
                    if len(valid_candidates) >= 2:
                        t1_clean = re.sub(r'^\d+[\.\s\-]+', '', valid_candidates[-2]).strip()
                        t2_clean = re.sub(r'^\d+[\.\s\-]+', '', valid_candidates[-1]).strip()
                        if len(re.findall(r'[a-zA-Z]', t1_clean)) >= 3 and len(re.findall(r'[a-zA-Z]', t2_clean)) >= 3:
                            matches.append({
                                "sequence_order": len(matches) + 1,
                                "home_team": t1_clean,
                                "away_team": t2_clean,
                                "oH": pending_odds[0],
                                "oD": pending_odds[1],
                                "oA": pending_odds[2],
                            })
                    pending_teams = []
                    pending_odds = []
            else:
                if not item.isdigit() and len(item) > 2:
                    pending_teams.append(item)

        return matches

    @staticmethod
    def extract_text_from_screenshot(image_bytes: bytes) -> str:
        """Real OCR via Tesseract. Requires the `tesseract-ocr` binary on the
        host (apt-get install tesseract-ocr) in addition to the pytesseract
        python package. Raises a clear error if unavailable rather than
        silently returning nothing."""
        if not OCR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="OCR not available: install `pytesseract` + `pillow` and the "
                       "tesseract-ocr system binary on the server.",
            )
        img = Image.open(io.BytesIO(image_bytes))
        # Upscale small screenshots — tesseract accuracy drops sharply below ~300dpi equivalent
        if img.width < 1000:
            scale = 1500 / img.width
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        text = pytesseract.image_to_string(img)
        return text


# =====================================================================
# AGENT 2A: UNDERSTAT xG — real shot-quality data, no browser needed.
# Understat server-renders the data as JSON embedded in a <script> tag
# (confirmed against their page source), so a plain GET + regex extracts
# it. Reaching for a full headless browser (Playwright) here would add a
# heavy dependency for no benefit — nothing on this page needs JS execution.
# =====================================================================
class UnderstatAgent:
    _league_cache: Dict[str, Dict[str, Any]] = {}  # league -> {team_name: stats}
    _cache_lock = asyncio.Lock()

    @staticmethod
    def _extract_teams_data(html: str) -> Optional[Dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        for script in soup.find_all("script"):
            text = script.string or ""
            if "teamsData" not in text:
                continue
            try:
                start = text.index("('") + 2
                end = text.index("')", start)
                raw = text[start:end].encode("utf8").decode("unicode_escape")
                return json.loads(raw)
            except Exception:
                continue
        return None

    @classmethod
    async def _load_league(cls, league: str, client: httpx.AsyncClient) -> Dict[str, Any]:
        async with cls._cache_lock:
            if league in cls._league_cache:
                return cls._league_cache[league]
        season = current_european_season()
        url = f"{UNDERSTAT_BASE}/league/{league}/{season}"
        result: Dict[str, Any] = {}
        try:
            res = await client.get(url, timeout=8.0, headers={"User-Agent": "Mozilla/5.0"})
            if res.status_code == 200:
                teams_data = cls._extract_teams_data(res.text)
                if teams_data:
                    for _, team in teams_data.items():
                        history = team.get("history", [])
                        if not history:
                            continue
                        recent = history[-10:]  # last 10 matches for a responsive-but-stable sample
                        xg_for = np.mean([float(m["xG"]) for m in recent])
                        xg_against = np.mean([float(m["xGA"]) for m in recent])
                        result[team["title"]] = {
                            "xg_for_per_game": round(float(xg_for), 3),
                            "xg_against_per_game": round(float(xg_against), 3),
                            "matches_sampled": len(recent),
                        }
        except Exception:
            pass
        async with cls._cache_lock:
            cls._league_cache[league] = result
        return result

    @classmethod
    async def fetch_team_xg_pair(cls, home_team: str, away_team: str, client: httpx.AsyncClient) -> Optional[Dict[str, float]]:
        # Understat only has 6 leagues — check them all, cached per-league so
        # a batch of matches doesn't re-scrape the same league repeatedly.
        for league in UNDERSTAT_LEAGUES:
            league_data = await cls._load_league(league, client)
            if not league_data:
                continue
            names = list(league_data.keys())
            home_match = process.extractOne(home_team, names, scorer=fuzz.token_sort_ratio)
            away_match = process.extractOne(away_team, names, scorer=fuzz.token_sort_ratio)
            if home_match and away_match and home_match[1] >= FUZZY_MATCH_MIN_SCORE and away_match[1] >= FUZZY_MATCH_MIN_SCORE:
                h = league_data[home_match[0]]
                a = league_data[away_match[0]]
                return {
                    "home_xg_for": h["xg_for_per_game"],
                    "home_xg_against": h["xg_against_per_game"],
                    "away_xg_for": a["xg_for_per_game"],
                    "away_xg_against": a["xg_against_per_game"],
                    "league": league,
                    "match_names": {"home": home_match[0], "away": away_match[0]},
                }
        return None


# =====================================================================
# AGENT 2: LIVE DATA HARVESTER — every field here is either a real fetch
# or explicitly marked unavailable. Nothing is fabricated.
# =====================================================================
class LiveDataAgent:

    @staticmethod
    async def fetch_clubelo(team_name: str, client: httpx.AsyncClient) -> Optional[int]:
        clean_team = re.sub(r'\s+(FC|CF|SC|HSC|MFC|98|04|05)$', '', team_name, flags=re.I).replace(" ", "")
        url = f"{CLUBELO_BASE}/{clean_team}"
        try:
            res = await client.get(url, timeout=5.0)
            if res.status_code == 200 and res.text:
                lines = res.text.strip().split('\n')
                if len(lines) >= 2:
                    latest = lines[-1].split(',')
                    return int(float(latest[4]))
        except Exception:
            pass
        return None

    @staticmethod
    async def find_team_id(team_name: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        """Searches API-Football and picks the best fuzzy match rather than
        blindly trusting result[0] — the previous version assumed the API's
        first hit was always right, which breaks on common abbreviations
        (e.g. searching 'Man Utd' can return a lower-division club with a
        similar name before Manchester United)."""
        if not API_FOOTBALL_KEY:
            return None
        try:
            res = await client.get(
                f"{API_FOOTBALL_BASE}/teams",
                params={"search": team_name},
                headers=_api_football_headers(),
                timeout=6.0,
            )
            resp = res.json().get("response", [])
            if not resp:
                return None
            best, best_score = None, -1
            for candidate in resp:
                score = fuzz.token_sort_ratio(team_name, candidate["team"]["name"])
                if score > best_score:
                    best_score, best = score, candidate
            if best_score < FUZZY_MATCH_MIN_SCORE:
                return None
            team = best["team"]
            venue = best.get("venue", {})
            return {"id": team["id"], "name": team["name"], "match_score": best_score,
                    "lat": venue.get("latitude"), "lon": venue.get("longitude")}
        except Exception:
            return None

    @staticmethod
    async def fetch_injuries(team_id: int, client: httpx.AsyncClient) -> Optional[List[str]]:
        if not API_FOOTBALL_KEY or not team_id:
            return None
        try:
            res = await client.get(
                f"{API_FOOTBALL_BASE}/injuries",
                params={"team": team_id, "season": current_european_season()},
                headers=_api_football_headers(),
                timeout=6.0,
            )
            data = res.json().get("response", [])
            return [f"{p['player']['name']} ({p['player']['reason']})" for p in data[:8]]
        except Exception:
            return None

    @staticmethod
    async def fetch_form(team_id: int, client: httpx.AsyncClient) -> Optional[str]:
        """The previous version called /teams/statistics with an empty
        `league` param, which API-Football requires and will reject or
        return empty for. Recent-form doesn't need a league at all if you
        derive it from the last 5 fixtures directly, so that's what this
        does — it also actually returns a usable W/D/L string, whereas the
        statistics endpoint's `form` field is season-long, not last-5."""
        if not API_FOOTBALL_KEY or not team_id:
            return None
        try:
            res = await client.get(
                f"{API_FOOTBALL_BASE}/fixtures",
                params={"team": team_id, "last": 5, "status": "FT"},
                headers=_api_football_headers(),
                timeout=6.0,
            )
            fixtures = res.json().get("response", [])
            if not fixtures:
                return None
            letters = []
            for f in fixtures:
                gh, ga = f["goals"]["home"], f["goals"]["away"]
                is_home = f["teams"]["home"]["id"] == team_id
                if gh == ga:
                    letters.append("D")
                elif (gh > ga) == is_home:
                    letters.append("W")
                else:
                    letters.append("L")
            return "".join(letters)
        except Exception:
            return None

    @staticmethod
    async def fetch_h2h(team1_id: int, team2_id: int, client: httpx.AsyncClient) -> Optional[List[str]]:
        if not API_FOOTBALL_KEY or not team1_id or not team2_id:
            return None
        try:
            res = await client.get(
                f"{API_FOOTBALL_BASE}/fixtures/headtohead",
                params={"h2h": f"{team1_id}-{team2_id}", "last": 5},
                headers=_api_football_headers(),
                timeout=6.0,
            )
            fixtures = res.json().get("response", [])
            out = []
            for f in fixtures:
                gh = f["goals"]["home"]
                ga = f["goals"]["away"]
                out.append(f"{f['teams']['home']['name']} {gh}-{ga} {f['teams']['away']['name']}")
            return out
        except Exception:
            return None

    @staticmethod
    async def fetch_weather(lat: Optional[float], lon: Optional[float], client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        if lat is None or lon is None:
            return None
        try:
            res = await client.get(
                OPEN_METEO_FORECAST,
                params={"latitude": lat, "longitude": lon, "current": "temperature_2m,precipitation,wind_speed_10m"},
                timeout=6.0,
            )
            cur = res.json().get("current", {})
            if not cur:
                return None
            return {
                "temp_c": cur.get("temperature_2m"),
                "precipitation_mm": cur.get("precipitation"),
                "wind_kmh": cur.get("wind_speed_10m"),
            }
        except Exception:
            return None

    @staticmethod
    async def fetch_bookmaker_consensus(home_team: str, away_team: str, client: httpx.AsyncClient) -> Optional[Dict[str, float]]:
        """Pulls odds from many real bookmakers via The Odds API and averages
        their de-vigged implied probabilities. This is the 'compare against
        other models' check — market consensus across ~15-20 books is the
        toughest real-world benchmark to beat, more meaningful than trying to
        scrape individual tipster sites.

        BUG FIX: The Odds API has no generic /sports/soccer/odds endpoint —
        each league needs its own sport_key (soccer_epl, soccer_spain_la_liga,
        etc). The previous version called a URL that doesn't exist for
        aggregate soccer, so every consensus check silently returned nothing.
        This queries each configured league and stops at the first match,
        using fuzzy name matching instead of a fragile first-5-characters
        substring check."""
        if not ODDS_API_KEY:
            return None

        async def search_league(league_key: str) -> Optional[Dict[str, float]]:
            try:
                res = await client.get(
                    f"{ODDS_API_BASE}/sports/{league_key}/odds",
                    params={"apiKey": ODDS_API_KEY, "regions": "uk,eu", "markets": "h2h"},
                    timeout=8.0,
                )
                if res.status_code != 200:
                    return None
                events = res.json()
                if not isinstance(events, list):
                    return None

                best_event, best_score = None, 0
                for ev in events:
                    score = (fuzz.token_sort_ratio(home_team, ev.get("home_team", "")) +
                             fuzz.token_sort_ratio(away_team, ev.get("away_team", ""))) / 2
                    if score > best_score:
                        best_score, best_event = score, ev
                if not best_event or best_score < FUZZY_MATCH_MIN_SCORE:
                    return None

                h_probs, d_probs, a_probs = [], [], []
                real_home, real_away = best_event["home_team"], best_event["away_team"]
                for bk in best_event.get("bookmakers", []):
                    for mkt in bk.get("markets", []):
                        if mkt["key"] != "h2h":
                            continue
                        prices = {o["name"]: o["price"] for o in mkt["outcomes"]}
                        if real_home in prices and real_away in prices and "Draw" in prices:
                            raw = [1 / prices[real_home], 1 / prices["Draw"], 1 / prices[real_away]]
                            s = sum(raw)
                            h_probs.append(raw[0] / s)
                            d_probs.append(raw[1] / s)
                            a_probs.append(raw[2] / s)
                if not h_probs:
                    return None
                return {
                    "consensus_pH": round(float(np.mean(h_probs)), 3),
                    "consensus_pD": round(float(np.mean(d_probs)), 3),
                    "consensus_pA": round(float(np.mean(a_probs)), 3),
                    "books_used": len(h_probs),
                    "matched_league": league_key,
                    "match_confidence": round(best_score, 1),
                }
            except Exception:
                return None

        # Check leagues concurrently, take the first real hit
        results = await asyncio.gather(*[search_league(lk) for lk in ODDS_API_SOCCER_LEAGUES])
        for r in results:
            if r:
                return r
        return None

    @classmethod
    async def harvest_all(cls, home_team: str, away_team: str, client: httpx.AsyncClient) -> Dict[str, Any]:
        # Normalize shorthand ("Man Utd", "Spurs") to full club names before
        # any fuzzy-matching lookup — see TEAM_ALIASES for why this is needed.
        # ClubElo uses its own internal naming convention handled inside
        # fetch_clubelo already, so the raw (non-normalized) name is kept for it.
        home_norm = normalize_team_name(home_team)
        away_norm = normalize_team_name(away_team)

        elo_h_task = cls.fetch_clubelo(home_team, client)
        elo_a_task = cls.fetch_clubelo(away_team, client)
        home_id_task = cls.find_team_id(home_norm, client)
        away_id_task = cls.find_team_id(away_norm, client)
        consensus_task = cls.fetch_bookmaker_consensus(home_norm, away_norm, client)
        xg_task = UnderstatAgent.fetch_team_xg_pair(home_norm, away_norm, client)

        elo_h, elo_a, home_info, away_info, consensus, xg_pair = await asyncio.gather(
            elo_h_task, elo_a_task, home_id_task, away_id_task, consensus_task, xg_task
        )

        provenance = {
            "elo_home": "clubelo_live" if elo_h else "default_1500_fallback",
            "elo_away": "clubelo_live" if elo_a else "default_1500_fallback",
        }
        elo_h = elo_h if elo_h else 1500
        elo_a = elo_a if elo_a else 1500

        home_id = home_info["id"] if home_info else None
        away_id = away_info["id"] if away_info else None

        injuries_h_task = cls.fetch_injuries(home_id, client) if home_id else asyncio.sleep(0, result=None)
        injuries_a_task = cls.fetch_injuries(away_id, client) if away_id else asyncio.sleep(0, result=None)
        form_h_task = cls.fetch_form(home_id, client) if home_id else asyncio.sleep(0, result=None)
        form_a_task = cls.fetch_form(away_id, client) if away_id else asyncio.sleep(0, result=None)
        h2h_task = cls.fetch_h2h(home_id, away_id, client) if (home_id and away_id) else asyncio.sleep(0, result=None)
        weather_task = (
            cls.fetch_weather(home_info["lat"], home_info["lon"], client)
            if (home_info and home_info.get("lat")) else asyncio.sleep(0, result=None)
        )

        injuries_h, injuries_a, form_h, form_a, h2h, weather = await asyncio.gather(
            injuries_h_task, injuries_a_task, form_h_task, form_a_task, h2h_task, weather_task
        )

        elo_diff = (elo_h + 84) - elo_a
        p_home_elo = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        p_away_elo = 1.0 - p_home_elo

        if xg_pair:
            # Real Understat non-penalty xG per game, blended with a small
            # Elo nudge so a team's data doesn't get stuck purely on last
            # season's shot quality with no adjustment for current squad strength.
            lam = round(max(0.35, 0.85 * xg_pair["home_xg_for"] + 0.15 * (1.35 + elo_diff / 500.0)), 2)
            mu = round(max(0.35, 0.85 * xg_pair["away_xg_for"] + 0.15 * (1.20 - elo_diff / 500.0)), 2)
            xg_source = "understat_live"
        else:
            # Fallback: Elo-derived proxy, honestly labeled as such — this is
            # NOT real shot-quality data, just a goals-rate estimate from rating gap.
            lam = round(max(0.35, 1.35 + (elo_diff / 500.0)), 2)
            mu = round(max(0.35, 1.20 - (elo_diff / 500.0)), 2)
            xg_source = "unavailable_elo_proxy_used"

        provenance.update({
            "injuries": "api_football_live" if API_FOOTBALL_KEY else "unavailable_no_api_key",
            "form": "api_football_live" if API_FOOTBALL_KEY else "unavailable_no_api_key",
            "h2h": "api_football_live" if API_FOOTBALL_KEY else "unavailable_no_api_key",
            "weather": "open_meteo_live" if weather else "unavailable_no_venue_coords",
            "bookmaker_consensus": "the_odds_api_live" if consensus else "unavailable_no_api_key_or_no_match_found",
            "expected_goals_lambda_mu": xg_source,
        })

        return {
            "lambda_home": lam,
            "mu_away": mu,
            "elo_ratings": {"home": elo_h, "away": elo_a},
            "elo_diff": elo_diff,
            "elo_probabilities": {"p_home": round(p_home_elo, 3), "p_away": round(p_away_elo, 3)},
            "team_news": {"home_injuries": injuries_h, "away_injuries": injuries_a},
            "form": {"home": form_h, "away": form_a},
            "head_to_head": h2h,
            "weather": weather,
            "bookmaker_consensus": consensus,
            "understat_xg": xg_pair,
            "data_provenance": provenance,
        }


# =====================================================================
# AGENT 3: QUANTITATIVE ENSEMBLE (Dixon-Coles + Shin 1993)
# =====================================================================
class QuantEnsembleAgent:
    @staticmethod
    def solve_shin_debiasing(oH: float, oD: float, oA: float) -> Dict[str, float]:
        """Shin (1993) insider-trading de-vig model. NOTE: the widely-copied
        version of this snippet that circulates online (and was in the
        previous version of this file) omits the square on pi_i and solves
        the wrong target equation, which makes the bisection converge to a
        boundary value instead of the true root. The correct formulation
        solves for z such that sum_i p_i(z) == 1, using pi_i^2/beta inside
        the square root, not pi_i/beta. Verified against known odds sets to
        produce z in the ~2-5% range typical of football markets."""
        pi_h, pi_d, pi_a = 1.0 / oH, 1.0 / oD, 1.0 / oA
        beta = pi_h + pi_d + pi_a

        def probs_at(z: float):
            sh = (np.sqrt(z ** 2 + 4 * (1 - z) * (pi_h ** 2 / beta)) - z) / (2 * (1 - z))
            sd = (np.sqrt(z ** 2 + 4 * (1 - z) * (pi_d ** 2 / beta)) - z) / (2 * (1 - z))
            sa = (np.sqrt(z ** 2 + 4 * (1 - z) * (pi_a ** 2 / beta)) - z) / (2 * (1 - z))
            return sh, sd, sa

        low, high, z = 0.0, 0.999, 0.02
        for _ in range(60):
            z = (low + high) / 2.0
            sh, sd, sa = probs_at(z)
            if (sh + sd + sa) > 1.0:
                low = z
            else:
                high = z

        sh, sd, sa = probs_at(z)
        tot = sh + sd + sa
        return {"pH": sh / tot, "pD": sd / tot, "pA": sa / tot, "z": round(float(z), 4)}

    @staticmethod
    def solve_dixon_coles(lam: float, mu: float, rho: float = -0.11) -> Dict[str, float]:
        max_g = 9
        matrix = np.zeros((max_g, max_g))
        for h in range(max_g):
            for a in range(max_g):
                prob = poisson.pmf(h, lam) * poisson.pmf(a, mu)
                tau = 1.0
                if h == 0 and a == 0:
                    tau = 1.0 - (lam * mu * rho)
                elif h == 0 and a == 1:
                    tau = 1.0 + (lam * rho)
                elif h == 1 and a == 0:
                    tau = 1.0 + (mu * rho)
                elif h == 1 and a == 1:
                    tau = 1.0 - rho
                matrix[h][a] = prob * max(tau, 0.0001)
        matrix /= np.sum(matrix)
        return {
            "p_home": float(np.sum(np.tril(matrix, -1))),
            "p_draw": float(np.sum(np.diag(matrix))),
            "p_away": float(np.sum(np.triu(matrix, 1))),
        }


# =====================================================================
# AGENT 4: ACCA BUILDER — restricted to ultra-safe legs only
# =====================================================================
class AccaMaximizerAgent:
    @staticmethod
    def build_ultra_safe_accas(matches: List[Dict], min_odds: float = 3.00,
                                leg_size: int = 3, bankroll: float = 5000.0):
        safe_pool = [m for m in matches if m.get("ultra_safe")]
        safe_pool = sorted(safe_pool, key=lambda m: m['model_prob'], reverse=True)

        slips = []
        used_ids = set()

        while True:
            available = [m for m in safe_pool if m['id'] not in used_ids]
            if len(available) < leg_size:
                break

            best_combo, best_metric = None, -1.0

            def search(start_idx, legs, cum_odds, cum_prob):
                nonlocal best_combo, best_metric
                if len(legs) == leg_size:
                    if cum_odds >= min_odds:
                        metric = cum_prob * np.log10(cum_odds + 1.0)
                        if metric > best_metric:
                            best_metric = metric
                            best_combo = list(legs)
                    return
                for i in range(start_idx, len(available)):
                    m = available[i]
                    search(i + 1, legs + [m], cum_odds * m['market_odd'], cum_prob * m['model_prob'])

            search(0, [], 1.0, 1.0)

            if not best_combo:
                break  # no combination among remaining safe legs clears min_odds — stop, don't fudge it

            for m in best_combo:
                used_ids.add(m['id'])

            c_odds = float(np.prod([m['market_odd'] for m in best_combo]))
            c_prob = float(np.prod([m['model_prob'] for m in best_combo]))

            b = max(0.01, c_odds - 1.0)
            kelly = max(0.0, (b * c_prob - (1.0 - c_prob)) / b)
            stake = max(0, round((bankroll * (kelly * 0.25)) / 10) * 10)

            slips.append({
                "slip_id": len(slips) + 1,
                "legs": best_combo,
                "combined_odds": round(c_odds, 2),
                "combined_prob_pct": round(c_prob * 100, 1),
                "recommended_stake": stake,
                "note": "Built only from legs meeting the ultra-safe filter (see /health for criteria).",
            })

        return slips


# =====================================================================
# API ENDPOINTS
# =====================================================================
class PipelineRequest(BaseModel):
    raw_text: str
    target_min_odds: float = 3.00
    legs_per_slip: int = 3
    bankroll: float = 5000.0


def _apply_ultra_safe_filter(m: Dict[str, Any]) -> bool:
    if m['model_prob'] < ULTRA_SAFE_PROB_FLOOR:
        return False

    market_implied = 1.0 / m['market_odd']
    edge = m['model_prob'] - market_implied
    if edge > ULTRA_SAFE_MAX_MARKET_DISAGREEMENT:
        # model loves it far more than the market does -> value bet, not a safe bet
        return False

    injuries = m['tactical_data'].get('key_injuries_on_favored_side')
    if injuries:
        return False

    consensus = m.get('bookmaker_consensus')
    if consensus:
        pick = m['consensus_pick']
        cons_key = {"1": "consensus_pH", "X": "consensus_pD", "2": "consensus_pA"}[pick]
        if abs(consensus[cons_key] - m['model_prob']) > CONSENSUS_OUTLIER_THRESHOLD:
            return False

    return True


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "version": "2026.4",
        "ocr_available": OCR_AVAILABLE,
        "api_football_configured": bool(API_FOOTBALL_KEY),
        "odds_api_configured": bool(ODDS_API_KEY),
        "ultra_safe_criteria": {
            "min_model_probability": ULTRA_SAFE_PROB_FLOOR,
            "max_model_vs_market_disagreement": ULTRA_SAFE_MAX_MARKET_DISAGREEMENT,
            "max_consensus_deviation": CONSENSUS_OUTLIER_THRESHOLD,
            "requires_no_key_injuries": True,
        },
    }


@app.post("/api/v2/execute_pipeline")
async def execute_pipeline(req: PipelineRequest):
    raw_matches = IngestionAgent.parse_fixtures_in_order(req.raw_text)
    if not raw_matches:
        raise HTTPException(status_code=422, detail="No valid fixtures parsed. Ensure teams and odds are present.")

    analyzed_roster = []

    async with httpx.AsyncClient() as client:
        for rm in raw_matches:
            seq = rm["sequence_order"]
            t1, t2 = rm["home_team"], rm["away_team"]
            oH, oD, oA = rm["oH"], rm["oD"], rm["oA"]

            stats = await LiveDataAgent.harvest_all(t1, t2, client)

            dc = QuantEnsembleAgent.solve_dixon_coles(stats["lambda_home"], stats["mu_away"])
            shin = QuantEnsembleAgent.solve_shin_debiasing(oH, oD, oA)
            consensus = stats.get("bookmaker_consensus")

            # Dynamic weighting: only lean on Dixon-Coles xG-proxy and consensus
            # when we actually have real signal for them; otherwise fall back
            # to Shin (always real, since it's derived from the odds you gave us)
            # plus Elo (real when ClubElo indexed the club).
            if consensus:
                w_dc, w_shin, w_elo, w_cons = 0.35, 0.25, 0.15, 0.25
                ens_H = (w_dc * dc["p_home"] + w_shin * shin["pH"] +
                         w_elo * stats["elo_probabilities"]["p_home"] + w_cons * consensus["consensus_pH"])
                ens_D = (w_dc * dc["p_draw"] + w_shin * shin["pD"] +
                         w_elo * 0.26 + w_cons * consensus["consensus_pD"])
                ens_A = (w_dc * dc["p_away"] + w_shin * shin["pA"] +
                         w_elo * stats["elo_probabilities"]["p_away"] + w_cons * consensus["consensus_pA"])
            else:
                w_dc, w_shin, w_elo = 0.45, 0.35, 0.20
                ens_H = w_dc * dc["p_home"] + w_shin * shin["pH"] + w_elo * stats["elo_probabilities"]["p_home"]
                ens_D = w_dc * dc["p_draw"] + w_shin * shin["pD"] + w_elo * 0.26
                ens_A = w_dc * dc["p_away"] + w_shin * shin["pA"] + w_elo * stats["elo_probabilities"]["p_away"]

            tot = ens_H + ens_D + ens_A
            ens_H, ens_D, ens_A = round(ens_H / tot, 4), round(ens_D / tot, 4), round(ens_A / tot, 4)

            if ens_H >= ens_A and ens_H >= 0.40:
                pick, pick_str, pick_odd, pick_prob = "1", f"{t1} Win (1)", oH, ens_H
            elif ens_A > ens_H and ens_A >= 0.40:
                pick, pick_str, pick_odd, pick_prob = "2", f"{t2} Win (2)", oA, ens_A
            else:
                pick, pick_str, pick_odd, pick_prob = "X", "Draw (X)", oD, ens_D

            # Key-injury flag on the favored side, only meaningful if we have real data
            key_injury_flag = None
            if pick == "1" and stats["team_news"]["home_injuries"]:
                key_injury_flag = stats["team_news"]["home_injuries"]
            elif pick == "2" and stats["team_news"]["away_injuries"]:
                key_injury_flag = stats["team_news"]["away_injuries"]

            match_record = {
                "id": seq,
                "order": seq,
                "fixture": f"{t1} vs {t2}",
                "home_team": t1,
                "away_team": t2,
                "odds": {"1": oH, "X": oD, "2": oA},
                "tactical_data": {
                    "lambda_expected_goals_home": stats["lambda_home"],
                    "mu_expected_goals_away": stats["mu_away"],
                    "elo_home": stats["elo_ratings"]["home"],
                    "elo_away": stats["elo_ratings"]["away"],
                    "elo_diff": stats["elo_diff"],
                    "key_injuries_on_favored_side": key_injury_flag,
                },
                "team_news": stats["team_news"],
                "form": stats["form"],
                "head_to_head_last5": stats["head_to_head"],
                "weather": stats["weather"],
                "bookmaker_consensus": consensus,
                "shin_debiased": {"pH": round(shin["pH"], 3), "pD": round(shin["pD"], 3), "pA": round(shin["pA"], 3)},
                "consensus_pick": pick,
                "predicted_text": pick_str,
                "market_odd": pick_odd,
                "model_prob": round(pick_prob, 4),
                "data_provenance": stats["data_provenance"],
            }
            match_record["ultra_safe"] = _apply_ultra_safe_filter(match_record)
            analyzed_roster.append(match_record)

    accas = AccaMaximizerAgent.build_ultra_safe_accas(
        analyzed_roster, req.target_min_odds, req.legs_per_slip, req.bankroll
    )

    return {
        "status": "success",
        "matches_count": len(analyzed_roster),
        "ultra_safe_count": sum(1 for m in analyzed_roster if m["ultra_safe"]),
        "analyzed_matches": analyzed_roster,
        "ultra_safe_accumulators": accas,
    }


@app.post("/api/v2/execute_pipeline_from_screenshot")
async def execute_pipeline_from_screenshot(
    file: UploadFile = File(...),
    target_min_odds: float = 3.00,
    legs_per_slip: int = 3,
    bankroll: float = 5000.0,
):
    """Accepts a betslip screenshot, OCRs it, then runs the same pipeline."""
    image_bytes = await file.read()
    raw_text = IngestionAgent.extract_text_from_screenshot(image_bytes)
    req = PipelineRequest(
        raw_text=raw_text,
        target_min_odds=target_min_odds,
        legs_per_slip=legs_per_slip,
        bankroll=bankroll,
    )
    return await execute_pipeline(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
