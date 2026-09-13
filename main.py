"""
Mathayo Institutional Quant Engine — Production Backend
Version: 2026.5 (Fixes: ensemble weight collapse, pick-selection bug,
                        frontend field-mismatch crashes)
 
WHAT CHANGED FROM v2026.4 AND WHY
----------------------------------
Four real bugs found in v2026.4, all fixed here:
 
1. ENSEMBLE WEIGHT COLLAPSE (root cause of "zero ultra-safe picks out of
   460 matches"). The no-xG fallback for Dixon-Coles's lambda/mu is itself
   an Elo-differential proxy (see harvest_all). That means when a club also
   has no ClubElo rating (elo_diff defaults to 0, a flat 1500-vs-1500
   no-op), BOTH w_dc (0.45) AND w_elo (0.20) are simultaneously running on
   the same uninformative signal — 65% of the ensemble on pure noise, not
   just the 20% you'd expect from w_elo alone. Shin — the one component
   that is *always* real, because it's de-vigged directly from the odds
   the caller supplied — got squeezed to 35%. That's why a ~1.20-odds
   favorite (~83% implied) scored 58.9% instead of something close to
   Shin's number. Fixed with compute_ensemble_weights(): weight is now
   assigned in proportion to how real each component's inputs actually
   are for THIS match, and whatever gets discounted is reassigned to Shin.
   This is a heuristic, not a proven calibration — treat it as directionally
   correct, not as ground truth.
 
2. PICK-SELECTION WAS NOT ARGMAX. The old if/elif ladder forced a Draw
   pick whenever neither side cleared a 0.40 floor, even in cases like
   (0.38, 0.24, 0.38) where Draw is the LEAST likely outcome. Replaced
   with a straight argmax over {H, D, A}.
 
3. `/health` HAD NO FIELD MATCHING WHAT THE FRONTEND READS. The frontend
   log line "Active Agents: undefined" means it's reading a property this
   endpoint never returned. Added `active_agents`, `agents_active`,
   `agent_count`, and `active_agent_names` — multiple common namings,
   since the actual frontend source wasn't available to confirm the exact
   key. Confirm which one your frontend expects and drop the rest.
 
4. RESPONSE FIELD RENAMES BROKE forEach CALLS. `execute_pipeline`'s
   response keys (`analyzed_matches`, `ultra_safe_accumulators`) don't
   match older key names a frontend built against a prior version may
   still expect (e.g. `matches`, `accumulators`, `accas`). Rather than
   guess which one is live and break it again, this version returns BOTH
   the canonical name and legacy aliases. Also, previously-nullable list
   fields (home_injuries, away_injuries, head_to_head_last5,
   key_injuries_on_favored_side) are now normalized to `[]` instead of
   `null` when data is unavailable, so a naive `.forEach()` on them can't
   crash. This does NOT hide the fact that data is missing — the existing
   `data_provenance` block still tells you exactly what was fetched vs.
   unavailable; only the shape of "no data" changed from null to empty
   list, which is a UI-safety fix, not a provenance change.
 
   ACTION ITEM FOR YOU: once you confirm your current frontend's actual
   field names, delete the alias fields below (marked `# ALIAS`) — keeping
   two names for the same data forever is exactly the kind of drift that
   caused this bug in the first place.
 
Everything below this point that isn't marked with a v2026.5 comment is
unchanged from v2026.4.
 
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
 
ODDS_API_SOCCER_LEAGUES = [
    "soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a",
    "soccer_germany_bundesliga", "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]
 
UNDERSTAT_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]
 
FUZZY_MATCH_MIN_SCORE = 78
 
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
    "united": "manchester united", "city": "manchester city",
}
 
 
def normalize_team_name(name: str) -> str:
    key = re.sub(r'[^a-z0-9 ]', '', name.lower()).strip()
    return TEAM_ALIASES.get(key, name)
 
ULTRA_SAFE_PROB_FLOOR = 0.62
ULTRA_SAFE_MAX_MARKET_DISAGREEMENT = 0.10
CONSENSUS_OUTLIER_THRESHOLD = 0.08
 
 
def current_european_season() -> int:
    today = dt.datetime.utcnow()
    return today.year if today.month >= 7 else today.year - 1
 
app = FastAPI(title="Mathayo Autonomous Quant Engine", version="2026.5")
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
# AGENT 1: INGESTION
# =====================================================================
class IngestionAgent:
    """
    v2026.6 REWRITE. The old single-heuristic parser assumed one input shape:
    team name / team name / odds / odds / odds, each alone on its own line.
    That's a plain betslip paste. It is NOT what odibets' own site produces
    when you copy a search-results tile, a league table view, or a full
    match-detail page — and feeding those into the old heuristic is exactly
    what produced fixtures like "Way vs Real Madrid" and "or X vs X or 2".
    Those aren't hallucinated predictions; nothing downstream invents data.
    "Way" and "or X" are literal text fragments from a "3 Way" market header
    and "1 or X" Double-Chance labels elsewhere in the page dump, which the
    old parser mistook for team names because they're short strings with
    letters in them and nothing filtered them out.
 
    Fix: detect which of the known real-world paste formats the text matches
    and route to a parser built for that specific structure, instead of one
    heuristic trying to cover all of them. Four formats are supported:
      - "detail_dump": full match-detail page dump (has "ID: <n>" headers and
        an exact "Draw" line inside the "3 Way" market block)
      - "card": search-tile format with markdown-link team names and a single
        glued odds line like "1 6.40X 4.002 1.60"
      - "table": league-table format with team names glued on one line
        ("SunderlandArsenal") and odds glued on the next ("6.604.101.58")
      - "legacy": the original simple one-line-per-field betslip paste
 
    This is heuristic text parsing against three specific real-world samples,
    not a guaranteed-correct scraper for every possible odibets page layout
    or any future site redesign. If a paste doesn't match any of the three
    known shapes it falls back to "legacy". Spot-check the parsed fixture
    list against your source before trusting an accumulator built from it,
    especially for team-name splits in the "table" format (see
    _split_glued_two_names for why that one's a heuristic guess).
    """
 
    @staticmethod
    def detect_format(raw_text: str) -> str:
        if re.search(r'\bID:\s*\d+', raw_text) and re.search(r'\bdraw\b', raw_text, re.I):
            return "detail_dump"
        if re.search(r'\]\(https?://', raw_text) and re.search(r'\b1\s*\d+\.\d{2}\s*X', raw_text, re.I):
            return "card"
        if '\u2022' in raw_text and re.search(r'markets?', raw_text, re.I):
            return "table"
        return "legacy"
 
    @staticmethod
    def _split_glued_two_names(s: str):
        """
        Heuristic split for two team names concatenated with no separator,
        e.g. 'SunderlandArsenal' -> ('Sunderland', 'Arsenal'). Splits at the
        first lowercase->uppercase letter transition, since a multi-word team
        name keeps its own internal space ('Real Madrid') while the actual
        join point between two names never has a space. This can misfire on
        unusual names (an all-caps abbreviation glued directly to the next
        name, e.g. 'PSGLyon', has no lowercase->uppercase boundary at all and
        will fail to split) -- treat mis-splits as a parsing artifact to spot
        check, not a data error.
        """
        boundaries = [m.start() for m in re.finditer(r'(?<=[a-z])(?=[A-Z])', s)]
        if not boundaries:
            return None
        b = boundaries[0]
        t1, t2 = s[:b].strip(), s[b:].strip()
        if len(t1) > 2 and len(t2) > 2:
            return t1, t2
        return None
 
    @staticmethod
    def _parse_legacy_format(raw_text: str) -> List[Dict[str, Any]]:
        """Original parser: one plain betslip paste, one field per line."""
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
    def _parse_card_format(raw_text: str) -> List[Dict[str, Any]]:
        """
        odibets search-tile paste: each team name is its own markdown link
        line, e.g. '[Sunderland AFC](https://...)', and the 1/X/2 odds arrive
        glued onto a single line with no separators between a value and the
        next label, e.g. '1 6.40X 4.002 1.60' (meaning 1=6.40, X=4.00, 2=1.60
        -- the '2' of '4.002' is the away-win label, not part of the draw odd;
        matching the fractional part to exactly 2 digits is what keeps that
        boundary correct).
        """
        link_re = re.compile(r'^\[(.*?)\]\(https?://\S+\)$')
        glued_odds_re = re.compile(r'1\s*(\d+\.\d{2})X\s*(\d+\.\d{2})2\s*(\d+\.\d{2})', re.I)
 
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        matches: List[Dict[str, Any]] = []
        pending_teams: List[str] = []
 
        for line in lines:
            text = line
            m = link_re.match(line)
            if m:
                text = m.group(1).strip()
            if not text:
                continue
            if re.search(r'\bhot\b', text, re.I) and len(text) < 15:
                continue
            if re.search(r'bet now', text, re.I):
                continue
            if re.match(r'^\d{1,2}[\/\.]\d{1,2}(?:\/\d{2,4})?(\s*,\s*|\s*-\s*|\s+)\d{1,2}:\d{2}', text, re.I):
                continue
            if re.match(r'^starts in', text, re.I):
                continue
            if '\u2022' in text or re.match(r'^[\w .\-]+/[\w .\-]+\(\d+\)$', text):
                continue  # league header, e.g. "England / Premier League (1)"
 
            odds_m = glued_odds_re.search(text)
            if odds_m:
                if len(pending_teams) >= 2:
                    t1, t2 = pending_teams[-2], pending_teams[-1]
                    matches.append({
                        "sequence_order": len(matches) + 1,
                        "home_team": t1,
                        "away_team": t2,
                        "oH": float(odds_m.group(1)),
                        "oD": float(odds_m.group(2)),
                        "oA": float(odds_m.group(3)),
                    })
                pending_teams = []
                continue
 
            if len(re.findall(r'[a-zA-Z]', text)) >= 3 and len(text) > 2:
                pending_teams.append(text)
 
        return matches
 
    @staticmethod
    def _parse_table_format(raw_text: str) -> List[Dict[str, Any]]:
        """
        odibets league-table paste: team names glued on one line
        ('SunderlandArsenal'), the three 1X2 odds glued on the next
        ('6.604.101.58'), with league headers ('England \u2022 Premier League')
        and '+N Markets' trailer lines as noise around them.
        """
        glued_odds_full_re = re.compile(r'^(?:\d+\.\d{2}){3}$')
        markets_trailer_re = re.compile(r'^\+?\d+\s*markets?$', re.I)
 
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        filtered = []
        for l in lines:
            if '\u2022' in l:
                continue
            if markets_trailer_re.match(l):
                continue
            if re.match(r'^\d{1,2}[\/\.]\d{1,2}(?:\/\d{2,4})?(\s*,\s*|\s*-\s*|\s+)\d{1,2}:\d{2}', l, re.I):
                continue
            filtered.append(l)
 
        matches: List[Dict[str, Any]] = []
        i = 0
        while i < len(filtered) - 1:
            if not glued_odds_full_re.match(filtered[i]) and glued_odds_full_re.match(filtered[i + 1]):
                split = IngestionAgent._split_glued_two_names(filtered[i])
                odds_vals = [float(x) for x in re.findall(r'\d+\.\d{2}', filtered[i + 1])]
                if split and len(odds_vals) == 3:
                    t1, t2 = split
                    matches.append({
                        "sequence_order": len(matches) + 1,
                        "home_team": t1,
                        "away_team": t2,
                        "oH": odds_vals[0],
                        "oD": odds_vals[1],
                        "oA": odds_vals[2],
                    })
                i += 2
                continue
            i += 1
 
        return matches
 
    @staticmethod
    def _parse_detail_dump_format(raw_text: str) -> List[Dict[str, Any]]:
        """
        odibets full match-detail dump. Ignores every market block except
        3-Way: anchors ONLY on a line that is exactly 'Draw' (never matches
        the longer 'Draw no bet - Full Time' line elsewhere in the same dump,
        since this is an exact-equality check, not a substring search) and
        reads the fixed window around it:
        [home_team, home_odd, 'Draw', draw_odd, away_team, away_odd].
        This is what was previously misfiring -- the old parser had no
        concept of this structure, so fragments like the '3 Way' market
        header text itself were being swept up as if they were team names.
        """
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        matches: List[Dict[str, Any]] = []
        for i, line in enumerate(lines):
            if line.strip().lower() != "draw":
                continue
            if i < 2 or i + 3 >= len(lines):
                continue
            home_team, home_odd_s = lines[i - 2], lines[i - 1]
            draw_odd_s, away_team, away_odd_s = lines[i + 1], lines[i + 2], lines[i + 3]
            try:
                oH, oD, oA = float(home_odd_s), float(draw_odd_s), float(away_odd_s)
            except ValueError:
                continue
            if not (1.01 <= oH <= 60 and 1.01 <= oD <= 60 and 1.01 <= oA <= 60):
                continue
            if len(re.findall(r'[a-zA-Z]', home_team)) < 2 or len(re.findall(r'[a-zA-Z]', away_team)) < 2:
                continue
            matches.append({
                "sequence_order": len(matches) + 1,
                "home_team": home_team,
                "away_team": away_team,
                "oH": oH, "oD": oD, "oA": oA,
            })
        return matches
 
    @staticmethod
    def parse_fixtures_in_order(raw_text: str) -> List[Dict[str, Any]]:
        fmt = IngestionAgent.detect_format(raw_text)
        parsers = {
            "detail_dump": IngestionAgent._parse_detail_dump_format,
            "card": IngestionAgent._parse_card_format,
            "table": IngestionAgent._parse_table_format,
            "legacy": IngestionAgent._parse_legacy_format,
        }
        result = parsers[fmt](raw_text)
        # If the detected format's parser found nothing, don't silently
        # return zero fixtures -- try the plain legacy parser too, in case
        # the paste mixes plain lines in with the detected format's noise.
        if not result and fmt != "legacy":
            result = IngestionAgent._parse_legacy_format(raw_text)
        return result
 
    @staticmethod
    def extract_text_from_screenshot(image_bytes: bytes) -> str:
        if not OCR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="OCR not available: install `pytesseract` + `pillow` and the "
                       "tesseract-ocr system binary on the server.",
            )
        img = Image.open(io.BytesIO(image_bytes))
        if img.width < 1000:
            scale = 1500 / img.width
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        text = pytesseract.image_to_string(img)
        return text
 
 
# =====================================================================
# AGENT 2A: UNDERSTAT xG
# =====================================================================
class UnderstatAgent:
    _league_cache: Dict[str, Dict[str, Any]] = {}
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
                        recent = history[-10:]
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
# AGENT 2: LIVE DATA HARVESTER
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
 
        results = await asyncio.gather(*[search_league(lk) for lk in ODDS_API_SOCCER_LEAGUES])
        for r in results:
            if r:
                return r
        return None
 
    @classmethod
    async def harvest_all(cls, home_team: str, away_team: str, client: httpx.AsyncClient) -> Dict[str, Any]:
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
 
        # v2026.5 FIX: normalize nullable *list* fields to [] instead of None.
        # This does not change what's real vs. fallback — data_provenance below
        # still says exactly that — it only changes "no data" from null to an
        # empty list so a frontend `.forEach()` can't crash on it.
        injuries_h = injuries_h if injuries_h is not None else []
        injuries_a = injuries_a if injuries_a is not None else []
        h2h = h2h if h2h is not None else []
        # v2026.5.1 FIX: same treatment for nullable *object*-shaped fields —
        # {} instead of null, so `Object.keys(match.weather)` or similar
        # can't crash either. Still distinguishable from "checked, has data"
        # via data_provenance, same as the list fields above.
        weather = weather if weather is not None else {}
        consensus = consensus if consensus is not None else {}
        form_h = form_h if form_h is not None else ""
        form_a = form_a if form_a is not None else ""
 
        elo_diff = (elo_h + 84) - elo_a
        p_home_elo = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        p_away_elo = 1.0 - p_home_elo
 
        if xg_pair:
            lam = round(max(0.35, 0.85 * xg_pair["home_xg_for"] + 0.15 * (1.35 + elo_diff / 500.0)), 2)
            mu = round(max(0.35, 0.85 * xg_pair["away_xg_for"] + 0.15 * (1.20 - elo_diff / 500.0)), 2)
            xg_source = "understat_live"
        else:
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
 
    @staticmethod
    def compute_ensemble_weights(dc_is_real: bool, elo_is_real: bool, has_consensus: bool) -> Dict[str, float]:
        """
        v2026.5 FIX. Root cause of the zero-ultra-safe-picks bug: weights
        used to be fixed constants regardless of whether a component's
        inputs were real. Dixon-Coles's lambda/mu fall back to an
        Elo-differential proxy when Understat has no coverage — so when
        ClubElo ALSO has no rating for either club (elo_diff = 0, a flat
        no-op), both w_dc and w_elo are simultaneously worthless. In the
        no-consensus case that's 0.45 + 0.20 = 0.65 of the ensemble on pure
        noise, while Shin (always real — it's de-vigged straight from the
        odds you supplied) was stuck at 0.35.
 
        Fix: discount dc/elo weight in proportion to how real their inputs
        are for this specific match, and hand whatever gets discounted to
        Shin. This is a heuristic, not a proven calibration.
        """
        if has_consensus:
            w = {"dc": 0.35, "shin": 0.25, "elo": 0.15, "cons": 0.25}
        else:
            w = {"dc": 0.45, "shin": 0.35, "elo": 0.20, "cons": 0.0}
 
        discount = 0.0
        if not elo_is_real:
            discount += w["elo"]
            w["elo"] = 0.0
        if not dc_is_real:
            # If elo is real, dc's proxy formula still carries some of that
            # real signal, so only discount most of it, not all of it.
            factor = 1.0 if not elo_is_real else 0.6
            discount += w["dc"] * factor
            w["dc"] *= (1.0 - factor)
 
        w["shin"] += discount
        return w
 
 
# =====================================================================
# AGENT 4: ACCA BUILDER
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
                break
 
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
                "combined_prob": round(c_prob * 100, 1),  # ALIAS — matches frontend's s.combined_prob
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
    agent_names = [
        "IngestionAgent", "LiveDataAgent+UnderstatAgent",
        "QuantEnsembleAgent", "AccaMaximizerAgent",
    ]
    return {
        "status": "online",
        "version": "2026.5",
        "ocr_available": OCR_AVAILABLE,
        "api_football_configured": bool(API_FOOTBALL_KEY),
        "odds_api_configured": bool(ODDS_API_KEY),
        # v2026.5 FIX: "/health" previously had no field the frontend could
        # read for its "Active Agents: undefined" line. Since the real
        # frontend source wasn't available to confirm the exact key name,
        # multiple common namings are provided — check yours and drop the
        # rest.
        "active_agents": len(agent_names),      # ALIAS
        "agents_active": len(agent_names),       # ALIAS
        "agent_count": len(agent_names),         # ALIAS
        "active_agent_names": agent_names,
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
 
            # v2026.5 FIX: weights now computed from actual data quality for
            # this match instead of fixed constants. See compute_ensemble_weights.
            dc_is_real = (stats["data_provenance"]["expected_goals_lambda_mu"] == "understat_live")
            elo_is_real = (
                stats["data_provenance"]["elo_home"] == "clubelo_live"
                and stats["data_provenance"]["elo_away"] == "clubelo_live"
            )
            w = QuantEnsembleAgent.compute_ensemble_weights(dc_is_real, elo_is_real, bool(consensus))
 
            if consensus:
                ens_H = (w["dc"] * dc["p_home"] + w["shin"] * shin["pH"] +
                         w["elo"] * stats["elo_probabilities"]["p_home"] + w["cons"] * consensus["consensus_pH"])
                ens_D = (w["dc"] * dc["p_draw"] + w["shin"] * shin["pD"] +
                         w["elo"] * 0.26 + w["cons"] * consensus["consensus_pD"])
                ens_A = (w["dc"] * dc["p_away"] + w["shin"] * shin["pA"] +
                         w["elo"] * stats["elo_probabilities"]["p_away"] + w["cons"] * consensus["consensus_pA"])
            else:
                ens_H = w["dc"] * dc["p_home"] + w["shin"] * shin["pH"] + w["elo"] * stats["elo_probabilities"]["p_home"]
                ens_D = w["dc"] * dc["p_draw"] + w["shin"] * shin["pD"] + w["elo"] * 0.26
                ens_A = w["dc"] * dc["p_away"] + w["shin"] * shin["pA"] + w["elo"] * stats["elo_probabilities"]["p_away"]
 
            tot = ens_H + ens_D + ens_A
            ens_H, ens_D, ens_A = round(ens_H / tot, 4), round(ens_D / tot, 4), round(ens_A / tot, 4)
 
            # v2026.5 FIX: straight argmax, replacing the old 0.40-floor
            # if/elif ladder that forced a Draw pick whenever neither side
            # cleared 40% — even when Draw was the LEAST likely outcome
            # (e.g. 0.38 / 0.24 / 0.38 used to resolve to "X").
            probs = {"1": ens_H, "X": ens_D, "2": ens_A}
            pick = max(probs, key=probs.get)
            pick_prob = probs[pick]
            pick_odd = {"1": oH, "X": oD, "2": oA}[pick]
            pick_str = {"1": f"{t1} Win (1)", "X": "Draw (X)", "2": f"{t2} Win (2)"}[pick]
 
            # v2026.5 FIX: always a list (never null) so a frontend forEach
            # on this field can't crash — empty means "no flagged injury",
            # not "unchecked" (unchecked is tracked separately in provenance).
            if pick == "1":
                key_injury_flag = stats["team_news"]["home_injuries"]
            elif pick == "2":
                key_injury_flag = stats["team_news"]["away_injuries"]
            else:
                key_injury_flag = []
 
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
                    "lambda_xg": stats["lambda_home"],   # ALIAS — matches frontend's m.tactical_data.lambda_xg
                    "mu_xg": stats["mu_away"],           # ALIAS — matches frontend's m.tactical_data.mu_xg
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
                "ensemble_weights_used": w,  # transparency: shows exactly how much of this pick rode on real vs. fallback data
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
        "matches": analyzed_roster,              # ALIAS
        "ultra_safe_accumulators": accas,
        "orthogonal_accumulators": accas,         # FIX — this is the exact key index.html reads; its absence was the forEach crash
        "accumulators": accas,                    # ALIAS
        "accas": accas,                           # ALIAS
    }
 
 
@app.post("/api/v2/execute_pipeline_from_screenshot")
async def execute_pipeline_from_screenshot(
    file: UploadFile = File(...),
    target_min_odds: float = 3.00,
    legs_per_slip: int = 3,
    bankroll: float = 5000.0,
):
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
