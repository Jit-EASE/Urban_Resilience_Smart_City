# Robust cache/failover bootstrapping (inserted by patch)
import os as _os_boot
import logging as _logging_boot
# Silence noisy cache runtime notices
_logging_boot.getLogger("streamlit.runtime.caching").setLevel(_logging_boot.ERROR)
# Optional: disable Streamlit caching at runtime for debugging freezes
if _os_boot.getenv("DISABLE_CACHE", "0") == "1":
    import streamlit as _st_boot
    _st_boot.cache_data = lambda **_: (lambda f: f)
    _st_boot.cache_resource = lambda **_: (lambda f: f)

# UNIC Urban Resilience & Smart Cities — Ireland (v4.0)
# Streamlit + Leaflet (Folium) — Dynamic, fast-loading, Agentic RAG, Knowledge Graph, MILP, Contextual RL
# -------------------------------------------------------------------------------------
# Core rewrite goals:
# • No brittle hardcoding: cities, centroids, layers, and popups are resolved dynamically (with robust OFFLINE fallback)
# • Fast first paint: map renders immediately; heavy/remote tasks are strictly opt‑in and cached
# • Adaptive NLP explanations: all popups/tooltips are generated from live context (signals + POI metadata), not fixed text
# • Geospatial-first: centroid → nodes; optional legs/whiskers; inland snapping to avoid ocean drift (e.g., Galway Bay)
# • Clean extensibility: add new POI definitions via CONFIG_POIS without touching core logic

# --- BEGIN v4.0 REWRITE ---
# (See instructions for full code)
# The full code is as provided in the user message.
# Please refer to the user's message for the replacement code.

# UNIC Urban Resilience & Smart Cities — Ireland (v4.0)
# Streamlit + Leaflet (Folium) — Dynamic, fast-loading, Agentic RAG, Knowledge Graph, MILP, Contextual RL
# -------------------------------------------------------------------------------------
# Core rewrite goals:
# • No brittle hardcoding: cities, centroids, layers, and popups are resolved dynamically (with robust OFFLINE fallback)
# • Fast first paint: map renders immediately; heavy/remote tasks are strictly opt‑in and cached
# • Adaptive NLP explanations: all popups/tooltips are generated from live context (signals + POI metadata), not fixed text
# • Geospatial-first: centroid → nodes; optional legs/whiskers; inland snapping to avoid ocean drift (e.g., Galway Bay)
# • Clean extensibility: add new POI definitions via CONFIG_POIS without touching core logic

import os
import json
import glob
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw, MarkerCluster

# Optional deps (lazy)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import pulp
except Exception:
    pulp = None

import requests

# --- Robust Overpass handling (multi-endpoint + failover to OFFLINE) ---
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",  # fast community mirror
    "https://overpass-api.de/api/interpreter",        # main DE instance
]

# session-level OFFLINE failover flag
if "_offline_failover" not in st.session_state:
    st.session_state["_offline_failover"] = False

def use_offline() -> bool:
    """Return True if DATA_MODE is OFFLINE or failover was triggered."""
    return CONFIG.get("DATA_MODE", "LIVE").upper() == "OFFLINE" or bool(st.session_state.get("_offline_failover", False))

# ---------------- CONFIG ----------------
APP_TITLE = "UNIC Regional Resilience (Ireland) — v4 (Dynamic)"
st.set_page_config(page_title=APP_TITLE, layout="wide")

CONFIG = {
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    "openai_key": os.getenv("OPENAI_API_KEY", ""),
    "data_dir": "data",
    "policy_dir": os.path.join("data", "policies"),
    "log_dir": os.path.join("data", "logs"),
    "DATA_MODE": os.getenv("DATA_MODE", "LIVE"),  # LIVE | OFFLINE
    "MET_RAIN_TILES": os.getenv("MET_RAIN_TILES", ""),
    # Define POI layers dynamically via Overpass tag filters (extend here, no code changes needed)
    "CONFIG_POIS": {
        "Hospitals": {"query": "amenity=hospital", "icon": {"kind": "marker", "icon": "plus", "color": "red"}},
        "Shelters": {"query": "amenity=shelter", "icon": {"kind": "marker", "icon": None, "color": "green"}},
        "Substations": {"query": "power=substation", "icon": {"kind": "circle", "radius": 6}},
        # Add more here, e.g., bus stops, fire stations, schools, etc.
        # "BusStops": {"query": "highway=bus_stop", "icon": {"kind": "marker", "icon": "bus", "color": "blue"}},
    },
}

os.makedirs(CONFIG["policy_dir"], exist_ok=True)
os.makedirs(CONFIG["log_dir"], exist_ok=True)

# ---------------- Utilities ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def _bbox(lat: float, lon: float, km: float = 8.0) -> Tuple[float, float, float, float]:
    dlat = km / 110.574
    dlon = km / (111.320 * max(0.2, math.cos(math.radians(lat))))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon

@st.cache_data(ttl=3600, show_spinner=False)
def _overpass(query: str) -> Dict[str, Any]:
    """Call Overpass with retries across mirrors. On repeated failure, trigger OFFLINE failover."""
    # If already in offline mode (explicit or failover), short-circuit
    if use_offline():
        return {"elements": []}
    timeouts = (12, 12)  # connect, read
    for ep in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(ep, data={"data": query}, timeout=timeouts)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    # If we reach here, both endpoints failed — switch to failover
    st.session_state["_offline_failover"] = True
    return {"elements": []}

# Inland snap: if near water w/out nearby landuse/roads, nudge inland
@st.cache_data(ttl=3600, show_spinner=False)
def snap_inland(lat: float, lon: float) -> Tuple[float, float]:
    try:
        s, w, n, e = _bbox(lat, lon, km=0.2)
        q_water = f"""
        [out:json][timeout:20];
        ( way["natural"="coastline"]({s},{w},{n},{e}); way["water"]({s},{w},{n},{e}); ); out count;"""
        q_land = f"""
        [out:json][timeout:20];
        ( way["highway"]({s},{w},{n},{e}); way["landuse"]({s},{w},{n},{e}); ); out count;"""
        water = _overpass(q_water).get("elements", [])
        land = _overpass(q_land).get("elements", [])
        if water and not land:
            return lat + 0.003, lon + 0.007
    except Exception:
        pass
    return lat, lon

# ---------------- Dynamic place discovery ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def discover_irish_places(kind: str = "city", limit: int = 50) -> pd.DataFrame:
    """Discover Irish places dynamically (city/town) using Overpass. Falls back to a minimal list if offline."""
    if use_offline():
        return pd.DataFrame([
            {"name": "Dublin", "lat": 53.35014, "lon": -6.266155},
            {"name": "Cork", "lat": 51.903614, "lon": -8.468399},
            {"name": "Limerick", "lat": 52.668018, "lon": -8.630498},
            {"name": "Galway", "lat": 53.270962, "lon": -9.047691},
            {"name": "Waterford", "lat": 52.259319, "lon": -7.110070},
        ])
    # Overpass query for Irish cities/towns (admin boundary=IE)
    q = f"""
    [out:json][timeout:25];
    area["ISO3166-1"="IE"][admin_level=2];
    ( node["place"="{kind}"](area); node["place"="town"](area); );
    out center qt;"""
    data = _overpass(q)
    rows = []
    for el in data.get("elements", [])[:limit]:
        rows.append({"name": el.get("tags", {}).get("name"), "lat": el.get("lat"), "lon": el.get("lon")})
    df = pd.DataFrame([r for r in rows if r["name"] and r["lat"] and r["lon"]]).drop_duplicates("name")
    if df.empty:
        # Minimal fallback if API limits are hit
        df = pd.DataFrame([
            {"name": "Dublin", "lat": 53.35014, "lon": -6.266155},
            {"name": "Cork", "lat": 51.903614, "lon": -8.468399},
        ])
    return df.sort_values("name").reset_index(drop=True)

# ---------------- Dynamic POIs ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pois(layer_name: str, lat: float, lon: float, km: float = 8.0) -> List[Dict[str, Any]]:
    cfg = CONFIG["CONFIG_POIS"][layer_name]
    tag = cfg["query"]  # e.g., amenity=hospital
    if use_offline():
        # deterministic synthetic around centroid
        rng = np.random.default_rng(abs(hash((layer_name, lat, lon))) % (2**32))
        n = 8 if "Shelter" in layer_name else 6
        spread = 0.016 if "Shelter" in layer_name else 0.012
        pts = []
        for _ in range(n):
            la = float(lat + rng.normal(0, spread))
            lo = float(lon + rng.normal(0, spread))
            la, lo = snap_inland(la, lo)
            pts.append({"lat": la, "lon": lo, "name": layer_name[:-1] if layer_name.endswith('s') else layer_name})
        return pts
    s, w, n, e = _bbox(lat, lon, km=km)
    key, val = tag.split("=")
    q = f"""
    [out:json][timeout:25];
    ( node["{key}"="{val}"]({s},{w},{n},{e}); way["{key}"="{val}"]({s},{w},{n},{e}); relation["{key}"="{val}"]({s},{w},{n},{e}); );
    out center tags;"""
    data = _overpass(q)
    pois = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or layer_name[:-1] if layer_name.endswith("s") else layer_name
        if "lat" in el and "lon" in el:
            la, lo = snap_inland(el["lat"], el["lon"]) 
            pois.append({"lat": la, "lon": lo, "name": name, "tags": tags})
        elif "center" in el:
            la, lo = snap_inland(el["center"]["lat"], el["center"]["lon"]) 
            pois.append({"lat": la, "lon": lo, "name": name, "tags": tags})
    return pois

# ---------------- Signals (wire real APIs later) ----------------
@st.cache_data(ttl=900, show_spinner=False)
def get_signals(place: str) -> Dict[str, Any]:
    rng = np.random.default_rng(abs(hash(place)) % (2**32))
    return {
        "rain_mm_24h": float(rng.gamma(2, 3)),
        "aqi": int(rng.integers(12, 65)),
        "pop_est": int(rng.integers(70000, 1500000)),
        "hour": int(pd.Timestamp.now().hour),
    }

# ---------------- Adaptive NLP popups (no fixed text) ----------------
class NLPPopup:
    def __init__(self):
        self.client = OpenAI(api_key=CONFIG["openai_key"]) if (OpenAI and CONFIG["openai_key"]) else None
        self.model = CONFIG["openai_model"]

    def render(self, city: str, layer: str, poi: Dict[str, Any], signals: Dict[str, Any]) -> str:
        # Compose minimal structured context for adaptive explanation
        ctx = {
            "city": city,
            "layer": layer,
            "name": poi.get("name"),
            "tags": poi.get("tags", {}),
            "signals": signals,
        }
        if not self.client:
            # Lightweight deterministic template if no OpenAI
            rain = signals.get("rain_mm_24h", 0.0); aqi = signals.get("aqi", 0); pop = signals.get("pop_est", 0)
            return (
                f"<b>{city}</b> — {layer}: keep access resilient. "
                f"Context: rain≈{rain:.1f}mm, AQI={aqi}, pop≈{pop:,}."
            )
        try:
            sys = "You write crisp, factual, Irish smart-city map popups. One sentence (≤26 words). Emphasise service continuity, safety, equity, and low emissions."
            user = json.dumps(ctx)
            out = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=0.2,
                max_tokens=60,
            )
            text = (out.choices[0].message.content or "").strip()
        except Exception:
            text = "Maintain continuity; coordinate shelters, routes, and energy buffers during wet periods."
        return f"<b>{city}</b>: {text}"

# ---------------- Knowledge Graph (built from UI goal + layers) ----------------
class KnowledgeGraph:
    def __init__(self):
        self.edges: List[Tuple[str,str,str]] = []
        self.types: Dict[str,str] = {}
    def add(self, s, r, t, stype, ttype):
        self.edges.append((s,r,t)); self.types[s]=stype; self.types[t]=ttype
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([{"source":s,"relation":r,"target":t,"source_type":self.types.get(s,""),"target_type":self.types.get(t,"") } for s,r,t in self.edges])
    @staticmethod
    def build(goal: str, active_layers: List[str]) -> "KnowledgeGraph":
        kg = KnowledgeGraph()
        kg.add(goal, "optimises", "Service-hours preserved", "Goal", "Measure")
        kg.add(goal, "constrains", "ΔtCO₂e ≤ 0", "Goal", "Constraint")
        for L in active_layers:
            if "Hospitals" in L:
                kg.add("Service-hours preserved", "requires", "Hospital accessibility", "Measure", "KPI")
            if "Shelters" in L:
                kg.add("Equity", "improves_with", "Shelter coverage", "Objective", "KPI")
            if "Substations" in L:
                kg.add("Continuity", "depends_on", "Grid nodes uptime", "Objective", "KPI")
        return kg

# ---------------- Agentic Layer ----------------
class Agents:
    def __init__(self):
        self.client = OpenAI(api_key=CONFIG["openai_key"]) if (OpenAI and CONFIG["openai_key"]) else None
        self.model = CONFIG["openai_model"]
    def _chat(self, sys: str, user: str, T=0.2, max_tokens=900) -> str:
        if not self.client:
            return "[OpenAI key missing]"
        r = self.client.chat.completions.create(model=self.model, messages=[{"role":"system","content":sys},{"role":"user","content":user}], temperature=T, max_tokens=max_tokens)
        return r.choices[0].message.content
    def planner(self, goal: str, city: str, evidence: List[str], jury: float) -> str:
        sys = "Planner Agent for Irish urban resilience. Return JSON: objectives, constraints, actions, kpis, checklist48h, notes; budget/emissions/equity explicit."
        ctx = "\n\n".join([f"[E{i+1}] {s}" for i,s in enumerate(evidence)])
        user = f"City: {city}\nJuryConsensus: {jury:.2f}\nGoal: {goal}\nEvidence:\n{ctx}"
        return self._chat(sys, user, T=0.1)
    def verifier(self, plan_json: str, evidence: List[str]) -> str:
        sys = "Verifier Agent. JSON: policy_checks:[{name,pass,reason,citations}], risk_flags:[], required_data:[], overall: pass|fail."
        ctx = "\n\n".join([f"[E{i+1}] {s}" for i,s in enumerate(evidence)])
        return self._chat(sys, f"PlanJSON:\n{plan_json}\n\nEvidence:\n{ctx}", T=0.0)
    def equity(self, plan_json: str) -> str:
        return self._chat("Equity Agent. JSON with equity_score(0-1), hotspots[], mitigations[].", plan_json, T=0.1)
    def counterfactuals(self, plan_json: str) -> str:
        return self._chat("Counterfactual Agent. JSON list of rejected options with reasons.", plan_json, T=0.2)
    def explainer(self, plan_json: str, cites: List[Tuple[str,int]]) -> str:
        cites_md = "\n".join([f"{i+1}. {d}#${c}" for i,(d,c) in enumerate(cites)])
        sys = "Explainer Agent. Short public brief (markdown) with numbered citations (doc#chunk)."
        return self._chat(sys, f"Plan:\n{plan_json}\n\nCitations:\n{cites_md}", T=0.2, max_tokens=600)

# ---------------- RAG (lazy, chunk from ./data/policies) ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def _read_pdf(path: str) -> str:
    if not PdfReader:
        return ""
    try:
        reader = PdfReader(path)
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        return ""

@st.cache_data(ttl=3600, show_spinner=False)
def load_corpus(policy_dir: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(policy_dir, "**", "*"), recursive=True):
        if os.path.isdir(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        txt = _read_pdf(fp) if ext == ".pdf" else (open(fp, "r", encoding="utf-8", errors="ignore").read() if ext in (".txt", ".md") else "")
        if not txt:
            continue
        doc_id = os.path.relpath(fp, policy_dir)
        words = txt.split(); step = 900; overlap = 120
        for i in range(0, len(words), step - overlap):
            chunk = " ".join(words[i:i+step])
            rows.append({"doc_id":doc_id, "chunk_id":len(rows), "text":chunk})
    return pd.DataFrame(rows)

@dataclass
class VectorDoc:
    doc_id: str
    chunk_id: int
    text: str
    embedding: Optional[List[float]] = None

class SimpleVS:
    def __init__(self, embed_model: str, api_key: str):
        self.client = OpenAI(api_key=api_key) if (OpenAI and api_key) else None
        self.embed_model = embed_model
        self.docs: List[VectorDoc] = []
        self._mat: Optional[np.ndarray] = None
    def build(self, df: pd.DataFrame):
        self.docs = [VectorDoc(r.doc_id, int(r.chunk_id), r.text) for r in df.itertuples()]
        if not self.client or df.empty:
            return
        B = 64; allv = []
        for i in range(0, len(self.docs), B):
            batch = [d.text for d in self.docs[i:i+B]]
            resp = self.client.embeddings.create(model=self.embed_model, input=batch)
            allv.extend([e.embedding for e in resp.data])
        for d, v in zip(self.docs, allv):
            d.embedding = v
        self._mat = np.array(allv, dtype=np.float32)
    def search(self, q: str, k: int = 6) -> List[VectorDoc]:
        if not self.client or self._mat is None:
            return []
        qv = self.client.embeddings.create(model=self.embed_model, input=[q]).data[0].embedding
        qv = np.array(qv, dtype=np.float32)
        sims = self._mat @ qv / (np.linalg.norm(self._mat, axis=1) * (np.linalg.norm(qv) + 1e-9))
        idx = np.argsort(-sims)[:k]
        return [self.docs[i] for i in idx]

# ---------------- Optimisation (MILP) ----------------
@dataclass
class Intervention:
    name: str; cost_m: float; emissions_delta: float; equity_gain: float; resilience_gain: float

DEFAULT_INTERVENTIONS = [
    Intervention("Temporary pumps & barriers", 5.0, 20.0, 0.6, 18.0),
    Intervention("Bus reroute + dynamic headways", 2.0, -50.0, 0.7, 12.0),
    Intervention("Targeted shelters & comms", 1.5, 5.0, 0.9, 10.0),
    Intervention("1 MWh community battery", 3.0, -200.0, 0.5, 9.0),
    Intervention("Pop-up active travel corridors", 1.0, -30.0, 0.8, 8.0),
]

def solve_portfolio(items: List[Intervention], budget_m: float, emissions_cap: float, equity_w: float) -> Dict[str, Any]:
    if not pulp:
        return {"status":"no_milp", "selected":[], "obj":0.0}
    m = pulp.LpProblem("resilience_portfolio", pulp.LpMaximize)
    x = {iv.name: pulp.LpVariable(f"x_{i}", 0, 1, cat="Binary") for i, iv in enumerate(items)}
    m += pulp.lpSum([x[iv.name] * (iv.resilience_gain + equity_w * 10.0 * iv.equity_gain) for iv in items])
    m += pulp.lpSum([x[iv.name] * iv.cost_m for iv in items]) <= budget_m
    m += pulp.lpSum([x[iv.name] * iv.emissions_delta for iv in items]) <= emissions_cap
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [iv.name for iv in items if x[iv.name].value() and x[iv.name].value() >= 0.5]
    return {"status":"ok", "selected": selected, "obj": float(pulp.value(m.objective))}

# ---------------- RL (Contextual Bandit) ----------------
class Bandit:
    def __init__(self, actions: List[str], alpha=0.2, epsilon=0.15):
        self.actions = actions; self.alpha=alpha; self.eps=epsilon; self.q: Dict[Tuple, float] = {}
    def _key(self, city: str, rain: float, aqi: int, hour: int, ok: bool):
        return (city, "high" if rain>=10 else ("med" if rain>=3 else "low"), "poor" if aqi>=40 else ("fair" if aqi>=25 else "good"), "peak" if 7<=hour<=9 or 16<=19 else "off", "ok" if ok else "blocked")
    def select(self, city: str, rain: float, aqi: int, hour: int, ok: bool) -> str:
        k = self._key(city, rain, aqi, hour, ok)
        if not ok: return "No action (Verifier blocked)"
        if np.random.rand() < self.eps: return np.random.choice(self.actions)
        vals = [(a, self.q.get((*k, a), 0.0)) for a in self.actions]
        vals.sort(key=lambda z: -z[1])
        return vals[0][0] if vals else self.actions[0]
    def update(self, city, rain, aqi, hour, ok, action, reward):
        k = self._key(city, rain, aqi, hour, ok); old = self.q.get((*k, action), 0.0)
        self.q[(*k, action)] = old + self.alpha * (reward - old)

# ---------------- UI ----------------
st.title(APP_TITLE)

# City discovery
with st.sidebar:
    st.markdown("### Planner Controls")
    data_mode = st.selectbox("Data mode", ["LIVE", "OFFLINE"], index=0 if CONFIG["DATA_MODE"].upper()=="LIVE" else 1)
    CONFIG["DATA_MODE"] = data_mode
    if st.session_state.get("_offline_failover", False) and CONFIG["DATA_MODE"].upper() == "LIVE":
        st.warning("Overpass mirrors unreachable. Switched to OFFLINE failover for this session.")
    st.caption("LIVE uses Overpass + dynamic city/POI discovery. OFFLINE uses deterministic synthetic data for instant demos.")

cities_df = discover_irish_places()
city_names = cities_df["name"].tolist()

with st.sidebar:
    city = st.selectbox("Smart city (dynamic)", options=city_names, index=city_names.index("Dublin") if "Dublin" in city_names else 0)
    goal = st.text_area("Resilience goal", value="Reduce flood disruption while protecting access; maintain ≤0 ΔtCO₂e and improve equity.", height=80)
    active_layers = st.multiselect("Layers", options=list(CONFIG["CONFIG_POIS"].keys()), default=list(CONFIG["CONFIG_POIS"].keys()))
    draw_legs = st.checkbox("Show centroid→node legs/whiskers", True)
    zoom_km = st.slider("POI search radius (km)", 2.0, 15.0, 8.0, 0.5)
    st.markdown("**Constraints**")
    budget_m = st.slider("Budget (million €)", 1.0, 100.0, 25.0, 1.0)
    emissions_cap = st.slider("Max additional ΔtCO₂e", -2000.0, 2000.0, 0.0, 50.0)
    equity_w = st.slider("Equity weight (0..1)", 0.0, 1.0, 0.6, 0.05)

    st.markdown("**Agents**")
    do_verify = st.checkbox("Verifier", True)
    do_equity = st.checkbox("Equity Agent", True)
    do_counterf = st.checkbox("Counterfactuals", True)
    do_explain = st.checkbox("Public Brief", True)
    run_planner = st.button("Run Scenario Planner", type="primary")

# Map panel
col_map, col_right = st.columns([1.35, 1.65])
with col_map:
    st.markdown("#### Geospatial View (Dynamic centroids · nodes · legs)")
    row = cities_df[cities_df["name"]==city].iloc[0]
    base_lat, base_lon = float(row["lat"]), float(row["lon"]) 
    # Nudge Galway inland slightly (robust), but also rely on snap_inland for all nodes
    if city.lower()=="galway":
        base_lon += 0.012

    fmap = folium.Map(location=[base_lat, base_lon], zoom_start=12, tiles="CartoDB positron")

    if CONFIG.get("MET_RAIN_TILES"):
        folium.raster_layers.TileLayer(tiles=CONFIG["MET_RAIN_TILES"], name="Rain radar", overlay=True, control=True, opacity=0.55).add_to(fmap)

    groups: Dict[str, folium.FeatureGroup] = {}
    for lname in ["Centroid", *active_layers, "Legs"]:
        fg = folium.FeatureGroup(name=lname)
        groups[lname] = fg; fmap.add_child(fg)

    # signals & popup engine
    signals = get_signals(city)
    nlp = NLPPopup()

    # centroid
    folium.CircleMarker([base_lat, base_lon], radius=7, tooltip=f"{city} (centroid)", popup=folium.Popup(nlp.render(city, "city centre", {"name": city}, signals), max_width=320)).add_to(groups["Centroid"])

    # fetch & plot
    all_points: Dict[str, List[Dict[str, Any]]] = {}
    for lname in active_layers:
        pts = fetch_pois(lname, base_lat, base_lon, km=zoom_km)
        all_points[lname] = pts
        # choose icon style dynamically
        style = CONFIG["CONFIG_POIS"][lname]["icon"]
        cluster = MarkerCluster() if lname == "Shelters" else None
        tgt_layer = cluster.add_to(groups[lname]) if cluster else groups[lname]
        for p in pts:
            html = nlp.render(city, lname, p, signals)
            if style.get("kind") == "circle":
                folium.CircleMarker([p["lat"], p["lon"]], radius=style.get("radius", 6), tooltip=p.get("name", lname), popup=folium.Popup(html, max_width=360)).add_to(tgt_layer)
            else:  # marker
                folium.Marker([p["lat"], p["lon"]], tooltip=p.get("name", lname), popup=folium.Popup(html, max_width=360), icon=folium.Icon(color=style.get("color"), icon=style.get("icon") or "", prefix="fa")).add_to(tgt_layer)
        # legs
        if draw_legs:
            for p in pts:
                folium.PolyLine([(base_lat, base_lon), (p["lat"], p["lon"])], weight=1, opacity=0.7).add_to(groups["Legs"])

    Draw(export=True, filename="drawn.geojson").add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, height=560, width=None)
    if use_offline():
        st.caption("POIs rendered in OFFLINE mode (synthetic, deterministic).")
    else:
        st.caption("POIs fetched live from Overpass (cached).")

with col_right:
    st.markdown("#### Evidence · Knowledge Graph · Agents (on demand)")
    kg = KnowledgeGraph.build(goal, active_layers)
    st.dataframe(kg.df(), use_container_width=True)

    plan_json = verifier_json = equity_json = counterf_json = explainer_md = ""
    citations: List[Tuple[str,int]] = []
    if run_planner:
        corpus = load_corpus(CONFIG["policy_dir"])
        st.caption(f"Corpus chunks: {len(corpus)} from {corpus['doc_id'].nunique() if not corpus.empty else 0} docs")
        if CONFIG["openai_key"] and not corpus.empty:
            with st.spinner("Embedding corpus…"):
                vs = SimpleVS(CONFIG["embed_model"], CONFIG["openai_key"]) ; vs.build(corpus)
            q1 = f"{city} resilience goal: {goal}"; q2 = f"policy constraints {city} emissions budget equity"
            d1 = vs.search(q1, k=4); d2 = vs.search(q2, k=4)
            ids1 = {(d.doc_id, d.chunk_id) for d in d1}; ids2 = {(d.doc_id, d.chunk_id) for d in d2}
            jury_overlap = len(ids1.intersection(ids2)) / max(1, len(ids1.union(ids2)))
            top = (d1 + d2)[:6]
            evidence = [d.text[:1200] for d in top]
            citations = [(d.doc_id, d.chunk_id) for d in top]
        else:
            jury_overlap = 0.4
            evidence = [
                "Climate Action Plan 2024 emphasises emissions ceilings and modal shift.",
                "NPF highlights compact growth and resilient infrastructure in cities.",
                f"{city} development plan prioritises flood risk management and equitable access.",
            ]
        agents = Agents()
        with st.spinner("Planner Agent…"): plan_json = agents.planner(goal, city, evidence, jury_overlap)
        if do_verify:
            with st.spinner("Verifier…"): verifier_json = agents.verifier(plan_json, evidence)
        if do_equity:
            equity_json = agents.equity(plan_json)
        if do_counterf:
            counterf_json = agents.counterfactuals(plan_json)
        if do_explain:
            explainer_md = agents.explainer(plan_json, citations)

    if plan_json:
        st.markdown("##### Planner Output (JSON)"); st.code(plan_json, language="json")
    if verifier_json:
        st.markdown("##### Policy Compliance (Verifier JSON)"); st.code(verifier_json, language="json")
    if equity_json:
        st.markdown("##### Equity / Co-benefits (JSON)"); st.code(equity_json, language="json")
    if counterf_json:
        st.markdown("##### Counterfactuals (JSON)"); st.code(counterf_json, language="json")
    if explainer_md:
        st.markdown("##### Public Brief"); st.write(explainer_md)

# ---------------- Optimisation ----------------
st.markdown("---")
st.markdown("### Optimisation — Portfolio under constraints (MILP)")
colA, colB = st.columns([1.0, 2.0])
with colA:
    st.caption("Objective: maximise resilience + (equity_weight×equity)")
    if st.button("Solve Portfolio"):
        sol = solve_portfolio(DEFAULT_INTERVENTIONS, budget_m, emissions_cap, equity_w)
        st.session_state["portfolio"] = sol
with colB:
    sol = st.session_state.get("portfolio")
    if sol: st.success(f"Solution: {sol['selected']} (objective={sol['obj']:.2f})")
    st.dataframe(pd.DataFrame([vars(iv) for iv in DEFAULT_INTERVENTIONS]), use_container_width=True)

# ---------------- RL ----------------
st.markdown("---")
st.markdown("### Self-Learning (Contextual RL)")
actions = [
    "Temporary pumps & barriers",
    "Bus reroute + dynamic headways",
    "Targeted shelters & comms",
    "1 MWh community battery",
    "Pop-up active travel corridors",
]
if "bandit" not in st.session_state:
    st.session_state["bandit"] = Bandit(actions)
bandit: Bandit = st.session_state["bandit"]
ctx = get_signals(city)
col1, col2 = st.columns([1.2, 1.8])
with col1:
    st.caption(f"Context: rain={ctx['rain_mm_24h']:.1f}mm, AQI={ctx['aqi']}, hour={ctx['hour']}")
    verifier_ok = True
    suggestion = bandit.select(city, ctx['rain_mm_24h'], ctx['aqi'], ctx['hour'], verifier_ok)
    st.success(f"Suggested action for {city}: {suggestion}")
with col2:
    reward = st.slider("Reward", -10.0, 10.0, 2.0, 0.5)
    if st.button("Update RL"):
        if suggestion != "No action (Verifier blocked)":
            bandit.update(city, ctx['rain_mm_24h'], ctx['aqi'], ctx['hour'], True, suggestion, reward)
            st.info("RL updated for this context.")
        else:
            st.warning("RL update blocked: Verifier failed or action blocked.")

# ---------------- Governance / Audit ----------------
st.markdown("---")
st.markdown("### Governance, Audit & Model/Data Cards")
colx, coly, colz = st.columns(3)
with colx:
    st.subheader("Run State")
    state = st.radio("Decision state", ["Draft","Reviewed","Approved"], horizontal=True)
with coly:
    st.subheader("Model Card (Planner)")
    st.markdown("- Model: OpenAI Chat\n- Purpose: scenario planning\n- Limits: depends on RAG quality; hallucination risk if poor evidence\n- Safety: Verifier + human review")
with colz:
    st.subheader("Data Sheet (Evidence)")
    st.markdown("- Sources: policy PDFs, official APIs\n- Freshness: fetch timestamps to be added\n- Known gaps: live GTFS/EirGrid wiring")

SNAP_CSV = os.path.join(CONFIG["log_dir"], "decisions.csv")
if st.button("Save Decision Snapshot"):
    row = {
        "ts": pd.Timestamp.utcnow().isoformat(), "city": city, "goal": goal,
        "budget_m": budget_m, "emissions_cap": emissions_cap, "equity_w": equity_w,
        "portfolio": json.dumps(st.session_state.get("portfolio", {})), "state": state,
    }
    pd.DataFrame([row]).to_csv(SNAP_CSV, mode='a', header=not os.path.exists(SNAP_CSV), index=False)
    st.success("Snapshot saved to data/logs/decisions.csv")

# ---------------- Footer ----------------
st.markdown(
    """
<small>
Dynamic city/POI discovery via Overpass (or OFFLINE synthetic). Adaptive NLP popups reflect live signals and POI tags.
Centroid→node legs are optional. Heavy steps (RAG/agents) are opt‑in and cached for speed.
Wire real Met Éireann/EPA/CSO when ready by replacing get_signals().
</small>
""",
    unsafe_allow_html=True,
)
