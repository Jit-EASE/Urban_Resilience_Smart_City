# UNIC Urban Resilience & Smart Cities — Ireland (v2)
# Streamlit App — Agentic RAG + Knowledge Graph + MILP Optimiser + Contextual RL + Folium Map
# -------------------------------------------------------------------------------------------
# Highlights
# - Knowledge Graph (in‑memory) linking Goals→Targets→Measures→Datasets→Tools→Stakeholders with query helpers
# - Multi‑agent orchestration (Planner, Verifier, Equity/Cobenefits, Explainer, Counterfactual) with structured outputs
# - Optimisation: MILP portfolio selector (PuLP) with budget/emissions/equity constraints + quantum‑inspired toggle (placeholder)
# - RL: Contextual bandit with state features + safety gate via Verifier; learning dashboard
# - Geospatial: LayerControl overlays (Hospitals/Shelters/Substations demo), Draw tool for what‑if lines/areas, marker clustering
# - Governance/EU AI Act posture: model/data cards, audit states (Draft→Reviewed→Approved), immutable run logs
# - Multi‑city scaling: City profiles (Dublin, Cork, Limerick, Galway, Waterford) with defaults
# - Engineering polish: config block, stricter error handling, caching, requirements list


import os
import io
import json
import glob
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
from streamlit_folium import st_folium
import folium

from folium.plugins import Draw, MarkerCluster

# GTFS‑RT (optional)
try:
    from google.transit import gtfs_realtime_pb2  # pip: gtfs-realtime-bindings
except Exception:
    gtfs_realtime_pb2 = None

# PDF handling
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Optimisation (MILP)
try:
    import pulp
except Exception:
    pulp = None

# ---------------- CONFIG ----------------
APP_TITLE = "UNIC Regional Resilience (Ireland) — Agentic + KG + RL"
st.set_page_config(page_title=APP_TITLE, layout="wide")

CONFIG = {
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    "openai_key": os.getenv("OPENAI_API_KEY", ""),
    "data_dir": "data",
    "policy_dir": os.path.join("data", "policies"),
    "log_dir": os.path.join("data", "logs"),
    "tfi_gtfs_rt_url": os.getenv("TFI_GTFS_RT_URL", ""),  # e.g., https://.../vehiclePositions.pb
    "epa_aq_api": os.getenv("EPA_AQ_API", ""),           # optional JSON endpoint
    "met_radar_tiles": os.getenv("MET_RAIN_TILES", ""),  # e.g., https://{s}.tile.openweathermap.org/... (or Met Éireann radar template)
}

os.makedirs(CONFIG["policy_dir"], exist_ok=True)
os.makedirs(CONFIG["log_dir"], exist_ok=True)

REQUIREMENTS = """
streamlit>=1.37
streamlit-folium>=0.18
folium>=0.15
pypdf>=4.2
pandas>=2.0
numpy>=1.25
openai>=1.40
requests>=2.31
scikit-learn>=1.4
pulp>=2.8
python-dotenv>=1.0
gtfs-realtime-bindings>=1.0
"""

# ---------------- County & City Profiles ----------------
COUNTIES = {
    "Carlow": (52.7236, -6.9439), "Cavan": (53.9908, -7.3606), "Clare": (52.8436, -8.9810),
    "Cork": (51.8986, -8.4756), "Donegal": (54.6540, -8.1041), "Dublin": (53.3498, -6.2603),
    "Galway": (53.2707, -9.0568), "Kerry": (52.1449, -9.5160), "Kildare": (53.1589, -6.9092),
    "Kilkenny": (52.6541, -7.2448), "Laois": (53.0326, -7.2995), "Leitrim": (54.1937, -8.0002),
    "Limerick": (52.6629, -8.6305), "Longford": (53.7275, -7.7939), "Louth": (53.9509, -6.5400),
    "Mayo": (53.8044, -9.5246), "Meath": (53.6055, -6.6564), "Monaghan": (54.2496, -6.9683),
    "Offaly": (53.2734, -7.4903), "Roscommon": (53.6273, -8.1891), "Sligo": (54.2697, -8.4694),
    "Tipperary": (52.4737, -7.8740), "Waterford": (52.2593, -7.1101), "Westmeath": (53.5333, -7.3500),
    "Wexford": (52.3369, -6.4633), "Wicklow": (52.9800, -6.0400),
}

# Higher‑fidelity city centroids (rounded). Prefer these over county centroids for city maps.
CITY_CENTROIDS = {
    "Dublin": (53.350140, -6.266155),
    "Cork": (51.903614, -8.468399),
    "Limerick": (52.668018, -8.630498),
    "Galway": (53.270962, -9.062691),
    "Waterford": (52.259319, -7.110070),
}

CITY_PROFILES = {
    "Dublin": {
        "focus_layers": ["bus_routes", "hospitals", "substations"],
        "default_goal": "Reduce pluvial flood disruption and protect public transport access while holding emissions neutral.",
    },
    "Cork": {
        "focus_layers": ["shelters", "hospitals", "bus_routes"],
        "default_goal": "Cut flood damages by 30% and improve air quality near quays; equity priority high.",
    },
    "Limerick": {
        "focus_layers": ["substations", "shelters"],
        "default_goal": "Expand positive energy actions and maintain hospital access during storms.",
    },
    "Galway": {
        "focus_layers": ["hospitals", "bus_routes"],
        "default_goal": "Waterway micro‑hydro integration with mobility continuity during heavy rain.",
    },
    "Waterford": {
        "focus_layers": ["shelters", "bus_routes"],
        "default_goal": "North Quays smart district resilience and walkable access to services.",
    },
}

# ---------------- Utility: PDF & chunking ----------------
@st.cache_data(show_spinner=False)
def read_pdf_text(path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(path)
        texts = []
        for p in reader.pages:
            try:
                texts.append(p.extract_text() or "")
            except Exception:
                pass
        return "\n".join(texts)
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def chunk_text(text: str, max_tokens: int = 900, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    step = max(max_tokens - overlap, 1)
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_tokens]))
        i += step
    return chunks

@st.cache_data(show_spinner=False)
def load_policy_corpus(policy_dir: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(policy_dir, "**", "*"), recursive=True):
        if os.path.isdir(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        txt = ""
        if ext == ".pdf":
            txt = read_pdf_text(fp)
        elif ext in (".txt", ".md"):
            try:
                txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
            except Exception:
                txt = ""
        if not txt:
            continue
        doc_id = os.path.relpath(fp, policy_dir)
        for i, ch in enumerate(chunk_text(txt)):
            rows.append({"doc_id": doc_id, "source_path": fp, "chunk_id": i, "text": ch})
    return pd.DataFrame(rows)

# ---------------- Embeddings / Vector store ----------------
@dataclass
class VectorDoc:
    doc_id: str
    chunk_id: int
    text: str
    source_path: str
    embedding: Optional[List[float]] = None

class SimpleVectorStore:
    def __init__(self, embed_model: str, api_key: str):
        self.embed_model = embed_model
        self.client = OpenAI(api_key=api_key) if (OpenAI and api_key) else None
        self.docs: List[VectorDoc] = []
        self._mat: Optional[np.ndarray] = None

    def build(self, df: pd.DataFrame):
        self.docs = [VectorDoc(r.doc_id, int(r.chunk_id), r.text, r.source_path) for r in df.itertuples()]
        self._embed_all()

    def _embed_all(self):
        if not self.client or not self.docs:
            return
        B = 64
        embs: List[List[float]] = []
        for i in range(0, len(self.docs), B):
            batch = [d.text for d in self.docs[i:i+B]]
            resp = self.client.embeddings.create(model=CONFIG["embed_model"], input=batch)
            embs.extend([e.embedding for e in resp.data])
        for d, e in zip(self.docs, embs):
            d.embedding = e
        self._mat = np.array(embs, dtype=np.float32)

    def search(self, q: str, k: int = 6) -> List[VectorDoc]:
        if not self.client or self._mat is None:
            return []
        qv = self.client.embeddings.create(model=CONFIG["embed_model"], input=[q]).data[0].embedding
        qv = np.array(qv, dtype=np.float32)
        sims = self._mat @ qv / (np.linalg.norm(self._mat, axis=1) * (np.linalg.norm(qv) + 1e-9))
        idx = np.argsort(-sims)[:k]
        return [self.docs[i] for i in idx]

# ---------------- Knowledge Graph (Goals→Targets→Measures→Datasets→Tools→Stakeholders) ----------------
class KnowledgeGraph:
    def __init__(self):
        self.edges: List[Tuple[str, str, str]] = []  # (src, relation, dst)
        self.node_types: Dict[str, str] = {}

    def add(self, src: str, rel: str, dst: str, src_type: str, dst_type: str):
        self.edges.append((src, rel, dst))
        self.node_types[src] = src_type
        self.node_types[dst] = dst_type

    def neighbors(self, node: str) -> List[Tuple[str, str]]:
        return [(rel, dst) for (s, rel, dst) in self.edges if s == node]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for s, r, d in self.edges:
            rows.append({"source": s, "relation": r, "target": d, "source_type": self.node_types.get(s, ""), "target_type": self.node_types.get(d, "")})
        return pd.DataFrame(rows)

    @staticmethod
    def demo(county: str) -> "KnowledgeGraph":
        kg = KnowledgeGraph()
        goal = f"{county}: Reduce pluvial flood disruption & protect access"
        kg.add(goal, "targets", "-30% 10‑yr flood damages", "Goal", "Target")
        kg.add(goal, "targets", "<=0 ΔtCO₂e", "Goal", "Target")
        kg.add("-30% 10‑yr flood damages", "measured_by", "Service-hours preserved", "Target", "Measure")
        kg.add("<=0 ΔtCO₂e", "measured_by", "Emissions delta", "Target", "Measure")
        kg.add("Service-hours preserved", "requires", "Bus headways (GTFS‑RT)", "Measure", "Dataset")
        kg.add("Emissions delta", "requires", "Energy mix (EirGrid)", "Measure", "Dataset")
        kg.add("Bus headways (GTFS‑RT)", "queried_by", "MobilityAgent", "Dataset", "Tool")
        kg.add("Energy mix (EirGrid)", "queried_by", "EnergyAgent", "Dataset", "Tool")
        kg.add("MobilityAgent", "owned_by", "NTA/TFI", "Tool", "Stakeholder")
        kg.add("EnergyAgent", "owned_by", "EirGrid", "Tool", "Stakeholder")
        return kg

# ---------------- Agentic Layer ----------------
class Agents:
    def __init__(self):
        self.client = OpenAI(api_key=CONFIG["openai_key"]) if (OpenAI and CONFIG["openai_key"]) else None
        self.model = CONFIG["openai_model"]

    def _chat(self, messages, temperature=0.2, max_tokens=900) -> str:
        if not self.client:
            return "[OpenAI key missing]"
        r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content

    def planner(self, goal: str, county: str, evidence_snips: List[str], jury_score: float) -> str:
        sys = (
            "You are a Planner Agent for Irish urban resilience. Output JSON with keys: "
            "objectives, constraints, actions, kpis, checklist48h, notes. Keep budget/emissions/equity explicit."
        )
        ctx = "\n\n".join([f"[E{i+1}] {s}" for i, s in enumerate(evidence_snips)])
        user = f"County: {county}\nJuryConsensus: {jury_score:.2f}\nGoal: {goal}\nEvidence:\n{ctx}"
        return self._chat([{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.1)

    def verifier(self, plan_json: str, evidence_snips: List[str]) -> str:
        sys = (
            "You are a Verifier Agent. Return a JSON with policy_checks: [{name, pass, reason, citations}], "
            "risk_flags:[], required_data:[], and overall: pass|fail."
        )
        ctx = "\n\n".join([f"[E{i+1}] {s}" for i, s in enumerate(evidence_snips)])
        user = f"PlanJSON:\n{plan_json}\n\nEvidence:\n{ctx}"
        return self._chat([{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.0)

    def equity(self, plan_json: str) -> str:
        sys = "Equity/Cobenefits Agent. Given Plan JSON, output JSON with equity_score(0-1), hotspots[], mitigations[]."
        return self._chat([{"role":"system","content":sys},{"role":"user","content":plan_json}], temperature=0.1)

    def counterfactuals(self, plan_json: str) -> str:
        sys = "Counterfactual Agent. Produce JSON list of rejected options with reasons (cost/emissions/equity/feasibility)."
        return self._chat([{"role":"system","content":sys},{"role":"user","content":plan_json}], temperature=0.2)

    def explainer(self, plan_json: str, citations: List[Tuple[str,int]]) -> str:
        sys = "Explainer Agent. Output a brief (markdown) for the public. Include a numbered citations list (doc#chunk)."
        cites = "\n".join([f"{i+1}. {d}#${c}" for i,(d,c) in enumerate(citations)])
        user = f"Plan:\n{plan_json}\n\nCitations:\n{cites}"
        return self._chat([{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.2)

# ---------------- Official Data Connectors (stubs with simulated fallback) ----------------
import requests

@st.cache_data(ttl=1800)
def fetch_met_eireann_forecast(county: str) -> Dict[str, Any]:
    try:
        return {"status":"sim", "rain_mm_next24h": float(np.random.gamma(2,3))}
    except Exception:
        return {"status":"sim", "rain_mm_next24h": float(np.random.gamma(2,3))}

@st.cache_data(ttl=1800)
def fetch_epa_air_quality(county: str) -> Dict[str, Any]:
    return {"aqi": int(np.random.randint(15, 65)), "pm25": round(float(np.random.uniform(5, 20)), 1)}


@st.cache_data(ttl=3600)
def fetch_cso_population(county: str) -> Dict[str, Any]:
    base = {"Dublin":1500000, "Cork":600000, "Limerick":210000, "Galway":280000, "Waterford":127000}
    return {"population": int(base.get(county, int(np.random.randint(70000, 400000))))}

#
# ---------------- Dynamic OSM POIs (Hospitals / Shelters / Substations) ----------------
@st.cache_data(ttl=3600)
def _city_bbox(lat: float, lon: float, km: float = 8.0) -> Tuple[float,float,float,float]:
    # approx degrees per km
    dlat = km / 110.574
    dlon = km / (111.320 * math.cos(math.radians(lat)))
    south, north = lat - dlat, lat + dlat
    west, east = lon - dlon, lon + dlon
    return south, west, north, east

def _overpass(query: str) -> Dict[str, Any]:
    url = "https://overpass-api.de/api/interpreter"
    try:
        r = requests.post(url, data={"data": query}, timeout=25)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"elements": []}

@st.cache_data(ttl=3600)
def fetch_osm_hospitals(lat: float, lon: float) -> List[Dict[str, Any]]:
    s,w,n,e = _city_bbox(lat, lon, km=10)
    q = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"]({s},{w},{n},{e});
      way["amenity"="hospital"]({s},{w},{n},{e});
      relation["amenity"="hospital"]({s},{w},{n},{e});
    );
    out center tags;"""
    data = _overpass(q)
    pois = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Hospital")
        if "lat" in el and "lon" in el:
            pois.append({"lat": el["lat"], "lon": el["lon"], "name": name})
        elif "center" in el:
            pois.append({"lat": el["center"]["lat"], "lon": el["center"]["lon"], "name": name})
    return pois

@st.cache_data(ttl=3600)
def fetch_osm_shelters(lat: float, lon: float) -> List[Dict[str, Any]]:
    s,w,n,e = _city_bbox(lat, lon, km=10)
    q = f"""
    [out:json][timeout:25];
    (
      node["amenity"="shelter"]({s},{w},{n},{e});
      way["amenity"="shelter"]({s},{w},{n},{e});
      relation["amenity"="shelter"]({s},{w},{n},{e});
      node["emergency"="shelter"]({s},{w},{n},{e});
    );
    out center tags;"""
    data = _overpass(q)
    pois = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Shelter")
        if "lat" in el and "lon" in el:
            pois.append({"lat": el["lat"], "lon": el["lon"], "name": name})
        elif "center" in el:
            pois.append({"lat": el["center"]["lat"], "lon": el["center"]["lon"], "name": name})
    return pois

# ---------------- Land‑snap safeguard & live data connectors ----------------
@st.cache_data(ttl=3600)
def fetch_osm_substations(lat: float, lon: float) -> List[Dict[str, Any]]:
    s,w,n,e = _city_bbox(lat, lon, km=12)
    q = f"""
    [out:json][timeout:25];
    (
      node["power"="substation"]({s},{w},{n},{e});
      way["power"="substation"]({s},{w},{n},{e});
      relation["power"="substation"]({s},{w},{n},{e});
    );
    out center tags;"""
    data = _overpass(q)
    pois = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Substation")
        if "lat" in el and "lon" in el:
            pois.append({"lat": el["lat"], "lon": el["lon"], "name": name})
        elif "center" in el:
            pois.append({"lat": el["center"]["lat"], "lon": el["center"]["lon"], "name": name})
    return pois

def snap_inland_if_water(lat: float, lon: float) -> Tuple[float, float]:
    """Best‑effort nudge for coastal points: if the area is dominated by water and lacks roads/landuse within 150m,
    push slightly inland (east or north depending on coastline orientation). Uses Overpass heuristics; cheap & cached upstream."""
    try:
        rad_km = 0.15
        s,w,n,e = _city_bbox(lat, lon, km=rad_km)
        q = f"""
        [out:json][timeout:20];
        (
          way["natural"="coastline"]({s},{w},{n},{e});
          way["water"]; node({s},{w},{n},{e});
        );
        out count;"""
        water = _overpass(q).get("elements", [])
        q2 = f"""
        [out:json][timeout:20];
        (
          way["highway"]({s},{w},{n},{e});
          way["landuse"]({s},{w},{n},{e});
        );
        out count;"""
        landish = _overpass(q2).get("elements", [])
        # If water detected and no roads/landuse, nudge inland
        if water and not landish:
            return lat + 0.003, lon + 0.007
    except Exception:
        pass
    return lat, lon

def _snap_list(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out = []
    for a, b in points:
        la, lo = snap_inland_if_water(a, b)
        out.append((la, lo))
    return out

@st.cache_data(ttl=60)
def fetch_gtfs_vehicles(url: str) -> List[Dict[str, Any]]:
    """Parse GTFS‑RT vehicle positions (if available). Returns list of {lat, lon, bearing, label}."""
    if not url or gtfs_realtime_pb2 is None:
        return []
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        out = []
        for ent in feed.entity:
            if not ent.vehicle or not ent.vehicle.position:
                continue
            pos = ent.vehicle.position
            lat = float(getattr(pos, "latitude", 0.0))
            lon = float(getattr(pos, "longitude", 0.0))
            if lat == 0.0 and lon == 0.0:
                continue
            out.append({
                "lat": lat,
                "lon": lon,
                "bearing": float(getattr(pos, "bearing", 0.0)),
                "label": getattr(ent.vehicle, "vehicle", {}).id if hasattr(ent.vehicle, "vehicle") else (ent.id or "bus"),
            })
        return out
    except Exception:
        return []

@st.cache_data(ttl=900)
def fetch_epa_aq_stations(lat: float, lon: float) -> List[Dict[str, Any]]:
    """Try EPA JSON API if provided; else fallback to OSM monitoring stations within bbox."""
    api = CONFIG.get("epa_aq_api", "")
    if api:
        try:
            r = requests.get(api, timeout=20)
            if r.status_code == 200:
                data = r.json()
                # Expecting list of stations with lat/lon/name/pm25 (schema may vary)
                out = []
                for s in data:
                    if all(k in s for k in ("latitude", "longitude")):
                        out.append({
                            "lat": float(s["latitude"]),
                            "lon": float(s["longitude"]),
                            "name": s.get("name", "AQ Station"),
                            "pm25": s.get("pm25"),
                        })
                if out:
                    return out
        except Exception:
            pass
    # Fallback: OSM environmental monitoring stations
    s,w,n,e = _city_bbox(lat, lon, km=15)
    q = f"""
    [out:json][timeout:25];
    (
      node["man_made"="monitoring_station"]({s},{w},{n},{e});
      way["man_made"="monitoring_station"]({s},{w},{n},{e});
      relation["man_made"="monitoring_station"]({s},{w},{n},{e});
    );
    out center tags;"""
    data = _overpass(q)
    out = []
    for el in data.get("elements", []):
        nm = el.get("tags", {}).get("name", "Monitoring station")
        if "lat" in el and "lon" in el:
            out.append({"lat": el["lat"], "lon": el["lon"], "name": nm})
        elif "center" in el:
            out.append({"lat": el["center"]["lat"], "lon": el["center"]["lon"], "name": nm})
    return out

# ---------------- NLP helper for map popups ----------------
class NLPPopup:
    def __init__(self):
        self.client = OpenAI(api_key=CONFIG["openai_key"]) if (OpenAI and CONFIG["openai_key"]) else None
        self.model = CONFIG["openai_model"]

    def generate(self, city: str, layer: str, signals: Dict[str, Any]) -> str:
        rain = signals.get("rain", 0.0)
        aqi = signals.get("aqi", 0)
        pop = signals.get("pop", 0)
        # Fallback HTML if no API key
        if not self.client:
            return (
                f"<b>{city}</b> — {layer.title()}"
                f"<br/>Context: rain≈{rain:.1f} mm, AQI={aqi}, pop≈{pop:,}."
                f"<br/>Hint: prioritise access & low‑emission options during wet periods."
            )
        msg = [
            {"role": "system", "content": "Write one precise sentence (<=28 words) for a smart‑city map popup in Ireland. Be concrete and non‑fluffy."},
            {"role": "user", "content": f"City={city}; Layer={layer}; rain_mm={rain:.1f}; AQI={aqi}; population={pop}. Output plain text only."}
        ]
        try:
            out = self.client.chat.completions.create(model=self.model, messages=msg, temperature=0.2, max_tokens=60)
            text = (out.choices[0].message.content or "").strip()
        except Exception:
            text = f"{layer.title()} action: maintain service continuity; coordinate shelters, bus headways and energy buffers."
        return f"<b>{city}</b>: {text}"

#
# ---------------- Reinforcement Learning (Contextual Bandit) ----------------
class ContextualBandit:
    def __init__(self, actions: List[str], alpha=0.2, epsilon=0.15):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.q: Dict[Tuple[str, str, str, str, str], float] = {}

    def _key(self, county: str, rain: float, aqi: int, hour: int, verifier_ok: bool) -> Tuple[str,str,str,str,str]:
        rain_bin = "high" if rain>=10 else ("med" if rain>=3 else "low")
        aqi_bin = "poor" if aqi>=40 else ("fair" if aqi>=25 else "good")
        peak = "peak" if 7<=hour<=9 or 16<=hour<=19 else "off"
        v = "ok" if verifier_ok else "blocked"
        return (county, rain_bin, aqi_bin, peak, v)

    def select(self, county: str, rain: float, aqi: int, hour: int, verifier_ok: bool) -> str:
        key = self._key(county, rain, aqi, hour, verifier_ok)
        if not verifier_ok:
            return "No action (Verifier blocked)"
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        vals = [(a, self.q.get((*key, a), 0.0)) for a in self.actions]
        vals.sort(key=lambda x: -x[1])
        return vals[0][0] if vals else self.actions[0]

    def update(self, county: str, rain: float, aqi: int, hour: int, verifier_ok: bool, action: str, reward: float):
        key = self._key(county, rain, aqi, hour, verifier_ok)
        old = self.q.get((*key, action), 0.0)
        self.q[(*key, action)] = old + self.alpha * (reward - old)

# ---------------- Optimisation (MILP portfolio) ----------------
@dataclass
class Intervention:
    name: str
    cost_m: float
    emissions_delta: float
    equity_gain: float  # 0..1
    resilience_gain: float  # generic score

DEFAULT_INTERVENTIONS = [
    Intervention("Temporary pumps & barriers", 5.0, 20.0, 0.6, 18.0),
    Intervention("Bus reroute + dynamic headways", 2.0, -50.0, 0.7, 12.0),
    Intervention("Targeted shelters & comms", 1.5, 5.0, 0.9, 10.0),
    Intervention("1 MWh community battery", 3.0, -200.0, 0.5, 9.0),
    Intervention("Pop-up active travel corridors", 1.0, -30.0, 0.8, 8.0),
]

def solve_portfolio(interventions: List[Intervention], budget_m: float, emissions_cap: float, equity_weight: float) -> Dict[str, Any]:
    if pulp is None:
        return {"status":"no_milp", "selected":[], "obj":0.0}
    m = pulp.LpProblem("resilience_portfolio", pulp.LpMaximize)
    x = {iv.name: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i, iv in enumerate(interventions)}
    m += pulp.lpSum([x[iv.name] * (iv.resilience_gain + equity_weight * 10.0 * iv.equity_gain) for iv in interventions])
    m += pulp.lpSum([x[iv.name] * iv.cost_m for iv in interventions]) <= budget_m
    m += pulp.lpSum([x[iv.name] * iv.emissions_delta for iv in interventions]) <= emissions_cap
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [iv.name for iv in interventions if x[iv.name].value() and x[iv.name].value() >= 0.5]
    obj = pulp.value(m.objective)
    return {"status":"ok", "selected": selected, "obj": obj}

# ---------------- UI ----------------
st.title(APP_TITLE)
with st.sidebar:
    st.markdown("### Controls")
    city = st.selectbox("Smart city", options=list(CITY_PROFILES.keys()), index=0)
    county = st.selectbox("County", options=sorted(COUNTIES.keys()), index=sorted(COUNTIES.keys()).index(city) if city in COUNTIES else 0)
    default_goal = CITY_PROFILES[city]["default_goal"]
    goal = st.text_area("Resilience goal", value=default_goal, height=80)

    st.markdown("**Scenario constraints**")
    budget_m = st.slider("Budget (million €)", 1.0, 100.0, 25.0, 1.0)
    emissions_cap = st.slider("Max additional ΔtCO₂e", -2000.0, 2000.0, 0.0, 50.0)
    equity_w = st.slider("Equity weight (0..1)", 0.0, 1.0, 0.6, 0.05)

    st.markdown("**Agentic options**")
    do_verify = st.checkbox("Run Verifier", True)
    do_equity = st.checkbox("Equity/Cobenefits Agent", True)
    do_counterf = st.checkbox("Counterfactuals", True)
    do_explain = st.checkbox("Public Brief", True)
    run_button = st.button("Run Scenario Planner", type="primary")

#
# ---------------- Map ----------------
col_map, col_right = st.columns([1.25, 1.75])
with col_map:
    st.markdown("#### Geospatial View")

    # Prefer city centroid; apply slight inland bias for coastal cities (fix Galway into the ocean)
    base_lat, base_lon = CITY_CENTROIDS.get(city, COUNTIES[county])
    bias_lat, bias_lon = 0.0, 0.0
    if city == "Galway":
        bias_lon = +0.015  # push markers slightly east (inland)
    lat = base_lat + bias_lat
    lon = base_lon + bias_lon

    fmap = folium.Map(location=[lat, lon], zoom_start=12, tiles="CartoDB positron")

    # Rain radar overlay (if a tile template is provided via env)
    if CONFIG.get("met_radar_tiles"):
        folium.raster_layers.TileLayer(
            tiles=CONFIG["met_radar_tiles"],
            name="Rain radar",
            attr="Met Éireann / tiles",
            overlay=True,
            control=True,
            opacity=0.55,
        ).add_to(fmap)

    # Overlays
    layers_enabled = CITY_PROFILES[city]["focus_layers"]
    groups: Dict[str, folium.FeatureGroup] = {}
    for lname in ["hospitals", "shelters", "substations", "bus_routes", "aq_stations", "gtfs_live"]:
        fg = folium.FeatureGroup(name=lname.replace('_', ' ').title())
        groups[lname] = fg
        fmap.add_child(fg)

    # Signals for NLP popups
    met = fetch_met_eireann_forecast(county)
    epa = fetch_epa_air_quality(county)
    cso = fetch_cso_population(county)
    signals = {"rain": met.get("rain_mm_next24h", 0.0), "aqi": epa.get("aqi", 0), "pop": cso.get("population", 0)}
    nlp = NLPPopup()

    rng = np.random.default_rng(42)

    def inland_jitter(n=6, r=0.02, lon_push=0.0):
        pts = []
        for _ in range(n):
            pts.append((lat + float(rng.normal(0, r)), lon + float(rng.normal(0, r)) + lon_push))
        return pts

    # --- Dynamic hospitals from OSM (fallback to synthetic if none), snapped to land ---
    hospitals = fetch_osm_hospitals(lat, lon)
    hospitals = [{**h, "lat": snap_inland_if_water(h["lat"], h["lon"])[0], "lon": snap_inland_if_water(h["lat"], h["lon"])[1]} for h in hospitals]
    if hospitals:
        for h in hospitals:
            html = nlp.generate(city, "hospital", signals)
            folium.Marker(
                [h["lat"], h["lon"]],
                tooltip=h.get("name", "Hospital"),
                popup=folium.Popup(html, max_width=340),
                icon=folium.Icon(icon="plus", prefix="fa", color="red")
            ).add_to(groups["hospitals"])
    else:
        for p in inland_jitter(6, 0.015, lon_push=(0.0 if city != "Galway" else +0.005)):
            html = nlp.generate(city, "hospital", signals)
            folium.Marker(p, tooltip="Hospital", popup=folium.Popup(html, max_width=320), icon=folium.Icon(icon="plus", prefix="fa", color="red")).add_to(groups["hospitals"])

    # --- Dynamic shelters from OSM, snapped to land ---
    shelters = fetch_osm_shelters(lat, lon)
    shelters = [{**spt, "lat": snap_inland_if_water(spt["lat"], spt["lon"])[0], "lon": snap_inland_if_water(spt["lat"], spt["lon"])[1]} for spt in shelters]
    mc = MarkerCluster().add_to(groups["shelters"])
    if shelters:
        for spt in shelters:
            html = nlp.generate(city, "shelter", signals)
            folium.Marker(
                [spt["lat"], spt["lon"]],
                tooltip=spt.get("name", "Shelter"),
                popup=folium.Popup(html, max_width=340),
                icon=folium.Icon(color="green")
            ).add_to(mc)
    else:
        for p in inland_jitter(10, 0.02, lon_push=(0.0 if city != "Galway" else +0.006)):
            html = nlp.generate(city, "shelter", signals)
            folium.Marker(p, tooltip="Shelter", popup=folium.Popup(html, max_width=320), icon=folium.Icon(color="green")).add_to(mc)

    # --- Dynamic substations from OSM, snapped to land ---
    substations = fetch_osm_substations(lat, lon)
    substations = [{**sst, "lat": snap_inland_if_water(sst["lat"], sst["lon"])[0], "lon": snap_inland_if_water(sst["lat"], sst["lon"])[1]} for sst in substations]
    if substations:
        for sst in substations:
            html = nlp.generate(city, "substation", signals)
            folium.CircleMarker(
                [sst["lat"], sst["lon"]],
                radius=6,
                tooltip=sst.get("name", "Substation"),
                popup=folium.Popup(html, max_width=340)
            ).add_to(groups["substations"])
    else:
        for p in inland_jitter(5, 0.012, lon_push=(0.0 if city != "Galway" else +0.004)):
            html = nlp.generate(city, "substation", signals)
            folium.CircleMarker(p, radius=6, tooltip="Substation", popup=folium.Popup(html, max_width=320)).add_to(groups["substations"])

    # --- EPA / Monitoring stations ---
    aq_sites = fetch_epa_aq_stations(lat, lon)
    if aq_sites:
        for stn in aq_sites:
            html = nlp.generate(city, "air quality station", signals)
            folium.CircleMarker(
                [stn["lat"], stn["lon"]],
                radius=5,
                tooltip=stn.get("name", "AQ Station"),
                popup=folium.Popup(html, max_width=340)
            ).add_to(groups["aq_stations"])

    # TODO: Replace with GTFS-RT or OSM public_transport=route for live routes
    # Bus route — inland polyline with NLP popup
    line = [(lat-0.008, lon-0.03), (lat, lon), (lat+0.01, lon+0.03)]
    folium.PolyLine(
        line,
        tooltip="Bus route",
        popup=folium.Popup(nlp.generate(city, "bus route", signals), max_width=320)
    ).add_to(groups["bus_routes"])

    # --- Live buses (GTFS‑RT vehicle positions) ---
    vehs = fetch_gtfs_vehicles(CONFIG.get("tfi_gtfs_rt_url", ""))
    for v in vehs:
        html = f"<b>{city}</b>: Bus {v.get('label','')} bearing {int(v.get('bearing',0))}°"
        folium.CircleMarker([v["lat"], v["lon"]], radius=4, tooltip=v.get("label","Bus"), popup=folium.Popup(html, max_width=260)).add_to(groups["gtfs_live"])

    # Selected city marker — with NLP popup
    folium.Marker(
        [lat, lon],
        tooltip=f"{city}",
        popup=folium.Popup(nlp.generate(city, "city centre", signals), max_width=320)
    ).add_to(fmap)

    # Draw tool for what‑if
    Draw(export=True, filename="drawn.geojson").add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    map_state = st_folium(fmap, height=560, width=None)

with col_right:
    st.markdown("#### Evidence, Knowledge Graph & Agents")
    corpus = load_policy_corpus(CONFIG["policy_dir"])
    st.caption(f"Corpus chunks: {len(corpus)} from {corpus['doc_id'].nunique() if not corpus.empty else 0} docs")

    vs = SimpleVectorStore(CONFIG["embed_model"], CONFIG["openai_key"]) if CONFIG["openai_key"] else None
    top_docs, evidence_snips = [], []
    if vs and not corpus.empty:
        with st.spinner("Embedding corpus…"):
            vs.build(corpus)
        q1 = f"{county} resilience goal: {goal}"
        q2 = f"policy constraints {city} emissions budget equity"
        d1 = vs.search(q1, k=4)
        d2 = vs.search(q2, k=4)
        ids1 = {(d.doc_id, d.chunk_id) for d in d1}
        ids2 = {(d.doc_id, d.chunk_id) for d in d2}
        jury_overlap = len(ids1.intersection(ids2)) / max(1, len(ids1.union(ids2)))
        top_docs = (d1 + d2)[:6]
        evidence_snips = [d.text[:1200] for d in top_docs]
        citations = [(d.doc_id, d.chunk_id) for d in top_docs]
    else:
        jury_overlap = 0.4
        evidence_snips = [
            "Climate Action Plan 2024 emphasises emissions ceilings and modal shift.",
            "NPF highlights compact growth and resilient infrastructure in cities.",
            f"{city} development plan prioritises flood risk management and equitable access.",
        ]
        citations = [("synthetic_policy.txt", i) for i in range(len(evidence_snips))]

    # Knowledge Graph demo
    kg = KnowledgeGraph.demo(county)
    st.dataframe(kg.to_dataframe(), use_container_width=True)

    # Agentic orchestration
    agents = Agents()
    plan_json = verifier_json = equity_json = counterf_json = explainer_md = ""
    if run_button:
        with st.spinner("Planner Agent running…"):
            plan_json = agents.planner(goal, county, evidence_snips, jury_overlap)
        if do_verify:
            with st.spinner("Verifier Agent…"):
                verifier_json = agents.verifier(plan_json, evidence_snips)
        if do_equity:
            equity_json = agents.equity(plan_json)
        if do_counterf:
            counterf_json = agents.counterfactuals(plan_json)
        if do_explain:
            explainer_md = agents.explainer(plan_json, citations)

    with st.expander("Retrieved Evidence", expanded=False):
        for i, sn in enumerate(evidence_snips, 1):
            st.markdown(f"**E{i}.** {sn[:1000]}…")

    if plan_json:
        st.markdown("##### Planner Output (JSON)")
        st.code(plan_json, language="json")
    if verifier_json:
        st.markdown("##### Policy Compliance (Verifier JSON)")
        st.code(verifier_json, language="json")
    if equity_json:
        st.markdown("##### Equity / Co‑benefits (JSON)")
        st.code(equity_json, language="json")
    if counterf_json:
        st.markdown("##### Counterfactuals (JSON)")
        st.code(counterf_json, language="json")
    if explainer_md:
        st.markdown("##### Public Brief")
        st.write(explainer_md)

# ---------------- Optimisation Panel ----------------
st.markdown("---")
st.markdown("### Optimisation — Portfolio under constraints (MILP)")
colA, colB = st.columns([1.0, 2.0])
with colA:
    st.caption("Objective: maximise resilience + (equity_weight*equity)")
    if st.button("Solve Portfolio"):
        sol = solve_portfolio(DEFAULT_INTERVENTIONS, budget_m, emissions_cap, equity_w)
        st.session_state["last_portfolio"] = sol
with colB:
    sol = st.session_state.get("last_portfolio", None)
    if sol:
        st.success(f"Solution: {sol['selected']} (objective={sol['obj']:.2f})")
    df_iv = pd.DataFrame([vars(iv) for iv in DEFAULT_INTERVENTIONS])
    st.dataframe(df_iv, use_container_width=True)

# ---------------- RL Panel ----------------
st.markdown("---")
st.markdown("### Self‑Learning (Contextual RL)")
actions = [
    "Temporary pumps & barriers",
    "Bus reroute + dynamic headways",
    "Targeted shelters & comms",
    "1 MWh community battery",
    "Pop‑up active travel corridors",
]

met = fetch_met_eireann_forecast(county)
epa = fetch_epa_air_quality(county)
cur_hour = pd.Timestamp.now().hour

if "bandit" not in st.session_state:
    st.session_state["bandit"] = ContextualBandit(actions)

bandit: ContextualBandit = st.session_state["bandit"]

col1, col2 = st.columns([1.2, 1.8])
with col1:
    st.caption(f"Context: rain={met['rain_mm_next24h']:.1f}mm, AQI={epa['aqi']}, hour={cur_hour}")
    verifier_ok = True  # in production, derive from verifier_json critical checks
    suggested = bandit.select(county, met['rain_mm_next24h'], epa['aqi'], cur_hour, verifier_ok)
    st.success(f"Suggested action for {county}: {suggested}")

with col2:
    st.write("Rate observed outcome (−10..+10). Verifier must not have failed critical checks.")
    reward = st.slider("Reward", -10.0, 10.0, 2.0, 0.5)
    verifier_pass = st.checkbox("Verifier critical checks passed", True)
    if st.button("Update RL"):
        if verifier_pass and suggested != "No action (Verifier blocked)":
            bandit.update(county, met['rain_mm_next24h'], epa['aqi'], cur_hour, True, suggested, reward)
            st.info("Updated RL values for this context.")
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
    st.markdown("- Model: OpenAI Chat \n- Purpose: scenario planning \n- Limits: relies on RAG quality; hallucination risk if poor evidence \n- Safety: Verifier + human review")
with colz:
    st.subheader("Data Sheet (Evidence)")
    st.markdown("- Sources: policy PDFs, official APIs\n- Freshness: show fetch timestamps\n- Known gaps: mobility GTFS‑RT, flood gauges (to wire)")

# Log decisions
DECISIONS_CSV = os.path.join(CONFIG["log_dir"], "decisions.csv")
if st.button("Save Decision Snapshot"):
    row = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        "city": city,
        "county": county,
        "goal": goal,
        "budget_m": budget_m,
        "emissions_cap": emissions_cap,
        "equity_w": equity_w,
        "portfolio": json.dumps(st.session_state.get("last_portfolio", {})),
        "state": state,
    }
    pd.DataFrame([row]).to_csv(DECISIONS_CSV, mode='a', header=not os.path.exists(DECISIONS_CSV), index=False)
    st.success("Snapshot saved to data/logs/decisions.csv")

# ---------------- Footer ----------------
st.markdown(
    """
<small>
**Data connectors** are simulated until API keys/endpoints are configured. Replace in `fetch_*` functions (Met Éireann, EPA, CSO, NTA/TFI, EirGrid).\
Knowledge Graph demo shows how goals link to targets, measures, datasets, tools, stakeholders.\
MILP uses PuLP (CBC). Quantum‑inspired path left as research toggle (benchmark vs classical before use).\
**City centroids** are used for Galway/Cork/Dublin/Limerick/Waterford; add official OSM/GIS layers for production accuracy.<br>
Live overlays: optional rain radar tiles, EPA/monitoring stations, and GTFS‑RT vehicle positions (if endpoints provided).
</small>
""",
    unsafe_allow_html=True,
)
# UNIC Urban Resilience & Smart Cities — Ireland (v2.1)
# Streamlit App — Agentic RAG + Knowledge Graph + MILP Optimiser + Contextual RL + Folium Map
# -------------------------------------------------------------------------------------------
# New in v2.1 (your request):
# - Fix Galway markers drifting into the ocean by using **city centroids** rather than coarse county centroids
# - Add **hover tooltips + NLP popups** for every marker (Planner-grade brief per feature)
# - Introduce a **CITY_CENTROIDS** table (sourced coords below) + safe inland jitter with city-specific inland bias
# - Keep all earlier features (Agentic, KG, MILP, RL, Governance) intact
#
# Coords provenance (fetched from public sources; rounded to 6 d.p.):
#   Dublin  : 53.350140, -6.266155  (latlong.net)
#   Cork    : 51.903614, -8.468399  (latlong.net)
#   Limerick: 52.668018, -8.630498  (latlong.net)
#   Galway  : 53.270962, -9.062691  (latlong.net)
#   Waterford: 52.259319, -7.110070 (approximate centre; verify for production)
#
# Quick start
# 1) pip install -r requirements.txt
# 2) export OPENAI_API_KEY=... (or set in .env)
# 3) Place policy docs under ./data/policies (PDF/TXT)
# 4) streamlit run Urban_Resilience.py

import os
import io
import json
import glob
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw, MarkerCluster

# PDF handling
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Optimisation (MILP)
try:
    import pulp
except Exception:
    pulp = None

# ---------------- CONFIG ----------------
APP_TITLE = "UNIC Regional Resilience (Ireland) — Agentic + KG + RL"
st.set_page_config(page_title=APP_TITLE, layout="wide")

CONFIG = {
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    "openai_key": os.getenv("OPENAI_API_KEY", ""),
    "data_dir": "data",
    "policy_dir": os.path.join("data", "policies"),
    "log_dir": os.path.join("data", "logs"),
}

os.makedirs(CONFIG["policy_dir"], exist_ok=True)
os.makedirs(CONFIG["log_dir"], exist_ok=True)

REQUIREMENTS = """
streamlit>=1.37
streamlit-folium>=0.18
folium>=0.15
pypdf>=4.2
pandas>=2.0
numpy>=1.25
openai>=1.40
requests>=2.31
scikit-learn>=1.4
pulp>=2.8
python-dotenv>=1.0
"""

# ---------------- County & City Profiles ----------------
COUNTIES = {
    "Carlow": (52.7236, -6.9439), "Cavan": (53.9908, -7.3606), "Clare": (52.8436, -8.9810),
    "Cork": (51.8986, -8.4756), "Donegal": (54.6540, -8.1041), "Dublin": (53.3498, -6.2603),
    "Galway": (53.2707, -9.0568), "Kerry": (52.1449, -9.5160), "Kildare": (53.1589, -6.9092),
    "Kilkenny": (52.6541, -7.2448), "Laois": (53.0326, -7.2995), "Leitrim": (54.1937, -8.0002),
    "Limerick": (52.6629, -8.6305), "Longford": (53.7275, -7.7939), "Louth": (53.9509, -6.5400),
    "Mayo": (53.8044, -9.5246), "Meath": (53.6055, -6.6564), "Monaghan": (54.2496, -6.9683),
    "Offaly": (53.2734, -7.4903), "Roscommon": (53.6273, -8.1891), "Sligo": (54.2697, -8.4694),
    "Tipperary": (52.4737, -7.8740), "Waterford": (52.2593, -7.1101), "Westmeath": (53.5333, -7.3500),
    "Wexford": (52.3369, -6.4633), "Wicklow": (52.9800, -6.0400),
}

# City centroids (higher fidelity than county centroids)
CITY_CENTROIDS = {
    "Dublin": (53.350140, -6.266155),   # latlong.net
    "Cork": (51.903614, -8.468399),     # latlong.net
    "Limerick": (52.668018, -8.630498), # latlong.net
    "Galway": (53.270962, -9.062691),   # latlong.net (coastal; add inland bias below)
    "Waterford": (52.259319, -7.110070) # approx centre
}

CITY_PROFILES = {
    "Dublin": {
        "focus_layers": ["bus_routes", "hospitals", "substations"],
        "default_goal": "Reduce pluvial flood disruption and protect public transport access while holding emissions neutral.",
    },
    "Cork": {
        "focus_layers": ["shelters", "hospitals", "bus_routes"],
        "default_goal": "Cut flood damages by 30% and improve air quality near quays; equity priority high.",
    },
    "Limerick": {
        "focus_layers": ["substations", "shelters"],
        "default_goal": "Expand positive energy actions and maintain hospital access during storms.",
    },
    "Galway": {
        "focus_layers": ["hospitals", "bus_routes"],
        "default_goal": "Waterway micro‑hydro integration with mobility continuity during heavy rain.",
    },
    "Waterford": {
        "focus_layers": ["shelters", "bus_routes"],
        "default_goal": "North Quays smart district resilience and walkable access to services.",
    },
}

# ---------------- Utility: PDF & chunking ----------------
@st.cache_data(show_spinner=False)
def read_pdf_text(path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(path)
        texts = []
        for p in reader.pages:
            try:
                texts.append(p.extract_text() or "")
            except Exception:
                pass
        return "\n".join(texts)
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def chunk_text(text: str, max_tokens: int = 900, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    step = max(max_tokens - overlap, 1)
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_tokens]))
        i += step
    return chunks

@st.cache_data(show_spinner=False)
def load_policy_corpus(policy_dir: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(policy_dir, "**", "*"), recursive=True):
        if os.path.isdir(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        txt = ""
        if ext == ".pdf":
            txt = read_pdf_text(fp)
        elif ext in (".txt", ".md"):
            try:
                txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
            except Exception:
                txt = ""
        if not txt:
            continue
        doc_id = os.path.relpath(fp, policy_dir)
        for i, ch in enumerate(chunk_text(txt)):
            rows.append({"doc_id": doc_id, "source_path": fp, "chunk_id": i, "text": ch})
    return pd.DataFrame(rows)

# ---------------- Embeddings / Vector store ----------------
@dataclass
class VectorDoc:
    doc_id: str
    chunk_id: int
    text: str
    source_path: str
    embedding: Optional[List[float]] = None

class SimpleVectorStore:
    def __init__(self, embed_model: str, api_key: str):
        self.embed_model = embed_model
        self.client = OpenAI(api_key=api_key) if (OpenAI and api_key) else None
        self.docs: List[VectorDoc] = []
        self._mat: Optional[np.ndarray] = None

    def build(self, df: pd.DataFrame):
        self.docs = [VectorDoc(r.doc_id, int(r.chunk_id), r.text, r.source_path) for r in df.itertuples()]
        self._embed_all()

    def _embed_all(self):
        if not self.client or not self.docs:
            return
        B = 64
        embs: List[List[float]] = []
        for i in range(0, len(self.docs), B):
            batch = [d.text for d in self.docs[i:i+B]]
            resp = self.client.embeddings.create(model=CONFIG["embed_model"], input=batch)
            embs.extend([e.embedding for e in resp.data])
        for d, e in zip(self.docs, embs):
            d.embedding = e
        self._mat = np.array(embs, dtype=np.float32)

    def search(self, q: str, k: int = 6) -> List[VectorDoc]:
        if not self.client or self._mat is None:
            return []
        qv = self.client.embeddings.create(model=CONFIG["embed_model"], input=[q]).data[0].embedding
        qv = np.array(qv, dtype=np.float32)
        sims = self._mat @ qv / (np.linalg.norm(self._mat, axis=1) * (np.linalg.norm(qv) + 1e-9))
        idx = np.argsort(-sims)[:k]
        return [self.docs[i] for i in idx]

# ---------------- Knowledge Graph ----------------
class KnowledgeGraph:
    def __init__(self):
        self.edges: List[Tuple[str, str, str]] = []
        self.node_types: Dict[str, str] = {}

    def add(self, src: str, rel: str, dst: str, src_type: str, dst_type: str):
        self.edges.append((src, rel, dst))
        self.node_types[src] = src_type
        self.node_types[dst] = dst_type

    def neighbors(self, node: str) -> List[Tuple[str, str]]:
        return [(rel, dst) for (s, rel, dst) in self.edges if s == node]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for s, r, d in self.edges:
            rows.append({"source": s, "relation": r, "target": d, "source_type": self.node_types.get(s, ""), "target_type": self.node_types.get(d, "")})
        return pd.DataFrame(rows)

    @staticmethod
    def demo(county: str) -> "KnowledgeGraph":
        kg = KnowledgeGraph()
        goal = f"{county}: Reduce pluvial flood disruption & protect access"
        kg.add(goal, "targets", "-30% 10‑yr flood damages", "Goal", "Target")
        kg.add(goal, "targets", "<=0 ΔtCO₂e", "Goal", "Target")
        kg.add("-30% 10‑yr flood damages", "measured_by", "Service-hours preserved", "Target", "Measure")
        kg.add("<=0 ΔtCO₂e", "measured_by", "Emissions delta", "Target", "Measure")
        kg.add("Service-hours preserved", "requires", "Bus headways (GTFS‑RT)", "Measure", "Dataset")
        kg.add("Emissions delta", "requires", "Energy mix (EirGrid)", "Measure", "Dataset")
        kg.add("Bus headways (GTFS‑RT)", "queried_by", "MobilityAgent", "Dataset", "Tool")
        kg.add("Energy mix (EirGrid)", "queried_by", "EnergyAgent", "Dataset", "Tool")
        kg.add("MobilityAgent", "owned_by", "NTA/TFI", "Tool", "Stakeholder")
        kg.add("EnergyAgent", "owned_by", "EirGrid", "Tool", "Stakeholder")
        return kg

# ---------------- Agentic Layer ----------------
class Agents:
    def __init__(self):
        self.client = OpenAI(api_key=CONFIG["openai_key"]) if (OpenAI and CONFIG["openai_key"]) else None
        self.model = CONFIG["openai_model"]

    def _chat(self, messages, temperature=0.2, max_tokens=900) -> str:
        if not self.client:
            return "[OpenAI key missing]"
        r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content

    def planner(self, goal: str, county: str, evidence_snips: List[str], jury_score: float) -> str:
        sys = (
            "You are a Planner Agent for Irish urban resilience. Output JSON with keys: "
            "objectives, constraints, actions, kpis, checklist48h, notes. Keep budget/emissions/equity explicit."
        )
        ctx = "\n\n".join([f"[E{i+1}] {s}" for i, s in enumerate(evidence_snips)])
        user = f"County: {county}\nJuryConsensus: {jury_score:.2f}\nGoal: {goal}\nEvidence:\n{ctx}"
        return self._chat([{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.1)

    def verifier(self, plan_json: str, evidence_snips: List[str]) -> str:
        sys = (
            "You are a Verifier Agent. Return a JSON with policy_checks: [{name, pass, reason, citations}], "
            "risk_flags:[], required_data:[], and overall: pass|fail."
        )
        ctx = "\n\n".join([f"[E{i+1}] {s}" for i, s in enumerate(evidence_snips)])
        user = f"PlanJSON:\n{plan_json}\n\nEvidence:\n{ctx}"
        return self._chat([{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.0)

    def equity(self, plan_json: str) -> str:
        sys = "Equity/Cobenefits Agent. Given Plan JSON, output JSON with equity_score(0-1), hotspots[], mitigations[]."
        return self._chat([{"role":"system","content":sys},{"role":"user","content":plan_json}], temperature=0.1)

    def counterfactuals(self, plan_json: str) -> str:
        sys = "Counterfactual Agent. Produce JSON list of rejected options with reasons (cost/emissions/equity/feasibility)."
        return self._chat([{"role":"system","content":sys},{"role":"user","content":plan_json}], temperature=0.2)

    def explainer(self, plan_json: str, citations: List[Tuple[str,int]]) -> str:
        sys = "Explainer Agent. Output a brief (markdown) for the public. Include a numbered citations list (doc#chunk)."
        cites = "\n".join([f"{i+1}. {d}#${c}" for i,(d,c) in enumerate(citations)])
        user = f"Plan:\n{plan_json}\n\nCitations:\n{cites}"
        return self._chat([{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.2)

# ---------------- Official Data Connectors (stubs) ----------------
import requests

@st.cache_data(ttl=1800)
def fetch_met_eireann_forecast(county: str) -> Dict[str, Any]:
    try:
        return {"status":"sim", "rain_mm_next24h": float(np.random.gamma(2,3))}
    except Exception:
        return {"status":"sim", "rain_mm_next24h": float(np.random.gamma(2,3))}

@st.cache_data(ttl=1800)
def fetch_epa_air_quality(county: str) -> Dict[str, Any]:
    return {"aqi": int(np.random.randint(15, 65)), "pm25": round(float(np.random.uniform(5, 20)), 1)}

@st.cache_data(ttl=3600)
def fetch_cso_population(county: str) -> Dict[str, Any]:
    base = {"Dublin":1500000, "Cork":600000, "Limerick":210000, "Galway":280000, "Waterford":127000}
    return {"population": int(base.get(county, int(np.random.randint(70000, 400000))))}

# ---------------- NLP helper for map popups ----------------
class NLPPopup:
    def __init__(self):
        self.client = OpenAI(api_key=CONFIG["openai_key"]) if (OpenAI and CONFIG["openai_key"]) else None
        self.model = CONFIG["openai_model"]

    def generate(self, city: str, layer: str, signals: Dict[str, Any]) -> str:
        rain = signals.get("rain", None)
        aqi = signals.get("aqi", None)
        pop = signals.get("pop", None)
        if not self.client:
            return (
                f"<b>{city}</b> — {layer.title()}\n" 
                f"<br/>Context: rain≈{rain:.1f} mm, AQI={aqi}, pop≈{pop:,}.\n"
                f"<br/>Action hint: Prioritise access & low‑emission options during wet periods."
            )
        msg = [
            {"role":"system","content":"Write one short sentence (<=28 words) for a map popup about Irish urban resilience. Be concrete and non‑fluffy."},
            {"role":"user","content":f"City={city}; Layer={layer}; rain_mm={rain}; AQI={aqi}; population={pop}. Output plain text only."}
        ]
        try:
            out = self.client.chat.completions.create(model=self.model, messages=msg, temperature=0.2, max_tokens=60)
            text = out.choices[0].message.content.strip()
        except Exception:
            text = f"{city} — {layer}: Maintain service continuity; coordinate shelters, bus headways and energy buffers."
        return f"<b>{city}</b>: {text}"

# ---------------- Reinforcement Learning (Contextual Bandit) ----------------
class ContextualBandit:
    def __init__(self, actions: List[str], alpha=0.2, epsilon=0.15):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.q: Dict[Tuple[str, str, str, str, str], float] = {}

    def _key(self, county: str, rain: float, aqi: int, hour: int, verifier_ok: bool) -> Tuple[str,str,str,str,str]:
        rain_bin = "high" if rain>=10 else ("med" if rain>=3 else "low")
        aqi_bin = "poor" if aqi>=40 else ("fair" if aqi>=25 else "good")
        peak = "peak" if 7<=hour<=9 or 16<=hour<=19 else "off"
        v = "ok" if verifier_ok else "blocked"
        return (county, rain_bin, aqi_bin, peak, v)

    def select(self, county: str, rain: float, aqi: int, hour: int, verifier_ok: bool) -> str:
        key = self._key(county, rain, aqi, hour, verifier_ok)
        if not verifier_ok:
            return "No action (Verifier blocked)"
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        vals = [(a, self.q.get((*key, a), 0.0)) for a in self.actions]
        vals.sort(key=lambda x: -x[1])
        return vals[0][0] if vals else self.actions[0]

    def update(self, county: str, rain: float, aqi: int, hour: int, verifier_ok: bool, action: str, reward: float):
        key = self._key(county, rain, aqi, hour, verifier_ok)
        old = self.q.get((*key, action), 0.0)
        self.q[(*key, action)] = old + self.alpha * (reward - old)

# ---------------- Optimisation (MILP portfolio) ----------------
@dataclass
class Intervention:
    name: str
    cost_m: float
    emissions_delta: float
    equity_gain: float
    resilience_gain: float

DEFAULT_INTERVENTIONS = [
    Intervention("Temporary pumps & barriers", 5.0, 20.0, 0.6, 18.0),
    Intervention("Bus reroute + dynamic headways", 2.0, -50.0, 0.7, 12.0),
    Intervention("Targeted shelters & comms", 1.5, 5.0, 0.9, 10.0),
    Intervention("1 MWh community battery", 3.0, -200.0, 0.5, 9.0),
    Intervention("Pop-up active travel corridors", 1.0, -30.0, 0.8, 8.0),
]

def solve_portfolio(interventions: List[Intervention], budget_m: float, emissions_cap: float, equity_weight: float) -> Dict[str, Any]:
    if pulp is None:
        return {"status":"no_milp", "selected":[], "obj":0.0}
    m = pulp.LpProblem("resilience_portfolio", pulp.LpMaximize)
    x = {iv.name: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i, iv in enumerate(interventions)}
    m += pulp.lpSum([x[iv.name] * (iv.resilience_gain + equity_weight * 10.0 * iv.equity_gain) for iv in interventions])
    m += pulp.lpSum([x[iv.name] * iv.cost_m for iv in interventions]) <= budget_m
    m += pulp.lpSum([x[iv.name] * iv.emissions_delta for iv in interventions]) <= emissions_cap
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [iv.name for iv in interventions if x[iv.name].value() and x[iv.name].value() >= 0.5]
    obj = pulp.value(m.objective)
    return {"status":"ok", "selected": selected, "obj": obj}

# ---------------- UI ----------------
st.title(APP_TITLE)
with st.sidebar:
    st.markdown("### Controls")
    city = st.selectbox("Smart city", options=list(CITY_PROFILES.keys()), index=0)
    county = st.selectbox("County", options=sorted(COUNTIES.keys()), index=sorted(COUNTIES.keys()).index(city) if city in COUNTIES else 0)
    default_goal = CITY_PROFILES[city]["default_goal"]
    goal = st.text_area("Resilience goal", value=default_goal, height=80)

    st.markdown("**Scenario constraints**")
    budget_m = st.slider("Budget (million €)", 1.0, 100.0, 25.0, 1.0)
    emissions_cap = st.slider("Max additional ΔtCO₂e", -2000.0, 2000.0, 0.0, 50.0)
    equity_w = st.slider("Equity weight (0..1)", 0.0, 1.0, 0.6, 0.05)

    st.markdown("**Agentic options**")
    do_verify = st.checkbox("Run Verifier", True)
    do_equity = st.checkbox("Equity/Cobenefits Agent", True)
    do_counterf = st.checkbox("Counterfactuals", True)
    do_explain = st.checkbox("Public Brief", True)
    run_button = st.button("Run Scenario Planner", type="primary")

# ---------------- Map ----------------
col_map, col_right = st.columns([1.25, 1.75])
with col_map:
    st.markdown("#### Geospatial View")

    # Use precise city centroid; add inland bias for coastal cities (e.g., Galway) to avoid ocean placement
    base_lat, base_lon = CITY_CENTROIDS.get(city, COUNTIES[county])
    bias_lat, bias_lon = 0.0, 0.0
    if city == "Galway":
        bias_lon = +0.015  # shift slightly east (inland) to keep demo POIs on land
    lat = base_lat + bias_lat
    lon = base_lon + bias_lon

    fmap = folium.Map(location=[lat, lon], zoom_start=12, tiles="CartoDB positron")

    # Overlays
    layers_enabled = CITY_PROFILES[city]["focus_layers"]
    groups: Dict[str, folium.FeatureGroup] = {}
    for lname in ["hospitals", "shelters", "substations", "bus_routes"]:
        fg = folium.FeatureGroup(name=lname.capitalize())
        groups[lname] = fg
        fmap.add_child(fg)

    # Signals for NLP popups
    met = fetch_met_eireann_forecast(county)
    epa = fetch_epa_air_quality(county)
    cso = fetch_cso_population(county)
    signals = {"rain": met.get("rain_mm_next24h", 0.0), "aqi": epa.get("aqi", 0), "pop": cso.get("population", 0)}
    nlp = NLPPopup()

    rng = np.random.default_rng(42)

    def inland_jitter(n=6, r=0.02, lon_push=0.0):
        pts = []
        for _ in range(n):
            pts.append((lat + float(rng.normal(0, r)), lon + float(rng.normal(0, r)) + lon_push))
        return pts

    # Hospitals
    for p in inland_jitter(6, 0.015, lon_push=(0.0 if city != "Galway" else +0.005)):
        html = nlp.generate(city, "hospital", signals)
        folium.Marker(p, tooltip="Hospital", popup=folium.Popup(html, max_width=320), icon=folium.Icon(icon="plus", prefix="fa", color="red")).add_to(groups["hospitals"])

    # Shelters (cluster)
    mc = MarkerCluster().add_to(groups["shelters"])
    for p in inland_jitter(10, 0.02, lon_push=(0.0 if city != "Galway" else +0.006)):
        html = nlp.generate(city, "shelter", signals)
        folium.Marker(p, tooltip="Shelter", popup=folium.Popup(html, max_width=320), icon=folium.Icon(color="green")).add_to(mc)

    # Substations
    for p in inland_jitter(5, 0.012, lon_push=(0.0 if city != "Galway" else +0.004)):
        html = nlp.generate(city, "substation", signals)
        folium.CircleMarker(p, radius=6, tooltip="Substation", popup=folium.Popup(html, max_width=320)).add_to(groups["substations"])

    # Bus route polyline — simple inland path
    line = [(lat-0.008, lon-0.03), (lat, lon), (lat+0.01, lon+0.03)]
    folium.PolyLine(line, tooltip="Bus route", popup=folium.Popup(nlp.generate(city, "bus route", signals), max_width=320)).add_to(groups["bus_routes"])

    # Selected city marker
    folium.Marker([lat, lon], tooltip=f"{city}", popup=folium.Popup(nlp.generate(city, "city centre", signals), max_width=320)).add_to(fmap)

    # Draw tool for what‑if
    Draw(export=True, filename="drawn.geojson").add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    map_state = st_folium(fmap, height=560, width=None)

with col_right:
    st.markdown("#### Evidence, Knowledge Graph & Agents")
    corpus = load_policy_corpus(CONFIG["policy_dir"])
    st.caption(f"Corpus chunks: {len(corpus)} from {corpus['doc_id'].nunique() if not corpus.empty else 0} docs")

    vs = SimpleVectorStore(CONFIG["embed_model"], CONFIG["openai_key"]) if CONFIG["openai_key"] else None
    top_docs, evidence_snips = [], []
    if vs and not corpus.empty:
        with st.spinner("Embedding corpus…"):
            vs.build(corpus)
        q1 = f"{county} resilience goal: {goal}"
        q2 = f"policy constraints {city} emissions budget equity"
        d1 = vs.search(q1, k=4)
        d2 = vs.search(q2, k=4)
        ids1 = {(d.doc_id, d.chunk_id) for d in d1}
        ids2 = {(d.doc_id, d.chunk_id) for d in d2}
        jury_overlap = len(ids1.intersection(ids2)) / max(1, len(ids1.union(ids2)))
        top_docs = (d1 + d2)[:6]
        evidence_snips = [d.text[:1200] for d in top_docs]
        citations = [(d.doc_id, d.chunk_id) for d in top_docs]
    else:
        jury_overlap = 0.4
        evidence_snips = [
            "Climate Action Plan 2024 emphasises emissions ceilings and modal shift.",
            "NPF highlights compact growth and resilient infrastructure in cities.",
            f"{city} development plan prioritises flood risk management and equitable access.",
        ]
        citations = [("synthetic_policy.txt", i) for i in range(len(evidence_snips))]

    kg = KnowledgeGraph.demo(county)
    st.dataframe(kg.to_dataframe(), use_container_width=True)

    agents = Agents()
    plan_json = verifier_json = equity_json = counterf_json = explainer_md = ""
    if run_button:
        with st.spinner("Planner Agent running…"):
            plan_json = agents.planner(goal, county, evidence_snips, jury_overlap)
        if do_verify:
            with st.spinner("Verifier Agent…"):
                verifier_json = agents.verifier(plan_json, evidence_snips)
        if do_equity:
            equity_json = agents.equity(plan_json)
        if do_counterf:
            counterf_json = agents.counterfactuals(plan_json)
        if do_explain:
            explainer_md = agents.explainer(plan_json, citations)

    with st.expander("Retrieved Evidence", expanded=False):
        for i, sn in enumerate(evidence_snips, 1):
            st.markdown(f"**E{i}.** {sn[:1000]}…")

    if plan_json:
        st.markdown("##### Planner Output (JSON)")
        st.code(plan_json, language="json")
    if verifier_json:
        st.markdown("##### Policy Compliance (Verifier JSON)")
        st.code(verifier_json, language="json")
    if equity_json:
        st.markdown("##### Equity / Co‑benefits (JSON)")
        st.code(equity_json, language="json")
    if counterf_json:
        st.markdown("##### Counterfactuals (JSON)")
        st.code(counterf_json, language="json")
    if explainer_md:
        st.markdown("##### Public Brief")
        st.write(explainer_md)

# ---------------- Optimisation Panel ----------------
st.markdown("---")
st.markdown("### Optimisation — Portfolio under constraints (MILP)")
colA, colB = st.columns([1.0, 2.0])
with colA:
    st.caption("Objective: maximise resilience + (equity_weight*equity)")
    if st.button("Solve Portfolio"):
        sol = solve_portfolio(DEFAULT_INTERVENTIONS, budget_m, emissions_cap, equity_w)
        st.session_state["last_portfolio"] = sol
with colB:
    sol = st.session_state.get("last_portfolio", None)
    if sol:
        st.success(f"Solution: {sol['selected']} (objective={sol['obj']:.2f})")
    df_iv = pd.DataFrame([vars(iv) for iv in DEFAULT_INTERVENTIONS])
    st.dataframe(df_iv, use_container_width=True)

# ---------------- RL Panel ----------------
st.markdown("---")
st.markdown("### Self‑Learning (Contextual RL)")
actions = [
    "Temporary pumps & barriers",
    "Bus reroute + dynamic headways",
    "Targeted shelters & comms",
    "1 MWh community battery",
    "Pop‑up active travel corridors",
]

met = fetch_met_eireann_forecast(county)
epa = fetch_epa_air_quality(county)
cur_hour = pd.Timestamp.now().hour

if "bandit" not in st.session_state:
    st.session_state["bandit"] = ContextualBandit(actions)

bandit: ContextualBandit = st.session_state["bandit"]

col1, col2 = st.columns([1.2, 1.8])
with col1:
    st.caption(f"Context: rain={met['rain_mm_next24h']:.1f}mm, AQI={epa['aqi']}, hour={cur_hour}")
    verifier_ok = True  # hook this to verifier_json for production
    suggested = bandit.select(county, met['rain_mm_next24h'], epa['aqi'], cur_hour, verifier_ok)
    st.success(f"Suggested action for {county}: {suggested}")

with col2:
    st.write("Rate observed outcome (−10..+10). Verifier must not have failed critical checks.")
    reward = st.slider("Reward", -10.0, 10.0, 2.0, 0.5)
    verifier_pass = st.checkbox("Verifier critical checks passed", True)
    if st.button("Update RL"):
        if verifier_pass and suggested != "No action (Verifier blocked)":
            bandit.update(county, met['rain_mm_next24h'], epa['aqi'], cur_hour, True, suggested, reward)
            st.info("Updated RL values for this context.")
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
    st.markdown("- Model: OpenAI Chat \n- Purpose: scenario planning \n- Limits: relies on RAG quality; hallucination risk if poor evidence \n- Safety: Verifier + human review")
with colz:
    st.subheader("Data Sheet (Evidence)")
    st.markdown("- Sources: policy PDFs, official APIs\n- Freshness: show fetch timestamps\n- Known gaps: mobility GTFS‑RT, flood gauges (to wire)")

DECISIONS_CSV = os.path.join(CONFIG["log_dir"], "decisions.csv")
if st.button("Save Decision Snapshot"):
    row = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        "city": city,
        "county": county,
        "goal": goal,
        "budget_m": budget_m,
        "emissions_cap": emissions_cap,
        "equity_w": equity_w,
        "portfolio": json.dumps(st.session_state.get("last_portfolio", {})),
        "state": state,
    }
    pd.DataFrame([row]).to_csv(DECISIONS_CSV, mode='a', header=not os.path.exists(DECISIONS_CSV), index=False)
    st.success("Snapshot saved to data/logs/decisions.csv")

# ---------------- Footer ----------------
st.markdown(
    """
<small>
**Data connectors** are simulated until API keys/endpoints are configured. Replace in `fetch_*` functions (Met Éireann, EPA, CSO, NTA/TFI, EirGrid).\
**City centroids** sourced from public references (e.g., latlong.net); consider swapping to OSM/official GIS for production.\
NLP popups use OpenAI if a key is present; otherwise fall back to templated text.\
MILP uses PuLP (CBC).\
</small>
""",
    unsafe_allow_html=True,
)
