# --- Prop Finder (Railway-friendly, writes to /tmp) ---
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict
from pathlib import Path
from os import getenv
import pandas as pd
import numpy as np
import json

# storage -> /tmp (works without volumes)
DATA_DIR = Path(getenv("DATA_DIR", "/tmp")).resolve()
HIST_DIR = DATA_DIR / "historical"
HIST_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Prop Finder", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def remove_vig_two_way(p1, p2):
    p1, p2 = float(p1 or 0.0), float(p2 or 0.0)
    s = p1 + p2
    return (0.5, 0.5) if s <= 0 else (p1/s, p2/s)

def remove_vig_three_way(ph, pd, pa):
    ph, pd, pa = float(ph or 0.0), float(pd or 0.0), float(pa or 0.0)
    s = ph + pd + pa
    return (1/3, 1/3, 1/3) if s <= 0 else (ph/s, pd/s, pa/s)

def remove_vig_nway(prob_map: dict):
    s = sum(float(v or 0.0) for v in prob_map.values())
    if s <= 0:
        n = max(1, len(prob_map))
        return {k: 1.0/n for k in prob_map}
    return {k: float(v or 0.0)/s for k, v in prob_map.items()}

def ev_decimal(d, p):
    d = float(d or 0.0); p = float(p or 0.0)
    return p * (d - 1.0) - (1.0 - p)

def compute_ev_row(r: dict):
    side = str(r.get("side") or "").upper()
    market_class = str(r.get("market_class") or "").upper()
    price = float(r.get("price_bet365") or 0.0)

    # OU
    if market_class == "OU" or side in ["OVER","UNDER"]:
        p_over_raw = float(r.get("baseline_over_prob", 0.0) or 0.0)
        p_under_raw = float(r.get("baseline_under_prob", 0.0) or 0.0)
        p_over, p_under = remove_vig_two_way(p_over_raw, p_under_raw)
        p_fair = p_over if side == "OVER" else p_under
        return ev_decimal(price, p_fair), p_fair

    # YN
    if market_class == "YN" or side in ["YES","NO"]:
        p_yes_raw = float(r.get("baseline_yes_prob", 0.0) or 0.0)
        p_no_raw  = float(r.get("baseline_no_prob", 1.0 - p_yes_raw) or (1.0 - p_yes_raw))
        p_yes, p_no = remove_vig_two_way(p_yes_raw, p_no_raw)
        p_fair = p_yes if side == "YES" else p_no
        return ev_decimal(price, p_fair), p_fair

    # THREE_WAY
    if market_class == "THREE_WAY" or side in ["HOME","DRAW","AWAY"]:
        ph = float(r.get("baseline_home_prob", 0.0) or 0.0)
        pd = float(r.get("baseline_draw_prob", 0.0) or 0.0)
        pa = float(r.get("baseline_away_prob", 0.0) or 0.0)
        ph, pd, pa = remove_vig_three_way(ph, pd, pa)
        mapping = {"HOME": ph, "DRAW": pd, "AWAY": pa}
        p_fair = mapping.get(side, 0.0)
        return ev_decimal(price, p_fair), p_fair

    # N_WAY
    if market_class == "N_WAY":
        probs = r.get("baseline_probs_json") or r.get("probs_json") or "{}"
        if isinstance(probs, str):
            try: probs = json.loads(probs)
            except: probs = {}
        probs = {str(k).upper(): float(v or 0.0) for k, v in probs.items()}
        probs = remove_vig_nway(probs)
        p_fair = probs.get(side, 0.0)
        return ev_decimal(price, p_fair), p_fair

    # BAND
    if market_class == "BAND":
        probs = r.get("band_probs_json") or "{}"
        if isinstance(probs, str):
            try: probs = json.loads(probs)
            except: probs = {}
        probs = {str(k): float(v or 0.0) for k, v in probs.items()}
        probs = remove_vig_nway(probs)
        p_fair = probs.get(r.get("side"), 0.0)
        return ev_decimal(price, p_fair), p_fair

    # JOINT
    if market_class == "JOINT":
        p_joint = float(r.get("joint_prob", 0.0) or 0.0)
        return ev_decimal(price, p_joint), p_joint

    p_fair = float(r.get("model_prob", 0.0) or 0.0)
    return ev_decimal(price, p_fair), p_fair

_CACHE = {"edges": None, "summary": None}
def latest_results(): return _CACHE if _CACHE.get("summary") else None

def _load_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted((data_dir / "historical").glob("*.csv"))
    if not files: return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if "event_datetime" in df.columns:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    return df

def run_backtest(data_dir: Path, edge_threshold: float = 0.02, kelly_fraction: float = 0.5):
    df = _load_csvs(data_dir)
    if df.empty:
        _CACHE["edges"] = pd.DataFrame()
        _CACHE["summary"] = {"n_bets": 0, "roi": 0.0, "pnl": 0.0,
                             "hit_rate": 0.0, "kelly_fraction": kelly_fraction, "by_sport": {}}
        return _CACHE["summary"]

    evs, pfs = [], []
    for _, r in df.iterrows():
        ev, p = compute_ev_row(r); evs.append(ev); pfs.append(p)
    df["p_fair"] = pfs; df["edge"] = evs

    sel = df[df["edge"] >= edge_threshold].copy()
    if sel.empty:
        _CACHE["edges"] = df
        _CACHE["summary"] = {"n_bets": 0, "roi": 0.0, "pnl": 0.0,
                             "hit_rate": 0.0, "kelly_fraction": kelly_fraction, "by_sport": {}}
        return _CACHE["summary"]

    d = sel["price_bet365"].astype(float); p = sel["p_fair"].astype(float)
    kelly_star = (p * d - 1.0) / (d - 1.0)
    kelly_star = kelly_star.clip(lower=0.0, upper=0.5)
    sel["stake"] = kelly_star * kelly_fraction

    won = sel["outcome"].astype(int)
    sel["pnl"] = np.where(won == 1, sel["stake"] * (d - 1.0), -sel["stake"])

    total_staked = sel["stake"].sum()
    total_pnl = sel["pnl"].sum()
    roi = (total_pnl / total_staked) if total_staked > 0 else 0.0
    hit_rate = won.mean()

    by_sport = (
        sel.groupby("sport")
        .agg(n_bets=("sport","size"), pnl=("pnl","sum"), staked=("stake","sum"))
        .assign(roi=lambda x: x["pnl"]/x["staked"]).to_dict(orient="index")
    )

    _CACHE["edges"] = sel.sort_values("edge", ascending=False).reset_index(drop=True)
    _CACHE["summary"] = {
        "n_bets": int(sel.shape[0]), "roi": float(roi), "pnl": float(total_pnl),
        "hit_rate": float(hit_rate), "kelly_fraction": float(kelly_fraction),
        "by_sport": {k: {"n_bets": int(v["n_bets"]), "pnl": float(v["pnl"]),
                         "staked": float(v["staked"]), "roi": float(v["roi"])} for k, v in by_sport.items()}
    }
    return _CACHE["summary"]

class BacktestResponse(BaseModel):
    n_bets: int
    roi: float
    pnl: float
    hit_rate: float
    kelly_fraction: float
    by_sport: Dict[str, Dict[str, float]]

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!doctype html><html lang='sv'><meta charset='utf-8'>
<title>Prop Finder</title><meta name=viewport content='width=device-width,initial-scale=1'>
<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/water.css@2/out/water.css'>
<h1>Prop Finder</h1>
<p>Ladda upp historik (CSV) och kör backtest.</p>
<h3>Ladda upp CSV</h3>
<input id=f type=file accept=.csv>
<button id=u>Ladda upp</button>
<div id=us></div>
<h3>Kör backtest</h3>
<label>Edge-tröskel <input id=e type=number step=0.01 value=0.02></label>
<label>Kelly-fraktion <input id=k type=number step=0.1 value=0.5></label>
<button id=r>Kör</button>
<div id=stats></div>
<h3>Edges</h3>
<label>Sportfilter <input id=s placeholder='FOOTBALL'></label>
<label>Min edge <input id=me type=number step=0.01 value=0.02></label>
<button id=ref>Uppdatera</button>
<table id=t></table>
<script>
const up=document.getElementById('u'), f=document.getElementById('f'), us=document.getElementById('us');
const r=document.getElementById('r'), e=document.getElementById('e'), k=document.getElementById('k'), stats=document.getElementById('stats');
const ref=document.getElementById('ref'), t=document.getElementById('t'), s=document.getElementById('s'), me=document.getElementById('me');

up.onclick=async()=>{
  if(!f.files[0]){us.textContent='Välj en CSV först.';return;}
  const fd=new FormData(); fd.append('file', f.files[0]);
  const res=await fetch('/api/data/upload',{method:'POST',body:fd}); const j=await res.json();
  us.textContent=j.ok?('Uppladdad: '+j.filename):('Fel: '+j.error);
};

r.onclick=async()=>{
  const res=await fetch(`/api/backtest/run?edge_threshold=${e.value||0.02}&kelly_fraction=${k.value||0.5}`,{method:'POST'});
  const j=await res.json();
  stats.innerHTML=`<b>Bets:</b> ${j.n_bets} • <b>ROI:</b> ${(j.roi*100).toFixed(2)}% • <b>Hit:</b> ${(j.hit_rate*100).toFixed(1)}%`;
  load();
};

ref.onclick=()=>load();

async function load(){
  const url=new URL('/api/edges', location.origin);
  if(s.value) url.searchParams.set('sport', s.value);
  url.searchParams.set('min_edge', me.value||0.02); url.searchParams.set('limit', 500);
  const res=await fetch(url); const arr=await res.json();
  t.innerHTML='<tr><th>Sport</th><th>Liga</th><th>Marknad</th><th>Spelare</th><th>Sida</th><th>Line</th><th>Pris</th><th>p_fair</th><th>Edge</th><th>Stake</th><th>PnL</th></tr>';
  arr.forEach(e=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${e.sport||''}</td><td>${e.league||''}</td><td>${e.market_type||''}</td><td>${e.player||''}</td>
    <td>${e.side||''}</td><td>${e.line??''}</td><td>${e.price_bet365??''}</td><td>${(e.p_fair||0).toFixed(3)}</td>
    <td><b>${(e.edge||0).toFixed(3)}</b></td><td>${(e.stake||0).toFixed(3)}</td>
    <td style="color:${(e.pnl||0)>=0?'#057a55':'#e11d48'}">${(e.pnl||0).toFixed(3)}</td>`;
    t.appendChild(tr);
  });
}
</script>
""")

@app.post("/api/data/upload")
async def upload_csv(file: UploadFile = File(...)):
    data = await file.read()
    out = HIST_DIR / file.filename
    out.write_bytes(data)
    try:
        pd.read_csv(out, nrows=2)
    except Exception as e:
        out.unlink(missing_ok=True)
        return {"ok": False, "error": str(e)}
    return {"ok": True, "filename": file.filename}

class BacktestResponse(BaseModel):
    n_bets: int; roi: float; pnl: float; hit_rate: float; kelly_fraction: float
    by_sport: Dict[str, Dict[str, float]]

@app.post("/api/backtest/run", response_model=BacktestResponse)
async def api_backtest_run(edge_threshold: float = 0.02, kelly_fraction: float = 0.5):
    summary = run_backtest(DATA_DIR, edge_threshold=edge_threshold, kelly_fraction=kelly_fraction)
    return BacktestResponse(**summary)

@app.get("/api/edges")
async def api_edges(min_edge: float = 0.02, sport: Optional[str] = None, limit: int = 500):
    res = latest_results()
    if not res or res.get("edges") is None:
        return []
    df = res["edges"].copy()
    if sport:
        df = df[df["sport"].astype(str).str.upper() == sport.upper()]
    df = df[df["edge"] >= min_edge].sort_values("edge", ascending=False).head(limit)
    return df.to_dict(orient="records")

