"""
eco_route_engine.py
--------------------
Master backend engine for eco-routing in Kolkata.

Features:
  - Loads Kolkata road network (fused_roads.geojson)
  - Predicts emission_factor per edge via CarbonFusionNet ML model
    (falls back to physics formula if model not found)
  - Vehicle count fallback: realistic random values by road type
    (used when no camera image is supplied)
  - Renames carbon_cost -> emission_factor throughout
  - 4 routing strategies: shortest, fastest, lowest emission, balanced
  - Saves route_compare.png
  - Returns JSON-serialisable results for the frontend
"""

import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, LineString

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
KOLKATA_GJ = BASE_DIR / "kolkata" / "fused_roads.geojson"
MODEL_PATH = BASE_DIR / "carbon_fusion_net.pt"

# ── Emission factors (g CO2 / km / vehicle) ───────────────────────────────────
EMISSIONS_G_PER_KM = {
    "car": 120, "motorcycle": 72, "bus": 822,
    "truck": 900, "van": 200, "bicycle": 0, "auto": 85,
}

# ── Realistic fallback vehicle counts by road type ────────────────────────────
FALLBACK_VEHICLE_COUNT = {
    "residential":    (15,  80),
    "tertiary":       (40, 140),
    "secondary":      (80, 200),
    "primary":        (100,250),
    "secondary_link": (40, 110),
    "tertiary_link":  (20,  80),
    "living_street":  (5,   40),
    "unclassified":   (20,  90),
    "busway":         (20,  80),
    "trunk":          (120, 280),
    "trunk_link":     (80, 200),
}
DEFAULT_COUNT_RANGE = (30, 130)

# ── Fallback vehicle-type mix ─────────────────────────────────────────────────
VEHICLE_MIX = {
    "residential":  {"car":0.50,"motorcycle":0.25,"bus":0.02,"truck":0.02,"van":0.05,"bicycle":0.08,"auto":0.08},
    "secondary":    {"car":0.40,"motorcycle":0.18,"bus":0.10,"truck":0.12,"van":0.08,"bicycle":0.04,"auto":0.08},
    "tertiary":     {"car":0.45,"motorcycle":0.22,"bus":0.06,"truck":0.06,"van":0.07,"bicycle":0.06,"auto":0.08},
    "primary":      {"car":0.38,"motorcycle":0.15,"bus":0.12,"truck":0.15,"van":0.08,"bicycle":0.03,"auto":0.09},
    "trunk":        {"car":0.35,"motorcycle":0.12,"bus":0.15,"truck":0.20,"van":0.08,"bicycle":0.02,"auto":0.08},
    "busway":       {"car":0.10,"motorcycle":0.05,"bus":0.70,"truck":0.05,"van":0.05,"bicycle":0.02,"auto":0.03},
}
DEFAULT_MIX = {"car":0.45,"motorcycle":0.20,"bus":0.07,"truck":0.08,"van":0.07,"bicycle":0.06,"auto":0.07}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VEHICLE COUNT & CO2 HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_highway_str(highway_val) -> str:
    """Normalise highway field (may be list-string or plain string)."""
    s = str(highway_val).strip("[]'\" ")
    return s.split(",")[0].strip().strip("'\"") if "," in s else s


def fallback_vehicle_count(highway: str, hour: int = 8) -> dict:
    """
    Generate a realistic vehicle breakdown for a road segment
    when no camera image is available.
    """
    hw = get_highway_str(highway)
    lo, hi = FALLBACK_VEHICLE_COUNT.get(hw, DEFAULT_COUNT_RANGE)

    # Hour-of-day multiplier
    hour_mult = {
        0:0.10,1:0.07,2:0.05,3:0.05,4:0.07,5:0.20,
        6:0.60,7:1.30,8:1.80,9:1.50,10:1.10,11:1.05,
        12:1.10,13:1.15,14:1.05,15:1.00,16:1.20,17:1.70,
        18:1.90,19:1.60,20:1.20,21:0.80,22:0.50,23:0.25,
    }.get(hour % 24, 1.0)

    base = (lo + hi) / 2
    total = max(1, int(round(base * hour_mult + random.gauss(0, (hi - lo) * 0.15))))
    total = max(lo // 2, min(total, hi * 2))

    mix = VEHICLE_MIX.get(hw, DEFAULT_MIX)
    types = list(mix.keys())
    probs = np.array([mix[t] for t in types])
    probs /= probs.sum()
    counts = np.random.multinomial(total, probs)
    breakdown = {f"n_{t}": int(c) for t, c in zip(types, counts)}
    co2_per_km = sum(breakdown.get(f"n_{t}", 0) * g
                     for t, g in EMISSIONS_G_PER_KM.items())
    return {"total": total, "co2_per_km_g": co2_per_km, **breakdown}


def physics_emission_factor(vehicle_count: int, avg_speed: float,
                             length_m: float, co2_per_km: float = None,
                             building_density: int = 5,
                             vegetation_score: int = 2,
                             aqi: int = 100) -> float:
    """
    Fallback emission_factor calculation (physics formula reverse-engineered
    from real data; corr=0.93 with actual carbon_cost column).
    """
    k = 6.0
    base = (vehicle_count / max(avg_speed, 5)) * length_m * k

    # Congestion extra
    if vehicle_count > 120 and avg_speed < 35:
        ci = (vehicle_count - 120) / 80 * (35 - avg_speed) / 20
        base += base * ci * 0.15

    aqi_pen  = (aqi / 200) * vehicle_count * 0.8
    density  = 1 + building_density * 0.005
    veg_off  = vegetation_score * 0.8

    factor = (base + aqi_pen) * density - veg_off
    return max(factor, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ML MODEL INFERENCE (with graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
_model_bundle = None   # loaded once

def _load_model_bundle():
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle
    if not MODEL_PATH.exists():
        return None
    try:
        import torch
        bundle = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
        # Import the model class
        sys.path.insert(0, str(BASE_DIR))
        from carbon_cost_model import CarbonFusionNet, ROAD_FEATURES, TRAFFIC_FEATURES
        from carbon_cost_model import EMISSION_FEATURES, ENV_FEATURES, TEMPORAL_FEATURES
        from carbon_cost_model import CATEGORICAL_COLS, predict_carbon_cost

        model = CarbonFusionNet(
            cat_vocab_sizes=bundle["cat_vocab_sizes"],
            embed_dim_each=bundle["embed_dim_each"],
            branch_out=bundle["branch_out"],
            dropout=bundle.get("dropout", 0.25),
        )
        model.load_state_dict(bundle["model_state"])
        model.eval()
        _model_bundle = {
            "model": model,
            "scalers": bundle["scalers"],
            "cat_encoders": bundle["cat_encoders"],
            "predict_fn": predict_carbon_cost,
        }
        print("[INFO] ML model loaded successfully.")
        return _model_bundle
    except Exception as e:
        print(f"[WARN] Could not load ML model: {e}. Using physics fallback.")
        return None


def predict_emission_factor(row: dict, vehicle_info: dict) -> float:
    """
    Predict emission_factor for one road segment.
    Uses ML model if available, otherwise physics formula.
    """
    bundle = _load_model_bundle()
    if bundle is None:
        return physics_emission_factor(
            vehicle_count=vehicle_info["total"],
            avg_speed=float(row.get("avg_speed_kmph", 35)),
            length_m=float(row.get("length", 100)),
            co2_per_km=vehicle_info.get("co2_per_km_g"),
            building_density=int(row.get("building_density", 5)),
            vegetation_score=int(row.get("vegetation_score", 2)),
            aqi=int(row.get("AQI", 100)),
        )

    row_input = {
        "length":             float(row.get("length", 100)),
        "lanes":              float(row.get("lanes") or 1),
        "building_density":   int(row.get("building_density", 5)),
        "vegetation_score":   int(row.get("vegetation_score", 2)),
        "maxspeed":           float(row.get("maxspeed") or 40),
        "width":              float(row.get("width") or 8),
        "vehicle_count":      vehicle_info["total"],
        "avg_speed_kmph":     float(row.get("avg_speed_kmph", 35)),
        "n_car":              vehicle_info.get("n_car", 0),
        "n_motorcycle":       vehicle_info.get("n_motorcycle", 0),
        "n_bus":              vehicle_info.get("n_bus", 0),
        "n_truck":            vehicle_info.get("n_truck", 0),
        "n_van":              vehicle_info.get("n_van", 0),
        "n_bicycle":          vehicle_info.get("n_bicycle", 0),
        "n_auto":             vehicle_info.get("n_auto", 0),
        "co2_per_km_g":       vehicle_info.get("co2_per_km_g", 10000),
        "pm2_5_ugm3":         float(row.get("pm2_5_ugm3", 40)),
        "pm10_ugm3":          float(row.get("pm10_ugm3", 50)),
        "no2_ugm3":           float(row.get("no2_ugm3", 1.5)),
        "o3_ugm3":            float(row.get("o3_ugm3", 160)),
        "so2_ugm3":           float(row.get("so2_ugm3", 3)),
        "co_ugm3":            float(row.get("co_ugm3", 300)),
        "AQI":                int(row.get("AQI", 100)),
        "openweather_aqi_1to5": int(row.get("openweather_aqi_1to5", 3)),
        "temperature_k":      float(row.get("temperature_k", 305)),
        "humidity_pct":       float(row.get("humidity_pct", 70)),
        "wind_speed_mps":     float(row.get("wind_speed_mps", 1)),
        "hour_of_day":        8,
        "day_of_week":        1,
        "is_weekend":         0,
        "highway":            get_highway_str(row.get("highway", "residential")),
        "weather_description": str(row.get("weather_description", "clear sky")),
        "junction":           row.get("junction") or None,
        "oneway":             str(row.get("oneway", True)),
        "reversed":           str(row.get("reversed", False)),
    }
    try:
        val = bundle["predict_fn"](
            bundle["model"], row_input,
            bundle["scalers"], bundle["cat_encoders"]
        )
        return max(float(val), 1.0)
    except Exception as e:
        print(f"[WARN] ML inference failed ({e}), using physics fallback.")
        return physics_emission_factor(
            vehicle_info["total"],
            float(row.get("avg_speed_kmph", 35)),
            float(row.get("length", 100)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────
_graph_cache = None

def build_emission_graph(geojson_path: Path = KOLKATA_GJ,
                          use_ml: bool = True,
                          hour: int = 8,
                          use_csv_emission: bool = True,
                          vehicle_override: int = None) -> nx.MultiDiGraph:
    """
    Build a directed road graph where edge weight = emission_factor.

    use_csv_emission=True  -> use the pre-computed carbon_cost column directly
                               (fastest, no ML call needed for each edge)
    use_csv_emission=False -> call ML model / physics formula per edge
    vehicle_override       -> if set (from YOLO detection), use this count
                               instead of CSV or random fallback for ALL edges
    """
    global _graph_cache
    if _graph_cache is not None and vehicle_override is None:
        return _graph_cache

    print(f"[INFO] Loading road network from: {geojson_path}")
    gdf = gpd.read_file(str(geojson_path))
    print(f"[INFO] {len(gdf):,} road segments loaded.")

    G = nx.MultiDiGraph()

    def safe_scalar(val, default=None):
        """Return a scalar even if val is a list/array (GeoJSON multi-value columns)."""
        if val is None:
            return default
        if hasattr(val, "__iter__") and not isinstance(val, str):
            items = list(val)
            return items[0] if items else default
        try:
            if pd.isna(val):
                return default
        except (TypeError, ValueError):
            pass
        return val

    for idx, row in gdf.iterrows():
        u = row["u"]
        v = row["v"]

        hw  = get_highway_str(safe_scalar(row.get("highway"), "residential"))
        spd_raw = safe_scalar(row.get("avg_speed_kmph"), 30)
        spd = max(float(spd_raw) if spd_raw is not None else 30.0, 5.0)
        lng_raw = safe_scalar(row.get("length"), 50)
        lng = float(lng_raw) if lng_raw is not None else 50.0

        # ── emission_factor ────────────────────────────────────────────────
        cc_raw = safe_scalar(row.get("carbon_cost"))
        if use_csv_emission and cc_raw is not None:
            try:
                emission_factor = float(cc_raw)
            except (ValueError, TypeError):
                emission_factor = None
        else:
            emission_factor = None

        if emission_factor is None or emission_factor <= 0:
            if vehicle_override is not None:
                # Use YOLO-detected vehicle count instead of random fallback
                vinfo = fallback_vehicle_count(hw, hour)
                vinfo["total"] = vehicle_override
            else:
                vinfo = fallback_vehicle_count(hw, hour)
            emission_factor = predict_emission_factor(dict(row), vinfo)

        # If vehicle_override is set and we have a CSV emission, scale it
        # by the ratio of detected vehicles to stored vehicle count
        if vehicle_override is not None and emission_factor > 0:
            stored_vc = int(float(vc_raw)) if vc_raw is not None else 60
            if stored_vc > 0:
                scale = vehicle_override / stored_vc
                emission_factor *= max(0.3, min(scale, 3.0))  # clamp [0.3x, 3x]

        # ── vehicle count for display ──────────────────────────────────────
        vc_raw = safe_scalar(row.get("vehicle_count"), 60)
        vc = int(float(vc_raw)) if vc_raw is not None else 60

        bd_raw = safe_scalar(row.get("building_density"), 5)
        vs_raw = safe_scalar(row.get("vegetation_score"), 2)
        aq_raw = safe_scalar(row.get("AQI"), 100)

        edge_data = {
            "length":           lng,
            "emission_factor":  emission_factor,
            "vehicle_count":    vc,
            "avg_speed":        spd,
            "time":             lng / (spd * 1000 / 3600),
            "highway":          hw,
            "name":             str(safe_scalar(row.get("name"), "") or ""),
            "geometry":         row.get("geometry"),
            "building_density": int(float(bd_raw)) if bd_raw is not None else 5,
            "vegetation_score": int(float(vs_raw)) if vs_raw is not None else 2,
            "AQI":              int(float(aq_raw)) if aq_raw is not None else 100,
        }
        G.add_edge(u, v, **edge_data)

    print(f"[INFO] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    if vehicle_override is None:
        _graph_cache = G
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ROUTING STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
def _safe_route(G, origin, dest, weight):
    try:
        return nx.shortest_path(G, origin, dest, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def compute_all_routes(G, origin, destination):
    """Run all 4 routing strategies. Returns dict of {strategy_name: route}."""
    # Add composite weight for balanced routing
    for u, v, data in G.edges(data=True):
        n_ef  = data["emission_factor"] / 5000
        n_t   = data["time"] / 300
        n_d   = data["length"] / 1000
        data["composite"] = 0.5 * n_ef + 0.3 * n_t + 0.2 * n_d

    strategies = {
        "Shortest Distance":    ("length",         "#3B82F6"),
        "Fastest Time":         ("time",            "#F59E0B"),
        "Lowest Emission":      ("emission_factor", "#10B981"),
        "Balanced":             ("composite",       "#8B5CF6"),
    }
    routes = {}
    for name, (weight, _) in strategies.items():
        r = _safe_route(G, origin, destination, weight)
        if r:
            routes[name] = r
    return routes, strategies


def route_stats(G, route):
    """Compute stats for a route. Returns dict."""
    dist = time_ = ef = veh = 0
    speeds = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if v not in G[u]:
            continue
        ed = min(G[u][v].values(), key=lambda x: x["emission_factor"])
        dist  += ed["length"]
        time_ += ed["time"]
        ef    += ed["emission_factor"]
        veh   += ed.get("vehicle_count", 0)
        speeds.append(ed["avg_speed"])
    return {
        "distance_km":    round(dist / 1000, 3),
        "time_min":       round(time_ / 60, 2),
        "emission_factor": round(ef, 1),
        "avg_vehicles":   round(veh / max(len(route) - 1, 1), 1),
        "avg_speed_kmh":  round(sum(speeds) / len(speeds), 1) if speeds else 0,
        "segments":       len(route) - 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NEAREST NODE FINDER
# ─────────────────────────────────────────────────────────────────────────────
def nearest_node(G, lat: float, lon: float) -> int:
    """Find graph node closest to (lat, lon) using Euclidean approx."""
    gdf = gpd.read_file(str(KOLKATA_GJ)).to_crs("EPSG:4326")
    nodes_u = gdf[["u", "geometry"]].copy()
    nodes_u["cx"] = nodes_u.geometry.apply(
        lambda g: g.coords[0][0] if g.geom_type == "Point" else g.centroid.x)
    nodes_u["cy"] = nodes_u.geometry.apply(
        lambda g: g.coords[0][1] if g.geom_type == "Point" else g.centroid.y)

    best_node = None
    best_dist = float("inf")
    for _, r in nodes_u.drop_duplicates("u").iterrows():
        d = (r["cx"] - lon) ** 2 + (r["cy"] - lat) ** 2
        if d < best_dist:
            best_dist = d
            best_node = r["u"]
    return int(best_node)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ROUTE COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_route_comparison(G, routes: dict, strategies: dict,
                          stats: dict, origin, destination,
                          save_path: str = "route_compare_kolkata.png"):
    """Generate a rich route comparison figure."""
    fig = plt.figure(figsize=(18, 10), facecolor="#0F172A")

    # ── Map panel (left) ──────────────────────────────────────────────────
    ax_map = fig.add_axes([0.01, 0.05, 0.55, 0.90])
    ax_map.set_facecolor("#1E293B")
    ax_map.tick_params(colors="#94A3B8")
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#334155")

    # Draw all background roads (grey)
    drawn = set()
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom and (u, v) not in drawn:
            if geom.geom_type == "LineString":
                xs, ys = geom.xy
                ax_map.plot(xs, ys, color="#334155", linewidth=0.4, alpha=0.6, zorder=1)
            drawn.add((u, v))

    # Draw each route
    legend_patches = []
    for name, (weight, color) in strategies.items():
        route = routes.get(name)
        if not route:
            continue
        xs_all, ys_all = [], []
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if v not in G[u]:
                continue
            ed = min(G[u][v].values(), key=lambda x: x["emission_factor"])
            geom = ed.get("geometry")
            if geom and geom.geom_type == "LineString":
                xs, ys = geom.xy
                ax_map.plot(xs, ys, color=color, linewidth=3.5,
                            alpha=0.92, zorder=5, solid_capstyle="round")
                xs_all.extend(xs); ys_all.extend(ys)

        st = stats.get(name, {})
        lbl = (f"{name}\n"
               f"  {st.get('distance_km','?')} km | "
               f"{st.get('time_min','?')} min | "
               f"EF={st.get('emission_factor','?'):.0f}")
        legend_patches.append(mpatches.Patch(color=color, label=lbl))

    # Origin / destination markers
    def get_node_coords(node_id):
        for u, v, data in G.edges(data=True):
            geom = data.get("geometry")
            if geom and geom.geom_type == "LineString":
                if u == node_id:
                    return geom.coords[0]
                if v == node_id:
                    return geom.coords[-1]
        return None

    oc = get_node_coords(origin)
    dc = get_node_coords(destination)
    if oc:
        ax_map.scatter(*oc, s=200, c="#22D3EE", zorder=10, edgecolors="white", linewidths=1.5)
        ax_map.annotate("  START", oc, color="#22D3EE", fontsize=9, fontweight="bold", zorder=11)
    if dc:
        ax_map.scatter(*dc, s=200, c="#F43F5E", zorder=10, edgecolors="white", linewidths=1.5)
        ax_map.annotate("  END", dc, color="#F43F5E", fontsize=9, fontweight="bold", zorder=11)

    ax_map.set_title("Kolkata Eco-Routing — Route Comparison", color="white",
                     fontsize=14, fontweight="bold", pad=10)
    ax_map.legend(handles=legend_patches, loc="lower left",
                  facecolor="#1E293B", edgecolor="#475569",
                  labelcolor="white", fontsize=8.5)
    ax_map.set_aspect("equal")

    # ── Stats panel (right) ───────────────────────────────────────────────
    ax_stats = fig.add_axes([0.58, 0.05, 0.40, 0.90])
    ax_stats.set_facecolor("#1E293B")
    ax_stats.axis("off")

    ax_stats.text(0.5, 0.97, "Route Statistics", color="white",
                  fontsize=15, fontweight="bold", ha="center", va="top",
                  transform=ax_stats.transAxes)
    ax_stats.text(0.5, 0.92, "Kolkata Road Network  |  Emission Factor = g CO\u2082/segment",
                  color="#94A3B8", fontsize=8, ha="center", va="top",
                  transform=ax_stats.transAxes)

    # Table
    col_labels = ["Strategy", "Dist (km)", "Time (min)", "Emission\nFactor", "Segments"]
    col_widths  = [0.32, 0.17, 0.17, 0.17, 0.14]
    y = 0.85
    # header row
    x = 0.01
    for lbl, w in zip(col_labels, col_widths):
        ax_stats.text(x + w / 2, y, lbl, color="#94A3B8", fontsize=8.5,
                      fontweight="bold", ha="center", va="center",
                      transform=ax_stats.transAxes)
        x += w
    ax_stats.plot([0.01, 0.99], [y - 0.03, y - 0.03], color="#475569",
                  linewidth=0.8, transform=ax_stats.transAxes, clip_on=False)
    y -= 0.07

    best_ef = min((stats[n]["emission_factor"] for n in stats), default=0)
    for name, (weight, color) in strategies.items():
        st = stats.get(name)
        if not st:
            continue
        bg_color = "#1E3A2F" if st["emission_factor"] == best_ef else "#1E293B"
        rect = plt.Rectangle((0.0, y - 0.055), 1.0, 0.075,
                               facecolor=bg_color, transform=ax_stats.transAxes,
                               clip_on=False, zorder=0)
        ax_stats.add_patch(rect)

        vals = [name, st["distance_km"], st["time_min"],
                f"{st['emission_factor']:,.0f}", st["segments"]]
        x = 0.01
        for val, w in zip(vals, col_widths):
            ax_stats.text(x + w / 2, y - 0.015, str(val),
                          color=color, fontsize=9, ha="center", va="center",
                          transform=ax_stats.transAxes, fontweight="bold")
            x += w
        y -= 0.085

    # Savings callout
    y -= 0.02
    ax_stats.plot([0.01, 0.99], [y, y], color="#475569",
                  linewidth=0.8, transform=ax_stats.transAxes, clip_on=False)
    y -= 0.06

    if "Shortest Distance" in stats and "Lowest Emission" in stats:
        saved_ef = (stats["Shortest Distance"]["emission_factor"]
                    - stats["Lowest Emission"]["emission_factor"])
        saved_pct = saved_ef / max(stats["Shortest Distance"]["emission_factor"], 1) * 100
        ax_stats.text(0.5, y, f"Eco-route saves {saved_ef:,.0f} g CO\u2082  ({saved_pct:.1f}%)",
                      color="#10B981", fontsize=12, fontweight="bold",
                      ha="center", va="center", transform=ax_stats.transAxes)
        y -= 0.06
        extra_km = (stats["Lowest Emission"]["distance_km"]
                    - stats["Shortest Distance"]["distance_km"])
        extra_min = (stats["Lowest Emission"]["time_min"]
                     - stats["Shortest Distance"]["time_min"])
        ax_stats.text(0.5, y,
                      f"Extra distance: +{extra_km:.2f} km  |  Extra time: +{extra_min:.1f} min",
                      color="#94A3B8", fontsize=9, ha="center", va="center",
                      transform=ax_stats.transAxes)

    # Emission factor breakdown bar chart
    y -= 0.10
    names_  = list(stats.keys())
    efs_    = [stats[n]["emission_factor"] for n in names_]
    colors_ = [strategies[n][1] for n in names_]
    bars_ax = fig.add_axes([0.595, y - 0.16, 0.385, 0.18])
    bars_ax.set_facecolor("#0F172A")
    bars = bars_ax.barh(names_, efs_, color=colors_, edgecolor="none", height=0.55)
    bars_ax.set_xlabel("Emission Factor (g CO\u2082)", color="#94A3B8", fontsize=8)
    bars_ax.tick_params(colors="#94A3B8", labelsize=8)
    for spine in bars_ax.spines.values():
        spine.set_edgecolor("#334155")
    for bar, val in zip(bars, efs_):
        bars_ax.text(bar.get_width() + max(efs_) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:,.0f}", color="white", fontsize=7.5, va="center")
    bars_ax.set_title("Emission Factor by Strategy", color="white", fontsize=9, pad=4)

    plt.savefig(save_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Route comparison saved -> {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def run_eco_routing(origin_lat: float, origin_lon: float,
                     dest_lat: float,   dest_lon: float,
                     hour: int = 8,
                     plot_path: str = "route_compare_kolkata.png",
                     use_ml: bool = True,
                     vehicle_override: int = None) -> dict:
    """
    Full pipeline:
      1. Load / cache Kolkata graph
      2. Find nearest graph nodes to (lat,lon) pairs
      3. Compute 4 routes
      4. Generate comparison plot
      5. Return JSON-serialisable result dict

    vehicle_override: if set (from YOLO), this vehicle count is used
                      to scale emission factors across all edges.
    """
    G = build_emission_graph(use_ml=use_ml, vehicle_override=vehicle_override)

    print(f"[INFO] Finding nearest nodes to origin ({origin_lat}, {origin_lon})...")
    origin_node = nearest_node(G, origin_lat, origin_lon)
    print(f"[INFO] Finding nearest nodes to dest ({dest_lat}, {dest_lon})...")
    dest_node   = nearest_node(G, dest_lat, dest_lon)
    print(f"[INFO] Origin node: {origin_node}  |  Dest node: {dest_node}")

    if origin_node == dest_node:
        return {"error": "Origin and destination are the same node. Move them further apart."}

    routes, strategies = compute_all_routes(G, origin_node, dest_node)

    if not routes:
        return {"error": "No path found between these two points. Try different coordinates."}

    all_stats = {name: route_stats(G, route) for name, route in routes.items()}

    # Best route (lowest emission_factor)
    best_name = min(all_stats, key=lambda n: all_stats[n]["emission_factor"])

    plot_route_comparison(G, routes, strategies, all_stats,
                          origin_node, dest_node, save_path=plot_path)

    # Build GeoJSON for each route (for frontend map)
    routes_geojson = {}
    for name, route in routes.items():
        coords = []
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if v not in G[u]:
                continue
            ed = min(G[u][v].values(), key=lambda x: x["emission_factor"])
            geom = ed.get("geometry")
            if geom and geom.geom_type == "LineString":
                # Convert from EPSG:3857 to lat/lon on-the-fly
                from pyproj import Transformer
                tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
                for x, y in geom.coords:
                    lon_, lat_ = tr.transform(x, y)
                    coords.append([lat_, lon_])
        routes_geojson[name] = coords

    return {
        "origin_node":  origin_node,
        "dest_node":    dest_node,
        "strategies":   {k: v[1] for k, v in strategies.items()},   # name→color
        "stats":        all_stats,
        "best_route":   best_name,
        "routes_latlng": routes_geojson,
        "plot_path":    plot_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, pprint
    parser = argparse.ArgumentParser(description="Eco-routing engine for Kolkata")
    parser.add_argument("--origin-lat", type=float, default=22.5680,
                        help="Origin latitude  (default: near Esplanade)")
    parser.add_argument("--origin-lon", type=float, default=88.3512,
                        help="Origin longitude")
    parser.add_argument("--dest-lat",   type=float, default=22.5756,
                        help="Destination latitude  (default: near Sealdah)")
    parser.add_argument("--dest-lon",   type=float, default=88.3697,
                        help="Destination longitude")
    parser.add_argument("--hour",       type=int,   default=8)
    parser.add_argument("--plot",       type=str,   default="route_compare_kolkata.png")
    parser.add_argument("--no-ml",      action="store_true")
    args = parser.parse_args()

    result = run_eco_routing(
        args.origin_lat, args.origin_lon,
        args.dest_lat,   args.dest_lon,
        hour=args.hour,
        plot_path=args.plot,
        use_ml=not args.no_ml,
    )

    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  ECO-ROUTING RESULTS — Kolkata")
    print("=" * 55)
    for name, st in result["stats"].items():
        marker = " <-- BEST ECO" if name == result["best_route"] else ""
        print(f"\n  {name}{marker}")
        print(f"    Distance     : {st['distance_km']} km")
        print(f"    Time         : {st['time_min']} min")
        print(f"    Emission Factor : {st['emission_factor']:,.1f} g CO2")
        print(f"    Avg vehicles : {st['avg_vehicles']}")
    print(f"\n  Plot saved: {result['plot_path']}")
    print("=" * 55)
