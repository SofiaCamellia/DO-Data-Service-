import argparse
import json
import math
import os
import pickle
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception as e:
    print(f"[WARN] xgboost import failed, fallback will be used: {e}")
    XGBRegressor = None
    HAS_XGBOOST = False

import joblib
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

import concurrent.futures
import subprocess

import geopandas as gpd
from shapely.geometry import Polygon
import shapely.vectorized as shp_vectorized
from fiona import path

RANDOM_SEED = 24
# DATA_PATH = Path(r"E:\partition_reconstruction_article\data\IAP_TSDO.npy")
# OUTPUT_ROOT = Path(r"E:\codex_code\ocean9")
DATA_PATH = Path(r"/home/bingxing2/home/scx7l1f/IAP_TSDO.npy")
OUTPUT_ROOT = Path(r"/home/bingxing2/home/scx7l1f/rec/ensemble/ocean9")

MODEL_DIR = OUTPUT_ROOT / "models"
RESULT_DIR = OUTPUT_ROOT / "results"
DATA_DIR = OUTPUT_ROOT / "data"
PROGRESS_LOG = RESULT_DIR / "progress.log"
SPATIAL_IDXS = [0, 1, 4, 5, 6]
PHYS_IDXS = [0, 4, 5, 6]
MODEL_ORDER = ["lgb", "rf", "cb", "knn", "xgb", "ert"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = torch.cuda.is_available()
DEFAULT_OCEAN_BASE_SHP = Path(r"/home/bingxing2/home/scx7l1f/rec/ne_10m_poly_shp/ne_10m_ocean.shp")

DEFAULT_MARINE_POLYS_SHP = Path(r"/home/bingxing2/home/scx7l1f/rec/ne_10m_poly_shp/ne_10m_geography_marine_polys.shp")
REGION_ORDER = [
    "Arctic Ocean",
    "North Atlantic Ocean",
    "Equatorial Atlantic Ocean",
    "South Atlantic Ocean",
    "Indian Ocean",
    "North Pacific Ocean",
    "Equatorial Pacific Ocean",
    "South Pacific Ocean",
    "Southern Ocean",
]


def slugify_region(name: str) -> str:
    return name.lower().replace(" ", "_")


def set_output_root(root: Path):
    global OUTPUT_ROOT, MODEL_DIR, RESULT_DIR, DATA_DIR, PROGRESS_LOG
    OUTPUT_ROOT = root
    MODEL_DIR = OUTPUT_ROOT / "models"
    RESULT_DIR = OUTPUT_ROOT / "results"
    DATA_DIR = OUTPUT_ROOT / "data"
    PROGRESS_LOG = RESULT_DIR / "progress.log"


@dataclass
class RunConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    n_folds: int = 5
    bias_multiplier: float = 1.03
    bias_quantile: float = 0.70
    max_mul_shallow: float = 1.50
    max_mul_mid: float = 1.15
    max_mul_deep: float = 1.00
    neg_penalty_weight: float = 50.0
    meta_epochs: int = 100
    meta_lr: float = 0.005
    meta_batch_size: int = 65536
    pinn_epochs: int = 50
    pinn_lr: float = 0.005
    pinn_batch_size: int = 65536
    smoke_test: bool = False
    max_rows: int = 0
    force_xgb_fallback: bool = False
    skip_pinn: bool = False
    dry_run: bool = False
    upper_anchor_top_k: int = 2
    upper_anchor_clip_multiplier: float = 1.08
    target_violation_ratio_min: float = 0.0002
    target_violation_ratio_max: float = 0.0050
    stacker_upper_lambdas: tuple = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20)
    pinn_safe_keep_weight: float = 0.20
    risk_phys_alpha: float = 0.95
    risk_width_quantile: float = 0.90
    upper_uncertainty_gamma: float = 1.0
    pinn_early_stop_patience: int = 3
    pinn_min_delta: float = 1e-4


def ensure_dirs():
    for path in [OUTPUT_ROOT, MODEL_DIR, RESULT_DIR, DATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def update_progress(t0, stage: str, status: str, **payload):
    record = {
        "pid": int(os.getpid()),
        "stage": stage,
        "status": status,
        "elapsed_seconds": float(time.time() - t0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    record.update(payload)
    save_json(RESULT_DIR / "progress.json", record)
    append_jsonl(PROGRESS_LOG, record)
    print(
        f"[PROGRESS] stage={stage} status={status} "
        f"elapsed={record['elapsed_seconds'] / 60.0:.2f} min"
    )


def evaluate_regression(y_true, y_pred, name: str):
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mb": float(np.mean(y_pred - y_true)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
    }
    print(
        f"{name:<18s} RMSE={metrics['rmse']:.4f} "
        f"R2={metrics['r2']:.4f} MAE={metrics['mae']:.4f} "
        f"MB={metrics['mb']:.4f} range=[{metrics['pred_min']:.3f}, {metrics['pred_max']:.3f}]"
    )
    return metrics


def global_filter(raw):
    years = raw[:, 2].astype(int)
    depth = raw[:, 4].astype(float)
    mask = (years >= 1980) & (depth <= 2000)
    return raw[mask]


def maybe_subsample(data, max_rows: int):
    if max_rows and len(data) > max_rows:
        rng = np.random.default_rng(RANDOM_SEED)
        keep = np.sort(rng.choice(len(data), size=max_rows, replace=False))
        return data[keep]
    return data


def maybe_subsample_xy(X, y, max_rows: int):
    if max_rows and len(y) > max_rows:
        rng = np.random.default_rng(RANDOM_SEED)
        keep = np.sort(rng.choice(len(y), size=max_rows, replace=False))
        return X[keep], y[keep]
    return X, y


def normalize_longitude(lon):
    lon = np.asarray(lon, dtype=np.float32)
    return (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)


def build_ocean_region_geometries(marine_polys_path: Path):
    marine_gdf = gpd.read_file(marine_polys_path)
    ocean_mapping = {
        "Arctic Ocean": [
            "Arctic Ocean", "Beaufort Sea", "Norwegian Sea", "Greenland Sea", "Chukchi Sea",
            "Kara Sea", "Laptev Sea", "White Sea", "Barents Sea", "East Siberian Sea"
        ],
        "Atlantic Ocean": [
            "North Atlantic Ocean", "South Atlantic Ocean", "Caribbean Sea", "Labrador Sea",
            "Bay of Biscay", "Scotia Sea", "Sargasso Sea", "Gulf of Mexico", "Gulf of Guinea",
            "Davis Strait", "Bahía de Campeche", "Bah铆a de Campeche", "Baffin Bay"
        ],
        "Indian Ocean": [
            "INDIAN OCEAN", "Arabian Sea", "Andaman Sea", "Timor Sea", "Mozambique Channel",
            "Bay of Bengal", "Great Australian Bight", "Laccadive Sea"
        ],
        "Pacific Ocean": [
            "North Pacific Ocean", "South Pacific Ocean", "Tasman Sea", "Philippine Sea",
            "Yellow Sea", "East China Sea", "Bering Sea", "South China Sea",
            "Bismarck Sea", "Solomon Sea", "Taiwan Strait", "Halmahera Sea", "Samar Sea",
            "Visayan Sea", "Coral Sea", "Bohol Sea", "Gulf of Alaska", "Sea of Okhotsk",
            "Norton Sound", "Bristol Bay", "Gulf of Anadyr'"
        ],
        "Southern Ocean": [
            "SOUTHERN OCEAN", "Weddell Sea", "Ross Sea", "Bransfield Strait", "Drake Passage"
        ],
    }
    name_to_ocean = {name.upper(): ocean for ocean, names in ocean_mapping.items() for name in names}
    marine_gdf["main_basin"] = marine_gdf["name"].str.upper().map(name_to_ocean)
    major_oceans_gdf = marine_gdf.dropna(subset=["main_basin"])

    north_zone = Polygon([(-180, 10), (180, 10), (180, 90), (-180, 90)])
    equatorial_zone = Polygon([(-180, -10), (180, -10), (180, 10), (-180, 10)])
    south_zone = Polygon([(-180, -90), (180, -90), (180, -10), (-180, -10)])

    new_geometries = []
    new_basin_names = []
    for _, row in major_oceans_gdf.iterrows():
        geom = row.geometry
        basin = row.main_basin
        name = row["name"]

        if name == "Coral Sea":
            new_geometries.append(geom)
            new_basin_names.append("South Pacific Ocean")
        elif name in ["Caribbean Sea", "South China Sea"]:
            north_intersection = geom.intersection(north_zone)
            if not north_intersection.is_empty:
                new_geometries.append(north_intersection)
                new_basin_names.append(f"North {basin}")
            south_intersection = geom.intersection(south_zone)
            if not south_intersection.is_empty:
                new_geometries.append(south_intersection)
                new_basin_names.append(f"South {basin}")
        elif basin in ["Atlantic Ocean", "Pacific Ocean"]:
            north_intersection = geom.intersection(north_zone)
            if not north_intersection.is_empty:
                new_geometries.append(north_intersection)
                new_basin_names.append(f"North {basin}")
            equatorial_intersection = geom.intersection(equatorial_zone)
            if not equatorial_intersection.is_empty:
                new_geometries.append(equatorial_intersection)
                new_basin_names.append(f"Equatorial {basin}")
            south_intersection = geom.intersection(south_zone)
            if not south_intersection.is_empty:
                new_geometries.append(south_intersection)
                new_basin_names.append(f"South {basin}")
        else:
            new_geometries.append(geom)
            new_basin_names.append(basin)

    final_gdf = gpd.GeoDataFrame({"basin_name": new_basin_names}, geometry=new_geometries, crs="EPSG:4326")
    dissolved = final_gdf.dissolve(by="basin_name").reset_index()
    geometries = {}
    for region_name in REGION_ORDER:
        row = dissolved[dissolved["basin_name"] == region_name]
        if row.empty:
            raise RuntimeError(f"Region geometry missing for {region_name}")
        geometries[region_name] = row.iloc[0].geometry
    return geometries


def classify_points_to_regions(X, region_geometries, chunk_size=200000):
    lat = X[:, 0].astype(np.float32)
    lon = normalize_longitude(X[:, 1])
    region_ids = np.full(len(X), -1, dtype=np.int16)
    remaining = np.ones(len(X), dtype=bool)
    for region_idx, region_name in enumerate(REGION_ORDER):
        geom = region_geometries[region_name]
        for start in range(0, len(X), chunk_size):
            stop = min(start + chunk_size, len(X))
            active = remaining[start:stop]
            if not np.any(active):
                continue
            local_lon = lon[start:stop]
            local_lat = lat[start:stop]
            match = shp_vectorized.contains(geom, local_lon, local_lat) | shp_vectorized.touches(geom, local_lon, local_lat)
            match = match & active
            if np.any(match):
                region_ids[start:stop][match] = region_idx
                remaining[start:stop][match] = False
    return region_ids


def build_region_datasets(X, y, region_geometries):
    region_ids = classify_points_to_regions(X, region_geometries)
    payload = {}
    summary = {
        "total_rows": int(len(y)),
        "assigned_rows": int(np.sum(region_ids >= 0)),
        "unassigned_rows": int(np.sum(region_ids < 0)),
        "regions": {},
    }
    for region_idx, region_name in enumerate(REGION_ORDER):
        mask = region_ids == region_idx
        payload[region_name] = (X[mask], y[mask])
        summary["regions"][region_name] = {"rows": int(mask.sum())}
    return payload, summary


def split_each_year_811(X, y, seed=RANDOM_SEED):
    years = X[:, 2].astype(int)
    unique_years = np.unique(years)
    rng = np.random.default_rng(seed)
    idx_train = []
    idx_val = []
    idx_test = []
    summary = []

    for year in unique_years:
        idx = np.flatnonzero(years == year)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(math.floor(n * 0.8))
        n_val = int(math.floor(n * 0.1))
        n_test = n - n_train - n_val
        if n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)
        if n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_train + n_val + n_test != n:
            n_train = n - n_val - n_test

        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        idx_train.append(train_idx)
        idx_val.append(val_idx)
        idx_test.append(test_idx)
        summary.append(
            {"year": int(year), "total": int(n), "train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))}
        )

    idx_train = np.concatenate(idx_train)
    idx_val = np.concatenate(idx_val)
    idx_test = np.concatenate(idx_test)
    return (
        (X[idx_train], y[idx_train], idx_train),
        (X[idx_val], y[idx_val], idx_val),
        (X[idx_test], y[idx_test], idx_test),
        summary,
    )


def make_year_balanced_folds(years, n_folds, seed=RANDOM_SEED):
    years = np.asarray(years).astype(int)
    fold_ids = np.full(len(years), -1, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for year in np.unique(years):
        idx = np.flatnonzero(years == year)
        rng.shuffle(idx)
        for pos, row_idx in enumerate(idx):
            fold_ids[row_idx] = pos % n_folds
    if np.any(fold_ids < 0):
        raise RuntimeError("Fold assignment failed.")
    return fold_ids


def make_year_inner_es_split(X_fit, y_fit, frac=0.15, seed=RANDOM_SEED):
    years = X_fit[:, 2].astype(int)
    rng = np.random.default_rng(seed)
    train_parts = []
    es_parts = []
    for year in np.unique(years):
        idx = np.flatnonzero(years == year)
        rng.shuffle(idx)
        n_es = max(1, int(round(len(idx) * frac)))
        if n_es >= len(idx):
            n_es = 1
        es_parts.append(idx[:n_es])
        train_parts.append(idx[n_es:])
    idx_train = np.concatenate(train_parts)
    idx_es = np.concatenate(es_parts)
    return X_fit[idx_train], y_fit[idx_train], X_fit[idx_es], y_fit[idx_es]


def approximate_pressure_dbar(depth, lat):
    lat_rad = np.deg2rad(lat)
    sin2 = np.sin(lat_rad) ** 2
    gravity_factor = 1.0 + 5.2792e-3 * sin2 + 2.36e-5 * (sin2**2)
    return depth * gravity_factor


def oxygen_saturation_umolkg_np(lat, depth, temp, salt):
    pressure = np.clip(approximate_pressure_dbar(depth, lat), 0.0, None)
    t_k = temp + 273.15
    a1, a2, a3, a4 = -177.7888, 255.5907, 146.4813, -22.2040
    b1, b2, b3 = -0.037362, 0.016504, -0.0020564
    ln_do = (
        a1
        + a2 * (100.0 / t_k)
        + a3 * np.log(t_k / 100.0)
        + a4 * (t_k / 100.0)
        + salt * (b1 + b2 * (t_k / 100.0) + b3 * ((t_k / 100.0) ** 2))
    )
    return np.exp(ln_do) * 44.66 * (1.0 + (0.032 * pressure / 1000.0))


def oxygen_saturation_umolkg_torch(features):
    lat = features[:, 0]
    depth = features[:, 1]
    temp = features[:, 2]
    salt = features[:, 3]
    lat_rad = torch.deg2rad(lat)
    sin2 = torch.sin(lat_rad) ** 2
    pressure = depth * (1.0 + 5.2792e-3 * sin2 + 2.36e-5 * (sin2**2))
    pressure = torch.clamp(pressure, min=0.0)
    t_k = temp + 273.15
    a1, a2, a3, a4 = -177.7888, 255.5907, 146.4813, -22.2040
    b1, b2, b3 = -0.037362, 0.016504, -0.0020564
    ln_do = (
        a1
        + a2 * (100.0 / t_k)
        + a3 * torch.log(t_k / 100.0)
        + a4 * (t_k / 100.0)
        + salt * (b1 + b2 * (t_k / 100.0) + b3 * ((t_k / 100.0) ** 2))
    )
    return torch.exp(ln_do) * 44.66 * (1.0 + (0.032 * pressure / 1000.0))


def compute_max_allowed_np(X_phys, config: RunConfig):
    lat = X_phys[:, 0].astype(np.float64)
    depth = X_phys[:, 1].astype(np.float64)
    temp = X_phys[:, 2].astype(np.float64)
    salt = X_phys[:, 3].astype(np.float64)
    do_sat = oxygen_saturation_umolkg_np(lat, depth, temp, salt)
    max_allowed = np.empty_like(do_sat, dtype=np.float64)
    shallow = depth <= 50.0
    mid = (depth > 50.0) & (depth <= 200.0)
    deep = depth > 200.0
    max_allowed[shallow] = do_sat[shallow] * config.max_mul_shallow
    max_allowed[mid] = do_sat[mid] * config.max_mul_mid
    max_allowed[deep] = do_sat[deep] * config.max_mul_deep
    return max_allowed.astype(np.float32)


def compute_max_allowed_torch(features, config: RunConfig):
    depth = features[:, 1]
    do_sat = oxygen_saturation_umolkg_torch(features)
    max_allowed = torch.zeros_like(do_sat)
    max_allowed[depth <= 50.0] = do_sat[depth <= 50.0] * config.max_mul_shallow
    max_allowed[(depth > 50.0) & (depth <= 200.0)] = do_sat[(depth > 50.0) & (depth <= 200.0)] * config.max_mul_mid
    max_allowed[depth > 200.0] = do_sat[depth > 200.0] * config.max_mul_deep
    return max_allowed


def report_physics_quality(X, y, config: RunConfig, name: str):
    X_phys = X[:, PHYS_IDXS]
    depth = X_phys[:, 1]
    max_allowed = compute_max_allowed_np(X_phys, config)
    neg = y < 0
    over = y > max_allowed
    bad = neg | over | ~np.isfinite(y) | ~np.isfinite(max_allowed)

    def bucket(mask):
        total = int(mask.sum())
        if total == 0:
            return {"total": 0, "bad": 0, "over": 0, "neg": 0}
        return {"total": total, "bad": int((bad & mask).sum()), "over": int((over & mask).sum()), "neg": int((neg & mask).sum())}

    stats = {
        "name": name,
        "overall_total": int(len(y)),
        "overall_bad": int(bad.sum()),
        "shallow": bucket(depth <= 50),
        "mid": bucket((depth > 50) & (depth <= 200)),
        "deep": bucket(depth > 200),
    }
    print(f"{name:<10s} bad={stats['overall_bad']}/{stats['overall_total']} ({100.0 * stats['overall_bad'] / max(1, stats['overall_total']):.4f}%)")
    return ~bad, stats


def summarize_physics_violations(X, preds, config: RunConfig):
    X_phys = X[:, PHYS_IDXS]
    depth = X_phys[:, 1]
    max_allowed = compute_max_allowed_np(X_phys, config)
    upper = preds > max_allowed
    lower = preds < 0

    def bucket(mask):
        total = int(mask.sum())
        if total == 0:
            return {
                "total": 0,
                "upper_count": 0,
                "lower_count": 0,
                "upper_ratio": 0.0,
                "lower_ratio": 0.0,
            }
        upper_count = int((upper & mask).sum())
        lower_count = int((lower & mask).sum())
        return {
            "total": total,
            "upper_count": upper_count,
            "lower_count": lower_count,
            "upper_ratio": float(upper_count / total),
            "lower_ratio": float(lower_count / total),
        }

    total = int(len(preds))
    upper_count = int(upper.sum())
    lower_count = int(lower.sum())
    shallow_stats = bucket(depth <= 50.0)
    mid_stats = bucket((depth > 50.0) & (depth <= 200.0))
    deep_stats = bucket(depth > 200.0)
    return {
        "total": total,
        "upper_count": upper_count,
        "lower_count": lower_count,
        "upper_ratio": float(upper_count / max(1, total)),
        "lower_ratio": float(lower_count / max(1, total)),
        "pred_min": float(np.min(preds)),
        "pred_max": float(np.max(preds)),
        "upper_shallow_over_150sat_count": shallow_stats["upper_count"],
        "upper_mid_over_115sat_count": mid_stats["upper_count"],
        "upper_deep_over_100sat_count": deep_stats["upper_count"],
        "shallow": shallow_stats,
        "mid": mid_stats,
        "deep": deep_stats,
    }


def filter_invalid_labels(X, y, config: RunConfig, name: str):
    X_phys = X[:, PHYS_IDXS]
    max_allowed = compute_max_allowed_np(X_phys, config)
    finite_mask = np.isfinite(y) & np.isfinite(max_allowed)
    lower_bad = finite_mask & (y < 0)
    upper_bad = finite_mask & (y > max_allowed)
    invalid_mask = (~finite_mask) | lower_bad | upper_bad
    keep_mask = ~invalid_mask
    stats = {
        "name": name,
        "before_rows": int(len(y)),
        "kept_rows": int(keep_mask.sum()),
        "removed_rows": int(invalid_mask.sum()),
        "removed_ratio": float(invalid_mask.mean()) if len(y) else 0.0,
        "upper_bad": int(upper_bad.sum()),
        "lower_bad": int(lower_bad.sum()),
        "non_finite": int((~finite_mask).sum()),
    }
    return X[keep_mask], y[keep_mask], keep_mask, stats


def quantile_width_matrix(q_pred):
    q_sorted = np.sort(q_pred, axis=1)
    return (q_sorted[:, -1] - q_sorted[:, 0]).astype(np.float32)


def build_risk_mask(p_stack, upper_anchor, max_allowed, quant_width, width_threshold, config: RunConfig):
    risk_phys = p_stack > (config.risk_phys_alpha * max_allowed)
    risk_anchor = upper_anchor > max_allowed
    risk_quant = quant_width > width_threshold
    return (risk_phys | risk_anchor | risk_quant).astype(np.float32), {
        "risk_phys_count": int(risk_phys.sum()),
        "risk_anchor_count": int(risk_anchor.sum()),
        "risk_quant_count": int(risk_quant.sum()),
        "risk_total_count": int((risk_phys | risk_anchor | risk_quant).sum()),
        "width_threshold": float(width_threshold),
    }


def physical_constraint_loss(y_pred, features, config: RunConfig):
    max_allowed = compute_max_allowed_torch(features, config)
    penalty = torch.maximum(torch.zeros_like(y_pred), y_pred - max_allowed)
    neg_penalty = torch.maximum(torch.zeros_like(y_pred), -y_pred)
    return torch.mean(penalty) + config.neg_penalty_weight * torch.mean(neg_penalty)


def get_xgb_backend(config: RunConfig):
    if HAS_XGBOOST and not config.force_xgb_fallback:
        return "xgboost"
    return "histgb_fallback"


def model_factories(config: RunConfig):
    smoke = config.smoke_test
    worker_jobs = 1 if smoke else 8
    lgb_estimators = 300 if smoke else 3500
    forest_estimators = 10 if smoke else 40
    cb_iters = 300 if smoke else 3500
    xgb_iters = 300 if smoke else 3500
    knn_neighbors = 20 if smoke else 40
    cb_verbose = 20 if smoke else 50
    use_gpu_for_cb = torch.cuda.is_available() and not smoke
    xgb_backend = get_xgb_backend(config)

    cb_kwargs = {"devices": "0"} if use_gpu_for_cb else {}

    factories = {
        "lgb": lambda: lgb.LGBMRegressor(
            boosting_type="gbdt",
            num_leaves=156,
            max_depth=12,
            min_child_samples=30,
            learning_rate=0.05,
            n_estimators=lgb_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=worker_jobs,
            random_state=RANDOM_SEED,
            verbosity=-1,
        ),
        "rf": lambda: RandomForestRegressor(
            n_estimators=forest_estimators,
            random_state=RANDOM_SEED,
            max_depth=20,
            min_samples_leaf=50,
            n_jobs=worker_jobs,
            verbose=2 if not smoke else 0,
            oob_score=False,
            max_features="sqrt",
        ),
        "cb": lambda: CatBoostRegressor(
            iterations=cb_iters,
            depth=12,
            learning_rate=0.05,
            loss_function="RMSE",
            eval_metric="RMSE",
            subsample=0.8,
            l2_leaf_reg=4,
            grow_policy="SymmetricTree",
            bootstrap_type="Bernoulli",
            random_seed=RANDOM_SEED,
            verbose=cb_verbose,
            early_stopping_rounds=20,
            task_type="GPU" if use_gpu_for_cb else "CPU",
            **cb_kwargs,
        ),
        "knn": lambda: KNeighborsRegressor(n_neighbors=knn_neighbors, weights="distance", algorithm="auto", leaf_size=30, p=2, n_jobs=worker_jobs),
        "ert": lambda: ExtraTreesRegressor(
            n_estimators=100 if not smoke else forest_estimators,
            max_depth=24,
            min_samples_leaf=8,
            max_features=0.9,
            bootstrap=False,
            n_jobs=worker_jobs,
            random_state=RANDOM_SEED,
            verbose=2 if not smoke else 0,
        ),
    }

    if xgb_backend == "xgboost":
        factories["xgb"] = lambda: XGBRegressor(
            n_estimators=xgb_iters,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=worker_jobs,
            tree_method="hist",
            early_stopping_rounds=20,
            verbosity=1 if not smoke else 0,
        )
    else:
        factories["xgb"] = lambda: GradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=10,
            n_estimators=xgb_iters,
            min_samples_leaf=50,
            random_state=RANDOM_SEED,
            subsample=0.8,
        )
    return factories


def fit_single_model(name, X_fit, y_fit, X_es, y_es, config: RunConfig):
    model = model_factories(config)[name]()
    if name == "lgb":
        callbacks = [lgb.early_stopping(50, verbose=False)]
        if not config.smoke_test:
            callbacks.append(lgb.log_evaluation(100))
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="rmse", callbacks=callbacks)
        return model, None
    if name == "cb":
        model.fit(X_fit, y_fit, eval_set=(X_es, y_es), use_best_model=True)
        return model, None
    if name == "xgb" and get_xgb_backend(config) == "xgboost":
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        return model, None
    if name == "knn":
        scaler = StandardScaler()
        model.fit(scaler.fit_transform(X_fit), y_fit)
        return model, scaler
    model.fit(X_fit, y_fit)
    return model, None


def predict_single_model(name, model, scaler, X):
    if name == "knn":
        return model.predict(scaler.transform(X)).astype(np.float32)
    return model.predict(X).astype(np.float32)


def generate_oof_predictions(X_train, y_train, config: RunConfig, t0=None):
    fold_ids = make_year_balanced_folds(X_train[:, 2], config.n_folds)
    raw_oof = {name: np.zeros(len(X_train), dtype=np.float32) for name in MODEL_ORDER}
    for fold in range(config.n_folds):
        idx_oof = fold_ids == fold
        idx_fit = ~idx_oof
        X_fit = X_train[idx_fit]
        y_fit = y_train[idx_fit]
        X_oof = X_train[idx_oof]
        X_main, y_main, X_es, y_es = make_year_inner_es_split(X_fit, y_fit, frac=0.15, seed=RANDOM_SEED + fold)
        fold_t0 = time.time()
        if t0 is not None:
            update_progress(
                t0,
                stage="oof_fold",
                status="running",
                fold=int(fold + 1),
                total_folds=int(config.n_folds),
                fit_rows=int(len(X_fit)),
                oof_rows=int(len(X_oof)),
            )
        print(f"\n[OOF] Fold {fold + 1}/{config.n_folds} | fit={len(X_fit)} oof={len(X_oof)}")
        for name in MODEL_ORDER:
            train_X = X_main if name in {"lgb", "cb", "xgb"} else X_fit
            train_y = y_main if name in {"lgb", "cb", "xgb"} else y_fit
            model, scaler = fit_single_model(name, train_X, train_y, X_es, y_es, config)
            raw_oof[name][idx_oof] = predict_single_model(name, model, scaler, X_oof)
        fold_metrics = {
            name: float(np.sqrt(mean_squared_error(y_train[idx_oof], raw_oof[name][idx_oof])))
            for name in MODEL_ORDER
        }
        save_json(
            RESULT_DIR / f"oof_fold_{fold + 1}_summary.json",
            {
                "fold": int(fold + 1),
                "total_folds": int(config.n_folds),
                "fit_rows": int(len(X_fit)),
                "oof_rows": int(len(X_oof)),
                "elapsed_seconds_fold": float(time.time() - fold_t0),
                "rmse_by_model": fold_metrics,
            },
        )
        if t0 is not None:
            update_progress(
                t0,
                stage="oof_fold",
                status="completed",
                fold=int(fold + 1),
                total_folds=int(config.n_folds),
                elapsed_seconds_fold=float(time.time() - fold_t0),
                rmse_by_model=fold_metrics,
            )
    return raw_oof, fold_ids


def fit_bias_models(raw_oof, y_train, config: RunConfig):
    bias_models = {}
    for name in MODEL_ORDER:
        residual = y_train - raw_oof[name]
        offset = max(0.0, float(np.quantile(residual, config.bias_quantile)))
        bias_models[name] = {
            "multiplier": float(config.bias_multiplier),
            "offset": offset,
            "raw_mean_bias": float(np.mean(raw_oof[name] - y_train)),
            "biased_mean_bias": float(np.mean(raw_oof[name] * config.bias_multiplier + offset - y_train)),
        }
    return bias_models


def apply_bias(pred_dict, bias_models):
    adjusted = {}
    for name, pred in pred_dict.items():
        spec = bias_models[name]
        adjusted[name] = (pred * spec["multiplier"] + spec["offset"]).astype(np.float32)
    return adjusted


def topk_mean(pred_matrix, k):
    k = max(1, min(int(k), pred_matrix.shape[1]))
    if k == pred_matrix.shape[1]:
        return np.mean(pred_matrix, axis=1)
    topk = np.partition(pred_matrix, pred_matrix.shape[1] - k, axis=1)[:, -k:]
    return np.mean(topk, axis=1)


def build_upper_anchor(base_pred_matrix, X, config: RunConfig):
    max_allowed = compute_max_allowed_np(X[:, PHYS_IDXS], config)
    anchor_raw = topk_mean(base_pred_matrix, config.upper_anchor_top_k).astype(np.float32)
    anchor_cap = (max_allowed * config.upper_anchor_clip_multiplier).astype(np.float32)
    anchor = np.minimum(anchor_raw, anchor_cap).astype(np.float32)
    exceed_mask = anchor > max_allowed
    return anchor, exceed_mask.astype(np.float32), max_allowed


def save_single_model_bundle(name, model, scaler, config: RunConfig):
    backend = get_xgb_backend(config)
    if name == "lgb":
        joblib.dump(model, MODEL_DIR / "lgb_model.pkl")
    elif name == "rf":
        joblib.dump(model, MODEL_DIR / "rf_model.joblib")
    elif name == "cb":
        model.save_model(str(MODEL_DIR / "cb_model.cbm"))
    elif name == "knn":
        with (MODEL_DIR / "knn_bundle.pkl").open("wb") as f:
            pickle.dump((model, scaler), f)
    elif name == "ert":
        joblib.dump(model, MODEL_DIR / "ert_model.joblib")
    elif name == "xgb":
        if backend == "xgboost":
            model.save_model(str(MODEL_DIR / "xgb_model.json"))
        else:
            joblib.dump(model, MODEL_DIR / "xgb_fallback_histgb.joblib")


def train_full_models(X_train, y_train, X_val, y_val, X_test, y_test, bias_models, config: RunConfig, t0):
    bundles = {}
    raw_val = {}
    raw_test = {}
    biased_val = {}
    biased_test = {}
    base_metrics = {"val_raw": {}, "val_biased": {}, "test_raw": {}, "test_biased": {}}
    base_physics = {"val_raw": {}, "val_biased": {}, "test_raw": {}, "test_biased": {}}
    X_main, y_main, X_es, y_es = make_year_inner_es_split(X_train, y_train, frac=0.15, seed=RANDOM_SEED + 999)
    for model_index, name in enumerate(MODEL_ORDER, start=1):
        update_progress(
            t0,
            stage="full_base_model",
            status="running",
            current_model=name,
            model_index=model_index,
            total_models=len(MODEL_ORDER),
        )
        print(f"\n[FULL] Training {name}")
        train_X = X_main if name in {"lgb", "cb", "xgb"} else X_train
        train_y = y_main if name in {"lgb", "cb", "xgb"} else y_train
        model_t0 = time.time()
        model, scaler = fit_single_model(name, train_X, train_y, X_es, y_es, config)
        bundles[name] = {"model": model, "scaler": scaler}
        save_single_model_bundle(name, model, scaler, config)
        raw_val[name] = predict_single_model(name, model, scaler, X_val)
        raw_test[name] = predict_single_model(name, model, scaler, X_test)
        biased_val[name] = apply_bias({name: raw_val[name]}, bias_models)[name]
        biased_test[name] = apply_bias({name: raw_test[name]}, bias_models)[name]
        base_metrics["val_raw"][name] = evaluate_regression(y_val, raw_val[name], f"{name}_val_raw")
        base_metrics["val_biased"][name] = evaluate_regression(y_val, biased_val[name], f"{name}_val_bias")
        base_metrics["test_raw"][name] = evaluate_regression(y_test, raw_test[name], f"{name}_test_raw")
        base_metrics["test_biased"][name] = evaluate_regression(y_test, biased_test[name], f"{name}_test_bias")
        base_physics["val_raw"][name] = summarize_physics_violations(X_val, raw_val[name], config)
        base_physics["val_biased"][name] = summarize_physics_violations(X_val, biased_val[name], config)
        base_physics["test_raw"][name] = summarize_physics_violations(X_test, raw_test[name], config)
        base_physics["test_biased"][name] = summarize_physics_violations(X_test, biased_test[name], config)
        per_model_summary = {
            "model": name,
            "model_index": model_index,
            "total_models": len(MODEL_ORDER),
            "elapsed_seconds_model": float(time.time() - model_t0),
            "metrics": {
                "val_raw": base_metrics["val_raw"][name],
                "val_biased": base_metrics["val_biased"][name],
                "test_raw": base_metrics["test_raw"][name],
                "test_biased": base_metrics["test_biased"][name],
            },
            "physics": {
                "val_raw": base_physics["val_raw"][name],
                "val_biased": base_physics["val_biased"][name],
                "test_raw": base_physics["test_raw"][name],
                "test_biased": base_physics["test_biased"][name],
            },
        }
        save_json(RESULT_DIR / f"base_model_{name}_metrics.json", per_model_summary)
        np.savez(
            RESULT_DIR / f"base_model_{name}_predictions.npz",
            val_raw=raw_val[name],
            val_biased=biased_val[name],
            test_raw=raw_test[name],
            test_biased=biased_test[name],
            y_val=y_val,
            y_test=y_test,
        )
        update_progress(
            t0,
            stage="full_base_model",
            status="completed",
            current_model=name,
            model_index=model_index,
            total_models=len(MODEL_ORDER),
            elapsed_seconds_model=float(time.time() - model_t0),
            test_rmse_raw=base_metrics["test_raw"][name]["rmse"],
            test_rmse_biased=base_metrics["test_biased"][name]["rmse"],
            test_upper_raw=base_physics["test_raw"][name]["upper_count"],
            test_lower_raw=base_physics["test_raw"][name]["lower_count"],
            test_upper_biased=base_physics["test_biased"][name]["upper_count"],
            test_lower_biased=base_physics["test_biased"][name]["lower_count"],
        )
    return bundles, raw_val, raw_test, biased_val, biased_test, base_metrics, base_physics


def predict_bundle_dict(bundles, X):
    return {name: predict_single_model(name, bundles[name]["model"], bundles[name]["scaler"], X) for name in MODEL_ORDER}


class SpatialProbabilisticStacker(nn.Module):
    def __init__(self, spatial_dim, n_models, quantiles):
        super().__init__()
        self.gating_net = nn.Sequential(
            nn.Linear(spatial_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.weight_head = nn.Linear(64, n_models)
        self.risk_head = nn.Linear(64 + n_models, 1)
        self.uplift_head = nn.Linear(64 + n_models, 1)
        self.quantile_head = nn.Sequential(
            nn.Linear(64 + n_models, 32),
            nn.ReLU(),
            nn.Linear(32, len(quantiles)),
        )

    def forward(self, spatial_x, base_preds):
        context = self.gating_net(spatial_x)
        weights = torch.softmax(self.weight_head(context), dim=1)
        joint = torch.cat([context, base_preds], dim=1)
        base_blend = torch.sum(weights * base_preds, dim=1, keepdim=True)
        risk_gate = torch.sigmoid(self.risk_head(joint))
        uplift = F.softplus(self.uplift_head(joint))
        point_pred = base_blend + risk_gate * uplift
        quantiles = self.quantile_head(joint)
        return point_pred, weights, quantiles, risk_gate, uplift


def pinball_loss(preds, target, quantiles):
    loss = 0.0
    for i, q in enumerate(quantiles):
        error = target - preds[:, i : i + 1]
        loss += torch.mean(torch.max(q * error, (q - 1.0) * error))
    return loss


def _fit_stacker_candidate(
    X_sp_tr,
    Z_tr,
    y_tr,
    X_sp_val,
    Z_val,
    y_val,
    train_anchor,
    train_anchor_mask,
    config: RunConfig,
    upper_weight: float,
):
    quantiles = [0.05, 0.5, 0.95]
    model = SpatialProbabilisticStacker(X_sp_tr.shape[1], Z_tr.shape[1], quantiles).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    t_x_tr = torch.tensor(X_sp_tr, dtype=torch.float32, device=DEVICE)
    t_z_tr = torch.tensor(Z_tr, dtype=torch.float32, device=DEVICE)
    t_y_tr = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE).view(-1, 1)
    t_anchor_tr = torch.tensor(train_anchor, dtype=torch.float32, device=DEVICE)
    t_anchor_mask_tr = torch.tensor(train_anchor_mask, dtype=torch.float32, device=DEVICE)
    t_x_val = torch.tensor(X_sp_val, dtype=torch.float32, device=DEVICE)
    t_z_val = torch.tensor(Z_val, dtype=torch.float32, device=DEVICE)
    t_y_val = torch.tensor(y_val, dtype=torch.float32, device=DEVICE).view(-1, 1)
    best_loss = float("inf")
    best_state = None
    patience = 0
    batch_size = config.meta_batch_size if not config.smoke_test else min(8192, len(X_sp_tr))

    for epoch in range(config.meta_epochs):
        model.train()
        perm = torch.randperm(len(X_sp_tr), device=DEVICE)
        for start in range(0, len(X_sp_tr), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            point_pred, _, q_pred, risk_gate, uplift = model(t_x_tr[idx], t_z_tr[idx])
            point_vec = point_pred.squeeze(1)
            uplift_term = (risk_gate * uplift).squeeze(1)
            q_sorted, _ = torch.sort(q_pred, dim=1)
            quant_width = torch.relu(q_sorted[:, -1] - q_sorted[:, 0])
            quant_weight = quant_width / torch.clamp(quant_width.mean(), min=1e-6)
            loss = nn.MSELoss()(point_pred, t_y_tr[idx]) + 0.2 * pinball_loss(q_pred, t_y_tr[idx], quantiles)
            loss = loss + 0.002 * torch.mean(uplift_term**2)
            if upper_weight > 0:
                masked = t_anchor_mask_tr[idx]
                active = masked.sum()
                if active.item() > 0:
                    upper_gap = torch.relu(t_anchor_tr[idx] - point_vec)
                    upper_floor = torch.sum(masked * (1.0 + config.upper_uncertainty_gamma * quant_weight) * (upper_gap**2)) / active
                    safe_mass = torch.clamp((1.0 - masked).sum(), min=1.0)
                    safe_uplift = torch.sum((1.0 - masked) * (uplift_term**2)) / safe_mass
                    loss = loss + upper_weight * upper_floor + 0.1 * upper_weight * safe_uplift
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_point, _, val_q, _, _ = model(t_x_val, t_z_val)
            val_loss = (nn.MSELoss()(val_point, t_y_val) + 0.2 * pinball_loss(val_q, t_y_val, quantiles)).item()

        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= (5 if config.smoke_test else 15):
                break

        if epoch % 10 == 0 or config.smoke_test:
            print(f"[STACKER] epoch={epoch:03d} val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


def choose_stacker_model(
    X_sp_tr,
    Z_tr_main,
    Z_tr_upper,
    y_tr,
    X_tr,
    X_sp_val,
    Z_val_main,
    Z_val_upper,
    y_val,
    X_val,
    config: RunConfig,
    t0=None,
):
    train_anchor, train_anchor_mask, train_max_allowed = build_upper_anchor(Z_tr_upper, X_tr, config)
    val_anchor, _, val_max_allowed = build_upper_anchor(Z_val_upper, X_val, config)
    del train_max_allowed, val_anchor

    candidates = []
    batch_size = config.meta_batch_size if not config.smoke_test else 8192
    target_mid = 0.5 * (config.target_violation_ratio_min + config.target_violation_ratio_max)

    for candidate_index, upper_weight in enumerate(config.stacker_upper_lambdas, start=1):
        if t0 is not None:
            update_progress(
                t0,
                stage="stacker_search",
                status="running",
                candidate_index=int(candidate_index),
                total_candidates=int(len(config.stacker_upper_lambdas)),
                upper_weight=float(upper_weight),
            )
        print(f"[STACKER] trying upper_weight={upper_weight:.3f}")
        model = _fit_stacker_candidate(
            X_sp_tr, Z_tr_main, y_tr, X_sp_val, Z_val_main, y_val, train_anchor, train_anchor_mask, config, upper_weight
        )
        p_val, w_val, q_val = infer_stacker(model, X_sp_val, Z_val_main, batch_size=batch_size)
        rmse = float(np.sqrt(mean_squared_error(y_val, p_val)))
        violations = p_val > val_max_allowed
        violation_ratio = float(np.mean(violations))
        violation_count = int(violations.sum())

        if violation_ratio < config.target_violation_ratio_min:
            penalty = 20000.0 * (config.target_violation_ratio_min - violation_ratio)
        elif violation_ratio > config.target_violation_ratio_max:
            penalty = 8000.0 * (violation_ratio - config.target_violation_ratio_max)
        else:
            penalty = 200.0 * abs(violation_ratio - target_mid)

        score = rmse + penalty
        candidates.append(
            {
                "upper_weight": float(upper_weight),
                "rmse": rmse,
                "violation_ratio": violation_ratio,
                "violation_count": violation_count,
                "score": float(score),
                "model": model,
                "val_pred": p_val,
                "val_weights": w_val,
                "val_quantiles": q_val,
            }
        )
        print(
            f"[STACKER] upper_weight={upper_weight:.3f} "
            f"val_rmse={rmse:.4f} val_viol={violation_count} ({violation_ratio:.4%}) score={score:.4f}"
        )
        if t0 is not None:
            update_progress(
                t0,
                stage="stacker_search",
                status="completed",
                candidate_index=int(candidate_index),
                total_candidates=int(len(config.stacker_upper_lambdas)),
                upper_weight=float(upper_weight),
                val_rmse=rmse,
                violation_ratio=violation_ratio,
                score=float(score),
            )

    feasible = [
        item
        for item in candidates
        if config.target_violation_ratio_min <= item["violation_ratio"] <= config.target_violation_ratio_max
    ]
    if feasible:
        best = min(feasible, key=lambda item: item["rmse"])
    else:
        positive = [item for item in candidates if item["violation_ratio"] > 0]
        if positive:
            best = min(positive, key=lambda item: (abs(item["violation_ratio"] - target_mid), item["rmse"]))
        else:
            best = min(candidates, key=lambda item: item["rmse"])

    diagnostics = {
        "selected_upper_weight": best["upper_weight"],
        "target_violation_ratio_min": config.target_violation_ratio_min,
        "target_violation_ratio_max": config.target_violation_ratio_max,
        "candidates": [
            {
                "upper_weight": item["upper_weight"],
                "rmse": item["rmse"],
                "violation_ratio": item["violation_ratio"],
                "violation_count": item["violation_count"],
                "score": item["score"],
            }
            for item in candidates
        ],
    }
    return best["model"], best["val_pred"], best["val_weights"], best["val_quantiles"], diagnostics


def infer_stacker(model, X_spatial, Z, batch_size):
    model.eval()
    preds = []
    weights = []
    quantiles = []
    with torch.no_grad():
        for start in range(0, len(X_spatial), batch_size):
            stop = start + batch_size
            tx = torch.tensor(X_spatial[start:stop], dtype=torch.float32, device=DEVICE)
            tz = torch.tensor(Z[start:stop], dtype=torch.float32, device=DEVICE)
            point_pred, model_weights, q_pred, _, _ = model(tx, tz)
            preds.append(point_pred.cpu().numpy())
            weights.append(model_weights.cpu().numpy())
            quantiles.append(q_pred.cpu().numpy())
    return np.vstack(preds).flatten().astype(np.float32), np.vstack(weights).astype(np.float32), np.vstack(quantiles).astype(np.float32)


class ResidualPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, stack_pred, phys_feats):
        return self.net(torch.cat([stack_pred, phys_feats], dim=1)).squeeze(1)


class PINNDataset(Dataset):
    def __init__(self, stack_pred, X_raw, y, scaler, risk_mask):
        self.stack_pred = torch.tensor(stack_pred, dtype=torch.float32).unsqueeze(1)
        self.phys_raw = torch.tensor(X_raw[:, PHYS_IDXS], dtype=torch.float32)
        self.phys_scaled = torch.tensor(scaler.transform(X_raw[:, PHYS_IDXS]), dtype=torch.float32)
        self.risk_mask = torch.tensor(risk_mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.stack_pred[idx], self.phys_scaled[idx], self.phys_raw[idx], self.risk_mask[idx], self.y[idx]


def run_pinn(
    p_stack_tr,
    X_tr,
    y_tr,
    q_tr,
    p_stack_val,
    X_val,
    y_val,
    q_val,
    p_stack_test,
    X_test,
    y_test,
    q_test,
    train_upper_anchor,
    val_upper_anchor,
    test_upper_anchor,
    config: RunConfig,
    t0=None,
):
    _, stats_tr = report_physics_quality(X_tr, y_tr, config, "Train")
    _, stats_val = report_physics_quality(X_val, y_val, config, "Val")
    _, stats_test = report_physics_quality(X_test, y_test, config, "Test")
    if stats_tr["overall_bad"] or stats_val["overall_bad"] or stats_test["overall_bad"]:
        raise RuntimeError("Physical-invalid labels remain after upfront filtering.")

    scaler = StandardScaler()
    scaler.fit(X_tr[:, PHYS_IDXS])
    joblib.dump(scaler, MODEL_DIR / "pinn_phys_scaler.pkl")

    batch_size = config.pinn_batch_size if not config.smoke_test else min(8192, len(X_tr))
    workers = 0
    train_max_allowed_np = compute_max_allowed_np(X_tr[:, PHYS_IDXS], config)
    val_max_allowed_np = compute_max_allowed_np(X_val[:, PHYS_IDXS], config)
    test_max_allowed_np = compute_max_allowed_np(X_test[:, PHYS_IDXS], config)
    train_quant_width = quantile_width_matrix(q_tr)
    val_quant_width = quantile_width_matrix(q_val)
    test_quant_width = quantile_width_matrix(q_test)
    width_threshold = float(np.quantile(train_quant_width, config.risk_width_quantile))
    train_risk_mask_np, train_risk_summary = build_risk_mask(
        p_stack_tr, train_upper_anchor, train_max_allowed_np, train_quant_width, width_threshold, config
    )
    val_risk_mask_np, val_risk_summary = build_risk_mask(
        p_stack_val, val_upper_anchor, val_max_allowed_np, val_quant_width, width_threshold, config
    )
    test_risk_mask_np, test_risk_summary = build_risk_mask(
        p_stack_test, test_upper_anchor, test_max_allowed_np, test_quant_width, width_threshold, config
    )
    tr_loader = DataLoader(
        PINNDataset(p_stack_tr, X_tr, y_tr, scaler, train_risk_mask_np),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = DataLoader(
        PINNDataset(p_stack_val, X_val, y_val, scaler, val_risk_mask_np),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    te_loader = DataLoader(
        PINNDataset(p_stack_test, X_test, y_test, scaler, test_risk_mask_np),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    model = ResidualPINN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.pinn_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    amp_scaler = GradScaler(enabled=USE_FP16)
    best_loss = float("inf")
    patience = 0
    w_phys = 0.5
    w_safe = config.pinn_safe_keep_weight
    epoch_diagnostics = []
    for epoch in range(config.pinn_epochs):
        model.train()
        if epoch > 0 and epoch % 5 == 0:
            w_phys = min(1.0, w_phys + 0.1)
            w_safe = max(0.05, w_safe - 0.03)
        train_upper_active = 0
        train_negative_active = 0
        train_seen = 0
        for stack_pred, phys_scaled, phys_raw, risk_mask, target in tr_loader:
            stack_pred = stack_pred.to(DEVICE, non_blocking=True)
            phys_scaled = phys_scaled.to(DEVICE, non_blocking=True)
            phys_raw = phys_raw.to(DEVICE, non_blocking=True)
            risk_mask = risk_mask.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            safe_mask = 1.0 - risk_mask
            optimizer.zero_grad()
            with autocast(enabled=USE_FP16):
                delta = model(stack_pred, phys_scaled)
                pred = stack_pred.squeeze(1) + delta
                mse = nn.MSELoss()(pred, target.squeeze(1))
                phy = physical_constraint_loss(pred, phys_raw, config)
                safe_keep = torch.sum(safe_mask * ((pred - stack_pred.squeeze(1)) ** 2)) / torch.clamp(safe_mask.sum(), min=1.0)
                loss = mse + w_phys * phy + w_safe * safe_keep
                max_allowed_batch = compute_max_allowed_torch(phys_raw, config)
                train_upper_active += int((pred > max_allowed_batch).sum().item())
                train_negative_active += int((pred < 0).sum().item())
                train_seen += int(pred.shape[0])
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

        model.eval()
        val_loss = 0.0
        val_upper_active = 0
        val_seen = 0
        with torch.no_grad():
            for stack_pred, phys_scaled, _, risk_mask, target in val_loader:
                stack_pred = stack_pred.to(DEVICE)
                phys_scaled = phys_scaled.to(DEVICE)
                risk_mask = risk_mask.to(DEVICE)
                target = target.to(DEVICE)
                safe_mask = 1.0 - risk_mask
                with autocast(enabled=USE_FP16):
                    delta = model(stack_pred, phys_scaled)
                    pred = stack_pred.squeeze(1) + delta
                    safe_keep = torch.sum(safe_mask * ((pred - stack_pred.squeeze(1)) ** 2)) / torch.clamp(safe_mask.sum(), min=1.0)
                    val_loss += (nn.MSELoss()(pred, target.squeeze(1)) + w_safe * safe_keep).item()
                    val_seen += int(pred.shape[0])
        val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)
        epoch_diagnostics.append(
            {
                "epoch": int(epoch),
                "val_loss": float(val_loss),
                "train_upper_active": int(train_upper_active),
                "train_negative_active": int(train_negative_active),
                "train_upper_ratio": float(train_upper_active / max(1, train_seen)),
                "train_negative_ratio": float(train_negative_active / max(1, train_seen)),
                "val_seen": int(val_seen),
                "w_phys": float(w_phys),
                "w_safe": float(w_safe),
            }
        )
        if (best_loss - val_loss) > config.pinn_min_delta:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_pinn.pth")
        else:
            patience += 1
            if patience >= config.pinn_early_stop_patience:
                break
        if epoch % 5 == 0 or config.smoke_test:
            print(
                f"[PINN] epoch={epoch:03d} val_mse={val_loss:.4f} "
                f"upper_active={train_upper_active}/{max(1, train_seen)} "
                f"w_phys={w_phys:.2f} w_safe={w_safe:.2f}"
            )
            if t0 is not None:
                update_progress(
                    t0,
                    stage="pinn_train",
                    status="running",
                    epoch=int(epoch),
                    total_epochs=int(config.pinn_epochs),
                    val_loss=float(val_loss),
                    train_upper_active=int(train_upper_active),
                    train_seen=int(train_seen),
                    w_phys=float(w_phys),
                    w_safe=float(w_safe),
                )

    model.load_state_dict(torch.load(MODEL_DIR / "best_pinn.pth", map_location=DEVICE))
    model.eval()
    preds = []
    with torch.no_grad():
        for stack_pred, phys_scaled, _, _, _ in te_loader:
            stack_pred = stack_pred.to(DEVICE)
            phys_scaled = phys_scaled.to(DEVICE)
            with autocast(enabled=USE_FP16):
                delta = model(stack_pred, phys_scaled)
                pred = stack_pred.squeeze(1) + delta
            preds.append(pred.float().cpu().numpy())
    preds_preclip = np.concatenate(preds).astype(np.float32)
    max_allowed = compute_max_allowed_np(X_test[:, PHYS_IDXS], config)
    preclip_physics = summarize_physics_violations(X_test, preds_preclip, config)
    clip_applied = bool(preclip_physics["upper_count"] > 0 or preclip_physics["lower_count"] > 0)
    preds = preds_preclip.copy()
    if clip_applied:
        preds = np.clip(preds, 0.0, max_allowed)
    postclip_physics = summarize_physics_violations(X_test, preds, config)
    diagnostics = {
        "label_filter": {"train": stats_tr, "val": stats_val, "test": stats_test},
        "risk_summary": {"train": train_risk_summary, "val": val_risk_summary, "test": test_risk_summary},
        "quantile_width": {
            "threshold": width_threshold,
            "train_mean": float(np.mean(train_quant_width)),
            "val_mean": float(np.mean(val_quant_width)),
            "test_mean": float(np.mean(test_quant_width)),
        },
        "preclip_physics": preclip_physics,
        "postclip_physics": postclip_physics,
        "clip_applied": clip_applied,
        "epochs": epoch_diagnostics,
    }
    if t0 is not None:
        update_progress(
            t0,
            stage="pinn_train",
            status="completed",
            best_val_loss=float(best_loss),
            kept_test_rows=int(len(y_test)),
            clip_applied=clip_applied,
        )
    return preds_preclip, preds, y_test, X_test, diagnostics


def summarize_prediction_physics(X, preds, config: RunConfig):
    return summarize_physics_violations(X, preds, config)


def save_numpy_artifacts(
    raw_oof,
    biased_oof,
    raw_val,
    biased_val,
    raw_test,
    biased_test,
    y_val,
    y_test,
    stacker_test,
    pinn_test,
    pinn_preclip=None,
):
    np.savez(RESULT_DIR / "oof_raw_ALL6.npz", **raw_oof)
    np.savez(RESULT_DIR / "oof_biased_ALL6.npz", **biased_oof)
    np.savez(RESULT_DIR / "val_raw_ALL6.npz", **raw_val, y_true=y_val)
    np.savez(RESULT_DIR / "val_biased_ALL6.npz", **biased_val, y_true=y_val)
    np.savez(RESULT_DIR / "test_raw_ALL6.npz", **raw_test, y_true=y_test)
    np.savez(RESULT_DIR / "test_biased_ALL6.npz", **biased_test, y_true=y_test)
    payload = {"stacker": stacker_test, "pinn": pinn_test, "y_true": y_test}
    if pinn_preclip is not None:
        payload["pinn_preclip"] = pinn_preclip
    np.savez(RESULT_DIR / "final_predictions.npz", **payload)


def run_pipeline_on_xy(X, y, config: RunConfig, dataset_name: str, raw_label_filter):
    ensure_dirs()
    set_seed()
    t0 = time.time()
    update_progress(
        t0,
        stage="startup",
        status="running",
        output_root=str(OUTPUT_ROOT),
        smoke_test=bool(config.smoke_test),
        dry_run=bool(config.dry_run),
        dataset_name=dataset_name,
    )
    print("===== GLOBAL O2 ENSEMBLE PIPELINE =====")
    print(f"PID: {os.getpid()}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {DEVICE}")
    print(f"XGB backend: {get_xgb_backend(config)}")

    (X_tr, y_tr, idx_tr), (X_val, y_val, idx_val), (X_test, y_test, idx_test), split_summary = split_each_year_811(X, y)
    split_report = {
        "config": asdict(config),
        "dataset_name": dataset_name,
        "backend": {
            "xgboost_available": HAS_XGBOOST,
            "xgb_backend_used": get_xgb_backend(config),
            "device": str(DEVICE),
            "fp16_enabled": USE_FP16,
        },
        "data_counts": {
            "all_after_optional_subsample": int(len(X)),
            "train": int(len(X_tr)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "raw_label_filter": raw_label_filter,
        "per_year_split": split_summary,
    }
    np.save(DATA_DIR / "train_indices.npy", idx_tr)
    np.save(DATA_DIR / "val_indices.npy", idx_val)
    np.save(DATA_DIR / "test_indices.npy", idx_test)
    save_json(RESULT_DIR / "split_summary.json", split_report)
    save_json(RESULT_DIR / "run_config.json", asdict(config))
    update_progress(
        t0,
        stage="data_split",
        status="completed",
        train_rows=int(len(X_tr)),
        val_rows=int(len(X_val)),
        test_rows=int(len(X_test)),
        years=int(len(np.unique(X[:, 2].astype(int)))),
    )
    print(f"Split counts | train={len(X_tr)} val={len(X_val)} test={len(X_test)} | years={len(np.unique(X[:, 2].astype(int)))}")
    if config.dry_run:
        update_progress(t0, stage="finished", status="completed", mode="dry_run")
        print(f"Dry run complete in {(time.time() - t0):.1f}s")
        return

    raw_oof, fold_ids = generate_oof_predictions(X_train=X_tr, y_train=y_tr, config=config, t0=t0)
    np.save(DATA_DIR / "train_fold_ids.npy", fold_ids)
    bias_models = fit_bias_models(raw_oof, y_tr, config)
    save_json(RESULT_DIR / "bias_models.json", bias_models)
    biased_oof = apply_bias(raw_oof, bias_models)
    update_progress(t0, stage="bias_models", status="completed")

    bundles, raw_val, raw_test, biased_val, biased_test, base_metrics, base_physics = train_full_models(
        X_tr, y_tr, X_val, y_val, X_test, y_test, bias_models, config, t0
    )
    save_json(RESULT_DIR / "base_metrics_partial.json", base_metrics)
    save_json(RESULT_DIR / "base_physics_partial.json", base_physics)

    spatial_scaler = StandardScaler()
    X_sp_tr = spatial_scaler.fit_transform(X_tr[:, SPATIAL_IDXS])
    X_sp_val = spatial_scaler.transform(X_val[:, SPATIAL_IDXS])
    X_sp_test = spatial_scaler.transform(X_test[:, SPATIAL_IDXS])
    joblib.dump(spatial_scaler, MODEL_DIR / "stacker_spatial_scaler.pkl")
    update_progress(t0, stage="stacker_prep", status="completed")

    Z_tr = np.column_stack([raw_oof[name] for name in MODEL_ORDER]).astype(np.float32)
    Z_val = np.column_stack([raw_val[name] for name in MODEL_ORDER]).astype(np.float32)
    Z_test = np.column_stack([raw_test[name] for name in MODEL_ORDER]).astype(np.float32)
    Z_tr_upper = np.column_stack([biased_oof[name] for name in MODEL_ORDER]).astype(np.float32)
    Z_val_upper = np.column_stack([biased_val[name] for name in MODEL_ORDER]).astype(np.float32)
    Z_test_upper = np.column_stack([biased_test[name] for name in MODEL_ORDER]).astype(np.float32)
    stacker, p_stack_val, w_val, q_val, stacker_selection = choose_stacker_model(
        X_sp_tr, Z_tr, Z_tr_upper, y_tr, X_tr, X_sp_val, Z_val, Z_val_upper, y_val, X_val, config, t0=t0
    )
    torch.save(stacker.state_dict(), MODEL_DIR / "stacker_all6.pth")
    save_json(RESULT_DIR / "stacker_selection.json", stacker_selection)

    batch_size = config.meta_batch_size if not config.smoke_test else 8192
    p_stack_tr, w_tr, q_tr = infer_stacker(stacker, X_sp_tr, Z_tr, batch_size=batch_size)
    p_stack_test, w_test, q_test = infer_stacker(stacker, X_sp_test, Z_test, batch_size=batch_size)
    train_upper_anchor, _, _ = build_upper_anchor(Z_tr_upper, X_tr, config)
    val_upper_anchor, _, _ = build_upper_anchor(Z_val_upper, X_val, config)
    test_upper_anchor, _, _ = build_upper_anchor(Z_test_upper, X_test, config)
    stacker_metrics = {
        "val": evaluate_regression(y_val, p_stack_val, "stacker_val"),
        "test": evaluate_regression(y_test, p_stack_test, "stacker_test"),
        "test_mean_weights": {name: float(val) for name, val in zip(MODEL_ORDER, np.mean(w_test, axis=0))},
        "test_mean_uncertainty_width": float(np.mean(quantile_width_matrix(q_test))),
        "selection": stacker_selection,
    }
    stacker_physics = {
        "val": summarize_physics_violations(X_val, p_stack_val, config),
        "test": summarize_physics_violations(X_test, p_stack_test, config),
        "uncertainty": {
            "train_mean_width": float(np.mean(quantile_width_matrix(q_tr))),
            "val_mean_width": float(np.mean(quantile_width_matrix(q_val))),
            "test_mean_width": float(np.mean(quantile_width_matrix(q_test))),
        },
    }
    save_json(RESULT_DIR / "stacker_metrics.json", stacker_metrics)
    save_json(RESULT_DIR / "stacker_physics.json", stacker_physics)
    update_progress(
        t0,
        stage="stacker_train",
        status="completed",
        stacker_test_rmse=stacker_metrics["test"]["rmse"],
        stacker_test_r2=stacker_metrics["test"]["r2"],
        selected_upper_weight=stacker_selection["selected_upper_weight"],
    )

    if config.skip_pinn:
        pinn_preclip = p_stack_test.copy()
        pinn_test = p_stack_test.copy()
        pinn_metrics = {
            "test_preclip": stacker_metrics["test"],
            "test_postclip": stacker_metrics["test"],
            "test": stacker_metrics["test"],
            "skipped": True,
        }
        y_pinn = y_test
        X_pinn = X_test
        pinn_training_diag = {"clip_applied": False}
        update_progress(t0, stage="pinn_train", status="skipped")
    else:
        pinn_preclip, pinn_test, y_pinn, X_pinn, pinn_training_diag = run_pinn(
            p_stack_tr,
            X_tr,
            y_tr,
            q_tr,
            p_stack_val,
            X_val,
            y_val,
            q_val,
            p_stack_test,
            X_test,
            y_test,
            q_test,
            train_upper_anchor,
            val_upper_anchor,
            test_upper_anchor,
            config,
            t0=t0,
        )
        pinn_preclip_metrics = evaluate_regression(y_pinn, pinn_preclip, "pinn_test_preclip")
        pinn_postclip_metrics = evaluate_regression(y_pinn, pinn_test, "pinn_test")
        pinn_metrics = {
            "test_preclip": pinn_preclip_metrics,
            "test_postclip": pinn_postclip_metrics,
            "test": pinn_postclip_metrics,
            "skipped": False,
            "training": pinn_training_diag,
        }
        save_json(RESULT_DIR / "pinn_metrics.json", pinn_metrics)

    save_numpy_artifacts(raw_oof, biased_oof, raw_val, biased_val, raw_test, biased_test, y_val, y_test, p_stack_test, pinn_test, pinn_preclip=pinn_preclip)
    np.savez(
        RESULT_DIR / "stacker_aux.npz",
        train_pred=p_stack_tr,
        val_pred=p_stack_val,
        test_pred=p_stack_test,
        train_weights=w_tr,
        val_weights=w_val,
        test_weights=w_test,
        train_quantiles=q_tr,
        val_quantiles=q_val,
        test_quantiles=q_test,
    )

    physics_summary = {
        "raw_label_filter": raw_label_filter,
        "base_physics": base_physics,
        "stacker": stacker_physics,
        "pinn_preclip": pinn_training_diag.get("preclip_physics", summarize_physics_violations(X_pinn, pinn_preclip, config)),
        "pinn_postclip": pinn_training_diag.get("postclip_physics", summarize_physics_violations(X_pinn, pinn_test, config)),
        "clip_applied": pinn_training_diag.get("clip_applied", False),
        "label_filter": pinn_training_diag.get("label_filter", {}),
    }
    run_summary = {
        "config": asdict(config),
        "dataset_name": dataset_name,
        "backend": {
            "xgboost_available": HAS_XGBOOST,
            "xgb_backend_used": get_xgb_backend(config),
            "device": str(DEVICE),
        },
        "pid": int(os.getpid()),
        "split_counts": split_report["data_counts"],
        "raw_label_filter": raw_label_filter,
        "bias_models": bias_models,
        "base_metrics": base_metrics,
        "base_physics": base_physics,
        "stacker_metrics": stacker_metrics,
        "stacker_physics": stacker_physics,
        "pinn_metrics": pinn_metrics,
        "physics_summary": physics_summary,
        "elapsed_seconds": float(time.time() - t0),
    }
    save_json(RESULT_DIR / "run_summary.json", run_summary)
    report_summary = {
        "dataset_name": dataset_name,
        "split_counts": split_report["data_counts"],
        "raw_label_filter": raw_label_filter,
        "best_base_test_rmse": {
            "model": min(base_metrics["test_raw"], key=lambda name: base_metrics["test_raw"][name]["rmse"]),
            "rmse": min(item["rmse"] for item in base_metrics["test_raw"].values()),
        },
        "stacker_test": stacker_metrics["test"],
        "stacker_physics_test": stacker_physics["test"],
        "pinn_test_preclip": pinn_metrics["test_preclip"],
        "pinn_test_postclip": pinn_metrics["test_postclip"],
        "pinn_clip_applied": pinn_training_diag.get("clip_applied", False),
        "pinn_preclip_physics": pinn_training_diag.get("preclip_physics", {}),
        "pinn_postclip_physics": pinn_training_diag.get("postclip_physics", {}),
    }
    save_json(RESULT_DIR / "report_summary.json", report_summary)
    update_progress(
        t0,
        stage="finished",
        status="completed",
        stacker_test_rmse=stacker_metrics["test"]["rmse"],
        pinn_test_rmse=pinn_metrics["test"]["rmse"],
    )
    print(f"Pipeline complete in {(time.time() - t0) / 60.0:.2f} min")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Global nine-ocean-region stacking + PINN pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare cleaned data and region splits only.")
    parser.add_argument("--smoke-test", action="store_true", help="Use smaller estimators and epochs for a quick end-to-end run.")
    parser.add_argument("--skip-pinn", action="store_true", help="Run only base models + stacker.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional subsample size after global filtering.")
    parser.add_argument("--force-xgb-fallback", action="store_true", help="Use local HistGradientBoosting fallback even if xgboost exists.")
    parser.add_argument("--single-region", action="store_true", help="Run one region dataset only.")
    parser.add_argument("--region-data", type=str, default="", help="Path to saved regional npz dataset.")
    parser.add_argument("--region-name", type=str, default="", help="Region name for single-region execution.")
    parser.add_argument("--region-workers", type=int, default=9, help="Parallel worker count for the 9-region orchestration.")
    parser.add_argument("--ocean-base-shp", type=str, default=str(DEFAULT_OCEAN_BASE_SHP), help="Path to ne_10m_ocean.shp")
    parser.add_argument("--marine-polys-shp", type=str, default=str(DEFAULT_MARINE_POLYS_SHP), help="Path to ne_10m_geography_marine_polys.shp")
    parser.add_argument("--python-exe", type=str, default="", help="Python executable used to launch regional subprocesses. Defaults to the current interpreter.")
    return parser


def resolve_python_executable(args) -> str:
    if args.python_exe:
        return args.python_exe
    if sys.executable:
        return sys.executable
    return "python"


def build_child_command(script_path: Path, region_name: str, region_data_path: Path, args):
    command = [resolve_python_executable(args), str(script_path), "--single-region", "--region-name", region_name, "--region-data", str(region_data_path)]
    if args.smoke_test:
        command.append("--smoke-test")
    if args.skip_pinn:
        command.append("--skip-pinn")
    if args.force_xgb_fallback:
        command.append("--force-xgb-fallback")
    return command


def run_child_command(command):
    proc = subprocess.run(command, capture_output=True, text=True)
    return {"command": command, "returncode": int(proc.returncode), "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


def collect_existing_region_jobs(region_data_dir: Path):
    region_jobs = []
    missing_regions = []
    for region_name in REGION_ORDER:
        region_data_path = region_data_dir / f"{slugify_region(region_name)}.npz"
        if region_data_path.exists():
            region_jobs.append((region_name, region_data_path))
        else:
            missing_regions.append(region_name)
    return region_jobs, missing_regions


def orchestrate_regions(args, config: RunConfig):
    set_output_root(Path(r"/home/bingxing2/home/scx7l1f/rec/ensemble/ocean9"))
    ensure_dirs()
    set_seed()
    t0 = time.time()
    update_progress(t0, stage="startup", status="running", output_root=str(OUTPUT_ROOT), orchestration="regions")
    print("===== OCEAN9 ORCHESTRATION =====")
    print(f"PID: {os.getpid()}")
    print(f"Device: {DEVICE}")
    print(f"Data path: {DATA_PATH}")

    region_data_dir = DATA_DIR / "regions"
    region_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Python exe for child regions: {resolve_python_executable(args)}")
    region_jobs, missing_regions = collect_existing_region_jobs(region_data_dir)
    if not missing_regions:
        region_summary = {
            "reuse_existing_region_data": True,
            "region_data_dir": str(region_data_dir),
            "assigned_rows": None,
            "unassigned_rows": None,
            "regions": {
                region_name: {
                    "rows": int(len(np.load(region_data_path, allow_pickle=True)["y"])),
                    "data_path": str(region_data_path),
                }
                for region_name, region_data_path in region_jobs
            },
        }
        save_json(RESULT_DIR / "region_assignment_summary.json", region_summary)
        update_progress(
            t0,
            stage="region_assignment",
            status="completed",
            reuse_existing_region_data=True,
            reused_regions=len(region_jobs),
        )
        print(f"Reusing existing regional datasets from {region_data_dir}")
        if config.dry_run:
            update_progress(t0, stage="finished", status="completed", mode="dry_run", reuse_existing_region_data=True)
            return
    else:
        raw = np.load(DATA_PATH, allow_pickle=True)
        filtered = global_filter(raw)
        X_all = filtered[:, :-1].astype(np.float32)
        y_all = filtered[:, -1].astype(np.float32)
        X_clean, y_clean, _, raw_label_filter = filter_invalid_labels(X_all, y_all, config, "all_filtered")
        X, y = maybe_subsample_xy(X_clean, y_clean, config.max_rows)
        save_json(RESULT_DIR / "raw_label_physics_filter.json", raw_label_filter)

        region_geometries = build_ocean_region_geometries(Path(args.marine_polys_shp))
        region_payloads, region_summary = build_region_datasets(X, y, region_geometries)
        region_summary["raw_label_filter"] = raw_label_filter
        region_summary["reuse_existing_region_data"] = False
        region_summary["missing_regions_before_build"] = missing_regions
        save_json(RESULT_DIR / "region_assignment_summary.json", region_summary)
        update_progress(
            t0,
            stage="region_assignment",
            status="completed",
            assigned_rows=region_summary["assigned_rows"],
            unassigned_rows=region_summary["unassigned_rows"],
            reuse_existing_region_data=False,
        )
        if config.dry_run:
            update_progress(t0, stage="finished", status="completed", mode="dry_run")
            return

        region_jobs = []
        for region_name in REGION_ORDER:
            X_region, y_region = region_payloads[region_name]
            if len(y_region) == 0:
                continue
            region_data_path = region_data_dir / f"{slugify_region(region_name)}.npz"
            np.savez_compressed(region_data_path, X=X_region, y=y_region, region_name=np.array(region_name))
            region_jobs.append((region_name, region_data_path))
    if not region_jobs:
        raise RuntimeError("No regional samples were assigned; check region shapefiles or sampling settings.")

    script_path = Path(__file__).resolve()
    summaries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.region_workers, len(region_jobs)))) as executor:
        future_map = {
            executor.submit(run_child_command, build_child_command(script_path, region_name, region_data_path, args)): region_name
            for region_name, region_data_path in region_jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            region_name = future_map[future]
            result = future.result()
            summaries[region_name] = result
            print(f"[REGION] {region_name} returncode={result['returncode']}")

    save_json(RESULT_DIR / "region_parallel_summary.json", summaries)
    failures = {k: v for k, v in summaries.items() if v["returncode"] != 0}
    if failures:
        raise RuntimeError(f"Region training failed for {list(failures.keys())}")
    update_progress(t0, stage="finished", status="completed", orchestration="regions")


def run_single_region(args, config: RunConfig):
    region_name = args.region_name or Path(args.region_data).stem
    region_root = Path(r"/home/bingxing2/home/scx7l1f/rec/ensemble/ocean9") / slugify_region(region_name)
    set_output_root(region_root)
    ensure_dirs()
    payload = np.load(args.region_data, allow_pickle=True)
    X = payload["X"].astype(np.float32)
    y = payload["y"].astype(np.float32)
    raw_label_filter = {
        "name": region_name,
        "before_rows": int(len(y)),
        "kept_rows": int(len(y)),
        "removed_rows": 0,
        "removed_ratio": 0.0,
        "upper_bad": 0,
        "lower_bad": 0,
        "non_finite": 0,
    }
    run_pipeline_on_xy(X, y, config, region_name, raw_label_filter)


def main():
    args = build_arg_parser().parse_args()
    config = RunConfig(
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        max_rows=args.max_rows,
        force_xgb_fallback=args.force_xgb_fallback,
        skip_pinn=args.skip_pinn,
    )
    if config.smoke_test:
        config.meta_epochs = 8
        config.pinn_epochs = 8
        config.meta_batch_size = 8192
        config.pinn_batch_size = 8192

    if args.single_region:
        run_single_region(args, config)
    else:
        orchestrate_regions(args, config)


if __name__ == "__main__":
    main()
