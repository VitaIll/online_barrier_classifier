"""Train the offline CatBoost model on this repo's existing labeled feature parquet.

Replaces the 13-tree stub at ``artifacts/offline_model/model.cbm`` with a real
3-member ensemble trained on the full feature set.

Recipe distilled from the sister repo (``~/Desktop/barrier_classifier``):
  * CatBoost classifier, depth 6, L2 leaf reg 3.0, learning rate 0.03
  * Iterations 2000 with early stopping at 100 rounds (validation log loss)
  * Time-ordered split: first 80% train, last 20% validation (no shuffling)
  * Allow-list of feature columns: everything in the parquet EXCEPT the
    explicit non-feature set ({open_time, close_time, label, segment_id,
    bar_in_segment, close}). NaNs are passed through (CatBoost handles them).
  * 3-member ensemble with seeds {42, 43, 44} so the FrozenCatBoostPredictor
    can compute virtual-ensemble σ_VE.

Outputs (overwrites existing):
  artifacts/offline_model/model.cbm           — first-member alias (used when ensemble_n=1)
  artifacts/offline_model/model_seed0.cbm     — ensemble member 0 (seed=42)
  artifacts/offline_model/model_seed1.cbm     — ensemble member 1 (seed=43)
  artifacts/offline_model/model_seed2.cbm     — ensemble member 2 (seed=44)
  artifacts/offline_model/selected_features.json  — actual feature list

Usage:
  python scripts/train_offline_model.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)

logger = logging.getLogger(__name__)

# Constants from sister's src/utils.py
CB_ITERATIONS = 2000
CB_LEARNING_RATE = 0.03
CB_DEPTH = 6
CB_L2_LEAF_REG = 3.0
CB_EARLY_STOPPING = 100
CB_BORDER_COUNT = 128
CB_LEAF_EST_ITERS = 10

# Asymmetric risk-averse weighting (from sister utils.py defaults).
# For negative-class samples (m_k < phi, didn't hit barrier):
#   d_k = max(0, phi - m_k)        (distance below barrier; deep losses → big d)
#   w_k = exp(lambda * d_k)        (capped at w_max)
#   lambda = log(w_max) / d_star   (continuity at the cap)
#   d_star = quantile(d_neg, 1 - q_tail)
# Positive-class samples get w = 1 by default (use_pos=False) — purely
# asymmetric: penalise misclassifying the deep-down moves more heavily.
WEIGHT_USE_BARRIER_DISTANCE = True
WEIGHT_DIST_W_MAX = 5.0
WEIGHT_DIST_Q_TAIL = 0.001
WEIGHT_DIST_USE_POSITIVE = False  # asymmetric

# Non-feature columns (label + identifiers + raw price level + label aux)
NON_FEATURE_COLS = {
    "open_time", "close_time", "close", "segment_id", "bar_in_segment",
    "label",
    # Label-aux columns (joined in from labeled parquet) — must NOT be features
    "log_excursion", "next_high", "next_segment_id",
    # Sample weight column (added downstream) — must NOT be a feature
    "weight",
}

# Paths
DATA_PARQUET = Path("data/model_data/BTCUSDT/bars_20m_features.parquet")
LABELED_PARQUET = Path("data/cleansed_data/btcusdt_20m_labeled.parquet")  # has log_excursion (= m_k)
PHI = 0.0041113  # barrier alpha (matches broker.label_alpha and broker.take_profit_log)
OUT_DIR = Path("artifacts/offline_model")
TRAIN_FRAC = 0.80   # walk-forward: first 80% train, last 20% validation


def select_features(columns: list[str]) -> list[str]:
    """Return the feature columns the model should train on.

    Two filters:
      1. Drop the explicit non-feature set ({open_time, label, ...}).
      2. Restrict to the columns the streaming pipeline actually produces
         at runtime — otherwise the FrozenCatBoostPredictor feeds NaN at
         predict-time and the model collapses to its default tree path.
         The streaming feature catalog is the source of truth.
    """
    from wagie.features.catalog import default_streaming_features
    from wagie.features import FeatureBuilder
    streaming = set(FeatureBuilder(default_streaming_features()).feature_names)
    base = [c for c in columns if c not in NON_FEATURE_COLS]
    aligned = [c for c in base if c in streaming]
    dropped = sorted(set(base) - set(aligned))
    if dropped:
        logger.warning("Dropping %d parquet features absent from streaming "
                       "catalog (else NaN at predict-time). First 5 dropped: %s",
                       len(dropped), dropped[:5])
    return aligned


def time_split(df: pl.DataFrame, train_frac: float = TRAIN_FRAC) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Walk-forward time split. Sorted by open_time, first train_frac is train."""
    df = df.sort("open_time")
    n = len(df)
    cut = int(n * train_frac)
    return df.slice(0, cut), df.slice(cut, n - cut)


def compute_asymmetric_weights(
    m_k: np.ndarray, y: np.ndarray, phi: float = PHI,
    *, w_max: float = WEIGHT_DIST_W_MAX, q_tail: float = WEIGHT_DIST_Q_TAIL,
    use_pos: bool = WEIGHT_DIST_USE_POSITIVE,
) -> tuple[np.ndarray, dict]:
    """Risk-averse asymmetric per-sample weights (sister utils.py recipe).

    Negatives: w = exp(lambda * d_k) where d_k = max(0, phi - m_k).
        lambda = log(w_max) / d_star, d_star = quantile(d_neg, 1 - q_tail).
        Caps at w_max (continuity at the cap).
        => Deep losses (large d) get exponentially HIGHER weight.
        => Almost-hits (d ≈ 0) get weight ≈ 1.
    Positives: w = 1 (default, use_pos=False).
    NaN m_k: w = 1 (defensive, shouldn't happen for label != null rows).
    """
    m = np.asarray(m_k, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(m)
    w = np.ones(n, dtype=float)

    # Replace NaN m_k with phi (so d=0 → w=1; defensive)
    m = np.where(np.isnan(m), phi, m)
    d_k = np.maximum(0.0, phi - m)

    neg_mask = (y == 0)
    pos_mask = (y == 1)

    info: dict = {"phi": float(phi), "n_pos": int(pos_mask.sum()),
                  "n_neg": int(neg_mask.sum())}

    if neg_mask.sum() > 0:
        d_neg = d_k[neg_mask]
        d_star = float(np.quantile(d_neg, 1.0 - q_tail))
        d_max = float(d_neg.max())
        if d_star > 0.0 and w_max > 1.0:
            lam = math.log(w_max) / d_star
            log_cap = math.log(w_max)
            w_neg = np.exp(np.minimum(lam * d_neg, log_cap))
        else:
            lam = 0.0
            w_neg = np.ones_like(d_neg)
        w[neg_mask] = w_neg
        info["d_star_neg"] = d_star
        info["d_max_neg"] = d_max
        info["lambda_neg"] = float(lam)
        info["n_capped_neg"] = int((d_neg >= d_star).sum())
        info["w_neg_min"] = float(w_neg.min())
        info["w_neg_max"] = float(w_neg.max())
        info["w_neg_mean"] = float(w_neg.mean())

    if use_pos and pos_mask.sum() > 0:
        # Positive class: distance ABOVE barrier (g_k = m_k - phi)
        # Symmetric formulation, but disabled by default
        g_k = np.maximum(0.0, m[pos_mask] - phi)
        g_star = float(np.quantile(g_k, 1.0 - q_tail))
        if g_star > 0.0 and w_max > 1.0:
            lam_pos = math.log(w_max) / g_star
            w_pos = np.exp(np.minimum(lam_pos * g_k, math.log(w_max)))
            w[pos_mask] = w_pos
            info["lambda_pos"] = float(lam_pos)
            info["g_star_pos"] = g_star

    info["effective_n"] = float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0
    info["w_overall_min"] = float(w.min())
    info["w_overall_max"] = float(w.max())
    info["w_overall_mean"] = float(w.mean())
    return w, info


def add_log_excursion_and_weights(df: pl.DataFrame) -> pl.DataFrame:
    """Compute log_excursion in-place from the features parquet's log_high / log_close.

    log_excursion[k] = log_high[k+1] - log_close[k]   (segment-aware: NaN at boundaries)
    Matches src/wagie/offline/label.py::compute_one_sided_excursion semantics.
    """
    required = {"log_high", "log_close", "segment_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"features parquet missing required cols: {missing}")
    df = df.sort("open_time").with_columns(
        next_log_high=pl.col("log_high").shift(-1),
        next_segment_id=pl.col("segment_id").shift(-1),
    ).with_columns(
        log_excursion=pl.when(pl.col("segment_id") == pl.col("next_segment_id"))
        .then(pl.col("next_log_high") - pl.col("log_close"))
        .otherwise(None),
    ).drop(["next_log_high", "next_segment_id"])
    return df


def train_one(
    train_X: np.ndarray, train_y: np.ndarray,
    val_X: np.ndarray, val_y: np.ndarray,
    feature_names: list[str], *,
    seed: int,
    train_w: np.ndarray | None = None,
    val_w: np.ndarray | None = None,
) -> CatBoostClassifier:
    """Train a single CatBoost member with optional per-sample weights."""
    pool_train = Pool(data=train_X, label=train_y, feature_names=feature_names, weight=train_w)
    pool_val   = Pool(data=val_X,   label=val_y,   feature_names=feature_names, weight=val_w)

    model = CatBoostClassifier(
        iterations=CB_ITERATIONS,
        learning_rate=CB_LEARNING_RATE,
        depth=CB_DEPTH,
        l2_leaf_reg=CB_L2_LEAF_REG,
        border_count=CB_BORDER_COUNT,
        leaf_estimation_iterations=CB_LEAF_EST_ITERS,
        early_stopping_rounds=CB_EARLY_STOPPING,
        random_seed=seed,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=200,
        thread_count=-1,           # use all cores
        allow_writing_files=False,
    )
    model.fit(pool_train, eval_set=pool_val, use_best_model=True)
    return model


def evaluate(model: CatBoostClassifier, X: np.ndarray, y: np.ndarray, label: str) -> dict:
    p = model.predict_proba(X)[:, 1]
    if len(np.unique(y)) > 1:
        auc = float(roc_auc_score(y, p))
        prauc = float(average_precision_score(y, p))
    else:
        auc = float("nan"); prauc = float("nan")
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))
    bs = float(brier_score_loss(y, p))
    base = float(np.mean(y))
    return {
        "label": label, "n": int(len(y)), "base_rate": base,
        "log_loss": ll, "brier": bs, "roc_auc": auc, "pr_auc": prauc,
        "n_trees": int(model.tree_count_),
        "p_min": float(p.min()), "p_max": float(p.max()),
        "p_mean": float(p.mean()), "p_std": float(p.std()),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not DATA_PARQUET.is_file():
        logger.error("missing parquet: %s", DATA_PARQUET)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", DATA_PARQUET)
    df = pl.read_parquet(DATA_PARQUET)
    logger.info("Rows=%d, cols=%d", len(df), len(df.columns))

    # Join log_excursion (= m_k) from the labeled parquet for asymmetric weights
    df = add_log_excursion_and_weights(df)
    logger.info("Joined log_excursion. Cols now: %d", len(df.columns))

    feature_cols = select_features(df.columns)
    logger.info("Feature columns: %d (excluded: %d non-feature cols)",
                len(feature_cols), len(NON_FEATURE_COLS))

    train_df, val_df = time_split(df, TRAIN_FRAC)
    logger.info("Train: %d rows (%s -> %s)", len(train_df),
                train_df["open_time"].min(), train_df["open_time"].max())
    logger.info("Val:   %d rows (%s -> %s)", len(val_df),
                val_df["open_time"].min(), val_df["open_time"].max())
    logger.info("Train base rate: %.4f", float(train_df["label"].mean()))
    logger.info("Val   base rate: %.4f", float(val_df["label"].mean()))

    # Drop any rows where label is null (boundary effect from log_excursion = NaN)
    train_df = train_df.filter(pl.col("label").is_not_null())
    val_df   = val_df.filter(pl.col("label").is_not_null())
    logger.info("After dropping null-label rows: train=%d val=%d", len(train_df), len(val_df))

    # Materialize numpy arrays once (avoids re-conversion per seed)
    train_X = train_df.select(feature_cols).to_numpy()
    train_y = train_df["label"].to_numpy().astype(int)
    val_X   = val_df.select(feature_cols).to_numpy()
    val_y   = val_df["label"].to_numpy().astype(int)
    train_m_k = train_df["log_excursion"].to_numpy()
    val_m_k   = val_df["log_excursion"].to_numpy()
    logger.info("Arrays: train_X %s   val_X %s", train_X.shape, val_X.shape)

    # Asymmetric risk-averse weights (negatives only by default)
    train_w, train_winfo = compute_asymmetric_weights(train_m_k, train_y, phi=PHI)
    val_w,   val_winfo   = compute_asymmetric_weights(val_m_k,   val_y,   phi=PHI)
    logger.info("Asymmetric weighting (TRAIN): phi=%.6f, d_star_neg=%.6f, lambda=%.2f, "
                "n_capped=%d, w_neg range=[%.3f, %.3f] mean=%.3f",
                train_winfo["phi"], train_winfo["d_star_neg"], train_winfo["lambda_neg"],
                train_winfo["n_capped_neg"], train_winfo["w_neg_min"],
                train_winfo["w_neg_max"], train_winfo["w_neg_mean"])
    logger.info("  effective_n=%d/%d (%.1f%%)",
                int(train_winfo["effective_n"]), len(train_w),
                100 * train_winfo["effective_n"] / len(train_w))

    seeds = [42, 43, 44]
    members = []
    eval_rows = []
    for i, sd in enumerate(seeds):
        logger.info("=" * 60)
        logger.info("Training member %d (seed=%d)", i, sd)
        t0 = time.time()
        m = train_one(train_X, train_y, val_X, val_y, feature_cols,
                      seed=sd, train_w=train_w, val_w=val_w)
        dt = time.time() - t0
        logger.info("Member %d trained in %.1fs, n_trees=%d (early-stopped)", i, dt, m.tree_count_)
        members.append(m)
        # Evaluate
        eval_rows.append(evaluate(m, train_X, train_y, f"seed{sd}_train"))
        eval_rows.append(evaluate(m, val_X,   val_y,   f"seed{sd}_val"))
        # Save member
        out = OUT_DIR / f"model_seed{i}.cbm"
        m.save_model(str(out))
        logger.info("saved %s", out)

    # Save the first member as the canonical model.cbm too (used when ensemble_n=1)
    members[0].save_model(str(OUT_DIR / "model.cbm"))
    logger.info("saved %s (alias of seed0)", OUT_DIR / "model.cbm")

    # Update selected_features.json
    sel_path = OUT_DIR / "selected_features.json"
    sel_path.write_text(json.dumps({"features": feature_cols}, indent=2))
    logger.info("saved %s with %d features", sel_path, len(feature_cols))

    # Ensemble evaluation: avg probability across members
    ps = np.stack([m.predict_proba(val_X)[:, 1] for m in members], axis=0)
    p_ens = ps.mean(axis=0)
    sigma_ve = ps.std(axis=0, ddof=0)
    ens_eval = {
        "label": "ensemble_val", "n": len(val_y),
        "base_rate": float(val_y.mean()),
        "log_loss": float(log_loss(val_y, np.clip(p_ens, 1e-7, 1 - 1e-7))),
        "brier": float(brier_score_loss(val_y, p_ens)),
        "roc_auc": float(roc_auc_score(val_y, p_ens)) if len(np.unique(val_y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(val_y, p_ens)) if len(np.unique(val_y)) > 1 else float("nan"),
        "p_min": float(p_ens.min()), "p_max": float(p_ens.max()),
        "p_mean": float(p_ens.mean()), "p_std": float(p_ens.std()),
        "sigma_ve_mean": float(sigma_ve.mean()), "sigma_ve_max": float(sigma_ve.max()),
    }
    eval_rows.append(ens_eval)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("%-22s %6s %8s %8s %8s %8s %8s %8s",
                "Set", "n", "base", "logloss", "brier", "ROC-AUC", "PR-AUC", "p_max")
    for r in eval_rows:
        logger.info("%-22s %6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f",
                    r["label"], r["n"], r["base_rate"],
                    r["log_loss"], r["brier"], r["roc_auc"], r["pr_auc"], r["p_max"])
    logger.info("ensemble σ_VE on val: mean=%.4f max=%.4f",
                ens_eval["sigma_ve_mean"], ens_eval["sigma_ve_max"])

    # Persist eval metadata
    (OUT_DIR / "training_eval.json").write_text(
        json.dumps({"eval": eval_rows, "feature_count": len(feature_cols),
                    "train_rows": len(train_df), "val_rows": len(val_df),
                    "params": {
                        "iterations": CB_ITERATIONS, "learning_rate": CB_LEARNING_RATE,
                        "depth": CB_DEPTH, "l2_leaf_reg": CB_L2_LEAF_REG,
                        "early_stopping": CB_EARLY_STOPPING,
                        "border_count": CB_BORDER_COUNT, "seeds": seeds,
                    }}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("saved %s", OUT_DIR / "training_eval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
