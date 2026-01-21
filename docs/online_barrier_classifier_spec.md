# Online Barrier Classifier: Implementation Specification (VALIDATED)

**Version:** 1.1  [MODIFIED]  
**Status:** Implementation-Ready (Validated)  [MODIFIED]  
**Target:** Agentic Coding Agent  
**Validation Role:** Senior Technical Validation Engineer  [ADDED]

---

## Validation Summary  [ADDED]
- Total issues found: 25
- Critical errors corrected: 10
- Clarifications added: 14
- External verifications: 14

## Change Log  [ADDED]
| Section | Change Type | Description | Justification | Original Line (v1.0) |
|---------|-------------|-------------|---------------|-----------------------|
| 2.1 | MODIFIED | Added dependency constraints compatible with River `0.21.0` (notably `numpy<2.0` and `scipy`) | Prevents resolver drift and import/runtime failures | 22 |
| 2.2.1 | MODIFIED | Corrected River rolling API usage (`utils.Rolling(stats.Mean/Var)`) and ARF import (`river.forest.ARFClassifier`) | Verified against River `0.21.0` public API | 55 |
| 2.2.3 | MODIFIED | Corrected `calibration_curve` import (`sklearn.calibration`) | Verified against scikit-learn `1.8.0` public API | 154 |
| 3.2 | ADDED | Added `segment_id` to decision-bar schema | Required for gap-safe labeling and state resets | 204 |
| 3.3 | CLARIFIED | Stated label is a one-sided, horizon-1 special case of triple-barrier labeling | Aligns claims with López de Prado (2018) terminology | 224 |
| 4.1 | MODIFIED | Updated Binance timestamp unit handling (ms vs µs from 2025-01-01) | Verified against `binance-public-data` README | 341 |
| 6.1 | MODIFIED | Added ms/µs timestamp inference + normalization utilities | Supports Binance timestamp change and internal ms standardization | 457 |
| 6.2 | MODIFIED | Specified exact gap/segment behavior in aggregator; removed invalid River `Transformer` typing | Eliminates ambiguity and prevents cross-gap leakage | 672 |
| 6.3 | MODIFIED | Implemented River-compatible rolling stats construction; defined window warm-up semantics | Verified against River `0.21.0` | 787 |
| 8.1 | MODIFIED | Persisted raw ZIPs; normalized timestamps to ms; created `segment_id`; recorded gap metadata | Enables auditability and correct gap handling | 1124 |
| 8.2 | MODIFIED | Removed hard-coded `0.6`; aligned barrier calibration and split fraction with `config/model.yaml` | Prevents configuration drift | 1288 |
| 8.2 | MODIFIED | Prevented barrier calibration leakage at train/test boundary | Ensures strict temporal isolation | 1288 |
| 8.3 | MODIFIED | Added within-train validation split for CatBoost early stopping | Prevents test-set contamination | 1452 |
| 8.4 | MODIFIED | Fixed warm-up/training alignment for delayed labels; added missing imports | Correct prequential evaluation semantics | 1664 |
| 10.1 | MODIFIED | Corrected `predict_proba_one` usage in invariant example | Matches River return types | 2059 |
| 10.4-10.5 | ADDED | Added explicit assumptions, limitations, and edge-case policy | Meets scientific rigor and implementation clarity gates | N/A |

## Verification Evidence  [ADDED]
### Formula Verifications
- Parkinson variance coefficient `1/(4 ln 2) ≈ 0.3606737602` verified numerically; DOI: `10.1086/296071` (original full text access restricted).
- Garman–Klass coefficient `(2 ln 2 - 1) ≈ 0.3862943611` verified numerically; DOI: `10.1086/296072` (original full text access restricted).
- Amihud illiquidity definition verified at DOI `10.1016/S1386-4181(01)00024-6` (original full text access restricted); spec clarifies the proxy mapping to Binance `quote_volume`.
- Barrier label definition verified as a one-sided specialization of the triple-barrier family described in López de Prado (2018).

### API Verifications (Local Introspection)
- River `0.21.0`: `river.utils.Rolling(obj, window_size)` exists; `stats.Mean()`, `stats.Var(ddof=1)` exist; `stats.RollingMin/Max(window_size)` exist; ARF classifier is `river.forest.ARFClassifier` with parameters `n_models`, `max_features`, `lambda_value`, `seed`.
- CatBoost `1.2.8`: `predict_proba` returns shape `(n_samples, 2)` for binary; `get_feature_importance(type="PredictionValuesChange")` is valid (`EFstrType.PredictionValuesChange`).
- scikit-learn `1.8.0`: `roc_auc_score(y_true, y_score)`; `precision_recall_curve` returns `(precision, recall, thresholds)`; `calibration_curve` is in `sklearn.calibration` with `strategy in {"uniform","quantile"}`.
- pandas `2.3.3` / pyarrow `22.0.0`: `DataFrame.to_parquet(engine="pyarrow")` valid; `pd.Timestamp(..., unit="ms")` valid.

### Data Schema Verification
- Binance public data schema and timestamp unit change verified against `https://raw.githubusercontent.com/binance/binance-public-data/master/README.md` (section “SPOT → Klines”).

### Terminology Consistency Matrix  [ADDED]

| Term | Definition Location | Usage Locations | Consistent? |
|------|---------------------|-----------------|-------------|
| Decision bar | Section 3.1-3.2 | Sections 3-10, notebooks | Yes |
| Decision interval (`n`) | Section 3.1, `config/pipeline.yaml` | Aggregator + feature build | Yes |
| Feature vector $\phi_k$ | Section 3.6 | Sections 8.4, 10.1 | Yes |
| Label $y_k$ | Section 3.3 | Sections 8.2, 8.4, 10.1 | Yes |
| Barrier threshold $\alpha$ | Section 3.3 | `feature_metadata.json`, online delayed labels | Yes |
| Segment (`segment_id`) | Sections 4.3, 3.2 | Aggregator + feature pipeline + labeling | Yes |
| Burn-in (`burn_in_bars`) | Section 3.5, `config/pipeline.yaml` | Feature build + invariants | Yes |

### Validation Findings (Issue Log)  [ADDED]

| ID | Severity | Location (v1.0) | Finding | Resolution |
|----|----------|------------------|---------|------------|
| F-01 | CRITICAL | 2.1 (line 22) | `numpy>=...` allowed NumPy 2.x, incompatible with River 0.21.0 | Added `numpy<2.0` and `scipy` constraints; pinned `river==0.21.0` |
| F-02 | CRITICAL | 2.2.1 (line 55) | Used non-existent River classes `stats.RollingMean/RollingVar` | Switched to `utils.Rolling(stats.Mean/Var)` per River 0.21.0 |
| F-03 | CRITICAL | 2.2.1 (line 55) | Incorrect ARF import (`river.ensemble.AdaptiveRandomForestClassifier`) for River 0.21.0 | Updated to `river.forest.ARFClassifier` everywhere |
| F-04 | CRITICAL | 2.2.3 (line 154) | Incorrect `calibration_curve` import path | Corrected to `sklearn.calibration.calibration_curve` throughout |
| F-05 | HIGH | 3.3 (line 224) | Labeling described as barrier crossing but not related to triple-barrier method | Clarified as one-sided, horizon-1 special case; added formal definition |
| F-06 | CRITICAL | 4.1 (line 341) | Assumed Binance timestamps are milliseconds | Added µs→ms normalization and documented 2025 Binance change |
| F-07 | HIGH | 4.3 (line 395) | Gap/segment behavior underspecified; segment split threshold inconsistent | Specified “any gap starts new segment”; recorded major gaps separately |
| F-08 | HIGH | 5 (line 406) | Project structure diagram contradicted notebook outputs | Updated tree to match artifacts and metadata actually written |
| F-09 | CRITICAL | 6.2 (line 672) | Aggregator violated River `Transformer` contract (`transform_one` returned `None`) | Redesigned as non-River component with `update()` returning `Optional[dict]` |
| F-10 | HIGH | 6.2 (line 672) | Aggregator did not handle minute gaps; could aggregate across gaps | Added explicit gap reset logic and `segment_id` support |
| F-11 | HIGH | 6.1 (line 457) | Timestamp utilities assumed ms only | Added unit inference + normalization helpers |
| F-12 | MEDIUM | 6.3 (line 787) | GK variance negativity not specified | Clipped negative values and added `flag__gk_negative` |
| F-13 | MEDIUM | 3.4 (line 239) | Edge-case flags used in code but not documented | Added Section 3.4.8 flags table |
| F-14 | CRITICAL | 8.1 (line 1124) | Raw ZIPs not persisted despite project structure claims | Persisted ZIPs and documented outputs |
| F-15 | HIGH | 8.1 (line 1124) | Segment IDs not created, preventing safe resets | Added `segment_id` generation in cleansing and persisted it |
| F-16 | CRITICAL | 8.2 (line 1288) | Hard-coded `0.6` split fraction | Loaded `train_fraction` from `config/model.yaml` |
| F-17 | CRITICAL | 8.2 (line 1288) | Alpha calibration leaked one bar from test into train | Excluded last training bar for calibration (`k+1 < k_split`) |
| F-18 | HIGH | 8.2 (line 1288) | Labels computed across segment boundaries | Masked excursions where `segment_id != next_segment_id` and dropped |
| F-19 | HIGH | 8.2 (line 1288) | Burn-in applied globally, not per segment | Applied burn-in per segment via `bar_in_segment` |
| F-20 | HIGH | 8.2 (line 1288) | Rolling update/read order ambiguous | Specified update-then-read and implemented accordingly |
| F-21 | CRITICAL | 8.4 (line 1664) | Warmup buffering misaligned delayed labels | Implemented single streaming pass (warmup+test) with correct delayed updates |
| F-22 | MEDIUM | 8.4 (line 1664) | Metrics could crash on single-class slices | Added safe ROC/PR metric helpers and guards |
| F-23 | HIGH | 8.3 (line 1452) | Early stopping used test set (`eval_set=X_test`) | Added within-train validation split (`val_fraction`) |
| F-24 | MEDIUM | 8.* (lines 1124+) | Notebook pathing depended on CWD assumptions | Added robust `ROOT` discovery in all notebooks |
| F-25 | LOW | 12 (line 2130) | Constants table used rounded coefficients | Updated to full-precision coefficient values |

## Remaining Ambiguities  [ADDED]
- None. All previously identified ambiguity points are resolved with explicit, testable behavior in Sections 3, 4, 6, 8, and 10.

---

## 1. Executive Summary

This document specifies a two-stage classification system that predicts whether a time-ordered price process will cross a predefined return barrier within a fixed forward horizon. The system combines:

1. **Offline Model:** CatBoost gradient boosting classifier trained on historical data (batch learning)
2. **Online Model:** River Adaptive Random Forest trained sequentially on streaming data (incremental learning)

**Implementation Philosophy:** Minimal functional code with maximum library exploitation. All orchestration logic resides in Jupyter notebooks; only custom River transformers and shared utilities exist in `/src`.

---

## 2. Technology Stack

### 2.1 Required Dependencies

```python
# requirements.txt
# Core Data Processing
pandas>=2.1.0,<3.0.0    # [MODIFIED] River 0.21.0 requires pandas>=2.1
numpy>=1.23.0,<2.0.0    # [MODIFIED] River 0.21.0 requires numpy<2.0
scipy>=1.8.1,<2.0.0     # [ADDED] River dependency (statistics/metrics)
pyarrow>=14.0.0         # Parquet file format support

# Machine Learning - Offline
catboost>=1.2.0         # Gradient boosting classifier

# Machine Learning - Online  
river==0.21.0           # [MODIFIED] Version validated in this specification

# Visualization
matplotlib>=3.7.0       # Static plots
seaborn>=0.12.0         # Statistical visualizations

# Evaluation
scikit-learn>=1.3.0     # Metrics (roc_auc_score, precision_recall_curve, calibration_curve)

# Configuration
pyyaml>=6.0             # YAML config parsing

# Utilities
requests>=2.31.0        # HTTP for data download
tqdm>=4.65.0            # Progress bars
hashlib                 # Built-in, config hashing
```

### 2.2 Library Usage Patterns

#### 2.2.1 River Core Patterns  [MODIFIED]

```python
"""
River operates on dictionaries (dict[str, float]) as feature containers.
[CLARIFIED] River has three distinct call-order patterns:
1) Streaming statistics: `update(value)` then `get()`
2) Transformers: `transform_one(x)` then `learn_one(x)` (if the transformer is stateful)
3) Models: `predict_*_one(x)` then `learn_one(x, y)` (prequential)
"""
from river import stats, utils, forest

# --- Rolling Statistics Pattern ---
# [MODIFIED] In River 0.21.0, rolling mean/variance are built by wrapping base stats.
rolling_mean = utils.Rolling(stats.Mean(), window_size=12)        # 12-bar rolling mean
rolling_var = utils.Rolling(stats.Var(ddof=1), window_size=12)    # 12-bar sample variance
rolling_min = stats.RollingMin(window_size=12)
rolling_max = stats.RollingMax(window_size=12)

# Update and retrieve pattern:
for value in stream:
    rolling_mean.update(value)
    current_mean = rolling_mean.get()

# --- Custom Transformer Pattern ---
# Inherit from river.base.Transformer for custom feature computation
from river.base import Transformer

class CustomFeature(Transformer):
    """Docstring: Describe the transformer's purpose."""
    
    def __init__(self, param: float = 1.0):
        self.param = param
        self._state = None  # Internal state
    
    def learn_one(self, x: dict) -> "CustomFeature":
        """Update internal state with observation x."""
        # Update logic here
        return self
    
    def transform_one(self, x: dict) -> dict:
        """Produce features from observation x."""
        # Transform logic here
        return {"feature_name": computed_value}

# --- Adaptive Random Forest Pattern ---
# [MODIFIED] In River 0.21.0, Adaptive Random Forest is `river.forest.ARFClassifier`.
from river.forest import ARFClassifier

model = ARFClassifier(
    n_models=10,           # Number of trees
    max_features="sqrt",   # Features per split
    lambda_value=6,        # Poisson λ for online bagging
    seed=42
)

# Predict then learn (prequential evaluation)
y_pred_proba = model.predict_proba_one(x)  # dict: {class_label: probability}
p_positive = y_pred_proba.get(1, 0.5)      # [CLARIFIED] cold-start default
model.learn_one(x, y)  # y is the true label
```

#### 2.2.2 CatBoost Core Patterns

```python
"""
CatBoost provides gradient boosting with native categorical feature support.
Key methods: fit, predict_proba, get_feature_importance.
"""
from catboost import CatBoostClassifier, Pool

# --- Training Pattern ---
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

# Create Pool for efficient data handling
train_pool = Pool(data=X_train, label=y_train)
eval_pool = Pool(data=X_val, label=y_val)

model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

# --- Probability Prediction ---
# Returns array of shape (n_samples, 2) for binary classification
probas = model.predict_proba(X_test)
p_positive = probas[:, 1]  # Probability of class 1

# --- Feature Importance ---
# PredictionValuesChange: how much prediction changes if feature changes
importance = model.get_feature_importance(type="PredictionValuesChange")
feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)
```

#### 2.2.3 Scikit-learn Evaluation Patterns  [MODIFIED]

```python
"""
Evaluation metrics for imbalanced binary classification.
"""
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve  # [MODIFIED] correct import location

# --- ROC Analysis ---
fpr, tpr, thresholds = roc_curve(y_true, y_prob_positive)
roc_auc = roc_auc_score(y_true, y_prob_positive)

# --- Precision-Recall (preferred for imbalanced data) ---
precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob_positive)
pr_auc = average_precision_score(y_true, y_prob_positive)

# --- Calibration Curve (Reliability Diagram) ---
# fraction_of_positives: actual positive rate in each bin
# mean_predicted_value: mean predicted probability in each bin
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_prob_positive, n_bins=10, strategy="uniform"
)

# --- Brier Score (lower is better, measures calibration) ---
brier = brier_score_loss(y_true, y_prob_positive)
```

---

## 3. Mathematical Definitions

### 3.1 Indexing Convention

| Symbol | Definition | Domain |
|--------|------------|--------|
| $i$ | Minute-bar index | $\{0, 1, 2, \ldots\}$ |
| $k$ | Decision-bar index | $\{0, 1, 2, \ldots\}$ |
| $n$ | Aggregation factor (minutes per decision bar) | Default: 20 |

**Decision bar $k$** aggregates minute bars $\{kn, kn+1, \ldots, (k+1)n - 1\}$.

### 3.2 Decision Bar Schema

Each decision bar $X_k$ is a dictionary:

```python
X_k = {
    "open_time": int,        # Unix timestamp (ms, normalized), start of bar
    "close_time": int,       # Unix timestamp (ms, normalized), end of bar
    "segment_id": int,       # [ADDED] Segment identifier (resets at any minute gap)
    "open": float,           # O_k: first trade price in interval
    "high": float,           # H_k: maximum trade price
    "low": float,            # L_k: minimum trade price
    "close": float,          # C_k: last trade price
    "volume": float,         # V_k: base asset volume
    "quote_volume": float,   # Q_k: quote asset volume
    "trades": int,           # N_k: number of trades
    "taker_buy_base": float, # B_k: taker buy base volume
    "taker_buy_quote": float # BQ_k: taker buy quote volume
}
```

### 3.3 Label Definition (Barrier Crossing)  [CLARIFIED]

[CLARIFIED] This specification uses a **one-sided, horizon-1 upper barrier** label (a special case of the triple-barrier family in López de Prado, 2018 with only an upper barrier and a vertical barrier at $k+1$).

At decision time $k$, predict whether bar $k+1$ crosses the upper barrier:

$$
y_k = \mathbf{1}\left[\ln\left(\frac{H_{k+1}}{C_k}\right) \geq \alpha\right]
$$

Where:
- $C_k$ is the reference price (close of current bar)
- $H_{k+1}$ is the high of the next bar
- $\alpha$ is the barrier threshold (log-return)

**Barrier Calibration (Default):** Set $\alpha$ to the 95th percentile of $\ln(H_{k+1}/C_k)$ on training data, yielding approximately 5% positive class rate.

[CLARIFIED] **No-leakage calibration rule:** when using a chronological split index $k_{\text{split}}$, estimate $\alpha$ using only excursions $e_k=\ln(H_{k+1}/C_k)$ where **both** $k < k_{\text{split}}$ and $(k+1) < k_{\text{split}}$ (i.e., exclude the last training bar because its label depends on the first test bar).

[CLARIFIED] **Gap/segment rule:** labels are **undefined** across discontinuities (Section 4.3). If $X_{k+1}$ is not the immediate next decision bar in time (e.g., a segment boundary), set $y_k=\text{NaN}$ and drop the sample from offline training/evaluation.

### 3.4 Feature Formulas

Let $\epsilon = 10^{-10}$ (numerical stability constant).

#### 3.4.1 Returns and Variation

| Feature | Formula | Description |
|---------|---------|-------------|
| `log_close` | $p_k = \ln(C_k)$ | Log price |
| `return` | $r_k = p_k - p_{k-1}$ | Log return |
| `abs_return` | $\|r_k\|$ | Absolute return |
| `squared_return` | $r_k^2$ | Squared return |

#### 3.4.2 Range-Based Volatility Estimators

**Parkinson Estimator** (Parkinson, 1980):
$$
\hat{\sigma}^2_{P,k} = \frac{1}{4\ln 2}\left(\ln\frac{H_k}{L_k}\right)^2
$$

**Garman-Klass Estimator** (Garman & Klass, 1980):
$$
\hat{\sigma}^2_{GK,k} = \frac{1}{2}\left(\ln\frac{H_k}{L_k}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_k}{O_k}\right)^2
$$

[MODIFIED] Note: $(2\ln 2 - 1) \approx 0.3862943611$ and $\frac{1}{4\ln 2} \approx 0.3606737602$.

[CLARIFIED] Numerical guard: in finite samples, the GK formula can be negative; the implementation **clips** `garman_klass_var` to `0.0` and emits `flag__gk_negative=1` (Section 6.3).

| Feature | Formula |
|---------|---------|
| `range_hl` | $\Delta^{hl}_k = \ln(H_k / L_k)$ |
| `parkinson_var` | $\hat{\sigma}^2_{P,k}$ |
| `garman_klass_var` | $\hat{\sigma}^2_{GK,k}$ |

#### 3.4.3 Candle Geometry

| Feature | Formula |
|---------|---------|
| `body` | $(C_k - O_k) / (O_k + \epsilon)$ |
| `upper_wick` | $(H_k - \max(O_k, C_k)) / (O_k + \epsilon)$ |
| `lower_wick` | $(\min(O_k, C_k) - L_k) / (O_k + \epsilon)$ |
| `body_fraction` | $\|C_k - O_k\| / (H_k - L_k + \epsilon)$ |

#### 3.4.4 Activity Features

| Feature | Formula |
|---------|---------|
| `log_volume` | $\ln(1 + V_k)$ |
| `log_quote_volume` | $\ln(1 + Q_k)$ |
| `trades` | $N_k$ |
| `avg_trade_size` | $Q_k / (N_k + \epsilon)$ |

#### 3.4.5 Order Flow Proxies

| Feature | Formula |
|---------|---------|
| `buy_ratio` | $B_k / (V_k + \epsilon)$ |
| `imbalance` | $(2B_k - V_k) / (V_k + \epsilon)$ |

#### 3.4.6 Illiquidity Proxy (Amihud, 2002)

$$
\text{illiq}_k = \frac{|r_k|}{Q_k + \epsilon}
$$

[CLARIFIED] In Amihud (2002), the denominator is **dollar volume**. For Binance klines, we use `quote_volume` ($Q_k$) as the closest proxy (e.g., for `BTCUSDT`, $Q_k$ is in USDT). Returns use the log-return $r_k$; for small returns this closely matches simple returns.

#### 3.4.7 Seasonality Encodings

| Feature | Formula |
|---------|---------|
| `minute_sin` | $\sin(2\pi m / 1440)$ where $m$ = minute of day |
| `minute_cos` | $\cos(2\pi m / 1440)$ |
| `dow_sin` | $\sin(2\pi d / 7)$ where $d$ = day of week (0=Monday) |
| `dow_cos` | $\cos(2\pi d / 7)$ |

#### 3.4.8 Edge-Case Flags  [ADDED]

These indicators make edge conditions explicit for downstream models. All flags are binary in $\{0,1\}$.

| Flag | Definition |
|------|------------|
| `flag__first_bar` | 1 if there is no valid `prev_close` for return computation |
| `flag__segment_start` | 1 on the first decision bar of a new segment (after any minute gap) |
| `flag__range_zero` | 1 if $H_k - L_k \le \epsilon$ (zero/near-zero range) |
| `flag__no_trades` | 1 if $N_k = 0$ (no trades) |
| `flag__no_volume` | 1 if $V_k \approx 0$ (no base volume) |
| `flag__gk_negative` | 1 if raw GK variance is negative before clipping to 0 |

### 3.5 Rolling Statistics  [CLARIFIED]

**Window Grid** (in decision bars):
$$
\mathcal{W} = \{1, 2, 4, 12, 48\}
$$

Corresponding to: 20min, 40min, 80min, 4h, 16h.

**Statistics Applied:**
- Mean
- Variance (sample, ddof=1)
- Min
- Max

**Applied to Base Series:**
- `return`, `squared_return`
- `range_hl`, `garman_klass_var`
- `log_quote_volume`
- `imbalance`, `illiq`

**Naming Convention:** `{base_feature}_rolling_{stat}_{window}`

Example: `return_rolling_mean_12`, `garman_klass_var_rolling_var_48`

[CLARIFIED] **Inclusion rule:** rolling statistics at time $k$ are computed **including** the current base value at $k$ (update-then-read).

[CLARIFIED] **Segment reset rule:** rolling state is reset when `flag__segment_start=1` (i.e., at any gap/segment boundary).

[CLARIFIED] **Burn-in rule:** burn-in is applied **per segment** (drop the first `burn_in_bars` decision bars of each segment).

### 3.6 Feature Vector Definition  [ADDED]

At each decision bar $k$, the feature vector $\phi_k$ is the union of:
- Base features (Section 3.4)
- Edge-case flags (Section 3.4.8)
- Rolling features (Section 3.5)

In implementation, $\phi_k$ is represented as `dict[str, float]` (River convention).

[ADDED] For the online correction model, the input vector $z_k$ is:

$$
z_k = \left(\phi_k[\text{selected\_features}]\right) \cup \{p^{\text{off}}_k\}
$$

where $p^{\text{off}}_k$ is the offline model probability for class 1 at time $k$ and is stored under the key `p_offline`.

---

## 4. Data Pipeline

### 4.1 Data Source  [MODIFIED]

**Primary Source:** Binance public kline data  
**URL Pattern:** `https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY}-{MM}.zip`

[MODIFIED] **Timestamp units (CRITICAL):** Binance SPOT public data uses **milliseconds** historically, but from **2025-01-01 onward uses microseconds** (Binance public-data README). This specification **normalizes all timestamps to integer milliseconds** in `data/cleansed_data/**/1m.parquet` and records the raw unit in `metadata.json`.

**Raw Kline Schema (12 columns):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | open_time | int64 | Unix timestamp (ms or µs in raw; normalized to ms in cleansed parquet) |
| 1 | open | float64 | Open price |
| 2 | high | float64 | High price |
| 3 | low | float64 | Low price |
| 4 | close | float64 | Close price |
| 5 | volume | float64 | Base asset volume |
| 6 | close_time | int64 | Unix timestamp (ms or µs in raw; normalized to ms in cleansed parquet) |
| 7 | quote_volume | float64 | Quote asset volume |
| 8 | trades | int64 | Number of trades |
| 9 | taker_buy_base | float64 | Taker buy base volume |
| 10 | taker_buy_quote | float64 | Taker buy quote volume |
| 11 | ignore | - | Unused field |

### 4.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Binance API/Vision]                                           │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  Raw ZIP/CSV │  /data/raw_data/{symbol}/                     │
│  │   (1-minute) │                                               │
│  └──────┬───────┘                                               │
│         │  data_download.ipynb                                  │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │   Cleansed   │  /data/cleansed_data/{symbol}/1m.parquet      │
│  │   1-minute   │                                               │
│  └──────┬───────┘                                               │
│         │  feature_build.ipynb                                  │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  20-minute   │  /data/model_data/{symbol}/                   │
│  │  bars +      │  bars_20m_features.parquet                    │
│  │  features +  │                                               │
│  │  labels      │                                               │
│  └──────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Cleansing Rules  [MODIFIED]

1. **Timestamp Normalization:** Infer raw unit (ms vs µs), then normalize `open_time` and `close_time` to **integer milliseconds** (Section 4.1)
2. **Deduplication:** Keep first occurrence on duplicate `open_time`
3. **Sorting:** Strictly ascending by `open_time`
4. **Type Enforcement:** Cast all price/volume columns to `float64`, timestamps to `int64`
5. **Gap Detection:** Treat any `open_time` step != `60_000` ms as a gap; do NOT interpolate
6. **Segment Splitting:** Any gap starts a new `segment_id`; additionally, gaps >= `max_gap_minutes` are reported as “major gaps” in metadata
7. **Segment Annotation:** Persist `segment_id` as an `int64` column in the cleansed parquet to enforce downstream resets (Sections 6.2, 6.3, 8.2)

---

## 5. Project Structure  [MODIFIED]

```
online_barrier_classifier/
  config/
    download.yaml            # Symbol, date range, data paths
    pipeline.yaml            # Decision interval, windows, barrier config
    model.yaml               # CatBoost params, online model params

  data/
    raw_data/{symbol}/       # Downloaded ZIP files (immutable)
    cleansed_data/{symbol}/  # Validated 1-minute parquet (timestamps normalized to ms)
      1m.parquet
      metadata.json
    model_data/{symbol}/     # Decision bars with features/labels
      bars_20m_features.parquet
      feature_metadata.json

  artifacts/
    offline_model/           # CatBoost artifacts
      model.cbm
      selected_features.json
      metrics.json
      feature_importance.csv
      config_snapshot.json
      roc_pr_curves.png
      calibration_curve.png
      probability_histogram.png
    online_eval/             # Streaming evaluation artifacts
      metrics.json
      predictions.parquet
      periodic_metrics.csv
      plots/

  notebooks/                 # All orchestration logic
    data_download.ipynb
    feature_build.ipynb
    offline_train.ipynb
    online_eval.ipynb

  src/
    __init__.py
    transformers/
      __init__.py
      bar_aggregator.py
      feature_pipeline.py
    utils.py

  requirements.txt
```

---

## 6. Source Code Specifications

### 6.1 `/src/utils.py`

```python
"""
Shared utility functions used across notebooks and transformers.

This module contains ONLY functions that are:
1. Used in multiple places, AND
2. Correctness-critical or performance-sensitive

All other helper functions should remain notebook-local.
"""

from __future__ import annotations
import hashlib
import yaml
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# NUMERICAL UTILITIES
# =============================================================================

def safe_divide(
    numerator: float,
    denominator: float,
    neutral: float = 0.0,
    eps: float = 1e-10
) -> tuple[float, bool]:
    """
    Perform safe division with explicit handling of near-zero denominators.
    
    Parameters
    ----------
    numerator : float
        The numerator value.
    denominator : float
        The denominator value.
    neutral : float, default 0.0
        Value to return when division is undefined.
    eps : float, default 1e-10
        Threshold below which denominator is considered zero.
    
    Returns
    -------
    tuple[float, bool]
        (result, flag) where flag=True indicates division was undefined.
    
    Examples
    --------
    >>> safe_divide(10.0, 2.0)
    (5.0, False)
    >>> safe_divide(10.0, 0.0)
    (0.0, True)
    """
    if abs(denominator) < eps:
        return neutral, True
    return numerator / denominator, False


def parkinson_variance(high: float, low: float) -> float:
    """
    Compute single-bar Parkinson variance estimator.
    
    σ²_P = (1 / 4ln2) * (ln(H/L))²
    
    Reference: Parkinson (1980), "The extreme value method for estimating
    the variance of the rate of return", Journal of Business.
    
    Parameters
    ----------
    high : float
        High price of the bar.
    low : float
        Low price of the bar.
    
    Returns
    -------
    float
        Estimated variance for the bar period.
    """
    if low <= 0 or high <= 0 or high < low:
        return 0.0
    log_range = np.log(high / low)
    return (log_range ** 2) / (4 * np.log(2))


def garman_klass_variance(
    open_: float,
    high: float,
    low: float,
    close: float
) -> float:
    """
    Compute single-bar Garman-Klass variance estimator.
    
    σ²_GK = 0.5 * (ln(H/L))² - (2ln2 - 1) * (ln(C/O))²
    
    Reference: Garman & Klass (1980), "On the estimation of security price
    volatilities from historical data", Journal of Business.
    
    Parameters
    ----------
    open_ : float
        Open price of the bar.
    high : float
        High price of the bar.
    low : float
        Low price of the bar.
    close : float
        Close price of the bar.
    
    Returns
    -------
    float
        Estimated variance for the bar period.
    """
    if low <= 0 or high <= 0 or open_ <= 0 or close <= 0:
        return 0.0
    if high < low:
        return 0.0
    
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    
    return 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.
    
    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def config_hash(config: dict[str, Any]) -> str:
    """
    Compute deterministic SHA-256 hash of configuration.
    
    Used for cache invalidation: if config hash changes,
    dependent artifacts should be regenerated.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    
    Returns
    -------
    str
        Hexadecimal hash string (first 16 characters).
    """
    config_str = yaml.dump(config, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# =============================================================================
# TIMESTAMP UTILITIES
# =============================================================================

def infer_unix_timestamp_unit(ts: int) -> str:
    """
    Infer whether a Unix timestamp is in milliseconds or microseconds.

    Binance SPOT public data uses microseconds from 2025-01-01 onward.
    This helper uses a magnitude heuristic that is stable for modern dates.

    Parameters
    ----------
    ts : int
        Unix timestamp.

    Returns
    -------
    str
        Either "ms" or "us".
    """
    # 2026-01-01 in milliseconds is ~1.77e12; in microseconds is ~1.77e15.
    return "us" if ts >= 10**14 else "ms"


def to_ms_timestamp(ts: int, unit: str | None = None) -> int:
    """
    Normalize a Unix timestamp to integer milliseconds.

    Parameters
    ----------
    ts : int
        Unix timestamp in either ms or us.
    unit : {"ms", "us"}, optional
        Explicit unit. If None, inferred via `infer_unix_timestamp_unit`.

    Returns
    -------
    int
        Unix timestamp in milliseconds.
    """
    unit = unit or infer_unix_timestamp_unit(ts)
    if unit == "ms":
        return int(ts)
    if unit == "us":
        return int(ts // 1_000)
    raise ValueError(f"Unsupported unit: {unit}")


def ts_to_datetime(ts: int, unit: str | None = None) -> pd.Timestamp:
    """
    Convert a Unix timestamp (ms or us) to a UTC pandas Timestamp.

    Parameters
    ----------
    ts : int
        Unix timestamp.
    unit : {"ms", "us"}, optional
        Explicit unit. If None, inferred via `infer_unix_timestamp_unit`.

    Returns
    -------
    pd.Timestamp
        UTC-localized timestamp.
    """
    unit = unit or infer_unix_timestamp_unit(ts)
    return pd.Timestamp(ts, unit=unit, tz="UTC")


def datetime_to_ts(dt: pd.Timestamp, unit: str = "ms") -> int:
    """
    Convert a pandas Timestamp to a Unix timestamp in the requested unit.

    Parameters
    ----------
    dt : pd.Timestamp
        Pandas timestamp (any timezone or naive).
    unit : {"ms", "us"}, default "ms"
        Target unit.

    Returns
    -------
    int
        Unix timestamp in requested unit.
    """
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    dt = dt.tz_convert("UTC")

    # `dt.value` is nanoseconds since epoch.
    ns = int(dt.value)
    if unit == "ms":
        return ns // 1_000_000
    if unit == "us":
        return ns // 1_000
    raise ValueError(f"Unsupported unit: {unit}")
```

### 6.2 `/src/transformers/bar_aggregator.py`  [MODIFIED]

```python
"""
[MODIFIED] Decision bar aggregator for converting minute bars to N-minute bars.

This component aggregates streaming minute-bar observations into decision bars
of configurable length (default: 20 minutes).

[CLARIFIED] This is *not* a River `Transformer` because it emits either:
- `None` (while buffering), or
- an aggregated bar dict (when ready),
which does not match River's `Transformer.transform_one -> dict` contract.
"""

from __future__ import annotations
from collections import deque
from typing import Any, Optional


class DecisionBarAggregator:
    """
    Aggregate minute bars into decision bars.
    
    Collects `n` strictly consecutive minute bars (in **time**) and emits a
    single decision bar with aggregated OHLCV statistics.

    [CLARIFIED] Any missing-minute gap resets the internal buffer and starts a
    new in-memory segment. Segment boundaries should also be provided explicitly
    via `segment_id` (produced in `data_download.ipynb`).
    
    Parameters
    ----------
    n : int, default 20
        Number of minute bars per decision bar.
    expected_step_ms : int, default 60_000
        Expected spacing between consecutive minute bars (milliseconds).
    
    Attributes
    ----------
    buffer : deque
        Internal buffer holding minute bars until aggregation.
    bar_count : int
        Number of decision bars emitted.
    
    Examples
    --------
    >>> agg = DecisionBarAggregator(n=20)
    >>> for minute_bar in minute_bars:
    ...     decision_bar = agg.update(minute_bar)
    ...     if decision_bar is not None:
    ...         process(decision_bar)
    """
    
    def __init__(self, n: int = 20, expected_step_ms: int = 60_000):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.expected_step_ms = expected_step_ms
        self.buffer: deque[dict[str, Any]] = deque()
        self.bar_count: int = 0
        self._prev_open_time: Optional[int] = None
        self._prev_segment_id: Optional[int] = None
    
    def reset(self) -> None:
        """Reset internal buffer and time/segment trackers."""
        self.buffer.clear()
        self._prev_open_time = None
        self._prev_segment_id = None

    def update(self, x: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Add a minute bar to the buffer and aggregate when full.
        
        Parameters
        ----------
        x : dict
            Minute bar with keys: open_time, open, high, low, close,
            volume, close_time, quote_volume, trades, taker_buy_base,
            taker_buy_quote.
        
        Returns
        -------
        dict or None
            Aggregated decision bar when ready, else None while buffering.
        """
        open_time = int(x["open_time"])
        segment_id = x.get("segment_id")

        # [CLARIFIED] Reset on explicit segment boundary.
        if self._prev_segment_id is not None and segment_id is not None and segment_id != self._prev_segment_id:
            self.reset()

        # [CLARIFIED] Reset on any missing-minute gap.
        if self._prev_open_time is not None:
            if open_time - self._prev_open_time != self.expected_step_ms:
                self.reset()

        self._prev_open_time = open_time
        if segment_id is not None:
            self._prev_segment_id = int(segment_id)

        self.buffer.append(x)
        if len(self.buffer) < self.n:
            return None

        bar = self._aggregate()
        self.buffer.clear()
        self.bar_count += 1
        return bar
    
    def _aggregate(self) -> dict[str, Any]:
        """Aggregate buffered minute bars into a single decision bar."""
        bars = list(self.buffer)

        segment_id = bars[0].get("segment_id")
        return {
            "open_time": bars[0]["open_time"],
            "close_time": bars[-1]["close_time"],
            "open": bars[0]["open"],
            "high": max(b["high"] for b in bars),
            "low": min(b["low"] for b in bars),
            "close": bars[-1]["close"],
            "volume": sum(b["volume"] for b in bars),
            "quote_volume": sum(b["quote_volume"] for b in bars),
            "trades": sum(b["trades"] for b in bars),
            "taker_buy_base": sum(b["taker_buy_base"] for b in bars),
            "taker_buy_quote": sum(b["taker_buy_quote"] for b in bars),
            "segment_id": segment_id,
        }
```

### 6.3 `/src/transformers/feature_pipeline.py`

```python
"""
Streaming feature pipeline for decision bars.

This module defines the feature extraction pipeline using River's
streaming statistics. Features are computed incrementally as each
decision bar arrives.
"""

from __future__ import annotations
from typing import Optional
import math

from river import stats, utils  # [MODIFIED] `utils.Rolling` used for rolling mean/var in River 0.21.0
from river.base import Transformer

from ..utils import (
    safe_divide,
    parkinson_variance,
    garman_klass_variance,
)


class BaseFeatureExtractor(Transformer):
    """
    Extract base features from a single decision bar.
    
    Computes non-rolling features that depend only on the current bar
    and optionally the previous close price.
    
    Parameters
    ----------
    eps : float, default 1e-10
        Numerical stability constant for division.
    
    Attributes
    ----------
    prev_close : float or None
        Close price of the previous bar (for return computation).
    """
    
    def __init__(self, eps: float = 1e-10):
        self.eps = eps
        self.prev_close: Optional[float] = None
        self.prev_segment_id: Optional[int] = None  # [ADDED] resets return at segment boundaries
    
    def learn_one(self, x: dict) -> "BaseFeatureExtractor":
        """Update state with the current bar's close price."""
        self.prev_close = x["close"]
        if "segment_id" in x and x["segment_id"] is not None:
            self.prev_segment_id = int(x["segment_id"])
        return self
    
    def transform_one(self, x: dict) -> dict:
        """
        Extract base features from decision bar.
        
        Parameters
        ----------
        x : dict
            Decision bar dictionary.
        
        Returns
        -------
        dict
            Feature dictionary with base features.
        """
        o, h, l, c = x["open"], x["high"], x["low"], x["close"]
        v, q = x["volume"], x["quote_volume"]
        n_trades = x["trades"]
        taker_buy_base = x["taker_buy_base"]
        open_time = x["open_time"]
        segment_id = x.get("segment_id")
        
        eps = self.eps
        features = {}
        flags = {}
        
        # --- Returns ---
        log_close = math.log(c) if c > 0 else 0.0

        # [ADDED] Segment-aware return: do not carry prev_close across gaps/segments.
        seg = int(segment_id) if segment_id is not None else None
        is_new_segment = (
            seg is not None
            and self.prev_segment_id is not None
            and seg != self.prev_segment_id
        )
        if is_new_segment:
            prev_close = None
            flags["flag__segment_start"] = 1
        else:
            prev_close = self.prev_close
            flags["flag__segment_start"] = 1 if self.prev_close is None else 0

        flags["flag__first_bar"] = 1 if prev_close is None else 0

        if prev_close is not None and prev_close > 0:
            ret = log_close - math.log(prev_close)
        else:
            ret = 0.0
        
        features["log_close"] = log_close
        features["return"] = ret
        features["abs_return"] = abs(ret)
        features["squared_return"] = ret ** 2
        
        # --- Range-based volatility ---
        features["range_hl"] = math.log(h / l) if h > 0 and l > 0 else 0.0
        features["parkinson_var"] = parkinson_variance(h, l)
        gk_var = garman_klass_variance(o, h, l, c)
        if gk_var < 0:
            features["garman_klass_var"] = 0.0
            flags["flag__gk_negative"] = 1
        else:
            features["garman_klass_var"] = gk_var
            flags["flag__gk_negative"] = 0
        
        # --- Candle geometry ---
        features["body"] = (c - o) / (o + eps)
        features["upper_wick"] = (h - max(o, c)) / (o + eps)
        features["lower_wick"] = (min(o, c) - l) / (o + eps)
        
        range_size = h - l
        if range_size > eps:
            features["body_fraction"] = abs(c - o) / range_size
            flags["flag__range_zero"] = 0
        else:
            features["body_fraction"] = 0.0
            flags["flag__range_zero"] = 1
        
        # --- Activity ---
        features["log_volume"] = math.log(1 + v)
        features["log_quote_volume"] = math.log(1 + q)
        features["trades"] = float(n_trades)
        
        avg_trade, flag_no_trades = safe_divide(q, n_trades, neutral=0.0, eps=1.0)
        features["avg_trade_size"] = avg_trade
        flags["flag__no_trades"] = int(flag_no_trades)
        
        # --- Order flow ---
        buy_ratio, flag_no_vol = safe_divide(taker_buy_base, v, neutral=0.5, eps=eps)
        features["buy_ratio"] = buy_ratio
        flags["flag__no_volume"] = int(flag_no_vol)
        
        imbalance, _ = safe_divide(2 * taker_buy_base - v, v, neutral=0.0, eps=eps)
        features["imbalance"] = imbalance
        
        # --- Illiquidity ---
        illiq, _ = safe_divide(abs(ret), q, neutral=0.0, eps=eps)
        features["illiq"] = illiq
        
        # --- Seasonality ---
        # Convert open_time (ms) to minute of day and day of week
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc)
        minute_of_day = dt.hour * 60 + dt.minute
        day_of_week = dt.weekday()  # 0 = Monday
        
        features["minute_sin"] = math.sin(2 * math.pi * minute_of_day / 1440)
        features["minute_cos"] = math.cos(2 * math.pi * minute_of_day / 1440)
        features["dow_sin"] = math.sin(2 * math.pi * day_of_week / 7)
        features["dow_cos"] = math.cos(2 * math.pi * day_of_week / 7)
        
        # Merge features and flags
        return {**features, **flags}


class RollingFeatureExtractor(Transformer):
    """
    Compute rolling statistics over base features.
    
    Maintains River rolling statistics for specified base features
    across multiple window sizes.
    
    Parameters
    ----------
    base_features : list of str
        Names of base features to compute rolling stats for.
    windows : list of int
        Window sizes (in decision bars).
    stats_to_compute : list of str
        Statistics: "mean", "var", "min", "max".
    """
    
    def __init__(
        self,
        base_features: Optional[list[str]] = None,
        windows: Optional[list[int]] = None,
        stats_to_compute: Optional[list[str]] = None,
    ):
        self.base_features = base_features or [
            "return", "squared_return", "range_hl", "garman_klass_var",
            "log_quote_volume", "imbalance", "illiq"
        ]
        self.windows = windows or [1, 2, 4, 12, 48]
        self.stats_to_compute = stats_to_compute or ["mean", "var", "min", "max"]

        self._init_stats()

    def reset(self) -> None:
        """[ADDED] Reset all rolling state (use at segment boundaries)."""
        self._init_stats()

    def _init_stats(self) -> None:
        """[ADDED] (Re)initialize rolling statistic objects."""
        self._rolling_stats: dict[str, dict] = {}
        for feat in self.base_features:
            self._rolling_stats[feat] = {}
            for w in self.windows:
                for stat_name in self.stats_to_compute:
                    key = f"{feat}_rolling_{stat_name}_{w}"
                    self._rolling_stats[feat][key] = self._create_stat(stat_name, w)
    
    def _create_stat(self, stat_name: str, window: int):
        """Create River rolling statistic object."""
        if stat_name == "mean":
            return utils.Rolling(stats.Mean(), window_size=window)  # [MODIFIED]
        elif stat_name == "var":
            return utils.Rolling(stats.Var(ddof=1), window_size=window)  # [MODIFIED]
        elif stat_name == "min":
            return stats.RollingMin(window_size=window)
        elif stat_name == "max":
            return stats.RollingMax(window_size=window)
        else:
            raise ValueError(f"Unknown stat: {stat_name}")
    
    def learn_one(self, x: dict) -> "RollingFeatureExtractor":
        """
        Update all rolling statistics with base feature values.
        
        Parameters
        ----------
        x : dict
            Base feature dictionary (output of BaseFeatureExtractor).
        """
        # [ADDED] Reset state at segment boundaries to avoid cross-gap leakage.
        if x.get("flag__segment_start") == 1:
            self.reset()

        for feat in self.base_features:
            if feat not in x:
                continue
            value = x[feat]
            for key, stat_obj in self._rolling_stats[feat].items():
                stat_obj.update(value)
        return self
    
    def transform_one(self, x: dict) -> dict:
        """
        Get current rolling statistic values.
        
        Parameters
        ----------
        x : dict
            Base feature dictionary.
        
        Returns
        -------
        dict
            Rolling feature dictionary.
        """
        result = {}
        for feat in self.base_features:
            for key, stat_obj in self._rolling_stats[feat].items():
                result[key] = stat_obj.get()
        return result
```

---

## 7. Configuration Schemas

### 7.1 `config/download.yaml`

```yaml
# Data download configuration
symbol: "BTCUSDT"
market_type: "spot"

# Date range (inclusive)
start_date: "2022-01-01"
end_date: "2023-12-31"

# Paths (relative to project root)
paths:
  raw_data: "data/raw_data"
  cleansed_data: "data/cleansed_data"

# Binance Vision URL template
url_template: "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{year}-{month:02d}.zip"

# Gap handling
max_gap_minutes: 60  # [MODIFIED] Gaps >= this are flagged as "major gaps" (all gaps still start a new segment)
```

### 7.2 `config/pipeline.yaml`

```yaml
# Feature pipeline configuration
decision_interval: 20  # Minutes per decision bar

# Rolling window sizes (in decision bars)
windows: [1, 2, 4, 12, 48]

# Rolling statistics to compute
rolling_stats: ["mean", "var", "min", "max"]

# Base features for rolling computation
rolling_base_features:
  - "return"
  - "squared_return"
  - "range_hl"
  - "garman_klass_var"
  - "log_quote_volume"
  - "imbalance"
  - "illiq"

# Label configuration
barrier:
  # [CLARIFIED] Alpha calibration uses the chronological split fraction in `config/model.yaml` (`train_fraction`)
  # Method: "quantile" or "fixed"
  method: "quantile"
  # For quantile method: target positive class rate
  quantile: 0.95
  # For fixed method: log-return threshold
  fixed_alpha: 0.025

# Burn-in: number of decision bars to skip for rolling stabilization
burn_in_bars: 48  # max(windows)
```

### 7.3 `config/model.yaml`

```yaml
# Model configuration

# Train/test split (chronological)
train_fraction: 0.6
val_fraction: 0.2  # [ADDED] fraction of the training window reserved for early stopping

# --- Offline Model (CatBoost) ---
catboost:
  iterations: 500
  learning_rate: 0.05
  depth: 6
  loss_function: "Logloss"
  eval_metric: "AUC"
  random_seed: 42
  verbose: 100
  early_stopping_rounds: 50
  
# Feature selection for online model
top_k_features: 50  # Number of features to pass to online model

# --- Online Model (River ARF) ---
online:
  model_type: "ARFClassifier"  # [MODIFIED] River 0.21.0 class name (river.forest.ARFClassifier)
  n_models: 10
  max_features: "sqrt"
  lambda_value: 6
  seed: 42

# --- Evaluation ---
evaluation:
  # Reporting cadence during online loop
  report_every_bars: 100
  # Probability threshold for binary predictions
  threshold: 0.5
```

---

## 8. Notebook Specifications

### 8.1 `notebooks/data_download.ipynb`

**Purpose:** Download, validate, and persist cleansed 1-minute data.

```python
# === CELL 1: Imports and Config ===
"""
Data Download Notebook

This notebook:
1. Downloads raw kline data from Binance Vision
2. Validates and cleanses the data
3. Persists to parquet format

Outputs:
- /data/raw_data/{symbol}/*.zip  [ADDED]
- /data/cleansed_data/{symbol}/1m.parquet
- /data/cleansed_data/{symbol}/metadata.json  [ADDED]
"""
import os
import sys
from pathlib import Path
import zipfile
import requests

import pandas as pd
import numpy as np
from tqdm import tqdm

# [MODIFIED] Robust project root discovery (works when launched from repo root or `notebooks/`)
ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "src"))
from utils import load_config, config_hash, ts_to_datetime, infer_unix_timestamp_unit

# === CELL 2: Load Configuration ===
config = load_config(ROOT / "config" / "download.yaml")
SYMBOL = config["symbol"]
START_DATE = pd.Timestamp(config["start_date"])
END_DATE = pd.Timestamp(config["end_date"])
RAW_PATH = ROOT / config["paths"]["raw_data"] / SYMBOL
CLEANSED_PATH = ROOT / config["paths"]["cleansed_data"] / SYMBOL

RAW_PATH.mkdir(parents=True, exist_ok=True)
CLEANSED_PATH.mkdir(parents=True, exist_ok=True)

print(f"Symbol: {SYMBOL}")
print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")

# === CELL 3: Generate Download URLs ===
def generate_month_range(start: pd.Timestamp, end: pd.Timestamp):
    """Generate (year, month) tuples for date range."""
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append((current.year, current.month))
        current += pd.DateOffset(months=1)
    return months

months = generate_month_range(START_DATE, END_DATE)
print(f"Months to download: {len(months)}")

# === CELL 4: Download Function ===
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", 
    "taker_buy_base", "taker_buy_quote", "ignore"
]

def download_month(year: int, month: int) -> pd.DataFrame:
    """Download and parse one month of kline data."""
    url = config["url_template"].format(
        symbol=SYMBOL, year=year, month=month
    )

    # [MODIFIED] Persist raw ZIPs for auditability and reproducibility
    zip_path = RAW_PATH / f"{SYMBOL}-1m-{year}-{month:02d}.zip"
    if not zip_path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
    
    return df

# === CELL 5: Download All Data ===
dfs = []
for year, month in tqdm(months, desc="Downloading"):
    try:
        df = download_month(year, month)
        dfs.append(df)
    except Exception as e:
        print(f"Failed {year}-{month:02d}: {e}")

raw_df = pd.concat(dfs, ignore_index=True)
print(f"Raw rows: {len(raw_df):,}")

# === CELL 6: Validation and Cleansing ===
# Drop unused column
raw_df = raw_df.drop(columns=["ignore"])

# Type enforcement
dtype_map = {
    "open_time": "int64",
    "close_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "quote_volume": "float64",
    "trades": "int64",
    "taker_buy_base": "float64",
    "taker_buy_quote": "float64",
}
raw_df = raw_df.astype(dtype_map)

# [MODIFIED] Normalize Binance timestamps (ms vs µs) to integer milliseconds
raw_unit = infer_unix_timestamp_unit(int(raw_df["open_time"].median()))
if raw_unit == "us":
    raw_df["open_time"] = (raw_df["open_time"] // 1_000).astype("int64")
    raw_df["close_time"] = (raw_df["close_time"] // 1_000).astype("int64")
normalized_unit = "ms"

# Sort by open_time
raw_df = raw_df.sort_values("open_time").reset_index(drop=True)

# Deduplicate (keep first)
n_before = len(raw_df)
raw_df = raw_df.drop_duplicates(subset=["open_time"], keep="first")
n_dups = n_before - len(raw_df)
print(f"Duplicates removed: {n_dups:,}")

# === CELL 7: Gap Analysis ===
raw_df["expected_next"] = raw_df["open_time"] + 60_000  # 1 minute in ms (normalized)
raw_df["actual_next"] = raw_df["open_time"].shift(-1)
raw_df["gap_minutes"] = (raw_df["actual_next"] - raw_df["expected_next"]) / 60_000

gaps = raw_df[raw_df["gap_minutes"] > 0].copy()
print(f"Total gaps: {len(gaps)}")
print(f"Max gap: {gaps['gap_minutes'].max():.0f} minutes")
major_gaps = gaps[gaps["gap_minutes"] >= config["max_gap_minutes"]]
print(f"Major gaps (>= {config['max_gap_minutes']} min): {len(major_gaps)}")

# Gap distribution
gap_summary = gaps.groupby(
    pd.cut(gaps["gap_minutes"], bins=[0, 5, 60, 1440, float("inf")])
).size()
print("\nGap distribution:")
print(gap_summary)

# [MODIFIED] Segment IDs: any gap starts a new segment
delta_ms = raw_df["open_time"].diff()
is_gap = (delta_ms != 60_000) & delta_ms.notna()
raw_df["segment_id"] = is_gap.cumsum().astype("int64")

# Clean up temporary columns
raw_df = raw_df.drop(columns=["expected_next", "actual_next", "gap_minutes"])

# === CELL 8: Save Cleansed Data ===
output_path = CLEANSED_PATH / "1m.parquet"
raw_df.to_parquet(output_path, index=False, engine="pyarrow")

# Save metadata
metadata = {
    "symbol": SYMBOL,
    "start_time": int(raw_df["open_time"].min()),
    "end_time": int(raw_df["open_time"].max()),
    "row_count": len(raw_df),
    "raw_timestamp_unit": raw_unit,
    "normalized_timestamp_unit": normalized_unit,
    "n_gaps": int(len(gaps)),
    "n_major_gaps": int(len(major_gaps)),
    "max_gap_minutes": float(gaps["gap_minutes"].max()) if len(gaps) else 0.0,
    "config_hash": config_hash(config),
}
pd.Series(metadata).to_json(CLEANSED_PATH / "metadata.json")

print(f"\nSaved to: {output_path}")
print(f"Rows: {len(raw_df):,}")
print(f"Time range: {ts_to_datetime(metadata['start_time'])} to {ts_to_datetime(metadata['end_time'])}")

# === CELL 9: Summary Statistics ===
print("\n=== Data Summary ===")
print(raw_df.describe())
```

### 8.2 `notebooks/feature_build.ipynb`

**Purpose:** Build decision bars, compute features, generate labels.

```python
# === CELL 1: Imports and Setup ===
"""
Feature Build Notebook

This notebook:
1. Loads cleansed 1-minute data
2. Aggregates to decision bars
3. Computes base and rolling features
4. Generates labels
5. Persists feature dataset

Outputs:
- /data/model_data/{symbol}/bars_20m_features.parquet
- /data/model_data/{symbol}/feature_metadata.json  [ADDED]
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# [MODIFIED] Robust project root discovery (works when launched from repo root or `notebooks/`)
ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "src"))
from utils import load_config, config_hash
from transformers.bar_aggregator import DecisionBarAggregator
from transformers.feature_pipeline import BaseFeatureExtractor, RollingFeatureExtractor

# === CELL 2: Load Configuration ===
download_config = load_config(ROOT / "config" / "download.yaml")
pipeline_config = load_config(ROOT / "config" / "pipeline.yaml")
model_config = load_config(ROOT / "config" / "model.yaml")  # [MODIFIED] single source of truth for `train_fraction`

SYMBOL = download_config["symbol"]
N = pipeline_config["decision_interval"]
WINDOWS = pipeline_config["windows"]
BURN_IN = pipeline_config["burn_in_bars"]
TRAIN_FRAC = model_config["train_fraction"]

CLEANSED_PATH = ROOT / download_config["paths"]["cleansed_data"] / SYMBOL
MODEL_DATA_PATH = ROOT / "data" / "model_data" / SYMBOL
MODEL_DATA_PATH.mkdir(parents=True, exist_ok=True)

print(f"Decision interval: {N} minutes")
print(f"Rolling windows: {WINDOWS}")

# === CELL 3: Load Minute Data ===
minute_df = pd.read_parquet(CLEANSED_PATH / "1m.parquet")
print(f"Loaded {len(minute_df):,} minute bars")

# === CELL 4: Aggregate to Decision Bars ===
aggregator = DecisionBarAggregator(n=N)
decision_bars = []

for _, row in tqdm(minute_df.iterrows(), total=len(minute_df), desc="Aggregating"):
    minute_bar = row.to_dict()
    bar = aggregator.update(minute_bar)  # [MODIFIED]
    if bar is not None:
        decision_bars.append(bar)

bars_df = pd.DataFrame(decision_bars)
print(f"Decision bars: {len(bars_df):,}")

# [ADDED] Per-segment bar index (used for burn-in and diagnostics)
bars_df["bar_in_segment"] = bars_df.groupby("segment_id").cumcount()

# === CELL 5: Compute Labels ===
# y_k = 1 if log(H_{k+1} / C_k) >= alpha
# [MODIFIED] Segment-aware label construction (no labels across gaps)
bars_df["next_high"] = bars_df["high"].shift(-1)
bars_df["next_segment_id"] = bars_df["segment_id"].shift(-1)
bars_df["log_excursion"] = np.where(
    bars_df["segment_id"] == bars_df["next_segment_id"],
    np.log(bars_df["next_high"] / bars_df["close"]),
    np.nan,
)

# Determine alpha from quantile on training portion
# [MODIFIED] Use `train_fraction` from config and prevent boundary leakage (exclude last train row)
train_end_idx = int(len(bars_df) * TRAIN_FRAC)
calibration_df = bars_df.iloc[: max(train_end_idx - 1, 0)]
train_excursions = calibration_df["log_excursion"].dropna()
if train_excursions.empty:
    raise RuntimeError("No valid excursions for alpha calibration (check gaps, date range, and split fractions).")

if pipeline_config["barrier"]["method"] == "quantile":
    quantile = pipeline_config["barrier"]["quantile"]
    alpha = train_excursions.quantile(quantile)
else:
    alpha = pipeline_config["barrier"]["fixed_alpha"]

# [MODIFIED] Preserve NaN labels at segment boundaries and final row
bars_df["label"] = np.where(
    bars_df["log_excursion"].notna(),
    (bars_df["log_excursion"] >= alpha).astype(int),
    np.nan,
)

print(f"Barrier alpha: {alpha:.6f} (log-return = {100*(np.exp(alpha)-1):.3f}%)")
print(f"Positive rate (train calibration subset): {train_excursions.ge(alpha).mean():.3%}")

# === CELL 6: Compute Features ===
base_extractor = None  # [MODIFIED] initialized per segment in the loop below
rolling_extractor = None  # [MODIFIED] initialized per segment in the loop below

features_list = []
prev_segment_id = None  # [ADDED] used to reset rolling state at segment boundaries
for _, row in tqdm(bars_df.iterrows(), total=len(bars_df), desc="Computing features"):
    bar = row.to_dict()

    # [ADDED] Reset stateful extractors at segment boundaries (prevents cross-gap leakage)
    if prev_segment_id is None or bar.get("segment_id") != prev_segment_id:
        base_extractor = BaseFeatureExtractor()
        rolling_extractor = RollingFeatureExtractor(
            base_features=pipeline_config["rolling_base_features"],
            windows=WINDOWS,
            stats_to_compute=pipeline_config["rolling_stats"]
        )
        prev_segment_id = bar.get("segment_id")
    
    # Base features
    base_feats = base_extractor.transform_one(bar)
    base_extractor.learn_one(bar)
    
    # Rolling features
    # [MODIFIED] Update then read => rolling features include current bar (no lookahead leakage)
    rolling_extractor.learn_one(base_feats)
    rolling_feats = rolling_extractor.transform_one(base_feats)
    
    features_list.append({**base_feats, **rolling_feats})

features_df = pd.DataFrame(features_list)
print(f"Feature columns: {len(features_df.columns)}")

# === CELL 7: Combine and Clean ===
# Merge features with bars
full_df = pd.concat([
    bars_df[["open_time", "close_time", "close", "segment_id", "bar_in_segment", "label"]].reset_index(drop=True),
    features_df.reset_index(drop=True)
], axis=1)

# Drop rows with NaN labels (last row)
full_df = full_df.dropna(subset=["label"])
full_df["label"] = full_df["label"].astype(int)  # [ADDED] restore integer labels after NaN removal

# Drop burn-in rows (per segment)
full_df = full_df[full_df["bar_in_segment"] >= BURN_IN].reset_index(drop=True)  # [MODIFIED]
print(f"Final rows (after burn-in): {len(full_df):,}")

# === CELL 8: Feature Sanity Checks ===
print("\n=== Sanity Checks ===")

# Check return calculation
sample_idx = 100
print(f"Sample return check at idx {sample_idx}:")
print(f"  log_close[{sample_idx}]: {full_df.loc[sample_idx, 'log_close']:.6f}")
print(f"  return[{sample_idx}]: {full_df.loc[sample_idx, 'return']:.6f}")

# Check rolling mean convergence
print(f"\nRolling mean convergence:")
print(f"  return_rolling_mean_12 mean: {full_df['return_rolling_mean_12'].mean():.6f}")
print(f"  return mean: {full_df['return'].mean():.6f}")

# === CELL 9: Save Dataset ===
id_cols = ["open_time", "close_time", "close", "segment_id", "bar_in_segment", "label"]  # [ADDED]
feature_cols = [c for c in full_df.columns if c not in id_cols]

output_path = MODEL_DATA_PATH / "bars_20m_features.parquet"
full_df.to_parquet(output_path, index=False, engine="pyarrow")

# Save feature metadata
feature_metadata = {
    "alpha": float(alpha),
    "barrier_method": pipeline_config["barrier"]["method"],
    "decision_interval": N,
    "train_fraction": float(TRAIN_FRAC),  # [ADDED] used for alpha calibration and downstream splits
    "windows": WINDOWS,
    "feature_names": feature_cols,
    "n_features": len(feature_cols),
    "n_samples": len(full_df),
    "positive_rate": float(full_df["label"].mean()),
    "config_hash": config_hash(pipeline_config),
}
pd.Series(feature_metadata).to_json(MODEL_DATA_PATH / "feature_metadata.json")

print(f"\nSaved to: {output_path}")
print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(full_df):,}")
print(f"Positive rate: {full_df['label'].mean():.3%}")
```

### 8.3 `notebooks/offline_train.ipynb`

**Purpose:** Train CatBoost classifier, evaluate, save artifacts.

```python
# === CELL 1: Imports and Setup ===
"""
Offline Training Notebook

This notebook:
1. Loads feature dataset
2. Performs chronological train/test split
3. Trains CatBoost classifier
4. Evaluates model performance
5. Saves model and artifacts

Outputs:
- /artifacts/offline_model/model.cbm
- /artifacts/offline_model/selected_features.json
- /artifacts/offline_model/metrics.json
- /artifacts/offline_model/feature_importance.csv  [ADDED]
- /artifacts/offline_model/config_snapshot.json  [ADDED]
- /artifacts/offline_model/*.png  [ADDED]
"""
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve  # [MODIFIED] correct import location

# [MODIFIED] Robust project root discovery (works when launched from repo root or `notebooks/`)
ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "src"))
from utils import load_config, config_hash

# === CELL 2: Load Configuration and Data ===
download_config = load_config(ROOT / "config" / "download.yaml")
model_config = load_config(ROOT / "config" / "model.yaml")

SYMBOL = download_config["symbol"]
TRAIN_FRAC = model_config["train_fraction"]
VAL_FRAC = model_config.get("val_fraction", 0.2)  # [ADDED] early-stopping validation fraction within train
TOP_K = model_config["top_k_features"]

MODEL_DATA_PATH = ROOT / "data" / "model_data" / SYMBOL
ARTIFACT_PATH = ROOT / "artifacts" / "offline_model"
ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_parquet(MODEL_DATA_PATH / "bars_20m_features.parquet")
with open(MODEL_DATA_PATH / "feature_metadata.json") as f:
    feature_meta = json.load(f)

feature_cols = feature_meta["feature_names"]
print(f"Samples: {len(df):,}, Features: {len(feature_cols)}")

# === CELL 3: Chronological Split ===
split_idx = int(len(df) * TRAIN_FRAC)
train_full_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# [ADDED] Validation split within the training window (prevents test-set contamination via early stopping)
val_size = int(len(train_full_df) * VAL_FRAC)
val_size = max(val_size, 1) if len(train_full_df) >= 2 else 0

if val_size > 0 and val_size < len(train_full_df):
    train_df = train_full_df.iloc[:-val_size].copy()
    val_df = train_full_df.iloc[-val_size:].copy()
else:
    train_df = train_full_df.copy()
    val_df = train_full_df.iloc[0:0].copy()  # empty

X_train = train_df[feature_cols]
y_train = train_df["label"]
X_val = val_df[feature_cols]
y_val = val_df["label"]
X_test = test_df[feature_cols]
y_test = test_df["label"]

print(f"Train: {len(train_df):,} ({y_train.mean():.3%} positive)")
print(f"Val:   {len(val_df):,} ({y_val.mean():.3%} positive)" if len(val_df) else "Val:   0 (skipped)")
print(f"Test:  {len(test_df):,} ({y_test.mean():.3%} positive)")

# === CELL 4: Train CatBoost ===
cb_params = model_config["catboost"]

model = CatBoostClassifier(**cb_params)

train_pool = Pool(X_train, y_train)
eval_pool = Pool(X_val, y_val) if len(val_df) else None

if eval_pool is not None:
    model.fit(
        train_pool,
        eval_set=eval_pool,
        use_best_model=True,
        plot=False
    )
else:
    model.fit(
        train_pool,
        use_best_model=False,
        plot=False
    )

print(f"Best iteration: {model.get_best_iteration()}")

# === CELL 5: Predictions ===
y_prob_train = model.predict_proba(X_train)[:, 1]
y_prob_test = model.predict_proba(X_test)[:, 1]

# === CELL 6: Evaluation Metrics ===
def safe_roc_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan

def safe_pr_auc(y, p):
    return average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan

metrics = {
    "train": {
        "roc_auc": safe_roc_auc(y_train, y_prob_train),
        "pr_auc": safe_pr_auc(y_train, y_prob_train),
        "brier_score": brier_score_loss(y_train, y_prob_train),
    },
    "test": {
        "roc_auc": safe_roc_auc(y_test, y_prob_test),
        "pr_auc": safe_pr_auc(y_test, y_prob_test),
        "brier_score": brier_score_loss(y_test, y_prob_test),
    }
}

print("\n=== Model Performance ===")
for split, m in metrics.items():
    print(f"{split.upper()}:")
    print(f"  ROC AUC: {m['roc_auc']:.4f}")
    print(f"  PR AUC: {m['pr_auc']:.4f}")
    print(f"  Brier Score: {m['brier_score']:.4f}")

# === CELL 7: ROC and PR Curves ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
if len(np.unique(y_test)) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    axes[0].plot(fpr, tpr, label=f"AUC = {metrics['test']['roc_auc']:.3f}")
else:
    axes[0].text(0.5, 0.5, "ROC undefined (single class)", ha="center", va="center")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve (Test)")
axes[0].legend()

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob_test)
baseline = y_test.mean()
axes[1].plot(recall, precision, label=f"AP = {metrics['test']['pr_auc']:.3f}")
axes[1].axhline(y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline = {baseline:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve (Test)")
axes[1].legend()

plt.tight_layout()
plt.savefig(ARTIFACT_PATH / "roc_pr_curves.png", dpi=150)
plt.show()

# === CELL 8: Calibration Curve ===
fig, ax = plt.subplots(figsize=(6, 6))

if 0 < y_test.sum() < len(y_test):
    fraction_pos, mean_pred = calibration_curve(y_test, y_prob_test, n_bins=10, strategy="uniform")

    ax.plot(mean_pred, fraction_pos, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Test)")
    ax.legend()
else:
    ax.text(0.5, 0.5, "Calibration undefined (single class)", ha="center", va="center")

plt.tight_layout()
plt.savefig(ARTIFACT_PATH / "calibration_curve.png", dpi=150)
plt.show()

# === CELL 9: Probability Histograms ===
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(y_prob_test[y_test == 0], bins=50, alpha=0.5, label="Negative", density=True)
ax.hist(y_prob_test[y_test == 1], bins=50, alpha=0.5, label="Positive", density=True)
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Density")
ax.set_title("Probability Distribution by Class (Test)")
ax.legend()

plt.tight_layout()
plt.savefig(ARTIFACT_PATH / "probability_histogram.png", dpi=150)
plt.show()

# === CELL 10: Feature Importance ===
importance = model.get_feature_importance(type="PredictionValuesChange")
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance
}).sort_values("importance", ascending=False)

# Select top-K features for online model
selected_features = importance_df.head(TOP_K)["feature"].tolist()

print(f"\n=== Top {TOP_K} Features ===")
print(importance_df.head(TOP_K).to_string(index=False))

# === CELL 11: Save Artifacts ===
# Save model
model.save_model(str(ARTIFACT_PATH / "model.cbm"))

# Save selected features
with open(ARTIFACT_PATH / "selected_features.json", "w") as f:
    json.dump({"features": selected_features}, f, indent=2)

# Save metrics
with open(ARTIFACT_PATH / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save feature importance
importance_df.to_csv(ARTIFACT_PATH / "feature_importance.csv", index=False)

# Save config snapshot
config_snapshot = {
    "model_config": model_config,
    "feature_metadata": feature_meta,
}
with open(ARTIFACT_PATH / "config_snapshot.json", "w") as f:
    json.dump(config_snapshot, f, indent=2)

print(f"\nArtifacts saved to: {ARTIFACT_PATH}")
```

### 8.4 `notebooks/online_eval.ipynb`

**Purpose:** Streaming evaluation with online correction layer.

```python
# === CELL 1: Imports and Setup ===
"""
Online Evaluation Notebook

This notebook:
1. Loads test data and trained offline model
2. Initializes online correction model
3. Runs streaming evaluation loop
4. Reports prequential metrics
5. Compares offline vs. corrected predictions

Outputs:
- /artifacts/online_eval/metrics.json
- /artifacts/online_eval/plots/
- /artifacts/online_eval/predictions.parquet  [ADDED]
- /artifacts/online_eval/periodic_metrics.csv  [ADDED]
"""
import sys
import json
from pathlib import Path
from collections import deque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from catboost import CatBoostClassifier
from river.forest import ARFClassifier  # [MODIFIED] River 0.21.0 ARF classifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.calibration import calibration_curve  # [MODIFIED] correct import location

# [MODIFIED] Robust project root discovery (works when launched from repo root or `notebooks/`)
ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "src"))
from utils import load_config

# === CELL 2: Load Configuration and Artifacts ===
download_config = load_config(ROOT / "config" / "download.yaml")
model_config = load_config(ROOT / "config" / "model.yaml")

SYMBOL = download_config["symbol"]
TRAIN_FRAC = model_config["train_fraction"]
REPORT_EVERY = model_config["evaluation"]["report_every_bars"]

MODEL_DATA_PATH = ROOT / "data" / "model_data" / SYMBOL
OFFLINE_PATH = ROOT / "artifacts" / "offline_model"
ONLINE_EVAL_PATH = ROOT / "artifacts" / "online_eval"
ONLINE_EVAL_PATH.mkdir(parents=True, exist_ok=True)
(ONLINE_EVAL_PATH / "plots").mkdir(exist_ok=True)

# Load data
df = pd.read_parquet(MODEL_DATA_PATH / "bars_20m_features.parquet")
with open(MODEL_DATA_PATH / "feature_metadata.json") as f:
    feature_meta = json.load(f)
feature_cols = feature_meta["feature_names"]

# Load offline model
offline_model = CatBoostClassifier()
offline_model.load_model(str(OFFLINE_PATH / "model.cbm"))

# Load selected features for online model
with open(OFFLINE_PATH / "selected_features.json") as f:
    selected_features = json.load(f)["features"]

print(f"Offline model loaded")
print(f"Selected features for online: {len(selected_features)}")

# === CELL 3: Prepare Test Data ===
split_idx = int(len(df) * TRAIN_FRAC)
test_df = df.iloc[split_idx:].reset_index(drop=True)

# Include warm-up prefix for rolling feature stability
WARMUP_BARS = max([int(w) for w in feature_meta.get("windows", [48])])
warmup_start = max(split_idx - WARMUP_BARS, 0)  # [ADDED] guard against small datasets
warmup_df = df.iloc[warmup_start:split_idx].reset_index(drop=True)

print(f"Test samples: {len(test_df):,}")
print(f"Warmup samples: {len(warmup_df):,}")

# === CELL 4: Initialize Online Model ===
online_config = model_config["online"]

online_model = ARFClassifier(
    n_models=online_config["n_models"],
    max_features=online_config["max_features"],
    lambda_value=online_config["lambda_value"],
    seed=online_config["seed"]
)

# Label delay buffer (horizon = 1 bar)
# Stores: (z_k, ref_close)
label_buffer = deque()

# === CELL 5: Streaming Loop ===
"""
Online Loop Order (per decision bar k):
1. Consume X_k
2. Compute phi_k (features - already computed in feature_build)
3. Get offline prediction p_off
4. Build online input z_k = concat(selected_features, p_off)
5. Get online prediction p_final
6. Store (z_k, C_k) in buffer awaiting label
7. When X_{k+1} arrives, compute y_k and train online model
"""

# Results storage
results = {
    "k": [],
    "y_true": [],
    "p_offline": [],
    "p_online": [],
    "p_final": [],
}

# [MODIFIED] Process warmup and test in a single prequential loop.
# This ensures labels are aligned correctly for the delayed-label update.
print("Processing warmup + test (prequential)...")
stream_df = pd.concat(
    [
        warmup_df.assign(_is_warmup=True),
        test_df.assign(_is_warmup=False),
    ],
    ignore_index=True,
)

for stream_idx in tqdm(range(len(stream_df)), desc="Streaming"):
    row = stream_df.iloc[stream_idx]
    is_warmup = bool(row["_is_warmup"])

    features = row[feature_cols].to_dict()

    # --- Step 3: Offline prediction ---
    X_row = pd.DataFrame([row[feature_cols]])
    p_off = float(offline_model.predict_proba(X_row)[0, 1])

    # --- Step 4: Build online input ---
    z = {f: features[f] for f in selected_features}
    z["p_offline"] = p_off

    # --- Step 5: Online prediction (before learning) ---
    p_online_dict = online_model.predict_proba_one(z)
    p_online = p_online_dict.get(1, 0.5) if p_online_dict else 0.5

    # Final prediction (weighted combination or just online)
    p_final = p_online  # Can implement weighted ensemble here

    # --- Step 7: Train online model with delayed label (horizon = 1) ---
    if len(label_buffer) > 0:
        buffered = label_buffer.popleft()
        z_prev = buffered["z"]
        ref_close = buffered["ref_close"]

        current_high = row["high"]
        log_excursion = np.log(current_high / ref_close)
        y_delayed = int(log_excursion >= feature_meta["alpha"])

        online_model.learn_one(z_prev, y_delayed)

    # --- Step 6: Buffer current sample ---
    label_buffer.append({
        "z": z.copy(),
        "ref_close": row["close"],
    })

    if is_warmup:
        continue

    idx = stream_idx - len(warmup_df)  # test-relative index
    y_true = int(row["label"])

    # Store results
    results["k"].append(idx)
    results["y_true"].append(y_true)
    results["p_offline"].append(p_off)
    results["p_online"].append(p_online)
    results["p_final"].append(p_final)

# === CELL 6: Convert Results to DataFrame ===
results_df = pd.DataFrame(results)
print(f"Evaluation complete: {len(results_df):,} predictions")

# === CELL 7: Overall Metrics ===
y_true = results_df["y_true"].values
p_offline = results_df["p_offline"].values
p_final = results_df["p_final"].values

# [ADDED] Robust metric helpers for edge cases (e.g., all-zero labels in a slice)
def safe_roc_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan

def safe_pr_auc(y, p):
    return average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan

metrics = {
    "offline": {
        "roc_auc": safe_roc_auc(y_true, p_offline),
        "pr_auc": safe_pr_auc(y_true, p_offline),
    },
    "online_corrected": {
        "roc_auc": safe_roc_auc(y_true, p_final),
        "pr_auc": safe_pr_auc(y_true, p_final),
    },
    "sample_count": len(y_true),
    "positive_rate": float(y_true.mean()),
}

print("\n=== Online Evaluation Results ===")
print(f"Samples: {metrics['sample_count']:,}")
print(f"Positive rate: {metrics['positive_rate']:.3%}")
print(f"\nOffline Model:")
print(f"  ROC AUC: {metrics['offline']['roc_auc']:.4f}")
print(f"  PR AUC: {metrics['offline']['pr_auc']:.4f}")
print(f"\nOnline Corrected:")
print(f"  ROC AUC: {metrics['online_corrected']['roc_auc']:.4f}")
print(f"  PR AUC: {metrics['online_corrected']['pr_auc']:.4f}")

# === CELL 8: Periodic Reporting ===
print("\n=== Periodic Metrics ===")
periodic_metrics = []

for i in range(0, len(results_df), REPORT_EVERY):
    chunk = results_df.iloc[i:i+REPORT_EVERY]
    if len(chunk) < 10:  # Skip tiny chunks
        continue
    
    y_chunk = chunk["y_true"].values
    p_off_chunk = chunk["p_offline"].values
    p_fin_chunk = chunk["p_final"].values
    
    periodic_metrics.append({
        "start_k": i,
        "end_k": i + len(chunk),
        "count": len(chunk),
        "positive_rate": y_chunk.mean(),
        "offline_roc_auc": roc_auc_score(y_chunk, p_off_chunk) if 0 < y_chunk.sum() < len(y_chunk) else np.nan,
        "final_roc_auc": roc_auc_score(y_chunk, p_fin_chunk) if 0 < y_chunk.sum() < len(y_chunk) else np.nan,
    })

periodic_df = pd.DataFrame(periodic_metrics)
print(periodic_df.to_string(index=False))

# === CELL 9: ROC/PR Comparison Plot ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curves
if len(np.unique(y_true)) == 2:
    for label, probs, color in [
        ("Offline", p_offline, "blue"),
        ("Online Corrected", p_final, "orange")
    ]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        axes[0].plot(fpr, tpr, color=color, label=f"{label} (AUC={auc:.3f})")
else:
    axes[0].text(0.5, 0.5, "ROC undefined (single class)", ha="center", va="center")

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve Comparison")
axes[0].legend()

# PR Curves
if 0 < y_true.sum() < len(y_true):
    for label, probs, color in [
        ("Offline", p_offline, "blue"),
        ("Online Corrected", p_final, "orange")
    ]:
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        axes[1].plot(recall, precision, color=color, label=f"{label} (AP={ap:.3f})")
else:
    axes[1].text(0.5, 0.5, "PR undefined (single class)", ha="center", va="center")

baseline = y_true.mean()
axes[1].axhline(y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline={baseline:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("PR Curve Comparison")
axes[1].legend()

plt.tight_layout()
plt.savefig(ONLINE_EVAL_PATH / "plots" / "roc_pr_comparison.png", dpi=150)
plt.show()

# === CELL 10: Calibration Comparison ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (label, probs) in zip(axes, [("Offline", p_offline), ("Online", p_final)]):
    if 0 < y_true.sum() < len(y_true):
        fraction_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, fraction_pos, "o-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration: {label}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Calibration undefined (single class)", ha="center", va="center")
        ax.set_title(f"Calibration: {label}")

plt.tight_layout()
plt.savefig(ONLINE_EVAL_PATH / "plots" / "calibration_comparison.png", dpi=150)
plt.show()

# === CELL 11: Time Series Plot ===
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Subsample for visualization
plot_df = results_df.iloc[::10]  # Every 10th point

axes[0].plot(plot_df["k"], plot_df["p_offline"], alpha=0.7, label="Offline")
axes[0].plot(plot_df["k"], plot_df["p_final"], alpha=0.7, label="Final")
axes[0].set_ylabel("Probability")
axes[0].set_title("Predictions Over Time")
axes[0].legend()

axes[1].scatter(plot_df["k"], plot_df["y_true"], alpha=0.3, s=1)
axes[1].set_ylabel("True Label")
axes[1].set_title("Actual Labels")

# Rolling accuracy
window = 100
results_df["offline_pred"] = (results_df["p_offline"] > 0.5).astype(int)
results_df["final_pred"] = (results_df["p_final"] > 0.5).astype(int)
results_df["offline_correct"] = (results_df["offline_pred"] == results_df["y_true"]).astype(int)
results_df["final_correct"] = (results_df["final_pred"] == results_df["y_true"]).astype(int)

axes[2].plot(results_df["k"], results_df["offline_correct"].rolling(window).mean(), 
             label="Offline", alpha=0.7)
axes[2].plot(results_df["k"], results_df["final_correct"].rolling(window).mean(), 
             label="Final", alpha=0.7)
axes[2].set_xlabel("Decision Bar Index")
axes[2].set_ylabel(f"Rolling Accuracy (w={window})")
axes[2].set_title("Rolling Accuracy Comparison")
axes[2].legend()

plt.tight_layout()
plt.savefig(ONLINE_EVAL_PATH / "plots" / "time_series.png", dpi=150)
plt.show()

# === CELL 12: Save Results ===
with open(ONLINE_EVAL_PATH / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

results_df.to_parquet(ONLINE_EVAL_PATH / "predictions.parquet", index=False)
periodic_df.to_csv(ONLINE_EVAL_PATH / "periodic_metrics.csv", index=False)

print(f"\nResults saved to: {ONLINE_EVAL_PATH}")
```

---

## 9. Execution Order

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXECUTION WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SETUP                                                       │
│     ├── pip install -r requirements.txt                         │
│     └── Configure config/*.yaml files                           │
│                                                                 │
│  2. DATA PREPARATION                                            │
│     └── Run: notebooks/data_download.ipynb                      │
│         ├── Downloads Binance kline data                        │
│         ├── Validates and cleanses                              │
│         └── Output: data/cleansed_data/{symbol}/1m.parquet      │
│                                                                 │
│  3. FEATURE ENGINEERING                                         │
│     └── Run: notebooks/feature_build.ipynb                      │
│         ├── Aggregates to decision bars                         │
│         ├── Computes features (base + rolling)                  │
│         ├── Generates labels                                    │
│         └── Output: data/model_data/{symbol}/bars_20m_*.parquet │
│                                                                 │
│  4. OFFLINE TRAINING                                            │
│     └── Run: notebooks/offline_train.ipynb                      │
│         ├── Chronological train/test split                      │
│         ├── Trains CatBoost classifier                          │
│         ├── Evaluates (ROC, PR, calibration)                    │
│         ├── Selects top-K features                              │
│         └── Output: artifacts/offline_model/*                   │
│                                                                 │
│  5. ONLINE EVALUATION                                           │
│     └── Run: notebooks/online_eval.ipynb                        │
│         ├── Loads offline model                                 │
│         ├── Initializes River ARF                               │
│         ├── Runs prequential evaluation                         │
│         ├── Compares offline vs. corrected                      │
│         └── Output: artifacts/online_eval/*                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Correctness Invariants

### 10.1 Temporal Leakage Prevention

**INVARIANT 1 (Feature Causality):** At decision time $k$, feature vector $\phi_k$ depends only on $\{X_0, X_1, \ldots, X_k\}$.

**INVARIANT 2 (Label Delay):** Label $y_k$ depends on $X_{k+1}$. Online model training for $y_k$ occurs only after $X_{k+1}$ is observed.

**Implementation Enforcement:**
```python
# In streaming loop:
# 1. Predict with current features (no future information)
p_dict = online_model.predict_proba_one(z_k)
p_final = p_dict.get(1, 0.5) if p_dict else 0.5  # [MODIFIED] River returns a dict

# 2. Buffer z_k for delayed training
label_buffer.append(z_k)

# 3. Only when X_{k+1} arrives, compute y_k and train
z_prev = label_buffer.popleft()
y_k = compute_label(H_{k+1}, C_k)
online_model.learn_one(z_prev, y_k)
```

### 10.2 Frequency Consistency

**INVARIANT 3:** Samples are emitted only at decision boundaries. No per-minute samples with overlapping horizons.

### 10.3 Rolling Statistic Stability

**INVARIANT 4:** [MODIFIED] First `max(windows)` decision bars of **each segment** are excluded from training/evaluation to ensure rolling statistics have stabilized.

### 10.4 Scientific Assumptions and Limitations  [ADDED]

**Volatility Estimators (Parkinson / Garman–Klass):**
- **Assumption (Idealized):** Continuous-time diffusion (often modeled as geometric Brownian motion) within the bar; observed $H_k$ and $L_k$ approximate continuous extremes.
- **Assumption (Parkinson):** Zero drift within the bar; estimator can be biased under strong drift.
- **Limitation:** Discrete sampling, microstructure noise, and data gaps violate assumptions and can bias estimators; this spec mitigates the most severe gap effects by segmenting and resetting state (Sections 4.3, 6.2, 6.3), but does not “fix” estimator bias.

**Rolling Statistics:**
- **Assumption:** Rolling estimates are treated as descriptive summaries, not IID statistics; no IID/stationarity assumption is required for computation.
- **Limitation:** Edge effects during warm-up; this is handled via per-segment burn-in (Section 3.5).

**Time Series (Non-IID):**
- **Assumption:** Observations are time-ordered and generally non-IID; evaluation must respect chronology.
- **Mitigation:** This spec enforces chronological splits (offline) and prequential evaluation with delayed labels (online), preventing lookahead leakage (Section 10.1).

**Class Imbalance:**
- **Assumption:** Positive events are rare by construction (quantile barrier calibration).
- **Mitigation:** Report PR-AUC and calibration diagnostics in addition to ROC-AUC; handle single-class slices explicitly (Sections 2.2.3, 8.3, 8.4).

**Online Correction Model:**
- **Assumption:** Concept drift may occur; prequential learning is used.
- **Limitation:** Cold start (before any `learn_one`) yields undefined probabilities; this spec uses a neutral default of `0.5` and reports performance only after data accrues (Sections 2.2.1, 8.4).

### 10.5 Edge Case Policy  [ADDED]

| Edge Case | Specified Handling |
|----------|---------------------|
| $H_k=L_k$ (zero range) | `range_hl=0`, `parkinson_var=0`, `body_fraction=0`, `flag__range_zero=1` |
| GK variance negative | Clip `garman_klass_var` to `0.0`, emit `flag__gk_negative=1` |
| $V_k=0$ (zero volume) | `buy_ratio=0.5`, `imbalance=0.0`, `flag__no_volume=1` |
| $N_k=0$ (zero trades) | `avg_trade_size=0.0`, `flag__no_trades=1` |
| First bar in segment | `return=0.0`, `flag__first_bar=1`, `flag__segment_start=1` |
| Missing minute bars / gaps | Do not interpolate; start a new segment and reset aggregators/rolling state; do not label across segments (Sections 4.3, 6.2, 3.3) |
| All labels are one class in a slice | Report ROC AUC / PR AUC as `NaN` for that slice (see safe metric helpers in notebooks) |

## 11. References

### Academic Papers

1. **Parkinson, M.** (1980). *The extreme value method for estimating the variance of the rate of return.* The Journal of Business, 53(1), 61-65.

2. **Garman, M. B., & Klass, M. J.** (1980). *On the estimation of security price volatilities from historical data.* The Journal of Business, 53(1), 67-78.

3. **Amihud, Y.** (2002). *Illiquidity and stock returns: cross-section and time-series effects.* Journal of Financial Markets, 5(1), 31-56.

4. **López de Prado, M.** (2018). *Advances in Financial Machine Learning.* Wiley.

5. **Gomes, H. M., et al.** (2017). *Adaptive random forests for evolving data stream classification.* Machine Learning, 106(9-10), 1469-1495.

6. **Montiel, J., et al.** (2021). *River: machine learning for streaming data in Python.* Journal of Machine Learning Research, 22(110), 1-8.

### Library Documentation

7. **River Documentation:** https://riverml.xyz/
   - [MODIFIED] Rolling stats: `river.utils.Rolling`, `river.stats.Mean`, `river.stats.Var`, `river.stats.RollingMin`, `river.stats.RollingMax`
   - [MODIFIED] Online forest classifier: `river.forest.ARFClassifier`
   - `river.base.Transformer`: Base class for custom transformers

8. **CatBoost Documentation:** https://catboost.ai/docs/
   - `CatBoostClassifier`: Gradient boosting classifier
   - `predict_proba`: Probability predictions
   - `get_feature_importance`: Feature importance extraction

9. **scikit-learn Metrics:** https://scikit-learn.org/stable/modules/model_evaluation.html
   - `roc_auc_score`, `roc_curve`: ROC analysis
   - `precision_recall_curve`, `average_precision_score`: PR analysis
   - [MODIFIED] `calibration_curve` (in `sklearn.calibration`): Reliability diagrams

10. **Binance Public Data:** https://github.com/binance/binance-public-data
    - Kline data format and download procedures

---

## 12. Appendix: Quick Reference

### Feature Naming Convention

```
{base_feature}_rolling_{statistic}_{window}
```

Examples:
- `return_rolling_mean_12` — 12-bar rolling mean of returns
- `garman_klass_var_rolling_var_48` — 48-bar rolling variance of GK estimator

### Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `n` | 20 | Minutes per decision bar |
| `epsilon` | 1e-10 | Numerical stability |
| `2*ln(2) - 1` | 0.3862943611 | Garman-Klass coefficient |
| `1 / (4*ln(2))` | 0.3606737602 | Parkinson coefficient |

### Default Configuration

```yaml
decision_interval: 20
windows: [1, 2, 4, 12, 48]
barrier:
  method: "quantile"
  quantile: 0.95
  fixed_alpha: 0.0025
burn_in_bars: 48
train_fraction: 0.6
val_fraction: 0.2
top_k_features: 50
catboost:
  iterations: 500
online:
  n_models: 10
```

---

[MODIFIED] *Validated specification. See “Validation Summary”, “Change Log”, and “Verification Evidence” for corrections and provenance.*
