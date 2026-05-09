# Local Literature Index

The autonomous loop's `LITERATURE-SCOUT` checks this index **before** any web search. Web search is the fallback when the local corpus is silent on a topic.

Documents catalogued: 2026-05-09. Re-run the triage agent to refresh after major Downloads/Desktop changes.

Tags: `[ONLINE]` `[FIN-ML]` `[CALIB-UQ]` `[MICROSTRUCT]` `[VOL]` `[GBM]` `[FEATURES]` `[BACKTEST]` `[STATS-CI]` `[MISC]` `[MATH-FDN]`

**Where the literature actually lives**:
- `C:\Users\vitil\Downloads\` — the **applied** finance/ML corpus (López de Prado AFML, conformal prediction, CatBoost paper, Sugiyama covariate shift, Multiscale Stochastic Volatility, Kaufman backtesting, etc.). Default search target for LITERATURE-SCOUT.
- `C:\Users\vitil\OneDrive\Desktop\paper_editing\sources\_all_pdfs\` and `C:\Users\vitil\OneDrive\Desktop\dynamic_network_games\literature\` — the **mathematical foundations** (probability, measure theory, empirical processes, high-dim stats). NOT primary finance references; pull these in only when a probabilistic claim needs textbook backing. See `[MATH-FDN]` section below.

---

## Best 10 (highest immediate relevance to the BACKLOG)

1. **[FIN-ML]** López de Prado, *Advances in Financial Machine Learning* (2018) — `C:\Users\vitil\Downloads\Advances in Financial Machine Learning -- López de Prado, Marcos -- 2018 -- 0645579add65f44b6809abb008da88d9 -- Anna's Archive.pdf`. Triple-barrier labeling (Ch. 3), backtest statistics (Ch. 14), meta-labeling (Ch. 3.6), purged CV / embargo (Ch. 7), feature importance (Ch. 8). The canonical reference for almost every BACKLOG item.
2. **[CALIB-UQ]** Manokhin, *Practical Guide to Applied Conformal Prediction in Python* (2024) — `C:\Users\vitil\Downloads\Valeri Manokhin - Practical Guide to Applied Conformal Prediction in Python (2024).pdf`. Direct implementation guide for H-011.
3. **[CALIB-UQ]** Vovk et al., *Conformal Decision Theory* / *Conformal Decisions* — `C:\Users\vitil\Downloads\conformal_decision_theory.pdf` / `conformal_decisions_vovk.pdf`. Foundations for distribution-free prediction sets and the decision-theoretic framing that maps `p`+UQ to a betting policy.
4. **[GBM/CALIB-UQ]** XGBoostLSS (1907.03178v4) — `C:\Users\vitil\Downloads\1907.03178v4.pdf`. Distributional gradient boosting predicting the full conditional distribution. Directly competes with CatBoost virtual ensembles for H-010.
5. **[CALIB-UQ]** "1st-place-catboost-cqr" Kaggle notebook — `C:\Users\vitil\Downloads\1st-place-catboost-cqr-21-07-2024-01.ipynb`. CatBoost + conformal quantile regression, winning solution. Reference implementation for H-010/H-011 fusion.
6. **[ONLINE]** Sugiyama & Kawanabe, *Machine Learning in Non-Stationary Environments* (MIT 2012) — `C:\Users\vitil\Downloads\(Adaptive Computation and Machine Learning series) Masashi Sugiyama, Motoaki Kawanabe - Machine Learning in Non-Stationary Environments_ Introduction to Covariate Shift Adaptation-The MIT Press (2012).pdf`. Covariate shift, importance weighting, model-validation under non-stationarity. Critical for H-030..H-034 (online ensemble work).
7. **[ONLINE]** Hazan, *Online Convex Optimization* — `C:\Users\vitil\Downloads\online-convext-optimization-book.pdf`. Theoretical floor for adaptive online algorithms and regret bounds.
8. **[VOL]** *Multiscale Stochastic Volatility for Equity, Interest Rate and Credit Derivatives* — `C:\Users\vitil\Downloads\Multiscale Stochastic Volatility For Equity, Interest Rate, And Credit Derivatives.pdf`. HAR-RV, multi-scale variance decomposition. Direct reference for H-040 (Hurst/DFA), H-042 (HAR-RV residuals).
9. **[GBM]** Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features* — `C:\Users\vitil\Downloads\catboostPaper.pdf`. The model's official paper; ordered boosting, target leakage, categorical handling.
10. **[BACKTEST]** Kaufman, *Trading Systems and Methods* (Wiley 2019) — `C:\Users\vitil\Downloads\Perry J. Kaufman - Trading Systems and Methods (Wiley Trading)`. Practical backtesting, position sizing, transaction-cost modelling — directly informs H-001 follow-ups (H-005, H-007).

---

## By topic

### [FIN-ML] Financial ML / triple-barrier / barrier classification
- López de Prado (2018), AFML — see Best 10 #1.
- `BTC_barrier_trading/docs/DESIGN.md` — local design doc for the production system this project will eventually feed; layered architecture (data ⇒ feature ⇒ CatBoost ⇒ online calibration ⇒ trading). Reference target for `H-061` (production engineering arc).
- `Binance BTC_USDT Derivatives Data & Feature Engineering.pdf` — direct project notes on the derivatives data this loop will optionally consume.
- Chan, *Quantitative Trading* (Wiley) — practical algorithmic-trading fundamentals.

### [CALIB-UQ] Calibration, conformal prediction, predictive intervals
- Manokhin (2024), *Practical Conformal Prediction* — see Best 10 #2.
- Vovk et al., *Conformal Decision Theory* — see Best 10 #3.
- `conformal_control.pdf` — adaptive/regime-aware conformal methods; relevant for `calibration_by_regime` (Section 11.4) extension.
- "1st-place-catboost-cqr" Kaggle notebook — see Best 10 #5.
- James, Witten, Hastie et al., *ISLP* (2023) — calibration curves, CV theory for cross-checking.

### [GBM] Gradient boosting / CatBoost / distributional ML
- CatBoost paper — see Best 10 #9.
- XGBoostLSS — see Best 10 #4.
- *Mastering Advanced Time Series Forecasting in Python: Probabilistic, Hierarchical, and Foundation Models* — broader probabilistic forecasting context.

### [ONLINE] Online learning / drift / streaming ML
- Sugiyama & Kawanabe (2012) — see Best 10 #6.
- Hazan, *Online Convex Optimization* — see Best 10 #7.
- `bfs_book_2023_online.pdf` — recent online-optimization book (verify title before citing).

### [VOL] Volatility estimators / regime detection
- *Multiscale Stochastic Volatility* — see Best 10 #8.
- *Generalized Additive Models for Location, Scale and Shape* (Stasinopoulos et al., Cambridge) — distributional regression with explicit scale modelling; alternative angle on regime stratification.

### [MICROSTRUCT] Microstructure / order flow
- `microstructure.pdf` — fundamentals (verify exact title; relevant for H-043 microprice proxy).
- *Binance BTC/USDT Derivatives Data* note — see [FIN-ML] section.

### [FEATURES] Time-series features
- `feature engineering.pdf` — generic reference.
- `temporal_hierachies_forecasting.pdf` — multi-scale temporal structure (relevant to H-041 wavelet decomposition).
- `ComplexSeasonality.pdf` — relevant for Group Q seasonality extension.

### [BACKTEST] Backtest methodology
- Kaufman (2019) — see Best 10 #10.
- López de Prado (2018) Ch. 11 + 14 — see Best 10 #1.

### [MISC] Adjacent / supporting
- Hansen, *Econometrics* (2022) — modern causal-inference reference for any ablation study.
- Kochenderfer & Wheeler, *Algorithms for Optimization* — Optuna context, multi-objective methods.
- Recursive Macroeconomic Theory (Ljungqvist & Sargent) — DP for inventory-aware control (relevant if H-007 extends to multi-asset inventory).

### [MATH-FDN] Mathematical foundations (Desktop, not applied finance)

These are pure math/probability/statistics texts on the user's Desktop (mostly under `paper_editing/sources/_all_pdfs/` and `dynamic_network_games/literature/`). They are **not direct finance references**; cite them only when a round needs rigorous justification of a probabilistic argument, concentration bound, or measure-theoretic detail. Loop sub-agents should default to applied references in Downloads first; pull these out when the THEORIST flags a non-trivial probabilistic claim that needs a textbook proof.

| Reference | Local path | Use when |
|---|---|---|
| **Wainwright**, *High-Dimensional Statistics* | `Desktop\paper_editing\sources\_all_pdfs\Wainwright - High-Dimensional Statistics.pdf` | Concentration inequalities for sample Sharpe / Brier; high-dim feature-selection theory; sub-gaussian / sub-exponential bounds for tail-truncated PnL distributions. |
| **Vershynin**, *High-Dimensional Probability* | `Desktop\paper_editing\sources\_all_pdfs\Vershynin - High-Dimensional Probability ... Second Edition.pdf` | Hoeffding/Bernstein for short-window estimators; covariance estimation under p ~ n; matrix concentration. |
| **van der Vaart & Wellner**, *Weak Convergence and Empirical Processes* | `Desktop\paper_editing\sources\_all_pdfs\vanDerVaartEtAl - Weak Convergence and Empirical Processes.pdf` | Bootstrap consistency for Sharpe / deflated Sharpe; uniform CLTs for the no-skill bootstrap p-value; empirical-process technology behind PBO/CSCV. |
| **Kallenberg**, *Foundations of Modern Probability* | `Desktop\paper_editing\sources\_all_pdfs\Kallenberg - Foundations of Modern Probability - 10e30a97.pdf` | First-touch / stopping-time / martingale arguments behind triple-barrier behaviour; rigorous probability for theoretical falsification claims. |
| **Cover & Thomas**, *Elements of Information Theory* | `Desktop\paper_editing\sources\_all_pdfs\Cover - Elements of information theory.pdf` | Entropy / mutual information arguments — directly supports H-040 (Hurst & sample entropy), and any future mutual-information-based feature-selection round. |
| **Csiszár & Shields**, *Information Theory and Statistics* | `Desktop\paper_editing\sources\_all_pdfs\CsiszarEtAl - Information Theory and Statistics_ A Tutorial.pdf` | KL divergence, importance sampling, KLIEP for H-035 (covariate-shift weighting). |
| **Tao**, *An Introduction to Measure Theory* | `Desktop\paper_editing\sources\_all_pdfs\Tao - An Introduction to Measure Theory.pdf` | Dominated convergence and integrability arguments when a theoretical claim hinges on measurability. Use sparingly. |
| **Bogachev**, *Measure Theory* (Vols I, II) | `Desktop\paper_editing\sources\_all_pdfs\Bogachev - Measure Theory Volume {I,II}.pdf` | Reference for advanced measure theory; same use pattern as Tao but more comprehensive. |
| **Axler**, *Measure, Integration & Real Analysis* | `Desktop\paper_editing\sources\_all_pdfs\Axler - Measure, Integration & Real Analysis.pdf` | Friendly intro covering the same ground as Tao. |
| **Lebl**, *Basic Analysis: Real Analysis* | `Desktop\paper_editing\sources\_all_pdfs\Lebl - Basic Analysis_ Introduction to Real Analysis.pdf` | Open-source real-analysis reference; the lightest math-foundations citation. |

**Coverage note**: there is no high-frequency-trading or stochastic-control textbook in this group. For continuous-time finance / Itô / jump-diffusion the closest available are *Mastering Mathematical Finance: Credit Risk* (Capiński & Zastawniak, in Downloads) and the *Multiscale Stochastic Volatility* book (Downloads, [VOL] section). Anything beyond that needs a web fetch (e.g., Cont & Tankov, *Financial Modelling with Jump Processes*).

---

## How to use this index

A LITERATURE-SCOUT sub-agent's first action on a hypothesis must be:

1. Open this file and grep the topic keywords.
2. If a local match exists, **read the relevant chapter or paper directly** (use the `Read` tool with `pages: "<range>"` for large PDFs).
3. Cite local pdf path + page range in HYPOTHESIS.md. No web fallback unless step 2 returns nothing relevant.
4. Web search is permitted only for: post-2024 publications, very specific implementation details (`river.forest.ARFClassifier` API), or to fill gaps the index has flagged as missing (e.g., explicit River-ecosystem papers).

**Coverage gaps** flagged by the triage (most fetched in round-014; remainder still web-fetch):
- River framework papers (Montiel et al., 2021) — **CATALOGUED round-014** (web-only; see `[ONLINE]` section).
- ARF / SRP / HAT papers (Gomes et al., Bifet et al.) — **CATALOGUED round-014** (web-only).
- Virtual-ensemble paper (Malinin et al. 2021 ICLR) — **CATALOGUED round-014** (web-only).
- Selective-prediction paper (Geifman & El-Yaniv 2017 NeurIPS) — **CATALOGUED round-014** (web-only).
- Per-metric bootstrap papers (DeLong 1988, Sun-Xu 2014, Boyd-Eng-Page 2013, Politis-Romano 1994, Politis-White 2004, Niculescu-Mizil-Caruana 2005, Roelofs 2022) — **CATALOGUED round-014** (web-only); see new `[STATS-CI]` section below.
- High-frequency limit-order-book microstructure beyond a generic primer — must web-fetch when needed.
- Bailey & López de Prado deflated-Sharpe paper (SSRN) — already cited via web; consider downloading.
- Barndorff-Nielsen & Shephard (2004) BPV paper — must web-fetch when H-312 runs.
- Peng et al. (1994) original DFA paper — must web-fetch when H-313 runs.
- Hasbrouck (1991) VAR-impulse trade-vs-price decomposition — must web-fetch when H-311 runs.

**Refresh trigger**: any time the user mentions a paper not catalogued here, update this file and re-run the HOUSEKEEPER triage.

---

## Newly catalogued (round-014, research-planning round)

These references are web-only (not in local `Downloads/Desktop` corpus). They back hypothesis cards H-302, H-305, H-320-a, H-320-b — see `RESEARCH/research_plan_round_014.md` for the per-card mapping.

### [CALIB-UQ] (additions)
- **Malinin, Prokhorenkova, Ustimenko (2021)** *Uncertainty in Gradient Boosting via Ensembles* — ICLR 2021 — `https://arxiv.org/abs/2006.10562`. Defines total / data / knowledge uncertainty for binary classification under SGLB virtual ensembles. Underlies CatBoost's `virtual_ensembles_predict` API. Backs H-302.
- **Geifman & El-Yaniv (2017)** *Selective Classification for Deep Neural Networks* — NeurIPS 2017 — `https://arxiv.org/abs/1705.08500`. Risk-Coverage curve definition; AURC formulation; selective prediction. Backs H-302 secondary diagnostic.

### [ONLINE] (additions)
- **Montiel et al. (2021)** *River: machine learning for streaming data in Python* — JMLR 22(110) — `https://www.jmlr.org/papers/v22/20-1380.html`. Framework + `predict_proba_one`/`learn_one` contract; `compose.Pipeline`. Backs H-305.
- **Gomes et al. (2017)** *Adaptive Random Forest for evolving data stream classification* — *Machine Learning* 106(9-10):1469-1495 — DOI `10.1007/s10994-017-5642-8` — Springer link `https://link.springer.com/article/10.1007/s10994-017-5642-8`. ARF + ADWIN drift detector; per-tree Poisson(λ) bagging; warning δ_w ≈ 1e-2, drift δ_d ≈ 1e-5. Backs H-204 + H-305.
- **Gomes, Read, Bifet (2019)** *Streaming Random Patches for Evolving Data Stream Classification* — ICDM 2019 LNCS — `https://albertbifet.com/streaming-random-patches/`. SRP = global-subspace random patches + Poisson(λ=6) online bagging. Backs H-305 (recommended on heavy-tailed streams).
- **Bifet & Gavaldà (2009)** *Adaptive Learning from Evolving Data Streams* — IDA 2009 LNCS 5772:249 — DOI `10.1007/978-3-642-03915-7_22`. HAT = Hoeffding tree with per-node ADWIN; alternate-subtree replacement on drift. Backs H-305.

### [STATS-CI] Per-metric confidence intervals & bootstrap schemes (NEW SECTION, round-014)
Backs H-320-a (per-metric bootstrap library) and H-320-b (re-render existing accepted numbers).

- **DeLong, DeLong, Clarke-Pearson (1988)** *Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach* — *Biometrics* 44(3):837-845 — DOI `10.2307/2531595` — `https://pubmed.ncbi.nlm.nih.gov/3203132/`. Closed-form ROC-AUC variance via Mann-Whitney U structural decomposition. **Failure**: O(n_pos·n_neg); ties require mid-rank handling.
- **Sun & Xu (2014)** *Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves* — *IEEE Signal Processing Letters* 21(11):1389-1393 — `https://ieeexplore.ieee.org/document/6851192/`. Mid-rank reformulation gives O((n+m) log(n+m)).
- **Boyd, Eng, Page (2013)** *Area under the Precision-Recall Curve: Point Estimates and Confidence Intervals* — ECML PKDD 2013, LNAI 8190:451-466 — `https://link.springer.com/chapter/10.1007/978-3-642-40994-3_29` (PDF `https://pages.cs.wisc.edu/~boyd/aucpr_final.pdf`). Stratified bootstrap PR-AUC with B≥1000; logit/binomial alternatives recommended over naïve bootstrap. **Failure**: bias-down at small n_pos.
- **Politis & Romano (1994)** *The Stationary Bootstrap* — *JASA* 89(428):1303-1313 — `https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870`. Geometric-block-length bootstrap; valid for stationary α-mixing series. **Failure**: sensitive to mis-specified p (block length).
- **Politis & White (2004)** *Automatic Block-Length Selection for the Dependent Bootstrap* — *Econometric Reviews* 23(1):53-70 — `https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf` (correction Patton-Politis-White 2009 same authors `https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf`). Plug-in optimal block length from flat-top-lag spectral-density estimate; b̂ ∝ N^(1/3).
- **Niculescu-Mizil & Caruana (2005)** *Predicting Good Probabilities With Supervised Learning* — ICML 2005 — `https://dl.acm.org/doi/10.1145/1102351.1102430` (PDF `https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf`). Calibration evaluation framework; per-bin Wilson recommended externally (paper does not propose its own per-bin CI).
- **Roelofs et al. (2022)** *Mitigating Bias in Calibration Error Estimation* — AISTATS 2022 PMLR 151 — `https://proceedings.mlr.press/v151/roelofs22a.html` (arXiv `https://arxiv.org/abs/2012.08668`). Debiased ECE estimator; equal-mass binning; ECE_sweep selection rule. **Failure**: naïve bootstrap on ECE_bin reproduces the bias inside replicates.

### [CALIB-UQ] (clarifications, round-014)
- **Lekeufack, Angelopoulos, Bajcsy, Jordan, Malik (2024)** *Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions* — local PDF `Downloads\conformal_decision_theory.pdf` — also arXiv `https://arxiv.org/abs/2310.05921`. Theorem 1 controller dynamics: λ-update bounded under "eventually safe" data. The σ-as-feature vs σ-as-decision-bound distinction (round-014 H-302 addendum) follows from Lekeufack §3-4.
