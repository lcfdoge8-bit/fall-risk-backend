# ml/experiments — Model Experiments & Validation (KS 嘅 Phase-1 + Phase-2 工作)

呢個資料夾放 **KS（Kero）** 寫嘅模型實驗 / 驗證 scripts。佢哋係「搵出 FINAL MODEL」過程嘅完整證據鏈：
Phase-1 基礎 pipeline → fair comparison → combo 實驗 → 5-fold stability（overfit 證明）→ SHAP 圖 → 最終模型輸出。

> 註：呢啲 scripts 原本由 KS 直接 upload 咗去 repo 根目錄（GitHub "Add files via upload"，commit
> `85f7a25` / `c0dd472` / `af389f8`），為咗保持專案結構整潔，已移入 `ml/experiments/`（用 `git mv`，commit history 冇斷）。
> 唯一改動：
> 1. data path 改為 `__file__` 相對路徑 `../../data/fall_risk_patients_2000_v2.csv`，任何位置 run 都得；
> 2. **Phase-1 嗰 8 個 scripts 原本讀 `fall_risk_patients_2000.csv`（舊版 CSV，唔喺 repo 度）**，已改為指向
>    `data/fall_risk_patients_2000_v2.csv` —— 想完整重現 Phase-1 原始數字，要叫 KS 提供返舊 CSV；
> 3. 修咗 `shap_analysis.py` 一個 typo（`figsi ze` → `figsize`，原碼第 33 行）。

## File 一覽

### Phase 2（FINAL MODEL 證據鏈）

| File | 用途（KS 原描述） | 輸出 |
|------|-------------------|------|
| `final_model.py` | **FINAL MODEL** — Alex 14 features + Kero 驗證（70/20/10 + 5-fold + threshold tuning + overfit check），LR `class_weight='balanced'` → **acc ~91.5%, HIGH recall ~100%** | `saved/`（俾 Lai 部署用） |
| `phase2_model.py` | **17 features 版** — Kero engineered (4) + sex + 3 clinical background；threshold tuning + overfit check（P2 改進版） | console metrics |
| `fair_compare.py` | **公平比較 Kero (17 features) vs Alex (14 features)** — same split / same test set / same metrics / same protocol，threshold 兩邊都係 0.5（冇 unfair advantage） | console metrics |
| `combo_model.py` | **18 features combo** — Kero engineered + Alex 保留 `polypharmacy_count` + Alex `days_since_last_fall=-1` 處理 | console metrics |
| `shap_overfit_test.py` | **SHAP 5-fold 穩定性** — feature ranking 每個 fold 一致 → 證明模型冇 overfit；另附 feature balance 圖 | `phase2_shap_overfit_report.txt` + PNG figures |
| `shap_stacked_bar.py` | **Multi-class SHAP 圖** — mean(\|SHAP\|) per feature，按 HIGH / MEDIUM / LOW 拆開 stacked bar | PNG figure |

### Phase 1（基礎 pipeline，KS 上傳 2026-08-27）

| File | 用途 | 備註 |
|------|------|------|
| `real_data_pipeline_kfold.py` | 最早期 pipeline — 10 features + LR/RF/XGB | 讀 v2 CSV |
| `real_data_pipeline_k_fold.py` | 5-fold CV 改進版（+4 interaction features, -polypharmacy, +StandardScaler） | 讀 v2 CSV |
| `real_data_pipeline_Alex.py` | Alex 版 pipeline（原本 hardcode 咗 Mac 路徑） | 已改 `__file__` 路徑；drop 欄位改為 conditional |
| `class_balance.py` | Class balance 圖 — 解釋點解 `class_weight='balanced'` 重要 | PNG figure |
| `risk_distribution.py` | 2000 病人 LOW/MEDIUM/HIGH 3 色 dot 圖 | PNG figure |
| `shap_analysis.py` | SHAP summary/bar 圖（10 features，multiple plots） | 4 張 PNG（`shap_*.png`） |
| `pprof_shap.py` | Professional SHAP 圖（publication/presentation 用） | PNG figures |
| `dashboard.py` | **Streamlit dashboard**（Phase-1 互動介面） | `streamlit run` |

## 點樣 Run（喺 repo root）

```bash
# Phase-2 scripts（只需要 pandas / numpy / scikit-learn / joblib）
python ml/experiments/final_model.py
python ml/experiments/phase2_model.py
python ml/experiments/fair_compare.py
python ml/experiments/combo_model.py

# SHAP scripts（需要 SHAP venv：pip install shap matplotlib）
python ml/experiments/shap_overfit_test.py
python ml/experiments/shap_stacked_bar.py
python ml/experiments/shap_analysis.py
python ml/experiments/pprof_shap.py

# 圖表 scripts（matplotlib）
python ml/experiments/class_balance.py
python ml/experiments/risk_distribution.py

# Streamlit dashboard（需要 streamlit + plotly）
python -m streamlit run ml/experiments/dashboard.py
```

## 部署輸出（Lai 用）

`final_model.py` run 完之後會喺 **repo root** 產生 `saved/`（而家嘅版本係 **KS 原版** upload 嘅 artifacts）：

| File | 內容 |
|------|------|
| `saved/final_model.pkl` | 訓練好嘅 Logistic Regression（14 features, balanced；KS 原版，sklearn 1.7.2 pickle） |
| `saved/label_encoder.pkl` | LabelEncoder（LOW/MEDIUM/HIGH 編碼） |
| `saved/features.json` | 14 個 feature 名稱（順序必須一致） |
| `saved/threshold.json` | `{"threshold", "high_idx", "classes"}` — 部署時用 threshold 決定 HIGH |

> 呢批 artifacts 係俾 Lai 接落 API 部署用（對應 `deploy/` + `backend/` 嘅 predict 流程）。
> 小提示：KS 原版 pkl 係用 sklearn 1.7.2 訓練，如果用新版 sklearn load 會出 version warning（照用得，預測正常）。
> 想用返本機版本就 run 一次 `final_model.py` 重新生成。
