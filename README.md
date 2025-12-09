# Dual-Output LSTM: Joint Forecasting of U.S. Electricity Demand and CO₂ Emissions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

This repository contains the code, data processing pipelines, and analysis notebooks for the research paper:

> **"A dual output deep learning framework for enhanced electricity demand and emissions forecasting for renewable integration"**

The project develops a dual-output LSTM network that jointly forecasts national electricity demand and CO₂ emissions, exploiting their inherent coupling through generation dispatch to deliver high-accuracy predictions and actionable insights for renewable integration planning.

---

## 🔍 Abstract

Accurate forecasting of electricity demand and CO₂ emissions is vital for grid planning and renewable integration. However, existing research often treats these as separate problems, overlooking their intrinsic linkage through generation mix. This study proposes a **dual-output LSTM network** that simultaneously predicts both outputs using six years (2018–2023) of US EIA-930 hourly data with VIF-corrected feature engineering (28→24 features).

### Key Results

| Metric | Demand | CO₂ Emissions |
|--------|--------|---------------|
| **MAPE** | 0.48% | 1.60% |
| **MAE** | 2,113 MWh | 2,475 metric tons |
| **RMSE** | 2,716 MWh | 2,872 metric tons |

### Baseline Comparison

| Model | Demand MAPE | Emissions MAPE | Improvement |
|-------|-------------|----------------|-------------|
| ARIMA | 9.14% | 15.43% | **19× better** |
| GRU | 2.65% | 3.95% | **5.5× better** |
| CNN-LSTM | 5.53% | 5.95% | **11.5× better** |
| **Proposed LSTM** | **0.48%** | **1.60%** | — |

All improvements are statistically significant (p < 0.0001, Cohen's d = -1.40 to -4.40).

### Scenario Analysis

Solar penetration simulations (10%–50%) reveal:
- **6.74% CO₂ reduction** at 50% solar penetration
- Equivalent to **~584,000 metric tons CO₂ annually**
- Comparable to removing **~127,000 passenger vehicles** from roads

### Explainability (SHAP Analysis)

Quantitative feature contributions:
- **CO₂ Forecasting**: Coal generation (68.38%), Natural gas (23.99%), Net demand (3.81%)
- **Demand Forecasting**: Lagged demand features (98.36%)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Features (24)                   │
│  Temporal + Lag + Rolling Stats + Generation Mix + ...   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 LSTM Layer 1 (64 units)                  │
│                    + Dropout (0.2)                       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 LSTM Layer 2 (64 units)                  │
│                    + Dropout (0.2)                       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Dense Layer (2 outputs)                     │
│         [Electricity Demand, CO₂ Emissions]              │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
dual-lstm-energy-emissions-forecasting/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Data acquisition and preprocessing
│   ├── 02_feature_engineering.ipynb # VIF analysis and feature selection
│   ├── 03_EDA.ipynb                # Exploratory data analysis
│   ├── 04_model_training.ipynb     # LSTM training with validation
│   ├── 05_baseline_comparison.ipynb # ARIMA, GRU, CNN-LSTM benchmarks
│   ├── 06_scenario_analysis.ipynb  # Solar penetration simulations
│   └── 07_shap_analysis.ipynb      # Explainability analysis
│
├── src/
│   ├── data_preprocessing.py       # Data loading and cleaning functions
│   ├── feature_engineering.py      # Feature creation and VIF analysis
│   ├── models.py                   # LSTM and baseline model definitions
│   ├── evaluation.py               # Metrics and statistical tests
│   └── visualization.py            # Plotting functions
│
├── data/
│   └── README.md                   # Data source information
│
├── results/
│   ├── figures/                    # Generated plots
│   └── tables/                     # Performance metrics
│
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/Ann-Mary-Thomas/dual-lstm-energy-emissions-forecasting.git
cd dual-lstm-energy-emissions-forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.4.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
shap>=0.41.0
statsmodels>=0.13.0
scipy>=1.8.0
```

### Running the Analysis

```bash
# 1. Data preprocessing
jupyter notebook notebooks/01_data_cleaning.ipynb

# 2. Feature engineering with VIF analysis
jupyter notebook notebooks/02_feature_engineering.ipynb

# 3. Exploratory data analysis
jupyter notebook notebooks/03_EDA.ipynb

# 4. Model training
jupyter notebook notebooks/04_model_training.ipynb

# 5. Baseline comparisons
jupyter notebook notebooks/05_baseline_comparison.ipynb

# 6. Solar penetration scenarios
jupyter notebook notebooks/06_scenario_analysis.ipynb

# 7. SHAP explainability
jupyter notebook notebooks/07_shap_analysis.ipynb
```

---

## 📊 Data

### Source
- **Dataset**: U.S. Energy Information Administration (EIA) EIA-930 Portal
- **Period**: January 1, 2018 – December 31, 2023
- **Resolution**: Hourly (43,785 complete records)
- **DOI**: [10.7910/DVN/OKEATQ](https://doi.org/10.7910/DVN/OKEATQ)

### Features (24 after VIF correction)

| Category | Features |
|----------|----------|
| **Temporal** | Hour, Day of Week, Month |
| **Lag Variables** | Demand (t-1, t-24, t-168), CO₂ (t-1, t-24) |
| **Rolling Statistics** | Mean/Std (3h, 24h windows) |
| **Generation Mix** | Coal, Gas, Nuclear, Hydro, Solar, Wind |
| **Derived Ratios** | Solar Share, Fossil Share, CO₂ Intensity |

### Data Split

| Split | Samples | Period | Purpose |
|-------|---------|--------|---------|
| Training | 30,650 (70%) | Jan 2018 – May 2022 | Model learning |
| Validation | 4,379 (10%) | Jun 2022 – Dec 2022 | Early stopping |
| Test | 8,756 (20%) | Jul 2022 – Jun 2023 | Final evaluation |

---

## 📈 Results

### Model Performance

<details>
<summary><b>Click to expand detailed metrics</b></summary>

#### Training Performance
| Metric | Demand | CO₂ Emissions |
|--------|--------|---------------|
| MAPE | 0.31% | 0.39% |
| MAE | 1,345 MWh | 618 metric tons |
| RMSE | 1,760 MWh | 819 metric tons |

#### Validation Performance
| Metric | Demand | CO₂ Emissions |
|--------|--------|---------------|
| MAPE | 0.65% | 2.27% |
| MAE | 2,872 MWh | 3,550 metric tons |
| RMSE | 3,387 MWh | 3,830 metric tons |

#### Test Performance
| Metric | Demand | CO₂ Emissions |
|--------|--------|---------------|
| MAPE | 0.48% | 1.60% |
| MAE | 2,113 MWh | 2,475 metric tons |
| RMSE | 2,716 MWh | 2,872 metric tons |

</details>

### Statistical Validation

| Comparison | Target | t-statistic | p-value | Cohen's d |
|------------|--------|-------------|---------|-----------|
| LSTM vs ARIMA | Demand | -92.77 | <0.0001 | -1.40 |
| LSTM vs ARIMA | Emissions | -93.81 | <0.0001 | -1.41 |
| LSTM vs GRU | Demand | -318.45 | <0.0001 | -3.63 |
| LSTM vs GRU | Emissions | -217.06 | <0.0001 | -2.14 |
| LSTM vs CNN-LSTM | Demand | -313.75 | <0.0001 | -4.40 |
| LSTM vs CNN-LSTM | Emissions | -158.80 | <0.0001 | -2.28 |

Bootstrap 95% Confidence Intervals (1,000 iterations):
- Demand MAPE: [0.47%, 0.49%]
- Emissions MAPE: [1.58%, 1.62%]

### Extreme Weather Robustness

| Condition | Samples | Demand MAPE | Emissions MAPE |
|-----------|---------|-------------|----------------|
| Extreme High (≥95th percentile) | 438 | 0.41% | 1.63% |
| Normal (5th–95th percentile) | 7,875 | 0.48% | 1.60% |
| Extreme Low (≤5th percentile) | 438 | 0.67% | 2.66% |

### Solar Penetration Scenarios

| Solar Increase | Predicted CO₂ (tons) | Reduction | MAPE |
|----------------|---------------------|-----------|------|
| +10% | 164,433 | 1.12% | 0.70% |
| +20% | 162,595 | 2.23% | 0.27% |
| +30% | 159,954 | 3.81% | 0.43% |
| +40% | 157,100 | 5.53% | 0.70% |
| +50% | 155,088 | **6.74%** | 0.45% |

---

## 🔬 Methodology

### Key Innovations

1. **Dual-Output Architecture**: Shared temporal representation exploits demand-emission coupling through generation dispatch

2. **Comprehensive Statistical Validation**:
   - VIF analysis for feature selection (28→24 features)
   - Paired t-tests with Bonferroni correction
   - Bootstrap confidence intervals (1,000 iterations)
   - Cohen's d effect size quantification

3. **Data-Driven Scenario Simulation**: Separate model training per penetration level captures realistic dispatch dynamics

4. **Quantitative SHAP Analysis**: Percentage contributions for both outputs with Random Forest surrogate model

### Training Configuration

```python
# Model architecture
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(2)  # Dual output: [demand, emissions]
])

# Training parameters
optimizer = Adam(learning_rate=0.001)
loss = 'mse'
epochs = 100
batch_size = 32
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
```

---

## 📚 Citation

If you use this code or methodology, please cite:

```bibtex
@article{thomas2025dual,
  title={A dual output deep learning framework for enhanced electricity demand and emissions forecasting for renewable integration},
  author={Thomas, Ann Mary and Dey, Maitreyee and Rana, Soumya Prakash and Debnath, Ramit},
  journal={[Journal Name]},
  year={2025},
  note={Under Review}
}
```

---

## 👥 Authors

- **Ann Mary Thomas** - GENESIS Research Lab, London Metropolitan University
- **Dr. Maitreyee Dey** - GENESIS Research Lab, London Metropolitan University
- **Dr. Soumya Prakash Rana** - School of Engineering, University of Greenwich
- **Dr. Ramit Debnath** - University of Cambridge

---

## 📫 Contact

- ✉️ Email: annmarytttt@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/ann-mary-thomas/)
- 🌐 [Portfolio Website](https://ann-mary-thomas.github.io/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- U.S. Energy Information Administration for the EIA-930 dataset
- GENESIS Research Lab at London Metropolitan University
- Reviewers for constructive feedback that improved this work

---

## 📝 Changelog

### v2.0 (Current)
- ✅ Improved model achieving 0.48%/1.60% MAPE (from 1.24%/3.21%)
- ✅ Added comprehensive baseline comparisons (ARIMA, GRU, CNN-LSTM)
- ✅ Statistical validation with paired t-tests and bootstrap CIs
- ✅ VIF analysis reducing features from 28 to 24
- ✅ Monte Carlo uncertainty quantification for emission coefficients
- ✅ Extreme weather robustness evaluation
- ✅ Quantitative SHAP contribution percentages
- ✅ Extended scenario analysis to 50% solar penetration
- ✅ Environmental impact quantification (127,000 vehicles equivalent)

### v1.0 (Initial)
- Initial dual-output LSTM implementation
- Basic scenario analysis (10%-50% solar)
- Preliminary SHAP analysis
