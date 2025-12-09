# Dual-Output LSTM: Joint Forecasting of U.S. Electricity Demand and CO₂ Emissions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code and notebooks for research on joint forecasting of national electricity demand and CO₂ emissions using a dual-output LSTM network. The project explores the inherent link between electricity demand, generation mix, and resulting emissions, delivering accurate predictions and actionable insights for renewable integration and grid planning.

---

## 🔍 Overview

Accurate forecasting of electricity demand and CO₂ emissions is crucial for effective grid planning, emissions mitigation, and renewable energy integration. This study develops a **dual-output LSTM network** trained on U.S. hourly EIA-930 data (2018–2023) to jointly predict electricity demand and CO₂ emissions.

### Key Features

- ✅ Joint modeling of U.S. electricity demand and CO₂ emissions
- ✅ Dual-output LSTM architecture with shared temporal representation
- ✅ Comprehensive baseline comparisons (ARIMA, GRU, CNN-LSTM)
- ✅ Statistical validation with significance testing and confidence intervals
- ✅ Renewable integration scenario analysis (solar penetration 10%–50%)
- ✅ Explainable AI (SHAP) analysis with quantitative feature contributions
- ✅ Extreme weather robustness evaluation

---

## 📁 Repository Structure

```
dual-lstm-energy-emissions-forecasting/
│
├── 01.data_cleaning.ipynb          # Data acquisition, preprocessing, and cleaning
├── 02.Feature_engg.ipynb           # Feature engineering and VIF analysis
├── 03.EDA.ipynb                    # Exploratory data analysis
├── 04.results.ipynb                # Model training, evaluation, and analysis
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

### Running the Analysis

```bash
# 1. Data preprocessing and cleaning
jupyter notebook 01.data_cleaning.ipynb

# 2. Feature engineering with VIF analysis
jupyter notebook 02.Feature_engg.ipynb

# 3. Exploratory data analysis
jupyter notebook 03.EDA.ipynb

# 4. Model training, evaluation, and scenario analysis
jupyter notebook 04.results.ipynb
```

---

## 📊 Data

- **Source**: U.S. Energy Information Administration (EIA) EIA-930 Portal
- **Period**: January 2018 – December 2023
- **Resolution**: Hourly observations
- **DOI**: [10.7910/DVN/OKEATQ](https://doi.org/10.7910/DVN/OKEATQ)

---

## 🔬 Methodology Highlights

- **Dual-Output Architecture**: Shared LSTM layers capture coupled dynamics between demand and emissions
- **Feature Engineering**: VIF-corrected feature selection to address multicollinearity
- **Statistical Validation**: Paired significance testing, bootstrap confidence intervals, effect size quantification
- **Scenario Simulation**: Data-driven solar penetration analysis (10%–50%)
- **Explainability**: SHAP analysis for transparent feature attribution

---

## 📫 Contact

- **Ann Mary Thomas**
- ✉️ Email: annmarytttt@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/ann-mary-thomas/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📝 Note

This repository accompanies a research paper currently under review. Detailed results and metrics will be updated upon publication.
