# 🌤️ AirTS Forecast: Urban Air Quality Modeling & Deep Learning
AirTS Forecast is an end-to-end Machine Learning Operations (MLOps) and Time-Series Forecasting pipeline designed to assess and predict urban air pollution. By fusing environmental/meteorological data with historical pollution concentrations (NO2, NOx, O3, PM10, PM25), this project establishes robust, multi-objective predictive models to inform public health and environmental policies.

The pipeline compares classical statistical baselines against advanced sequential Deep Learning architectures, featuring automated hyperparameter optimization, reverse-anchored temporal alignment, and presentation-ready visual reporting.

🚀 Key Achievements & Capabilities
Scale-Dependent Outlier Filtration: Eliminates simulation explosions (RMSE > 10^75) across various pollution emission scales to ensure absolute mathematical stability.

Flawless Sequential Preprocessing: Guarantees zero index desynchronization across thousands of chronological samples using reverse-anchored timeline slicing and Fourier feature mapping.

Deep Learning Optimization: Custom topologies (RNN, LSTM, Bi-LSTM, GRU, and Hybrids) validated through stochastic ensemble training to ensure tight stability margins (± standard deviation tracking).

Automated Hyperparameter DOE: Discovers Pareto-optimal frontiers (minimizing MAPE while maximizing R²) via 4^n Full Factorial Designs of Experiments and Optuna database search loops.

Executive Visualization: Generates globally scale-synchronized, 16:9 presentation-ready matrices and residual error envelopes using Seaborn and Matplotlib.

## 📂 Repository Structure
Plaintext
AirTS-Forecast/
│
├── EnvironmentalDataAnalysis/               # Weather & Environmental Feature Engineering
│   ├── Analysis 02 round/                   # EDA visual outputs and generated profiles
│   ├── environmental_data_etl_extraction.py # 1. API/Data source extraction
│   ├── environmental_data_etl_transform.py  # 2. Cleaning and temporal alignment
│   ├── environmental_data_etl_loading_orchestration.py # 3. Final ETL pipeline execution
│   ├── environmental_data_exp_analysis_controller.py   # Main EDA orchestrator
│   ├── environmental_data_exp_retrieval.py             # Data fetching for EDA
│   ├── environmental_data_exp_timeseries.py            # Feature timelines
│   ├── environmental_data_exp_overall_monthly_avg.py   # Monthly seasonality analysis
│   ├── environmental_data_exp_animated_period_evo.py   # Animated temporal evolution
│   ├── environmental_data_exp_visualization_orchestration_core.py # Visual rendering engine
│   └── recognizing_land.py (beta)           # Spatial/Land-use feature recognition
│
├── PollutionDataAnalysis/                   # Pollution Forecasting & ML Optimization
│   ├── pollution_data_exp_core.py           # Core time-series functions & metrics
│   ├── pollution_data_exp_pollutants_comparison.py # Inter-pollutant correlation
│   ├── pollution_data_exp_report.ipynb      # Interactive EDA report for pollutants
│   ├── Pollution_data_ARIMA_Lucas.py        # Classical Modeling: Original Baseline (Lucas)
│   ├── Pollution_data_ARIMA.py              # Classical Modeling: Extended DOE & Optimization
│   ├── pollution_DL_models_single_variable.py      # Univariate DL Pipeline (Pollution autoregression)
│   ├── pollution_DL_models_multivariate.py         # Multivariate DL Pipeline (Pollution + Weather)
│   ├── Pollution_DL_models_hyperparameter_search.py # Optuna Bayesian Optimization Engine
│   └── pollution_DL_model_comparison.py     # Final Evaluation: SV vs MV vs Classical Models
│
├── Pollution_environmental_data_merging.py  # Central script merging Weather + Pollution datasets
├── Pollution_DL_models_report.ipy           # Executive summary notebook for DL models
├── Project_Charter.md                       # Project scope, objectives, and stakeholder info
└── README.md                                # You are here!
## 🧠 Core Modules
1. Data Engineering (ETL & Merging)
Located in EnvironmentalDataAnalysis and the root directory. This module handles the automated extraction, transformation, and loading of raw meteorological data. Pollution_environmental_data_merging.py serves as the crucial bridge, aligning diverse timestamps and resolving missing values to create the master multivariate dataset.

2. Exploratory Data Analysis (EDA)
Highly modularized visual exploration. Evaluates seasonal trends, monthly averages, and spatial correlations to determine the optimal subset of environmental features to feed into the forecasting models.

3. Classical Time-Series Baselines
Scripts utilizing ARIMA, SARIMA, and Holt-Winters architectures. These models establish the performance floor, utilizing exact mathematical derivation and Response Surface Methodology (RSM) for grid tuning.

4. Deep Learning & Hyperparameter Search
The core predictive engine. Implements univariate (SV) and multivariate (MV) PyTorch models. The Pollution_DL_models_hyperparameter_search.py script utilizes Optuna backed by an SQLite database to dynamically navigate complex hyperparameter spaces (Look Back, Horizon, Layers, Batch Size) with automated pruning and SQLite collision safeguards.

🛠️ Tech Stack & Libraries
Language: Python 3.10+

Deep Learning: PyTorch (torch, torch.nn)

Optimization & DOE: Optuna, Statsmodels

Data Manipulation: Pandas, NumPy, Scikit-Learn

Visualization: Matplotlib, Seaborn, Jupyter

MLOps Tracking: Pickle, JSON configuration pipelines

## 👥 Contributors
Tiago Toloczko Ross - Data Engineering, Deep Learning Architecture, Optuna Integration, and ML-Ops Pipeline.

Lucas - Classical ARIMA baselines and initial statistical foundations.