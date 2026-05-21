# Project: AirTS-Forecast

S8 
Students: Tiago Toloczko Ross

## General Description

The project develops and compares statistical and deep learning models for short-term urban air quality prediction with explicit uncertainty quantification. Using ARMA/ARIMA, Exponential Smoothing (ES), and LSTM models, the project forecasts pollutant concentrations (e.g., PM2.5, NO₂) while estimating prediction intervals. The focus is on model comparison, robustness, probabilistic evaluation, and interpretability to support informed environmental decision-making rather than deterministic automation.

## Objectives

### **Current State [WHY]:**


### **Project Content [WHAT FOR]:**

1. **Data Description** 
   1. Pollutants: PM2.5, PM10, NO₂, O₃ 
   2. Exogenous Variables: Temperature, Wind speed/direction, Humidity, Pressure
2. **Exploratory Analysis:** Trend & seasonality decomposition, Autocorrelation (ACF/PACF), Stationarity tests (ADF), Missing data analysis 
3. **Statistical Models**
   1. ARMA / ARIMA: Model identification (ACF/PACF), Order selection (AIC/BIC), Residual diagnostics, Forecast intervals 
   2. Exponential Smoothing (ES / ETS)
      - Trend and seasonality modeling, State-space formulation, Probabilistic forecasting
      - Comparison criteria: MAE, RMSE
      - Coverage probability of intervals
      - Sharpness of intervals
4. **Deep Learning Model**
   1. LSTM Architecture
   Input windowing, Hidden state representation, Multi-step forecasting 
   2. Probabilistic Extension
      - Monte Carlo Dropout, Ensemble LSTM, Quantile regression LSTM
      - Evaluation: Deterministic accuracy, Prediction interval coverage, Calibration curves
5. **Uncertainty Quantification**
   - Types of uncertainty addressed: Aleatoric (inherent variability), Epistemic (model uncertainty)
   - Methods: Analytical intervals (ARIMA/ETS), Bootstrap, Monte Carlo Dropout, Quantile loss
   - Metrics: PICP (Prediction Interval Coverage Probability), PINAW (Prediction Interval Normalized Average Width), CRPS (Continuous Ranked Probability Score)
6. **Experimental Protocol**
   - Train/validation/test split (time-aware)
   - Hyperparameter research by Bayesian Research for LSTM
   - Rolling-origin evaluation
   - Extreme pollution episode analysis
   - Sensitivity to input window size
   - 
## Out of Scope

This project excludes every industrial application, dev activites

### Key Performance Indicators [WHAT]:

* Error measures - RMSE, MAPE, 

### **Budget [HOW MUCH]:**

* No budget was assigned

### **Deadline [WHEN]:**

* From March 1st to June 17th

### **Means [HOW]:**

* Python for coding and data analysis
* Jupyter notebooks for data visualization and sharing
* GEOD'AIR and ERA5-Land databases for pollution data
* 

## Clients and stakeholders

* **The tutor (M. HOUÉ NGOUNA):** main stakeholder, project and deliverable evaluation, objective definition, project execution follow-up
* **The students (M. REINOSO U.; M. T. ROSS):** Time management, task coordination, deliverable preparation and handling, objective accomplition, project execution
* **Pollution and environmental data databases (GEOD'AIR; COPERNICUS' Era5-Land):** Data protection measures
* **Future users:** Reliability of the models, clear usage instructions
* **Toulouse's Mayor's office:** Reliability of the models for decision-making
* **The Environment:** Global warming concerns

## Deliverables

### 1. Literature Review Report
- Statistical vs deep learning forecasting
- Probabilistic methods

### 2. Data Analysis Notebook
- EDA
- Stationarity tests
- ACF/PACF

### 3. Modeling Notebook(s)
- ARIMA implementation
- ETS implementation
- LSTM implementation
- Uncertainty estimation

### 4. Comparative Evaluation Report
- Performance metrics
- Interval calibration
- Robustness analysis

### 5. Final Report (30–40 pages)
- Methodology
- Results
- Discussion
- Environmental interpretation

### 6. Oral Defense (20 minutes)

## Millestones and calender

Pollution data exploration => Training on data science models => Basic statistical models => Environmental data exploration
=> Deel learning models => Model comparison => final presentation 
 


## Risk Management

The students do not have much previous experience with data analysis

The students have to manage well their time so they can conduct the project while studying for semester classes

The databases may be slow