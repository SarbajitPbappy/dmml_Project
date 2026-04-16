## Slide 1
DMML LAB PROJECT
Climate-Driven Machine Learning for
Dengue Outbreak Prediction
7 March 2026
Presenter
Sarbajit Paul Bappy
ID: 222-15-6155
University
Daffodil International University
Presented To
Md Ibrahim Patwary Khokan
Lecturer

## Slide 2
The Global Dengue Crisis
Understanding the magnitude of the public health challenge
Global Impact
~400M
people infected yearly
Dengue fever represents one of the fastest-growing mosquito-borne viral diseases globally, with major burden concentrated in tropical and subtropical regions where Aedes aegypti mosquitoes thrive.
Why This Matters
Hospitals become overwhelmed during outbreaks
Healthcare costs strain national budgets
Delayed response leads to preventable deaths
Bangladesh 2023 Outbreak
300K+
Total Cases
1600+
Deaths
The 2023 outbreak was the largest in Bangladesh's history, overwhelming healthcare systems and highlighting critical gaps in early warning capabilities. This crisis underscores the urgent need for predictive systems.
The Opportunity
Biological Lag: There exists a 2–4 week window between climate conditions → mosquito growth → human infection.
Can we predict outbreaks before cases appear?

## Slide 3
Research Problem & Objectives
Defining the research scope and expected contributions
Research Question
Can climate data predict dengue outbreaks 4 weeks in advance ?
Sub-Questions:
1.
Which climate factors most strongly drive outbreaks?
2.
Can ML models improve forecasting accuracy?
3.
Can predictions become actionable alerts?
Primary Objective
Develop an early warning system using machine learning that transforms climate data into actionable outbreak predictions, enabling proactive public health interventions.
Secondary Objectives
Create ERI
Epidemic Risk Index combining climate variables
Develop ERS
3-level alert system for outbreak classification
Compare Models
Evaluate multiple ML approaches
Explainability
Validate using SHAP for model transparency

## Slide 4
Dataset & Methodology Overview
Data foundation and system architecture
DengAI Dataset (Kaggle)
18
Years Data
Weekly
Records
2
Cities
Study Locations:
San Juan, Puerto Rico
Tropical Caribbean climate
Iquitos, Peru
Amazon rainforest region
Climate Features
Temperature
Rainfall
Humidity
NDVI (Vegetation)
Target Variable: Weekly dengue case counts
System Pipeline
1
Climate Data Collection
Temperature, rainfall, humidity, NDVI
2
Data Preprocessing
Cleaning, normalization, missing value handling
3
Feature Engineering
Lag features, rolling means, autoregressive terms
4
ML Models & Ensemble
Multiple algorithms combined for prediction
5
Alert System
ERS classification and actionable alerts

## Slide 5
Feature Engineering Innovation
Capturing temporal dynamics and biological lag patterns
Lag Features
Climate conditions from previous weeks influence current mosquito populations and transmission rates.
Implementation:
1-week lag
2-week lag
3-week lag
4-week lag
Biological Insight: Captures mosquito life cycle from egg to adult (~2 weeks) plus extrinsic incubation period
Rolling Features
Smoothed averages over time windows capture sustained climate patterns rather than single-week anomalies.
Implementation:
4-week mean
Short-term
8-week mean
Medium-term
Purpose: Reduces noise and identifies persistent climate trends that favor mosquito breeding
Autoregressive Features
Previous dengue case counts provide crucial information about epidemic momentum and population immunity.
Implementation:
Previous week cases
2-weeks ago cases
Case trend (difference)
Epidemiological Value: Captures disease transmission dynamics and herd immunity effects
Total Engineered Features
~68

## Slide 6
Epidemic Risk Index (ERI)
Original contribution: Synthesizing multi-variable climate data into actionable risk metric
Concept & Innovation
The Epidemic Risk Index (ERI) represents a novel approach to synthesizing multiple climate variables into a single, interpretable risk score. This simplifies complex multi-dimensional analysis into actionable intelligence for public health officials.
Key Innovation
Transforms 4+ climate variables into one metric that correlates strongly with outbreak probability, enabling rapid decision-making without requiring domain expertise in climate science.
ERI Formula
ERI =
0.35
× Temperature
0.30
× Humidity
0.25
× Rainfall
0.10
× Vegetation (NDVI)
Weights derived from: Domain knowledge and correlation analysis with historical outbreaks
Risk Scale
0.0 - 0.3
LOW
Minimal outbreak risk. Standard surveillance sufficient.
0.3 - 0.7
MODERATE
Elevated risk. Enhanced monitoring recommended.
0.7 - 1.0
HIGH
Critical risk. Immediate intervention required.
Advantages
Simplified interpretation for non-technical stakeholders
Standardized metric across different regions
Early warning capability with 4-week lead time
Actionable thresholds for policy decisions

## Slide 7
Machine Learning Models & Evaluation
Comprehensive model comparison and rigorous evaluation framework
Models Compared
SARIMA
Statistical
Seasonal ARIMA baseline for time-series
Random Forest
ML
Ensemble of decision trees
XGBoost
Gradient Boosting
Extreme Gradient Boosting
LightGBM
Boosting
Light Gradient Boosting Machine
LSTM
Deep Learning
Long Short-Term Memory networks
Ensemble Strategy
Combine predictions from multiple models using weighted averaging to leverage strengths of different approaches and improve robustness.
Evaluation Metrics
MAE
Mean Absolute Error
Average magnitude of prediction errors
RMSE
Root Mean Square Error
Penalizes large errors more heavily
R²
Coefficient of Determination
Proportion of variance explained
Note: MAPE avoided due to instability with zero-case periods
Validation Strategy
Time-series cross-validation: Respects temporal order
Prevents data leakage: No future information in training
Realistic evaluation: Simulates real-world forecasting

## Slide 8
Model Performance Results
Comparative analysis reveals Random Forest as top performer
Performance Comparison (R² Score)
Best Performer
Random Forest
0.886
R²
Explains 88.6% of variance in dengue cases
Key Insights
Tree-based models excel: Random Forest, LightGBM, XGBoost all achieve R² > 0.82
Deep learning underperforms: LSTM (0.489) suggests insufficient data for complex patterns
Statistical baseline fails: SARIMA (-0.03) cannot capture non-linear climate-disease relationships
Feature engineering critical: Lag and rolling features capture biological dynamics

## Slide 9
Explainability with SHAP Analysis
Understanding model decisions through SHAP (SHapley Additive exPlanations)
Two-Layer Prediction Logic
SHAP analysis reveals the model operates through two distinct prediction layers, each capturing different aspects of outbreak dynamics.
LAYER 1
Epidemic Momentum
Key Feature: Previous week case counts
Captures disease transmission dynamics, population immunity status, and ongoing outbreak trajectory. High SHAP values indicate strong momentum effects.
LAYER 2
Climate Drivers
Key Features: Temperature lag, rainfall, humidity
Identifies environmental conditions that enable mosquito breeding and virus transmission. Reveals optimal climate windows for outbreaks.
Critical Discovery
3-Week Biological Lag
The model automatically discovered the 3-week lag between climate conditions and human cases, validating established domain knowledge.
Timeline: Week 0 (climate) → Week 2 (mosquito emergence) → Week 3 (human infection)
Why This Matters:
Validates model: Learned patterns align with entomological science
Enables prediction: 4-week forecast window confirmed
Builds trust: Explainable predictions for stakeholders
Top SHAP Features
1. Previous week cases
High Impact
2. Temperature lag (3wk)
High Impact
3. Rainfall (2-4wk)
Medium Impact
4. Humidity (3wk)
Medium Impact

## Slide 10
Early Warning Alert System
Transforming predictions into actionable public health intelligence
Epidemic Risk Score (ERS)
The ERS translates model predictions into a 3-level alert system, enabling rapid public health response before outbreaks escalate.
Normal
< 39 cases
Standard surveillance. No additional measures required.
Elevated
39-71 cases
Enhanced monitoring. Prepare resources and staff.
High Alert
> 71 cases
Immediate intervention. Activate emergency protocols.
4-Week Forecast Demo
Average Error:~1 case per week
Government Response
4-week lead time enables proactive preparation
Resource allocation: Deploy staff, supplies, beds
Public messaging: Issue prevention guidelines
Vector control: Targeted spraying campaigns

## Slide 11
Deployment, Impact & Limitations
Practical implementation pathway and research constraints
System Requirements
Data Sources
Weather station data (temperature, rainfall, humidity)
Satellite NDVI (vegetation index)
Historical dengue case records
Technical Infrastructure
Standard laptop/computer
Python-based pipeline
Cloud deployment ready
Cost: ~$20/month for cloud deployment
Deployment Targets
Bangladesh Meteorological Dept
Integrate with existing weather monitoring infrastructure
Health Ministry
Direct integration with disease surveillance systems
SMS Alert System
Automated notifications to hospitals and clinics
Limitations
Geographic gap: Dataset from Puerto Rico/Peru, needs Bangladesh validation
Data constraints: Limited samples for deep learning optimization
Social factors: Population density, sanitation not included
Climate change: Model may need retraining for shifting patterns

## Slide 12
Conclusion & Future Directions
Key contributions and path forward
Key Contributions
ERI Climate Index
Novel composite risk score combining multiple climate variables
ERS Alert System
3-level classification enabling proactive response
Multi-Model Ensemble
Random Forest achieves 0.886 R², outperforming baselines
SHAP Explainability
Revealed 3-week biological lag, validating domain knowledge
Future Work
Bangladesh validation: Collect local data and retrain models
Social factors: Incorporate population density, sanitation, mobility
Deep learning: Explore with larger datasets for complex patterns
Real-time deployment: Live system integration with health ministries
Final Message
This system demonstrates that machine learning + climate data can successfully predict dengue outbreaks 4 weeks in advance, transforming reactive healthcare into proactive prevention.
Contact & Resources
Sarbajit Paul Bappy
Daffodil International University
Email: sarbajit.paul@example.com
GitHub: github.com/sarbajit/dengue-prediction
Questions & Discussion Welcome!
