# COVID-19 Data Analysis and Prediction

This project analyzes COVID-19 data and performs predictive modeling using various epidemiological variables.

## Data Source

The analysis uses data from `covid.csv`, which contains global COVID-19 statistics from Our World in Data.

## Variables Used

The notebook `covidpredictTotal.ipynb` has been updated to use the following key variables for analysis and prediction:

- **reproduction_rate**: The estimated reproduction number (R) of the virus
- **new_cases_smoothed_per_million**: 7-day smoothed new cases per million people
- **people_fully_vaccinated_per_hundred**: Percentage of population fully vaccinated
- **weekly_hosp_admissions_per_million**: Weekly hospital admissions per million people

## Analysis Components

### 1. Time Series Analysis
- Plots new cases and deaths for Indonesia and United Kingdom
- Examines total deaths per million over time

### 2. Cross-Sectional Country Analysis
- Selects data for a specific date (2022-05-01)
- Performs clustering analysis using the four key variables
- Creates 3D scatter plots showing relationships between variables
- Sorts countries by weekly hospital admissions per million

### 3. Predictive Modeling
The notebook includes three linear regression models:

1. **Simple Linear Regression**: Predicts weekly hospital admissions using new cases smoothed per million
2. **Simple Linear Regression**: Predicts weekly hospital admissions using people fully vaccinated per hundred
3. **Multiple Linear Regression**: Predicts weekly hospital admissions using reproduction rate, new cases smoothed per million, and people fully vaccinated per hundred

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Ensure `covid.csv` is in the same directory as the notebook
2. Open `covidpredictTotal.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells to perform the complete analysis

## Files

- `covidpredictTotal.ipynb`: Main analysis notebook
- `covid.csv`: COVID-19 dataset
- `penjelasan_grafik.txt`: Additional explanations (in Indonesian)
- `README.md`: This file

## Results

The analysis provides insights into COVID-19 patterns and predictive relationships between epidemiological variables across different countries.