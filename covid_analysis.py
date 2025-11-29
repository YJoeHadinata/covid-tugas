import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#reading data from file
df = pd.read_csv('covid.csv')

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','new_cases']]
print(df1.tail())

df2 = df.loc[df['location'] == 'United Kingdom']
df2=df2[['location','date','new_cases']]

#Reseting data index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

#Trimming data to the same length
df2=df2.iloc[len(df2) - len(df1):]
df2.reset_index(drop=True, inplace=True)

#Setting the last XX days data, 720 = 24 months
df1=df1.iloc[len(df1)-720:]
df2=df2.iloc[len(df2)-720:]

plt.plot(df1['new_cases'], label='Indonesia')
plt.plot(df2['new_cases'],'r', label='United Kingdom')
plt.title('Perbandingan Kasus Baru Harian: Indonesia vs United Kingdom')
plt.xlabel('Hari')
plt.ylabel('Jumlah Kasus Baru')
plt.legend()
plt.grid()
plt.savefig('new_cases_comparison.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','reproduction_rate', 'new_cases_smoothed_per_million','people_fully_vaccinated_per_hundred', 'weekly_hosp_admissions_per_million']]

df2 = df.loc[df['location'] == 'United Kingdom']
df2=df2[['location','date','reproduction_rate', 'new_cases_smoothed_per_million','people_fully_vaccinated_per_hundred', 'weekly_hosp_admissions_per_million']]

#Reseting data index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

#Trimming data to the same length
df2=df2.iloc[len(df2) - len(df1):]
df2.reset_index(drop=True, inplace=True)

#Setting the last XX days data, 720 = 24 months; 300 = 10 months
xdays=720
df1=df1.iloc[len(df1)-xdays:]
df2=df2.iloc[len(df2)-xdays:]

#For colormap to show time elapsing
numbers=range(len(df1))

df1.plot.scatter(x='reproduction_rate',y='new_cases_smoothed_per_million',s=50, c=numbers, colormap='plasma')
plt.title('Indonesia: Reproduction Rate vs New Cases per Million')
plt.xlabel('Reproduction Rate')
plt.ylabel('New Cases per Million')
plt.grid()
plt.savefig('indonesia_reproduction_vs_cases_per_million.png')
plt.show()

df2.plot.scatter(x='reproduction_rate',y='new_cases_smoothed_per_million',s=50, c=numbers, colormap='plasma')
plt.title('United Kingdom: Reproduction Rate vs New Cases per Million')
plt.xlabel('Reproduction Rate')
plt.ylabel('New Cases per Million')
plt.grid()
plt.savefig('uk_reproduction_vs_cases_per_million.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','reproduction_rate', 'people_fully_vaccinated_per_hundred']]

df2 = df.loc[df['location'] == 'United Kingdom']
df2=df2[['location','date','reproduction_rate', 'people_fully_vaccinated_per_hundred']]

#Reseting data index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

#Trimming data to the same length
df2=df2.iloc[len(df2) - len(df1):]
df2.reset_index(drop=True, inplace=True)

#Setting the last XX days data, 720 = 24 months
xdays=720
df1=df1.iloc[len(df1)-xdays:]
df2=df2.iloc[len(df2)-xdays:]

#For colormap to show time elapsing
numbers=range(len(df1))

df1.plot.scatter(x='reproduction_rate',y='people_fully_vaccinated_per_hundred',s=50, c=numbers, colormap='plasma')
plt.title('Indonesia: Reproduction Rate vs Vaccination Rate (%)')
plt.xlabel('Reproduction Rate')
plt.ylabel('People Fully Vaccinated (%)')
plt.grid()
plt.savefig('indonesia_reproduction_vs_vaccination.png')
plt.show()

df2.plot.scatter(x='reproduction_rate',y='people_fully_vaccinated_per_hundred',s=50, c=numbers, colormap='plasma')
plt.title('United Kingdom: Reproduction Rate vs Vaccination Rate (%)')
plt.xlabel('Reproduction Rate')
plt.ylabel('People Fully Vaccinated (%)')
plt.grid()
plt.savefig('uk_reproduction_vs_vaccination.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','new_cases_smoothed_per_million', 'people_fully_vaccinated_per_hundred']]

df2 = df.loc[df['location'] == 'United Kingdom']
df2=df2[['location','date','new_cases_smoothed_per_million', 'people_fully_vaccinated_per_hundred']]

#Reseting data index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

#Trimming data to the same length
df2=df2.iloc[len(df2) - len(df1):]
df2.reset_index(drop=True, inplace=True)

#Setting the last XX days data, 720 = 24 months
xdays=720
df1=df1.iloc[len(df1)-xdays:]
df2=df2.iloc[len(df2)-xdays:]

#For colormap to show time elapsing
numbers=range(len(df1))

df1.plot.scatter(x='new_cases_smoothed_per_million',y='people_fully_vaccinated_per_hundred',s=50, c=numbers, colormap='plasma')
plt.title('Indonesia: New Cases per Million vs Vaccination Rate (%)')
plt.xlabel('New Cases per Million')
plt.ylabel('People Fully Vaccinated (%)')
plt.grid()
plt.savefig('indonesia_cases_per_million_vs_vaccination.png')
plt.show()

df2.plot.scatter(x='new_cases_smoothed_per_million',y='people_fully_vaccinated_per_hundred',s=50, c=numbers, colormap='plasma')
plt.title('United Kingdom: New Cases per Million vs Vaccination Rate (%)')
plt.xlabel('New Cases per Million')
plt.ylabel('People Fully Vaccinated (%)')
plt.grid()
plt.savefig('uk_cases_per_million_vs_vaccination.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','reproduction_rate','weekly_hosp_admissions_per_million']]

df2 = df.loc[df['location'] == 'United Kingdom']
df2=df2[['location','date','reproduction_rate','weekly_hosp_admissions_per_million']]

df1=df1.dropna()
df2=df2.dropna()

#Reseting data index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

#For colormap to show time elapsing
numbers1=range(len(df1))
numbers2=range(len(df2))

df1.plot.scatter(x='reproduction_rate',y='weekly_hosp_admissions_per_million',s=50, c=numbers1, colormap='plasma')
plt.title('Indonesia: Reproduction Rate vs Weekly Hospital Admissions per Million')
plt.xlabel('Reproduction Rate')
plt.ylabel('Weekly Hosp Admissions per Million')
plt.grid()
plt.savefig('indonesia_reproduction_vs_hospitalizations.png')
plt.show()

df2.plot.scatter(x='reproduction_rate',y='weekly_hosp_admissions_per_million',s=50, c=numbers2, colormap='plasma')
plt.title('United Kingdom: Reproduction Rate vs Weekly Hospital Admissions per Million')
plt.xlabel('Reproduction Rate')
plt.ylabel('Weekly Hosp Admissions per Million')
plt.grid()
plt.savefig('uk_reproduction_vs_hospitalizations.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','new_cases_smoothed_per_million','weekly_hosp_admissions_per_million']]

df2 = df.loc[df['location'] == 'United Kingdom']
df2=df2[['location','date','new_cases_smoothed_per_million','weekly_hosp_admissions_per_million']]

df1=df1.dropna()
df2=df2.dropna()

#Reseting data index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

#For colormap to show time elapsing
numbers1=range(len(df1))
numbers2=range(len(df2))

df1.plot.scatter(x='new_cases_smoothed_per_million',y='weekly_hosp_admissions_per_million',s=50, c=numbers1, colormap='plasma')
plt.title('Indonesia: New Cases per Million vs Weekly Hospital Admissions per Million')
plt.xlabel('New Cases per Million')
plt.ylabel('Weekly Hosp Admissions per Million')
plt.grid()
plt.savefig('indonesia_cases_per_million_vs_hospitalizations.png')
plt.show()

df2.plot.scatter(x='new_cases_smoothed_per_million',y='weekly_hosp_admissions_per_million',s=50, c=numbers2, colormap='plasma')
plt.title('United Kingdom: New Cases per Million vs Weekly Hospital Admissions per Million')
plt.xlabel('New Cases per Million')
plt.ylabel('Weekly Hosp Admissions per Million')
plt.grid()
plt.savefig('uk_cases_per_million_vs_hospitalizations.png')
plt.show()

#Selecting certain country in data columns
df_vac1 = df.loc[df['location'] == 'Indonesia']
df_vac1=df_vac1[['location','date','people_fully_vaccinated_per_hundred','weekly_hosp_admissions_per_million']]

df_vac2 = df.loc[df['location'] == 'United Kingdom']
df_vac2=df_vac2[['location','date','people_fully_vaccinated_per_hundred','weekly_hosp_admissions_per_million']]

df_vac1=df_vac1.dropna()
df_vac2=df_vac2.dropna()

#Reseting data index
df_vac1.reset_index(drop=True, inplace=True)
df_vac2.reset_index(drop=True, inplace=True)

#For colormap to show time elapsing
numbers1=range(len(df_vac1))
numbers2=range(len(df_vac2))

df_vac1.plot.scatter(x='people_fully_vaccinated_per_hundred',y='weekly_hosp_admissions_per_million',s=50, c=numbers1, colormap='plasma')
plt.title('Indonesia: Vaccination Rate (%) vs Weekly Hospital Admissions per Million')
plt.xlabel('People Fully Vaccinated (%)')
plt.ylabel('Weekly Hosp Admissions per Million')
plt.grid()
plt.savefig('indonesia_vaccination_vs_hospitalizations.png')
plt.show()

df_vac2.plot.scatter(x='people_fully_vaccinated_per_hundred',y='weekly_hosp_admissions_per_million',s=50, c=numbers2, colormap='plasma')
plt.title('United Kingdom: Vaccination Rate (%) vs Weekly Hospital Admissions per Million')
plt.xlabel('People Fully Vaccinated (%)')
plt.ylabel('Weekly Hosp Admissions per Million')
plt.grid()
plt.savefig('uk_vaccination_vs_hospitalizations.png')
plt.show()

#Selecting certain parameters for prediction
df1 = df.loc[df['location'] == 'Indonesia']
df3=df1[['new_cases_smoothed_per_million','people_fully_vaccinated_per_hundred']]
print(df3.shape)
print(df3.head())

df3=df3.dropna()
df3.reset_index(drop=True, inplace=True)
print(df3.shape)
print(df3.head())

x=df3[['people_fully_vaccinated_per_hundred']].values
y=df3[['new_cases_smoothed_per_million']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=False)

reg = LinearRegression().fit(x_train, y_train)
print(reg.score(x_train, y_train))
print(reg.coef_)
print(reg.intercept_)

y_pred=reg.predict(x_test)

plt.plot(y_test, label='Actual')
plt.plot(y_pred,'r', label='Predicted')
plt.title('Linear Regression: Cases per Million Prediction from Vaccination')
plt.xlabel('Test Samples')
plt.ylabel('Cases per Million')
plt.legend()
plt.grid()
plt.savefig('cases_per_million_prediction_linear.png')
plt.show()

print(len(y_pred))

print(r2_score(y_test, y_pred))

#Selecting certain parameters for prediction
df3=df1[['new_cases_smoothed_per_million','reproduction_rate','people_fully_vaccinated_per_hundred']]
print(df3.shape)
print(df3.head())

df3=df3.dropna()
df3.reset_index(drop=True, inplace=True)
print(df3.shape)
print(df3.head())

x=df3[['reproduction_rate','people_fully_vaccinated_per_hundred']].values
y=df3[['new_cases_smoothed_per_million']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=False)

reg = LinearRegression().fit(x_train, y_train)
print(reg.score(x_train, y_train))
print(reg.coef_)
print(reg.intercept_)

y_pred=reg.predict(x_test)

plt.plot(y_test, label='Actual')
plt.plot(y_pred,'r', label='Predicted')
plt.title('Linear Regression: Cases per Million Prediction (Multivariate)')
plt.xlabel('Test Samples')
plt.ylabel('Cases per Million')
plt.legend()
plt.grid()
plt.savefig('cases_per_million_prediction_linear_multi.png')
plt.show()

print(len(y_pred))

print(r2_score(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor

x=df3[['reproduction_rate','people_fully_vaccinated_per_hundred']].values
y=df3[['new_cases_smoothed_per_million']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)

forest_reg = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=5)
forest_reg.fit(x_train, np.ravel(y_train))

y_pred=forest_reg.predict(x_test)

plt.plot(y_test, label='Actual')
plt.plot(y_pred,'r', label='Predicted')
plt.title('Random Forest: Cases per Million Prediction (Multivariate)')
plt.xlabel('Test Samples')
plt.ylabel('Cases per Million')
plt.legend()
plt.grid()
plt.savefig('cases_per_million_prediction_rf_multi.png')
plt.show()

print(len(y_pred))

print(r2_score(y_test, y_pred))

# Cross-sectional analysis from covidpredictTotal.ipynb
print("\n" + "="*50)
print("CROSS-SECTIONAL COUNTRY ANALYSIS")
print("="*50)

# Picking data from a certain date (snapshot analysis)
snapshot_date = '2022-05-01'
df_snapshot = df.loc[df['date'] == snapshot_date]

# Selecting certain parameters for cross-sectional analysis
df_snapshot = df_snapshot[['location', 'total_cases_per_million', 'total_deaths_per_million',
                          'total_vaccinations_per_hundred', 'total_tests_per_thousand']]

print(f"Cross-sectional data shape for {snapshot_date}:", df_snapshot.shape)
print("Sample countries:")
print(df_snapshot.head(10))

# Selecting data from Indonesia for highlighting
df_indonesia = df_snapshot.loc[df_snapshot['location'] == 'Indonesia']
print("\nIndonesia data:")
print(df_indonesia)

# Sorting values by total deaths per million
df_snapshot_sorted = df_snapshot.sort_values(by=['total_deaths_per_million'], ascending=False)
print("\nTop 20 countries by total deaths per million:")
print(df_snapshot_sorted.head(20)[['location', 'total_deaths_per_million']])

# 3D Scatter Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot all countries
ax.scatter(df_snapshot['total_cases_per_million'],
           df_snapshot['total_deaths_per_million'],
           df_snapshot['total_vaccinations_per_hundred'],
           s=50, edgecolor="k", alpha=0.6, label='All Countries')

# Highlight Indonesia
if not df_indonesia.empty:
    ax.scatter(df_indonesia['total_cases_per_million'],
               df_indonesia['total_deaths_per_million'],
               df_indonesia['total_vaccinations_per_hundred'],
               c='red', marker="s", s=200, edgecolor="k", label='Indonesia')

ax.set_title(f'COVID-19 Cross-Sectional Analysis ({snapshot_date})')
ax.set_xlabel('Total Cases per Million')
ax.set_ylabel('Total Deaths per Million')
ax.set_zlabel('Total Vaccinations per Hundred')
ax.legend()
plt.savefig('cross_sectional_3d_analysis.png')
plt.show()

# Country-level predictions
print("\n" + "="*50)
print("COUNTRY-LEVEL PREDICTIONS")
print("="*50)

# Prediction 1: Total deaths from total cases
df_pred1 = df_snapshot[['total_cases_per_million', 'total_deaths_per_million']].dropna()
print(f"Prediction 1 - Data shape: {df_pred1.shape}")

if len(df_pred1) > 10:
    x = df_pred1[['total_cases_per_million']].values
    y = df_pred1[['total_deaths_per_million']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=True)

    reg1 = LinearRegression().fit(x_train, y_train)
    print(f"R² Score (train): {reg1.score(x_train, y_train)}")
    print(f"Coefficients: {reg1.coef_}")
    print(f"Intercept: {reg1.intercept_}")

    y_pred = reg1.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score (test): {r2}")

    # Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.6, label='Training Data')
    plt.scatter(x_test, y_test, alpha=0.6, c='orange', label='Test Data')
    plt.plot(x_test, y_pred, 'r-', linewidth=2, label='Predictions')
    plt.title('Total Deaths vs Total Cases per Million (Cross-Sectional)')
    plt.xlabel('Total Cases per Million')
    plt.ylabel('Total Deaths per Million')
    plt.legend()
    plt.grid(True)
    plt.savefig('cross_sectional_deaths_vs_cases.png')
    plt.show()

# Prediction 2: Total deaths from vaccination rates
df_pred2 = df_snapshot[['total_vaccinations_per_hundred', 'total_deaths_per_million']].dropna()
print(f"\nPrediction 2 - Data shape: {df_pred2.shape}")

if len(df_pred2) > 10:
    x = df_pred2[['total_vaccinations_per_hundred']].values
    y = df_pred2[['total_deaths_per_million']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=True)

    reg2 = LinearRegression().fit(x_train, y_train)
    print(f"R² Score (train): {reg2.score(x_train, y_train)}")
    print(f"Coefficients: {reg2.coef_}")
    print(f"Intercept: {reg2.intercept_}")

    y_pred = reg2.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score (test): {r2}")

    # Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.6, label='Training Data')
    plt.scatter(x_test, y_test, alpha=0.6, c='orange', label='Test Data')
    plt.plot(x_test, y_pred, 'r-', linewidth=2, label='Predictions')
    plt.title('Total Deaths vs Vaccination Rate per Hundred (Cross-Sectional)')
    plt.xlabel('Total Vaccinations per Hundred')
    plt.ylabel('Total Deaths per Million')
    plt.legend()
    plt.grid(True)
    plt.savefig('cross_sectional_deaths_vs_vaccinations.png')
    plt.show()

# Prediction 3: Multivariate - deaths from cases and vaccinations
df_pred3 = df_snapshot[['total_vaccinations_per_hundred', 'total_cases_per_million', 'total_deaths_per_million']].dropna()
print(f"\nPrediction 3 (Multivariate) - Data shape: {df_pred3.shape}")

if len(df_pred3) > 10:
    x = df_pred3[['total_vaccinations_per_hundred', 'total_cases_per_million']].values
    y = df_pred3[['total_deaths_per_million']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=True)

    reg3 = LinearRegression().fit(x_train, y_train)
    print(f"R² Score (train): {reg3.score(x_train, y_train)}")
    print(f"Coefficients: {reg3.coef_}")
    print(f"Intercept: {reg3.intercept_}")

    y_pred = reg3.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score (test): {r2}")

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    plt.title('Multivariate Prediction: Deaths from Cases & Vaccinations')
    plt.xlabel('Actual Total Deaths per Million')
    plt.ylabel('Predicted Total Deaths per Million')
    plt.legend()
    plt.grid(True)
    plt.savefig('cross_sectional_multivariate_prediction.png')
    plt.show()