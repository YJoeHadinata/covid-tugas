import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

plt.plot(df1['new_cases'])
plt.plot(df2['new_cases'],'r')
plt.grid()
plt.savefig('new_cases_comparison.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','new_deaths', 'new_cases','positive_rate', 'new_cases_smoothed','new_deaths_smoothed']]

df2 = df.loc[df['location'] == 'United Kingdom']
df2=df2[['location','date','new_deaths', 'new_cases','positive_rate', 'new_cases_smoothed','new_deaths_smoothed']]

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

df1.plot.scatter(x='new_cases_smoothed',y='new_deaths_smoothed',s=50, c=numbers, colormap='plasma')
plt.grid()
plt.savefig('indonesia_cases_vs_deaths.png')
plt.show()

df2.plot.scatter(x='new_cases_smoothed',y='new_deaths_smoothed',s=50, c=numbers, colormap='plasma')
plt.grid()
plt.savefig('uk_cases_vs_deaths.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','new_deaths', 'stringency_index']]

df2 = df.loc[df['location'] == 'United Kingdom']
df2=df2[['location','date','new_deaths', 'stringency_index']]

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

df1.plot.scatter(x='stringency_index',y='new_deaths',s=50, c=numbers, colormap='plasma')
plt.grid()
plt.savefig('indonesia_stringency_vs_deaths.png')
plt.show()

df2.plot.scatter(x='stringency_index',y='new_deaths',s=50, c=numbers, colormap='plasma')
plt.grid()
plt.savefig('uk_stringency_vs_deaths.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','new_cases', 'stringency_index']]

df2 = df.loc[df['location'] == 'United Kingdom']
df2=df2[['location','date','new_cases', 'stringency_index']]

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

df1.plot.scatter(x='stringency_index',y='new_cases',s=50, c=numbers, colormap='plasma')
plt.grid()
plt.savefig('indonesia_stringency_vs_cases.png')
plt.show()

df2.plot.scatter(x='stringency_index',y='new_cases',s=50, c=numbers, colormap='plasma')
plt.grid()
plt.savefig('uk_stringency_vs_cases.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','stringency_index','reproduction_rate']]

df2 = df.loc[df['location'] == 'South Korea']
df2=df2[['location','date','stringency_index','reproduction_rate']]

df1=df1.dropna()
df2=df2.dropna()

#Reseting data index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

#For colormap to show time elapsing
numbers1=range(len(df1))
numbers2=range(len(df2))

df1.plot.scatter(x='stringency_index',y='reproduction_rate',s=50, c=numbers1, colormap='plasma')
plt.grid()
plt.savefig('indonesia_stringency_vs_reproduction.png')
plt.show()

df2.plot.scatter(x='stringency_index',y='reproduction_rate',s=50, c=numbers2, colormap='plasma')
plt.grid()
plt.savefig('south_korea_stringency_vs_reproduction.png')
plt.show()

#Selecting certain country in data columns
df1 = df.loc[df['location'] == 'Indonesia']
df1=df1[['location','date','new_vaccinations','reproduction_rate']]

df2 = df.loc[df['location'] == 'South Korea']
df2=df2[['location','date','new_vaccinations','reproduction_rate']]

df1=df1.dropna()
df2=df2.dropna()

#Reseting data index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

#For colormap to show time elapsing
numbers1=range(len(df1))
numbers2=range(len(df2))

df1.plot.scatter(x='new_vaccinations',y='reproduction_rate',s=50, c=numbers1, colormap='plasma')
plt.grid()
plt.savefig('indonesia_vaccinations_vs_reproduction.png')
plt.show()

df2.plot.scatter(x='new_vaccinations',y='reproduction_rate',s=50, c=numbers2, colormap='plasma')
plt.grid()
plt.savefig('south_korea_vaccinations_vs_reproduction.png')
plt.show()

#Selecting certain country in data columns
df_vac1 = df.loc[df['location'] == 'Indonesia']
df_vac1=df_vac1[['location','date','new_vaccinations','new_deaths']]

df_vac2 = df.loc[df['location'] == 'United Kingdom']
df_vac2=df_vac2[['location','date','new_vaccinations','new_deaths']]

df_vac1=df_vac1.dropna()
df_vac2=df_vac2.dropna()

#Reseting data index
df_vac1.reset_index(drop=True, inplace=True)
df_vac2.reset_index(drop=True, inplace=True)

#For colormap to show time elapsing
numbers1=range(len(df_vac1))
numbers2=range(len(df_vac2))

df_vac1.plot.scatter(x='new_vaccinations',y='new_deaths',s=50, c=numbers1, colormap='plasma')
plt.grid()
plt.savefig('indonesia_vaccinations_vs_deaths.png')
plt.show()

df_vac2.plot.scatter(x='new_vaccinations',y='new_deaths',s=50, c=numbers2, colormap='plasma')
plt.grid()
plt.savefig('uk_vaccinations_vs_deaths.png')
plt.show()

#Selecting certain parameters for prediction
df1 = df.loc[df['location'] == 'Indonesia']
df3=df1[['new_cases','new_deaths']]
print(df3.shape)
print(df3.head())

df3=df3.dropna()
df3.reset_index(drop=True, inplace=True)
print(df3.shape)
print(df3.head())

from sklearn.model_selection import train_test_split

x=df3[['new_cases']].values
y=df3[['new_deaths']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=False)

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x_train, y_train)
print(reg.score(x_train, y_train))
print(reg.coef_)
print(reg.intercept_)

y_pred=reg.predict(x_test)
#assume deaths number will be delayed by 14 days and delta is 2x more deadly than previous data
y_predpad=np.pad(0.15*y_pred[0:len(y_pred)-10],(10,0),'minimum')

plt.plot(y_test)
plt.plot(y_predpad,'r')
plt.grid()
plt.savefig('deaths_prediction_linear.png')
plt.show()

print(len(y_pred))

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

#Selecting certain parameters for prediction
df3=df1[['new_cases','new_vaccinations','stringency_index','new_tests','new_deaths']]
print(df3.shape)
print(df3.head())

df3=df3.dropna()
df3.reset_index(drop=True, inplace=True)
print(df3.shape)
print(df3.head())

x=df3[['stringency_index','new_vaccinations','new_tests']].values
y=df3[['new_cases']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=False)

reg = LinearRegression().fit(x_train, y_train)
print(reg.score(x_train, y_train))
print(reg.coef_)
print(reg.intercept_)

y_pred=reg.predict(x_test)

plt.plot(y_test)
plt.plot(y_pred,'r')
plt.grid()
plt.savefig('cases_prediction_linear.png')
plt.show()

print(len(y_pred))

print(r2_score(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor

x=df3[['new_vaccinations','new_tests','stringency_index']].values
y=df3[['new_cases']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)

forest_reg = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=5)
forest_reg.fit(x_train, np.ravel(y_train))

y_pred=forest_reg.predict(x_test)

plt.plot(y_test)
plt.plot(y_pred,'r')
plt.grid()
plt.savefig('cases_prediction_rf.png')
plt.show()

print(len(y_pred))

print(r2_score(y_test, y_pred))