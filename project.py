import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

print("\n=======GROUP 6========\n\n========COVID ICU Prediction is listing as the following details:========")
ICUdata = pd.read_csv('COVIDICUPrediction.csv')

print("\n",ICUdata.head())

print("\n ===================================\n"
      "There are 1925 entries, 0 to 1924 and 231 Columns(from PATIENT_VISIT_IDENTIFIER to ICU) base on the following details:\n")
print(ICUdata.info())
print("\n=======Analyzing purpose #1 =======\n")
print("\n=======#1: Organize patients by different age brackets and identify any common risks within each group.=======\n")
print("New Age_data will drop the ICU column.")
Age_data = ICUdata.drop(columns='ICU')

col_search = 'AGE_PERCENTIL'
plt.bar(Age_data[col_search].value_counts().index, Age_data[col_search].value_counts())
plt.ylabel("Amount", rotation='vertical')
plt.xlabel("AGE_PERCENTIL")
plt.show()

print(f'Unique values: {Age_data[col_search].unique()}')


print("Defining the group of Patient ID and Window as each group age and each window group")
definitions = []
for i in Age_data.columns:
    if type(Age_data[i].iloc[0]) == str:
        factor = pd.factorize(Age_data[i],sort=True)
        Age_data[i] = factor[0]
        definitions.append([np.unique(factor[0]), factor[1]])
print(definitions,"\n")

print("""New Age data define the group age as
    0 : 10th
    1 : 20th
    2 : 30th
    3 : 40th
    4 : 50th  
    5 : 60th
    6 : 70th
    7 : 80th
    8 : 90th
    9 : Above 90th
and The Window group: 
    0: 0-2
    1: 2-4
    2: 4-6
    3: 6-12
    4: ABOVE_12"""
)

col_search = 'AGE_PERCENTIL'
sns.barplot(y=ICUdata[col_search].value_counts().index,x=ICUdata[col_search].value_counts())
plt.show()
print(f'Unique values: {ICUdata[col_search].unique()}')
print("Selecting 50 row of two column AGE_PERCENTIL and WINDOW and drop the duplicated to show Age_data after defination")
print(Age_data.loc[0:50, ['AGE_PERCENTIL', 'WINDOW']].drop_duplicates())
print("====================================================")




print("Check Missing Values in each Column of Age_data as following detail:")
print(Age_data.isnull().sum()[Age_data.isnull().sum()>0])
print("====================================================")
print("All columns contains an value for each group except the unrelated column PATIENT_VISIT_IDENTIFIER ID is increasing from 0-1924 row.\n "
      "So Age_data will drop the PATIENT_VISIT_IDENTIFIER to have the good "
      " data to find the correlation between these groups")

from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(strategy='mean')
Correlation_data = pd.DataFrame(Imputer.fit_transform(Age_data.drop(columns='PATIENT_VISIT_IDENTIFIER')))
Correlation_data.columns = Age_data.drop(columns='PATIENT_VISIT_IDENTIFIER').columns
print("Selecting 50 row to show new data name : Correlation_data ")
print(Correlation_data.loc[0:50, :])


cor = pd.DataFrame(np.triu(Correlation_data.corr().values), index = Correlation_data.corr().index,
                   columns=Correlation_data.corr().columns).round(3)

cor_data = cor.unstack()

cor_data_2 = cor_data.sort_values()


most_correlated = cor_data_2[cor_data_2>0.95][cor_data_2[cor_data_2>0.90] !=1]
print("====================================================")
print("Correlation table ")
print(pd.DataFrame(most_correlated, columns=['correlation']))
n = len(most_correlated)
print('Number of feature most correlated are: {}'.format(n))
print("====================================================")
print("Showing the Correlation between left and right column for comparison")

for i in range(1, 10):
    x = most_correlated.index[i-1][0]
    y = most_correlated.index[i-1][1]
    sns.scatterplot(x=x,y=y,data=Correlation_data)
    plt.show()

