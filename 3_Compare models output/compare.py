#Import packages

import pandas as pd
import numpy as np

#Import the results from the models

df1=pd.read_excel('C:/Users/sneha/OneDrive/Desktop/Snehal/Masters_Study/Study-SEM2/CaseStudy_Pwc/python_scripts/Result1.xlsx')
df2=pd.read_excel('C:/Users/sneha/OneDrive/Desktop/Snehal/Masters_Study/Study-SEM2/CaseStudy_Pwc/python_scripts/Labelled_Output.xlsx')

# Data Preprocessing
df2 = df2.drop('Date', axis =1)
df2 = df2.drop('Counts_acct', axis =1)
df2 = df2.drop('Counts_location', axis =1)

df2 = df2.rename(columns={'label': 'isFraud'})

# Comparison of the results

df1.equals(df2)

comparison_values = df1.values == df2.values

rows,cols=np.where(comparison_values==False)

for item in zip(rows,cols):
    df1.iloc[item[0], item[1]] = '{} --> {}'.format(df1.iloc[item[0], item[1]],df2.iloc[item[0], item[1]])

df1.to_excel('./Excel_diff.xlsx',index=False,header=True)