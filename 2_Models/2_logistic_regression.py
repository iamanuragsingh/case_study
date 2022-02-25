# Import packages

from itertools import count
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sn
from openpyxl.workbook import Workbook
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Import the training data

df = pd.read_csv("1_Input/training_data.csv")

# Data Preprocessing

df['isFraud'] = df['isFraud'].astype('object')
df['Type_code'] = df['Type_code'].astype('object')
df = df.drop('Date', axis =1)

# Summary statistics of numerical and categorical variables

print(df.describe(include='all'))

print('Maximum number of missing values in any column: ' + str(df.isnull().sum().max()))

# Check that there are no negative amounts
print('Number of transactions where the transaction amount is negative: ' + str(sum(df['amount'] < 0)))

# Check instances where transacted amount is 0
print('Number of transactions where the transaction amount is equal to zero: ' + str(sum(df['amount'] == 0)))

# Count the occurrences of fraud and no fraud and print them
occ = df['isFraud'].value_counts()
print(occ)

# Print the ratio of fraud cases
ratio_cases = occ/len(df.index)
print("Ratio of fraudulent cases:",ratio_cases[1])
print("Ratio of non-fraudulent cases:",ratio_cases[0])

# Creating dummy variables through one hot encoding

df1 = pd.get_dummies(df, columns=['Type'], prefix=['Type'])
df1 = pd.get_dummies(df1, columns=['Orig_Location'], prefix=['Orig_Location'])
df1 = pd.get_dummies(df1, columns=['Dest_Location'], prefix=['Dest_Location'])

# Data Modeling

X = df1.loc[:, df1.columns != 'isFraud']
y = df1.loc[:, df1.columns == 'isFraud']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train.values.ravel())
y_test = label_encoder.fit_transform(y_test.values.ravel())

logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)

# Predicting the values of test data

y_pred=logistic_regression.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
plt.show()

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))

# Get importance

importance = logistic_regression.coef_[0]

# summarize feature importance

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance

plt.bar([x for x in range(len(importance))], importance)
plt.show()

#Import pickle Package

import pickle

#Save the Model to file in the current working directory

Pkl_Filename = "Pickle_logistic_regression.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(logistic_regression, file)

# Load the Model back from file

with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

print(Pickled_LR_Model)

# Use the Reloaded Model to Calculate the accuracy score and predict target values

df2 = pd.read_csv("1_Input/testing_data.csv")
print(df2)

df2['Type_code'] = df2['Type_code'].astype(object)
df2 = df2.drop('Date', axis =1)

#Creating dummy variables through one hot encoding 

df3 = pd.get_dummies(df2, columns=['Type'], prefix=['Type'])
df3 = pd.get_dummies(df3, columns=['Orig_Location'], prefix=['Orig_Location'])
df3 = pd.get_dummies(df3, columns=['Dest_Location'], prefix=['Dest_Location'])


#Use the loaded pickled model to make predictions

Ypredict = Pickled_LR_Model.predict(df3)  

df2['isFraud'] = Ypredict

# Export the results

def highlight_sentiment(row):
    if row["isFraud"] == 1:
        return ['background-color: yellow'] * len(row)
    else:
        return ['background-color: white'] * len(row)
    
df2 = df2.style.apply(highlight_sentiment, axis=1)

df2.to_excel("5_Output/Result1.xlsx")
