# Import packages 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.tree import plot_tree
from sklearn import tree

# Import training dataset 

df = pd.read_csv("1_Input/training_data.csv")

# Data Preprocessing

df['nameOrig'] = df['nameOrig'].astype('object')
df['nameDest'] = df['nameDest'].astype('object')
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

df1 = pd.get_dummies(df, columns=['Type'], prefix=['Type'])
df1 = pd.get_dummies(df1, columns=['Orig_Location'], prefix=['Orig_Location'])
df1 = pd.get_dummies(df1, columns=['Dest_Location'], prefix=['Dest_Location'])

# Getting information of dataset

print(df1.info())

df1.isnull().any()

target = df['isFraud']
df2 = df1.copy()
df2 = df2.drop('isFraud', axis =1)

# Defining the attributes

X = df2

# Label encoding

le = LabelEncoder()
target = le.fit_transform(target)
target

y = target

# Splitting the data - 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.3, stratify = y, random_state = 42)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)


# Defining the decision tree algorithm

dtree=DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=1)
dtree.fit(X_train,y_train)
print('Decision Tree Classifier Created')
dtree_representation = tree.export_text(dtree)
print(dtree_representation)

tree.plot_tree(dtree)
plt.show()

# Predicting the values of test data

y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

score = metrics.accuracy_score(y_test, y_pred)
print(score)

#Import pickle Package

import pickle

#Save the Model to file in the current working directory

Pkl_Filename = "Pickle_dtree.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(dtree, file)

# Load the Model back from file

with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

print(Pickled_LR_Model)

# Use the Reloaded Model to Calculate the accuracy score and predict target values

# Import testing dataset

df3 = pd.read_csv("1_Input/testing_data.csv")

# Data Preprocessing 

df3['nameOrig'] = df3['nameOrig'].astype('object')
df3['nameDest'] = df3['nameDest'].astype('object')
df3 = df3.drop('Date', axis =1)

# Creating dummy variables through one hot encoding for 'type' column

df4 = pd.get_dummies(df3, columns=['Type'], prefix=['Type'])
df4 = pd.get_dummies(df4, columns=['Orig_Location'], prefix=['Orig_Location'])
df4 = pd.get_dummies(df4, columns=['Dest_Location'], prefix=['Dest_Location'])


# Use the loaded pickled model to make predictions

Ypredict = Pickled_LR_Model.predict(df4)  

df3['isFraud'] = Ypredict


# Export the results

def highlight_sentiment(row):
    if row["isFraud"] == 1:
        return ['background-color: yellow'] * len(row)
    else:
        return ['background-color: white'] * len(row)
    
df3 = df3.style.apply(highlight_sentiment, axis=1)

df3.to_excel("5_Output/Result2.xlsx")
