# Anti Money Laundering Machine Learning & Deep Learning

This repository contains 2 Machine Learning & 1 Deep Learning Model to implement money laundering fraud detection. We have used Logistic Regression, Decision Tree and Graphical Neural Network (Node classification and Graph Classification) Models in the case study. Additionally we have also implemented Rule based model which will always check for certain rules before passing the data to any of the models and raise an suspicious alert on the particular accounts. In this project we have created a synthetic bank transaction data, which represents various fraud and non-fraud patterns of bank transactions. In the end our model predicts, whether the transaction is Fraud or Non-Fraud.

## Contents of the file

- The Input file 'data.json' given to cleaning pipeline for processing the data is stored inside Input folder.
```
Input/transactions.csv
Input/accounts.csv
```

- The final fraud transaction and accounts highlighted csv file 'Fraud-transaction.csv' is stored inside Output folder
```
Output/Fraud-transaction.csv
```
- All the machine learning and deep learning models are stored inside Models folder.
```
Model/logistic_regression.py
Model/decision_tree.py
Model/graphical_neural_network.py
Model/rule_based.py
```
- The document consisting of model research is stored inside Document folder.
```
Document/manuscript.pdf
```
- To install Python packages with pip and requirements.txt
```
requirements.txt
```

## Pipeline for Logistic regression, decision tree and graphical neural networks model is as follows :

![alt text](https://github.com/Big-Data-And-Data-Analytics/case-study-1-october2020-beyond-analytics/blob/main/6_Images/Money_Launderin_Project_basic_pipeline.png)

![alt text](https://github.com/Big-Data-And-Data-Analytics/case-study-1-october2020-beyond-analytics/blob/main/6_Images/Logistic%20regression%20pipeline.PNG)

![alt text](https://github.com/Big-Data-And-Data-Analytics/case-study-1-october2020-beyond-analytics/blob/main/6_Images/DecisionTree%20pipeline.png)

# GCN Model pipeline
![alt text](https://github.com/Big-Data-And-Data-Analytics/case-study-1-october2020-beyond-analytics/blob/main/6_Images/GCN%20Model.PNG)

## Environment

We need a python environment for executing the models. Latest python version (3.9.6) for 64 bit Windows 10 can be supportive to run these files.

## Installation

pip version 21.1.3

Install packages with pip: -r requirements.txt

The following command will install the packages according to the configuration file requirements.txt. 
Run the following command where requirements.txt file is located.
```
pip install -r requirements.txt
```

Description of the python packages used in the text cleaning pipeline.

1. pandas is used to read the input json file.

2. torch-geometric

3. torch-sparse

4. torch-scatter

5. sklearn

## Execution

Change the directory of the file where project is located.

1. Run the 1_Rule_based.py python file using below command

``` python Models/1_Rule_based.py ```

2. Run the 2_logistic_regression.py python file using below command

``` python Models/2_logistic_regression.py ```

3. Run the 3_decision_tree.py python file using below command

``` python Models/3_decision_tree.py ```

4. Run the 4_gnn_node_classification.py python file using below command

``` python Models/4_gnn_node_classification.py ```

5. Run the 5_gnn_graph_classification.py python file using below command

``` python Models/5_gnn_graph_classification.py ```

After the command is executed, output will be stored in ``` 5_Output/Labelled_Output.csv ``` location



