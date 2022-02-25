#Import packages

import pandas as pd
import numpy as np
from collections import Counter

# Import the data
df = pd.read_csv("1_Input/testing_data.csv")

# Rules to Identify Known Fraud-based

# Check the number of transaction occured through the account
df['Counts_acct'] = df.groupby(['nameOrig'])['nameDest'].transform('count')

# Check the location of the account and number of transactions exceeding the given limit from that location
df['Counts_location'] = df.groupby(['Orig_Location'])['Dest_Location'].transform('count')


conditions = [
    (df['amount'] > 10000),
    (df['Counts_acct'] > 10),
    (df['Counts_location'] > 70)
    
    ]

values = ['1', '1', '1']

df['label'] = np.select(conditions, values)

# Export the results

def highlight_sentiment(row):
    
    ''' Function to highlight the rows which meet the conditions/rules'''
    if row['label'] == "1":
        return ['background-color: yellow'] * len(row)
    else:
        return ['background-color: white'] * len(row)
        
df1 = df.style.apply(highlight_sentiment, axis=1)

# Storing the output in Excel file
df1.to_excel("5_Output/Labelled_Output.xlsx")







