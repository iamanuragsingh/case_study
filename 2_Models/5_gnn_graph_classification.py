# -*- coding: utf-8 -*-
"""GNN_Graph_Classification"""

import os
import re
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn.functional as F 
import torch.optim as optim
import torch_scatter
import torch_sparse
import torch_geometric
import scipy.sparse as sp
from tqdm import tqdm
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data,Dataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv #GATConv
torch.manual_seed(42)
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



print(torch.__version__)
print(torch.version.cuda)

# Reading the data from csv file
transaction=pd.read_csv('1_Input/fraud_transactions_data.csv')
accounts=pd.read_csv('1_Input/accounts_data.csv')

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)


def data_preprocess:

    accounts.IS_FRAUD=accounts.IS_FRAUD.astype(int)
   
    # drop the columns with less correlation
    accounts.drop(columns=['COUNTRY','ACCOUNT_TYPE','CUSTOMER_ID'],inplace=True)

    # Remove comma from the amount
    transaction['amount']  = transaction['amount'].apply(lambda x: re.sub(r"[,]", '', x))
    
    #convert the datatype to float
    transaction['amount'] = transaction['amount'].astype(object).astype(float)
    
    #convert the datatype to object
    transaction['nameOrig'] = transaction['nameOrig'].astype(int).astype(object)
    transaction['nameDest'] = transaction['nameDest'].astype(int).astype(object)
    
    #convert the datatype to int
    transaction.isFraud=transaction.isFraud.astype(int)

    # create dummy values
    transaction_type_dummies = pd.get_dummies(transaction['Type'])

    # drop unwanted column
    transaction.drop(columns=['Type'],inplace=True)
    
    # preprocessing on transaction column
    frames = [transaction, transaction_type_dummies]
    transaction_final = pd.concat(frames)

    # visualize the correlation between columns
    sns.heatmap(transaction.corr(),annot=True,fmt='.1g')

    
    node_features=accounts.drop(columns='IS_FRAUD')

def get_node_features(tran_data):

  all_accounts = [x for i in zip(tran_data['nameOrig'], tran_data['nameDest']) for x in i]

  node_features = accounts.loc[accounts['ACCOUNT_ID'].isin(all_accounts)]

  node_feats=[]

  for i in range(len(node_features)):
    node_feats.append(list(node_features.iloc[i]))
    
  node_feats=torch.tensor(node_feats,dtype=torch.float)

  return node_feats

def get_edge_index(tran_data):

  edge_index=[]

  tran_index=tran_data[['nameOrig','nameDest']]

  for i in range(len(tran_index)):
   edge_index.append(list(tran_index.iloc[i]))

  edge_index=torch.tensor(edge_index).T

  return edge_index

def get_edge_features(tran_data):

  edge_feats=[]

  trans_feats=tran_data.drop(columns=['nameOrig','nameDest'])

  for i in range(len(trans_feats)):
    edge_feats.append(list(trans_feats.iloc[i]))

  edge_feats=torch.tensor(edge_feats,dtype=torch.float)

  return edge_feats

def get_label(tran_data):

  if 1 in tran_data['isFraud']:
    label = 1
    label = np.asarray([label])
  else:
    label = 0
    label = np.asarray([label])

  return torch.tensor(label,dtype=torch.int64)

def create_data:

    torch.ones(1)>=1

    fraud_pattern = transaction.groupby(['Pattern'])
    train_dataset = []
    test_dataset = []
    p = iter(fraud_pattern)
    for i in range(1 , 26):
    frame = fraud_pattern.get_group(i)

    train_dataset.append(Data(x=get_node_features(frame),
          edge_index=get_edge_index(frame),
          edge_attr=get_edge_features(frame),
          y=get_label(frame)))
  
    for i in range(25,46):
    frame = fraud_pattern.get_group(i)

    test_dataset.append(Data(x=get_node_features(frame),
          edge_index=get_edge_index(frame),
          edge_attr=get_edge_features(frame),
          y=get_label(frame)))

    data = train_dataset[1]



                             
class GCN(torch.nn.Module):

    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        print("First Message Passing Layer (Transformation)")
        print("x = self.conv1(x, edge_index)")
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        print("x = x.relu()")
        x = x.relu()
        print("x = F.dropout(x, p=0.5, training=self.training)")
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        print("Second Message Passing Layer")
        print("x = self.conv1(x, edge_index)")
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = F.softmax(self.out(x), dim=1)
        return x


def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(data.x, data.edge_index)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc
      
      
      
      
if __name__ == "__main__":

    data_preprocess()
 
    create_data()
    # Initialize model
    model = GCN(hidden_channels=16)
  
    # Define loss function (CrossEntropyLoss for Classification Problems with 
    # probability distributions)
    criterion = torch.nn.CrossEntropyLoss()

    train()
    test()
