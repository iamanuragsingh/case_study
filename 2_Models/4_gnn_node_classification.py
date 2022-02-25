# -*- coding: utf-8 -*-
"""GNN_node_classification"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch_scatter
import torch_sparse
import torch_geometric
torch.manual_seed(42)

from torch_geometric.nn import GCNConv #GATConv
from torch_geometric.data import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data,Dataset

print(torch.__version__)
print(torch.version.cuda)

# Reading the data from csv
transaction = pd.read_csv('1_Input/fraud_transactions_data.csv')
accounts = pd.read_csv('1_Input/accounts_data.csv')

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)
                             
def data_preprocess:

    """Preprocessing the data"""

    ### Transactions
    
    # Fill null values with string 'NA'
    transaction.ALERT_TYPE.fillna('NA',inplace=True)

    # visualizing heatmap for correlation
    sns.heatmap(transaction.corr(),annot=True,fmt='.1g')

    # drop the column which has no corelation with target column
    transaction.drop(columns='TX_TYPE',inplace=True)

    # Convert column to int to get better results
    transaction.IS_FRAUD = transaction.IS_FRAUD.astype(int)

    # creating one hot encoding for transaction data
    transaction_final = pd.get_dummies(transaction)
    
    #  creating transaction index
    transaction_index=transaction_final[['SENDER_ACCOUNT_ID','RECEIVER_ACCOUNT_ID']]

    ### Accounts 
    
    # drop the column which has no corelation with target column
    accounts.drop(columns=['COUNTRY','ACCOUNT_TYPE','CUSTOMER_ID'],inplace=True)

    # Convert the type to int
    accounts.IS_FRAUD=accounts.IS_FRAUD.astype(int)

    # Drop target column from node features
    node_features=accounts.drop(columns='IS_FRAUD')


def create_data:

    """Creating the data"""
    
    # Creating the node features
    node_feats=[]
    for i in range(len(node_features)):
    node_feats.append(list(node_features.iloc[i]))

    node_feats=torch.tensor(node_feats,dtype=torch.float)

    label=list(accounts['IS_FRAUD'])

    label=torch.tensor(label,dtype=torch.int64)

    # creating edge index
    edge_index=[]
    for i in range(len(transaction_index)):
    edge_index.append(list(transaction_index.iloc[i]))

    edge_index=torch.tensor(edge_index).T

    edge_index.shape

    transaction_feats=transaction_final.drop(columns=['SENDER_ACCOUNT_ID','RECEIVER_ACCOUNT_ID','TX_ID'])

    transaction_feats

    # creating edge features
    edge_feats=[]
    for i in range(len(transaction_feats)):
        edge_feats.append(list(transaction_feats.iloc[i]))

    edge_feats=torch.tensor(edge_feats,dtype=torch.float)

    # creating data object
    data=Data(x=node_feats,
          edge_index=edge_index,
          edge_attr=edge_feats,
          y=label,
          train_mask=torch.ones(1000)>=1,
          test_mask=torch.ones(1000)>=1)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, 3)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = F.softmax(self.out(x), dim=1)
        return x

def train():

    """Training the data"""
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(data.x ,data.edge_index)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():

      """Calculate test metric"""
      model.eval()
      out = model(data.x, data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc



def visualize_result:

    losses_float = [float(loss.cpu().detach().numpy()) for loss in losses[1:]] 
    loss_indices = [i for i,l in enumerate(losses_float)] 
    plt = sns.lineplot(loss_indices, losses_float)
    plt

    sample = 12
    sns.set_theme(style="whitegrid")
    pred = model(data.x, data.edge_index)
    sns.barplot(x=np.array(range(3)), y=pred[sample].detach().cpu().numpy())

if __name__ == "__main__":

    data_preprocess()
 
    create_data()
    # Initialize model
    model = GCN(hidden_channels=16)
    
    # Define loss function (CrossEntropyLoss for Classification Problems with 
    # probability distributions)
    criterion = torch.nn.CrossEntropyLoss()
    
     """Training the data"""
    losses = []
    for epoch in range(0, 1001):
        loss = train()
        losses.append(loss)
        if epoch % 100 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    """Calculate test metric"""
    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')
    
    """Visualize the result"""
    visualize_result()
    

