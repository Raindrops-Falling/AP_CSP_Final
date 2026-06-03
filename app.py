#number stuff

import pandas as pd
import numpy as np

#the ai
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import TensorDataset

#Others
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class GRUModel(nn.Module):
    def __init__(self, hidden_dim, delta_dim=16):
        super(GRUModel, self).__init__()
        self.gru=nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.delta_layer=nn.Linear(1, delta_dim)
        self.output_layer=nn.Linear(hidden_dim+delta_dim, 1)
        self.act=nn.ReLU()
    def forward(self, padded, length_tensor, delta_tensor):
        
        packed=pack_padded_sequence(padded, length_tensor, batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)
        sequence_features=h_n.squeeze(0)
        delta_t=delta_tensor.unsqueeze(1)
        delta_features=self.act(self.delta_layer(delta_t))
        x=torch.cat([sequence_features, delta_features], dim=1)
        out=self.output_layer(x)
        return out


class noDELTAGRU(nn.Module):
    def __init__(self, hidden_dim):
        super(noDELTAGRU, self).__init__()
        self.gru=nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.output_layer=nn.Linear(hidden_dim, 1)
    def forward(self, padded, length_tensor):
        packed=pack_padded_sequence(padded, length_tensor, batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)
        sequence_features=h_n.squeeze(0)
        out=self.output_layer(sequence_features)
        return out



def processingNormalized():
    #initialize lists
    sequence=[]
    lengths=[]

    #read from csv
    df=pd.read_csv('opensource_dataset_forgetting_curve.tsv',sep='\t')
    vals = df[["t_history", "r_history"]].to_numpy()
    deltas=(df["delta_t"].to_numpy())
    recall=(df["p_recall"].to_numpy())

    #create loaded and associated sequences
    for i in range (len(vals)):
        e = [int(x) for x in vals[i][0].split(",")]
        f = [int(x) for x in vals[i][1].split(",")]
        sequence.append([[e_i, f_i] for e_i, f_i in zip(e, f)])
        lengths.append(len(list(zip(e,f))))
    
    #create tensors and pad sequences
    sequence_tensors=[]
    length_tensor=torch.tensor(lengths, dtype=torch.long)
    output_tensor=torch.tensor(recall, dtype=torch.float32)
    delta_tensor=torch.tensor(deltas, dtype=torch.float32)
    for s in sequence:
        sequence_tensors.append(torch.tensor(s,dtype=torch.float32))
    padded=pad_sequence(sequence_tensors, batch_first=True, padding_value=-1)  

    return padded, length_tensor, delta_tensor, output_tensor
    
def processing():

    #initialize lists
    sequence=[]
    lengths=[]

    #read from csv
    df=pd.read_csv('opensource_dataset_forgetting_curve.tsv',sep='\t')
    vals = df[["t_history", "r_history"]].to_numpy()
    deltas=(df["delta_t"].to_numpy())
    recall=(df["p_recall"].to_numpy())

    #create loaded and associated sequences
    for i in range (len(vals)):
        e = [int(x) for x in vals[i][0].split(",")]
        f = [int(x) for x in vals[i][1].split(",")]
        sequence.append([[e_i, f_i] for e_i, f_i in zip(e, f)])
        lengths.append(len(list(zip(e,f))))
    
    #create tensors and pad sequences
    sequence_tensors=[]
    length_tensor=torch.tensor(lengths, dtype=torch.long)
    output_tensor=torch.tensor(recall, dtype=torch.float32)
    delta_tensor=torch.tensor(deltas, dtype=torch.float32)
    for s in sequence:
        sequence_tensors.append(torch.tensor(s,dtype=torch.float32))
    padded=pad_sequence(sequence_tensors, batch_first=True, padding_value=-1)  

    
    
    # sorting for the train/test splits
    indices = np.arange(len(output_tensor))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    padded_train = padded[train_idx]
    padded_test = padded[test_idx]
    length_tensor_train=length_tensor[train_idx]
    length_tensor_test=length_tensor[test_idx]
    delta_tensor_train=delta_tensor[train_idx]
    delta_tensor_test=delta_tensor[test_idx]
    output_tensor_train=output_tensor[train_idx]
    output_tensor_test=output_tensor[test_idx]
    
    return padded_train, length_tensor_train, delta_tensor_train, output_tensor_train, padded_test, length_tensor_test, delta_tensor_test, output_tensor_test
    
    
    
criterion = nn.BCEWithLogitsLoss() 

def train(model, padded_train, length_tensor_train, delta_tensor_train, output_tensor_train, epochs=10, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(padded_train, length_tensor_train, delta_tensor_train, output_tensor_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_padded, batch_lengths, batch_deltas, batch_outputs in dataloader:
            optimizer.zero_grad()
            preds = model(batch_padded, batch_lengths, batch_deltas).squeeze(1)
            loss = criterion(preds, batch_outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_padded.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model

def run():
    padded_train, length_tensor_train, delta_tensor_train, output_tensor_train, padded_test, length_tensor_test, delta_tensor_test, output_tensor_test = processing()
    model = GRUModel(hidden_dim=64)
    train(model, padded_train, length_tensor_train, delta_tensor_train, output_tensor_train)
    torch.save(model.state_dict(), "dataset_prediction.pth")
    model.eval()
    with torch.no_grad():
        preds = model(padded_test, length_tensor_test, delta_tensor_test).squeeze(1)
        loss = criterion(preds, output_tensor_test).item()
        probs = torch.sigmoid(preds)
        mae = torch.mean(torch.abs(probs - output_tensor_test)).item()
        print((list(probs))[-1])
        print("Test loss:", loss)
        print("Test MAE:", mae)


run()

def usePreStored():
    padded, length_tensor, delta_tensor, output_tensor=processingNormalized()
    model=GRUModel(hidden_dim=64)
    model.load_state_dict(torch.load("dataset_prediction.pth"))
    model.eval()
    with torch.no_grad():
        preds = model(padded, length_tensor, delta_tensor).squeeze(1)
        loss = criterion(preds, output_tensor).item()
        probs = torch.sigmoid(preds)
        mae = torch.mean(torch.abs(probs - output_tensor)).item()
        print((list(probs))[:50])
        print("Test loss:", loss)
        print("Test MAE:", mae)

usePreStored()

