from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple
import torch
    
def fit_labelencoder(X_windows, y_windows) -> LabelEncoder: 
    '''
    Fit preprocessing labels TRAINING only:
      - LabelEncoder on labels  
    X_windows: [#_windows, L, F]
    y_windows: [#_windows]
    '''
    # Label Encoder. On Training Labels
    label_encoder = LabelEncoder().fit(y_windows)    
    return label_encoder 

class Dataset_HAR(Dataset):
    ''' 
    DataLoader needs an instance of a Class 
    '''
    def __init__(self, X_windows, y_windows, label_encoder): 
        self.label_encoder = label_encoder
        self.X = torch.tensor(X_windows, dtype=torch.float32)
        self.y = torch.tensor(self.label_encoder.transform(np.asarray(y_windows)), dtype=torch.long)
        
    def __len__(self):
        return self.X.shape[0] # self.X.shape = [10699, 150, 45]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] # self.X[5] (window 5) -> shape [150, 45]