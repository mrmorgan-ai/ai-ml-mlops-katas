import os
import json
import pandas as pd
import numpy as np

from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLDataLoader:
    def __init__(self, data_path: str, random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError("Path to dataset does not exists")
        
        df = pd.read_csv(self.data_path)
        print(f"Loaded {df.shape[0]} registers")
        return df
    
    def remove_high_cardinality(self, df: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
        if df.shape[0] == 0:
            raise ValueError(f"the dataframe {df} is empty")
        
        cols_to_keep = []
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == "object":
                if df[col].nunique() < threshold:
                    cols_to_keep.append(col)
            
            else:
                cols_to_keep.append(col)
        return df[cols_to_keep]
    
    def split_features_and_target(self,
        df: pd.DataFrame,
        target_col_name: str
        )-> Tuple[pd.DataFrame, pd.Series]: # type: ignore
        
        if not target_col_name:
            raise ValueError(f"Targe column {target_col_name} does not exist")
        
        X = df.drop(columns=[target_col_name])
        y = df[target_col_name]
        return X,y
    
    def train_test_split(self, 
        X: pd.DataFrame,
        y: pd.Series,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2
        ): # type: ignore
    
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, val_size, random_state=42)
        
        return X_train, y_train, X_val, y_val, X_test, y_test       
