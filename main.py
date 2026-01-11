import pandas as pd
import numpy as np

from src.pipelines.preprocessing import MLDataLoader

def main():
    data_path = r"C:\Users\jhoni\Documents\LooperAI\repositorios\ai-ml-mlops-katas\data\raw\Exam_Score_Prediction.csv"
    
    # Initialize loader
    data_loader = MLDataLoader(data_path)
    
    # Upload raw data
    df_raw = data_loader.load_data()

    # Remove high cardinality
    df_remove_hc = data_loader.remove_high_cardinality(df_raw)
    
    # Split feature and target
    X,y = data_loader.split_features_and_target(df_remove_hc, "exam_score")
    
    # Split train, val, test datasets
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.train_test_split(X,y,val_size=0.2,test_size=0.2)

if __name__ == "__main__":
    main()
