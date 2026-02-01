import pandas as pd
import numpy as np
import pandera as pa
from pandera.errors import SchemaError

from src.pipelines.preprocessing import MLDataPreprocessor
from src.monitoring.data_validation import validate_exam_data

from config.config import DATA_PATH

def main():    
    # Initialize loader
    data_loader = MLDataPreprocessor(DATA_PATH)
    
    # Upload raw data
    df_raw = data_loader.load_data()

    # Validate raw data
    try:
        validated = validate_exam_data(df_raw)
        print("Validation succeed!!")
    except SchemaError as e:
        print(f"Validation failed: {e}")
        
    # Remove high cardinality
    df_remove_hc = data_loader.remove_high_cardinality(df_raw)
    
    # Split feature and target
    X,y = data_loader.split_features_and_target(df_remove_hc, "exam_score")
    
    # Split train, val, test datasets
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.train_test_split(X,y,test_size=0.2, val_size=0.2)

    print("Preprocessing complete!!\n")
    print(f"Found {X_train.shape[0]} registers in train dataset")
    print(f"Found {X_val.shape[0]} registers in train dataset")
    print(f"Found {X_test.shape[0]} registers in train dataset")
    
if __name__ == "__main__":
    main()
