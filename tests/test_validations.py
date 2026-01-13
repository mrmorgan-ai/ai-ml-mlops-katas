import sys
import os
import pytest
import pandas as pd
import numpy as np
import pandera as pa
from pathlib import Path
from pandera.errors import SchemaError

sys.path.insert(0, str(Path(__file__).parent.parent))
print(os.getcwd())

from src.pipelines.preprocessing import MLDataLoader
from src.pipelines.data_validation import validate_transactions, check_feature_drift

data_path = r"C:\Users\jhoni\Documents\LooperAI\repositorios\ai-ml-mlops-katas\data\raw\Exam_Score_Prediction.csv"
data_loader = MLDataLoader(data_path)

class TestTransactionValidation:
    @pytest.fixture
    def valid_transactions(self):
        df = data_loader.load_data()
        return df
    
    def test_valid_data_passes(self, valid_transactions):
        result = valid_transactions(valid_transactions)
        assert len(result) == 3
if __name__ == "__main__":
    test_validation = TestTransactionValidation()
    
    print(test_validation.test_valid_data_passes(test_validation.valid_transactions))
