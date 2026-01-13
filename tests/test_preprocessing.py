import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipelines.preprocessing import MLDataLoader

class TestDataLoaderClass:
    data_path = r"C:\Users\jhoni\Documents\LooperAI\repositorios\ai-ml-mlops-katas\data\raw\Exam_Score_Prediction.csv"
    data_loader = MLDataLoader(data_path)
    
    def test_split_returns_correct_shapes(self):
        """X should have  n-1 columns and y should be a series"""
        df = self.data_loader.load_data()
        X,y 


p = TestDataLoaderClass()

p.test_split_returns_correct_shapes()
