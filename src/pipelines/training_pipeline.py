"""
dataclass — For creating clean data container classes
Optional — Type hint meaning "this value can be None"
Dict, Any, List — Type hints for dictionaries, any type, and lists
BaseEstimator — Base class for all sklearn models (for type hints)
Pipeline — Chains preprocessing and model steps
StandardScaler — Normalizes features to mean=0, std=1
RandomForestClassifier — Ensemble model good for tabular data
roc_auc_score — Metric for binary classification
joblib — Efficient serialization for numpy arrays and sklearn models
"""
import os
import sys
from pathlib import Path

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any, List, Dict, Tuple

import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Model artifact dataclass
@dataclass
class ModelArtifact:
    """
    Container for trained model and associated metadata.
    
    Attributes
    model : BaseEstimator
        The trained sklearn model or pipeline.
    feature_names : List[str]
        Ordered list of feature names the model expects.
    training_metrics : Dict[str, float]
        Metrics computed during training (accuracy, AUC, etc.).
    version : str
        Version identifier for model tracking.
    """
    model: BaseEstimator
    feature_names: List[str]
    training_metrics: Dict[str,float]
    version: str

class ExamScorePrediction:
    """
    End-to-end fraud detection pipeline.
    
    Parameters
    model_params : Dict[str, Any], optional
        Parameters passed to RandomForestClassifier.
        Defaults to balanced class weights for fraud detection.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        self.model_params = model_params or {
            "n_estimators":10,
            "max_depth":10,
            "random_state":42,
            "class_weight":"balanced"
        }
