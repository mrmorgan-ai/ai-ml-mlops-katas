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
import joblib
import tempfile
from pathlib import Path

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any, List, Dict, Tuple

import mlflow
import mlflow.sklearn

import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .preprocessing import MLDataPreprocessor
from config.config import DEFAULT_MODEL_HYPERPARAMETERS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, DATA_PATH

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
    numerical_features: List[str]
    categorical_features: List[str]
    training_metrics: Dict[str,float]
    version: str
    mlflow_run_id: Optional[str] = None 

class ExamScorePipeline:
    """
    End-to-end exam score prediction
    
    This class encapsulates the complete ML workflow:
    1. PREPROCESSING
       - Scale numeric features (StandardScaler)
       - Encode categorical features (OneHotEncoder)
    
    2. MODEL TRAINING
       - Fit Random Forest Regressor
       - Calculate training metrics
       - Log everything to MLflow
    
    3. PREDICTION
       - Validate input features
       - Generate exam score predictions
    
    4. EVALUATION
       - Calculate RMSE, MAE, R² on any dataset
       - Log evaluation metrics to MLflow
    
    5. PERSISTENCE
       - Save trained pipeline to disk
       - Load trained pipeline from disk
       - Register model in MLflow Model Registry
           
    Parameters
    model_params : Dict[str, Any], optional
        Parameters passed to RandomForestClassifier.
        Defaults to balanced class weights for fraud detection.
    """
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        experiment_name: str = "exam_score_prediction",
        tracking_uri: Optional[str] = None,
        enable_mlflow: bool = True                 
    ):
        """
        Initialize the pipeline with MLflow configuration.
        
        Parameters
        ----------
        model_params : Dict[str, Any], optional
            Hyperparameters to customize the model.
            Any parameter not specified uses the default.
            
            Available hyperparameters:
            - n_estimators: Number of trees (default: 100)
            - max_depth: Maximum tree depth (default: 15)
            - min_samples_split: Min samples to split (default: 5)
            - min_samples_leaf: Min samples in leaf (default: 3)
            - max_features: Features per split (default: 0.7)
            - random_state: Random seed (default: 42)
            - n_jobs: CPU cores (default: -1)
        
        experiment_name : str, default="exam_score_prediction"
            Name of the MLflow experiment.
            All runs with same experiment_name are grouped together.
            
        tracking_uri : str, optional
            Where to store MLflow data.
            - None: Uses default ./mlruns folder
            - "mlruns": Local folder (good for development)
            - "http://localhost:5000": MLflow tracking server
            - "databricks": Databricks workspace
            
        enable_mlflow : bool, default=True
            Whether to enable MLflow tracking.
            Set to False for testing or when MLflow is not available.
        """
        self.model_params = DEFAULT_MODEL_HYPERPARAMETERS.copy()
        if model_params:
            self.model_params.update(model_params)
            
        # mlflow config
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enable_mlflow = enable_mlflow
        self._mlflow_run_id: Optional[str] = None
        
        # Initialize data preprocessor
        self.data_preprocessor = MLDataPreprocessor(DATA_PATH)
        
        # setup mlflow
        if self.enable_mlflow:
            self._setup_mlflow()
            
        # instance variables
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: Optional[List[str]] = None

        # copy class attributes to instance
        self.numeric_features = NUMERIC_FEATURES.copy()
        self.categorical_features = CATEGORICAL_FEATURES.copy()
        
    def _setup_mlflow(self)-> None:
        """
        Configure MLflow tracking.
        
        WHAT THIS DOES:
        1. Set tracking URI (where data is stored)
        2. Create or get experiment
        3. Set experiment as active
        
        TRACKING URI OPTIONS:
        - None or "mlruns": Local folder ./mlruns
        - "file:///path/to/folder": Specific local path
        - "http://server:5000": MLflow tracking server
        - "databricks": Databricks workspace
        """
        # set tracking uri
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            
        # create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id
            
        # set as active experiment
        mlflow.set_experiment(self.experiment_name)

    def build_pipeline(self) -> Pipeline:
        """
        Build complete sklearn Pipeline (preprocessing + model)
        
        Returns
        -------
        Pipeline
            Complete unfitted pipeline.
        """
        return Pipeline([
            ("preprocessor", self.data_preprocessor.build_columns_preprocessor()),
            ("regressor", RandomForestRegressor(**self.model_params))
        ]) # type: ignore
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str,Any]] = None
    ) -> ModelArtifact: # type: ignore
        """
        Train pipeline and return artifact
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features.
            Must contain all columns in NUMERIC_FEATURES and CATEGORICAL_FEATURES.
            
        y : pd.Series
            Training target (exam_score).
            
        run_name : str, optional
            Name for this MLflow run.
            Default: Auto-generated timestamp.
            Example: "baseline_model", "more_trees", "tuned_v1"
            
        tags : Dict[str, str], optional
            Tags to attach to the MLflow run.
            Example: {"author": "soledad", "dataset_version": "v2"}
        
        Returns
        -------
        ModelArtifact
            Complete artifact with:
            - Trained model
            - Feature configuration
            - Training metrics
            - MLflow run ID
        
        Raises
        ------
        ValueError
            If X is empty or missing required features
        """
        # =====================================================================
        # VALIDATE INPUTS
        # =====================================================================
        if X.empty:
            raise ValueError("Training data cannot be empty")
        
        # Validate features
        self._validate_features(X)
        
        # =====================================================================
        # START MLFLOW RUN
        # =====================================================================
        if self.enable_mlflow:
            # Start a new run
            mlflow.start_run(run_name=run_name)
            self._mlflow_run_id = mlflow.active_run().info.run_id # type: ignore
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
                
            # Log hyperparameters
            mlflow.log_params(self.model_params)
            
            # Log dataset info
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", len(X.columns))
            mlflow.log_param("numeric_features", self.numeric_features)
            mlflow.log_param("categorical_features", self.categorical_features)
        
        # Store feature information
        self.feature_names = list(X.columns)
        
        # =====================================================================
        # BUILD AND FIT PIPELINE
        # =====================================================================
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X,y)
        # fit() does:
        # 1. Preprocessor.fit_transform(X) - learn and apply transformations
        # 2. Regressor.fit(X_transformed, y) - train on transformed features
        
        # =====================================================================
        # CALCULATE METRICS
        # =====================================================================
        train_predictions = self.pipeline.predict(X)
        metrics = self._calculate_metrics(y, train_predictions) # type: ignore
        training_metrics = {f"train_{k}": v for k, v in metrics.items()}
        
        # =====================================================================
        # LOG TO MLFLOW
        # =====================================================================
        
        if self.enable_mlflow:
            # Log training metrics
            mlflow.log_metrics({
                "train_rmse": training_metrics["train_rmse"],
                "train_mae": training_metrics["train_mae"],
                "train_r2": training_metrics["train_r2"]
            })
            
            # Log model
            mlflow.sklearn.log_model( # type: ignore
                self.pipeline,
                "model",
                registered_model_name=None # Register in model registre
            )
            
            # Log feature importance
            self._log_feature_importance()
            
            # End run
            mlflow.end_run()
        
        # =====================================================================
        # RETURN ARTIFACT
        # =====================================================================
        return ModelArtifact(
            model=self.pipeline,
            feature_names=self.feature_names,
            numerical_features=self.numeric_features,
            categorical_features=self.categorical_features,
            training_metrics=training_metrics,
            version="1.0.0",
            mlflow_run_id=self._mlflow_run_id
        )
        
    def _log_feature_importance(self)->None:
        """
        Log feature importance to MLflow.
        
        Creates:
        1. CSV file with importance values
        2. Bar chart visualization (optional)
        """
        importance_df = self.get_feature_importance()
        
        # Create temp directories for artifacts
        with tempfile.TemporaryDirectory() as tempdir:
            # Save csv
            csv_path = os.path.join(tempdir,"feature_importance.csv")
            importance_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            
            # Log top features as metrics
            for i,row in importance_df.head(5).iterrows():
                # Clean feature name for mlflow
                clean_name = row['feature'].replace(' ','_').replace('-','_')
                mlflow.log_metric(f"importance_{clean_name}",row['importance'])

    # =========================================================================
    # PREDICTION
    # =========================================================================
    def predict(self, X: pd.DataFrame)->np.ndarray:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction.
        
        Returns
        -------
        np.ndarray
            Predicted exam scores (0-100 range typically).
        
        Raises
        ------
        RuntimeError
            If pipeline not trained.
        ValueError
            If required features missing.
        """
        if self.pipeline is None:
            raise RuntimeError(f"Pipeline not trained!!")
        
        self._validate_features(X)
        
        return self.pipeline.predict(X) # type: ignore
    
    # =========================================================================
    # EVALUATION WITH MLFLOW
    # =========================================================================
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = 'test',
    )->Dict[str, float]:
        """
        Evaluate model and log metrics to MLflow.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for evaluation.
            
        y : pd.Series
            True exam scores.
            
        dataset_name : str, default="test"
            Name prefix for metrics (e.g., "test_rmse", "validation_rmse").
        
        Returns
        -------
        Dict[str, float]
            Dictionary with rmse, mae, r2.
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not trained!!")
        
        predictions = self.predict(X)
        metrics = self._calculate_metrics(y, predictions)
        
        # Log to mlflow
        if self.enable_mlflow and self._mlflow_run_id:
            with mlflow.start_run(run_id=self._mlflow_run_id):
                mlflow.log_metrics({
                    f"{dataset_name}_rmse": metrics["rmse"],
                    f"{dataset_name}_mae": metrics["mae"],
                    f"{dataset_name}_r2": metrics["r2"]
                })
        
        return metrics
    # =========================================================================
    # METRICS CALCULATION
    # =========================================================================
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    )->Dict[str, float]:
        """
        Calculate regression metrics.
        """
        return {
            'rmse': np.sqrt(mean_squared_error(y_true,y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    # =========================================================================
    # FEATURE VALIDATION
    # =========================================================================
    def _validate_features(self, X: pd.DataFrame)->None:
        """
        Ensure input features match training configuration
        """
        missing_numeric = set(self.numeric_features) - set(X.columns)
        if missing_numeric:
            raise ValueError(
                f"Missing numeric features: {missing_numeric}."
                f"\nExpected: {self.numeric_features}"
            )
            
        missing_categorical = set(self.categorical_features) - set(X.columns)
        if missing_categorical:
            raise ValueError(
                f"Missing categorical features: {missing_categorical}"
                f"\nExpected: {self.categorical_features}"
            )
            
    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    def get_feature_importance(self)->pd.DataFrame:
        """
        Get feature importance from trained Random Forest.
        
        Returns
        -------
        pd.DataFrame
            Columns: 'feature', 'importance'
            Sorted by importance (highest first).
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot get feature importance from untrained pipeline!!")
        
        regressor = self.pipeline.named_steps['regressor']
        importances = regressor.feature_importances_
        
        # Get feature names
        feature_names = list(self.numeric_features)
        
        preprocessor = self.pipeline.named_steps['preprocessor']
        encoder = preprocessor.named_transformers_['cat']
        cat_names = encoder.get_feature_names_out(self.categorical_features)
        
        feature_names.extend(cat_names)
        
        # Create and sort dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance',ascending=False).reset_index(drop=True)
        
        return importance_df
    
    # =========================================================================
    # MODEL PERSISTENCE
    # =========================================================================
    def save(self, path: str)->None:
        """
        Save trained pipeline to disk.
        
        Parameters
        ----------
        path : str
            File path (use .joblib extension)
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save untrained pipeline!!")
        
        artifact = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'model_params': self.model_params,
            'mlflow_run_id': self._mlflow_run_id
        }
        
        joblib.dump(artifact, path)
        
    @classmethod
    def load(cls, path: str)->"ExamScorePipeline":
        """
        Load trained pipeline from disk.
        
        Parameters
        ----------
        path : str
            File path to load from.
        
        Returns
        -------
        ExamScorePipeline
            Loaded pipeline ready for predictions.
        """
        artifact = joblib.load(path)
        
        instance = cls(
            model_params=artifact['model_params'],
            enable_mlflow=False # Dont create new experiment on load
        )
        
        instance.pipeline = artifact['pipeline']
        instance.feature_names = artifact['feature_names']
        instance.numeric_features = artifact['numeric_features']
        instance.categorical_features = artifact['categorical_features']
        instance._mlflow_run_id = artifact['mlflow_run_id']
        
        return instance

    # =========================================================================
    # MLFLOW MODEL REGISTRY
    # =========================================================================
    def register_model(
        self,
        model_name: str,
        description: Optional[str] = None
    )->str:
        """
        Register trained model in MLflow Model Registry.
        
        Parameters
        ----------
        model_name : str
            Name for the registered model.
            
        description : str, optional
            Description of this model version.
        
        Returns
        -------
        str
            Model version number.
        """
        if not self.enable_mlflow or not self._mlflow_run_id:
            raise RuntimeError("MLFLow not enabled or no run exists")
        
        # Register model from logged artifact
        model_uri = f"runs:/{self._mlflow_run_id}/model"
        
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        if description:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.update_model_version(
                name=model_name,
                version=result.version,
                description=description
            )
            
        return result.description # type: ignore
