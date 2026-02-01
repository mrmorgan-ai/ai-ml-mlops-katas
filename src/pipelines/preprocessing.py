import os
import json
import pandas as pd
import numpy as np

from typing import Dict, Tuple, List, Any, Optional

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ID_PATTERNS

class MLDataPreprocessor:
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
        """ 
        Remove categorical columns with cardinality above threshold.
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
            
        threshold : int, default=50
            Maximum allowed unique values for categorical columns.
            Columns with MORE than threshold are removed.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with high-cardinality categorical columns removed.
            Numeric columns are NEVER removed.
        """
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
    
    def build_columns_preprocessor(self)->ColumnTransformer:
        """
        Build the preprocessing step using ColumnTransformer.
        
        Returns
        -------
        ColumnTransformer
            Configured preprocessing pipeline
        """
        # numeric transformations
        ## During fit: learns mean and std of each column
        ## During transform: applies (x - mean) / std
        numeric_transformer = self.scaler
        
        # Categorical transformations
        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore',
            # If new category appears during prediction:
            # 'ignore': encode as zeros
            # 'error': raise exception
            
            sparse_output=False
            # False: regular numpy array (easier to debug)
            # True: sparse matrix (memory efficient)
        )
        
        # Combine with ColumnTransformer
        columns_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERIC_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ],
            remainder='drop'
            # What to do with unlisted columns:
            # 'drop': remove them (safest)
            # 'passthrough': keep unchanged
        )
        return columns_preprocessor

    def remove_id_columns(
        self,
        df: pd.DataFrame,
        id_patterns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Remove columns that appear to be identifiers.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
            
        id_patterns : List[str], optional
            Substrings that identify ID columns (case-insensitive).
            Default: ['_id', 'id_', 'student_id', 'index', 'key']
        
        Returns
        -------
        pd.DataFrame
            DataFrame with ID columns removed.
        """
        if id_patterns is None:
            id_patterns = ID_PATTERNS
        
        cols_to_keep = []
        
        for col in df.columns:
            col_lower = col.lower()  # Case-insensitive matching
            
            # Check if column name matches any ID pattern
            is_id_column = any(pattern in col_lower for pattern in id_patterns)
            
            if not is_id_column:
                cols_to_keep.append(col)
        
        return df[cols_to_keep]

    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to analyze.
        
        Returns
        -------
        pd.DataFrame
        Summary with columns:
        - column: Column name
        - missing_count: Number of missing values
        - missing_percentage: Percentage missing
        - dtype: Data type
        
        Only columns WITH missing values are included.
        Sorted by missing_percentage (highest first).
        Empty DataFrame if no missing values.
        """
        # Count missing per column
        missing_counts = df.isnull().sum()
        
        # Filter to only columns with missing values
        missing_counts = missing_counts[missing_counts > 0]
        
        if len(missing_counts) == 0:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=['column', 'missing_count', 'missing_percentage', 'dtype']
            )
        
        # Build summary
        summary = pd.DataFrame({
            'column': missing_counts.index,
            'missing_count': missing_counts.values,
            'missing_percentage': (missing_counts.values / len(df)) * 100, # type: ignore
            'dtype': [str(df[col].dtype) for col in missing_counts.index]
        })
        
        # Sort by percentage (worst first)
        summary = summary.sort_values('missing_percentage', ascending=False)
        summary = summary.reset_index(drop=True)
        
        return summary

    def drop_high_missing_columns(
        self,
        df: pd.DataFrame,
        threshold: float = 30.0
    ) -> pd.DataFrame:
        """
        Remove columns with missing value percentage above threshold.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
            
        threshold : float, default=30.0
            Maximum allowed missing percentage.
            Columns with MORE than threshold are removed.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with high-missing columns removed.
        """
        # Calculate missing percentage per column
        missing_pct = (df.isnull().sum() / len(df)) * 100
        
        # Keep columns at or below threshold
        cols_to_keep = missing_pct[missing_pct <= threshold].index.tolist()
        
        return df[cols_to_keep]

    def validate_numeric_columns(
        self,
        df: pd.DataFrame,
        expected_numeric: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Check that specified columns are numeric.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
            
        expected_numeric : List[str]
            Column names that SHOULD be numeric.
        
        Returns
        -------
        Tuple[bool, List[str]]
            - is_valid: True if all columns are numeric
            - invalid_columns: List of columns that are NOT numeric
        """
        invalid_columns = []
        
        for col in expected_numeric:
            if col not in df.columns:
                invalid_columns.append(col)  # Column doesn't exist
            elif not pd.api.types.is_numeric_dtype(df[col]):
                invalid_columns.append(col)  # Exists but not numeric
        
        is_valid = len(invalid_columns) == 0
        return is_valid, invalid_columns

    def validate_categorical_columns(
        self,
        df: pd.DataFrame,
        expected_categorical: List[str],
        allowed_values: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[bool, Dict[str, dict]]:
        """
        Check that categorical columns contain expected values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
            
        expected_categorical : List[str]
            Column names that should be categorical.
            
        allowed_values : Dict[str, List[str]], optional
            Mapping of column names to allowed values.
            Example: {'gender': ['male', 'female', 'other']}
        
        Returns
        -------
        Tuple[bool, Dict[str, dict]]
            - is_valid: True if all validations pass
            - issues: Dictionary with details about problems
        """
        issues = {}
        
        for col in expected_categorical:
            if col not in df.columns:
                issues[col] = {'error': 'column_not_found'}
                continue
            
            if allowed_values and col in allowed_values:
                actual_values = set(df[col].dropna().unique())
                allowed_set = set(allowed_values[col])
                unexpected = actual_values - allowed_set
                
                if unexpected:
                    issues[col] = {'unexpected_values': list(unexpected)}
        
        is_valid = len(issues) == 0
        return is_valid, issues

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
    
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

