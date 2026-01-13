import pandera as pa
from pandera.pandas import Column, Check, DataFrameSchema
import pandas as pd
import numpy as np

transaction_schema = DataFrameSchema(
    columns={
        "age": Column(
            int,
            checks=[
                Check.ge(0, error="Age cannot be negative"),
                Check.le(130, error="Age cannot be more than 130")
            ],
            nullable=True
        ),
        "exam_score": Column(
            float,
            checks=[
                Check.ge(0, error="Exam score cannot be negative"),
                Check.le(100, error="Exam error's max value is 100")
            ],
            nullable=False
        ),
        "exam_difficulty": Column(
            object,
            checks=[
                Check.isin(["hard", "moderate", "easy"])
            ]
        )
    },
    strict = True,
    coerce = True
)

def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate transaction data against schema.
    
    Raises:
        pandera.errors.SchemaError: If validation fails
    
    Returns:
        Validated DataFrame with coerced types
    """
    return transaction_schema.validate(df)

# For statistical validation (data drift detection)
def check_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = 0.1
) -> dict:
    """Compare distributions using PSI (Population Stability Index).
    
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate change, investigate
    PSI >= 0.25: Significant change, retrain
    """
    # Simplified PSI calculation
    ref_mean, ref_std = reference.mean(), reference.std()
    curr_mean, curr_std = current.mean(), current.std()
    
    mean_shift = abs(curr_mean - ref_mean) / (ref_std + 1e-10)
    
    return {
        "mean_shift_zscore": mean_shift,
        "drift_detected": mean_shift > threshold,
        "reference_mean": ref_mean,
        "current_mean": curr_mean
    }
