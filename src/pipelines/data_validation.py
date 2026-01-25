import pandera as pa
from pandera.pandas import Column, Check, DataFrameSchema
import pandas as pd
import numpy as np

exam_data_schema = DataFrameSchema(
    # NUMERIC COLUMNS
    columns={
        "age": Column(
            int,
            checks=[
                Check.ge(15, error="Age cannot be negative"),
                Check.le(30, error="Age cannot be more than 24 hours")
            ],
            nullable=False,
            coerce=True
        ),
        "study_hours": Column(
            float,
            checks=[
                Check.ge(0, error="Study hours cannot be negative"),
                Check.le(24, error="Study hours cannot be more than 24 hours")
            ],
            nullable=False,
            coerce=True
        ),
        "class_attendance": Column(
            float,
            checks=[
                Check.ge(0, error="Class attendance cannot be negative"),
                Check.le(100, error="Class attendance cannot be more than 100%")
            ],
            nullable=False,
            coerce=True
        ),
        "sleep_hours": Column(
            float,
            checks=[
                Check.ge(0, error="Sleep hours cannot be negative"),
                Check.le(24, error="Sleep hours cannot be more than 24 hours")
            ],
            nullable=False,
            coerce=True
        ),
        # CATEGORICAL COLUMNS
        "gender": Column(
            str,
            checks=[
                Check.isin(
                    ["male","female","other"],
                    error="Gender must be male, female or other"
                )
            ],
            nullable=False
        ),
        "course": Column(
            str,
            checks=[
                Check.isin(
                    ["diploma", "bca", "b.sc", "b.tech", "bba", "ba", "b.com"],
                    error="Invalid course value"
                ),
            ],
            nullable=False,
        ),
        "internet_access": Column(
            str,
            checks=[
                Check.isin(["yes", "no"]),
            ],
            nullable=False,
        ),
        "sleep_quality": Column(
            str,
            checks=[
                Check.isin(["poor", "average", "good"]),
            ],
            nullable=False,
        ),
        "study_method": Column(
            str,
            checks=[
                Check.isin([
                    "coaching", "online videos", "self-study", 
                    "group study", "mixed"
                ]),
            ],
            nullable=False,
        ),
        "facility_rating": Column(
            str,
            checks=[
                Check.isin(["low", "medium", "high"]),
            ],
            nullable=False,
        ),
        "exam_difficulty": Column(
            object,
            checks=[
                Check.isin(["hard", "moderate", "easy"])
            ],
            nullable=False
        ),
        # TARGET COLUMN
        "exam_score": Column(
            float,
            checks=[
                Check.ge(0, error="Exam score cannot be negative"),
                Check.le(100, error="Exam error's max value is 100")
            ],
            nullable=False,
            coerce=True
        )
    },
    strict = False,
    coerce = True
)

def validate_exam_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate exam data against the schema.
    
    Parameters
    df : pd.DataFrame
        Raw exam data to validate.
    
    Returns
    pd.DataFrame
        Validated DataFrame with coerced types.
    
    Raises
    pandera.errors.SchemaError
        If any validation check fails.
    """
    return exam_data_schema.validate(df)

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
