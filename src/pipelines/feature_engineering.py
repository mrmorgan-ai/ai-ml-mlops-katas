import pandas as pd
import numpy as np

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_study_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features related to study habits.
    
    FEATURES CREATED:
    1. study_intensity
       - Formula: study_hours / 8
       - Meaning: How close to maximum study hours (assumed 8)
       - Range: 0 to 1
       - Interpretation: 0.5 = half of maximum effort
    
    2. attendance_study_interaction
       - Formula: (class_attendance * study_hours) / 100
       - Meaning: Combined effect of attendance AND study
       - Intuition: Student who attends AND studies does best
    
    3. sleep_study_balance
       - Formula: sleep_hours / (study_hours + 0.1)
       - Meaning: Ratio of sleep to study
       - High value: Sleeping more than studying
       - Low value: Studying more than sleeping (burnout risk?)
       - Note: +0.1 prevents division by zero
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns:
        - study_hours
        - class_attendance
        - sleep_hours
    
    Returns
    -------
    pd.DataFrame
        DataFrame with original columns PLUS new features.
        Original DataFrame is NOT modified.
    """
    # Create copy to avoid modifying original
    result = df.copy()
    
    # Study intensity: normalized study hours (0-1 scale)
    result['study_intensity'] = result['study_hours'] / 8.0
    
    # Interaction: students who attend AND study perform best
    result['attendance_study_interaction'] = (
        result['class_attendance'] * result['study_hours'] / 100
    )
    
    # Balance: ratio of sleep to study
    # Add 0.1 to avoid division by zero when study_hours=0
    result['sleep_study_balance'] = (
        result['sleep_hours'] / (result['study_hours'] + 0.1)
    )
    
    return result
