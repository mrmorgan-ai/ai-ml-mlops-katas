import json
import sys
import pandas as pd
import numpy as np
import argparse
from scipy import stats

from pipelines.preprocessing import MLDataPreprocessor

from config.config import DATA_PATH

def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10
)-> float:
    """
    Calculate Population Stability Index between two distributions.
    
    Parameters
    reference : pd.Series
        Historical/training distribution (baseline).
    current : pd.Series
        New/production distribution.
    n_bins : int, default=10
        Number of bins for discretizing.
    
    Returns
    float
        PSI value
    """
    # Handle edge cases
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    
    # Create bins
    _, bin_edges = np.histogram(reference, bins=n_bins)
    
    # Extend edges to handle values outside reference range
    bin_edges[0] = -np.inf # All low values
    bin_edges[-1] = np.inf # All high values
    
    # Count values in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert percentages
    ref_pct = ref_counts / len(reference)
    curr_pct = curr_counts / len(current)
    
    # Avoid 0 division
    ref_pct = np.maximum(ref_pct,0.0001)
    curr_pct = np.maximum(curr_pct, 0.0001)
    
    # Calculate psi
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct/ref_pct))   
    
    return psi

# For statistical validation (data drift detection)
def check_numerical_drift_by_column(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = 0.1
) -> dict:
    """Compare distributions using PSI (Population Stability Index).
    
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate change, investigate
    PSI >= 0.25: Significant change, retrain
    """
    # Simplified drift calculation (Z score)
    ref_mean, ref_std = reference.mean(), reference.std()
    curr_mean, curr_std = current.mean(), current.std()
    
    mean_shift = abs(curr_mean - ref_mean) / (ref_std + 1e-10)
    
    # Calculate psi
    psi = calculate_psi(current=current, reference=reference)
    return {
        "mean_shift_zscore": mean_shift,
        "psi": psi,
        "drift_detected_zcore": mean_shift > threshold,
        "drift_detected_psi": psi > threshold,
        "reference_mean": ref_mean,
        "current_mean": curr_mean
    }

def check_categorical_drif(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    categorical_columns: list,
    pvalue_threshold: float = 0.05
)-> dict:
    """
    Check for drift in categorical features using Chi-squared test.
    
    Parameters
    ----------
    reference_df : pd.DataFrame
        Training/historical data.
    current_df : pd.DataFrame
        New/production data.
    categorical_columns : list
        Columns to check.
    pvalue_threshold : float
        P-value below which drift is flagged.
    
    Returns
    -------
    dict
        Drift analysis for each column.    
    """
    drift_report = {}
    
    for col in categorical_columns:
        # Get value counts
        ref_counts = reference_df[col].value_counts()
        curr_counts = current_df[col].value_counts()
        
        # Handle new/missing categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_counts = ref_counts.reindex(all_categories, fill_value=0)
        curr_counts = curr_counts.reindex(all_categories, fill_value=0)
    
        # Chi-Square Test
        try:
            # Expect counts = references proportions x current total
            # References proportions = values / values.sum()
            expected = ref_counts.values * (curr_counts.sum() / ref_counts.sum())
            
            chi2_stat, p_value = stats.chisquare(
                f_obs = curr_counts.values,
                f_exp = expected
            )
        except:
            # Handle edge cases
            chi2_stat, p_value = 0.0, 1.0
        
        drift_report[col] = {
            'chi2_statistics': chi2_stat,
            'p_value': p_value,
            'drift_detected': p_value < pvalue_threshold,
            'reference_distribution': (ref_counts / ref_counts.sum()).to_dict(),
            'current_distribution': (curr_counts / curr_counts.sum()).to_dict()
        }
    return drift_report

def main():
    parser = argparse.ArgumentParser(description="Check data drift")
    parser.add_argument("--reference", required=True, help="Reference data json path")
    parser.add_argument("--current", required=True, help="Current data csv path")
    parser.add_argument("--threshold", type=float, default=0.25, help="psi threshold")
    args = parser.parse_args()
    
    print("Data drift detection!!")
   
    with open(args.reference) as f:
       reference_stats = json.load(f)
       
    # load current data
    data_loader = MLDataPreprocessor(DATA_PATH)
    current_df = data_loader.load_data()
    
    numeric_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_cols = ['gender', 'course', 'study_method']
    
    drift_detected = False
    
    # Check numerical drift
    print("Checking numerical drift ...")
    for col in numeric_cols:
        drift_stats = check_numerical_drift_by_column(
            current=current_df[col],
            reference=pd.Series(reference_stats.get(col)),
            threshold=args.threshold
        )
        
        is_drift = drift_stats.get("drift_detected_psi",0.0)
        psi = drift_stats.get("psi")
                
        status = "DRIFT" if is_drift else "OK"
        print(f"{col}: psi={psi:2f} [{status}]")
        
        if is_drift:
            drift_detected = True
    
    print("SUMMARY")
    if drift_detected:
        print("DRIFT DETECTED - Analyze deeper")
        sys.exit(1)
        
    else:
        print("No significant drift detected")
        sys.exit(0)

if __name__ == "__main__":
    main()
