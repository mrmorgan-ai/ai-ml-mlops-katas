import math
import pandas as pd
import numpy as np
from scipy import stats

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
    sorted_reference = reference.sort_values(ascending=True)
    sorted_current = current.sort_values(ascending=True)
    
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
        # Expect counts = references proportions x current total
        # References proportions = values / values.sum()
        expected = ref_counts.values * (curr_counts.sum() / ref_counts.sum())
        
        chi2_stat, p_value = stats.chisquare(
            f_obs = curr_counts.values,
            f_exp = expected
        )
        
        drift_report[col] = {
            'chi2_statistics': chi2_stat,
            'p_value': p_value,
            'drift_detected': p_value < pvalue_threshold,
            'reference_distribution': (ref_counts / ref_counts.sum()).to_dict(),
            'current_distribution': (curr_counts / curr_counts.sum()).to_dict()
        }
    return drift_report
