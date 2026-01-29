import math
import pandas as pd
import numpy as np

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
