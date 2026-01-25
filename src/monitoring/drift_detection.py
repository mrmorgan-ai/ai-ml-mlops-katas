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
    
    
    return psi
