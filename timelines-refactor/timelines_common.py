import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gaussian_kde, norm, rankdata
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_external_data(csv_path: str = "../external/headline.csv") -> pd.DataFrame:
    """Load external benchmark data points for overlay on plots."""
    if not Path(csv_path).exists():
        print(f"Warning: External data file {csv_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Parse release_date to decimal year format (matching our timeline plots)
    # Handle missing dates (like for 'human' row)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year_decimal'] = df['release_year'].dt.year + (df['release_year'].dt.month - 1) / 12 + (df['release_year'].dt.day - 1) / 365
    
    # Remove rows without valid dates
    df = df.dropna(subset=['release_year_decimal'])
    
    print(f"Loaded {len(df)} external data points from {csv_path}")
    return df

def get_lognormal_from_80_ci(lower_bound, upper_bound):
    """Generate a lognormal distribution from 80% confidence interval."""
    # Convert to natural log space
    ln_lower = np.log(lower_bound)
    ln_upper = np.log(upper_bound)
    
    # Z-scores for 10th and 90th percentiles
    z_low = -1.28  # norm.ppf(0.1)
    z_high = 1.28  # norm.ppf(0.9)
    
    # Calculate mu and sigma in log space
    sigma = (ln_upper - ln_lower) / (z_high - z_low)
    mu = (ln_upper + ln_lower) / 2
    
    return lognorm(s=sigma, scale=np.exp(mu))

def format_year_month(year_decimal: float, max_time: float = 2050.0) -> str:
    """Convert decimal year to Month Year format."""
    if year_decimal >= max_time:
        return f">{int(max_time)}"
        
    year = int(year_decimal)
    month = int((year_decimal % 1) * 12) + 1
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
    return f"{month_name} {year}"

# Helper function to interpolate horizon from mapping
def get_horizon_at_progress(mapping, progress_months):
    """Get horizon at given progress using linear interpolation."""

    # Find the appropriate time point in the mapping
    times = [t for t, h in mapping]
    horizons = [h for t, h in mapping]
    
    if progress_months <= times[0]:
        return horizons[0]
    elif progress_months >= times[-1]:
        return horizons[-1]
    else:
        # Linear interpolation
        for i in range(len(times) - 1):
            if times[i] <= progress_months <= times[i + 1]:
                t1, t2 = times[i], times[i + 1]
                h1, h2 = horizons[i], horizons[i + 1]
                # Linear interpolation
                ratio = (progress_months - t1) / (t2 - t1)
                return h1 + ratio * (h2 - h1)
    
    return horizons[-1]  # Fallback
