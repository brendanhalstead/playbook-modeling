import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import curve_fit

def convert_to_2024_years(df):
    """Convert calendar dates to 2024-years based on progress multiplier adjustments"""
    # Claude 3.7 Sonnet as reference point (Feb 2025 = 0 in 2024-years)
    reference_date = datetime(2025, 2, 1)
    reference_progress_mult = None
    
    # Find Claude 3.7 Sonnet progress multiplier
    for idx, row in df.iterrows():
        if 'Claude 3.7 Sonnet' in str(row['Column 1']):
            reference_progress_mult = row['Progress multiplier']
            break
    
    if reference_progress_mult is None:
        reference_progress_mult = 1.099328839  # Default from data
    
    # Parse dates and calculate 2024-years
    model_2024_years = []
    parsed_dates = []
    
    for idx, row in df.iterrows():
        try:
            # Parse date
            month, year = row['Date'].split('/')
            date = datetime(int(year), int(month), 1)
            parsed_dates.append(date)
            
            # Calculate calendar years from reference
            calendar_years = (date.year - reference_date.year) + (date.month - reference_date.month) / 12.0
            
            # For dates before Claude 3.7 Sonnet, adjust based on progress multiplier
            if date < reference_date:
                # Get progress multiplier for this model
                progress_mult = row['Progress multiplier']
                
                # Adjust the time based on progress multiplier ratio
                # If progress was slower (lower multiplier), the same "work" took more calendar time
                # but represents less "2024-years" of equivalent progress
                adjustment_factor = progress_mult / reference_progress_mult
                adjusted_years = calendar_years * adjustment_factor
                model_2024_years.append(adjusted_years)
            else:
                # For dates after Claude 3.7 Sonnet, use calendar years directly
                model_2024_years.append(calendar_years)
                
        except:
            model_2024_years.append(None)
            parsed_dates.append(None)
    
    return model_2024_years, parsed_dates

def create_interpolation_model():
    """Create interpolation model based on Claude 3.7 Sonnet baseline"""
    # Claude 3.7 Sonnet baseline (Feb 2025): progress multiplier = 1.1
    baseline_progress_multiplier = 1.1
    baseline_date = datetime(2025, 2, 1)
    
    # 70 months later: progress multiplier = 8.5
    future_progress_multiplier = 8.5
    future_months = 70
    
    # Calculate monthly multiplier on (progress multiplier - 1)
    # (progress_multiplier - 1) = baseline_excess * r^months
    # where baseline_excess = 1.1 - 1 = 0.1
    # and future_excess = 8.5 - 1 = 7.5
    baseline_excess = baseline_progress_multiplier - 1  # 0.1
    future_excess = future_progress_multiplier - 1      # 7.5
    
    # r^70 = future_excess / baseline_excess = 7.5 / 0.1 = 75
    monthly_multiplier = (future_excess / baseline_excess) ** (1/future_months)
    
    print(f"Monthly multiplier (r): {monthly_multiplier:.6f}")
    print(f"This means (progress_multiplier - 1) grows by {(monthly_multiplier-1)*100:.4f}% per month")
    
    def interpolate_progress_multiplier(months_from_baseline):
        """Calculate progress multiplier at given months from baseline"""
        excess = baseline_excess * (monthly_multiplier ** months_from_baseline)
        return excess + 1
    
    return interpolate_progress_multiplier, baseline_date, monthly_multiplier

def create_gpt3_to_claude37_interpolation(df):
    """Create interpolation model based on GPT-3 to Claude 3.7 Sonnet progression"""
    # Find GPT-3 and Claude 3.7 Sonnet data points
    gpt3_progress_mult = None
    claude37_progress_mult = None
    gpt3_date = None
    claude37_date = None
    
    for idx, row in df.iterrows():
        try:
            model_name = row['Column 1']
            if 'GPT-3' in model_name and 'GPT-3.5' not in model_name:
                gpt3_progress_mult = row['Progress multiplier']
                month, year = row['Date'].split('/')
                gpt3_date = datetime(int(year), int(month), 1)
            elif 'Claude 3.7 Sonnet' in model_name:
                claude37_progress_mult = row['Progress multiplier']
                month, year = row['Date'].split('/')
                claude37_date = datetime(int(year), int(month), 1)
        except:
            continue
    
    if gpt3_progress_mult is None or claude37_progress_mult is None:
        print("Could not find GPT-3 or Claude 3.7 Sonnet data points")
        return None, None, None
    
    # Calculate months between GPT-3 and Claude 3.7 Sonnet
    months_between = (claude37_date.year - gpt3_date.year) * 12 + (claude37_date.month - gpt3_date.month)
    
    # Calculate monthly multiplier on (progress multiplier - 1)
    gpt3_excess = gpt3_progress_mult - 1
    claude37_excess = claude37_progress_mult - 1
    
    # r^months_between = claude37_excess / gpt3_excess
    monthly_multiplier = (claude37_excess / gpt3_excess) ** (1/months_between)
    
    print(f"\nGPT-3 to Claude 3.7 Interpolation:")
    print(f"GPT-3 ({gpt3_date.strftime('%b %Y')}): progress multiplier = {gpt3_progress_mult:.3f}")
    print(f"Claude 3.7 Sonnet ({claude37_date.strftime('%b %Y')}): progress multiplier = {claude37_progress_mult:.3f}")
    print(f"Months between: {months_between}")
    print(f"Monthly multiplier (r): {monthly_multiplier:.6f}")
    print(f"Monthly growth rate: {(monthly_multiplier-1)*100:.4f}%")
    
    def interpolate_from_gpt3(months_from_gpt3):
        """Calculate progress multiplier at given months from GPT-3"""
        excess = gpt3_excess * (monthly_multiplier ** months_from_gpt3)
        return excess + 1
    
    return interpolate_from_gpt3, gpt3_date, monthly_multiplier

def double_exponential(x, a, b, c):
    """Double exponential function: y = a * exp(b * exp(c * x))"""
    try:
        # Prevent overflow by clamping the inner exponential
        inner_exp = np.clip(c * x, -50, 50)  # Prevent extreme values
        outer_exp = np.clip(b * np.exp(inner_exp), -50, 50)  # Prevent extreme values
        return a * np.exp(outer_exp)
    except:
        # If any calculation fails, return a reasonable default
        return np.full_like(x, 1.0)

def fit_double_exponential_to_data_original(df):
    """Fit a double exponential to the original progress multiplier data (without target point)"""
    # Convert to 2024-years for each data point
    model_2024_years_orig, _ = convert_to_2024_years(df)
    
    # Parse data
    years_2024 = []
    progress_multiplier_apr2025 = []
    model_names = []
    
    for idx, row in df.iterrows():
        try:
            if model_2024_years_orig[idx] is not None:
                years_2024.append(model_2024_years_orig[idx])
                # Get progress multiplier w/Apr 2025 as 1
                progress_multiplier_apr2025.append(row['Progress multiplier w/Apr 2025 as 1'])
                model_names.append(row['Column 1'])
        except:
            continue
    
    # Convert to numpy arrays (NO target point added)
    x_data = np.array(years_2024)
    y_data = np.array(progress_multiplier_apr2025)
    
    # Fit double exponential
    # Initial guess for parameters [a, b, c]
    # We expect the function to grow, so let's start with reasonable values
    initial_guess = [0.9, 0.001, 0.01]
    
    try:
        # Set bounds for parameters to help with convergence
        # Lower bounds: [a_min, b_min, c_min]
        # Upper bounds: [a_max, b_max, c_max]
        lower_bounds = [0.01, -10, 0.001]
        upper_bounds = [2.0, 10, 0.5]
        
        # Fit the double exponential
        popt, pcov = curve_fit(double_exponential, x_data, y_data, 
                              p0=initial_guess, bounds=(lower_bounds, upper_bounds), 
                              maxfev=10000)
        
        # Calculate R-squared
        y_pred = double_exponential(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\nDouble Exponential Fit Results (Original Data Only):")
        print(f"Parameters: a={popt[0]:.6f}, b={popt[1]:.6f}, c={popt[2]:.6f}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Data points used: {len(x_data)} (historical data only)")
        
        # Print residuals for debugging
        y_pred = double_exponential(x_data, *popt)
        print(f"\nOriginal Fit Residuals:")
        for i, model in enumerate(model_names):
            residual = y_data[i] - y_pred[i]
            print(f"{model}: actual={y_data[i]:.4f}, predicted={y_pred[i]:.4f}, residual={residual:.4f}")
        
        print(f"Mean absolute error: {np.mean(np.abs(y_data - y_pred)):.6f}")
        print(f"Max absolute error: {np.max(np.abs(y_data - y_pred)):.6f}")
        
        return popt, x_data, y_data, model_names, r_squared
        
    except Exception as e:
        print(f"Original double exponential fitting failed: {e}")
        # Try a simpler exponential fit as fallback
        try:
            print("Trying simple exponential fit as fallback...")
            def simple_exp(x, a, b):
                return a * np.exp(b * x)
            
            popt_simple, _ = curve_fit(simple_exp, x_data, y_data, maxfev=10000)
            y_pred = simple_exp(x_data, *popt_simple)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"Simple exponential fit: a={popt_simple[0]:.6f}, b={popt_simple[1]:.6f}")
            print(f"R-squared: {r_squared:.4f}")
            
            # Convert to double exponential format for consistency
            popt_converted = [popt_simple[0], popt_simple[1], 0.0]
            return popt_converted, x_data, y_data, model_names, r_squared
        except:
            return None, x_data, y_data, model_names, 0

def fit_double_exponential_to_data(df):
    """Fit a double exponential to the progress multiplier data"""
    # Convert to 2024-years for each data point
    model_2024_years_orig, _ = convert_to_2024_years(df)
    
    # Parse data
    years_2024 = []
    progress_multiplier_apr2025 = []
    model_names = []
    
    for idx, row in df.iterrows():
        try:
            if model_2024_years_orig[idx] is not None:
                years_2024.append(model_2024_years_orig[idx])
                # Get progress multiplier w/Apr 2025 as 1
                progress_multiplier_apr2025.append(row['Progress multiplier w/Apr 2025 as 1'])
                model_names.append(row['Column 1'])
        except:
            continue
    
    # Add the 60-month future point: 60 months after Claude 3.7 Sonnet (Feb 2025) with value 8.5
    # 60 months = 5 years in 2024-years
    target_2024_years = 5.0
    
    years_2024.append(target_2024_years)
    progress_multiplier_apr2025.append(8.5)
    model_names.append("Target Point (60m after Claude 3.7)")
    
    # Convert to numpy arrays
    x_data = np.array(years_2024)
    y_data = np.array(progress_multiplier_apr2025)
    
    # Fit double exponential
    # Initial guess for parameters [a, b, c]
    # We expect the function to grow, so let's start with reasonable values
    initial_guess = [0.9, 0.001, 0.01]
    
    try:
        # Set bounds for parameters to help with convergence
        # Lower bounds: [a_min, b_min, c_min]
        # Upper bounds: [a_max, b_max, c_max]
        lower_bounds = [0.01, -10, 0.001]
        upper_bounds = [2.0, 10, 0.5]
        
        # Fit the double exponential
        popt, pcov = curve_fit(double_exponential, x_data, y_data, 
                              p0=initial_guess, bounds=(lower_bounds, upper_bounds), 
                              maxfev=10000)
        
        # Calculate R-squared
        y_pred = double_exponential(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate R-squared for historical points only (excluding target point)
        historical_mask = np.array([i for i, model in enumerate(model_names) if "Target Point" not in model])
        if len(historical_mask) > 0:
            x_hist = x_data[historical_mask]
            y_hist = y_data[historical_mask]
            y_pred_hist = double_exponential(x_hist, *popt)
            ss_res_hist = np.sum((y_hist - y_pred_hist) ** 2)
            ss_tot_hist = np.sum((y_hist - np.mean(y_hist)) ** 2)
            r_squared_hist = 1 - (ss_res_hist / ss_tot_hist)
        else:
            r_squared_hist = 0
        
        print(f"\nDouble Exponential Fit Results (With Target Point):")
        print(f"Parameters: a={popt[0]:.6f}, b={popt[1]:.6f}, c={popt[2]:.6f}")
        print(f"R-squared (all points): {r_squared:.4f}")
        print(f"R-squared (historical only): {r_squared_hist:.4f}")
        print(f"Data points used: {len(x_data)} (including target point)")
        
        # Print residuals for debugging
        print(f"\nResiduals Analysis:")
        for i, model in enumerate(model_names):
            residual = y_data[i] - y_pred[i]
            print(f"{model}: actual={y_data[i]:.4f}, predicted={y_pred[i]:.4f}, residual={residual:.4f}")
        
        # Calculate weighted errors (since target point has much higher value)
        print(f"\nError Analysis:")
        print(f"Mean absolute error (all): {np.mean(np.abs(y_data - y_pred)):.6f}")
        print(f"Mean absolute error (historical): {np.mean(np.abs(y_hist - y_pred_hist)):.6f}")
        print(f"Max absolute error: {np.max(np.abs(y_data - y_pred)):.6f}")
        
        # Show the influence of the target point
        target_mask = np.array([i for i, model in enumerate(model_names) if "Target Point" in model])
        if len(target_mask) > 0:
            target_residual = y_data[target_mask[0]] - y_pred[target_mask[0]]
            print(f"Target point residual: {target_residual:.6f}")
            print(f"Target point dominates R² calculation due to high value ({y_data[target_mask[0]]:.1f} vs ~1.0 for historical)")
        
        return popt, x_data, y_data, model_names, r_squared, r_squared_hist
        
    except Exception as e:
        print(f"Double exponential fitting failed: {e}")
        # Try a simpler exponential fit as fallback
        try:
            print("Trying simple exponential fit as fallback...")
            def simple_exp(x, a, b):
                return a * np.exp(b * x)
            
            popt_simple, _ = curve_fit(simple_exp, x_data, y_data, maxfev=10000)
            y_pred = simple_exp(x_data, *popt_simple)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"Simple exponential fit: a={popt_simple[0]:.6f}, b={popt_simple[1]:.6f}")
            print(f"R-squared: {r_squared:.4f}")
            
            # Convert to double exponential format for consistency
            popt_converted = [popt_simple[0], popt_simple[1], 0.0]
            return popt_converted, x_data, y_data, model_names, r_squared, 0
        except:
            return None, x_data, y_data, model_names, 0, 0

def fit_double_exponential_to_data_months_since_gpt2(df):
    """Fit a double exponential to the progress multiplier data using 2024-months since GPT-2 (historical data only)"""
    # Convert to 2024-years for each data point
    model_2024_years_orig, parsed_dates = convert_to_2024_years(df)
    
    # Find GPT-2 date and its 2024-years value
    gpt2_date = None
    gpt2_2024_years = None
    reference_date = datetime(2025, 2, 1)  # Claude 3.7 Sonnet reference
    
    for idx, row in df.iterrows():
        try:
            if 'GPT-2' in str(row['Column 1']):
                month, year = row['Date'].split('/')
                gpt2_date = datetime(int(year), int(month), 1)
                gpt2_2024_years = model_2024_years_orig[idx]
                break
        except:
            continue
    
    if gpt2_date is None or gpt2_2024_years is None:
        print("Could not find GPT-2 data point")
        return None, None, None, None, 0
    
    # Parse data and convert to 2024-months since GPT-2 (NO target point added)
    months_since_gpt2 = []
    progress_multiplier_apr2025 = []
    model_names = []
    
    for idx, row in df.iterrows():
        try:
            if model_2024_years_orig[idx] is not None:
                # Convert 2024-years to 2024-months since GPT-2
                months_since_gpt2_val = (model_2024_years_orig[idx] - gpt2_2024_years) * 12
                months_since_gpt2.append(months_since_gpt2_val)
                # Get progress multiplier w/Apr 2025 as 1
                progress_multiplier_apr2025.append(row['Progress multiplier w/Apr 2025 as 1'])
                model_names.append(row['Column 1'])
        except:
            continue
    
    # Convert to numpy arrays (NO target point added)
    x_data = np.array(months_since_gpt2)
    y_data = np.array(progress_multiplier_apr2025)
    
    # Fit double exponential
    # Initial guess for parameters [a, b, c]
    # We expect the function to grow, so let's start with reasonable values
    initial_guess = [0.9, 0.001, 0.001]  # Smaller c since we're using months
    
    try:
        # Set bounds for parameters to help with convergence
        # Lower bounds: [a_min, b_min, c_min]
        # Upper bounds: [a_max, b_max, c_max]
        lower_bounds = [0.01, -10, 0.0001]  # Smaller c bounds for months
        upper_bounds = [2.0, 10, 0.1]       # Smaller c bounds for months
        
        # Fit the double exponential
        popt, pcov = curve_fit(double_exponential, x_data, y_data, 
                              p0=initial_guess, bounds=(lower_bounds, upper_bounds), 
                              maxfev=10000)
        
        # Calculate R-squared
        y_pred = double_exponential(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\nDouble Exponential Fit Results (2024-months since GPT-2, historical data only):")
        print(f"Parameters: a={popt[0]:.6f}, b={popt[1]:.6f}, c={popt[2]:.6f}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Data points used: {len(x_data)} (historical data only)")
        print(f"GPT-2 date: {gpt2_date.strftime('%b %Y')}")
        print(f"GPT-2 is at 0 months in this scale")
        
        # Print residuals for debugging
        print(f"\nResiduals Analysis (2024-months since GPT-2):")
        for i, model in enumerate(model_names):
            residual = y_data[i] - y_pred[i]
            print(f"{model}: actual={y_data[i]:.4f}, predicted={y_pred[i]:.4f}, residual={residual:.4f}, months={x_data[i]:.1f}")
        
        print(f"Mean absolute error: {np.mean(np.abs(y_data - y_pred)):.6f}")
        print(f"Max absolute error: {np.max(np.abs(y_data - y_pred)):.6f}")
        
        return popt, x_data, y_data, model_names, r_squared
        
    except Exception as e:
        print(f"Double exponential fitting (months since GPT-2) failed: {e}")
        # Try a simpler exponential fit as fallback
        try:
            print("Trying simple exponential fit as fallback...")
            def simple_exp(x, a, b):
                return a * np.exp(b * x)
            
            popt_simple, _ = curve_fit(simple_exp, x_data, y_data, maxfev=10000)
            y_pred = simple_exp(x_data, *popt_simple)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"Simple exponential fit: a={popt_simple[0]:.6f}, b={popt_simple[1]:.6f}")
            print(f"R-squared: {r_squared:.4f}")
            
            # Convert to double exponential format for consistency
            popt_converted = [popt_simple[0], popt_simple[1], 0.0]
            return popt_converted, x_data, y_data, model_names, r_squared
        except:
            return None, x_data, y_data, model_names, 0

def load_and_plot_aird_estimates():
    """Load AIRD estimates CSV and plot progress multiplier vs. 80% time horizon"""
    
    # Read the CSV file
    df = pd.read_csv('aird_estimates.csv')
    
    # Remove rows where all values are NaN or empty
    df = df.dropna(how='all')
    
    # Also remove rows where the essential columns are empty
    df = df.dropna(subset=['80% time horizon', 'Progress multiplier'])
    
    # Print column names to debug
    print("Available columns:", df.columns.tolist())
    
    # Extract the columns we need
    time_horizon = df['80% time horizon'].astype(float)
    progress_multiplier = df['Progress multiplier'].astype(float)
    model_names = df['Column 1']
    dates = df['Date']
    
    # Parse dates - convert from "MM/YYYY" format to datetime
    parsed_dates = []
    for date_str in dates:
        try:
            # Split "MM/YYYY" format
            month, year = date_str.split('/')
            # Create datetime object (using day 1 as default)
            parsed_date = datetime(int(year), int(month), 1)
            parsed_dates.append(parsed_date)
        except:
            parsed_dates.append(None)
    
    # Convert dates to 2024-years
    model_2024_years, parsed_dates_2024 = convert_to_2024_years(df)
    
    # Create interpolation model
    interpolate_fn, baseline_date, monthly_multiplier = create_interpolation_model()
    
    # Create interpolation data from GPT-2 (Feb 2019) to 60 months after Claude 3.7 Sonnet
    gpt2_date = datetime(2019, 2, 1)  # GPT-2 release date
    end_date = baseline_date + relativedelta(months=60)  # 60 months after Claude 3.7 Sonnet
    
    # Create monthly date range and convert to 2024-years
    interpolation_dates = []
    interpolation_2024_years = []
    current_date = gpt2_date
    reference_date = datetime(2025, 2, 1)  # Claude 3.7 Sonnet reference
    reference_progress_mult = 1.099328839  # Claude 3.7 Sonnet progress multiplier
    
    while current_date <= end_date:
        interpolation_dates.append(current_date)
        
        # Convert to 2024-years
        calendar_years = (current_date.year - reference_date.year) + (current_date.month - reference_date.month) / 12.0
        if current_date < reference_date:
            # For dates before Claude 3.7 Sonnet, estimate progress multiplier from interpolation
            months_diff = (current_date.year - reference_date.year) * 12 + (current_date.month - reference_date.month)
            interp_progress_mult = interpolate_fn(months_diff)
            adjustment_factor = interp_progress_mult / reference_progress_mult
            adjusted_years = calendar_years * adjustment_factor
            interpolation_2024_years.append(adjusted_years)
        else:
            interpolation_2024_years.append(calendar_years)
        
        current_date += relativedelta(months=1)
    
    # Calculate interpolation progress multipliers
    interpolation_progress_multipliers = []
    for date in interpolation_dates:
        months_diff = (date.year - baseline_date.year) * 12 + (date.month - baseline_date.month)
        interp_progress_mult = interpolate_fn(months_diff)
        interpolation_progress_multipliers.append(interp_progress_mult)
    
    # Filter interpolation data to only show backcast (before baseline_date)
    backcast_2024_years = [interpolation_2024_years[i] for i, d in enumerate(interpolation_dates) if d < baseline_date]
    backcast_progress_multipliers = [interpolation_progress_multipliers[i] for i, d in enumerate(interpolation_dates) if d < baseline_date]
    
    # Filter actual data to include backcast + Claude 3.7 Sonnet baseline point
    backcast_actual_2024_years = [model_2024_years[i] for i, d in enumerate(parsed_dates) if d is not None and d <= baseline_date]
    backcast_actual_progress = [progress_multiplier.iloc[i] for i, d in enumerate(parsed_dates) if d is not None and d <= baseline_date]
    backcast_actual_models = [model_names.iloc[i] for i, d in enumerate(parsed_dates) if d is not None and d <= baseline_date]
    
    # Calculate old method positions (calendar years from Claude 3.7 Sonnet)
    backcast_actual_calendar_years = []
    for i, d in enumerate(parsed_dates):
        if d is not None and d <= baseline_date:
            calendar_years = (d.year - baseline_date.year) + (d.month - baseline_date.month) / 12.0
            backcast_actual_calendar_years.append(calendar_years)
    
    # Create figure for 2024-years method
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot backcast data - 2024-years method
    ax.scatter(backcast_actual_2024_years, backcast_actual_progress, s=100, alpha=0.7, color='red', 
               label='Actual Data', zorder=5)
    
    # Add backcast interpolation curve for 2024-years
    ax.plot(backcast_2024_years, backcast_progress_multipliers, 
             color='green', linewidth=2, alpha=0.8, label='Claude 3.7 Baseline Backcast Model')
    
    # Add vertical line at Claude 3.7 Sonnet baseline
    ax.axvline(x=0, color='purple', linestyle=':', alpha=0.7, label='Claude 3.7 Sonnet Baseline')
    
    # Add model names as labels for backcast
    for i, model in enumerate(backcast_actual_models):
        ax.annotate(model, (backcast_actual_2024_years[i], backcast_actual_progress[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Connect actual data points with lines to show progression
    ax.plot(backcast_actual_2024_years, backcast_actual_progress, alpha=0.5, color='red', linestyle='--')
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Customize plot
    ax.set_xlabel('2024-years from Claude 3.7 Sonnet', fontsize=12)
    ax.set_ylabel('Progress Multiplier (log scale)', fontsize=12)
    ax.set_title('AI R&D progress multiplier backcast (2024-years method)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.margins(0.1)
    ax.legend()
    
    # Make layout tight
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print some basic statistics
    print("\nBasic Statistics:")
    print(f"Number of models: {len(df)}")
    print(f"Time horizon range: {time_horizon.min():.3f} to {time_horizon.max():.3f}")
    print(f"Progress multiplier range: {progress_multiplier.min():.3f} to {progress_multiplier.max():.3f}")
    # Filter out None values for 2024-years range calculation
    valid_2024_years = [y for y in model_2024_years if y is not None]
    if valid_2024_years:
        print(f"2024-years range: {min(valid_2024_years):.3f} to {max(valid_2024_years):.3f}")
    
    # Print interpolation validation
    print("\nClaude 3.7 Sonnet Baseline Interpolation Validation:")
    print(f"At baseline (Claude 3.7 Sonnet, Feb 2025, 0 2024-years): {interpolate_fn(0):.3f}")
    print(f"At 35 months (halfway to target, 2.92 2024-years): {interpolate_fn(35):.3f}")
    print(f"At 70 months (target, 5.83 2024-years): {interpolate_fn(70):.3f}")
    
    # Calculate what the geomean formula gives for verification
    geomean_result = np.sqrt((1.1 - 1) * (8.5 - 1)) + 1
    print(f"Geomean formula result for 35 months: {geomean_result:.3f}")
    
    # Print backcast examples (comparison of methods)
    print("\nBackcast Examples (Calendar years vs 2024-years comparison):")
    # Calculate both calendar and 2024-years for key models
    gpt2_2024_years = None
    gpt3_2024_years = None
    gpt4_2024_years = None
    gpt2_calendar_years = None
    gpt3_calendar_years = None
    gpt4_calendar_years = None
    
    for i, model in enumerate(model_names):
        if 'GPT-2' in str(model):
            gpt2_2024_years = model_2024_years[i]
            if parsed_dates[i] is not None:
                gpt2_calendar_years = (parsed_dates[i].year - baseline_date.year) + (parsed_dates[i].month - baseline_date.month) / 12.0
        elif 'GPT-3' in str(model) and 'GPT-3.5' not in str(model):
            gpt3_2024_years = model_2024_years[i]
            if parsed_dates[i] is not None:
                gpt3_calendar_years = (parsed_dates[i].year - baseline_date.year) + (parsed_dates[i].month - baseline_date.month) / 12.0
        elif 'GPT-4 0314' in str(model):
            gpt4_2024_years = model_2024_years[i]
            if parsed_dates[i] is not None:
                gpt4_calendar_years = (parsed_dates[i].year - baseline_date.year) + (parsed_dates[i].month - baseline_date.month) / 12.0
    
    if gpt2_2024_years is not None and gpt2_calendar_years is not None:
        print(f"GPT-2 (Feb 2019): {gpt2_calendar_years:.3f} calendar years, {gpt2_2024_years:.3f} 2024-years")
    if gpt3_2024_years is not None and gpt3_calendar_years is not None:
        print(f"GPT-3 (Jun 2020): {gpt3_calendar_years:.3f} calendar years, {gpt3_2024_years:.3f} 2024-years")
    if gpt4_2024_years is not None and gpt4_calendar_years is not None:
        print(f"GPT-4 (Mar 2023): {gpt4_calendar_years:.3f} calendar years, {gpt4_2024_years:.3f} 2024-years")
    
    # GPT-3 to Claude 3.7 Sonnet interpolation analysis
    print("\n" + "="*50)
    print("GPT-3 TO CLAUDE 3.7 SONNET INTERPOLATION")
    print("="*50)
    
    gpt3_interpolate_fn, gpt3_date, gpt3_monthly_multiplier = create_gpt3_to_claude37_interpolation(df)
    
    if gpt3_interpolate_fn is not None:
        # Calculate months from GPT-3 for key dates
        claude_37_date = datetime(2025, 2, 1)
        target_date = claude_37_date + relativedelta(months=60)
        
        gpt3_to_claude37_months = (claude_37_date.year - gpt3_date.year) * 12 + (claude_37_date.month - gpt3_date.month)
        gpt3_to_target_months = (target_date.year - gpt3_date.year) * 12 + (target_date.month - gpt3_date.month)
        
        # Calculate predicted value at 60 months after Claude 3.7 Sonnet
        predicted_60m_value = gpt3_interpolate_fn(gpt3_to_target_months)
        
        print(f"Predicted value at 60 months after Claude 3.7 Sonnet: {predicted_60m_value:.6f}")
        
        # Create interpolation data from GPT-3 to 60 months after Claude 3.7 Sonnet
        end_date = target_date
        
        # Create monthly date range and convert to 2024-years
        gpt3_interpolation_dates = []
        gpt3_interpolation_2024_years = []
        current_date = gpt3_date
        while current_date <= end_date:
            gpt3_interpolation_dates.append(current_date)
            
            # Convert to 2024-years
            calendar_years = (current_date.year - reference_date.year) + (current_date.month - reference_date.month) / 12.0
            if current_date < reference_date:
                # For dates before Claude 3.7 Sonnet, estimate progress multiplier from GPT-3 interpolation
                months_from_gpt3 = (current_date.year - gpt3_date.year) * 12 + (current_date.month - gpt3_date.month)
                interp_progress_mult = gpt3_interpolate_fn(months_from_gpt3)
                adjustment_factor = interp_progress_mult / reference_progress_mult
                adjusted_years = calendar_years * adjustment_factor
                gpt3_interpolation_2024_years.append(adjusted_years)
            else:
                gpt3_interpolation_2024_years.append(calendar_years)
            
            current_date += relativedelta(months=1)
        
        # Calculate progress multipliers
        gpt3_interpolation_progress_multipliers = []
        for date in gpt3_interpolation_dates:
            months_from_gpt3 = (date.year - gpt3_date.year) * 12 + (date.month - gpt3_date.month)
            interp_progress_mult = gpt3_interpolate_fn(months_from_gpt3)
            gpt3_interpolation_progress_multipliers.append(interp_progress_mult)
        
        # Convert key dates to 2024-years
        gpt3_2024_years = (gpt3_date.year - reference_date.year) + (gpt3_date.month - reference_date.month) / 12.0
        gpt3_progress_mult_val = None
        for idx, row in df.iterrows():
            if 'GPT-3' in str(row['Column 1']) and 'GPT-3.5' not in str(row['Column 1']):
                gpt3_progress_mult_val = row['Progress multiplier']
                break
        if gpt3_progress_mult_val:
            gpt3_2024_years *= gpt3_progress_mult_val / reference_progress_mult
        
        target_2024_years = (target_date.year - reference_date.year) + (target_date.month - reference_date.month) / 12.0
        
        # Create first window for full timeline forecast
        fig_gpt3_forecast = plt.figure(figsize=(12, 8))
        ax1 = fig_gpt3_forecast.add_subplot(111)
        
        # First plot: Full timeline with forecast
        ax1.scatter(model_2024_years, progress_multiplier, s=100, alpha=0.7, color='red', label='Actual Data')
        
        # Add interpolation curve
        ax1.plot(gpt3_interpolation_2024_years, gpt3_interpolation_progress_multipliers, 
                 color='green', linewidth=2, alpha=0.8, label='GPT-3 to Claude 3.7 Interpolation')
        
        # Add vertical lines for key dates
        ax1.axvline(x=gpt3_2024_years, color='blue', linestyle=':', alpha=0.7, label='GPT-3 (Jun 2020)')
        ax1.axvline(x=0, color='purple', linestyle=':', alpha=0.7, label='Claude 3.7 Sonnet (Feb 2025)')
        ax1.axvline(x=target_2024_years, color='orange', linestyle=':', alpha=0.7, label='60 months after Claude 3.7 Sonnet')
        
        # Add horizontal line at predicted value
        ax1.axhline(y=predicted_60m_value, color='orange', linestyle='--', alpha=0.5)
        
        # Add horizontal line at 8.5 (median SC progress multiplier)
        ax1.axhline(y=8.5, color='red', linestyle='-', alpha=0.7, linewidth=2)
        
        # Add text showing predicted value
        ax1.text(target_2024_years, predicted_60m_value, f'  {predicted_60m_value:.3f}', 
                verticalalignment='center', fontsize=12, color='orange', fontweight='bold')
        
        # Add text showing 8.5 median SC progress multiplier
        ax1.text(target_2024_years, 8.5, '  8.5 (median SC)', 
                verticalalignment='center', fontsize=12, color='red', fontweight='bold')
        
        # Add model names as labels
        for i, model in enumerate(model_names):
            if model_2024_years[i] is not None:
                ax1.annotate(model, (model_2024_years[i], progress_multiplier.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
        
        # Connect actual data points with lines
        ax1.plot(model_2024_years, progress_multiplier, alpha=0.5, color='gray', linestyle='--')
        
        # Set log scale for y-axis
        ax1.set_yscale('log')
        
        # Customize first plot
        ax1.set_xlabel('2024-years', fontsize=12)
        ax1.set_ylabel('Progress Multiplier (log scale)', fontsize=12)
        ax1.set_title('SC progress multiplier forecast from GPT-3 to Claude 3.7 interpolation', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Make layout tight
        plt.tight_layout()
        
        # Show the first plot
        plt.show()
        
        # Create second window for recent focus (no forecast)
        fig_gpt3_recent = plt.figure(figsize=(12, 8))
        ax2 = fig_gpt3_recent.add_subplot(111)
        
        # Second plot: Recent points only (no forecast to 60 months)
        # Show from GPT-3 to Claude 3.7 Sonnet period
        recent_end_date = claude_37_date + relativedelta(months=6)  # A bit past Claude 3.7 Sonnet
        recent_end_2024_years = (recent_end_date.year - reference_date.year) + (recent_end_date.month - reference_date.month) / 12.0
        
        # Filter data for recent period
        recent_actual_2024_years = [model_2024_years[i] for i, d in enumerate(parsed_dates) 
                                   if d is not None and gpt3_date <= d <= recent_end_date]
        recent_actual_progress = [progress_multiplier.iloc[i] for i, d in enumerate(parsed_dates) 
                                 if d is not None and gpt3_date <= d <= recent_end_date]
        recent_actual_models = [model_names.iloc[i] for i, d in enumerate(parsed_dates) 
                               if d is not None and gpt3_date <= d <= recent_end_date]
        
        # Filter interpolation data for recent period
        recent_interp_2024_years = [gpt3_interpolation_2024_years[i] for i, d in enumerate(gpt3_interpolation_dates) 
                                   if d <= recent_end_date]
        recent_interp_progress = [gpt3_interpolation_progress_multipliers[i] for i, d in enumerate(gpt3_interpolation_dates) 
                                 if d <= recent_end_date]
        
        # Plot recent data
        ax2.scatter(recent_actual_2024_years, recent_actual_progress, s=100, alpha=0.7, color='red', 
                   label='Actual Data', zorder=5)
        
        # Add interpolation curve for recent period
        ax2.plot(recent_interp_2024_years, recent_interp_progress, 
                 color='green', linewidth=2, alpha=0.8, label='GPT-3 to Claude 3.7 Interpolation')
        
        # Add vertical lines
        ax2.axvline(x=gpt3_2024_years, color='blue', linestyle=':', alpha=0.7, label='GPT-3 (Jun 2020)')
        ax2.axvline(x=0, color='purple', linestyle=':', alpha=0.7, label='Claude 3.7 Sonnet (Feb 2025)')
        
        # Add model names as labels
        for i, model in enumerate(recent_actual_models):
            ax2.annotate(model, (recent_actual_2024_years[i], recent_actual_progress[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
        
        # Connect actual data points with lines
        ax2.plot(recent_actual_2024_years, recent_actual_progress, alpha=0.5, color='gray', linestyle='--')
        
        # Set log scale for y-axis
        ax2.set_yscale('log')
        
        # Customize second plot
        ax2.set_xlabel('2024-years', fontsize=12)
        ax2.set_ylabel('Progress Multiplier (log scale)', fontsize=12)
        ax2.set_title('GPT-3 to Claude 3.7 interpolation', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Make layout tight
        plt.tight_layout()
        
        # Show the second plot
        plt.show()
        
    else:
        print("GPT-3 to Claude 3.7 Sonnet interpolation failed - skipping plot")
    
    # Add the two new "benchmarks and gaps" plots
    print("\n" + "="*50)
    print("BENCHMARKS AND GAPS ANALYSIS")
    print("="*50)
    
    # Calculate end of 2026 date
    end_of_2026 = datetime(2026, 12, 31)
    
    # Graph (a): GPT-3 to Claude 3.7 interpolation with end of 2026 marker
    if gpt3_interpolate_fn is not None:
        # Calculate months from GPT-3 to end of 2026
        gpt3_to_end2026_months = (end_of_2026.year - gpt3_date.year) * 12 + (end_of_2026.month - gpt3_date.month)
        end2026_value_gpt3 = gpt3_interpolate_fn(gpt3_to_end2026_months)
        
        print(f"GPT-3 to Claude 3.7 interpolation value at end of 2026: {end2026_value_gpt3:.6f}")
        
        # For benchmarks and gaps, use 70 months target
        bg_target_date = claude_37_date + relativedelta(months=70)
        gpt3_to_bg_target_months = (bg_target_date.year - gpt3_date.year) * 12 + (bg_target_date.month - gpt3_date.month)
        predicted_70m_value_bg = gpt3_interpolate_fn(gpt3_to_bg_target_months)
        
        # Create extended interpolation data for benchmarks and gaps (70 months)
        bg_end_date = bg_target_date
        bg_interpolation_dates = []
        current_date = gpt3_date
        while current_date <= bg_end_date:
            bg_interpolation_dates.append(current_date)
            current_date += relativedelta(months=1)
        
        bg_interpolation_progress_multipliers = []
        for date in bg_interpolation_dates:
            months_from_gpt3 = (date.year - gpt3_date.year) * 12 + (date.month - gpt3_date.month)
            interp_progress_mult = gpt3_interpolate_fn(months_from_gpt3)
            bg_interpolation_progress_multipliers.append(interp_progress_mult)
        
        # Convert additional dates to 2024-years for benchmarks and gaps
        end_of_2026_2024_years = (end_of_2026.year - reference_date.year) + (end_of_2026.month - reference_date.month) / 12.0
        bg_target_2024_years = (bg_target_date.year - reference_date.year) + (bg_target_date.month - reference_date.month) / 12.0
        
        # Convert bg_interpolation_dates to 2024-years
        bg_interpolation_2024_years = []
        for date in bg_interpolation_dates:
            calendar_years = (date.year - reference_date.year) + (date.month - reference_date.month) / 12.0
            if date < reference_date:
                months_from_gpt3 = (date.year - gpt3_date.year) * 12 + (date.month - gpt3_date.month)
                interp_progress_mult = gpt3_interpolate_fn(months_from_gpt3)
                adjustment_factor = interp_progress_mult / reference_progress_mult
                adjusted_years = calendar_years * adjustment_factor
                bg_interpolation_2024_years.append(adjusted_years)
            else:
                bg_interpolation_2024_years.append(calendar_years)
        
        # Create figure for GPT-3 interpolation benchmarks and gaps
        fig_gpt3_bg = plt.figure(figsize=(12, 8))
        ax_gpt3_bg = fig_gpt3_bg.add_subplot(111)
        
        # Plot actual data
        ax_gpt3_bg.scatter(model_2024_years, progress_multiplier, s=100, alpha=0.7, color='red', label='Actual Data')
        
        # Add interpolation curve (extended to 70 months)
        ax_gpt3_bg.plot(bg_interpolation_2024_years, bg_interpolation_progress_multipliers, 
                        color='green', linewidth=2, alpha=0.8, label='GPT-3 to Claude 3.7 Interpolation')
        
        # Add vertical lines for key dates
        ax_gpt3_bg.axvline(x=gpt3_2024_years, color='blue', linestyle=':', alpha=0.7, label='GPT-3 (Jun 2020)')
        ax_gpt3_bg.axvline(x=0, color='purple', linestyle=':', alpha=0.7, label='Claude 3.7 Sonnet (Feb 2025)')
        ax_gpt3_bg.axvline(x=end_of_2026_2024_years, color='cyan', linestyle=':', alpha=0.7, label='End of 2026')
        ax_gpt3_bg.axvline(x=bg_target_2024_years, color='orange', linestyle=':', alpha=0.7, label='70 months after Claude 3.7 Sonnet')
        
        # Add horizontal lines
        ax_gpt3_bg.axhline(y=predicted_70m_value_bg, color='orange', linestyle='--', alpha=0.5)
        ax_gpt3_bg.axhline(y=8.5, color='red', linestyle='-', alpha=0.7, linewidth=2)
        ax_gpt3_bg.axhline(y=end2026_value_gpt3, color='cyan', linestyle='-', alpha=0.7, linewidth=2)
        
        # Add text annotations
        ax_gpt3_bg.text(bg_target_2024_years, predicted_70m_value_bg, f'  {predicted_70m_value_bg:.3f}', 
                       verticalalignment='center', fontsize=12, color='orange', fontweight='bold')
        ax_gpt3_bg.text(bg_target_2024_years, 8.5, '  8.5 (median SC)', 
                       verticalalignment='center', fontsize=12, color='red', fontweight='bold')
        ax_gpt3_bg.text(end_of_2026_2024_years, end2026_value_gpt3, f'  {end2026_value_gpt3:.3f}', 
                       verticalalignment='center', fontsize=12, color='cyan', fontweight='bold')
        
        # Add model names as labels
        for i, model in enumerate(model_names):
            if model_2024_years[i] is not None:
                ax_gpt3_bg.annotate(model, (model_2024_years[i], progress_multiplier.iloc[i]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
        
        # Connect actual data points with lines
        ax_gpt3_bg.plot(model_2024_years, progress_multiplier, alpha=0.5, color='gray', linestyle='--')
        
        # Set log scale for y-axis
        ax_gpt3_bg.set_yscale('log')
        
        # Customize plot
        ax_gpt3_bg.set_xlabel('2024-years', fontsize=12)
        ax_gpt3_bg.set_ylabel('Progress Multiplier (log scale)', fontsize=12)
        ax_gpt3_bg.set_title('GPT-3 to Claude 3.7 interpolation benchmarks and gaps', fontsize=14)
        ax_gpt3_bg.grid(True, alpha=0.3)
        ax_gpt3_bg.legend()
        
        # Make layout tight
        plt.tight_layout()
        
        # Show the plot
        plt.show()
    
    # Graph (b): Claude 3.7 baseline interpolation with end of 2026 marker
    # Calculate months from Claude 3.7 baseline to end of 2026
    baseline_to_end2026_months = (end_of_2026.year - baseline_date.year) * 12 + (end_of_2026.month - baseline_date.month)
    end2026_value_baseline = interpolate_fn(baseline_to_end2026_months)
    
    print(f"Claude 3.7 baseline interpolation value at end of 2026: {end2026_value_baseline:.6f}")
    
    # For benchmarks and gaps, create a special interpolation for 70 months
    # This uses the same monthly multiplier but extends to 70 months
    bg_baseline_target_date = baseline_date + relativedelta(months=70)
    
    # Create figure for Claude 3.7 baseline benchmarks and gaps
    fig_baseline_bg = plt.figure(figsize=(12, 8))
    ax_baseline_bg = fig_baseline_bg.add_subplot(111)
    
    # Create extended interpolation data to show forecast to 70 months
    extended_end_date = bg_baseline_target_date  # Show up to 70 months after Claude 3.7 Sonnet
    
    # Create monthly date range and convert to 2024-years
    extended_interpolation_dates = []
    extended_interpolation_2024_years = []
    current_date = gpt2_date
    while current_date <= extended_end_date:
        extended_interpolation_dates.append(current_date)
        
        # Convert to 2024-years
        calendar_years = (current_date.year - reference_date.year) + (current_date.month - reference_date.month) / 12.0
        if current_date < reference_date:
            months_diff = (current_date.year - baseline_date.year) * 12 + (current_date.month - baseline_date.month)
            interp_progress_mult = interpolate_fn(months_diff)
            adjustment_factor = interp_progress_mult / reference_progress_mult
            adjusted_years = calendar_years * adjustment_factor
            extended_interpolation_2024_years.append(adjusted_years)
        else:
            extended_interpolation_2024_years.append(calendar_years)
        
        current_date += relativedelta(months=1)
    
    # Calculate progress multipliers for extended range
    extended_interpolation_progress_multipliers = []
    for date in extended_interpolation_dates:
        months_diff = (date.year - baseline_date.year) * 12 + (date.month - baseline_date.month)
        interp_progress_mult = interpolate_fn(months_diff)
        extended_interpolation_progress_multipliers.append(interp_progress_mult)
    
    # Convert additional dates to 2024-years
    bg_baseline_target_2024_years = (bg_baseline_target_date.year - reference_date.year) + (bg_baseline_target_date.month - reference_date.month) / 12.0
    
    # Plot actual data
    ax_baseline_bg.scatter(model_2024_years, progress_multiplier, s=100, alpha=0.7, color='red', label='Actual Data')
    
    # Add interpolation curve
    ax_baseline_bg.plot(extended_interpolation_2024_years, extended_interpolation_progress_multipliers, 
                        color='green', linewidth=2, alpha=0.8, label='Claude 3.7 Baseline Interpolation')
    
    # Add vertical lines for key dates
    ax_baseline_bg.axvline(x=0, color='purple', linestyle=':', alpha=0.7, label='Claude 3.7 Sonnet (Feb 2025)')
    ax_baseline_bg.axvline(x=end_of_2026_2024_years, color='cyan', linestyle=':', alpha=0.7, label='End of 2026')
    ax_baseline_bg.axvline(x=bg_baseline_target_2024_years, color='orange', linestyle=':', alpha=0.7, label='70 months after Claude 3.7 Sonnet')
    
    # Add horizontal lines
    ax_baseline_bg.axhline(y=8.5, color='red', linestyle='-', alpha=0.7, linewidth=2)
    ax_baseline_bg.axhline(y=end2026_value_baseline, color='cyan', linestyle='-', alpha=0.7, linewidth=2)
    
    # Add text annotations
    ax_baseline_bg.text(bg_baseline_target_2024_years, 8.5, '  8.5 (median SC)', 
                       verticalalignment='center', fontsize=12, color='red', fontweight='bold')
    ax_baseline_bg.text(end_of_2026_2024_years, end2026_value_baseline, f'  {end2026_value_baseline:.3f}', 
                       verticalalignment='center', fontsize=12, color='cyan', fontweight='bold')
    
    # Add model names as labels
    for i, model in enumerate(model_names):
        if model_2024_years[i] is not None:
            ax_baseline_bg.annotate(model, (model_2024_years[i], progress_multiplier.iloc[i]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Connect actual data points with lines
    ax_baseline_bg.plot(model_2024_years, progress_multiplier, alpha=0.5, color='gray', linestyle='--')
    
    # Set log scale for y-axis
    ax_baseline_bg.set_yscale('log')
    
    # Customize plot
    ax_baseline_bg.set_xlabel('2024-years', fontsize=12)
    ax_baseline_bg.set_ylabel('Progress Multiplier (log scale)', fontsize=12)
    ax_baseline_bg.set_title('Claude 3.7 baseline interpolation benchmarks and gaps', fontsize=14)
    ax_baseline_bg.grid(True, alpha=0.3)
    ax_baseline_bg.legend()
    
    # Make layout tight
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Fit double exponential to the "Progress multiplier w/Apr 2025 as 1" data
    print("\n" + "="*50)
    print("DOUBLE EXPONENTIAL ANALYSIS")
    print("(Including 60-month target point in fit)")
    print("="*50)
    
    # First fit without target point (original historical data only)
    popt_orig, x_data_orig, y_data_orig, model_names_orig, r_squared_orig = fit_double_exponential_to_data_original(df)
    
    # Second fit with target point included
    popt, x_data, y_data, model_names_de, r_squared, r_squared_hist = fit_double_exponential_to_data(df)
    
    # Third fit using 2024-months since GPT-2
    popt_months, x_data_months, y_data_months, model_names_months, r_squared_months = fit_double_exponential_to_data_months_since_gpt2(df)
    
    if popt is not None or popt_orig is not None or popt_months is not None:
        # Calculate 2024-years for Claude 3.7 Sonnet and 60 months later
        claude_37_2024_years = 0.0  # Claude 3.7 Sonnet is the reference point
        target_2024_years = 5.0  # 60 months = 5 years in 2024-years
        
        # Extrapolate to 60 months after Claude 3.7 Sonnet
        if popt is not None:
            target_value = double_exponential(target_2024_years, *popt)
            claude_37_value = double_exponential(claude_37_2024_years, *popt)
            
            print(f"Constrained fit - Claude 3.7 Sonnet (Feb 2025, {claude_37_2024_years} 2024-years): {claude_37_value:.6f}")
            print(f"Constrained fit - 60 months after Claude 3.7 Sonnet ({target_2024_years} 2024-years): {target_value:.6f}")
            print(f"Note: Target point (8.5) was included in the fit, so this should be close to 8.5")
        else:
            target_value = 8.5  # Default for display purposes
            claude_37_value = 1.0  # Default for display purposes
            print("Constrained fit failed - using default values for display")
        
        # Create separate window for original data fit (without target point) + extrapolation
        if popt_orig is not None:
            fig_orig = plt.figure(figsize=(12, 8))
            ax1 = fig_orig.add_subplot(111)
            # Plot original historical data points
            ax1.scatter(x_data_orig, y_data_orig, s=100, alpha=0.7, color='red', 
                       label='Historical Data', zorder=5)
            
            # Add model names as labels
            for i, model in enumerate(model_names_orig):
                ax1.annotate(model, (x_data_orig[i], y_data_orig[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
            
            # Create smooth curve for fitted function (extrapolated)
            x_smooth = np.linspace(min(x_data_orig), target_2024_years, 1000)
            y_smooth = double_exponential(x_smooth, *popt_orig)
            
            # Plot fitted curve
            ax1.plot(x_smooth, y_smooth, color='blue', linewidth=2, alpha=0.8, 
                    label=f'Double Exponential Fit (R² hist = {r_squared_orig:.4f})')
            
            # Calculate extrapolated value at target point
            target_value_orig = double_exponential(target_2024_years, *popt_orig)
            
            # Plot extrapolated target point
            ax1.scatter([target_2024_years], [target_value_orig], s=100, alpha=0.7, color='green', 
                       label=f'Extrapolated Target: {target_value_orig:.3f}', zorder=5, marker='s')
            
            # Add vertical lines for key dates
            ax1.axvline(x=claude_37_2024_years, color='purple', linestyle=':', alpha=0.7, 
                      label='Claude 3.7 Sonnet (Feb 2025)')
            ax1.axvline(x=target_2024_years, color='orange', linestyle=':', alpha=0.7, 
                      label='60 months after Claude 3.7 Sonnet')
            
            # Add horizontal line at extrapolated target value
            ax1.axhline(y=target_value_orig, color='green', linestyle='--', alpha=0.5)
            
            # Add text showing extrapolated target value
            ax1.text(target_2024_years, target_value_orig, f'  {target_value_orig:.3f}', 
                    verticalalignment='center', fontsize=12, color='green', fontweight='bold')
            
            print(f"Original fit extrapolated to 60 months: {target_value_orig:.6f}")
            
            # Set log scale for y-axis
            ax1.set_yscale('log')
            
            # Customize first plot
            ax1.set_xlabel('2024-years', fontsize=12)
            ax1.set_ylabel('Progress Multiplier w/Apr 2025 as 1 (log scale)', fontsize=12)
            ax1.set_title('Double Exponential Fit (2024-years, includes negatives)', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Make layout tight and show the plot
            plt.tight_layout()
            plt.show()
        else:
            # Fallback if original fit failed
            print("Original fit failed - skipping first plot")
        
        # Create new plot window for the remaining 2 subplots
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Second subplot: Constrained fit with target point (full timeline)
        if popt is not None:
            # Separate actual data from target point
            actual_indices = [i for i, model in enumerate(model_names_de) if "Target Point" not in model]
            target_indices = [i for i, model in enumerate(model_names_de) if "Target Point" in model]
            
            # Plot actual data points
            ax2.scatter(x_data[actual_indices], y_data[actual_indices], s=100, alpha=0.7, color='red', 
                       label='Historical Data', zorder=5)
            
            # Plot target point separately
            if target_indices:
                ax2.scatter(x_data[target_indices], y_data[target_indices], s=100, alpha=0.7, color='orange', 
                           label='Target Point (8.5)', zorder=5, marker='D')
            
            # Add model names as labels
            for i, model in enumerate(model_names_de):
                ax2.annotate(model, (x_data[i], y_data[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
            
            # Create smooth curve for fitted function
            x_smooth = np.linspace(min(x_data), target_2024_years, 1000)
            y_smooth = double_exponential(x_smooth, *popt)
            
            # Plot fitted curve
            ax2.plot(x_smooth, y_smooth, color='blue', linewidth=2, alpha=0.8, 
                    label=f'Constrained Double Exponential (R² all = {r_squared:.4f}, R² hist = {r_squared_hist:.4f})')
            
            # Add vertical lines for key dates
            ax2.axvline(x=claude_37_2024_years, color='purple', linestyle=':', alpha=0.7, 
                      label='Claude 3.7 Sonnet (Feb 2025)')
            ax2.axvline(x=target_2024_years, color='orange', linestyle=':', alpha=0.7, 
                      label='60 months after Claude 3.7 Sonnet')
            
            # Add horizontal line at target value
            ax2.axhline(y=target_value, color='orange', linestyle='--', alpha=0.5)
            
            # Add text showing target value
            ax2.text(target_2024_years, target_value, f'  {target_value:.3f}', 
                    verticalalignment='center', fontsize=12, color='orange', fontweight='bold')
        else:
            # Fallback if constrained fit failed
            ax2.text(0.5, 0.5, 'Constrained fit failed', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=16, color='red')
        
        # Set log scale for y-axis
        ax2.set_yscale('log')
        
        # Customize second plot
        ax2.set_xlabel('2024-years', fontsize=12)
        ax2.set_ylabel('Progress Multiplier w/Apr 2025 as 1 (log scale)', fontsize=12)
        ax2.set_title('Double Exponential Fit (Constrained to Target)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Third subplot: Backcast only (before Claude 3.7 Sonnet)
        if popt is not None:
            backcast_mask = x_data < claude_37_2024_years
            backcast_x = x_data[backcast_mask]
            backcast_y = y_data[backcast_mask]
            backcast_models = [model_names_de[i] for i in range(len(model_names_de)) if backcast_mask[i]]
            
            # Plot backcast data
            ax3.scatter(backcast_x, backcast_y, s=100, alpha=0.7, color='red', 
                       label='Historical Data', zorder=5)
            
            # Add model names as labels
            for i, model in enumerate(backcast_models):
                ax3.annotate(model, (backcast_x[i], backcast_y[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
            
            # Plot fitted curve for backcast period
            x_backcast_smooth = np.linspace(min(x_data), claude_37_2024_years, 500)
            y_backcast_smooth = double_exponential(x_backcast_smooth, *popt)
            ax3.plot(x_backcast_smooth, y_backcast_smooth, color='blue', linewidth=2, alpha=0.8, 
                    label=f'Constrained Double Exponential Backcast (R² hist = {r_squared_hist:.4f})')
            
            # Add vertical line at Claude 3.7 Sonnet
            ax3.axvline(x=claude_37_2024_years, color='purple', linestyle=':', alpha=0.7, 
                      label='Claude 3.7 Sonnet')
        else:
            # Fallback if constrained fit failed
            ax3.text(0.5, 0.5, 'Constrained fit failed', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=16, color='red')
        
        # Set log scale for y-axis
        ax3.set_yscale('log')
        
        # Customize third plot
        ax3.set_xlabel('2024-years', fontsize=12)
        ax3.set_ylabel('Progress Multiplier w/Apr 2025 as 1 (log scale)', fontsize=12)
        ax3.set_title('Double Exponential Backcast Only (Constrained)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Make layout tight
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        
        # NEW: Add the 2024-months since GPT-2 plot
        if popt_months is not None:
            print(f"\n" + "="*50)
            print("DOUBLE EXPONENTIAL ANALYSIS (2024-months since GPT-2)")
            print("="*50)
            
            # Calculate extrapolated target value
            # Find Claude 3.7 Sonnet months value first
            claude_37_months = 0  # Will be calculated from data
            for i, model in enumerate(model_names_months):
                if 'Claude 3.7 Sonnet' in model:
                    claude_37_months = x_data_months[i]
                    break
            
            # Calculate target at 60 months after Claude 3.7 Sonnet
            target_months = claude_37_months + 60
            target_value_months = double_exponential(target_months, *popt_months)
            
            print(f"Extrapolated value at 60 months after Claude 3.7 Sonnet: {target_value_months:.6f}")
            
            fig_months = plt.figure(figsize=(12, 8))
            ax_months = fig_months.add_subplot(111)
            
            # Plot historical data points
            ax_months.scatter(x_data_months, y_data_months, 
                             s=100, alpha=0.7, color='red', label='Historical Data', zorder=5)
            
            # Add model names as labels
            for i, model in enumerate(model_names_months):
                ax_months.annotate(model, (x_data_months[i], y_data_months[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
            
            # Create smooth curve for fitted function (extended to show extrapolation)
            x_smooth_months = np.linspace(min(x_data_months), target_months, 1000)
            y_smooth_months = double_exponential(x_smooth_months, *popt_months)
            
            # Plot fitted curve
            ax_months.plot(x_smooth_months, y_smooth_months, color='blue', linewidth=2, alpha=0.8, 
                          label=f'Double Exponential Fit (R² = {r_squared_months:.4f})')
            
            # Plot extrapolated target point
            ax_months.scatter([target_months], [target_value_months], s=100, alpha=0.7, color='green', 
                             label=f'Extrapolated Target: {target_value_months:.3f}', zorder=5, marker='s')
            
            # Add vertical line at GPT-2 (0 months)
            ax_months.axvline(x=0, color='green', linestyle=':', alpha=0.7, label='GPT-2 (Feb 2019)')
            
            # Add vertical line at Claude 3.7 Sonnet
            ax_months.axvline(x=claude_37_months, color='purple', linestyle=':', alpha=0.7, 
                             label='Claude 3.7 Sonnet (Feb 2025)')
            
            # Add vertical line at target (60 months after Claude 3.7 Sonnet)
            ax_months.axvline(x=target_months, color='orange', linestyle=':', alpha=0.7, 
                             label='60 months after Claude 3.7 Sonnet')
            
            # Add horizontal line at extrapolated target value
            ax_months.axhline(y=target_value_months, color='green', linestyle='--', alpha=0.5)
            
            # Add text showing extrapolated target value
            ax_months.text(target_months, target_value_months, f'  {target_value_months:.3f}', 
                          verticalalignment='center', fontsize=12, color='green', fontweight='bold')
            
            # Set log scale for y-axis
            ax_months.set_yscale('log')
            
            # Customize plot
            ax_months.set_xlabel('2024-months since GPT-2', fontsize=12)
            ax_months.set_ylabel('Progress Multiplier w/Apr 2025 as 1 (log scale)', fontsize=12)
            ax_months.set_title('Double Exponential Fit (2024-months from GPT-2=0)', fontsize=14)
            ax_months.grid(True, alpha=0.3)
            ax_months.legend()
            
            # Make layout tight
            plt.tight_layout()
            
            # Show the plot
            plt.show()
        
    else:
        print("Double exponential fitting failed - skipping plot")
    
    return df

if __name__ == "__main__":
    data = load_and_plot_aird_estimates()
