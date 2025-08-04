import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import curve_fit, fsolve

# Import useful functions from the existing file
from explore_aird_interpolation_methods import convert_to_2024_years, double_exponential

def fixed_decay(t, H_0, a, T_0):
    """
    New curve: H_0*(1-a*(t/T_0))^(1/log2(1-a))
    Where:
    - t is the 2024-years
    - H_0, a, T_0 are parameters to be fitted
    """
    try:
        # Ensure we don't hit singularities or invalid domains
        ratio = t / T_0
        inner_term = 1 - a * ratio
        
        # More conservative clipping to avoid numerical issues
        # The inner term must be positive and not equal to 1
        inner_term = np.clip(inner_term, 1e-15, 1 - 1e-15)
        
        # Calculate the exponent with more care
        # 1/log2(1-a) where (1-a) must be positive and not equal to 1
        base_term = 1 - a
        base_term = np.clip(base_term, 1e-15, 1 - 1e-15)
        
        # Calculate exponent
        exponent = 1 / np.log2(base_term)
        
        # Calculate the result
        result = H_0 * np.power(inner_term, exponent)
        
        # Ensure result is positive and finite
        result = np.clip(result, 1e-15, 1e30)
        
        return result
    except Exception as e:
        print(f"Error in fixed_decay: {e}")
        print(f"Parameters: H_0={H_0}, a={a}, T_0={T_0}")
        print(f"t range: {np.min(t)} to {np.max(t)}")
        # If any calculation fails, return a reasonable default
        return np.full_like(t, 1.0)

def fit_double_exponential_to_time_horizon_2024_years(df):
    """Fit a double exponential to the 80% time horizon data using 2024-years anchored to Claude 3.7 Sonnet as 0"""
    # Convert to 2024-years for each data point
    model_2024_years_orig, parsed_dates = convert_to_2024_years(df)
    
    # Parse data and use 2024-years directly (Claude 3.7 Sonnet is already at 0)
    years_2024 = []
    time_horizon_values = []
    model_names = []
    
    for idx, row in df.iterrows():
        try:
            if model_2024_years_orig[idx] is not None:
                years_2024.append(model_2024_years_orig[idx])
                # Get 80% time horizon value
                time_horizon_values.append(row['80% time horizon'])
                model_names.append(row['Column 1'])
        except:
            continue
    
    # Convert to numpy arrays
    x_data = np.array(years_2024)
    y_data = np.array(time_horizon_values)
    
    # Fit double exponential
    # Initial guess for parameters [a, b, c]
    # For time horizon, we expect growth, so start with reasonable values
    initial_guess = [0.001, 0.001, 0.01]  # Much more conservative c parameter for years
    
    try:
        # Set bounds for parameters to help with convergence
        # Lower bounds: [a_min, b_min, c_min]
        # Upper bounds: [a_max, b_max, c_max]
        lower_bounds = [0.0001, -10, 0.001]  # Allow small starting values
        upper_bounds = [1.0, 10, 2.0]        # Allow much higher c values for natural fit
        
        # Fit the double exponential
        popt, pcov = curve_fit(double_exponential, x_data, y_data, 
                              p0=initial_guess, bounds=(lower_bounds, upper_bounds), 
                              maxfev=10000)
        
        # Calculate R-squared
        y_pred = double_exponential(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\nDouble Exponential Fit Results (80% Time Horizon vs 2024-years from Claude 3.7 Sonnet):")
        print(f"Parameters: a={popt[0]:.6f}, b={popt[1]:.6f}, c={popt[2]:.6f}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Data points used: {len(x_data)} (historical data only)")
        print(f"Claude 3.7 Sonnet is at 0 2024-years in this scale")
        
        return popt, x_data, y_data, model_names, r_squared
        
    except Exception as e:
        print(f"Double exponential fitting (80% time horizon vs 2024-years) failed: {e}")
        return None, x_data, y_data, model_names, 0

def fit_fixed_decay_to_time_horizon_2024_years(df):
    """Fit the fixed decay curve H_0*(1-a*(t/T_0))^(1/log2(1-a)) to the 80% time horizon data"""
    # Convert to 2024-years for each data point
    model_2024_years_orig, parsed_dates = convert_to_2024_years(df)
    
    # Parse data and use 2024-years directly (Claude 3.7 Sonnet is already at 0)
    years_2024 = []
    time_horizon_values = []
    model_names = []
    
    for idx, row in df.iterrows():
        try:
            if model_2024_years_orig[idx] is not None:
                years_2024.append(model_2024_years_orig[idx])
                # Get 80% time horizon value
                time_horizon_values.append(row['80% time horizon'])
                model_names.append(row['Column 1'])
        except:
            continue
    
    # Convert to numpy arrays
    x_data_original = np.array(years_2024)
    y_data = np.array(time_horizon_values)
    
    # Shift x-values to make them all positive
    # Add the absolute value of the minimum plus a small buffer
    x_shift = abs(min(x_data_original)) + 0.1
    x_data = x_data_original + x_shift
    
    print(f"\nDebugging new curve fit:")
    print(f"Original data range: x from {min(x_data_original):.3f} to {max(x_data_original):.3f}")
    print(f"Shifted data range: x from {min(x_data):.3f} to {max(x_data):.3f} (shift = +{x_shift:.3f})")
    print(f"Data range: y from {min(y_data):.3f} to {max(y_data):.3f}")
    
    # Test the fixed_decay function with some sample parameters
    print(f"\nTesting fixed_decay function with shifted positive x:")
    test_params = [1.0, 0.5, 10.0]  # H_0, a, T_0
    test_x = np.array([0.1, 1.0, 2.0])  # All positive now
    try:
        test_y = fixed_decay(test_x, *test_params)
        print(f"  Test successful: f({test_x}) = {test_y}")
    except Exception as e:
        print(f"  Test failed: {e}")
        return None, x_data_original, y_data, model_names, 0
    
    # Try multiple initial guesses and parameter bounds
    initial_guesses = [
        [1.0, 0.1, 5.0],    # Conservative
        [0.1, 0.3, 20.0],   # Moderate  
        [10.0, 0.7, 50.0],  # Aggressive
        [0.01, 0.05, 2.0],  # Very conservative
        [max(y_data), 0.5, 10.0],  # Scale H_0 to data
        [min(y_data), 0.9, max(x_data)*2],  # Scale to data range
    ]
    
    bounds_options = [
        # Option 1: Conservative bounds
        ([1e-6, 0.01, 0.1], [100, 0.99, 100]),
        # Option 2: Wider bounds
        ([1e-6, 0.001, 0.01], [1000, 0.999, 1000]),
        # Option 3: Very wide bounds
        ([1e-10, 0.0001, 0.001], [10000, 0.9999, 10000]),
    ]
    
    best_fit = None
    best_r_squared = -np.inf
    best_params = None
    
    for i, initial_guess in enumerate(initial_guesses):
        for j, bounds in enumerate(bounds_options):
            try:
                print(f"\nTrying initial guess {i+1}: H_0={initial_guess[0]:.3f}, a={initial_guess[1]:.3f}, T_0={initial_guess[2]:.3f}")
                print(f"  with bounds {j+1}: H_0=[{bounds[0][0]:.1e}, {bounds[1][0]:.1e}], a=[{bounds[0][1]:.3f}, {bounds[1][1]:.3f}], T_0=[{bounds[0][2]:.3f}, {bounds[1][2]:.3f}]")
                
                # To improve the fit for early, small-value data points, use weighted least squares.
                # We can give more weight to points with smaller y-values. A common choice is 1/y or 1/y^2.
                # Let's use 1/y as the weight. Add a small epsilon to avoid division by zero.
                weights = 1.0 / (np.abs(y_data) + 1e-6)
                
                # Fit the fixed decay curve with weights
                popt, pcov = curve_fit(fixed_decay, x_data, y_data,
                                      p0=initial_guess, bounds=bounds,
                                      sigma=1.0/weights,  # sigma is the inverse of the weight
                                      absolute_sigma=True,
                                      maxfev=50000)
                
                # Calculate R-squared (unweighted, for standard comparison)
                y_pred = fixed_decay(x_data, *popt)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Calculate weighted R-squared for a more representative metric of the fit
                weighted_mean = np.sum(weights * y_data) / np.sum(weights)
                ss_res_weighted = np.sum(weights * (y_data - y_pred) ** 2)
                ss_tot_weighted = np.sum(weights * (y_data - weighted_mean) ** 2)
                if ss_tot_weighted == 0:
                    r_squared_weighted = r_squared # Fallback
                else:
                    r_squared_weighted = 1 - (ss_res_weighted / ss_tot_weighted)
                
                print(f"  Result: H_0={popt[0]:.6f}, a={popt[1]:.6f}, T_0={popt[2]:.6f}")
                print(f"  R² (unweighted): {r_squared:.6f}")
                print(f"  R² (weighted): {r_squared_weighted:.6f}")
                
                # Check if this is the best fit so far (using weighted R-squared)
                if r_squared_weighted > best_r_squared:
                    best_fit = popt
                    best_r_squared = r_squared_weighted
                    best_params = (i+1, j+1)
                    print(f"  *** New best fit! ***")
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
    
    # Use the best fit found
    if best_fit is not None:
        popt = best_fit
        r_squared = best_r_squared
        
        print(f"\nBest fit found using initial guess {best_params[0]}, bounds {best_params[1]}:")
        print(f"Fixed Decay Fit Results (H_0*(1-a*(t/T_0))^(1/log2(1-a))) with shifted x:")
        print(f"Parameters: H_0={popt[0]:.6f}, a={popt[1]:.6f}, T_0={popt[2]:.6f}")
        
        # Recalculate final R-squared values for printing
        y_pred_final = fixed_decay(x_data, *popt)
        ss_res_final = np.sum((y_data - y_pred_final) ** 2)
        ss_tot_final = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared_final = 1 - (ss_res_final / ss_tot_final)
        
        weights_final = 1.0 / (np.abs(y_data) + 1e-6)
        weighted_mean_final = np.sum(weights_final * y_data) / np.sum(weights_final)
        ss_res_weighted_final = np.sum(weights_final * (y_data - y_pred_final) ** 2)
        ss_tot_weighted_final = np.sum(weights_final * (y_data - weighted_mean_final) ** 2)
        r_squared_weighted_final = 1 - (ss_res_weighted_final / ss_tot_weighted_final)
        
        print(f"R-squared (unweighted): {r_squared_final:.4f}")
        print(f"R-squared (weighted): {r_squared_weighted_final:.4f}")
        print(f"Data points used: {len(x_data)} (historical data only)")
        print(f"X-shift applied: +{x_shift:.3f} years")
        print(f"Singularity at t = {popt[2]/popt[1]:.3f} years (in shifted coordinates)")
        print(f"Singularity at t = {popt[2]/popt[1] - x_shift:.3f} years from Claude 3.7 Sonnet (original coordinates)")
        
        # Print residuals for debugging
        y_pred = fixed_decay(x_data, *popt)
        print(f"\nResiduals analysis:")
        for i, model in enumerate(model_names):
            residual = y_data[i] - y_pred[i]
            print(f"{model}: actual={y_data[i]:.4f}, predicted={y_pred[i]:.4f}, residual={residual:.4f}")
        
        # Store the shift for later use in plotting
        return (popt, x_shift), x_data_original, y_data, model_names, r_squared
    else:
        print(f"\nAll fitting attempts failed!")
        return None, x_data_original, y_data, model_names, 0

def find_fixed_decay_boundary(popt, x_start, x_max_search=1000, max_value=1e50):
    """Find where the new curve function becomes numerically unstable (hits overflow or singularity)"""
    try:
        # Test points progressively further out
        x_test = x_start
        step_size = 1.0  # Start with small steps
        
        # For the new curve, we need to be careful about the singularity at T_0/a
        H_0, a, T_0 = popt
        singularity_point = T_0 / a
        
        while x_test < min(x_max_search, singularity_point - 0.1):  # Stay away from singularity
            try:
                value = fixed_decay(x_test, *popt)
                if np.isnan(value) or np.isinf(value) or value > max_value:
                    return x_test - step_size  # Return the last stable point
                x_test += step_size
                
                # Increase step size as we go further to speed up search
                if x_test > x_start + 50:
                    step_size = 5.0
                elif x_test > x_start + 20:
                    step_size = 2.0
                    
            except (OverflowError, ValueError, RuntimeError):
                return x_test - step_size  # Return the last stable point
        
        # If we get close to the singularity, return a point safely before it
        return min(x_max_search, singularity_point - 0.1)
        
    except Exception as e:
        print(f"Error in find_new_curve_boundary: {e}")
        return x_start + 10  # Return a safe default

def find_numerical_boundary(popt, x_start, x_max_search=1000, max_value=1e50):
    """Find where the double exponential function becomes numerically unstable (hits overflow)"""
    try:
        # Test points progressively further out
        x_test = x_start
        step_size = 0.1  # Start with smaller steps for better precision
        
        while x_test < x_max_search:
            try:
                value = double_exponential(x_test, *popt)
                if np.isnan(value) or np.isinf(value) or value > max_value:
                    return x_test - step_size  # Return the last stable point
                x_test += step_size
                
                # Increase step size as we go further to speed up search
                if x_test > x_start + 100:
                    step_size = 5.0
                elif x_test > x_start + 50:
                    step_size = 2.0
                elif x_test > x_start + 10:
                    step_size = 0.5
                    
            except:
                return x_test - step_size  # Return the last stable point
        
        return x_max_search  # If we made it through the whole range
    except:
        return x_start + 10  # Fallback if detection fails

def find_sc_crossing(popt, x_range, target_value_minutes=10*365.25*24*60, data_x=None, curve_type='double_exponential'):
    """Find when the fitted curve crosses the SC threshold (10 year time horizon = 5,259,600 minutes)"""
    try:
        # Define function to find root (where curve equals target_value_minutes)
        if curve_type == 'double_exponential':
            def curve_minus_target(x):
                return double_exponential(x, *popt) - target_value_minutes
        else:  # fixed_decay
            def curve_minus_target(x):
                return fixed_decay(x, *popt) - target_value_minutes
        
        # Extend search range significantly to ensure we find SC
        x_min, x_max = x_range
        
        # For approaches that might have late SC crossings, extend the search
        if data_x is not None:
            data_max_x = max(data_x)
            # Extend search to at least 10x the data range or 100 units, whichever is larger
            extended_max = max(x_max, data_max_x + max(abs(data_max_x) * 10, 100))
        else:
            extended_max = max(x_max, 100)  # Minimum search range
        
        # For fixed_decay, don't go too close to singularity, but get closer than before
        if curve_type == 'fixed_decay':
            H_0, a, T_0 = popt
            singularity_point = T_0 / a
            extended_max = min(extended_max, singularity_point * 0.999)  # Get much closer to singularity
        
        # Check if target is reachable in this range
        if curve_type == 'double_exponential':
            y_min = double_exponential(x_min, *popt)
        else:
            y_min = fixed_decay(x_min, *popt)
        
        # Test progressively to find a range where target is achievable
        search_x = x_min
        
        # For fixed_decay, use much smaller steps near the singularity
        if curve_type == 'fixed_decay':
            H_0, a, T_0 = popt
            singularity_point = T_0 / a
            # Use adaptive step size: smaller steps as we approach singularity
            base_step = (extended_max - x_min) / 5000  # Start with 5000 points
            search_step = max(0.001, base_step)  # Very small steps
        else:
            search_step = max(0.01, (extended_max - x_min) / 1000)  # Test 1000 points with smaller steps
        
        print(f"\nDEBUG SC crossing search ({curve_type}):")
        print(f"  Initial range: {x_min:.3f} to {x_max:.3f}")
        print(f"  Extended search to: {extended_max:.3f}")
        print(f"  Target: {target_value_minutes:.0f} minutes")
        print(f"  Function value at x_min ({x_min:.3f}): {y_min:.2f} minutes")
        
        # For the fixed_decay, we know it approaches infinity at the singularity
        if curve_type == 'fixed_decay':
            H_0, a, T_0 = popt
            singularity_point = T_0 / a
            print(f"  Singularity at x={singularity_point:.3f}")
            
            # Check if we're already close to the singularity
            if extended_max > singularity_point * 0.99:
                print(f"  Warning: Search range very close to singularity!")
        
        # Find where we exceed the target
        last_valid_x = x_min
        last_valid_y = y_min
        
        while search_x <= extended_max:
            try:
                if curve_type == 'double_exponential':
                    y_test = double_exponential(search_x, *popt)
                else:
                    y_test = fixed_decay(search_x, *popt)
                
                # Check for numerical issues
                if np.isnan(y_test) or np.isinf(y_test):
                    print(f"  Hit numerical instability at x={search_x:.3f}")
                    break
                    
                if y_test >= target_value_minutes:
                    # Found the crossing point, now use numerical solver
                    if search_x == x_min:
                        print(f"  Target already exceeded at range start!")
                        return x_min
                    
                    # Use fsolve to find precise crossing with better bounds
                    try:
                        # Use bisection-like approach first to get a better starting point
                        x_low = last_valid_x
                        x_high = search_x
                        
                        # Use the midpoint as starting guess for fsolve
                        x_guess = (x_low + x_high) / 2
                        
                        crossing_x = fsolve(curve_minus_target, x_guess)[0]
                        
                        if curve_type == 'double_exponential':
                            crossing_y = double_exponential(crossing_x, *popt)
                        else:
                            crossing_y = fixed_decay(crossing_x, *popt)
                        
                        # Verify the solution is reasonable
                        if abs(crossing_y - target_value_minutes) < target_value_minutes * 0.05:  # Within 5%
                            print(f"  Found SC crossing at x={crossing_x:.3f}, y={crossing_y:.0f} minutes")
                            return crossing_x
                        else:
                            print(f"  Solution verification failed - y={crossing_y:.0f} is not close to target {target_value_minutes:.0f}")
                            print(f"  Error: {abs(crossing_y - target_value_minutes):.0f} minutes ({abs(crossing_y - target_value_minutes)/target_value_minutes*100:.1f}%)")
                            # Still return the result if it's not too far off
                            if abs(crossing_y - target_value_minutes) < target_value_minutes * 0.5:  # Within 50%
                                print(f"  Accepting result despite verification issues")
                                return crossing_x
                            return None
                    except Exception as e:
                        print(f"  fsolve failed: {e}")
                        return None
                
                # Store the last valid point
                last_valid_x = search_x
                last_valid_y = y_test
                
                # For fixed_decay, make step size even smaller as we approach singularity
                if curve_type == 'fixed_decay':
                    H_0, a, T_0 = popt
                    singularity_point = T_0 / a
                    distance_to_singularity = singularity_point - search_x
                    
                    # Make step size proportional to distance to singularity
                    if distance_to_singularity > 0:
                        adaptive_step = min(search_step, distance_to_singularity / 1000)
                        search_x += max(0.0001, adaptive_step)  # Minimum step size
                    else:
                        break  # We've reached the singularity
                else:
                    search_x += search_step
                
            except Exception as e:
                # Hit numerical overflow before finding target
                print(f"  Hit numerical overflow at x={search_x:.3f} before reaching SC: {e}")
                return None
        
        print(f"  Target not reached even in extended range")
        print(f"  Last valid point: x={last_valid_x:.3f}, y={last_valid_y:.0f} minutes")
        
        # For new curve, let's test extremely close to singularity
        if curve_type == 'fixed_decay':
            H_0, a, T_0 = popt
            singularity_point = T_0 / a
            print(f"\n  Testing function values very close to singularity:")
            
            # Test at increasingly close points to singularity
            test_distances = [0.1, 0.01, 0.001, 0.0001, 0.00001]
            for distance in test_distances:
                test_x = singularity_point - distance
                try:
                    test_y = fixed_decay(test_x, *popt)
                    if not np.isnan(test_y) and not np.isinf(test_y):
                        print(f"    At x={test_x:.5f} (distance {distance:.5f} from singularity): y={test_y:.0f} minutes")
                        if test_y >= target_value_minutes:
                            print(f"    *** SC THRESHOLD REACHED! ***")
                            # Use bisection to find the precise crossing point
                            try:
                                from scipy.optimize import brentq
                                # We know the crossing is between this point and the previous distance
                                if distance < 0.1:  # Only do bisection for closer points
                                    prev_distance = test_distances[test_distances.index(distance) - 1] if distance != test_distances[0] else 0.2
                                    x_low = singularity_point - prev_distance
                                    x_high = test_x
                                    
                                    def curve_minus_target_local(x):
                                        return fixed_decay(x, *popt) - target_value_minutes
                                    
                                    precise_x = brentq(curve_minus_target_local, x_low, x_high)
                                    precise_y = fixed_decay(precise_x, *popt)
                                    print(f"    Precise SC crossing at x={precise_x:.6f}, y={precise_y:.0f} minutes")
                                    return precise_x
                                else:
                                    return test_x
                            except:
                                print(f"    Precise search failed, using approximate point")
                                return test_x
                    else:
                        print(f"    At x={test_x:.5f}: numerical instability")
                        break
                except:
                    print(f"    At x={test_x:.5f}: calculation failed")
                    break
        
        return None
        
    except Exception as e:
        print(f"  SC crossing search failed: {e}")
        return None

def load_and_analyze_time_horizon_focused():
    """Load AIRD estimates CSV and analyze 80% time horizon data - focused on 2024-years from Claude 3.7 Sonnet"""
    
    # Read the CSV file
    csv_path = 'aird_estimates.csv'
    df = pd.read_csv(csv_path)
    
    # Remove rows where all values are NaN or empty
    df = df.dropna(how='all')
    
    # Also remove rows where the essential columns are empty
    df = df.dropna(subset=['80% time horizon', 'Date'])
    
    # Calculate 2024-years (initially anchored to Claude 3.7 Sonnet)
    model_2024_years_claude_anchor, _ = convert_to_2024_years(df)

    # Find the offset value for GPT-2 to set it as the zero point
    gpt2_offset_value = None
    gpt2_index = -1
    for i, name in enumerate(df['Column 1']):
        if 'GPT-2' in str(name):
            gpt2_index = i
            break
            
    if gpt2_index != -1 and gpt2_index < len(model_2024_years_claude_anchor) and model_2024_years_claude_anchor[gpt2_index] is not None:
        gpt2_offset_value = model_2024_years_claude_anchor[gpt2_index]
        
        # Re-anchor the 2024-years so that GPT-2 is at 0
        model_2024_years_gpt2_anchor = [(val - gpt2_offset_value) if val is not None else None for val in model_2024_years_claude_anchor]
        
        # Replace the '2024-years' column with the new values
        df['2024-years'] = model_2024_years_gpt2_anchor
        
        # Save the updated dataframe back to the CSV
        df.to_csv(csv_path, index=False)
        print(f"\nUpdated '{csv_path}' with '2024-years' column (re-anchored to GPT-2=0).")
    else:
        print("\nCould not find GPT-2 to re-anchor '2024-years' column. Saving with original anchor.")
        # If we couldn't re-anchor, just save the original calculation for consistency
        df['2024-years'] = model_2024_years_claude_anchor
        df.to_csv(csv_path, index=False)
    
    # Print column names to debug
    print("Available columns:", df.columns.tolist())
    
    # Extract the columns we need
    time_horizon = df['80% time horizon'].astype(float)
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
    
    # Fit both curves
    print("="*50)
    print("FITTING BOTH CURVES (2024-years from Claude 3.7 Sonnet)")
    print("="*50)
    
    # Fit double exponential
    popt_double, x_data, y_data, model_names_fit, r_squared_double = fit_double_exponential_to_time_horizon_2024_years(df)
    
    # Fit new curve
    popt_new_result, x_data_new, y_data_new, model_names_new, r_squared_new = fit_fixed_decay_to_time_horizon_2024_years(df)
    
    # Handle the new curve result (which now includes shift information)
    if popt_new_result is not None:
        popt_new, x_shift = popt_new_result
    else:
        popt_new, x_shift = None, 0
    
    # Create side-by-side plots: full extrapolation and zoomed-in historical fit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Common SC threshold
    sc_threshold_minutes = 10 * 365.25 * 24 * 60  # 10 years in minutes
    
    # Reference dates
    reference_date = datetime(2025, 2, 1)  # Claude 3.7 Sonnet reference
    
    # Plot historical data on both axes
    ax1.scatter(x_data, y_data, s=100, alpha=0.7, color='red', label='Historical Data', zorder=5)
    ax2.scatter(x_data, y_data, s=100, alpha=0.7, color='red', label='Historical Data', zorder=5)
    
    # Add model names on both axes
    for i, model in enumerate(model_names_fit):
        ax1.annotate(model, (x_data[i], y_data[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
        ax2.annotate(model, (x_data[i], y_data[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Fit and plot double exponential if successful
    if popt_double is not None:
        # Find SC crossing
        search_max_years = 50
        sc_crossing_double = find_sc_crossing(popt_double, (min(x_data), search_max_years), 
                                            sc_threshold_minutes, x_data, 'double_exponential')
        
        # Find numerical boundary
        data_end = max(x_data)
        numerical_boundary_double = find_numerical_boundary(popt_double, data_end, x_max_search=50)
        
        # Determine plot end for double exponential
        if sc_crossing_double is not None:
            sc_plus_3_years = sc_crossing_double + 3  # 3 years past SC
            plot_end_double = min(sc_plus_3_years, numerical_boundary_double)
        else:
            plot_end_double = numerical_boundary_double
        
        # Plot double exponential curve - full extrapolation on left, historical fit on right
        # Left plot: Full extrapolation
        x_smooth_double = np.linspace(min(x_data), plot_end_double, 1000)
        y_smooth_double = double_exponential(x_smooth_double, *popt_double)
        ax1.plot(x_smooth_double, y_smooth_double, color='blue', linewidth=2, alpha=0.8, 
                 label=f'Double Exponential (R² = {r_squared_double:.4f})')
        
        # Right plot: Historical fit only
        x_historical_double = np.linspace(min(x_data), max(x_data), 1000)
        y_historical_double = double_exponential(x_historical_double, *popt_double)
        ax2.plot(x_historical_double, y_historical_double, color='blue', linewidth=2, alpha=0.8, 
                 label=f'Double Exponential (R² = {r_squared_double:.4f})')
        
        # Plot SC crossing for double exponential (left plot only)
        if sc_crossing_double is not None:
            sc_date_double = reference_date + relativedelta(months=int(sc_crossing_double * 12))
            ax1.scatter([sc_crossing_double], [sc_threshold_minutes], s=150, alpha=0.9, color='orange', 
                        label=f'SC (Double Exp): {sc_date_double.strftime("%b %Y")}', zorder=6, marker='*')
    
    # Fit and plot new curve if successful
    if popt_new is not None:
        # For the new curve, we need to work with shifted coordinates internally but plot in original coordinates
        
        # Find SC crossing (using shifted coordinates for calculation)
        search_max_years = 50
        x_data_shifted = x_data_new + x_shift
        sc_crossing_new_shifted = find_sc_crossing(popt_new, (min(x_data_shifted), search_max_years + x_shift), 
                                                 sc_threshold_minutes, x_data_shifted, 'fixed_decay')
        
        # Convert back to original coordinates
        if sc_crossing_new_shifted is not None:
            sc_crossing_new = sc_crossing_new_shifted - x_shift
        else:
            sc_crossing_new = None
        
        # Find numerical boundary (in shifted coordinates, then convert back)
        data_end_shifted = max(x_data_shifted)
        numerical_boundary_new_shifted = find_fixed_decay_boundary(popt_new, data_end_shifted, x_max_search=50 + x_shift)
        numerical_boundary_new = numerical_boundary_new_shifted - x_shift
        
        # Determine plot end for new curve (in original coordinates)
        if sc_crossing_new is not None:
            sc_plus_3_years = sc_crossing_new + 3  # 3 years past SC
            plot_end_new = min(sc_plus_3_years, numerical_boundary_new)
        else:
            plot_end_new = numerical_boundary_new
        
        # Plot new curve - full extrapolation on left, historical fit on right
        # Left plot: Full extrapolation (convert back to original coordinates for plotting)
        x_smooth_new_original = np.linspace(min(x_data_new), plot_end_new, 1000)
        x_smooth_new_shifted = x_smooth_new_original + x_shift  # Shift for function evaluation
        y_smooth_new = fixed_decay(x_smooth_new_shifted, *popt_new)
        ax1.plot(x_smooth_new_original, y_smooth_new, color='green', linewidth=2, alpha=0.8, 
                 label=f'Fixed Decay (Weighted R² = {r_squared_new:.4f})')
        
        # Right plot: Historical fit only
        x_historical_new_original = np.linspace(min(x_data_new), max(x_data_new), 1000)
        x_historical_new_shifted = x_historical_new_original + x_shift  # Shift for function evaluation
        y_historical_new = fixed_decay(x_historical_new_shifted, *popt_new)
        ax2.plot(x_historical_new_original, y_historical_new, color='green', linewidth=2, alpha=0.8, 
                 label=f'Fixed Decay (Weighted R² = {r_squared_new:.4f})')
        
        # Plot SC crossing for new curve (left plot only)
        if sc_crossing_new is not None:
            sc_date_new = reference_date + relativedelta(months=int(sc_crossing_new * 12))
            ax1.scatter([sc_crossing_new], [sc_threshold_minutes], s=150, alpha=0.9, color='purple', 
                        label=f'SC (Fixed Decay): {sc_date_new.strftime("%b %Y")}', zorder=6, marker='D')
        
        # Show singularity point (in original coordinates) on left plot only
        H_0, a, T_0 = popt_new
        singularity_point_shifted = T_0 / a
        singularity_point = singularity_point_shifted - x_shift
        ax1.axvline(x=singularity_point, color='green', linestyle=':', alpha=0.7, 
                    label=f'Singularity: {singularity_point:.1f} years')
    
    # Reference lines on both plots
    ax1.axvline(x=0, color='purple', linestyle=':', alpha=0.7, label='Claude 3.7 Sonnet (Feb 2025)')
    ax1.axhline(y=sc_threshold_minutes, color='orange', linestyle='-', alpha=0.7, linewidth=2, 
                label='SC Threshold (10 years)')
    
    ax2.axvline(x=0, color='purple', linestyle=':', alpha=0.7, label='Claude 3.7 Sonnet (Feb 2025)')
    # Don't show SC threshold on right plot since it's out of range
    
    # Formatting for both plots
    ax1.set_yscale('log')
    ax1.set_xlabel('2024-years from Claude 3.7 Sonnet', fontsize=12)
    ax1.set_ylabel('80% Time Horizon (minutes, log scale)', fontsize=12)
    ax1.set_title('Full Extrapolation with SC Crossings', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('2024-years from Claude 3.7 Sonnet', fontsize=12)
    ax2.set_ylabel('80% Time Horizon (minutes, log scale)', fontsize=12)
    ax2.set_title('Historical Data Fit (Zoomed)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Set appropriate axis limits for the zoomed plot
    ax2.set_xlim(min(x_data) - 0.5, max(x_data) + 0.5)
    ax2.set_ylim(min(y_data) * 0.5, max(y_data) * 2)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary comparison
    print("\n" + "="*50)
    print("CURVE COMPARISON SUMMARY")
    print("="*50)
    print(f"Double Exponential: y = a * exp(b * exp(c * x))")
    if popt_double is not None:
        print(f"  Parameters: a={popt_double[0]:.6f}, b={popt_double[1]:.6f}, c={popt_double[2]:.6f}")
        print(f"  R²: {r_squared_double:.4f}")
        if sc_crossing_double is not None:
            print(f"  SC crossing: {sc_crossing_double:.3f} years from Claude 3.7 Sonnet")
    else:
        print("  Failed to fit")
    
    print(f"\nFixed Decay: y = H_0 * (1 - a * (t/T_0))^(1/log2(1-a)) [with x-shift +{x_shift:.3f}]")
    if popt_new is not None:
        print(f"  Parameters: H_0={popt_new[0]:.6f}, a={popt_new[1]:.6f}, T_0={popt_new[2]:.6f}")
        # Recalculate final R-squared values for printing summary
        y_pred_summary = fixed_decay(x_data_new + x_shift, *popt_new)
        ss_res_summary = np.sum((y_data_new - y_pred_summary) ** 2)
        ss_tot_summary = np.sum((y_data_new - np.mean(y_data_new)) ** 2)
        r_squared_summary = 1 - (ss_res_summary / ss_tot_summary)
        
        weights_summary = 1.0 / (np.abs(y_data_new) + 1e-6)
        weighted_mean_summary = np.sum(weights_summary * y_data_new) / np.sum(weights_summary)
        ss_res_weighted_summary = np.sum(weights_summary * (y_data_new - y_pred_summary) ** 2)
        ss_tot_weighted_summary = np.sum(weights_summary * (y_data_new - weighted_mean_summary) ** 2)
        r_squared_weighted_summary = 1 - (ss_res_weighted_summary / ss_tot_weighted_summary)
        
        print(f"  R² (unweighted): {r_squared_summary:.4f}")
        print(f"  R² (weighted): {r_squared_weighted_summary:.4f}")
        singularity_original = (popt_new[2]/popt_new[1]) - x_shift
        print(f"  Singularity at: {singularity_original:.3f} years from Claude 3.7 Sonnet")
        if sc_crossing_new is not None:
            print(f"  SC crossing: {sc_crossing_new:.3f} years from Claude 3.7 Sonnet")
    else:
        print("  Failed to fit")
    
    return df

if __name__ == "__main__":
    data = load_and_analyze_time_horizon_focused()
