import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm


def lookup_compute(t: datetime, schedule_params: dict) -> tuple[float, float]:
    """Return the experiment compute and memory bandwidth for a project at time t.
    
    Args:
        t: datetime
        schedule_params: dict

    Returns:
        tuple[float, float]: (experiment_H100e, agent_H100be)
    """
    #TODO: this should vary over time
    return (schedule_params["experiment_H100e"], schedule_params["agent_H100be"])

def sw_progress_rate_schedule(t: datetime,  schedule_params: dict, milestone: str = None, start_speed: float = None, end_speed: float = None, debug: bool = False) -> float:
    """Return the software progress rate at time t_days.
    
    schedule_params is a dictionary with the following keys:
        type: "constant" or "decay"
        eta_days: float
    
    If schedule_params is None or "type" is "constant", this returns the initial_sw_progress_rate.
    If "type" is "decay", this returns the initial_sw_progress_rate * np.exp(-t_days / eta_days).
    """
    assert schedule_params is not None
    initial_rate = schedule_params["initial_sw_progress_rate"]
    if schedule_params["type"] == "constant":
        return initial_rate
    elif schedule_params["type"] == "decay":
        if "eta_days" not in schedule_params:
            raise ValueError("eta_days is required for decay schedule")
        return initial_rate * np.exp(-(t - schedule_params["start_date"]).days / schedule_params["eta_days"])
    elif schedule_params["type"] == "compute_and_talent":
        # get compute and memory bandwidth at time t
        experiment_H100e, agent_H100be = lookup_compute(t, schedule_params)
        ref_exp_H100e = schedule_params["reference_exp_H100e"]

        # get talent at time t
        # TODO: talent
        
        if milestone == "PRESENT_DAY":
            return schedule_params["initial_sw_progress_rate"]
        elif milestone == "SC":
            return progress_rate_with_SC(experiment_H100e, ref_exp_H100e, agent_H100be, schedule_params, start_speed, debug)
        elif milestone == "SAR":
            return progress_rate_with_SAR(experiment_H100e, ref_exp_H100e, agent_H100be, start_speed, debug)
        elif milestone == "SIAR":
            return progress_rate_with_SIAR(experiment_H100e, ref_exp_H100e, agent_H100be, start_speed, debug)
        elif milestone == "ASI":
            return progress_rate_with_ASI(experiment_H100e, ref_exp_H100e, agent_H100be, start_speed, debug)        
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_params['type']}")

def progress_rate_with_SC(experiment_H100e: float, reference_exp_H100e: float, agent_H100be: float, schedule_params: dict, SC_speedup: float = 5, debug: bool = False) -> float:
    """Return the progress rate with SC.
    perhaps the first three params should be in schedule_params
    
    Args:
        experiment_H100e: Compute for experiments, in H100e
        reference_exp_H100e: Experiment compute available to a 2027 leading project, in H100e
        agent_H100be: Memory bandwidth for running automated researchers, in H100be
    """
    def get_multiplier(agent_H100be: float) -> float:
        """
        Return the maximal N for which the quantity agent_H100be of memory bandwidth
        permits running N/1700 agents, each accomplishing tasks N times faster than
        the fastest 2027-OpenBrain researcher.
        """
        #TODO: actually calculate this
        H100BE_PER_2027_IC = 200
        REF_NUMBER_OF_2027_IC = 4000
        return 30 * np.sqrt(agent_H100be/(H100BE_PER_2027_IC * REF_NUMBER_OF_2027_IC))
    # speedup_with_reference_exp_H100e_and_scientist_talent = 5 #TODO: put interpolation based on agent_H100be here

    # TODO: use a CES between engineer and scientist talent?
    N = get_multiplier(agent_H100be)
    exp_compute_ratio = experiment_H100e / reference_exp_H100e
    # F = 1.7
    # alpha = 0.06
    # beta = 0.3
    # rho = -1.43
    F = schedule_params["SC_CES_params"]["scale_F"]
    alpha = schedule_params["SC_CES_params"]["share_sci_to_eng_alpha"]
    beta = schedule_params["SC_CES_params"]["share_compute_labor_beta"]
    rho = schedule_params["SC_CES_params"]["substitution_sci_eng_rho"]
    engineer_talent_ratio = N**2
    scientist_talent_ratio = schedule_params["scientist_talent_ratio"]
    labor_factor = (alpha * scientist_talent_ratio**rho + (1 - alpha) * engineer_talent_ratio**rho)**(1/rho)

    
    # Known reference points:
    # N = 30, scientist_talent_ratio = 1.0 -> speedup_with_reference_exp_H100e = SC_speedup
    # N = 1, scientist_talent_ratio = 1.0 -> speedup_with_reference_exp_H100e = speedup from replacing every coder with a top coder

    # N_ref1, S_ref1 = 1.0, 1.5
    # N_ref2, S_ref2 = 30.0, SC_speedup
    # b = np.log(S_ref2 / S_ref1) / np.log(N_ref2 / N_ref1)
    # a = S_ref1 / (N_ref1 ** b)
    # N = get_multiplier(agent_H100be)
    # speedup_with_reference_exp_H100e = a * N ** b

    # exp_compute_ratio = experiment_H100e / reference_exp_H100e

    speed = F * exp_compute_ratio**beta * labor_factor**(1-beta)
    
    if debug:
        print(f"SC speedup_with_reference_exp_H100e: {F * labor_factor**(1-beta)}")
        print(f"SC exp_compute_ratio: {exp_compute_ratio}")
        print(f"SC progress rate: {speed}")
    return speed

def progress_rate_with_SAR(experiment_H100e: float, reference_exp_H100e: float, agent_H100be: float, SAR_speedup: float = 25, debug: bool = False) -> float:
    """Return the progress rate with SAR.
    
    Args:
        experiment_H100e: Compute for experiments, in H100e
        reference_exp_H100e: Experiment compute available to a 2027 leading project, in H100e
        agent_H100be: Memory bandwidth for running automated researchers, in H100be
    """
    def get_multiplier(agent_H100be: float) -> float:
        """
        Return the maximal N for which the quantity agent_H100be of memory bandwidth
        permits running N/1700 agents, each accomplishing tasks N times faster than
        the fastest 2027-OpenBrain researcher.
        """
        #TODO: actually calculate this
        H100BE_PER_2027_IC = 200
        REF_NUMBER_OF_2027_IC = 4000
        return 30 * np.sqrt(agent_H100be/(H100BE_PER_2027_IC * REF_NUMBER_OF_2027_IC))
    

    # Intepolate between:
    # (30, 25)-forecasted SAR speedup (with 30x, 30x) and 
    # (1, 2)-estimated speedup from upgrading every researcher to the best researcher (1x, 1x)
    # this assumes human labor contributes negligibly to the speedup.
    N_ref1, S_ref1 = 1.0, 2.0
    N_ref2, S_ref2 = 30.0, SAR_speedup
    b = np.log(S_ref2 / S_ref1) / np.log(N_ref2 / N_ref1)
    a = S_ref1 / (N_ref1 ** b)
    N = get_multiplier(agent_H100be)
    speedup_with_reference_exp_H100e = a * N ** b
        
    # scale by experiment compute ratio
    exp_compute_ratio = experiment_H100e / reference_exp_H100e
    speed = exp_compute_ratio**0.3 * speedup_with_reference_exp_H100e
    if debug:
        print(f"SAR multiplier (N): {N}")
        print(f"SAR speedup_with_reference_exp_H100e: {speedup_with_reference_exp_H100e}")
        print(f"SAR exp_compute_ratio: {exp_compute_ratio}")
        print(f"SAR progress rate: {speed}")

    return speed

def progress_rate_with_SIAR(experiment_H100e: float, reference_exp_H100e: float, agent_H100be: float, SIAR_speedup: float = 250, debug: bool = False) -> float:
    """Return the progress rate with SAR.
    
    Args:
        experiment_H100e: Compute for experiments, in H100e
        reference_exp_H100e: Experiment compute available to a 2027 leading project, in H100e
        agent_H100be: Memory bandwidth for running automated researchers, in H100be
    """
    def get_multiplier(agent_H100be: float) -> float:
        """
        Return the maximal N for which the quantity agent_H100be of memory bandwidth
        permits running N/1700 agents, each accomplishing tasks N times faster than
        the fastest 2027-OpenBrain researcher.
        """
        #TODO: actually calculate this
        H100BE_PER_2027_IC = 200
        REF_NUMBER_OF_2027_IC = 4000
        return 30 * np.sqrt(agent_H100be/(H100BE_PER_2027_IC * REF_NUMBER_OF_2027_IC))
    
    # Intepolate between:
    # (30, 250)-forecasted SIAR speedup (with 30x, 250x) and 
    # (1, 10)-estimated speedup from upgrading every researcher to a "median * (best/median)^2 researcher" (1x, 1x)
    # this assumes human labor contributes negligibly to the speedup.
    N_ref1, S_ref1 = 1.0, 10.0
    N_ref2, S_ref2 = 30.0, SIAR_speedup
    b = np.log(S_ref2 / S_ref1) / np.log(N_ref2 / N_ref1)
    a = S_ref1 / (N_ref1 ** b)
    N = get_multiplier(agent_H100be)
    speedup_with_reference_exp_H100e = a * N ** b
    
    exp_compute_ratio = experiment_H100e / reference_exp_H100e
    speed = exp_compute_ratio**0.3 * speedup_with_reference_exp_H100e
    if debug:
        print(f"SIAR multiplier (N): {N}")
        print(f"SIAR speedup_with_reference_exp_H100e: {speedup_with_reference_exp_H100e}")
        print(f"SIAR exp_compute_ratio: {exp_compute_ratio}")
        print(f"SIAR progress rate: {speed}")

    return speed
def progress_rate_with_ASI(experiment_H100e: float, reference_exp_H100e: float, agent_H100be: float, ASI_speedup: float = 2000, debug: bool = False) -> float:
    """Return the progress rate with ASI.
    
    Args:
        experiment_H100e: Compute for experiments, in H100e
        reference_exp_H100e: Experiment compute available to a 2027 leading project, in H100e
        agent_H100be: Memory bandwidth for running automated researchers, in H100be
        ASI_speedup: Reference speedup to 2027-OpenBrain from running ASI
    """
    def get_multiplier(agent_H100be: float) -> float:
        """
        Return the maximal N for which the quantity agent_H100be of memory bandwidth
        permits running N/1700 agents, each accomplishing tasks N times faster than
        the fastest 2027-OpenBrain researcher.
        """
        #TODO: actually calculate this
        H100BE_PER_2027_IC = 200
        REF_NUMBER_OF_2027_IC = 4000
        return 30 * np.sqrt(agent_H100be/(H100BE_PER_2027_IC * REF_NUMBER_OF_2027_IC))
    # Intepolate between:
    # (30, 2000)-forecasted ASI speedup (with 30x, 30x) and 
    # (1, 80x)-estimated speedup from replacing every researcher with a superintelligence (1x, 1x)
    # this assumes human labor contributes negligibly to the speedup.
    N_ref1, S_ref1 = 1.0, 80.0
    N_ref2, S_ref2 = 30.0, ASI_speedup
    b = np.log(S_ref2 / S_ref1) / np.log(N_ref2 / N_ref1)
    a = S_ref1 / (N_ref1 ** b)
    N = get_multiplier(agent_H100be)
    speedup_with_reference_exp_H100e = a * N ** b

    exp_compute_ratio = experiment_H100e / reference_exp_H100e
    speed = exp_compute_ratio**0.3 * speedup_with_reference_exp_H100e

    if debug:
        print(f"ASI multiplier (N): {N}")
        print(f"ASI speedup_with_reference_exp_H100e: {speedup_with_reference_exp_H100e}")
        print(f"ASI exp_compute_ratio: {exp_compute_ratio}")
        print(f"ASI progress rate: {speed}")

    return speed

# Add helper to compute reliability based on failure model parameters
def compute_reliability(t_days: float, model_params: dict | None) -> float:
    """Return the reliability R(t) (fraction of GPUs still working) after t_days.

    Currently only Weibull models are supported.  The YAML failure_models entry
    should contain at least the keys:
        type: "weibull"
        beta: <float>
        eta_years: <float>
    If model_params is None or unsupported, this returns 1.0 (no attrition).
    """
    if not model_params:
        return 1.0  # No attrition / unlimited resupply

    model_type = model_params.get("type", "weibull").lower()

    # Convert generic inputs
    if model_type == "weibull":
        beta = model_params.get("beta", 1.0)
        eta_years = model_params.get("eta_years", 1.0)
        eta_days = eta_years * 365.0
        if eta_days <= 0:
            return 1.0
        return np.exp(- (t_days / eta_days) ** beta)
    # Fallback – unsupported model
    return 1.0
