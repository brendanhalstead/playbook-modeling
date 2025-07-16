from simple_forecasting_timelines import *

class THE_Model:
    def __init__(self, config: dict):
        self.config = config
        self.forecasters = apply_inheritance_to_forecasters(config["forecasters"])
        self.samples = self.sample_from_config(config["forecasters"], config["simulation"]["n_sims"])

    def sample_from_config(self, config: dict, n_sims: int = 1000) -> dict:
        return get_distribution_samples(config, n_sims)
    
class TH_Trajectory:
    def __init__(self, anchor_year: float, anchor_horizon_minutes: float, dt: float, horizon_minutes_at_SC: float, *args, **kwargs):
        self.dt = dt
        self.anchor_year = anchor_year
        self.anchor_horizon_minutes = anchor_horizon_minutes
        self.horizon_minutes_at_SC = horizon_minutes_at_SC
    
    def get_horizon_at_year(self, year: float) -> float:
        '''Returns the 80% time horizon in minutes at the given decimal year.'''
        raise NotImplementedError("Not implemented")
    
class TH_Trajectory_Original(TH_Trajectory):
    def __init__(self, anchor_year: float, anchor_horizon_minutes: float, dt: float, horizon_minutes_at_SC: float, *args, **kwargs):
        super().__init__(anchor_year, anchor_horizon_minutes, dt, horizon_minutes_at_SC, *args, **kwargs)

    def calculate_base_time_months(self, year: float) -> float:
        '''Returns the base time in months at the given decimal year.'''
        raise NotImplementedError("Not implemented")

    def get_horizon_at_year(self, year: float) -> float:
        '''Returns the 80% time horizon in minutes at the given decimal year.'''
        raise NotImplementedError("Not implemented")
    