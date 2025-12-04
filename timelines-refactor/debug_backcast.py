"""Debug script for backcast_trajectories with Eli_exp_only_patched_rd_speedup median params."""
import yaml
import sys
sys.path.insert(0, '.')
from simple_forecasting_timelines import get_median_samples, backcast_trajectories

# Load config using UnsafeLoader to handle OrderedDict
with open('output/2025-12-02_12-07-49/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

# Get eli_exp_only_patched_rd_speedup config
forecasters = dict(config['forecasters'])
forecaster_config = forecasters['eli_exp_only_patched_rd_speedup']
print('Forecaster:', forecaster_config['name'])

# Get median samples
forecaster_dist_config = {'distributions': forecaster_config['distributions']}
median_samples = get_median_samples(forecaster_dist_config)
print('\nMedian samples:')
for k, v in median_samples.items():
    print(f'  {k}: {v}')

# Run backcast_trajectories
current_horizon = config['simulation']['current_horizon']
dt = config['simulation']['dt']
print(f'\nRunning backcast_trajectories with current_horizon={current_horizon}, dt={dt}')

backcast_trajs = backcast_trajectories(median_samples, current_horizon, dt, backcast_years=5)
print(f'\nBackcast complete! Got {len(backcast_trajs)} trajectory(ies)')
