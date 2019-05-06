import os
import yaml
import sys

from env import Environment
from munch import Munch

def create_env(env_name):
    env_name += '.yaml'
    yaml_name = os.path.join('env/room_envs', env_name)
    with open(yaml_name, 'r') as f:

        yaml_file = yaml.load(f)

        cfg = Munch(yaml_file)
        end_positions = []
        for x in cfg.end_positions:
            end_positions.append(tuple(x))
        blocked_positions = []
        for x in cfg.blocked_positions:
             blocked_positions.append(tuple(x))

    env = Environment(cfg.gridH, cfg.gridW, end_positions, cfg.end_rewards, blocked_positions, cfg.default_reward)
    return env
