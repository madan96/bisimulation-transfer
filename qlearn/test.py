import numpy as np
import os
import time
import sys
sys.path.append('./')

from env import Environment, create_env
from .agent import QLearningAgent

def test(opts):
    env = create_env(opts.env_name)
    action_space = env.action_space
    state_space = env.state_space
    tp_matrix = env.tp_matrix
    agent = QLearningAgent(opts.alpha, opts.epsilon, opts.discount, action_space, state_space, tp_matrix, env.blocked_positions)

    qvalues = np.load(os.path.join(opts.policy_dir, opts.env_name + '.npy' ))
    agent.qvalues = qvalues

    env.render(agent.qvalues)
    state = env.get_state()

    for i in range(200):
        possible_actions = env.get_possible_actions()
        action = agent.get_best_action(state, possible_actions)
        time.sleep(0.1)
        next_state, reward, done, next_possible_states = env.step(action)
        env.render(agent.qvalues)

        next_state_possible_actions = env.get_possible_actions()
        state = next_state

        print (reward)
        if done == True:	
            env.reset_state()
            env.render(agent.qvalues)
            time.sleep(0.5)
            state = env.get_state()
            continue