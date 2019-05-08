import numpy as np
import os
import sys
sys.path.append('./')
from env import Environment, create_env
from .agent import QLearningAgent

def train(opts):
	env = create_env(opts.env_name)
	action_space = env.action_space
	state_space = env.state_space
	tp_matrix = env.tp_matrix

	agent = QLearningAgent(opts.alpha, opts.epsilon, opts.discount, action_space, state_space, tp_matrix, env.blocked_positions)

	env.render(agent.qvalues)
	state = env.get_state()

	for i in range(opts.num_iters):

		possible_actions = env.get_possible_actions()
		action = agent.get_action(state, possible_actions)
		next_state, reward, done, next_possible_states = env.step(action)
		env.render(agent.qvalues)

		next_state_possible_actions = env.get_possible_actions()
		agent.update(state, action, reward, next_state, next_state_possible_actions, next_possible_states, done)
		state = next_state

		if done == True:	
			env.reset_state()
			env.render(agent.qvalues)
			state = env.get_state()
			continue

	if not os.path.exists(opts.policy_dir):
		os.makedirs(opts.policy_dir)
	np.save(os.path.join(opts.policy_dir, opts.env_name + '.npy'), np.asarray(agent.qvalues))
