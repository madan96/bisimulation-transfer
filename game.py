import numpy as np

import environment
import agent

# ------------------------------------ environment 1 Four Room - 8 states - Source Domain -----------------------------------------
# gridH, gridW = 3, 3
# start_pos = None
# end_positions = [(1, 2)]
# end_rewards = [10.0]
# blocked_positions = [(1, 1)]
# default_reward= -0.2
# ------------------------------------ environment 2 Four Room - 44 States - Target Domain -----------------------------------------

gridH, gridW = 5, 11 
start_pos = None
end_positions = [(2, 8)]
end_rewards = [10.0]
blocked_positions = [(0, 5), (2, 5), (4, 5), (2, 0), (2, 1), (2, 3), (2, 4), (2, 6), (2, 7), (2, 9), (2, 10)]
default_reward= -0.2

# ------------------------------------ environment 3 -----------------------------------------
'''
gridH, gridW = 8, 9
start_pos = None
end_positions = [(2, 2), (3, 5), (4, 5), (5, 5), (6, 5)]
end_rewards = [10.0, -30.0, -30.0, -30.0, -30.0]
blocked_positions = [(i, 1) for i in range(1, 7)]+ [(1, i) for i in range(1, 8)] + [(i, 7) for i in range(1, 7)]
default_reward= -0.5
'''
# ------------------------------------ environment 4 -----------------------------------------
'''
gridH, gridW = 9, 7
start_pos = None
end_positions = [(0, 3), (2, 4), (6, 2)]
end_rewards = [20.0, -50.0, -50.0]
blocked_positions = [(2, i) for i in range(3)] + [(6, i) for i in range(4, 7)]
default_reward = -0.1
'''
# --------------------------------------------------------------------------------------------

env = environment.Environment(gridH, gridW, end_positions, end_rewards, blocked_positions, start_pos, default_reward)

alpha = 0.2
epsilon = 0.5
discount = 0.99
action_space = env.action_space
state_space = env.state_space
tp_matrix = env.tp_matrix

agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, state_space, tp_matrix, blocked_positions)
# agent = agent.EVSarsaAgent(alpha, epsilon, discount, action_space, state_space)

env.render(agent.qvalues)
state = env.get_state()

for i in range(20000):

	possible_actions = env.get_possible_actions()
	action = agent.get_action(state, possible_actions)
	next_state, reward, done, next_possible_states = env.step(action)
	# print ("next state ", next_state)
	env.render(agent.qvalues)

	next_state_possible_actions = env.get_possible_actions()
	agent.update(state, action, reward, next_state, next_state_possible_actions, next_possible_states, done)
	state = next_state

	if done == True:	
		env.reset_state()
		env.render(agent.qvalues)
		state = env.get_state()
		continue

qval = np.asarray(agent.qvalues)
print (qval.shape)
np.save('optimal_qvalues_44_new_prob_states.npy', qval)
