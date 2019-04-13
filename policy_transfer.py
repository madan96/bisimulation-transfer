import math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
import time

import environment
import agent

alpha = 0.2
epsilon = 0.05
discount = 0.99


gridH, gridW = 3, 3
start_pos = None
end_positions = [(1, 2)]
end_rewards = [5.0]
src_blocked_positions = [(1, 1)]
default_reward = -0.2

src_env = environment.Environment(gridH, gridW, end_positions, end_rewards, src_blocked_positions, start_pos, default_reward)
action_space = src_env.action_space
src_state_space = src_env.state_space
src_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, src_state_space, src_env.tp_matrix, src_blocked_positions)

gridH, gridW = 5, 11 
start_pos = None
end_positions = [(2, 8)]
end_rewards = [10.0]
tgt_blocked_positions = [(0, 5), (2, 5), (4, 5), (2, 0), (2, 1), (2, 3), (2, 4), (2, 6), (2, 7), (2, 9), (2, 10)]
default_reward= -0.2

tgt_env = environment.Environment(gridH, gridW, end_positions, end_rewards, tgt_blocked_positions, start_pos, default_reward)
action_space = tgt_env.action_space
tgt_state_space = tgt_env.state_space
tgt_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, tgt_state_space, tgt_env.tp_matrix, tgt_blocked_positions)

src_agent.qvalues = np.load('optimal_qvalues_8_new_prob_states.npy')
# tgt_qval = np.zeros((tgt_env.state_space, 4))

src_possible_actions = src_env.get_possible_actions()
tgt_possible_actions = tgt_env.get_possible_actions()

def compute_d():
    """
    Computes state-action bisimulation metric
    """
    # u - v <= d(si, sj)
    A = [1, -1]
    bounds = [(-1, 1), (-1, 1)]

    d = np.zeros((src_state_space, action_space, tgt_state_space, action_space))
    for s1_pos, s1_state in src_env.state2idx.items():
        src_env.position = s1_pos
        for a in range(action_space):
            next_state, reward_a, done, next_possible_states = src_env.step(a)
            src_env.position = s1_pos
            for s2_pos, s2_state in tgt_env.state2idx.items():
                tgt_env.position = s2_pos
                for b in range(action_space):
                    p1 = -np.sum(src_env.tp_matrix[s1_state,a])
                    p2 = np.sum(tgt_env.tp_matrix[s2_state,b])
                    c = [p1, p2]
                    # print (c)
                    next_state, reward_b, done, next_possible_states = tgt_env.step(b)
                    d[s1_state,a,s2_state,b] = math.fabs(reward_a - reward_b) + wasserstein_distance(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b])
                    b = [d[s1_state,a,s2_state,b]]
                    # res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, options={"disp": True})
                    # print (res.fun)
                    tgt_env.position = s2_pos
    return d


def compute_dl(d):
    """
    Computes state bisimulation metric
    """
    S1 = src_env.state_space
    S2 = tgt_env.state_space
    lax_bisim_state_metric = np.zeros((S1, S2))
    for i in range(S1):
        for j in range(S2):
            d_st = d[i,:,j,:]
            dl_st = max(np.max(np.min(d_st, axis=1)),np.max(np.min(d_st, axis=0)))
            lax_bisim_state_metric[i][j] = dl_st
    return lax_bisim_state_metric

def laxBisimTransfer(S1, S2):
    dl_sa = compute_d()
    bisim_state_metric = compute_dl(dl_sa)

    for t in range(S2):
        s_t = np.argmin(bisim_state_metric[:,t])
        b_t = np.argmin(dl_sa[s_t, src_agent.get_best_action(s_t, src_possible_actions), t])
        qv = src_agent.qvalues[s_t][b_t]
        tgt_agent.update_qvalue(t, b_t, qv)
    
laxBisimTransfer(src_state_space, tgt_state_space)

tgt_env.render(tgt_agent.qvalues)
state = tgt_env.get_state()

for i in range(1000):
    possible_actions = tgt_env.get_possible_actions()
    action = tgt_agent.get_best_action(state, possible_actions)
    next_state, reward, done, next_possible_states = tgt_env.step(action)
    tgt_env.render(tgt_agent.qvalues)

    next_state_possible_actions = tgt_env.get_possible_actions()
    state = next_state

    if done == True:	
        tgt_env.reset_state()
        tgt_env.render(tgt_agent.qvalues)
        time.sleep(0.5)
        state = tgt_env.get_state()
        continue