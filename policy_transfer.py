import math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import time
import environment
import agent
from pyemd import emd
import pdb

alpha = 0.2
epsilon = 0.05
discount = 0.99


gridH, gridW = 3, 3
start_pos = None
end_positions = [(1, 2)]
end_rewards = [1.0]
src_blocked_positions = [(1, 1)]
default_reward = 0.0

# gridH, gridW = 5, 11 
# start_pos = None
# end_positions = [(2, 8)]
# end_rewards = [1.0]
# src_blocked_positions = [(0, 5), (2, 5), (4, 5), (2, 0), (2, 1), (2, 3), (2, 4), (2, 6), (2, 7), (2, 9), (2, 10)]
# default_reward= 0.0

src_env = environment.Environment(gridH, gridW, end_positions, end_rewards, src_blocked_positions, start_pos, default_reward)
action_space = src_env.action_space
src_state_space = src_env.state_space
src_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, src_state_space, src_env.tp_matrix, src_blocked_positions)

# gridH, gridW = 5, 11 
# start_pos = None
# end_positions = [(2, 8)]
# end_rewards = [1.0]
# tgt_blocked_positions = [(0, 5), (2, 5), (4, 5), (2, 0), (2, 1), (2, 3), (2, 4), (2, 6), (2, 7), (2, 9), (2, 10)]
# default_reward= 0.0

gridH, gridW = 3, 3
start_pos = None
end_positions = [(1, 2)]
end_rewards = [1.0]
tgt_blocked_positions = [(1, 1)]
default_reward = 0.0

tgt_env = environment.Environment(gridH, gridW, end_positions, end_rewards, tgt_blocked_positions, start_pos, default_reward)
action_space = tgt_env.action_space
tgt_state_space = tgt_env.state_space
tgt_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, tgt_state_space, tgt_env.tp_matrix, tgt_blocked_positions)

src_agent.qvalues = np.load('optimal_qvalues_8_new_prob_states.npy')
# src_agent.qvalues = np.load('optimal_qvalues_44_new_prob_states.npy')
# tgt_qval = np.zeros((tgt_env.state_space, 4))

src_possible_actions = src_env.get_possible_actions()
tgt_possible_actions = tgt_env.get_possible_actions()

def compute_d(use_reward=True, use_wasserstein=True, use_reward_as_d=False, use_manhattan_as_d=False, solver=False):
    """
    Computes state-action bisimulation metric
    """
    # u - v <= d(si, sj)
    A = [1, -1]
    bounds = [(-1, 1), (-1, 1)]

    d = np.zeros((src_state_space, action_space, tgt_state_space, action_space))
    dist_matrix = np.random.rand(src_state_space, tgt_state_space)
    reward_matrix_tmp = np.zeros((src_state_space, 4, tgt_state_space, 4))
    reward_matrix = np.zeros((src_state_space, tgt_state_space))
    for s1_pos, s1_state in src_env.state2idx.items():
        src_env.position = s1_pos
        src_env.start_position = s1_pos
        for s2_pos, s2_state in tgt_env.state2idx.items():
            tgt_env.position = s2_pos
            tgt_env.start_position = s2_pos
            for a in range(action_space):
                next_state, reward_a, done, next_possible_states = src_env.step(a)
                src_env.start_position = s1_pos
                src_env.position = s1_pos
                for b in range(action_space):
                    # print("source state: ", s1_state, "| source action: ", a, "| P(s'): ", src_env.tp_matrix[s1_state, a])
                    # print("target state: ", s2_state, "| target action: ", b, "| P(t'): ", tgt_env.tp_matrix[s2_state, b])
                    next_state, reward_b, done, next_possible_states = tgt_env.step(b)
                    # d[s1_state,a,s2_state,b] = 0
                    # d[s1_state,a,s2_state,b] += math.fabs(reward_a - reward_b)
                    reward_matrix_tmp[s1_state, a, s2_state, b] = math.fabs(reward_a - reward_b)
                    # b = [d[s1_state,a,s2_state,b]]
                    # res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, options={"disp": True})
                    # print (res.fun)
                    tgt_env.start_position = s2_pos
                    tgt_env.position = s2_pos
    
    for s1_pos, s1_state in src_env.state2idx.items():
        for s2_pos, s2_state in tgt_env.state2idx.items():
            reward_matrix[s1_state, s2_state] = np.max(reward_matrix_tmp[s1_state,:,s2_state,:])

    # Supply Manhattan distance as an alternative to reward distance for calculation of EMD
    # DO NOT USE when S1 and S2 are of different sizes
    manhattan_distance = np.zeros((src_env.state_space, tgt_env.state_space))
    for s1_pos, s1_state in src_env.state2idx.items():
        for s2_pos, s2_state in tgt_env.state2idx.items():
            manhattan_distance[s1_state, s2_state] = distance.cityblock(s1_pos, s2_pos)

    # np.fill_diagonal(reward_matrix, 0)
    # print (reward_matrix)
    sum1 = np.sum(dist_matrix)
    for s1_pos, s1_state in src_env.state2idx.items():
        for s2_pos, s2_state in tgt_env.state2idx.items():
            while True:
                val = -10.
                ctr = 0
                for a in range(action_space):
                    new_val = reward_matrix_tmp[s1_state, a, s2_state, a] + emd(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,a], dist_matrix)
                    val = max(new_val, val)
                ctr += 1
                if math.fabs(val - dist_matrix[s1_state, s2_state]) < 0.1:
                    break
                dist_matrix[s1_state, s2_state] = val

    print ("Updated: ", sum1 - np.sum(dist_matrix))
    for s1_pos, s1_state in src_env.state2idx.items():
        src_env.position = s1_pos
        src_env.start_position = s1_pos
        for s2_pos, s2_state in tgt_env.state2idx.items():
            tgt_env.position = s2_pos
            tgt_env.start_position = s2_pos
            for a in range(action_space):
                next_state, reward_a, done, next_possible_states = src_env.step(a)
                src_env.start_position = s1_pos
                src_env.position = s1_pos
                for b in range(action_space):
                    next_state, reward_b, done, next_possible_states = tgt_env.step(b)
                    # print (src_env.tp_matrix[s1_state,a].shape)
                    # print (tgt_env.tp_matrix[s2_state,b].shape)
                    if use_reward:
                        d[s1_state,a,s2_state,b] += reward_matrix_tmp[s1_state, a, s2_state, b]
                    if use_wasserstein:
                        # pdb.set_trace()
                        if use_reward_as_d:
                            d[s1_state,a,s2_state,b] += emd(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b], reward_matrix)
                        elif use_manhattan_as_d:
                            d[s1_state,a,s2_state,b] += emd(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b], manhattan_distance)
                        elif solver:
                            d[s1_state,a,s2_state,b] += emd(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b], dist_matrix)
                    tgt_env.start_position = s2_pos
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

def laxBisimTransfer(S1, S2, debugging=False):
    dl_sa = compute_d(use_reward=True, use_wasserstein=True, use_manhattan_as_d=False, solver=True)
    print (dl_sa[0,0,0,0])
    bisim_state_metric = compute_dl(dl_sa)
    print(bisim_state_metric)
    if debugging:
        num_misclassified_states = 0  # to be used only when source and target domains are same
    for t in range(S2):
        s_t = np.argmin(bisim_state_metric[:,t])
        if debugging:
            num_misclassified_states += int(t!=s_t)
        print("target state: %d, matching source state: %d"%(t, s_t))
        b_t = np.argmin(dl_sa[s_t, src_agent.get_best_action(s_t, src_possible_actions), t])
        print("source best action: %d, target best action: %d"%(src_agent.get_best_action(s_t, src_possible_actions), b_t))
        qv = src_agent.qvalues[s_t][b_t]
        tgt_agent.update_qvalue(t, b_t, qv)
    if debugging:
        print("State misclassification rate: %f percent"%(100*num_misclassified_states/S2))
    
laxBisimTransfer(src_state_space, tgt_state_space, debugging=True)

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
        time.sleep(0.1)
        state = tgt_env.get_state()
        continue