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
import cv2
import ot

import environment
import agent

np.random.seed(712)

counts = 0
alpha = 0.2
epsilon = 0.05
discount = 0.99
end_rewards = [1.0]
default_reward = 0.0
start_pos = None


gridH, gridW = 3, 3
end_positions = [(1, 2)]
src_blocked_positions = [(1, 1)]

src_env = environment.Environment(gridH, gridW, end_positions, end_rewards, src_blocked_positions, start_pos, default_reward)
action_space = src_env.action_space
src_state_space = src_env.state_space
src_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, src_state_space, src_env.tp_matrix, src_blocked_positions)

# gridH, gridW = 5, 11 
# end_positions = [(2, 8)]
# tgt_blocked_positions = [(0, 5), (2, 5), (4, 5), (2, 0), (2, 1), (2, 3), (2, 4), (2, 6), (2, 7), (2, 9), (2, 10)]

# gridH, gridW = 6, 9 
# end_positions = [(2, 6)]
# tgt_blocked_positions = [(3,0), (0, 4), (5, 4), (3, 2), (3, 3), (3, 4), (2, 4), (2, 5), (2, 7), (2, 8)]

# gridH, gridW = 5, 5
# end_positions = [(2, 3)]
# tgt_blocked_positions = [(2, 0), (0, 2), (4, 2), (2, 2), (2, 4)]

gridH, gridW = 3, 3
end_positions = [(1, 2)]
tgt_blocked_positions = []

tgt_env = environment.Environment(gridH, gridW, end_positions, end_rewards, tgt_blocked_positions, start_pos, default_reward)
action_space = tgt_env.action_space
tgt_state_space = tgt_env.state_space
tgt_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, tgt_state_space, tgt_env.tp_matrix, tgt_blocked_positions)

src_agent.qvalues = np.load('optimal_qvalues_8_new_prob_states.npy')
# tgt_qval = np.zeros((tgt_env.state_space, 4))

src_possible_actions = src_env.get_possible_actions()
tgt_possible_actions = tgt_env.get_possible_actions()

def compute_d_opt(use_manhattan_as_d=False, solver=True, emd_func='cv2'):
    """
    Computes state-action and state-state bisimulation metric

    Parameters: 
    use_manhattan_as_d (bool): If True, use manhattan distance as distance matrix
    solver (bool): If true, perform fixed point iteration for calculating distance matrix
    emd (str): Specify which function to use for calculating the Earth Mover's 
                distance or Wasserstein distance or Kantorovich metric
  
    Returns: 
    d_final: Bisimulation state-action metric with dim S1 x a x S2 x b
    dist_matrix_final: Bisimulation state-state metric with dim S1 x S2
    """
    # u - v <= d(si, sj)
    A = [1, -1]
    bounds = [(-1, 1), (-1, 1)]
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
                    next_state, reward_b, done, next_possible_states = tgt_env.step(b)
                    reward_matrix_tmp[s1_state, a, s2_state, b] = math.fabs(reward_a - reward_b)
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

    #######################
    # dist_matrix_final = np.random.rand(src_state_space, tgt_state_space)
    tmp_dist_matrix = np.random.rand(src_state_space, tgt_state_space)
    dist_matrix_final = np.zeros((src_state_space, tgt_state_space))
    dist_matrix_final.fill(1000.0)
    d_final = np.zeros((src_state_space, 1, tgt_state_space, action_space))
    d_final.fill(1000.0)
    m = src_state_space
    n = tgt_state_space

    global counts
    while True:
        counts += 1
        print (counts)
        if counts > 30:
            break
        for s1_pos, s1_state in src_env.state2idx.items():
            for s2_pos, s2_state in sorted(tgt_env.state2idx.items()):
                a = src_agent.get_best_action(s1_state, src_possible_actions)
                p = src_env.tp_matrix[s1_state, a]
                for b in range(action_space):
                    
                    A_r = np.zeros((m, m, n))
                    A_t = np.zeros((n, m, n))

                    for i in range(m):
                        for j in range(n):
                            A_r[i, i, j] = 1
                    
                    for i in range(n):
                        for j in range(m):
                            A_t[i, j, i] = 1
                    
                    A = np.concatenate((A_r.reshape((m, m*n)), A_t.reshape((n, m*n))), axis=0)
                    q = tgt_env.tp_matrix[s2_state, b]
                    b_mat = np.concatenate((p, q), axis=0)
                    c = tmp_dist_matrix.reshape((src_state_space * tgt_state_space))
                    # opt_res = linprog(c, A_eq=A, b_eq=b_mat, method='interior-point') #, options={'sym_pos':False})
                    opt_res = linprog(-b_mat, A.T, c, bounds=(-1, 1))
                    # opt_res = linprog(-b_mat, A.T, c, bounds=(-1, 1), method='interior-point')
                    # value = opt_res.fun
                    value = -1 * opt_res.fun
                    # print (d_final.shape)
                    # print (s1_state, a, s2_state, b)
                    d_final[s1_state, 0, s2_state, b] = 0.1 * reward_matrix_tmp[s1_state, a, s2_state, b] + 0.9 * value
                    # print (value)
                # tmp_dist_matrix[s1_state, s2_state] = np.min(d_final[s1_state, 0, s2_state])

        for s1_pos, s1_state in src_env.state2idx.items():
            for s2_pos, s2_state in sorted(tgt_env.state2idx.items()):
                tmp_dist_matrix[s1_state, s2_state] = np.min(d_final[s1_state, 0, s2_state])
        
        print (np.mean(np.abs(dist_matrix_final - tmp_dist_matrix)))
        if np.mean(np.abs(dist_matrix_final - tmp_dist_matrix)) < 0.01:
            break
        dist_matrix_final = tmp_dist_matrix.copy()
    #######################

    print (dist_matrix_final)
    return d_final, dist_matrix_final
    
def compute_dl_opt(d):
    """
    Computes state bisimulation metric for pessimistic transfer
    """
    S1 = src_env.state_space
    S2 = tgt_env.state_space
    pess_bisim_state_metric = np.zeros((S1, S2))
    for i in range(S1):
        for j in range(S2):
            d_st = d[i,:,j,:]
            print (d_st.shape)
            dl_st = np.min(d_st)
            pess_bisim_state_metric[i][j] = dl_st
    return pess_bisim_state_metric


def optBisimTransfer(S1, S2):
    dl_sa, bisim_state_metric = compute_d_opt()
    # bisim_state_metric = compute_dl_opt(dl_sa)
    # tgt_agent.qvalues = np.zeros((tgt_state_space, action_space))
    # tgt_agent.qvalues.fill(-10.0)
    lower_bound = np.zeros((S1, S2))
    for t in range (S2):
        for s in range(S1):
            lower_bound[s, t] = np.max(src_agent.qvalues[s]) - bisim_state_metric[s, t]
        s_t = np.argmax(lower_bound[:, t])
        b_t = np.argmin(dl_sa[s_t,0,t,:])
        print("target state: %d, matching source state: %d"%(t, s_t))
        print (dl_sa[s_t, 0, t])
        b_t = np.argmin(dl_sa[s_t, 0, t])
        print("source best action: %d, target best action: %d"%(src_agent.get_best_action(s_t, src_possible_actions), b_t))
        qv = 1000.0 #src_agent.qvalues[s_t][b_t]
        tgt_agent.update_qvalue(t, b_t, qv)

optBisimTransfer(src_state_space, tgt_state_space)
qval = np.asarray(tgt_agent.qvalues)
print (qval.shape)
np.save('ip_opt_transfer_44_states_count_' + str(counts) + '.npy', qval)

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

print ("Num iters: ", counts)