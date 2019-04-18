import math
import numpy as np
import pdb
from scipy.optimize import linprog
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import time

import environment
import agent

#### EMD imports ####
from pyemd import emd
import cv2
import ot

np.random.seed(712)

alpha = 0.2
epsilon = 0.05
discount = 0.99
alpha = 0.2
epsilon = 0.05
discount = 0.99

end_rewards = [10.0]
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

# gridH, gridW = 5, 5
# end_positions = [(2, 3)]
# tgt_blocked_positions = [(2, 0), (0, 2), (4, 2), (2, 2), (2, 4)]

gridH, gridW = 3, 3
end_positions = [(1, 2)]
tgt_blocked_positions = [(1, 1)]

tgt_env = environment.Environment(gridH, gridW, end_positions, end_rewards, tgt_blocked_positions, start_pos, default_reward)
action_space = tgt_env.action_space
tgt_state_space = tgt_env.state_space
tgt_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, tgt_state_space, tgt_env.tp_matrix, tgt_blocked_positions)

src_agent.qvalues = np.load('optimal_qvalues_8_new_prob_states.npy')
# tgt_qval = np.zeros((tgt_env.state_space, 4))

src_possible_actions = src_env.get_possible_actions()
tgt_possible_actions = tgt_env.get_possible_actions()

def compute_d(least_fixed_iters=10, threshold=0.00001, emd_func='cv2', use_manhattan_as_d=False):
    """
    Computes state-action and state-state bisimulation metric

    Parameters: 
    least_fixed_iters (int): Number of iterations of random init and solving
    threshold (float): Threshold value for stopping the solver for distance matrix
    emd (str): Specify which function to use for calculating the Earth Mover's 
                distance or Wasserstein distance or Kantorovich metric
                Options: ['scipy', 'cv2', 'opt', 'pyemd']
    use_manhattan_as_d (bool): If True, use manhattan distance as distance matrix
    
  
    Returns: 
    d_final: Bisimulation state-action metric with dim (S1 x a x S2 x b)
    dist_matrix_final: Bisimulation state-state metric with dim (S1 x S2)
    """

    print ("EMD computed using: ", emd_func)

    if emd_func == 'pyemd':
        reward_matrix_tmp = np.zeros((src_state_space, action_space, tgt_state_space, action_space))
        reward_matrix = np.zeros((src_state_space, tgt_state_space))
        dist_matrix_final = np.zeros((src_state_space, tgt_state_space))
        d_final = np.zeros((src_state_space, action_space, tgt_state_space, action_space))
    else:
        reward_matrix_tmp = np.zeros((src_state_space, action_space, tgt_state_space, action_space)).astype(np.float32)
        reward_matrix = np.zeros((src_state_space, tgt_state_space)).astype(np.float32)
        src_env.tp_matrix = src_env.tp_matrix.astype(np.float32)
        tgt_env.tp_matrix = tgt_env.tp_matrix.astype(np.float32)
        dist_matrix_final = np.zeros((src_state_space, tgt_state_space)).astype(np.float32)
        d_final = np.zeros((src_state_space, action_space, tgt_state_space, action_space)).astype(np.float32)

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

    dist_matrix_final.fill(1000.0)
    d_final.fill(1000.0)

    for i in range(least_fixed_iters):
        if emd_func == 'pyemd':
            d = np.zeros((src_state_space, action_space, tgt_state_space, action_space))
            dist_matrix = np.random.rand(src_state_space, tgt_state_space)
            tmp_dist_matrix = np.random.rand(src_state_space, tgt_state_space)
        else:
            d = np.zeros((src_state_space, action_space, tgt_state_space, action_space)).astype(np.float32)
            dist_matrix = np.random.rand(src_state_space, tgt_state_space).astype(np.float32)
            tmp_dist_matrix = np.random.rand(src_state_space, tgt_state_space).astype(np.float32)

        ctr = 0
        while True:
            for s1_pos, s1_state in src_env.state2idx.items():
                for s2_pos, s2_state in sorted(tgt_env.state2idx.items()):
                    for a in range(action_space):
                        for b in range(action_space):
                            # kd = cv2.EMD(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b], cv2.DIST_USER, cost=dist_matrix)[0] # cv2
                            # kd, log = ot.emd2(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b], dist_matrix, log=True, numItermax=100000) # ot
                            # kd = wasserstein_distance(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b],
                            # u_weights=dist_matrix[:,s2_state], v_weights=dist_matrix[s1_state,:]) # scipy
                            kd = emd(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b], dist_matrix) # pyemd
                            # if log['result_code'] == 0:
                            #     continue
                            new_val = reward_matrix_tmp[s1_state, a, s2_state, b] + 0.9 * kd
                            d[s1_state, a, s2_state, b] = new_val
                            d_st = d[s1_state, :, s2_state, :]
                            val = max(np.max(np.min(d_st, axis=1)),np.max(np.min(d_st, axis=0)))
                    tmp_dist_matrix[s1_state, s2_state] = val
            dist_matrix = tmp_dist_matrix
            if np.mean(np.abs(dist_matrix - tmp_dist_matrix)) < threshold:
                break

        for s1_pos, s1_state in src_env.state2idx.items():
            for s2_pos, s2_state in tgt_env.state2idx.items():
                if dist_matrix[s1_state, s2_state] < dist_matrix_final[s1_state, s2_state]:
                    dist_matrix_final[s1_state, s2_state] = dist_matrix[s1_state, s2_state]
                    d_final[s1_state, :, s2_state, :] = d[s1_state, :, s2_state, :]

    return d_final, dist_matrix_final

def laxBisimTransfer(S1, S2, debugging=False):
    dl_sa, bisim_state_metric = compute_d(100, 0.0001, emd_func='pyemd', use_manhattan_as_d=False)
    if debugging:
        num_misclassified_states = 0  # to be used only when source and target domains are same
    for t in range(S2):
        s_t = np.argmin(bisim_state_metric[:,t])
        if debugging:
            num_misclassified_states += int(t!=s_t)
        print("target state: %d, matching source state: %d" % (t, s_t))
        print ("s: ", bisim_state_metric[:,t])
        b_t = np.argmin(dl_sa[s_t, src_agent.get_best_action(s_t, src_possible_actions), t])
        print ("sa: ", dl_sa[s_t, src_agent.get_best_action(s_t, src_possible_actions), t])
        print("source best action: %d, target best action: %d" % (src_agent.get_best_action(s_t, src_possible_actions), b_t))
        qv = 1000.0 #src_agent.qvalues[s_t][src_agent.get_best_action(s_t, src_possible_actions)] #[b_t]
        tgt_agent.update_qvalue(t, b_t, qv)
    if debugging:
        print("State misclassification rate: %f percent" % (100 * num_misclassified_states / S2))
    
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