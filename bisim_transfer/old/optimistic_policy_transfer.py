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

import matplotlib.pyplot as plt
import environment
import agent
import time
import copy

np.random.seed(712)

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
# end_positions = [(3, 1)]
# tgt_blocked_positions = [(3,0), (0, 4), (5, 4), (3, 2), (3, 3), (3, 4), (2, 4), (2, 5), (2, 7), (2, 8)]

# gridH, gridW = 6, 9 
# end_positions = [(2, 6)]
# tgt_blocked_positions = [(3,0), (0, 4), (5, 4), (3, 2), (3, 3), (3, 4), (2, 4), (2, 5), (2, 7), (2, 8), (3, 5), (4, 5), (5, 5), (3,6), (4,6), (5,6), (3,7), (4,7), (5,7), (3,8), (4,8), (5,8)]

# gridH, gridW = 5, 5
# end_positions = [(2, 3)]
# tgt_blocked_positions = [(2, 0), (0, 2), (4, 2), (2, 2), (2, 4)]

gridH, gridW = 3, 3
end_positions = [(1, 2)]
tgt_blocked_positions = [(2, 0)]

tgt_env = environment.Environment(gridH, gridW, end_positions, end_rewards, tgt_blocked_positions, start_pos, default_reward)
action_space = tgt_env.action_space
tgt_state_space = tgt_env.state_space
tgt_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, tgt_state_space, tgt_env.tp_matrix, tgt_blocked_positions)
test_tgt_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, tgt_state_space, tgt_env.tp_matrix, tgt_blocked_positions)

src_agent.qvalues = np.load('optimal_qvalues_8_new_prob_states.npy')
test_tgt_agent.qvalues = np.load('optimal_qvalues_8_new_prob_states.npy')
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


    dist_matrix_final = np.zeros((src_state_space, tgt_state_space))
    dist_matrix_final.fill(100.0)
    d_final = np.zeros((src_state_space, 1, tgt_state_space, action_space))
    d = np.zeros((src_state_space, 1, tgt_state_space, action_space))
    d.fill(1000.0)
    dist_matrix = np.zeros((src_state_space, tgt_state_space))
    dist_matrix.fill(0.01) # 3 is a good inits
    tmp_dist_matrix = np.zeros((src_state_space, tgt_state_space))
    tmp_dist_matrix.fill(0.1)
    num_iters = 10
    for i in range(num_iters):
        print ("Iteration: ", i, "/", num_iters, " Loss: ", np.mean(np.abs(dist_matrix_final - dist_matrix)))
        dist_matrix_final = copy.deepcopy(dist_matrix)
        while True:
            for s1_pos, s1_state in src_env.state2idx.items():
                a = src_agent.get_best_action(s1_state, src_possible_actions)
                for s2_pos, s2_state in tgt_env.state2idx.items():
                    for b in range(action_space): 
                        kd = emd(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b], dist_matrix) # pyemd
                        new_val = 0.1 * reward_matrix_tmp[s1_state, a, s2_state, b] + 0.9 * kd
                        d[s1_state, 0, s2_state, b] = new_val
                        val = np.min(d[s1_state, 0, s2_state])
                    tmp_dist_matrix[s1_state, s2_state] = val
            
            if np.mean(np.abs(dist_matrix - tmp_dist_matrix)) < 0.01:
                dist_matrix = copy.deepcopy(tmp_dist_matrix)
                break
            
            dist_matrix = copy.deepcopy(tmp_dist_matrix)
        
    d_final = d.copy()
    dist_matrix_final = dist_matrix.copy()

    print (np.min(dist_matrix_final), np.max(dist_matrix_final), np.mean(dist_matrix_final))

    return d_final, dist_matrix_final

def optBisimTransfer(S1, S2):
    dl_sa, bisim_state_metric = compute_d_opt()
    plt.imshow(bisim_state_metric, cmap='flag' )
    plt.savefig('/home/rishabh/Pictures/bisimulation/optimistic/opt_dm_obs_' + str(tgt_blocked_positions[0][0]) + str(tgt_blocked_positions[0][1]) + '.png')
    # plt.show()

    lower_bound = np.zeros((S1, S2))
    match = 0.
    # action_mat = np.zeros(6, 9)
    for t in range(S2):
        for s in range(S1):
            lower_bound[s, t] = np.max(src_agent.qvalues[s]) - bisim_state_metric[s, t]
        s_t = np.argmax(lower_bound[:, t])
        b_t = np.argmin(dl_sa[s_t,0,t,:])
        print (dl_sa[s_t, 0, t])
        b_t = np.argmin(dl_sa[s_t, 0, t])
        gt_bt = test_tgt_agent.get_best_action(t, tgt_possible_actions)

        if b_t == gt_bt:
            match += 1.
        qv = 1000.0 # src_agent.qvalues[s_t][b_t]
        tgt_agent.update_qvalue(t, b_t, qv)
    
    accuracy = (match / float(S2)) * 100.
    return accuracy

start = time.time()
accuracy = optBisimTransfer(src_state_space, tgt_state_space)
end = time.time()
tgt_q = np.asarray(tgt_agent.qvalues)
# np.save('/home/rishabh/work/btp/rl-grid-world/policy/optimistic_fastemd_transfer_44.npy', tgt_q)
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

print ("Accuracy: ", accuracy)
print ("Transfer time: ", end - start)