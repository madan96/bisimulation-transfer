import os
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.optimize import linprog
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import sys
import time
import matplotlib.pyplot as plt
import environment
import agent

#### EMD imports ####
from pyemd import emd
import cv2
import ot

log_path = "logs"
if not os.path.isdir(log_path):
    os.mkdir(log_path)

fname = str(datetime.datetime.now())
# sys.stdout = open(log_path + "/" + fname, 'w')
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
# end_positions = [(2, 6)]
# tgt_blocked_positions = [(3,0), (0, 4), (5, 4), (3, 2), (3, 3), (3, 4), (2, 4), (2, 5), (2, 7), (2, 8)]

gridH, gridW = 6, 9 
end_positions = [(2, 6)]
tgt_blocked_positions = [(3,0), (0, 4), (5, 4), (3, 2), (3, 3), (3, 4), (2, 4), (2, 5), (2, 7), (2, 8), (3, 5), (4, 5), (5, 5), (3,6), (4,6), (5,6), (3,7), (4,7), (5,7), (3,8), (4,8), (5,8)]

# gridH, gridW = 5, 5
# end_positions = [(2, 3)]
# tgt_blocked_positions = [(2, 0), (0, 2), (4, 2), (2, 2), (2, 4)]

# gridH, gridW = 3, 3
# end_positions = [(1, 2)]
# tgt_blocked_positions = [(2, 21)]

tgt_env = environment.Environment(gridH, gridW, end_positions, end_rewards, tgt_blocked_positions, start_pos, default_reward)
action_space = tgt_env.action_space
tgt_state_space = tgt_env.state_space
tgt_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, tgt_state_space, tgt_env.tp_matrix, tgt_blocked_positions)
test_tgt_agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, tgt_state_space, tgt_env.tp_matrix, tgt_blocked_positions)

src_agent.qvalues = np.load('optimal_qvalues_8_new_prob_states.npy')
test_tgt_agent.qvalues = np.load('optimal_qvalues_44_new_prob_states.npy')

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
    print ("Number of lfp iterations: ", least_fixed_iters)
    print ("Threshold value: ", threshold)
    print ("Source size: ", src_state_space, " Target size: ", tgt_state_space)

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
    if emd_func == 'pyemd':
        d = np.zeros((src_state_space, action_space, tgt_state_space, action_space))
        dist_matrix = np.zeros((src_state_space, tgt_state_space))
        dist_matrix.fill(0.01)
        tmp_dist_matrix = np.zeros((src_state_space, tgt_state_space))
    else:
        d = np.zeros((src_state_space, action_space, tgt_state_space, action_space)).astype(np.float32)
        dist_matrix = np.random.rand(src_state_space, tgt_state_space).astype(np.float32)
        tmp_dist_matrix = np.random.rand(src_state_space, tgt_state_space).astype(np.float32)

    for i in range(least_fixed_iters):
        print ("Iteration: ", i, "/", least_fixed_iters, " Loss: ", np.mean(np.abs(dist_matrix_final - dist_matrix)))
        dist_matrix_final = dist_matrix.copy()
        ctr = 0
        while True:
            for s1_pos, s1_state in src_env.state2idx.items():
                for s2_pos, s2_state in sorted(tgt_env.state2idx.items()):
                    for a in range(action_space):
                        for b in range(action_space):
                            kd = emd(src_env.tp_matrix[s1_state,a], tgt_env.tp_matrix[s2_state,b], dist_matrix) # pyemd
                            new_val = 0.1 * reward_matrix_tmp[s1_state, a, s2_state, b] + 0.9 * kd
                            d[s1_state, a, s2_state, b] = new_val
                            d_st = d[s1_state, :, s2_state, :]
                    val = max(np.max(np.min(d_st, axis=1)), np.max(np.min(d_st, axis=0)))
                    tmp_dist_matrix[s1_state, s2_state] = val

            if np.mean(np.abs(dist_matrix - tmp_dist_matrix)) < threshold:
                dist_matrix = tmp_dist_matrix.copy()
                break
            
            dist_matrix = tmp_dist_matrix.copy()

    dist_matrix_final = dist_matrix.copy()
    d_final = d.copy()

    return d_final, dist_matrix_final

def laxBisimTransfer(S1, S2, debugging=False):
    dl_sa, bisim_state_metric = compute_d(5, 0.01, emd_func='pyemd', use_manhattan_as_d=False)
    plt.imshow(bisim_state_metric, cmap='flag' )
    plt.savefig('/home/rishabh/Pictures/bisimulation/lax/lax_dm_obs_' + str(tgt_blocked_positions[0][0]) + str(tgt_blocked_positions[0][1]) + '.png')
    # plt.show()
    match = 0.
    for t in range(S2):
        s_t = np.argmin(bisim_state_metric[:,t])
        print ("s: ", bisim_state_metric[:,t])
        b_t = np.argmin(dl_sa[s_t, src_agent.get_best_action(s_t, src_possible_actions), t])
        gt_bt = test_tgt_agent.get_best_action(t, tgt_possible_actions)
        print ("sa: ", dl_sa[s_t, src_agent.get_best_action(s_t, src_possible_actions), t])
        qv = 1000.0
        tgt_agent.update_qvalue(t, b_t, qv)
        if gt_bt == b_t:
            match += 1.
    accuracy = (match / S2) * 100.
    return accuracy

start = time.time()
accuracy = laxBisimTransfer(src_state_space, tgt_state_space, debugging=True)
end = time.time()

tgt_env.render(tgt_agent.qvalues)

returns = []
for i in range(100):
    tgt_env.reset_state()
    state = tgt_env.get_state()
    score = 0
    for j in range(100):
        possible_actions = tgt_env.get_possible_actions()
        action = tgt_agent.get_best_action(state, possible_actions)
        next_state, reward, done, next_possible_states = tgt_env.step(action)
        score += reward
        tgt_env.render(tgt_agent.qvalues)

        next_state_possible_actions = tgt_env.get_possible_actions()
        state = next_state

        if done == True:
            tgt_env.reset_state()
            tgt_env.render(tgt_agent.qvalues)
            time.sleep(0.1)
            state = tgt_env.get_state()
            break

    returns.append(score)

print ("Accuracy: ", accuracy)
print ("Transfer time: ", end - start)
plt.plot(returns)
plt.ylabel('Return')
# plt.show()
plt.savefig(log_path + '/' + fname[:-6])
