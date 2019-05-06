import math
import numpy as np
import os
import sys
import time

from copy import deepcopy
from pyemd import emd
from scipy.optimize import linprog

from env import create_env
from qlearn import QLearningAgent

alpha = 0.2
epsilon = 0.05
discount = 0.99

class Bisimulation(object):
    def __init__(self, opts):
        self.opts = opts
        self.src_env = create_env(opts.src_env)
        self.tgt_env = create_env(opts.tgt_env)
        self.src_agent = QLearningAgent(alpha, epsilon, discount, 4, self.src_env.state_space, self.src_env.tp_matrix, self.src_env.blocked_positions)
        self.tgt_agent = QLearningAgent(alpha, epsilon, discount, 4, self.tgt_env.state_space, self.tgt_env.tp_matrix, self.tgt_env.blocked_positions)
        self.transferred_agent = QLearningAgent(alpha, epsilon, discount, 4, self.tgt_env.state_space, self.tgt_env.tp_matrix, self.tgt_env.blocked_positions)
        self.action_space = self.src_env.action_space
        self.solver = opts.solver
        self.lfp_iters = opts.lfp_iters
        self.threshold = opts.threshold
        self.discount_kd = opts.discount_kd
        self.discount_kd = opts.discount_r
        if self.solver == 'lp':
            m = self.src_env.state_space
            n = self.tgt_env.state_space
            A_r = np.zeros((m, m, n))
            A_t = np.zeros((n, m, n))

            for i in range(m):
                for j in range(n):
                    A_r[i, i, j] = 1
            
            for i in range(n):
                for j in range(m):
                    A_t[i, j, i] = 1
            self.A = np.concatenate((A_r.reshape((m, m*n)), A_t.reshape((n, m*n))), axis=0)
        
        self.src_possible_actions = self.src_env.get_possible_actions()
        self.tgt_possible_actions = self.tgt_env.get_possible_actions()

        # Initialize Q-Values
        self.src_agent.qvalues = np.load(os.path.join(opts.policy_dir, opts.src_env + '.npy'))
        self.tgt_agent.qvalues = np.load(os.path.join(opts.policy_dir, opts.tgt_env + '.npy'))
        
        # Distance and reward matrices
        self.d_sa_final = np.zeros((self.src_env.state_space, self.action_space, self.tgt_env.state_space, self.action_space))
        self.dist_matrix_final = np.zeros((self.src_env.state_space, self.tgt_env.state_space))
        self.reward_matrix_tmp = np.zeros((self.src_env.state_space, self.action_space, self.tgt_env.state_space, self.action_space))
        self.reward_matrix = np.zeros((self.src_env.state_space, self.tgt_env.state_space))
        self.init_reward_matix()

        self.accuracy = None
    
    def init_reward_matix(self):
        for s1_pos, s1_state in self.src_env.state2idx.items():
            self.src_env.position = s1_pos
            self.src_env.start_position = s1_pos
            for s2_pos, s2_state in self.tgt_env.state2idx.items():
                self.tgt_env.position = s2_pos
                self.tgt_env.start_position = s2_pos
                for a in range(self.action_space):
                    next_state, reward_a, done, next_possible_states = self.src_env.step(a)
                    self.src_env.start_position = s1_pos
                    self.src_env.position = s1_pos
                    for b in range(self.action_space):
                        next_state, reward_b, done, next_possible_states = self.tgt_env.step(b)
                        self.reward_matrix_tmp[s1_state, a, s2_state, b] = math.fabs(reward_a - reward_b)
                        self.tgt_env.start_position = s2_pos
                        self.tgt_env.position = s2_pos
    
        for s1_pos, s1_state in self.src_env.state2idx.items():
            for s2_pos, s2_state in self.tgt_env.state2idx.items():
                self.reward_matrix[s1_state, s2_state] = np.max(self.reward_matrix_tmp[s1_state,:,s2_state,:])

    def solver_lp(self):
        for s1_pos, s1_state in self.src_env.state2idx.items():
            for s2_pos, s2_state in sorted(self.tgt_env.state2idx.items()):
                for a in range(self.action_space):
                    for b in range(self.action_space):
                        p = self.src_env.tp_matrix[s1_state,a]
                        q = self.tgt_env.tp_matrix[s2_state,b]
                        b_mat = np.concatenate((p, q), axis=0)
                        c = self.dist_matrix.reshape((self.src_env.state_space * self.tgt_env.state_space))
                        # opt_res = linprog(c, A_eq=self.A, b_eq=b_mat)#, bounds=(-1, 1)) #method='interior-point', options={'sym_pos':False})
                        opt_res = linprog(-b_mat, self.A.T, c, bounds=(-1, 1), method='interior-point')
                        # value = opt_res.fun
                        value = -1 * opt_res.fun
                        self.d[s1_state, a, s2_state, b] = value

                d_st = self.d[s1_state, :, s2_state, :]
                self.dist_matrix[s1_state, s2_state] = max(np.max(np.min(d_st, axis=1)), np.max(np.min(d_st, axis=0)))
    
    def solver_pyemd(self):
        while True:
            for s1_pos, s1_state in self.src_env.state2idx.items():
                for s2_pos, s2_state in sorted(self.tgt_env.state2idx.items()):
                    for a in range(self.action_space):
                        for b in range(self.action_space):
                            kd = emd(self.src_env.tp_matrix[s1_state,a], self.tgt_env.tp_matrix[s2_state,b], self.dist_matrix) # pyemd
                            new_val = 0.1 * self.reward_matrix_tmp[s1_state, a, s2_state, b] + 0.9 * kd
                            self.d[s1_state, a, s2_state, b] = new_val
                    d_st = self.d[s1_state, :, s2_state, :]
                    val = max(np.max(np.min(d_st, axis=1)), np.max(np.min(d_st, axis=0)))
                    self.tmp_dist_matrix[s1_state, s2_state] = val

            if np.mean(np.abs(self.dist_matrix - self.tmp_dist_matrix)) < self.opts.threshold:
                self.dist_matrix = self.tmp_dist_matrix.copy()
                break
            
            self.dist_matrix = self.tmp_dist_matrix.copy()
    
    def bisimulation(self):
        raise NotImplementedError
    
    def lax_bisimulation(self):
        self.d = np.zeros((self.src_env.state_space, self.action_space, self.tgt_env.state_space, self.action_space))
        self.d.fill(1000.0)
        if self.solver == 'lp':
            solver = self.solver_lp
            self.dist_matrix = np.random.rand(self.src_env.state_space, self.tgt_env.state_space)
        else:
            solver = self.solver_pyemd
            self.dist_matrix = np.zeros((self.src_env.state_space, self.tgt_env.state_space))
            self.dist_matrix.fill(0.01)
        self.tmp_dist_matrix = np.zeros((self.src_env.state_space, self.tgt_env.state_space))
            
        for i in range(self.opts.lfp_iters):
            print ("Iteration: ", i, "/", self.opts.lfp_iters, " Loss: ", np.mean(np.abs(self.dist_matrix_final - self.dist_matrix)))
            self.dist_matrix_final = self.dist_matrix.copy()
            solver()
        self.dist_matrix_final = self.dist_matrix.copy()
        self.d_sa_final = self.d.copy()

        match = 0.
        for t in range(self.tgt_env.state_space):
            s_t = np.argmin(self.dist_matrix_final[:,t])
            b_t = np.argmin(self.d_sa_final[s_t, self.src_agent.get_best_action(s_t, self.src_possible_actions), t])
            gt_bt = self.tgt_agent.get_best_action(t, self.tgt_possible_actions)
            qv = 1000.0
            self.transferred_agent.update_qvalue(t, b_t, qv)
            if gt_bt == b_t:
                match += 1.
        np.save(os.path.join(os.path.join(self.opts.save_dir, 'solver_' + self.opts.solver), self.opts.tgt_env + '.npy'), np.asarray(self.transferred_agent.qvalues))
        self.accuracy = (match / self.tgt_env.state_space) * 100.

    def pess_bisimulation(self):
        raise NotImplementedError

    def opt_bisimulation(self):
        raise NotImplementedError
    
    def generate_logs(self):
        raise NotImplementedError
    
    def render(self):
        self.tgt_env.render(self.transferred_agent.qvalues)
        for i in range(10):
            self.tgt_env.reset_state()
            state = self.tgt_env.get_state()
            score = 0
            for j in range(100):
                possible_actions = self.tgt_env.get_possible_actions()
                action = self.transferred_agent.get_best_action(state, possible_actions)
                next_state, reward, done, next_possible_states = self.tgt_env.step(action)
                score += reward
                self.tgt_env.render(self.transferred_agent.qvalues)

                next_state_possible_actions = self.tgt_env.get_possible_actions()
                state = next_state

                if done == True:
                    self.tgt_env.reset_state()
                    self.tgt_env.render(self.transferred_agent.qvalues)
                    time.sleep(0.1)
                    state = self.tgt_env.get_state()
                    break