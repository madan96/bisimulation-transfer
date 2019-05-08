import math
import numpy as np
import os
import sys
import time

from copy import deepcopy
from pyemd import emd
from scipy.optimize import linprog

from .base_class import Bisimulation
from env import create_env
from qlearn import QLearningAgent

alpha = 0.2
epsilon = 0.05
discount = 0.99

# TODO: Implement basic and pessimistic bisimulation

class LaxBisimulation(Bisimulation):
    """
    Bisimulation class for Lax-Bisimulation transfer method
    """
    def solver_lp(self):
        for s1_pos, s1_state in self.src_env.state2idx.items():
            for s2_pos, s2_state in sorted(self.tgt_env.state2idx.items()):
                for a in range(self.action_space):
                    for b in range(self.action_space):
                        p = self.src_env.tp_matrix[s1_state,a]
                        q = self.tgt_env.tp_matrix[s2_state,b]
                        b_mat = np.concatenate((p, q), axis=0)
                        c = self.dist_matrix.reshape((self.src_env.state_space * self.tgt_env.state_space))
                        # opt_res = linprog(c, A_eq=self.A, b_eq=b_mat) #, bounds=(-1, 1)) #method='interior-point', options={'sym_pos':False})
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
                            new_val = self.opts.discount_r * self.reward_matrix_tmp[s1_state, a, s2_state, b] + self.opts.discount_kd * kd
                            self.d[s1_state, a, s2_state, b] = new_val
                    d_st = self.d[s1_state, :, s2_state, :]
                    val = max(np.max(np.min(d_st, axis=1)), np.max(np.min(d_st, axis=0)))
                    self.tmp_dist_matrix[s1_state, s2_state] = val

            if np.mean(np.abs(self.dist_matrix - self.tmp_dist_matrix)) < self.opts.threshold:
                self.dist_matrix = self.tmp_dist_matrix.copy()
                break
            
            self.dist_matrix = self.tmp_dist_matrix.copy()
    
    def execute_transfer(self):
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
        if not os.path.exists(self.opts.save_dir):
            os.makedirs(self.opts.save_dir)
        np.save(os.path.join(os.path.join(self.opts.save_dir, 'solver_' + self.opts.solver), self.opts.tgt_env + '.npy'), np.asarray(self.transferred_agent.qvalues))
        self.accuracy = (match / self.tgt_env.state_space) * 100.


class PessBisimulation(Bisimulation):
    def solver_lp(self):
        raise NotImplementedError
    
    def solver_pyemd(self):
        raise NotImplementedError
    
    def execute_transfer(self):
        raise NotImplementedError


class OptBisimulation(Bisimulation):
    def __init__(self, opts):
        Bisimulation.__init__(self, opts)
        self.d_sa_final = np.zeros((self.src_env.state_space, 1, self.tgt_env.state_space, self.action_space))

    def solver_lp(self):
        for s1_pos, s1_state in self.src_env.state2idx.items():
            a = self.src_agent.get_best_action(s1_state, self.src_possible_actions)
            for s2_pos, s2_state in sorted(self.tgt_env.state2idx.items()):
                for b in range(self.action_space):
                    p = self.src_env.tp_matrix[s1_state,a]
                    q = self.tgt_env.tp_matrix[s2_state,b]
                    b_mat = np.concatenate((p, q), axis=0)
                    c = self.dist_matrix.reshape((self.src_env.state_space * self.tgt_env.state_space))
                    opt_res = linprog(c, A_eq=self.A, b_eq=b_mat) #, bounds=(-1, 1)) #method='interior-point', options={'sym_pos':False})
                    # opt_res = linprog(-b_mat, self.A.T, c, bounds=(-1, 1), method='interior-point')
                    value = opt_res.fun
                    # value = -1 * opt_res.fun
                    self.d[s1_state, 0, s2_state, b] = value

                self.dist_matrix[s1_state, s2_state] = np.min(self.d[s1_state, 0, s2_state])
    
    def solver_pyemd(self):
        while True:
            for s1_pos, s1_state in self.src_env.state2idx.items():
                a = self.src_agent.get_best_action(s1_state, self.src_possible_actions)
                for s2_pos, s2_state in self.tgt_env.state2idx.items():
                    for b in range(self.action_space): 
                        kd = emd(self.src_env.tp_matrix[s1_state,a], self.tgt_env.tp_matrix[s2_state,b], self.dist_matrix) # pyemd
                        new_val = self.opts.discount_r * self.reward_matrix_tmp[s1_state, a, s2_state, b] + self.opts.discount_kd * kd
                        self.d[s1_state, 0, s2_state, b] = new_val
                    val = np.min(self.d[s1_state, 0, s2_state])
                    self.tmp_dist_matrix[s1_state, s2_state] = val
            
            if np.mean(np.abs(self.dist_matrix - self.tmp_dist_matrix)) < self.opts.threshold:
                self.dist_matrix = deepcopy(self.tmp_dist_matrix)
                break
            
            self.dist_matrix = deepcopy(self.tmp_dist_matrix)
    
    def execute_transfer(self):
        self.d = np.zeros((self.src_env.state_space, 1, self.tgt_env.state_space, self.action_space))
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

        lower_bound = np.zeros((self.src_env.state_space, self.tgt_env.state_space))
        match = 0.
        for t in range(self.tgt_env.state_space):
            for s in range(self.src_env.state_space):
                lower_bound[s, t] = np.max(self.src_agent.qvalues[s]) - self.dist_matrix_final[s, t]
            s_t = np.argmax(lower_bound[:, t])
            b_t = np.argmin(self.d_sa_final[s_t, 0, t])
            gt_bt = self.tgt_agent.get_best_action(t, self.tgt_possible_actions)

            if b_t == gt_bt:
                match += 1.
            qv = 1000.0
            self.transferred_agent.update_qvalue(t, b_t, qv)
        
        self.accuracy = (match / float(self.tgt_env.state_space)) * 100.
