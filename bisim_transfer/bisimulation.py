import numpy as np

class Bisimulation(object):
    def __init__(self, opts):
        self.src_env = None
        self.tgt_env = None
        self.src_agent = None
        self.tgt_agent = None
        self.transferred_agent = None
        self.solver = None
        self.d_sa_final = np.zeros((src_env.state_space, tgt_env.state_space))
        self.dist_matrix_final = np.zeros((src_env.state_space, tgt_env.state_space))
        self.lfp_iters = opts.lfp_iters
        self.threshold = opts.threshold
        self.discount_factor_kd = opts.df_kd
        self.discount_factor_reward = opts.df_reward
        self.sparse_A = None
    
    def solver_lp(self):
        raise NotImplementedError
    
    def solver_pyemd(self):
        raise NotImplementedError
    
    def bisimulation(self):
        raise NotImplementedError
    
    def lax_bisimulation(self):
        raise NotImplementedError
    
    def pess_bisimulations(self):
        raise NotImplementedError

    def opt_bisimulations(self):
        raise NotImplementedError
    
    def generate_logs(self):
        raise NotImplementedError