from __future__ import division
import numpy as np
from jbd import jbd, ajbd
import statsmodels.api as sm
from cvxopt.solvers import qp
from cvxopt import matrix


class Estimator():
    # Estimator: initializes estimators from data
    # X: nxp matrix of observed covariates
    # Y: nx1 vector of observed responses
    # win_size: list of window sizes to be used for estimation

    def __init__(self, X, Y, win_size):
        self.X = X
        self.Y = Y
        self.m = len(win_size)
        self.w = win_size
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.w_st = np.linspace(0, self.n-self.w[-1], num=self.m,
                                endpoint=True, dtype=int)
        self.w_end = self.w_st + self.w

        # estimates initialization
        self.beta_maximin = None
        self.beta_pcm = None
        self.beta_inv = np.zeros((self.p, 1))
        self.beta_ev = None
        self.blocks = list([self.p])

    def invariant_est(self, diag=False):
        # estimator for the invariant component beta_inv
        # diag: True if Sigmas are assumed to be jointly diagonalizable

        # auxiliary functions
        def comp_thresholds(X, Y, w, mu):
            # compute invariance threshold T for constant coefficient selection
            # for a given block
            # X, Y: observed predictors and responses
            # w: list of window dimensions over which mu is computed
            # mu: mean coefficient value

            m = len(w)
            n = X.shape[0]
            w_start = np.linspace(0, n-w[-1], num=m,
                                  endpoint=True, dtype=int)
            c = np.zeros((m, ))
            v = np.zeros((m, ))
            for k in range(m):
                X_w = X[w_start[k]:w_start[k]+w[k], :]
                Y_w = Y[w_start[k]:w_start[k]+w[k], :]
                cov_w = np.cov(Y_w-X_w@mu, X_w@mu, rowvar=False)
                c[k] = cov_w[0, 1]
                v[k] = np.sqrt(cov_w[0, 0]*cov_w[1, 1])

            T = np.mean([np.abs((c[k])/v[k]) for k in range(m)])
            return T

        def check_const(blocks_shape, th_const, th):
            # check which blocks are invariant for a given threshold
            # blocks shape: list of blocks dimensions
            # th_const: list of thresholds corresponding to each block
            # th: reference threshold to check
            # returns:
            #   const_blocks: boolean vector for constant blocks
            #   const_idxs: list of covariates indices corresponding
            #               to constant blocks
            #   v_idxs: list of covariates indices corresponding
            #           to time-varying blocks
            const_blocks = np.zeros(len(blocks_shape), dtype=bool)
            const_idxs = []
            v_idxs = []
            for b, bs in enumerate(blocks_shape):
                if b == 0:
                    block_idxs = list(range(bs))
                else:
                    block_idxs = [j+sum(blocks_shape[:b])
                                  for j in range(bs)]
                if th_const[b] < th:
                    const_blocks[b] = True
                    for idx in block_idxs:
                        const_idxs.append(idx)
                else:
                    for idx in block_idxs:
                        v_idxs.append(idx)
            return const_blocks, const_idxs, v_idxs

        def xpl_var(X, Y, beta):
            # compute explained varinace by beta (under zero mean ass.)
            # X, Y: observed covariates and response
            # beta: linear parameter
            if len(X.shape) == 1:
                n = 1
                X = X.reshape(1, -1)
                Y = Y.reshape(1, -1)
            else:
                n = X.shape[0]
            return (1/n)*(2*Y.T@X@beta - beta.T@X.T@X@beta)

        # 1) Compute windows covariance and coefficients estimates
        Sigma = np.zeros((self.m, self.p, self.p))
        gamma = np.zeros((self.m, self.p+1, 1))
        # stack constant term in X
        X_1 = np.concatenate((self.X, np.ones((self.n, 1))), axis=1)

        w_start = self.w_st
        for idx, win in enumerate(self.w):
            X_w = self.X[w_start[idx]:w_start[idx]+win, :]
            X_1_w = X_1[w_start[idx]:w_start[idx]+win, :]
            Y_w = self.Y[w_start[idx]:w_start[idx]+win, :]
            reg_w = sm.OLS(Y_w, X_1_w).fit()
            gamma[idx, :, :] = reg_w.params.reshape(-1, 1)
            Sigma[idx, :, :] = np.cov(X_w, rowvar=False)

        # Save pooled regression coefficient
        self.beta_pcm = np.mean(gamma, axis=0)

        # 2) Joint block diagonalization
        if diag:
            U, converged, iteration, meanoffdiag = jbd(Sigma,
                                                       threshold=0,
                                                       diag=True)
            blocks = [set([j]) for j in range(self.p)]
            Sigma_diag = np.zeros_like(Sigma)
            for k in range(self.m):
                Sigma_diag[k, :, :] = U@Sigma[k, :, :]@U.T.conj()
        else:
            U, blocks, Sigma_diag, t_opt, boff_opt = ajbd(Sigma)

        blocks_shape = [len(block) for block in blocks]

        # Transform estimated coefficients using JBD U
        # and check if they are constant
        gamma_til = np.zeros((self.m, self.p, 1))
        for k in range(self.m):
            gamma_til[k, :, :] = U@gamma[k, :-1, :]

        beta_icpt = self.beta_pcm[-1]
        # Compute constant thresholds for every estimated block
        th_const = []
        for b, bs in enumerate(blocks_shape):
            if b == 0:
                block_idxs = list(range(bs))
            else:
                block_idxs = [j+sum(blocks_shape[:b])
                              for j in range(bs)]
            gamma_block = gamma_til[:, block_idxs, :]
            mu_block = np.mean(gamma_block, axis=0)
            th_const.append(comp_thresholds(self.X@(U.T[:, block_idxs]),
                                            self.Y,
                                            self.w,
                                            mu_block
                                            ))
        th_const.append(1)

        # 3) Ivariant parameter estimation
        #  Cross-validation for threshold selection
        xi_inv = np.zeros((self.p, 1))
        k_fold = 10
        w_fold = int(self.n/k_fold)
        wst_fold = np.linspace(0, self.n-w_fold, num=k_fold,
                               endpoint=True, dtype=int)
        ws_test = 2*self.p
        xpl_v = np.zeros(len(th_const))
        xpl_v_th = np.zeros((len(th_const), k_fold))
        for th_idx, th in enumerate(th_const):
            # Detect constant blocks for threshold th
            const_blocks, const_idxs, v_idxs = check_const(blocks_shape,
                                                           th_const,
                                                           th)
            for fd_idx in range(k_fold):
                # Leave out one fold for testing
                T_test = w_fold-ws_test-1
                xpl_v_fd = np.zeros(T_test)
                X_tr = np.delete(self.X,
                                 np.s_[wst_fold[fd_idx]:
                                       wst_fold[fd_idx]+w_fold],
                                 axis=0)
                Y_tr = np.delete(self.Y,
                                 np.s_[wst_fold[fd_idx]:
                                       wst_fold[fd_idx]+w_fold],
                                 axis=0)
                X_ts = self.X[wst_fold[fd_idx]:
                              wst_fold[fd_idx]+w_fold, :]
                Y_ts = self.Y[wst_fold[fd_idx]:
                              wst_fold[fd_idx]+w_fold, :]

                # Compute candidate invariant parameter
                if const_idxs:
                    X_inv = X_tr@(U.T[:, const_idxs])
                    X_inv = np.concatenate((X_inv,
                                            np.ones((X_inv.shape[0], 1))),
                                           axis=1)
                    xi_inv_1 = sm.OLS(Y_tr, X_inv).fit()\
                        .params[:-1].reshape(-1, 1)
                    beta_inv = U.T[:, const_idxs]@xi_inv_1
                else:
                    beta_inv = np.zeros((self.p, 1))

                # Compute empirical explained variance by adapted
                # invariant parameter on test fold
                for t in range(T_test):
                    X_test = X_ts[t:t+ws_test, :]
                    Y_test = Y_ts[t:t+ws_test, :]
                    X_val = X_ts[t+ws_test:t+ws_test+1, :]
                    Y_val = Y_ts[t+ws_test:t+ws_test+1, :]
                    if v_idxs:
                        Y_res_inv = Y_test-X_test@beta_inv
                        X_var = X_test@(U.T)
                        X_var = np.concatenate((X_var[:, v_idxs],
                                                np.ones((X_var.shape[0], 1))),
                                               axis=1)
                        delta_res = U.T[:, v_idxs] @ \
                            (sm.OLS(Y_res_inv, X_var).fit()
                                .params[:-1].reshape(-1, 1))
                    else:
                        delta_res = np.zeros((self.p, 1))
                    xpl_v_fd[t] = xpl_var(X_val, Y_val, delta_res+beta_inv)

                xpl_v_th[th_idx, fd_idx] = np.mean(xpl_v_fd)
            # Average over folds
            xpl_v[th_idx] = np.mean(xpl_v_th[th_idx, :])
        # Standard error across folds for all thresholds
        std_th = (1/np.sqrt(k_fold))*np.std(xpl_v_th, axis=1)
        # Minimum explained variance across folds for all thresholds
        xpl_v_min = np.min(xpl_v_th, axis=1)

        # Optimal threshold selection
        if np.all(xpl_v_min[:-1] <= 0):
            th_opt = 0
        else:
            sort_th = np.argsort(th_const)
            xpl_v_s = xpl_v[sort_th]
            xv_max_idx = np.argmax(xpl_v_s)
            xv_max = xpl_v_s[xv_max_idx]
            xv_min = xv_max - std_th[sort_th][xv_max_idx]
            th_cand = np.where(xpl_v_s[0:xv_max_idx+1] > xv_min)[0][0]
            th_opt = np.array(th_const)[sort_th][th_cand]

        # Compute invariant parameter for optimal threshold
        xi_inv = np.zeros((self.p, 1))
        const_blocks, const_idxs, v_idxs = check_const(blocks_shape,
                                                       th_const,
                                                       th_opt)

        if const_idxs:
            X_inv = self.X@(U.T[:, const_idxs])
            X_1_inv = np.concatenate((X_inv, np.ones((self.n, 1))), axis=1)
            reg_inv = sm.OLS(self.Y, X_1_inv).fit()
            xi_inv = reg_inv.params[:-1].reshape(-1, 1)
            beta_inv = U.T[:, const_idxs]@xi_inv
        else:
            beta_inv = np.zeros((self.p, 1))

        return beta_inv, beta_icpt, U, blocks_shape, const_blocks

    def magging_estimator(self):
        # Implementation of the maximin effect following the pseudocode
        # provided in the reference paper
        # Reference: Bühlmann, P., & Meinshausen, N. (2015).
        #            Magging: maximin aggregation for inhomogeneous
        #            large-scale data.
        #            Proceedings of the IEEE, 104(1), 126-135.

        beta_hat_g_list = []
        w_start = self.w_st
        for g, gs in enumerate(self.w):
            X_g = self.X[w_start[g]:w_start[g]+gs, :]
            X_1_g = np.concatenate((X_g, np.ones((gs, 1))), axis=1)
            Y_g = self.Y[w_start[g]:w_start[g]+gs, :]
            beta_hat_g_list.append(sm.OLS(Y_g, X_1_g).fit().params[:-1])

        beta_hat = np.array(beta_hat_g_list).T
        g_num = beta_hat.shape[1]

        Sigma_hat = (1/self.n)*self.X.T@self.X

        H = beta_hat.T@Sigma_hat@beta_hat
        H = matrix(H, (g_num, g_num))
        A = matrix(np.concatenate((np.ones((1, g_num)),
                                   np.eye(g_num)), axis=0),
                   (g_num+1, g_num))

        B = np.zeros((g_num+1, 1))
        B[0, 0] = 1
        B = matrix(B)
        Q = matrix(np.zeros((g_num, 1)))

        weights = qp(H, Q, -A, -B, options={'show_progress': False})
        w = np.array(weights['x'])

        self.beta_maximin = beta_hat@w
        return self.beta_maximin

    def get_pooled_est(self):
        # Returns pooled OLS solution computed by invariant_est()
        return self.beta_pcm
