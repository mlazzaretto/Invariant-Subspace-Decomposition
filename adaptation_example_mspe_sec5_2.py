from __future__ import division
import numpy as np
from isd import ISD
from scipy.linalg import block_diag
from scipy.stats import ortho_group
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.2,
    'axes.unicode_minus': True,
})
width = 6.0
# functions


def gen_data(n, p, m, block_sizes, c_coeffs, OM, rng,
             test=False, test_value=0):

    mu_x = np.zeros(p)

    X = np.zeros((n, p))
    Y = np.zeros((n, 1))
    eps = 0.8*rng.normal(size=(n, 1))
    gamma_0 = np.zeros((n, p))
    Sigma_list = np.zeros((n, p, p))
    ws = int(n/m)
    w_start = [j*ws for j in range(m)]

    y_mean = 0
    const = 0.2
    v_coeffs = [k not in c_coeffs for k in list(range(p))]
    shift_test = test_value
    b_ends = np.cumsum(block_sizes)
    beta_0 = np.zeros((p, 1))
    for j, b in enumerate(block_sizes):
        b_idxs = list(np.arange(b_ends[j]-block_sizes[j], b_ends[j]))
        if np.all(np.isin(b_idxs, c_coeffs)):
            beta_0[b_idxs] = const

    rng_sigma = np.random.default_rng(0)
    for idx, w in enumerate(w_start):
        if idx == m-1:
            ws = n - w_start[-1]
        A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
        X[w:w+ws, :] = rng.multivariate_normal(mean=mu_x,
                                               cov=A@A.T,
                                               size=ws)
        X[w:w+ws, :] = X[w:w+ws, :]@OM
        Sigma = OM.T@(A@A.T+0*np.eye(p))@OM
        for i in range(ws):
            Sigma_list[w+i, :, :] = Sigma

        gamma_0_w = np.zeros((p, 1))
        gamma_0_w[c_coeffs] = const
        if test:
            for j, var in enumerate(v_coeffs):
                if var:
                    gamma_0_w[j] = shift_test
            gamma_0_w = OM.T@gamma_0_w
            gamma_0[w:w+ws, :] = gamma_0_w.T
            Y[w:w+ws, :] = X[w:w+ws, :]@gamma_0_w

    if not test:
        gamma_0 = np.zeros((n, p))
        for t in range(n):
            for j in range(p):
                if j in c_coeffs:
                    gamma_0[t, j] = const
                else:
                    gamma_0[t, j] = 1-1.5*(t/n)*(np.sin((j+1)*t/n+(j+1))**2)
            gamma_0[t, :] = OM.T@gamma_0[t, :]
            Y[t, 0] = X[t, :]@gamma_0[t, :].T

    Y = Y + y_mean + eps

    return X, Y, OM.T@beta_0, gamma_0


def ols(Y, X):
    X_1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    return sm.OLS(Y, X_1).fit().params[:-1].reshape(-1, 1)


def xpl_var(X, Y, beta):
    n = X.shape[0]
    beta = beta.reshape(-1, 1)
    res = Y-X@beta
    return (1/n)*(Y.T@Y - res.T@res)


def main():
    rng = np.random.default_rng(0)

    n_iter = 20
    p = 10
    gt_bs = [2, 4, 3, 1]
    gt_inv_blocks = np.array([False, True, True, False])
    gt_const_coeffs = list(range(2, 9))

    n_hist = 6000
    m_hist = 10
    n_rw = 25
    ws_hist = int(n_hist/8)
    OM = ortho_group.rvs(dim=p, random_state=rng)
    tv_list = [-0.5, -2]
    m_train = len(tv_list)
    ws_shifts = 1000
    n_train = ws_shifts

    ws_train_list = [int(1.5*p), 2*p, 3*p, 4*p, 5*p, int(7.5*p), 10*p]
    n_test = 1
    n_est = 4
    n_ex = len(ws_train_list)

    mse = np.zeros((m_train, n_est, n_iter, n_ex))
    xpl_v = np.zeros((m_train, n_est, n_iter, n_ex))
    beta_err = np.zeros((n_iter))

    for iter in tqdm(range(n_iter)):
        X_hist, Y_hist, beta_0, gamma_0 = gen_data(n_hist,
                                                   p, m_hist,
                                                   gt_bs,
                                                   gt_const_coeffs,
                                                   OM, rng)
        est = ISD(X_hist, Y_hist, [ws_hist]*n_rw)
        beta_inv, beta_icpt, U, blocks, c_blocks = \
            est.invariant_estimator(k_fold=10)
        beta_ols = est.get_pooled_est()[:, :]
        beta_err[iter] = np.mean((beta_0-beta_inv)**2)

        for tv_idx, tv in enumerate(tv_list):
            X_train_env, Y_train_env, _, gamma_train = \
                gen_data(n_train,
                         p,
                         1,
                         gt_bs,
                         gt_const_coeffs,
                         OM,
                         rng,
                         test=True,
                         test_value=tv)

            for m_idx, ws_train in enumerate(ws_train_list):
                T = n_train-ws_train-n_test
                xpl_v_w = np.zeros((n_est, T))
                mse_w = np.zeros((n_est, T))

                for t in range(T):
                    X_train = X_train_env[t:t+ws_train, :]
                    Y_train = Y_train_env[t:t+ws_train]
                    X_test = X_train_env[t+ws_train:t+ws_train+n_test, :]
                    Y_test = Y_train_env[t+ws_train:t+ws_train+n_test]
                    gamma_0_t = gamma_train[0, :].reshape(-1, 1)

                    # Test
                    # OLS on test window
                    gamma_ols = ols(Y_train, X_train)

                    # OLS on inv residuals
                    if gt_inv_blocks.any():
                        v_idxs = []
                        be = np.cumsum(gt_bs)
                        for j, b in enumerate(gt_inv_blocks):
                            if not b:
                                for idx in np.arange(be[j]-gt_bs[j], be[j]):
                                    v_idxs.append(idx)
                        Y_res_inv_0 = Y_train-X_train@beta_0
                        X_var_0 = X_train@(OM.T)
                        X_var_0 = X_var_0[:, v_idxs]
                        delta_res_0 = OM.T[:, v_idxs]@ols(Y_res_inv_0, X_var_0)
                    else:
                        delta_res_0 = gamma_ols

                    if c_blocks.any():
                        v_idxs = []
                        be = np.cumsum(blocks)
                        for j, b in enumerate(c_blocks):
                            if not b:
                                for idx in np.arange(be[j]-blocks[j], be[j]):
                                    v_idxs.append(idx)
                        Y_res_inv = Y_train-X_train@beta_inv
                        X_var = X_train@(U.T)
                        X_var = X_var[:, v_idxs]
                        delta_res = U.T[:, v_idxs]@ols(Y_res_inv, X_var)
                    else:
                        delta_res = gamma_ols

                    xpl_v_w[0, t] = xpl_var(X_test, Y_test, delta_res_0+beta_0)
                    xpl_v_w[1, t] = xpl_var(X_test, Y_test, delta_res+beta_inv)
                    xpl_v_w[2, t] = xpl_var(X_test, Y_test, gamma_ols)
                    xpl_v_w[3, t] = xpl_var(X_test, Y_test, beta_ols)

                    mse_w[0, t] = np.mean((Y_test -
                                           X_test@(delta_res_0+beta_0))**2)
                    mse_w[1, t] = np.mean((Y_test -
                                           X_test@(delta_res+beta_inv))**2)
                    mse_w[2, t] = np.mean((Y_test-X_test@gamma_ols)**2)
                    mse_w[0, t] = np.mean((X_test@gamma_0_t -
                                           X_test@(delta_res_0+beta_0))**2)
                    mse_w[1, t] = np.mean((X_test@gamma_0_t -
                                           X_test@(delta_res+beta_inv))**2)
                    mse_w[2, t] = np.mean((X_test@gamma_0_t -
                                           X_test@gamma_ols)**2)
                    mse_w[3, t] = np.mean((X_test@gamma_0_t -
                                           X_test@beta_ols)**2)

                for est_idx in range(n_est):
                    xpl_v[tv_idx, est_idx, iter, m_idx] = \
                        np.mean(xpl_v_w[est_idx, :])
                    mse[tv_idx, est_idx, iter, m_idx] = \
                        np.mean(mse_w[est_idx, :])

    c = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    leg = ['m=1.5p', 'm=2p', 'm=5p', 'm=10p']
    lines = [None]*2
    lines2 = [None]*2
    idxs = []
    if len(ws_train_list) == len(leg):
        idxs = list(range(ws_train_list))
    else:
        idxs = [0, 1, 4, 6]
    width = 6.0
    fig, ax = plt.subplots(1, len(leg), figsize=(width, width/(4*len(leg))*5),
                           sharey=True)
    for k in range(len(tv_list)):
        for i, j in enumerate(idxs):
            lines[k] = ax[i].scatter(mse[k, 1, :, j], mse[k, 2, :, j],
                                     color=c[0], alpha=0.35)
    for k in range(len(tv_list)):
        for i, j in enumerate(idxs):
            lines2[k] = ax[i].scatter(mse[k, 0, :, j], mse[k, 2, :, j],
                                      color=c[1], alpha=0.2)

    for j in range(len(idxs)):
        lim = 5
        liml = 0.005
        temp = np.linspace(0, lim, 100)
        ax[j].plot(temp, temp, '-.', color=c[3], alpha=0.3)
        ax[j].axvline(0, ls='-.', color=c[3], alpha=0.3)
        ax[j].axhline(0, ls='-.', color=c[3], alpha=0.3)
        ax[j].set_xscale('log')
        ax[j].set_yscale('log')
        ax[j].set_xlim([liml, lim])
        ax[j].set_ylim([liml, lim])
        ax[j].set_title(leg[j], fontsize=10)
        ax[j].grid(color='grey', axis='both', linestyle='-', linewidth=0.25,
                   alpha=0.25)
    ax[0].set_ylabel(r'$\mathsf{MSPE}(\hat{\gamma}^{\text{OLS}}_t)$')
    fig.supxlabel('\n', fontsize=2)
    leg = fig.legend([lines[0], lines2[0]],
                     [r'$\mathsf{MSPE}(\hat{\gamma}^{\text{ISD}}_t)$',
                      r'$\mathsf{MSPE}(\beta^{\text{inv}}+$' +
                      r'$\hat{\delta}^{\text{res}}_t)$'],
                     ncol=2, loc='lower center', frameon=False)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    plt.tight_layout()
    plt.show()

    #
    width = 6.0
    st = 0
    fig, ax = plt.subplots(figsize=(0.5*width, 0.5*width*2.2/3))
    x = 3*(0.8**2)*np.reciprocal(np.array(ws_train_list[st:]), dtype=float)
    bp = ax.boxplot(mse[0, 2, :, st:] - mse[0, 1, :, st:],
                    positions=x,
                    widths=0.005,
                    showmeans=True,
                    boxprops=dict(color=c[0]),
                    capprops=dict(color=c[0]),
                    whiskerprops=dict(color=c[0]),
                    flierprops=dict(markeredgecolor=c[0]),
                    meanprops=dict(marker='o',
                                   markerfacecolor=c[0],
                                   markeredgecolor='none'),
                    medianprops=dict(linestyle=None, linewidth=0))
    bp2 = ax.boxplot(mse[0, 2, :, st:] - mse[0, 0, :, st:],
                     positions=x,
                     widths=0.005,
                     showmeans=True,
                     boxprops=dict(color=c[1]),
                     capprops=dict(color=c[1]),
                     whiskerprops=dict(color=c[1]),
                     flierprops=dict(markeredgecolor=c[1]),
                     meanprops=dict(marker='o',
                                    markerfacecolor=c[1],
                                    markeredgecolor='none'),
                     medianprops=dict(linestyle=None, linewidth=0))
    for b in bp2['boxes']:
        b.set_alpha(0.8)
    for b in bp2['whiskers']:
        b.set_alpha(0.8)
    for b in bp2['caps']:
        b.set_alpha(0.8)
    for b in bp2['fliers']:
        b.set_alpha(0.8)
    for b in bp2['means']:
        b.set_alpha(0.8)
    ax.set_xlim([0.015, 0.135])
    bound = ax.axline((0, 0), (1, 1), color='r', linestyle='--')
    ax.legend([bp['boxes'][0], bp2['boxes'][0], bound],
              [r'$\mathsf{MSPE}(\hat{\gamma}^{\text{OLS}}_t)-$' +
               r'$\mathsf{MSPE}(\hat{\gamma}^{\text{ISD}}_t)$',
               r'$\mathsf{MSPE}(\hat{\gamma}^{\text{OLS}}_t)-$' +
               r'$\mathsf{MSPE}(\beta^{\text{inv}}+$' +
               r'$\hat{\delta}^{\text{res}}_t)$',
               r'$\sigma_{\text{ad}}^2$' +
               r'$\frac{\text{dim}(\mathcal{S}^{\text{inv}})}{m}$'],
              fontsize=8)
    ax.set_xlabel(r'$\sigma_{\text{ad}}^2$' +
                  r'$\frac{\text{dim}(\mathcal{S}^{\text{inv}})}{m}$')
    ax.grid(color='grey', axis='both', linestyle='-', linewidth=0.25,
            alpha=0.25)
    ax.set_xticks(x)
    plt.xticks(rotation=45, ha='right')
    ax.set_xticklabels([r'$'+str(xx)+'$' for xx in np.around(x, 2)])
    plt.tight_layout()
    plt.show()

    width = 6.0
    st = 0
    fig, ax = plt.subplots(figsize=(0.5*width, 0.5*width*2.2/3))
    x = 7*(0.8**2)*np.reciprocal(np.array(ws_train_list[st:]), dtype=float)
    bp = ax.boxplot(mse[0, 1, :, st:],
                    positions=x,
                    widths=0.015,
                    showmeans=True,
                    boxprops=dict(color=c[0]),
                    capprops=dict(color=c[0]),
                    whiskerprops=dict(color=c[0]),
                    flierprops=dict(markeredgecolor=c[0]),
                    meanprops=dict(marker='o',
                                   markerfacecolor=c[0],
                                   markeredgecolor='none'),
                    medianprops=dict(linestyle=None, linewidth=0))
    bp2 = ax.boxplot(mse[0, 0, :, st:],
                     positions=x,
                     widths=0.015,
                     showmeans=True,
                     boxprops=dict(color=c[1]),
                     capprops=dict(color=c[1]),
                     whiskerprops=dict(color=c[1]),
                     flierprops=dict(markeredgecolor=c[1]),
                     meanprops=dict(marker='o',
                                    markerfacecolor=c[1],
                                    markeredgecolor='none'),
                     medianprops=dict(linestyle=None, linewidth=0))
    for b in bp2['boxes']:
        b.set_alpha(0.8)
    for b in bp2['whiskers']:
        b.set_alpha(0.8)
    for b in bp2['caps']:
        b.set_alpha(0.8)
    for b in bp2['fliers']:
        b.set_alpha(0.8)
    for b in bp2['means']:
        b.set_alpha(0.8)
    ax.set_xlim([0.03, 0.32])
    bound = ax.axline((0, 0), (1, 1), color='r', linestyle='--')
    ax.legend([bp['boxes'][0], bp2['boxes'][0], bound],
              [r'$\mathsf{MSPE}(\hat{\gamma}^{\text{ISD}}_t)$',
               r'$\mathsf{MSPE}(\beta^{\text{inv}}+$' +
               r'$\hat{\delta}^{\text{res}}_t)$',
               r'$\sigma_{\text{ad}}^2$' +
               r'$\frac{\text{dim}(\mathcal{S}^{\text{res}})}{m}$'],
              fontsize=8)
    ax.set_xlabel(r'$\sigma_{\text{ad}}^2$' +
                  r'$\frac{\text{dim}(\mathcal{S}^{\text{res}})}{m}$')
    ax.grid(color='grey', axis='both', linestyle='-', linewidth=0.25,
            alpha=0.25)
    ax.set_xticks(x)
    plt.xticks(rotation=45, ha='right')
    ax.set_xticklabels([r'$'+str(xx)+'$' for xx in np.around(x, 2)])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
