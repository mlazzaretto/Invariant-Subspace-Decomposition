from __future__ import division
import numpy as np
from isd import ISD
from scipy.linalg import block_diag
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.2,
    'axes.unicode_minus': True,
    'axes.facecolor': 'white',
    'axes.edgecolor': '0.85',
})


def gen_data(n, p, m, block_sizes, c_coeffs, OM, rng,
             test=False, test_value=0):

    mu_x = np.zeros(p)

    X = np.zeros((n, p))
    Y = np.zeros((n, 1))
    eps = 0.55*rng.normal(size=(n, 1))
    gamma_0 = np.zeros((n, p))
    Sigma_list = np.zeros((n, p, p))
    ws = int(n/m)
    w_start = [j*ws for j in range(m)]

    y_mean = 0
    const = 2
    v_coeffs = [k not in c_coeffs for k in list(range(p))]
    shift_test = test_value
    b_ends = np.cumsum(block_sizes)
    beta_0 = np.zeros((p, 1))
    for j, b in enumerate(block_sizes):
        b_idxs = list(np.arange(b_ends[j]-block_sizes[j], b_ends[j]))
        if np.all(np.isin(b_idxs, c_coeffs)):
            beta_0[b_idxs] = const

    rng_sigma = np.random.default_rng(1)
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
                    gamma_0_w[j] = shift_test-idx/3

            gamma_0_w = OM.T@gamma_0_w
            gamma_0[w:w+ws, :] = gamma_0_w.T
            Y[w:w+ws, :] = X[w:w+ws, :]@gamma_0_w

    gamma_0 = np.zeros((n, p))
    for t in range(n):
        for j in range(p):
            if j in c_coeffs:
                gamma_0[t, j] = const
            else:
                if test:
                    gamma_0[t, j] = 1-3*(t/n)*(np.sin((j+1)*t/n+(j+1))**2)
                else:
                    gamma_0[t, j] = (3-2*t/n)
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

    p = 2
    gt_bs = [1, 1]
    gt_const_coeffs = [1]

    n_hist = 1000
    m_hist = 10
    n_rw = 20
    ws_hist = int(n_hist/5)
    OM = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    tv = -0.3
    m_train = 3
    ws_shifts = 120
    n_train = ws_shifts*m_train

    ws_train = int(8*p)
    n_test = 1
    T = n_train-ws_train-n_test
    delta_res_t = np.zeros((T, p))
    gamma_ols_t = np.zeros((T, p))

    X_hist, Y_hist, beta_0, gamma_0 = gen_data(n_hist,
                                               p, m_hist,
                                               gt_bs,
                                               gt_const_coeffs,
                                               OM, rng)
    X_train_env, Y_train_env, _, gamma_train = gen_data(n_train,
                                                        p,
                                                        m_train,
                                                        gt_bs,
                                                        gt_const_coeffs,
                                                        OM,
                                                        rng,
                                                        test=True,
                                                        test_value=tv)
    est = ISD(X_hist, Y_hist, [ws_hist]*n_rw)
    beta_inv, _, U, blocks, c_blocks = est.invariant_estimator(k_fold=10)

    for t in tqdm(range(T)):

        if t % ws_hist == 0:
            n_rw += 1
        X_train = X_train_env[t:t+ws_train, :]
        Y_train = Y_train_env[t:t+ws_train]

        gamma_ols = ols(Y_train, X_train)
        gamma_ols_t[t, :] = gamma_ols.squeeze()

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
        delta_res_t[t, :] = delta_res.squeeze()

    # Plots
    time = np.arange(T)
    sk = 10
    width = 6.0
    cm = 'viridis'
    fig, ax = plt.subplots(1, 2, figsize=(width, width*1.5/3), sharey=True,
                           sharex=True)
    sc_g0 = ax[0].scatter(gamma_train[-T:n_train+1:sk, 0],
                          gamma_train[-T:n_train+1:sk, 1],
                          c=time[::sk],
                          cmap=cm,
                          marker='h',
                          s=30,
                          linewidths=0,
                          alpha=0.7,
                          zorder=2,
                          label=r'true parameter $\gamma_{0,t}$'+'\n' +
                                r'(test data)',
                          )
    _ = ax[0].scatter(gamma_ols_t[0:T+1:sk, 0],
                      gamma_ols_t[0:T+1:sk, 1],
                      c=time[::sk],
                      cmap=cm,
                      marker=10,
                      s=40,
                      linewidths=0.5,
                      label=r'rolling window OLS' +
                      r'$\hat{\gamma}^{\text{OLS}}_t$')
    ax[0].set_xlabel(r'$\gamma^1_t$')
    ax[0].set_ylabel(r'$\gamma^2_t$')
    ax[0].grid(color='grey', axis='both', linestyle='-.', linewidth=0.25,
               alpha=0.2)
    ax[0].axhline(y=0, color='grey', linestyle='-', alpha=0.2)
    ax[0].axvline(x=0, color='grey', linestyle='-', alpha=0.2, zorder=0)
    ax[0].set_ylim(bottom=-0.15, top=3.5)
    ax[0].set_title('OLS')

    _ = ax[1].scatter(gamma_train[-T:n_train+1:sk, 0],
                      gamma_train[-T:n_train+1:sk, 1],
                      c=time[::sk],
                      cmap=cm,
                      marker='h',
                      s=30,
                      linewidths=0,
                      alpha=0.7,
                      zorder=2)
    _ = ax[1].scatter(delta_res_t[0:T+1:sk, 0]+beta_inv[0, 0],
                      delta_res_t[0:T+1:sk, 1]+beta_inv[1, 0],
                      c=time[::sk],
                      cmap=cm,
                      marker='+',
                      s=40,
                      linewidths=1.5,
                      label=r'ISD $\hat{\gamma}^{\text{ISD}}_t = $' +
                            r'$\hat{\beta}^{\text{inv}}+$' +
                            r'$\hat{\delta}^{\text{res}}_t$',
                      zorder=3)
    ax[1].axline([0, 0], [beta_0[0, 0], beta_0[1, 0]], color='tab:red',
                 linestyle='--', alpha=0.25,
                 label=r"true subspaces" + '\n' +
                 r"$\mathcal{S}^{\text{inv}}$," +
                 r"$\mathcal{S}^{\text{res}}$", zorder=0)
    ax[1].axline([0, 0], [beta_inv[0, 0], beta_inv[1, 0]], color='tab:orange',
                 linestyle='-.', alpha=0.25,
                 label=r"estimated subspaces" + '\n' +
                 r" $\hat{\mathcal{S}}^{\text{inv}}$," +
                 r"$\hat{\mathcal{S}}^{\text{res}}$", zorder=0)
    ax[1].axline(gamma_train[0, :], gamma_train[-1, :], color='tab:red',
                 linestyle='--', alpha=0.25, zorder=0)
    ax[1].axline(delta_res_t[0, :]+beta_inv.squeeze(),
                 delta_res_t[-1, :]+beta_inv.squeeze(),
                 color='tab:orange',
                 linestyle='-.', alpha=0.25, zorder=0)
    ax[1].scatter(beta_0[0, 0], beta_0[1, 0], color='tab:red', s=50,
                  marker='o', ec='tab:red', lw=0.8,
                  label=r'true invariant' + '\n'
                        + r'component $\beta^{\text{inv}}$', zorder=4)
    ax[1].scatter(beta_inv[0, 0], beta_inv[1, 0], color='tab:orange', s=50,
                  marker='x', ec='tab:orange', lw=2,
                  label=r'estimated invariant' + '\n'
                        + r'component $\hat{\beta}^{\text{inv}}$', zorder=5)
    ax[1].set_xlabel(r'$\gamma^1_t$')
    ax[1].grid(color='grey', axis='both', linestyle='-.', linewidth=0.25,
               alpha=0.2)
    plt.axhline(y=0, color='grey', linestyle='-', alpha=0.2, zorder=0)
    plt.axvline(x=0, color='grey', linestyle='-', alpha=0.2, zorder=0)
    ax[1].set_ylim(bottom=-0.1, top=3.5)
    ax[1].set_title('ISD')
    ax[1].annotate(r'$\mathcal{S}^{\text{inv}}$', (1.8, 2.9),
                   color='tab:red', alpha=1)
    ax[1].annotate(r'$\mathcal{S}^{\text{res}}$', (-0.6, 2.75),
                   color='tab:red', alpha=1)
    ax[1].annotate(r'$\hat{\mathcal{S}}^{\text{inv}}$', (1.2, 2.9),
                   color='tab:orange', alpha=1)
    ax[1].annotate(r'$\hat{\mathcal{S}}^{\text{res}}$', (-0.9, 2.3),
                   color='tab:orange', alpha=1)

    leg = fig.legend(bbox_to_anchor=(0.035, 0.05), ncol=4,
                     loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    leg.legend_handles[0].set_alpha(0.7)
    leg.legend_handles[1].set_alpha(0.7)
    leg.legend_handles[2].set_alpha(0.7)
    leg.legend_handles[3].set_alpha(0.5)
    leg.legend_handles[4].set_alpha(0.5)

    cbar_ax = fig.add_axes([1, 0.18, 0.025, 0.7])
    fig.colorbar(sc_g0, cax=cbar_ax, pad=0, label='$t$ (test)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
