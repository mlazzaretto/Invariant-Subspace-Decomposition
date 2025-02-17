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
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.2,
    'axes.unicode_minus': True,
    'axes.titlesize': 'small'
})
# functions


def gen_data(n, p, m, block_sizes, c_coeffs, OM, rng,
             test=False, test_value=0):
    # Generate observational data
    # n: sample size
    # p: covariates dimension
    # m: number of covariates shifts
    # block_sizes: sizes of diagonal blocks/partition subspaces
    # c_coeffs: indexes of constant coefficients
    # OM: orthonormal matrix used for rotation
    # rng: random generator
    # test: if True, generates test (adaptation) data
    # test_value: (list of) values of time-varying coefficients

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

    if not test:
        gamma_0 = np.zeros((n, p))
        for t in range(n):
            for j in range(p):
                if j in c_coeffs:
                    gamma_0[t, j] = const
                else:
                    gamma_0[t, j] = 0.5-1*(t/n)*(np.sin((j+1)*t/n+(j+1))**2)
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
    tv = -0.3
    m_train = 3
    ws_shifts = 150
    n_train = ws_shifts*m_train

    ws_train = 3*p
    n_test = 1
    n_est = 9

    T = n_train-ws_train-n_test
    xpl_v_train = np.zeros((n_iter, n_est, n_hist))
    xpl_v = np.zeros((n_iter, n_est, T))

    for iter in range(n_iter):
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
        beta_ols = est.get_pooled_est()[:, :]
        beta_mm = est.magging_estimator()
        beta_mm_a = beta_mm
        beta_ols_full = ols(Y_hist, X_hist)

        for t in tqdm(range(n_hist)):
            # true time-var parameter
            xpl_v_train[iter, 0, t] = xpl_var(X_hist[t:t+1, :],
                                              Y_hist[t:t+1, :],
                                              gamma_0[t:t+1, :])
            # time-inv subspace effect
            xpl_v_train[iter, 1, t] = xpl_var(X_hist[t:t+1, :],
                                              Y_hist[t:t+1, :],
                                              beta_inv)
            # maximin
            xpl_v_train[iter, 2, t] = xpl_var(X_hist[t:t+1, :],
                                              Y_hist[t:t+1, :],
                                              beta_mm)
            # pooled ols
            xpl_v_train[iter, 3, t] = xpl_var(X_hist[t:t+1, :],
                                              Y_hist[t:t+1, :],
                                              beta_ols)
            # full ols
            xpl_v_train[iter, 4, t] = xpl_var(X_hist[t:t+1, :],
                                              Y_hist[t:t+1, :],
                                              beta_ols_full)
            # oracle time-inv sub effect
            xpl_v_train[iter, 5, t] = xpl_var(X_hist[t:t+1, :],
                                              Y_hist[t:t+1, :],
                                              beta_0)

        for t in tqdm(range(T)):
            X = np.concatenate((X_hist, X_train_env[:t+1, :]))
            Y = np.concatenate((Y_hist, Y_train_env[:t+1]))

            if t % ws_hist == 0:
                n_rw += 1
                est_a = ISD(X, Y, [ws_hist]*n_rw)
                beta_mm_a = est_a.magging_estimator()
            X_train = X[-ws_train:, :]
            Y_train = Y[-ws_train:]
            X_test = X_train_env[t+1:t+1+n_test, :]
            Y_test = Y_train_env[t+1:t+1+n_test]
            gamma_0_t = gamma_train[t, :].reshape(-1, 1)

            gamma_ols = ols(Y_train, X_train)
            beta_ols_full = ols(Y_hist, X_hist)

            # Oracle time adaptation
            if gt_inv_blocks.any():
                v_idxs_gt = []
                be = np.cumsum(gt_bs)
                for j, b in enumerate(gt_inv_blocks):
                    if not b:
                        for idx in np.arange(be[j]-gt_bs[j], be[j]):
                            v_idxs_gt.append(idx)
                Y_res_inv_0 = Y_train-X_train@beta_0
                X_var_0 = X_train@(OM.T)
                X_var_0 = X_var_0[:, v_idxs_gt]
                delta_res_0 = OM.T[:, v_idxs_gt]@ols(Y_res_inv_0, X_var_0)
            else:
                delta_res_0 = gamma_ols

            # Time adaptation
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

            xpl_v[iter, 0, t] = xpl_var(X_test, Y_test, gamma_0_t)
            xpl_v[iter, 1, t] = xpl_var(X_test, Y_test, delta_res_0+beta_0)
            xpl_v[iter, 2, t] = xpl_var(X_test, Y_test, beta_0)
            xpl_v[iter, 3, t] = xpl_var(X_test, Y_test, delta_res+beta_inv)
            xpl_v[iter, 4, t] = xpl_var(X_test, Y_test, beta_inv)
            xpl_v[iter, 5, t] = xpl_var(X_test, Y_test, gamma_ols)
            xpl_v[iter, 6, t] = xpl_var(X_test, Y_test, beta_ols_full)
            xpl_v[iter, 7, t] = xpl_var(X_test, Y_test, beta_mm)
            xpl_v[iter, 8, t] = xpl_var(X_test, Y_test, beta_mm_a)

    # Plotting
    c = ['tab:blue', 'tab:orange', 'tab:cyan', 'tab:red',
         'tab:purple', 'tab:brown', 'tab:green', 'tab:gray',
         'tab:olive', 'tab:cyan']
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
    })
    width = 6.0
    mean = True
    std = True
    fig, ax = plt.subplots(1, 1, figsize=(0.5*width, 0.5*width*2/3),
                           layout='constrained')
    if mean:
        a = 1
        l2, = ax.plot(np.arange(T),
                      np.cumsum(np.mean(xpl_v[:, 4, :], axis=0)),
                      label=r'$\hat{\beta}^{\text{inv}}$',
                      linestyle='--',
                      color=c[1],
                      alpha=a)
        l3, = ax.plot(np.arange(T),
                      np.cumsum(np.mean(xpl_v[:, 3, :], axis=0)),
                      label=r'$\hat{\gamma}^{\text{ISD}}_t=$' +
                      r'$\hat{\beta}^{\text{inv}}+$' +
                      r'$\hat{\delta}^{\text{res}}_t$',
                      color=c[2],
                      alpha=a)
        l4, = ax.plot(np.arange(T),
                      np.cumsum(np.mean(xpl_v[:, 5, :], axis=0)),
                      label=r'$\hat{\gamma}^{\text{OLS}}_t$',
                      color=c[3],
                      alpha=a)
        l5, = ax.plot(np.arange(T),
                      np.cumsum(np.mean(xpl_v[:, 8, :], axis=0)),
                      label=r'$\hat{\beta}^{\text{mm}}$',
                      linestyle='--',
                      color=c[4],
                      alpha=a)
        l6, = ax.plot(np.arange(T),
                      np.cumsum(np.mean(xpl_v[:, 0, :], axis=0)),
                      label=r'$\gamma_{0, t}$',
                      linestyle=':',
                      color=c[5],
                      alpha=a)
        l7, = ax.plot(np.arange(T),
                      np.cumsum(np.mean(xpl_v[:, 6, :], axis=0)),
                      label=r'$\hat{\beta}^{\text{OLS}}$',
                      linestyle='--',
                      color=c[6],
                      alpha=a)
    if std:
        a = 0.15
        ax.fill_between(np.arange(T),
                        np.cumsum(np.mean(xpl_v[:, 4, :], axis=0)) -
                        (np.std(np.cumsum(xpl_v[:, 4, :], axis=-1), axis=0)),
                        np.cumsum(np.mean(xpl_v[:, 4, :], axis=0)) +
                        (np.std(np.cumsum(xpl_v[:, 4, :], axis=-1), axis=0)),
                        label=r'$\hat{\beta}^{\text{inv}}$',
                        color=c[1],
                        alpha=a)
        ax.fill_between(np.arange(T),
                        np.cumsum(np.mean(xpl_v[:, 3, :], axis=0)) -
                        (np.std(np.cumsum(xpl_v[:, 3, :], axis=-1), axis=0)),
                        np.cumsum(np.mean(xpl_v[:, 3, :], axis=0)) +
                        (np.std(np.cumsum(xpl_v[:, 3, :], axis=-1), axis=0)),
                        label=r'$\hat{\beta}^{\text{inv}}+$' +
                        r'$\hat{\delta}^{\text{res}}_t$',
                        color=c[2],
                        alpha=a)
        ax.fill_between(np.arange(T),
                        np.cumsum(np.mean(xpl_v[:, 5, :], axis=0)) -
                        (np.std(np.cumsum(xpl_v[:, 5, :], axis=-1), axis=0)),
                        np.cumsum(np.mean(xpl_v[:, 5, :], axis=0)) +
                        (np.std(np.cumsum(xpl_v[:, 5, :], axis=-1), axis=0)),
                        label=r'$\hat{\gamma}^{\text{OLS}}_t$',
                        color=c[3],
                        alpha=a)
        ax.fill_between(np.arange(T),
                        np.cumsum(np.mean(xpl_v[:, 8, :], axis=0)) -
                        (np.std(np.cumsum(xpl_v[:, 8, :], axis=-1), axis=0)),
                        np.cumsum(np.mean(xpl_v[:, 8, :], axis=0)) +
                        (np.std(np.cumsum(xpl_v[:, 8, :], axis=-1), axis=0)),
                        label=r'$\hat{\beta}^{\text{mm}}$',
                        color=c[4],
                        alpha=a)
        ax.fill_between(np.arange(T),
                        np.cumsum(np.mean(xpl_v[:, 0, :], axis=0)) -
                        (np.std(np.cumsum(xpl_v[:, 0, :], axis=-1), axis=0)),
                        np.cumsum(np.mean(xpl_v[:, 0, :], axis=0)) +
                        (np.std(np.cumsum(xpl_v[:, 0, :], axis=-1), axis=0)),
                        label=r'$\gamma_{0, t}$',
                        color=c[5],
                        alpha=a)
        ax.fill_between(np.arange(T),
                        np.cumsum(np.mean(xpl_v[:, 6, :], axis=0)) -
                        (np.std(np.cumsum(xpl_v[:, 6, :], axis=-1), axis=0)),
                        np.cumsum(np.mean(xpl_v[:, 6, :], axis=0)) +
                        (np.std(np.cumsum(xpl_v[:, 6, :], axis=-1), axis=0)),
                        label=r'$\hat{\beta}^{\text{OLS}}$',
                        color=c[6],
                        alpha=a)
    ax.grid(color='grey', axis='both', linestyle='-', linewidth=0.25,
            alpha=0.25)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\sum_{r=1}^t\Delta$Var$_r(\hat{\beta})$')
    plt.rcParams.update({'font.size': 8})
    leg3 = fig.legend(bbox_to_anchor=(1, 1.02),
                      handles=[l6, ], loc='outside left upper',
                      title=r'\textbf{Ground truth}')
    leg1 = fig.legend(bbox_to_anchor=(1, 0.3),
                      handles=[l2, l7, l5],
                      loc='outside left center',
                      title=r'\textbf{Zero-shot prediction}')
    leg2 = fig.legend(bbox_to_anchor=(1, 0.48),
                      handles=[l3, l4],
                      loc='outside left lower',
                      title=r'\textbf{Time adaptation estimators}')
    fig.add_artist(leg1)
    fig.add_artist(leg3)
    fig.add_artist(leg2)
    plt.show()

    # Introduction plot
    c = ['tab:blue', 'tab:orange', 'tab:cyan', 'tab:red',
         'tab:purple', 'tab:brown', 'tab:green', 'tab:gray',
         'tab:olive', 'tab:cyan']

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.2,
        'axes.unicode_minus': True,
        'axes.titlesize': 'medium'

    })
    a = 1
    mean = False
    fig, ax = plt.subplots(1, 2, figsize=(width, width*3.2/8),
                           layout='constrained')
    iter = 4
    axs0 = [None]*5
    bp = [None]*5
    wd = 0.1

    for i in range(5):
        if i == 0:
            axs0[0] = ax[0]
            axs0[0].set_xticks([])
        else:
            axs0[i] = ax[0].twinx()
            axs0[i].set_yticks([])
            axs0[i].set_xticks([])
    bp[0] = axs0[0].violinplot(np.mean(xpl_v_train[:, 0, :], axis=1),
                               showmeans=True,
                               showextrema=False,
                               widths=wd,
                               positions=[0])
    bp[1] = axs0[1].violinplot(np.mean(xpl_v_train[:, 1, :], axis=1),
                               showmeans=True,
                               showextrema=False,
                               widths=wd,
                               positions=[2/7])
    bp[2] = axs0[2].violinplot(np.mean(xpl_v_train[:, 2, :], axis=1),
                               showmeans=True,
                               showextrema=False,
                               widths=wd,
                               positions=[4/7])
    bp[3] = axs0[3].violinplot(np.mean(xpl_v_train[:, 4, :], axis=1),
                               showmeans=True,
                               showextrema=False,
                               widths=wd,
                               positions=[3/7])
    bp[4] = axs0[4].violinplot(np.mean(xpl_v_train[:, 5, :], axis=1),
                               showmeans=True,
                               showextrema=False,
                               widths=wd,
                               positions=[1/7])

    ord = [5, 1, 4, 6, 0]
    for i in range(5):
        for vp in bp[i]['bodies']:
            vp.set_color(c[ord[i]])
        bp[i]['cmeans'].set_color(c[ord[i]])
        axs0[i].set_ylim([0, 1.5])

    st = 1

    l2, = ax[1].plot(np.arange(T-st), np.cumsum(xpl_v[iter, 4, st:]),
                     label=r'$\hat{\beta}^{\text{inv}}$',
                     linestyle='--',
                     color=c[1],
                     alpha=a)
    l3, = ax[1].plot(np.arange(T-st), np.cumsum(xpl_v[iter, 3, st:]),
                     label=r'$\hat{\gamma}^{\text{ISD}}_t=$' +
                     r'$\hat{\beta}^{\text{inv}}+$' +
                     r'$\hat{\delta}^{\text{res}}_t$',
                     color=c[2],
                     alpha=a)
    l4, = ax[1].plot(np.arange(T-st), np.cumsum(xpl_v[iter, 5, st:]),
                     label=r'$\hat{\gamma}^{\text{OLS}}_t$',
                     color=c[3],
                     alpha=a)
    l5, = ax[1].plot(np.arange(T-st), np.cumsum(xpl_v[iter, 8, st:]),
                     label=r'$\hat{\beta}^{\text{mm}}$',
                     linestyle='--',
                     color=c[4],
                     alpha=a)
    l6, = ax[1].plot(np.arange(T-st), np.cumsum(xpl_v[iter, 0, st:]),
                     label=r'$\gamma_{0, t}$',
                     linestyle=':',
                     color=c[5],
                     alpha=a)
    l7, = ax[1].plot(np.arange(T-st), np.cumsum(xpl_v[iter, 6, st:]),
                     label=r'$\hat{\beta}^{\text{OLS}}$',
                     linestyle='--',
                     color=c[6],
                     alpha=a)
    temp, = axs0[0].plot(np.zeros(3), np.zeros(3), alpha=0, label='')
    ax[0].grid(color='grey', axis='y', linestyle='-.', linewidth=0.25,
               alpha=0.2)
    ax[0].set_ylabel(r'$\overline{\Delta \text{Var}}(\hat{\beta})$')
    ax[1].grid(color='grey', axis='both', linestyle='-.', linewidth=0.25,
               alpha=0.2)
    ax[1].set_xlabel('t')
    ax[1].set_ylabel(r'$\sum_{r=1}^t\Delta$Var$_r(\hat{\beta})$')
    plt.rcParams.update({'font.size': 12})
    ax[0].set_title('historical data')
    ax[1].set_title('adaptation data')
    plt.rcParams.update({'font.size': 8})
    ax[0].legend(handles=[bp[0]['cmeans'], bp[4]['cmeans'],
                          bp[1]['cmeans'], bp[3]['cmeans'], bp[2]['cmeans']],
                 labels=[r'$\gamma_{0,t}$'+''+r'$\text{ (ground truth)}$',
                         r'$\beta^{\text{inv}}$'+''+r'$\text{ (oracle)}$',
                         r'$\hat{\beta}^{\text{inv}}$',
                         r'$\hat{\beta}^{\text{OLS}}$',
                         r'$\hat{\beta}^{\text{mm}}$'],
                 loc='lower left', ncol=2)
    leg3 = fig.legend(bbox_to_anchor=(1, .92),
                      handles=[l6, ], loc='outside left upper',
                      title=r'\textbf{Ground truth}',)
    leg2 = fig.legend(bbox_to_anchor=(1, 0.47),
                      handles=[l3, l4],
                      loc='outside left lower',
                      title=r'\textbf{Time adaptation estimators}')
    leg1 = fig.legend(bbox_to_anchor=(1, 0.315),
                      handles=[l2, l7, l5],
                      loc='outside left center',
                      title=r'\textbf{Zero-shot estimators}')
    fig.add_artist(leg1)
    fig.add_artist(leg2)
    fig.add_artist(leg3)
    plt.show()


if __name__ == '__main__':
    main()
