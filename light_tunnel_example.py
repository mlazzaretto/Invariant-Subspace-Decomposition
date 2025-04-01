from __future__ import division

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from isd_i import ISD_I
from causalchamber.datasets import Dataset


def xpl_var(X, Y, beta):
    # compute explained varinace by beta (under zero mean ass.)
    # X, Y: observed covariates and response
    # beta: linear parameter
    if len(X.shape) == 1:
        n = 1
        X = X.reshape(1, -1)
        Y = Y.reshape(1, -1)
        return (1/n)*(2*Y.T@X@beta - beta.T@X.T@X@beta)
    else:
        n = X.shape[0]
        cov_xy = np.cov(Y, Y-X@beta, rowvar=False)
    return cov_xy[0, 0]-cov_xy[1, 1]


def main():
    # Download the dataset and store it in the current directory
    dataset = Dataset('lt_walks_v1', root='./', download=True)

    # Load the data from the experiment
    experiment = dataset.get_experiment(name='smooth_polarizers')
    df = experiment.as_pandas_dataframe()
    df['rgb'] = df.red + df.green + df.blue
    df['c2pol_1'] = np.cos(np.deg2rad(df.pol_1))**2

    dft = df.sort_values(by='c2pol_1', ascending=False)

    n = 7000
    X_hist = np.array(dft[['rgb', 'l_31', 'l_32']][:n])
    Y_hist = np.array(dft[['ir_3']][:n])
    p = X_hist.shape[1]
    X_test = np.array(dft[['rgb', 'l_31', 'l_32']][n+1000:n+2000])
    Y_test = np.array(dft[['ir_3']][n+1000:n+2000])
    n_test = X_test.shape[0]
    ws_hist = int(n/15)
    nw_hist = 15

    # ######## Data plotting ###########################

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        'axes.labelsize': 22,
        'font.size': 22,
        'legend.fontsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'lines.linewidth': 1.2,
        'axes.unicode_minus': True,
    })
    sk = 3
    start = n+1000
    stop = n+2000
    fig, ax = plt.subplots(1, 5, figsize=(17, 3))
    l0 = ax[0].scatter(dft[:n:sk].rgb, dft[:n:sk].ir_3,
                       c=np.arange(stop=n, step=sk), marker='.',
                       label='historical data')
    l1 = ax[0].scatter(dft[start:stop:sk].rgb, dft[start:stop:sk].ir_3,
                       marker='.', c='tab:orange', label='test data')
    ax[0].set_ylabel(r"$Y$")
    ax[0].set_xlabel(r"$X_{\text{RGB}}$")
    ax[1].scatter(dft[:n:sk].l_31, dft[:n:sk].ir_3, marker='.',
                  c=np.arange(stop=n, step=sk))
    ax[1].scatter(dft[start:stop:sk].l_31, dft[start:stop:sk].ir_3,
                  marker='.', c='tab:orange')
    ax[1].set_ylabel(r"$Y$")
    ax[1].set_xlabel(r"$X_{L_{31}}$")
    ax[1].set_xlim([55, 260])
    ax[2].scatter(dft[:n:sk].l_32, dft[:n:sk].ir_3, marker='.',
                  c=np.arange(stop=n, step=sk))
    ax[2].scatter(dft[start:stop:sk].l_32, dft[start:stop:sk].ir_3,
                  marker='.', c='tab:orange')
    ax[2].set_ylabel(r"$Y$")
    ax[2].set_xlabel(r"$X_{L_{32}}$")
    ax[2].set_xlim([55, 260])
    ax[3].scatter(np.arange(stop=n, step=sk), dft[:n:sk].ir_3, marker='.',
                  c=np.arange(stop=n, step=sk))
    ax[3].scatter(np.arange(start=n, stop=n+n_test, step=sk),
                  dft[start:stop:sk].ir_3, marker='.', c='tab:orange')
    ax[3].set_ylabel(r"$Y$")
    ax[3].set_xlabel(r"$t$")
    l3 = ax[4].scatter(np.arange(stop=n, step=sk), dft[:n:sk].c2pol_1,
                       marker='.', c=np.arange(stop=n, step=sk))
    ax[4].scatter(np.arange(start=n, stop=n+n_test, step=sk),
                  dft[start:stop:sk].c2pol_1, marker='.', c='tab:orange')
    ax[4].set_ylabel(r"$\text{cos}^2(\theta)$")
    ax[4].set_xlabel(r"$t$")
    for aa in range(5):
        ax[aa].grid(color='grey', axis='both', linestyle='-',
                    linewidth=0.25, alpha=0.25)
    cbar_ax = fig.add_axes([0.855, 0.01, 0.13, 0.05])
    fig.colorbar(l3, cax=cbar_ax, pad=0, location='bottom')
    cbar_ax.text(-900, -1.8, r'$t$')
    fig.legend(handles=[l0, l1],
               bbox_to_anchor=(0.5, 0.0),
               loc='outside center', ncol=2)
    plt.tight_layout()
    plt.show()

    sk = 1
    start = n+1000
    stop = n+2000
    fig, ax = plt.subplots(1, 5, figsize=(16.5, 3))
    ax[0].scatter(dft['rgb'][start:stop], dft['ir_3'][start:stop],
                  c='tab:orange', marker='.')
    ax[0].set_ylabel(r"$Y$")
    ax[0].set_xlabel(r"$X_{\text{RGB}}$")
    ax[1].scatter(dft['l_31'][start:stop], dft['ir_3'][start:stop],
                  c='tab:orange', marker='.')
    ax[1].set_ylabel(r"$Y$")
    ax[1].set_xlabel(r"$X_{L_{31}}$")
    ax[2].scatter(dft['l_32'][start:stop], dft['ir_3'][start:stop],
                  c='tab:orange', marker='.')
    ax[2].set_ylabel(r"$Y$")
    ax[2].set_xlabel(r"$X_{L_{32}}$")
    ax[3].scatter(np.arange(start=n, stop=n+n_test, step=sk),
                  dft[start:stop:sk].ir_3, marker='.', c='tab:orange')
    ax[3].set_ylabel(r"$Y$")
    ax[3].set_xlabel(r"$t$")
    ax[4].scatter(np.arange(start=n, stop=n+n_test, step=sk),
                  dft[start:stop:sk].c2pol_1, marker='.', c='tab:orange')
    ax[4].set_ylabel(r"$\text{cos}^2(\theta)$")
    ax[4].set_xlabel(r"$t$")
    for aa in range(5):
        ax[aa].grid(color='grey', axis='both', linestyle='-',
                    linewidth=0.25, alpha=0.25)
    plt.tight_layout()
    plt.show()

    ################################################################

    # ISD
    est = ISD_I(X_hist, Y_hist, [ws_hist]*nw_hist)

    # Invariant component estimation
    beta_inv, beta_icpt, U, blocks, c_blocks = \
        est.invariant_estimator(k_fold=10)
    # OLS on historical data
    beta_ols = sm.OLS(Y_hist, sm.add_constant(X_hist)).fit()\
        .params.reshape(-1, 1)
    # Maximin on historical data
    beta_mm = est.magging_estimator()

    # Explained variance in historical data
    n_est = 3
    n_c = 5
    xpl_v_hist = np.zeros((n_est+1, n-n_c))
    xpl_v_test = np.zeros((n_est+1, n_test-n_c))

    beta = np.zeros((n_est, p))
    beta[0, :] = beta_inv.squeeze()
    beta[1, :] = beta_ols[1:].squeeze()
    beta[2, :] = beta_mm.squeeze()

    for j in range(n_est):
        b = beta[j, :].reshape(-1, 1)
        for t in range(n-n_c):
            xpl_v_hist[j, t] = xpl_var(X_hist[t:t+n_c, :],
                                       Y_hist[t:t+n_c, :], b)
        for t in range(n_test-n_c):
            xpl_v_test[j, t] = xpl_var(X_test[t:t+n_c, :],
                                       Y_test[t:t+n_c, :], b)

    for t in range(n-n_c):
        xpl_v_hist[-1, t] = np.cov(Y_hist[t:t+n_c, :], rowvar=False)
    for t in range(n_test-n_c):
        xpl_v_test[-1, t] = np.cov(Y_test[t:t+n_c, :], rowvar=False)

    # Fraction of explained variance
    xvf_hist = np.zeros((n_est,))
    xvf_test = np.zeros((n_est,))
    for j in range(n_est):
        xvf_hist[j] = np.mean(xpl_v_hist[j, :])/np.mean(xpl_v_hist[-1, :])
        xvf_test[j] = np.mean(xpl_v_test[j, :])/np.mean(xpl_v_test[-1, :])

    print(r'Fraction of explained variance by' +
          r'[beta^inv, beta^OLS, beta^mm]')
    print('Historical data:', xvf_hist)
    print('Test data', xvf_test)

    # Explained variance in adaptation step
    n_est_ad = 2
    ws_ad = p*2+2
    n_val = 5
    T = n_test-ws_ad-n_val
    xpl_v = np.zeros((n_est+n_est_ad+1, T))

    for t in range(T):
        X_ad = X_test[t:t+ws_ad, :]
        Y_ad = Y_test[t:t+ws_ad]
        X_val = X_test[t+ws_ad:t+ws_ad+n_val, :]
        Y_val = Y_test[t+ws_ad:t+ws_ad+n_val]

        gamma_ols_t = sm.OLS(Y_ad,
                             sm.add_constant(X_ad)).fit().params.reshape(-1, 1)
        # Time adaptation
        if c_blocks.any():
            delta_res_t, icpt_t = est.adapt(X_ad, Y_ad)
        else:
            delta_res_t = gamma_ols_t
        gamma_isd_t = beta_inv + delta_res_t

        xpl_v[0, t] = xpl_var(X_val, Y_val, gamma_isd_t)            # gamma ISD
        xpl_v[1, t] = xpl_var(X_val, Y_val, gamma_ols_t[1:, :])     # gamma OLS
        xpl_v[2, t] = xpl_var(X_val, Y_val, beta_inv)               # beta inv
        xpl_v[3, t] = xpl_var(X_val, Y_val, beta_ols[1:, :])        # beta OLS
        xpl_v[4, t] = xpl_var(X_val, Y_val, beta_mm)                # beta mm
        xpl_v[5, t] = np.cov(Y_val, rowvar=False)

    # Cumulative variance plot
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
    fig, ax = plt.subplots(1, 1, figsize=(0.5*width, 0.5*width*2.25/3),
                           layout='constrained')
    a = 1
    l2, = ax.plot(np.arange(n, T+n),
                  np.cumsum(xpl_v[2, :]),
                  label=r'$\hat{\beta}^{\text{inv}}$',
                  linestyle='--',
                  color=c[1],
                  alpha=a)
    l3, = ax.plot(np.arange(n, T+n),
                  np.cumsum(xpl_v[0, :]),
                  label=r'$\hat{\gamma}^{\text{ISD}}_t=$' +
                  r'$\hat{\beta}^{\text{inv}}+$' +
                  r'$\hat{\delta}^{\text{res}}_t$',
                  color=c[2],
                  alpha=a)
    l4, = ax.plot(np.arange(n, T+n),
                  np.cumsum(xpl_v[1, :]),
                  label=r'$\hat{\gamma}^{\text{OLS}}_t$',
                  color=c[3],
                  alpha=a)
    l6, = ax.plot(np.arange(n, T+n),
                  np.cumsum(xpl_v[5, :]),
                  label=r'$\sum_{r=1}^t \text{Var}(Y_r)$',
                  linestyle=':',
                  color=c[5],
                  alpha=a)
    l7, = ax.plot(np.arange(n, n+100),
                  np.cumsum(xpl_v[3, :100]),
                  label=r'$\hat{\beta}^{\text{OLS}}$',
                  linestyle='--',
                  color=c[6],
                  alpha=a)

    ax.grid(color='grey', axis='both', linestyle='-', linewidth=0.25,
            alpha=0.25)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\sum_{r=1}^t\Delta$Var$_r(\hat{\beta})$')
    ax.set_yscale('linear')
    plt.rcParams.update({'font.size': 8})
    leg3 = fig.legend(bbox_to_anchor=(1, 0.95),
                      handles=[l6, ], loc='outside left upper',
                      title=r'\textbf{Response variance}')
    leg1 = fig.legend(bbox_to_anchor=(1, 0.32),
                      handles=[l2, l7],
                      loc='outside left center',
                      title=r'\textbf{Zero-shot prediction}')
    leg2 = fig.legend(bbox_to_anchor=(1, 0.45),
                      handles=[l3, l4],
                      loc='outside left lower',
                      title=r'\textbf{Time adaptation estimators}')
    fig.add_artist(leg1)
    fig.add_artist(leg3)
    fig.add_artist(leg2)
    plt.show()


if __name__ == '__main__':
    main()

# %%
