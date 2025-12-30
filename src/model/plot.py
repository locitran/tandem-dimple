import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..features import TANDEM_FEATS, all_feat, dynamics_feat, structure_feat, sequence_feat

featSet = TANDEM_FEATS['v1.1']
error_params = {
    'capsize': 2,
    'capthick': 1.5,
    'elinewidth': 1
}
feat_colors = np.array([
    'lightcoral' if f in dynamics_feat else
    'lightgreen'   if f in structure_feat else
    'skyblue' if f in sequence_feat else
    'gray'  # default/fallback color if not in any group
    for f in featSet
])
feat_labels = np.array([
    'Dynamics' if f in dynamics_feat else
    'Structure'   if f in structure_feat else
    'Sequence&Chemical' if f in sequence_feat else
    'gray'  # default/fallback color if not in any group
    for f in featSet
])
feat_hatches = np.array([
    '//' if f in dynamics_feat else
    '..'   if f in structure_feat else
    '' if f in sequence_feat else
    ''  # default/fallback hatch if not in any group
    for f in featSet
])

featnames = np.array([all_feat[f] for f in featSet])
n_features = len(featnames)

def _plotSHAP_bar(
        phi,
        phi_sem,
        feature_order,
        title,
        axis_fontsize=12, 
        legend_fontsize=10,
    ):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(
        x=np.arange(n_features) + 0.5,
        height=phi[feature_order][::-1],
        yerr=phi_sem[feature_order][::-1],
        color=feat_colors[feature_order][::-1],
        label=feat_labels[feature_order][::-1],
        hatch=feat_hatches[feature_order][::-1],
        edgecolor='grey', capsize=2, width=0.7,
        error_kw=error_params
    )
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.set_ylabel('SHAP value', fontsize=axis_fontsize)
    ax.set_xticks(np.arange(n_features)+0.5)
    ax.set_xticklabels(featnames[feature_order][::-1], rotation=90, fontsize=legend_fontsize)
    ax.set_xlabel('Protein features', fontsize=axis_fontsize)
    ax.set_title(title, fontsize=axis_fontsize)

    handles, labels_ = ax.get_legend_handles_labels()
    unique_labels = set(labels_)
    unique_labels = ['Sequence&Chemical', 'Structure', 'Dynamics']
    handles = [handles[labels_.index(label)] for label in unique_labels]
    ax.legend(
        handles, unique_labels, fontsize=legend_fontsize, 
        title='Feature category', title_fontsize=str(legend_fontsize))
    return fig, ax

def plotSHAP_bar(struct_featImp, title, folder='.', filename=None,
        axis_fontsize=12, legend_fontsize=10, globalshap=True,
    ):  

    # globalSHAP: (nSAVs * n_models, n_features) 
    # individualSHAP: (n_models, n_features)
    featImp_arr    = np.vstack(struct_featImp) 
    if globalshap:
        phi = np.abs(featImp_arr)
    else:
        phi = featImp_arr

    phi_mean = phi.mean(axis=0) # 
    phi_sem  = phi.std(axis=0, ddof=1) / np.sqrt(phi.shape[0])
    feature_order  = np.argsort(np.abs(phi_mean))

    fig, ax = _plotSHAP_bar(
        phi_mean, phi_sem, feature_order, title,
        axis_fontsize=axis_fontsize, legend_fontsize=legend_fontsize
    )
    if filename:
        filepath = os.path.join(folder, filename)
        plt.savefig(
            filepath,
            dpi=300,               # higher resolution (300–600 for papers)
            bbox_inches='tight',   # avoid cutting off labels
        )
    plt.close(fig)

def plotLoss(
        folder, filename=None, patient=50,
        figsize=(15,9), grid_alpha=0.3, title_fontsize=12,
        x_fontsize=12, y_fontsize=12, legend_fontsize=10,
    ):

    row = 3 ; col = 5
    fig, ax = plt.subplots(row, col, figsize=figsize, sharey=True, dpi=300)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    history = [os.path.join(folder, f'TD_{i+1}') for i in range(3)]
    history = [[os.path.join(h, f'history_{i+1}.csv') for i in range(5)] for h in history]

    for td_idx in range(row): # td: tandem-dimple # 3-fold CV
        for tandem_idx in range(col): # 5-fold CV
            _ax = ax[td_idx, tandem_idx]
            h = history[td_idx][tandem_idx]
            df_full = pd.read_csv(h)

            # ----- Early-stop marker (last_epoch - 50; clamp to 0) -----
            stop_idx   = max(0, len(df_full) - patient)
            stop_epoch = df_full['epoch'].iloc[stop_idx]
            df_history = df_full[:-1]
            
            # ----- Curves -----
            _ax.plot(df_history['epoch'], df_history['train_loss'], 'b--', linewidth=2)
            _ax.plot(df_history['epoch'], df_history['val_loss'],   'r-',  linewidth=2)
            _ax.grid(axis='y', linestyle='--', alpha=grid_alpha)

            # ----- Early-stop “×” markers (both train & val at stop_epoch) -----
            _ax.scatter(stop_epoch, df_full['train_loss'].iloc[stop_idx], marker='x', s=30, color='black', zorder=5)
            _ax.scatter(stop_epoch, df_full['val_loss'].iloc[stop_idx], marker='x', s=30, color='black', linewidths=1.2, zorder=6)

            # (1) ticks inward for both axes
            _ax.tick_params(axis='both', which='both', direction='in')

            # Titles on top row
            if td_idx == 0:
                _ax.set_title(f'TANDEM {tandem_idx+1}', fontsize=title_fontsize)

            # Y-label on right side of last column
            if tandem_idx == col - 1:
                _ax.set_ylabel(f'TANDEM-DIMPLE {td_idx+1}', fontsize=y_fontsize, rotation=270, labelpad=18)
                _ax.yaxis.set_label_position("right")

            # (2) only show left y-ticks in each row
            if tandem_idx > 0:
                _ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # Per-row legend (adds early-stop entry too)
        ax[td_idx, 0].legend(['Train', 'Val', 'Early stop'], fontsize=legend_fontsize)
            
    # Expands the ylim by 10% (min, max)
    for td_idx in range(row):
        # get all axes in this row
        row_axes = [ax[td_idx, j] for j in range(col)]

        # collect current limits
        ymins = []
        ymaxs = []
        for _ax in row_axes:
            ymin, ymax = _ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)

        # common min/max for this row
        row_ymin = min(ymins)
        row_ymax = max(ymaxs)

        # expand range by 10%
        yrange = row_ymax - row_ymin
        pad = 0.05 * yrange

        new_ymin = row_ymin - pad
        new_ymax = row_ymax + pad
        # apply back to all axes in this row
        for _ax in row_axes:
            _ax.set_ylim(new_ymin, new_ymax)

    # fig.set_facecolor('none')  # or 'none' for transparent
    # plt.setp(ax, facecolor='none')  # Set axes background to transparent
    fig.supxlabel('Number of Epochs', fontsize=x_fontsize, y=0.06)
    fig.supylabel('Loss',  fontsize=y_fontsize, x=0.09)

    if filename:
        filepath = os.path.join(folder, filename)
        plt.savefig(
            filepath,
            dpi=300,               # higher resolution (300–600 for papers)
            bbox_inches='tight',   # avoid cutting off labels
        )
    plt.close()
