import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import confusion_matrix

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

def star_from_p(p, alpha=0.05):
    try:
        return "*" if (p is not None) and np.isfinite(p) and (p < alpha) else ""
    except Exception:
        return ""

def annotate_sig(ax, x1, x2, y, h, text, lw=0.5, fs=8, clip_on=False, text_offset=0.3):
    """
    Draw a significance bracket from x1 to x2 at height y with vertical height h
    and put 'text' (e.g., '*') above it.
    """
    if not text:
        return
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c="k", clip_on=clip_on)
    ax.text((x1 + x2) / 2, y + h + text_offset, text, ha="center", va="bottom", fontsize=fs)

def _ensure_ylim(ax, needed):
    y0, y1 = ax.get_ylim()
    if needed > y1:
        ax.set_ylim(y0, needed)

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
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title='Confusion matrix'):
    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='2d', cmap='Blues')#, normalize='true')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title, fontsize=20)
    plt.rcParams.update({'font.size': 18})
    plt.show()

def styled_legend(ax, handles, ncol=1, loc='upper right'):
    legend = ax.legend(
        handles=handles,
        loc=loc, ncol=ncol, fontsize=8, title_fontsize=9, frameon=True,
        borderpad=0.3, labelspacing=0.5, handletextpad=0.5, columnspacing=0.5
    )
    frame = legend.get_frame()
    frame.set_facecolor('none')
    frame.set_edgecolor('grey')
    return legend

def pl_gene_general_performance(
    tandem,
    rhapsodyDNN,
    rhapsody_R20000_metrics,
    alphamissense_R20000_metrics,
    rhapsody_GJB2_metrics,
    alphamissense_GJB2_metrics,
    rhapsody_RYR1_metrics,
    alphamissense_RYR1_metrics,
    param,
    txt_abv_bar,
    alpha=0.05,
    show_bar_values=False,
    show_sigstars=True,
    save_path=None,
):
    def add_sigstars_for_panel(
        ax, means, sems,
        p_vs_rhapsodyDNN, p_vs_rhapsody, p_vs_alpha,
        alpha=0.05, width=0.2, base_pad=1.0, level_gap=1.0, bar_h=0.3,
    ):
        """
        means: [RhapsodyDNN, Tandem, Rhapsody, AlphaMissense]
        sems : same order
        """
        means = [np.asarray(m) for m in means]
        sems  = [np.asarray(s) for s in sems]

        for j in range(len(means[0])):
            # x positions (shift Alpha left if we skip Rhapsody)
            x_rdd = j + 0*width
            x_tan = j + 1*width
            x_r = j + 2*width
            x_a = j + 3*width   

            # heights (percent scale)
            y_rdd = (means[0][j] + sems[0][j]) * 100.0
            y_tan = (means[1][j] + sems[1][j]) * 100.0
            y_r = (means[2][j] + sems[2][j]) * 100.0
            y_a   = (means[3][j] + sems[3][j]) * 100.0

            # baseline for brackets: ignore Rhapsody height if skipping
            candidates = y_rdd, y_tan, y_r, y_a
            y_base = max(candidates) + base_pad

            level = 0
            # 1) Tandem vs RhapsodyDNN
            s1 = star_from_p(p_vs_rhapsodyDNN[j], alpha=alpha)
            annotate_sig(ax, x_tan, x_rdd, y_base + level*level_gap, bar_h, s1);  level += (1 if s1 else 0)

            # 2) Tandem vs Rhapsody (only if included)
            s2 = star_from_p(p_vs_rhapsody[j], alpha=alpha)
            annotate_sig(ax, x_tan, x_r,   y_base + level*level_gap, bar_h, s2);  level += (1 if s2 else 0)

            # 3) Tandem vs AlphaMissense
            s3 = star_from_p(p_vs_alpha[j], alpha=alpha)
            annotate_sig(ax, x_tan, x_a,   y_base + level*level_gap, bar_h, s3);  level += (1 if s3 else 0)

            # ensure visibility
            ymax_needed = y_base + max(level-1, 0)*level_gap + bar_h + 2.0
            _ensure_ylim(ax, ymax_needed)

    # R20000
    tandem_r20000_test = [tandem['test_accuracy'].values, tandem['test_auc'].values, tandem['test_precision'].values, tandem['test_recall'].values, tandem['test_f1'].values]
    rhapsodyDNN_20000_test = [rhapsodyDNN['test_accuracy'].values, rhapsodyDNN['test_auc'].values, rhapsodyDNN['test_precision'].values, rhapsodyDNN['test_recall'].values, rhapsodyDNN['test_f1'].values]
    rhapsody_20000_test = [rhapsody_R20000_metrics.get(k) for k in ['accuracy', 'auc', 'precision', 'recall', 'f1_score']] # 
    alphamissense_20000_test = [alphamissense_R20000_metrics.get(k) for k in ['accuracy', 'auc', 'precision', 'recall', 'f1_score']] # 
    # GJB2
    tandem_GJB2 = [tandem['GJB2_notnan_accuracy'].values, tandem['GJB2_notnan_auc'].values, tandem['GJB2_notnan_precision'].values, tandem['GJB2_notnan_recall'].values, tandem['GJB2_notnan_f1'].values]
    rhapsodyDNN_GJB2 = [rhapsodyDNN['GJB2_notnan_accuracy'].values, rhapsodyDNN['GJB2_notnan_auc'].values, rhapsodyDNN['GJB2_notnan_precision'].values, rhapsodyDNN['GJB2_notnan_recall'].values, rhapsodyDNN['GJB2_notnan_f1'].values]
    rhapsody_GJB2 = [rhapsody_GJB2_metrics.get(k) for k in ['accuracy', 'auc', 'precision', 'recall', 'f1_score']] # 
    alphamissense_GJB2 = [alphamissense_GJB2_metrics.get(k) for k in ['accuracy', 'auc', 'precision', 'recall', 'f1_score']] # 
    # RYR1
    tandem_RYR1 = [tandem['RYR1_notnan_accuracy'].values, tandem['RYR1_notnan_auc'].values, tandem['RYR1_notnan_precision'].values, tandem['RYR1_notnan_recall'].values, tandem['RYR1_notnan_f1'].values]
    rhapsodyDNN_RYR1 = [rhapsodyDNN['RYR1_notnan_accuracy'].values, rhapsodyDNN['RYR1_notnan_auc'].values, rhapsodyDNN['RYR1_notnan_precision'].values, rhapsodyDNN['RYR1_notnan_recall'].values, rhapsodyDNN['RYR1_notnan_f1'].values]
    rhapsody_RYR1 = [rhapsody_RYR1_metrics.get(k) for k in ['accuracy', 'auc', 'precision', 'recall', 'f1_score']] # 
    alphamissense_RYR1 = [alphamissense_RYR1_metrics.get(k) for k in ['accuracy', 'auc', 'precision', 'recall', 'f1_score']] # 

    # Statistic test
    _, tandem_rhapsodyDNN_r20000    = stats.ttest_ind(tandem_r20000_test, rhapsodyDNN_20000_test, axis=1)
    _, tandem_rhapsody_r20000       = stats.ttest_1samp(tandem_r20000_test, [[x] for x in rhapsody_20000_test], axis=1)
    _, tandem_alphamissense_r20000  = stats.ttest_1samp(tandem_r20000_test, [[x] for x in alphamissense_20000_test], axis=1)

    _, tandem_rhapsodyDNN_GJB2    = stats.ttest_ind(tandem_GJB2, rhapsodyDNN_GJB2, axis=1)
    _, tandem_rhapsody_GJB2       = stats.ttest_1samp(tandem_GJB2, [[x] for x in rhapsody_GJB2], axis=1)
    _, tandem_alphamissense_GJB2  = stats.ttest_1samp(tandem_GJB2, [[x] for x in alphamissense_GJB2], axis=1)

    _, tandem_rhapsodyDNN_RYR1    = stats.ttest_ind(tandem_RYR1, rhapsodyDNN_RYR1, axis=1)
    _, tandem_rhapsody_RYR1       = stats.ttest_1samp(tandem_RYR1, [[x] for x in rhapsody_RYR1], axis=1)
    _, tandem_alphamissense_RYR1  = stats.ttest_1samp(tandem_RYR1, [[x] for x in alphamissense_RYR1], axis=1)
    ################

    data = [rhapsodyDNN_20000_test, tandem_r20000_test, rhapsody_20000_test, alphamissense_20000_test]
    data_mean = []
    for d in data:
        data_mean.append(np.array([np.mean(d[i]) for i in range(len(d))]))
    data_sem = [
        np.array([np.std(rhapsodyDNN_20000_test[i], ddof=1) / np.sqrt(len(rhapsodyDNN_20000_test[i])) for i in range(len(rhapsodyDNN_20000_test))]),
        np.array([np.std(tandem_r20000_test[i], ddof=1) / np.sqrt(len(tandem_r20000_test[i])) for i in range(len(tandem_r20000_test))]),
        np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])
    ]

    GJB2 = [rhapsodyDNN_GJB2, tandem_GJB2, rhapsody_GJB2, alphamissense_GJB2]
    GJB2_mean = []
    for d in GJB2:
        GJB2_mean.append(np.array([np.mean(d[i]) for i in range(len(d))]))
    GJB2_sem = [
        np.array([np.std(rhapsodyDNN_GJB2[i], ddof=1) / np.sqrt(len(rhapsodyDNN_GJB2[i])) for i in range(len(rhapsodyDNN_GJB2))]),
        np.array([np.std(tandem_GJB2[i], ddof=1) / np.sqrt(len(tandem_GJB2[i])) for i in range(len(tandem_GJB2))]),
        np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])
    ]
    RYR1 = [rhapsodyDNN_RYR1, tandem_RYR1, rhapsody_RYR1, alphamissense_RYR1]
    RYR1_mean = []
    for d in RYR1:
        RYR1_mean.append(np.array([np.mean(d[i]) for i in range(len(d))]))
    RYR1_sem = [
        np.array([np.std(rhapsodyDNN_RYR1[i], ddof=1) / np.sqrt(len(rhapsodyDNN_RYR1[i])) for i in range(len(rhapsodyDNN_RYR1))]),
        np.array([np.std(tandem_RYR1[i], ddof=1) / np.sqrt(len(tandem_RYR1[i])) for i in range(len(tandem_RYR1))]),
        np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])
    ]

    # Gene-general models
    exp_labels = [r'Rhapsody$_{DNN}$', r'TANDEM', r'Rhapsody', r'AlphaMissense'] # model name
    x_labels = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1'] # x-axis
    data_hatches = ['', '', '//', '\\\\'] # bar hatches
    data_colors = ['orange', 'lightblue', 'orange', 'lightgreen'] # bar colors

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharey=True, dpi=300,)
    plt.subplots_adjust(wspace=0, hspace=0) # Space the subplots
    ax_r20000 = ax[0]
    ax_GJB2 = ax[1]
    ax_RYR1 = ax[2]

    # Put number on the top of the bar
    for i in range(len(data_mean)):

        ax_r20000.bar(np.arange(len(data_mean[i])) + i * 0.2, data_mean[i]*100, yerr=data_sem[i]*100, 
            label=exp_labels[i], **param, hatch=data_hatches[i], error_kw=error_params, color=data_colors[i])
        ax_GJB2.bar(np.arange(len(GJB2_mean[i])) + i * 0.2, GJB2_mean[i]*100, yerr=GJB2_sem[i]*100, 
            label=exp_labels[i], **param, hatch=data_hatches[i], error_kw=error_params, color=data_colors[i])
        ax_RYR1.bar(np.arange(len(RYR1_mean[i])) + i * 0.2, RYR1_mean[i]*100, yerr=RYR1_sem[i]*100, 
            label=exp_labels[i], **param, hatch=data_hatches[i], error_kw=error_params, color=data_colors[i])

        if show_bar_values:
            # Add text on top of the bars
            for j in range(len(data_mean[i])):
                ax_r20000.text(j + i * 0.2, data_mean[i][j]*100 + data_sem[i][j]*100 + txt_abv_bar, f'{data_mean[i][j]*100:.1f}', ha='center', va='bottom', fontsize=5)
                ax_GJB2.text(j + i * 0.2, GJB2_mean[i][j]*100 + GJB2_sem[i][j]*100 + txt_abv_bar, f'{GJB2_mean[i][j]*100:.1f}', ha='center', va='bottom', fontsize=5)
                ax_RYR1.text(j + i * 0.2, RYR1_mean[i][j]*100 + RYR1_sem[i][j]*100 + txt_abv_bar, f'{RYR1_mean[i][j]*100:.1f}', ha='center', va='bottom', fontsize=5)

    for x in ax:
        x.grid(axis='y', linestyle='--', linewidth=0.1)
        x.set_xticks(np.arange(len(x_labels)) + 0.3)
        x.set_xticklabels(x_labels, fontsize=12)
        x.set_xlim(-0.3, len(x_labels)-0.1)
        x.set_ylim(0, 120)
        # x.set_ylim(0, 111)
        
    for i, title in zip([0, 1, 2], ['R20000$_{test}$', 'GJB2$_{knw}$', 'RYR1$_{knw}$']):
        ax[i].set_title(title, fontsize=14)
    ax_GJB2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    ax_RYR1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    legend_batch = ax_GJB2.legend(loc='upper center', fontsize=9, frameon=True, ncol=2, borderpad=0.3, labelspacing=0.5, handletextpad=0.5, columnspacing=0.5,)
    legend_batch.get_frame().set_facecolor('none')
    legend_batch.get_frame().set_edgecolor('grey')

    ax_r20000.set_ylabel('Performance (%)', fontsize=14)
    ax_r20000.set_yticks(np.arange(0, 105, 5))
    ax_r20000.set_yticklabels([str(y) if y % 10 == 0 else "" for y in np.arange(0, 105, 5)], fontsize=10)


    # --- add significance stars (alpha = 0.05) ---
    bar_width = 0.2  # matches your i * 0.2 offsets

    if show_sigstars:
        # R20000: [RhapsodyDNN, Tandem, Rhapsody, AlphaMissense]
        add_sigstars_for_panel(
            ax_r20000,
            means=[data_mean[0], data_mean[1], data_mean[2], data_mean[3]],
            sems =[data_sem[0],  data_sem[1],  data_sem[2],  data_sem[3]],
            p_vs_rhapsodyDNN=tandem_rhapsodyDNN_r20000,
            p_vs_rhapsody   =tandem_rhapsody_r20000,
            p_vs_alpha      =tandem_alphamissense_r20000,
            alpha=alpha,
            width=bar_width,
        )

        # GJB2
        add_sigstars_for_panel(
            ax_GJB2,
            means=[GJB2_mean[0], GJB2_mean[1], GJB2_mean[2], GJB2_mean[3]],
            sems =[GJB2_sem[0],  GJB2_sem[1],  GJB2_sem[2],  GJB2_sem[3]],
            p_vs_rhapsodyDNN=tandem_rhapsodyDNN_GJB2,
            p_vs_rhapsody   =tandem_rhapsody_GJB2,
            p_vs_alpha      =tandem_alphamissense_GJB2,
            alpha=alpha,
            width=bar_width,
        )

        # RYR1
        add_sigstars_for_panel(
            ax_RYR1,
            means=[RYR1_mean[0], RYR1_mean[1], RYR1_mean[2], RYR1_mean[3]],
            sems =[RYR1_sem[0],  RYR1_sem[1],  RYR1_sem[2],  RYR1_sem[3]],
            p_vs_rhapsodyDNN=tandem_rhapsodyDNN_RYR1,
            p_vs_rhapsody   =tandem_rhapsody_RYR1,
            p_vs_alpha      =tandem_alphamissense_RYR1,
            alpha=alpha,
            width=bar_width,
        )

    # Remove background of the plot, transparent
    # fig.set_facecolor('none')  # or 'none' for transparent
    # plt.setp(ax, facecolor='none')  # Set axes background to transparent
    # plt.savefig('/mnt/nas_1/YangLab/loci/tandem/models/5metrics_disease_general_models.png', bbox_inches='tight', transparent=True, dpi=300)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def pl_gene_specific_performance(
    tf_gjb2_after,
    tf_ryr1_after,
    tf_gjb2_before,
    tf_ryr1_before,
    rhapsody_R20000_metrics,
    alphamissense_R20000_metrics,
    rhapsody_GJB2_test_metrics,
    alphamissense_GJB2_test_metrics,
    rhapsody_RYR1_test_metrics,
    alphamissense_RYR1_test_metrics,
    txt_abv_bar,
    alpha=0.05,
    show_bar_values=False,
    show_sigstars=True,
    save_path=None,
):
    # --- main function ---
    def add_transfer_sigstars(ax, means, sems, pvals, idx_transfer, compare_indices,
                            bar_width, alpha=0.05, base_pad=1.0, level_gap=1.0, bar_h=0.3):
        means = np.asarray(means)
        sems  = np.asarray(sems)
        pvals = np.asarray(pvals)

        n_metrics, n_bars = means.shape
        for m in range(n_metrics):
            # x positions of bars for this metric
            x_positions = np.array([m + j*bar_width for j in range(n_bars)])

            # height reference: top of tallest bar (mean+sem), same scale as plot (e.g., %)
            heights = means[m] + sems[m]
            y_base = float(np.max(heights)) + base_pad

            level = 0
            for col_k, j_idx in enumerate(compare_indices):
                s = star_from_p(pvals[m, col_k])
                if s:
                    annotate_sig(ax,
                                x_positions[idx_transfer],
                                x_positions[j_idx],
                                y_base + level*level_gap,
                                h=bar_h,
                                text=s,
                                lw=0.8,
                                fs=8,
                                clip_on=False)
                    level += 1

            # ensure space for brackets if any were drawn
            if level > 0:
                need = y_base + (level-1)*level_gap + bar_h + 2.0
                _ensure_ylim(ax, need)


    # # Precision, Recall, F1
    x_labels = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']

    r20000_test = [
        [tf_gjb2_after['R20000_test_accuracy'].values,  tf_ryr1_after['R20000_test_accuracy'].values,   tf_gjb2_before['R20000_test_accuracy'].values, rhapsody_R20000_metrics['accuracy'], alphamissense_R20000_metrics['accuracy']],
        [tf_gjb2_after['R20000_test_auc'].values,       tf_ryr1_after['R20000_test_auc'].values,        tf_gjb2_before['R20000_test_auc'].values, rhapsody_R20000_metrics['auc'], alphamissense_R20000_metrics['auc']],
        [tf_gjb2_after['R20000_test_precision'].values, tf_ryr1_after['R20000_test_precision'].values,  tf_gjb2_before['R20000_test_precision'].values, rhapsody_R20000_metrics['precision'], alphamissense_R20000_metrics['precision']],
        [tf_gjb2_after['R20000_test_recall'].values,    tf_ryr1_after['R20000_test_recall'].values,     tf_gjb2_before['R20000_test_recall'].values, rhapsody_R20000_metrics['recall'], alphamissense_R20000_metrics['recall']],
        [tf_gjb2_after['R20000_test_f1'].values,        tf_ryr1_after['R20000_test_f1'].values,         tf_gjb2_before['R20000_test_f1'].values, rhapsody_R20000_metrics['f1_score'], alphamissense_R20000_metrics['f1_score']]
    ]
    gjb2_test = [
        [tf_gjb2_after['test_accuracy'].values,  tf_gjb2_before['test_accuracy'].values,    rhapsody_GJB2_test_metrics['accuracy'], alphamissense_GJB2_test_metrics['accuracy']],
        [tf_gjb2_after['test_auc'].values,       tf_gjb2_before['test_auc'].values,         rhapsody_GJB2_test_metrics['auc'],      alphamissense_GJB2_test_metrics['auc']],
        [tf_gjb2_after['test_precision'].values, tf_gjb2_before['test_precision'].values,   rhapsody_GJB2_test_metrics['precision'], alphamissense_GJB2_test_metrics['precision']],
        [tf_gjb2_after['test_recall'].values,    tf_gjb2_before['test_recall'].values,      rhapsody_GJB2_test_metrics['recall'],   alphamissense_GJB2_test_metrics['recall']],
        [tf_gjb2_after['test_f1'].values,        tf_gjb2_before['test_f1'].values,          rhapsody_GJB2_test_metrics['f1_score'], alphamissense_GJB2_test_metrics['f1_score']],
    ]
    ryr1_test = [
        [tf_ryr1_after['test_accuracy'].values,     tf_ryr1_before['test_accuracy'].values, rhapsody_RYR1_test_metrics['accuracy'], alphamissense_RYR1_test_metrics['accuracy']],
        [tf_ryr1_after['test_auc'].values,          tf_ryr1_before['test_auc'].values,      rhapsody_RYR1_test_metrics['auc'], alphamissense_RYR1_test_metrics['auc']],
        [tf_ryr1_after['test_precision'].values,    tf_ryr1_before['test_precision'].values, rhapsody_RYR1_test_metrics['precision'], alphamissense_RYR1_test_metrics['precision']],
        [tf_ryr1_after['test_recall'].values,       tf_ryr1_before['test_recall'].values,   rhapsody_RYR1_test_metrics['recall'], alphamissense_RYR1_test_metrics['recall']],
        [tf_ryr1_after['test_f1'].values,           tf_ryr1_before['test_f1'].values,       rhapsody_RYR1_test_metrics['f1_score'], alphamissense_RYR1_test_metrics['f1_score']],
    ]

    pvalues_gjb2 = []
    for tf_gjb2, fd_gjb2, rhd_gjb2, alm_gjb2 in gjb2_test:
        _, tf_fd_gjb2 = stats.ttest_ind(tf_gjb2, fd_gjb2)
        _, tf_rhd_gjb2 = stats.ttest_1samp(tf_gjb2, rhd_gjb2)
        _, tf_alm_gjb2 = stats.ttest_1samp(tf_gjb2, alm_gjb2)
        pvalues_gjb2.append([tf_fd_gjb2, tf_rhd_gjb2, tf_alm_gjb2])

    pvalues_ryr1 = []
    for tf_ryr1, fd_ryr1, rhd_ryr1, alm_ryr1 in ryr1_test:
        _, tf_fd_ryr1 = stats.ttest_ind(tf_ryr1, fd_ryr1)
        _, tf_rhd_ryr1 = stats.ttest_1samp(tf_ryr1, rhd_ryr1)
        _, tf_alm_ryr1 = stats.ttest_1samp(tf_ryr1, alm_ryr1)
        pvalues_ryr1.append([tf_fd_ryr1, tf_rhd_ryr1, tf_alm_ryr1])

    pvalues_gjb2 = np.array(pvalues_gjb2)  # shape: (5 metrics, 3 comparisons)
    pvalues_ryr1 = np.array(pvalues_ryr1)  # shape: (5 metrics, 3 comparisons)

    r20000_test_mean = np.array([
        [100*np.mean(ele) for ele in metric] for metric in r20000_test
    ])
    r20000_test_sem = np.array([
        [100*np.std(ele, ddof=1) / np.sqrt(np.array(ele).shape[0]) if np.size(ele) > 1 else 0 for ele in metric] for metric in r20000_test
    ])
    gjb2_test_mean = np.array([
        [100*np.mean(ele) for ele in metric] for metric in gjb2_test
    ])
    gjb2_test_sem = np.array([
        [100*np.std(ele, ddof=1) / np.sqrt(np.array(ele).shape[0]) if np.size(ele) > 1 else 0 for ele in metric] for metric in gjb2_test
    ])
    ryr1_test_mean = np.array([
        [100*np.mean(ele) for ele in metric] for metric in ryr1_test
    ])
    ryr1_test_sem = np.array([
        [100*np.std(ele, ddof=1) / np.sqrt(np.array(ele).shape[0]) if np.size(ele) > 1 else 0 for ele in metric] for metric in ryr1_test
    ])

    r20000_test_plot = {
        'label': [r'TANDEM-DIMPLE for GJB2', r'TANDEM-DIMPLE for RYR1', r'TANDEM', r'Rhapsody', r'AlphaMissense'],
        'hatch': ['', '..', '', '//', '\\\\'],
        'color': ['lightcoral', 'lightcoral', 'lightblue', 'orange', 'lightgreen']
    }
    gjb2_test_plot = {
        'label': [r'TANDEM-DIMPLE for GJB2', r'TANDEM', r'Rhapsody', r'AlphaMissense'],
        'hatch': ['', '', '//', '\\\\'],
        'color': ['lightcoral', 'lightblue', 'orange', 'lightgreen']
    }
    ryr1_test_plot = {
        'label': [r'TANDEM-DIMPLE for RYR1', r'TANDEM', r'Rhapsody', r'AlphaMissense'],
        'hatch': ['..', '', '//', '\\\\'],
        'color': ['lightcoral', 'lightblue', 'orange', 'lightgreen']
    }

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharey=True, dpi=300, width_ratios=[1.25, 1, 1])
    # Space the subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    ax_r20000 = ax[0]
    ax_GJB2 = ax[1]
    ax_RYR1 = ax[2]

    for i in range(len(x_labels)):
        # R20000 plot
        bar_x = np.array([i + j * 0.18 for j in range(len(r20000_test_mean[0]))])
        bar_heights = r20000_test_mean[i]
        bar_errors = r20000_test_sem[i]
        ax_r20000.bar(
            x=bar_x,
            height=bar_heights,
            yerr=bar_errors,
            label=r20000_test_plot['label'],
            hatch=r20000_test_plot['hatch'],
            color=r20000_test_plot['color'],
            error_kw=error_params,
            edgecolor='black',
            width=0.18,
        )
        # Add text labels above bars
        if show_bar_values:
            for x, h, e in zip(bar_x, bar_heights, bar_errors):
                ax_r20000.text(x, h + e + txt_abv_bar, f'{h:.1f}', ha='center', va='bottom', fontsize=5)
        
        # GJB2 plot
        bar_x = np.array([i + j * 0.2 for j in range(len(gjb2_test_mean[0]))])
        bar_heights = gjb2_test_mean[i]
        bar_errors = gjb2_test_sem[i]
        ax_GJB2.bar(
            x=bar_x,
            height=bar_heights,
            yerr=bar_errors,
            label=gjb2_test_plot['label'],
            hatch=gjb2_test_plot['hatch'],
            color=gjb2_test_plot['color'],
            error_kw=error_params,
            edgecolor='black',
            width=0.2,
        )
        # Add text labels above bars
        if show_bar_values:
            for x, h, e in zip(bar_x, bar_heights, bar_errors):
                ax_GJB2.text(x, h + e + txt_abv_bar, f'{h:.1f}', ha='center', va='bottom', fontsize=5)

        # RYR1 plot
        bar_x = np.array([i + j * 0.2 for j in range(len(ryr1_test_mean[0]))])
        bar_heights = ryr1_test_mean[i]
        bar_errors = ryr1_test_sem[i]
        ax_RYR1.bar(
            x=bar_x,
            height=bar_heights,
            yerr=bar_errors,
            label=ryr1_test_plot['label'],
            hatch=ryr1_test_plot['hatch'],
            color=ryr1_test_plot['color'],
            error_kw=error_params,
            edgecolor='black',
            width=0.2,
        )
        # Add text labels above bars
        if show_bar_values:
            for x, h, e in zip(bar_x, bar_heights, bar_errors):
                ax_RYR1.text(x, h + e + txt_abv_bar, f'{h:.1f}', ha='center', va='bottom', fontsize=5)


    # Define custom handles
    general_models = [
        Patch(facecolor='lightblue', edgecolor='black', label='TANDEM', hatch=''),
        Patch(facecolor='orange', edgecolor='black', label='Rhapsody', hatch='//'),
        Patch(facecolor='lightgreen', edgecolor='black', label='AlphaMissense', hatch='\\\\'),
    ]
    specific_models = [
        Patch(facecolor='lightcoral', edgecolor='black', label='TANDEM-DIMPLE for GJB2', hatch=''),
        Patch(facecolor='lightcoral', edgecolor='black', label='TANDEM-DIMPLE for RYR1', hatch='..'),
    ]

    legend = ax_r20000.legend(
        handles=general_models,
        loc='upper right',
        ncol=len(general_models),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.5,
        handletextpad=0.5,
        columnspacing=0.5
    )
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('grey')
    legend = ax_GJB2.legend(
        handles=[Patch(facecolor='lightcoral', edgecolor='black', label='TANDEM-DIMPLE for GJB2', hatch='')],
        loc='upper right',
        ncol=1,
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.5,
        handletextpad=0.5,
        columnspacing=0.5
    )
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('grey')
    legend = ax_RYR1.legend(
        handles=[Patch(facecolor='lightcoral', edgecolor='black', label='TANDEM-DIMPLE for RYR1', hatch='..')],
        loc='upper right',
        ncol=1,
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.5,
        handletextpad=0.5,
        columnspacing=0.5
    )
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('grey')


    ax_r20000.set_xticks(np.arange(len(x_labels)) + 0.27+0.08)
    ax_GJB2.set_xticks(np.arange(len(x_labels)) + 0.3)
    ax_RYR1.set_xticks(np.arange(len(x_labels)) + 0.3)
    for x in ax:
        x.grid(axis='y', linestyle='--', linewidth=0.1)
        # x.set_xticks(np.arange(len(x_labels)) + 0.3)
        x.set_xticklabels(x_labels, fontsize=12)
        x.set_xlim(-0.3, len(x_labels)-0.1)
        # x.set_ylim(0, 113)
        x.set_ylim(0, 117)
        
    for i, title in zip([0, 1, 2], ['R20000$_{test}$', 'GJB2$_{test}$', 'RYR1$_{test}$']):
        ax[i].set_title(title, fontsize=14)
    ax_GJB2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    ax_RYR1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    ax_r20000.set_ylabel('Performance (%)', fontsize=14)
    ax_r20000.set_yticks(np.arange(0, 105, 5))
    ax_r20000.set_yticklabels([str(y) if y % 10 == 0 else "" for y in np.arange(0, 105, 5)], fontsize=10)

    if show_sigstars:
        # --- Significance annotations: GJB2 (transfer vs others) ---
        # GJB2 panel: bar order = [TANDEM_GJB2 (transfer), TANDEM, Rhapsody, AlphaMissense]
        add_transfer_sigstars(
            ax=ax_GJB2,
            means=gjb2_test_mean,         # shape (5, 4) in percent already
            sems=gjb2_test_sem,           # shape (5, 4)
            pvals=pvalues_gjb2,           # columns correspond to [vs TANDEM, vs Rhapsody, vs Alpha]
            idx_transfer=0,               # TANDEM_GJB2
            compare_indices=[1, 2, 3],    # compare to TANDEM, Rhapsody, Alpha
            bar_width=0.2,                # matches your plotting width/offset
            alpha=alpha,
        )

        # RYR1 panel: bar order = [TANDEM_RYR1 (transfer), TANDEM, Rhapsody, AlphaMissense]
        add_transfer_sigstars(
            ax=ax_RYR1,
            means=ryr1_test_mean,         # shape (5, 4)
            sems=ryr1_test_sem,           # shape (5, 4)
            pvals=pvalues_ryr1,           # columns: [vs TANDEM, vs Rhapsody, vs Alpha]
            idx_transfer=0,               # TANDEM_RYR1
            compare_indices=[1, 2, 3],
            bar_width=0.2,
        )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


SAV_PATTERN = re.compile(r"^(\S+)\s+([A-Z])(\d+)([A-Z])$")

def _parse_sav_label(sav):
    """Parse compact SAV labels such as 'Q46897 M1A'."""
    match = SAV_PATTERN.match(str(sav).strip().upper())
    if match is None:
        raise ValueError(f"Cannot parse SAV label: {sav}")
    acc, wt_aa, resid, mut_aa = match.groups()
    return acc, int(resid), wt_aa, mut_aa

def _parse_prediction_value(value):
    """Parse prediction strings such as '0.769±0.026 (pathogenic)'."""
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.lower().startswith("not available"):
        return np.nan
    try:
        return float(text.split("±", 1)[0])
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+", text)
        return float(match.group(0)) if match else np.nan

def _model_to_data_key(model):
    model_name = str(model).strip().upper()
    if model_name in {"TANDEM", "FOUNDATION"}:
        return "tandem"
    if model_name in {"TANDEM-DIMPLE", "TANDEM_DIMPLE", "DIMPLE"}:
        return "tandem_dimple"
    raise ValueError("model must be 'TANDEM' or 'TANDEM-DIMPLE'.")

def _extract_saturation_from_data(data, model):
    model_key = _model_to_data_key(model)
    if model_key not in data.dtype.names:
        raise ValueError(f"Cannot find model predictions in self.data: {model_key}")
    savs = np.asarray(data["SAVs"]).astype(str)
    scores = np.asarray(data[model_key]["path_prob"], dtype=float)
    return savs, scores

def _extract_saturation_from_file(prediction_file, model):
    df = pd.read_csv(prediction_file)
    if "SAV" not in df.columns:
        raise ValueError(f"Prediction file must contain a SAV column: {prediction_file}")
    if model not in df.columns:
        raise ValueError(
            f"Prediction file does not contain '{model}'. "
            f"Available columns: {', '.join(df.columns)}"
        )
    savs = df["SAV"].astype(str).values
    scores = df[model].map(_parse_prediction_value).astype(float).values
    return savs, scores

def _build_saturation_matrix(savs, scores, aa_order):
    aa_to_row = {aa: i for i, aa in enumerate(aa_order)}
    parsed = []
    wt_by_resid = {}
    accs = set()
    max_resid = 0

    for sav, score in zip(savs, scores):
        acc, resid, wt_aa, mut_aa = _parse_sav_label(sav)
        if mut_aa not in aa_to_row:
            continue
        parsed.append((acc, resid, wt_aa, mut_aa, score))
        accs.add(acc)
        wt_by_resid[resid] = wt_aa
        max_resid = max(max_resid, resid)

    if not parsed:
        raise ValueError("No valid saturation mutagenesis SAVs were found.")

    matrix = np.full((len(aa_order), max_resid), np.nan, dtype=float)
    for _, resid, _, mut_aa, score in parsed:
        matrix[aa_to_row[mut_aa], resid - 1] = score

    acc = sorted(accs)[0] if len(accs) == 1 else "multiple proteins"
    return matrix, wt_by_resid, acc

def saturation_mutagenesis(
    data=None,
    prediction_file=None,
    model="TANDEM",
    folder=".",
    filename="saturation_mutagenesis_heatmap.png",
    aa_order="ACDEFGHIKLMNPQRSTVWY",
    title=None,
    figsize=None,
    dpi=300,
    cmap="coolwarm",
    vmin=0,
    vmax=1,
):
    """Plot a saturation mutagenesis pathogenicity heatmap.

    The function accepts either a Tandem self.data array or a saved
    Main_Predictions.txt file. If both are provided, self.data is used.
    """
    if data is not None:
        savs, scores = _extract_saturation_from_data(data, model)
    elif prediction_file is not None:
        savs, scores = _extract_saturation_from_file(prediction_file, model)
    else:
        raise ValueError("Provide either data or prediction_file.")

    matrix, wt_by_resid, acc = _build_saturation_matrix(savs, scores, aa_order)
    n_resid = matrix.shape[1]
    if figsize is None:
        figsize = (max(8, min(30, n_resid / 10)), 6)

    masked = np.ma.masked_invalid(matrix)
    plot_cmap = plt.get_cmap(cmap).copy()
    plot_cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(masked, cmap=plot_cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="none")

    ax.set_yticks(np.arange(len(aa_order)))
    ax.set_yticklabels(list(aa_order))
    tick_step = max(1, int(np.ceil(n_resid / 20)))
    xticks = np.arange(0, n_resid, tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks + 1)
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Mutant amino acid")
    ax.set_title(title or f"Saturation mutagenesis: {acc} ({model})")

    aa_to_row = {aa: i for i, aa in enumerate(aa_order)}
    for resid, wt_aa in wt_by_resid.items():
        row = aa_to_row.get(wt_aa)
        if row is None:
            continue
        ax.add_patch(
            plt.Rectangle(
                (resid - 1.5, row - 0.5), 1, 1,
                linewidth=0.8,
                edgecolor="black",
                facecolor="none",
            )
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Pathogenic probability")
    cbar.set_ticks([0, 0.5, 1])

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    fig.tight_layout()

    filepath = None
    if filename:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": filepath,
        "matrix": matrix,
        "wt_by_resid": wt_by_resid,
        "accession": acc,
    }

    # import matplotlib.pyplot as plt
    # import matplotlib.colors as mcolors
    # from matplotlib.colors import ListedColormap, BoundaryNorm
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from matplotlib import gridspec as gridspec
    # from matplotlib.colors import LinearSegmentedColormap
    # import numpy as np
    # from matplotlib import colors
    # import matplotlib.patches as patches
    # import numpy as np

    # import matplotlib.patches as mpatches
    # from matplotlib.legend_handler import HandlerPatch

    # def plot_ax_patho(ax, data, mask_col=[221, 222, 223, 224, 225], xlim=None):
    #     # data for these columns is masked
    #     data[:, mask_col] = np.nan
        
    #     aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    #     # new_aa_list = 'WYFLIVMCAGPSTNQHRKDE'    
        
    #     # Create a mapping from aa_list to row indices
    #     # aa_to_index = {aa: i for i, aa in enumerate(aa_list)}

    #     # Create a list of indices to reorder the data
    #     # new_order = [aa_to_index[aa] for aa in new_aa_list]

    #     # Reorder the data
    #     # data = data[new_order, :]
        
    #     im = ax.imshow(data, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    #     ax.imshow(np.ma.getmask(data), cmap='Greys', alpha=0.0, aspect='auto')
    
    #     # Assuming masked_data is a numpy.ma.MaskedArray
    #     mask = np.ma.getmask(data)
    #     # Loop through the mask and add borders where mask is True
    #     n_rows, n_cols = mask.shape
    #     for row in range(n_rows):
    #         for col in range(n_cols):
    #             if col in mask_col:
    #                 continue
    #             if mask[row, col]:
    #                 rect = patches.Rectangle(
    #                     (col - 0.5, row - 0.5), 1, 1,  # (x, y), width, height
    #                     linewidth=0.5,
    #                     edgecolor='black',
    #                     facecolor='none'
    #                 )
    #                 ax.add_patch(rect)
                    
    #     # Set y-axis labels
        
    #     ax.set_yticks(np.arange(data.shape[0]))
    #     ax.set_yticklabels(list(aa_list), rotation=0, fontsize=15)
    #     # ax.set_yticklabels(list(new_aa_list), rotation=0, fontsize=15)
    #     # Set x-axis labels
    #     ax.set_xticks([]) # Hide x-axis ticks
    #     ax.set_ylabel('Pathogenic Probability', fontsize=20)
    #     if xlim is not None:
    #         ax.set_xlim(xlim)
    #     # Remove spines
    #     for spine in ax.spines.values():
    #         spine.set_visible(False)
    #     return im

    # def mark_heatmap(ax, resids, aa_indices, labels, pathogenic='red', benign='blue'):
    #     for resid, idx, label in zip(resids, aa_indices, labels):
    #         if label == 1:
    #             edgecolor = pathogenic
    #         elif label == 0:
    #             edgecolor = benign
    #         ax.add_patch(
    #             plt.Rectangle(
    #                 (resid-0.5, idx-0.5), 1, 1,
    #                 linewidth=2,
    #                 edgecolor=edgecolor,
    #                 facecolor='none'
    #             )
    #         )

    # def plot_ax_avg(ax, data, label, color, linewidth=2, linestyle='solid',
    #                 min_max=True, setup=True, get_xlim=None,
    #                 plot_peaks=True):
        
    #     avg_data = np.nanmean(data, axis=0)
    #     min_data = np.nanmin(data, axis=0)
    #     max_data = np.nanmax(data, axis=0)
        
    #     ax.plot(avg_data, color=color, label=label, linewidth=linewidth, linestyle=linestyle)
    #     if min_max:
    #         # Fill from lowest to highest
    #         ax.fill_between(np.arange(len(avg_data)), min_data, max_data, color=color, alpha=0.2)
    #     if setup:
    #         ax.set_xlim(get_xlim)
    #         ax.set_xticks([])
    #         ax.set_ylim(-0.1, 1.1)
    #         ax.set_yticks([0, 0.5, 1])
    #         ax.set_yticklabels([0, 0.5, 1], fontsize=15)
    #         ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
    #         ax.set_ylabel('Average', fontsize=20)
            
    #     if plot_peaks:
    #         # Peaks and troughs
    #         peaks = np.where(np.diff(np.sign(np.diff(avg_data))) < 0)[0] + 1
    #         # top x peaks > 0.8
    #         topxpeaks = np.where(avg_data[peaks] > 0.5)[0]
    #         troughs = np.where(np.diff(np.sign(np.diff(avg_data))) > 0)[0] + 1
    #         # top 20 troughs < 0.2
    #         top20troughs = np.where(avg_data[troughs] < 0.5)[0]
    #         # Plot peaks and troughs
    #         ax.plot(peaks[topxpeaks], avg_data[peaks][topxpeaks], 'o', color='red', markersize=6)
    #         ax.plot(troughs[top20troughs], avg_data[troughs][top20troughs], 'o', color='blue', markersize=6)

    # # Custom handler to scale the patch size in the legend
    # class TallRectangleHandler(HandlerPatch):
    #     def create_artists(self, legend, orig_handle,
    #                     xdescent, ydescent, width, height, fontsize, trans):
    #         # Custom rectangle with scaled width and height
    #         w = width * 0.3
    #         h = height * 1.5
    #         rect = mpatches.Rectangle(
    #             [xdescent + (width - w) / 2, ydescent + (height - h) / 2],
    #             w, h,
    #             edgecolor=orig_handle.get_edgecolor(),
    #             facecolor=orig_handle.get_facecolor(),
    #             linewidth=orig_handle.get_linewidth(),
    #             transform=trans
    #         )
    #         return [rect]


    # def plot_ax_V1(ax, data, get_xlim=None):
    #     white_to_darkred = LinearSegmentedColormap.from_list(
    #     'white_to_darkred', ['white', 'darkred'])

    #     ax.imshow(data,
    #             cmap=white_to_darkred,
    #             vmin=0.0, 
    #             vmax=100,
    #             aspect='auto',
    #             interpolation='none')
    #     ax.set_xlim(get_xlim)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_ylabel(r'$‖V_{1,i}‖$', fontsize=15, rotation=0, labelpad=5)
    #     bbox = ax.get_position()
    #     ax.yaxis.set_label_coords(bbox.x0-0.15, 0)  # y=0.5 is always vertical center

    # def plot_ax_consurf(ax, data, get_xlim=None):
    #     # Define hex colors for cons1 to cons10 (as given before)
    #     hex_colors = [
    #         "#0A7D82",  # cons1
    #         "#4AB0BF",  # cons2
    #         "#A6DBE6",  # cons3
    #         "#D6EFEF",  # cons4
    #         "#FFFFFF",  # cons5
    #         "#FAEAF5",  # cons6
    #         "#FAC7DB",  # cons7
    #         "#F07DAB",  # cons8
    #         "#A02960",  # cons9
    #         "#FFFF96",  # cons10
    #     ]
    #     # Create discrete colormap and normalization
    #     cmap = ListedColormap(hex_colors)
    #     bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     norm = BoundaryNorm(bounds, cmap.N)

    #     # Plot with the custom discrete colormap
    #     ax.imshow(data, cmap=cmap, aspect='auto', interpolation='none')
    #     ax.set_xlim(get_xlim)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     # Set label text and rotation
    #     ax.set_ylabel('ConSurf', fontsize=15, rotation=0, labelpad=5)
    #     bbox = ax.get_position()
    #     ax.yaxis.set_label_coords(bbox.x0-0.15, 0)  # y=0.5 is always vertical center
        
    # def plot_ax_dssp(ax, data, get_xlim=None):
    #     x = 0
    #     i = 0
    #     while i < len(data):
    #         s = data[i]
    #         # s nan continue
    #         if np.isnan(s):
    #             i += 1
    #             continue
    #         # Group consecutive H, E, or -
    #         start = i
    #         current = s
    #         while i < len(dssp) and dssp[i] == current:
    #             i += 1
    #         length = i - start
    #         if current == 0.5:
    #             # # Draw helix as a sine wave
    #             # xs = np.linspace(x, x + length, 100)
    #             # ys = 0.2 * np.sin(10 * np.linspace(0, 2 * np.pi, 100))
    #             # ax_dssp.plot(xs, ys, color='navy', linewidth=2)
    #             # Control frequency: N full sine cycles per residue
    #             cycles_per_residue = 1  # e.g., 1 wave per residue
    #             total_cycles = cycles_per_residue * length
    #             xs = np.linspace(x, x + length, 100)
    #             phase = np.linspace(0, 2 * np.pi * total_cycles, 100)
    #             ys = 0.2 * np.sin(phase)
    #             ax.plot(xs, ys, color='navy', linewidth=2)
                
    #         elif current == 0:
    #             # Draw sheet as a single yellow arrow
    #             rect = patches.FancyArrow(
    #                 x, 0,
    #                 length, 0,
    #                 width=0.5,
    #                 head_width=1,
    #                 head_length=1,
    #                 length_includes_head=True,
    #                 color='darkblue',
    #                 linewidth=0
    #             )
    #             ax.add_patch(rect)
    #         else:
    #             # Draw loop as straight black line
    #             ax.plot([x, x + length], [0, 0], color='black', linewidth=1)
    #         x += length

    #     ax.set_xlim(get_xlim)
    #     ax.set_yticks([])
    #     # # Set label text and rotation
    #     ax.set_ylabel('Sec. Str.', fontsize=15, rotation=0, labelpad=5)
    #     bbox = ax.get_position()
    #     ax.yaxis.set_label_coords(bbox.x0-0.15, 0)  # y=0.5 is always vertical center
    #     # Remove spines
    #     for spine in ax.spines.values():
    #         spine.set_visible(False)

    #     # # Domain information
    #     cx26_domains_dict = {
    #         (1, 13): {"name": "NTH", "type": "Intramembrane"},
    #         # (14, 20): {"name": "TD1", "type": "Topological domain", "location": "Cytoplasmic"},
    #         (14, 20): {"name": "CL1", "type": "Topological domain", "location": "Cytoplasmic"},
    #         (21, 40): {"name": "TM1", "type": "Transmembrane", "structure": "Helical"},
    #         (41, 73): {"name": "EL1", "type": "Topological domain", "location": "Extracellular"},
    #         (74, 94): {"name": "TM2", "type": "Transmembrane", "structure": "Helical"},
    #         (95, 135): {"name": "CL2", "type": "Topological domain", "location": "Cytoplasmic"},
    #         (136, 156): {"name": "TM3", "type": "Transmembrane", "structure": "Helical"},
    #         (157, 189): {"name": "EL2", "type": "Topological domain", "location": "Extracellular"},
    #         (190, 210): {"name": "TM4", "type": "Transmembrane", "structure": "Helical"},
    #         (211, 226): {"name": "CT", "type": "Topological domain", "location": "Cytoplasmic"},
    #     }
    #     # Separate the domains by drawing rectangles
    #     for (start, end), domain_info in cx26_domains_dict.items():
    #         # Plot the name in the center of the rectangle
    #         # Plot a vertical line at the start and end of the domain
    #         if end < get_xlim[0]:
    #             continue
    #         # if start >= get_xlim[1]+0.5:
    #         #     continue
    #         ax.axvline(start-1.0, color='black', linewidth=0.5)
    #         if end == data.shape[0]:
    #             ax.axvline(end-1, color='black', linewidth=0.5)
    #         ax.text(
    #             (start + end) / 2 - 0.5, 1.3,
    #             domain_info["name"],
    #             ha='center', va='center',
    #             fontsize=12,
    #             color='black',
    #             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    #         )
    #         # 94.5
    #     xticks = np.array([1, 14, 21, 41, 74, 95, 136, 157, 190, 211, 226, 27, 34, 197, 203 ])
    #     show_xticks = xticks[(xticks-0.5 >= get_xlim[0]) & (xticks-0.5 <= get_xlim[1])]

    #     xticks = np.array(show_xticks)
    #     ax.set_xticks(xticks-1)
    #     ax.set_xticklabels(xticks, rotation=0, fontsize=15)
    #     # ax x title
    #     ax.set_xlabel('Residue number', fontsize=20) # 
    #     # Shift x label upper
    #     # bbox = ax.get_position()
    #     # ax.xaxis.set_label_coords(bbox.x0+0.5, -0.15)  # x=0.5 is always horizontal center
    #     # ax.xaxis.set_label_coords(bbox.x0+0.5, -0.15)  # x=0.5 is always horizontal center

    # import matplotlib.pyplot as plt
    # from matplotlib import gridspec as gridspec
    # import numpy as np


    # # # AlphaMissense data
    # # avg_aln_preds, masked_aln_preds
    # # # TANDEM_GJB2 data
    # # masked_td_GJB2_preds, avg_td_GJB2_preds, min_td_GJB2_preds, max_td_GJB2_preds
    # fig_height=8
    # fig_width=25
    # fig_width=30
    # dpi=300
    # num_intervals=9
    # fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    # # Create GridSpec with custom row and column size ratios
    # gs = gridspec.GridSpec(
    #     # nrows=7, ncols=2, 
    #     nrows=7, ncols=1, 
    #     height_ratios=[7.8, 0.02, 2, 0.3, 0.3, 0.1, 0.3], 
    #     # width_ratios=[9.9, 0.15],
    #     hspace=0.1,  # global hspace for others
    #     # wspace=0.02
    # )

    # # Create axes individually
    # ax_patho = fig.add_subplot(gs[0, 0])     # Heatmap
    # ax_avg = fig.add_subplot(gs[2, 0])       # Line plot (avg, sem)
    # ax_consurf = fig.add_subplot(gs[3, 0])   # ConSurf
    # ax_V1 = fig.add_subplot(gs[4, 0])        # V1
    # ax_dssp = fig.add_subplot(gs[6, 0])      # Secondary Structure
    # # ax_cb = fig.add_subplot(gs[0, 1])        # Colorbar
    # ax_spacer = fig.add_subplot(gs[1, 0])       # Spacer for bottom
    # ax_spacer.axis('off')  # Hide the spacer axis
    # ax_spacer = fig.add_subplot(gs[5, 0])       # Spacer for bottom
    # ax_spacer.axis('off')  # Hide the spacer axis

    # # # Pathogenicity heatmap
    # im = plot_ax_patho(ax_patho, masked_td_GJB2_preds, xlim=(-0.5, 225+8))

    # # xlim1 = (-0.5, 94+0.5)
    # # xlim2 = (95+0.5, 225+8)
    # # ax_patho.set_xlim(-0.5, 94+0.5)
    # # ax_patho.set_xlim(xlim2[0], xlim2[1])
    # # ax_patho.set_xlim(xlim1[0], xlim1[1])

    # # Draw first so layout settles
    # plt.draw()

    # # Get ax_patho position to calculate where to place colorbar
    # x0, y0, width, height = ax_patho.get_position().bounds

    # # Now place a new inset Axes inside that region for colorbar
    # cbar_ax = fig.add_axes([
    #     x0 + width - 0.035,  # push a bit inward from right edge
    #     y0 + 0.095 * height,  # start from ~15% up
    #     0.012,               # narrow width
    #     0.7 * height         # about 70% of the heatmap height
    # ])

    # # Now draw the colorbar in the inset axes
    # cbar = fig.colorbar(im, cax=cbar_ax)
    # # Customize appearance
    # cbar.set_ticks([0, 0.5, 1])
    # cbar.set_ticklabels([0, 0.5, 1], fontsize=15)
    # cbar.ax.text(0.5, 0.25, 'B', ha='center', va='center', fontsize=15)
    # cbar.ax.text(0.5, 0.75, 'P', ha='center', va='center', fontsize=15)
    # cbar.ax.axhline(0.5, color='black', linestyle='--')


    # # Mark known SAVs on the heaxtmap
    # mark_heatmap(ax_patho, resids, mut_aa_indices, labels)
    # # Add legend to the heatmap
    # # Define handles
    # red_rect = mpatches.Rectangle((0, 0), 1, 1, edgecolor='red', facecolor='none', linewidth=2, label='knwP')
    # blue_rect = mpatches.Rectangle((0, 0), 1, 1, edgecolor='blue', facecolor='none', linewidth=2, label='knwB')
    # white_rect = mpatches.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='none', linewidth=2, label='WT')
    # # Add legend with custom handler
    # ax_patho.legend(
    #     handles=[red_rect, blue_rect, white_rect],
    #     handler_map={mpatches.Rectangle: TallRectangleHandler()},
    #     # bbox_to_anchor=(1.05, -0.05),
    #     # bbox_to_anchor=(1.05, -0.05),
    #     loc='upper right',
    #     frameon=False,
    #     handletextpad=0.1,
    #     fontsize=12,
    #     labelspacing=0.5
    # )
    # # Plot avg
    # plot_ax_avg(ax_avg, masked_td_GJB2_preds, label=r'TANDEM$_{GJB2}$', color='red', linewidth=2, linestyle='solid', min_max=True, setup=True, get_xlim=ax_patho.get_xlim(), plot_peaks=True)
    # # Plot avg for AlphaMissense
    # plot_ax_avg(ax_avg, masked_aln_preds, label='AlphaMissense', color='darkgreen', linewidth=2, linestyle=(5, (10,3)), min_max=False, setup=False, get_xlim=ax_patho.get_xlim(), plot_peaks=False)
    # ax_avg.legend(
    #     loc='upper right', 
    #     fontsize=15,
    #     bbox_to_anchor=(0.4, 0.41),
    #     bbox_transform=ax_avg.transAxes,
    #     frameon=False,
    #     handlelength=1.5,
    #     handletextpad=0.5,
    #     borderpad=0.5,
    #     borderaxespad=0.5,
    #     labelspacing=0.5,
    #     ncol=2,
    #     # ncol=1,
    # )
    # # Plot V1
    # plot_ax_V1(ax_V1, V1_normalized, get_xlim=ax_patho.get_xlim())
    # # Plot ConSurf
    # plot_ax_consurf(ax_consurf, consurf_color, get_xlim=ax_patho.get_xlim())
    # # Plot DSSP
    # dssp = df_feat.dssp.values[:-1:19]
    # dssp[-5:-1] = 1
    # # plot_ax_dssp(ax_dssp, dssp, get_xlim=(-0.5, 225))
    # plot_ax_dssp(ax_dssp, dssp, get_xlim=ax_patho.get_xlim())

    # arrow_style = dict(shrink=0.01, width=1, headwidth=8, headlength=5)
    # x_pos = [ 34,  37,  44,  44,  50,  59,  75,  75,  84,  90,  95, 143, 143, 161, 163, 179, 184, 195, 197, 202, 205, 206]
    # x_pos = [v for v in x_pos if v-0.5 >= ax_patho.get_xlim()[0] and v-0.5 <= ax_patho.get_xlim()[1]]
    # for pos in x_pos:
    #     pos -= 1
    #     y_base = ax_patho.get_ylim()[1] - 0.2
    #     dy = 0.1  # same relative arrow length
    #     ax_patho.annotate(
    #         '',
    #         xy=(pos, y_base + dy),         # arrow tip
    #         xytext=(pos, y_base - dy),     # arrow tail
    #         arrowprops={**arrow_style, 'facecolor': 'red', 'edgecolor': 'red'},
    #         annotation_clip=False
    #     )

    # x_pos = [217, 215, 214, 210, 203, 197, 170, 170, 168, 156, 153, 127, 123, 121, 115, 114, 111, 107, 100,  83,  27,  16,   4,   4, 165]
    # x_pos = [v for v in x_pos if v-0.5 >= ax_patho.get_xlim()[0] and v-0.5 <= ax_patho.get_xlim()[1]]
    # for pos in x_pos:
    #     pos -= 1
    #     y_base = ax_patho.get_ylim()[1] - 0.2
    #     dy = 0.1  # same relative arrow length
    #     ax_patho.annotate(
    #         '',
    #         xy=(pos, y_base + dy),         # arrow tip
    #         xytext=(pos, y_base - dy),     # arrow tail
    #         arrowprops={**arrow_style, 'facecolor': 'blue', 'edgecolor': 'blue'},
    #         annotation_clip=False
    #     )
        
    # x_pos = [27, 34, 197, 203]
    # x_pos = [v for v in x_pos if v >= ax_patho.get_xlim()[0] and v <= ax_patho.get_xlim()[1]]
    # for pos in x_pos:
    #     pos -= 1
    #     y_base = ax_patho.get_ylim()[0] + 0.2
    #     # dy = 0.1  # fixed relative length

    #     # y_pos = ax_patho.get_ylim()[0]+1  # bottom of the y-axis
    #     ax_patho.annotate(
    #         '',
    #         xy=(pos, y_base - dy),        # arrow tip
    #         xytext=(pos, y_base + dy),    # arrow start (a bit above)
    #         arrowprops={**arrow_style, 'facecolor': 'black', 'edgecolor': 'black'},
    #         annotation_clip=False
    #     )
    #     y_pos = ax_dssp.get_ylim()[1]-0.2  # top of the y-axis
    #     ax_dssp.annotate(
    #         '',
    #         xy=(pos, y_pos -0.1),        # arrow tip
    #         xytext=(pos, y_pos +0.1),    # arrow start (a bit above)
    #         arrowprops={**arrow_style, 'facecolor': 'black', 'edgecolor': 'black'},
    #         annotation_clip=False
    #     )

    # # plt.suptitle(r'$\it{In\ silico}$ Saturation Mutagenesis of GJB2', fontsize=20, y=0.95)
    # plt.subplots_adjust(top=0.9)  # adjust as needed (smaller = closer)
    # # plt.savefig('/mnt/nas_1/YangLab/loci/tandem/models/saturation_mutagenesis_GJB2.png', dpi=300, bbox_inches='tight')
    # # plt.close()
    # plt.show()
