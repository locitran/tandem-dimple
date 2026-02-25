import os
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prody
from matplotlib import colors as mcolors
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import FancyArrow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.features import TANDEM_FEATS
from src.predict.inference import ModelInference
from src.utils.settings import ROOT_DIR

basedir = os.path.join(ROOT_DIR, 'models')
tf_gjb2 = os.path.join(basedir, 'TANDEM_GJB2')
FIGURE_OUTDIR = os.path.join(ROOT_DIR, 'jobs', 'figures')
os.makedirs(FIGURE_OUTDIR, exist_ok=True)
feat = f'{ROOT_DIR}/jobs/GJB2_full/GJB2_full-features.csv'
df_GJB2 = pd.read_csv(feat)
feat_names = TANDEM_FEATS['v1.1']
cols = ['SAV_coords'] + feat_names
df_GJB2 = df_GJB2[cols]

acc, pos, wt, mt = np.array([i.split() for i in df_GJB2['SAV_coords'].values]).T
wt_seq = np.array([wt[i*19] for i in range(int(len(wt) / 19))])
pos_seq = np.array([pos[i*19] for i in range(int(len(pos) / 19))])

aa_list = 'ACDEFGHIKLMNPQRSTVWY'
wt_idx = np.array([aa_list.index(res) for res in wt_seq])

mask_idx = df_GJB2['GNM_V1_full'].isna().values
mask_idx = mask_idx.reshape(-1, 1)

mi_GJB2 = ModelInference(folder=tf_gjb2)
mi = ModelInference(folder='models/TANDEM')

fm = df_GJB2[feat_names].values.astype(np.float32).reshape(df_GJB2.shape[0], -1)
mi_GJB2.calcPredictions(fm)
mi.calcPredictions(fm)

df_aln = pd.read_csv(f"{ROOT_DIR}/data/GJB2/AlphaMissense-Search-P29033.tsv", sep="\t")

aln_wt = df_aln['a.a.1'].values[:-1:19]
aa_list = 'ACDEFGHIKLMNPQRSTVWY'
aln_wt_idx = np.array([10] + [aa_list.index(res) for res in aln_wt])
aln_preds = df_aln['pathogenicity score'].values
add_one_residue = np.full(aln_preds.shape[0] + 19, np.nan, dtype=np.float32)
add_one_residue[19:] = df_aln['pathogenicity score'].values

aln_preds = add_one_residue.reshape(-1, 19)
masked_aln_preds  = np.ma.masked_all((aln_preds.shape[0], 20), dtype=aln_preds.dtype)
masked_aln_preds[:, :19] = aln_preds

for row, idx in enumerate(aln_wt_idx):
    if row == 0:
        continue
    for col in range(19, idx, -1):
        masked_aln_preds[row, col] = masked_aln_preds[row, col - 1]
    masked_aln_preds[row, idx] = np.nan
        
masked_aln_preds = masked_aln_preds.T
avg_aln_preds = np.nanmean(masked_aln_preds, axis=0)

tdGJB2_preds = mi_GJB2.final_probs
tdGJB2_preds = np.ma.masked_where(mask_idx, tdGJB2_preds)
tdGJB2_preds = tdGJB2_preds.reshape(-1, 19)

masked_td_GJB2_preds = np.full((tdGJB2_preds.shape[0], 20), np.nan, dtype=tdGJB2_preds.dtype)
masked_td_GJB2_preds[:, :19] = tdGJB2_preds

for row, idx in enumerate(wt_idx):
    for col in range(19, idx, -1):
        masked_td_GJB2_preds[row, col]  = masked_td_GJB2_preds[row, col - 1]
    masked_td_GJB2_preds[row, idx] = -1
        
masked_td_GJB2_preds = masked_td_GJB2_preds.T

masked_td_GJB2_preds = np.ma.masked_where(masked_td_GJB2_preds == -1, masked_td_GJB2_preds)
avg_td_GJB2_preds = np.mean(masked_td_GJB2_preds, axis=0)
min_td_GJB2_preds = np.min(masked_td_GJB2_preds, axis=0)
max_td_GJB2_preds = np.max(masked_td_GJB2_preds, axis=0)

avg_data = avg_td_GJB2_preds*100

pdbpath= f'{ROOT_DIR}/data/GJB2/structures/8qa2_opm_25Apr03.pdb'
pdb = prody.parsePDB(pdbpath)
residues = list(pdb.select('chain A').getHierView().iterResidues())
residues = [
    res.setBetas(avg_data[i]) for i, res in enumerate(residues)
]
residues = list(pdb.select('not chain A').getHierView().iterResidues())
residues = [res.setBetas(-1) for i, res in enumerate(residues)]
prody.writePDB(f'{tf_gjb2}/8qa2-mem-pathogenicity.pdb', pdb)

####### Consurf color
consurf_color = f'{ROOT_DIR}/jobs/consurf_color_GJB2/consurf_color_GJB2-features.csv'
df_feat = pd.read_csv(consurf_color)
consurf_color = df_feat.consurf_color.values[:-1:19][np.newaxis, :]
# color > 10 assign 10
consurf_color[consurf_color > 10] = 10

####### V_1
V1_masked = np.ma.masked_where(mask_idx, df_GJB2['GNM_V1_full'].values.reshape(-1, 1))
vmin = V1_masked.min()
vmax = V1_masked.max()
V1_normalized = (V1_masked - vmin) / (vmax - vmin)
V1_normalized = V1_normalized[0: -1: 19][np.newaxis, :]
V1_normalized = V1_normalized*100

df_pred = pd.DataFrame(columns=['SAV_coords', 'TANDEM_GJB2'])
df_pred['SAV_coords'] = df_GJB2['SAV_coords'].values
df_pred['TANDEM_GJB2'] = mi_GJB2.final_probs
df_pred['TANDEM'] = mi.final_probs
df_pred['AlphaMissense'] = add_one_residue

df_GJB2_VUS = pd.read_csv(f"{ROOT_DIR}/data/GJB2/SAVs.csv")
df_GJB2_VUS = df_GJB2_VUS[df_GJB2_VUS['labels'].notna()].copy()

df_GJB2_VUS = df_GJB2_VUS.merge(
    df_pred[['SAV_coords', 'TANDEM', 'TANDEM_GJB2', 'AlphaMissense']],
    on='SAV_coords',
    how='left',
)

pos = df_GJB2_VUS['SAV_coords'].apply(lambda x: int(x.split()[1]))

labels = ['TP', 'FP', 'TN', 'FN']
for i, row in df_GJB2_VUS.iterrows():
    for col in ['TANDEM', 'TANDEM_GJB2', 'AlphaMissense']:
        col_label = col + '_label'
        col_color = col + '_color'
        if row[col] > 0.5 and row['labels'] == 1:
            df_GJB2_VUS.at[i, col_label] = 'TP'
            df_GJB2_VUS.at[i, col_color] = 'red'
            
        elif row[col] > 0.5 and row['labels'] == 0:
            df_GJB2_VUS.at[i, col_label] = 'FP'
            df_GJB2_VUS.at[i, col_color] = 'royalblue'

        elif row[col] <= 0.5 and row['labels'] == 1:
            df_GJB2_VUS.at[i, col_label] = 'FN'
            df_GJB2_VUS.at[i, col_color] = 'coral'
            
        elif row[col] <= 0.5 and row['labels'] == 0:
            df_GJB2_VUS.at[i, col_label] = 'TN'
            df_GJB2_VUS.at[i, col_color] = 'blue'
df_GJB2_VUS['pos'] = pos

# y-axis df_GJB2_VUS['TANDEM_GJB2']

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 6), sharey=True)

plt.subplots_adjust(hspace=0, wspace=0)
ax0 = df_GJB2_VUS[df_GJB2_VUS['TANDEM_GJB2_label'].isin(['FP', 'TN'])]
ax1 = df_GJB2_VUS[df_GJB2_VUS['TANDEM_GJB2_label'].isin(['TP', 'FN'])]
ax[0].scatter(ax0['pos'], ax0['TANDEM_GJB2'], label='TANDEM GJB2', color=ax0['TANDEM_GJB2_color'])
ax[1].scatter(ax1['pos'], ax1['TANDEM_GJB2'], label='TANDEM GJB2', color=ax1['TANDEM_GJB2_color'])

# Set x-title of ax[0]
ax[0].set_title('Benign')
ax[1].set_title('Pathogenic')
# Set y-title of fig
ax[0].set_ylabel('Pathogenic Probability', fontsize=15)
# Set x-label of fig
ax[0].set_xlim(-10, 232)
ax[1].set_xlim(-10, 232)
ax[0].set_ylim(0, 1)
ax[0].axhline(y=0.5, color='black', linestyle='--', linewidth=0.5)
ax[1].axhline(y=0.5, color='black', linestyle='--', linewidth=0.5)

fig.supxlabel('Residue number', fontsize=15)

# Notation for 'FP', 'TN', 'TP', 'FN'
# Middle 
ax[0].text(0.9, 0.51, 'FP', transform=ax[0].transAxes, fontsize=15, color='royalblue', ha='center')
ax[0].text(0.9, 0.45, 'TN', transform=ax[0].transAxes, fontsize=15, color='blue', ha='center')
ax[1].text(0.09, 0.51, 'TP', transform=ax[1].transAxes, fontsize=15, color='red', ha='center')
ax[1].text(0.09, 0.45, 'FN', transform=ax[1].transAxes, fontsize=15, color='coral', ha='center')
# fig.set_facecolor('none')  # or 'none' for transparent
# plt.setp(ax, facecolor='none')  # Set axes background to transparent
fig.savefig(
    os.path.join(FIGURE_OUTDIR, 'figure4_scatter_pathogenic_probability.png'),
    dpi=300,
    bbox_inches='tight',
)
plt.show()

def plot_ax_patho(ax, data, mask_col=[221, 222, 223, 224, 225], xlim=None):
    # data for these columns is masked
    data[:, mask_col] = np.nan
    
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    # new_aa_list = 'WYFLIVMCAGPSTNQHRKDE'    
    
    # Create a mapping from aa_list to row indices
    # aa_to_index = {aa: i for i, aa in enumerate(aa_list)}

    # Create a list of indices to reorder the data
    # new_order = [aa_to_index[aa] for aa in new_aa_list]

    # Reorder the data
    # data = data[new_order, :]
    
    im = ax.imshow(data, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    ax.imshow(np.ma.getmask(data), cmap='Greys', alpha=0.0, aspect='auto')
   
    # Assuming masked_data is a numpy.ma.MaskedArray
    mask = np.ma.getmask(data)
    # Loop through the mask and add borders where mask is True
    n_rows, n_cols = mask.shape
    for row in range(n_rows):
        for col in range(n_cols):
            if col in mask_col:
                continue
            if mask[row, col]:
                rect = patches.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,  # (x, y), width, height
                    linewidth=0.5,
                    edgecolor='black',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
    # Set y-axis labels
    
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(list(aa_list), rotation=0, fontsize=15)
    # ax.set_yticklabels(list(new_aa_list), rotation=0, fontsize=15)
    # Set x-axis labels
    ax.set_xticks([]) # Hide x-axis ticks
    ax.set_ylabel('Pathogenic Probability', fontsize=20)
    if xlim is not None:
        ax.set_xlim(xlim)
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im

def mark_heatmap(ax, resids, aa_indices, labels, pathogenic='red', benign='blue'):
    for resid, idx, label in zip(resids, aa_indices, labels):
        if label == 1:
            edgecolor = pathogenic
        elif label == 0:
            edgecolor = benign
        ax.add_patch(
            plt.Rectangle(
                (resid-0.5, idx-0.5), 1, 1,
                linewidth=2,
                edgecolor=edgecolor,
                facecolor='none'
            )
        )

def plot_ax_avg(ax, data, label, color, linewidth=2, linestyle='solid',
                min_max=True, setup=True, get_xlim=None,
                plot_peaks=True):
    
    avg_data = np.nanmean(data, axis=0)
    min_data = np.nanmin(data, axis=0)
    max_data = np.nanmax(data, axis=0)
    
    ax.plot(avg_data, color=color, label=label, linewidth=linewidth, linestyle=linestyle)
    if min_max:
        # Fill from lowest to highest
        ax.fill_between(np.arange(len(avg_data)), min_data, max_data, color=color, alpha=0.2)
    if setup:
        ax.set_xlim(get_xlim)
        ax.set_xticks([])
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1], fontsize=15)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Average', fontsize=20)
        
    if plot_peaks:
        # Peaks and troughs
        peaks = np.where(np.diff(np.sign(np.diff(avg_data))) < 0)[0] + 1
        # top x peaks > 0.8
        topxpeaks = np.where(avg_data[peaks] > 0.5)[0]
        troughs = np.where(np.diff(np.sign(np.diff(avg_data))) > 0)[0] + 1
        # top 20 troughs < 0.2
        top20troughs = np.where(avg_data[troughs] < 0.5)[0]
        # Plot peaks and troughs
        ax.plot(peaks[topxpeaks], avg_data[peaks][topxpeaks], 'o', color='red', markersize=6)
        ax.plot(troughs[top20troughs], avg_data[troughs][top20troughs], 'o', color='blue', markersize=6)

# Custom handler to scale the patch size in the legend
class TallRectangleHandler(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Custom rectangle with scaled width and height
        w = width * 0.3
        h = height * 1.5
        rect = mpatches.Rectangle(
            [xdescent + (width - w) / 2, ydescent + (height - h) / 2],
            w, h,
            edgecolor=orig_handle.get_edgecolor(),
            facecolor=orig_handle.get_facecolor(),
            linewidth=orig_handle.get_linewidth(),
            transform=trans
        )
        return [rect]

def plot_ax_V1(ax, data, get_xlim=None):
    white_to_darkred = LinearSegmentedColormap.from_list(
    'white_to_darkred', ['white', 'darkred'])

    ax.imshow(data,
            cmap=white_to_darkred,
            vmin=0.0, 
            vmax=100,
            aspect='auto',
            interpolation='none')
    ax.set_xlim(get_xlim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r'$‖V_{1,i}‖$', fontsize=15, rotation=0, labelpad=5)
    bbox = ax.get_position()
    ax.yaxis.set_label_coords(bbox.x0-0.15, 0)  # y=0.5 is always vertical center

def plot_ax_consurf(ax, data, get_xlim=None):
    # Define hex colors for cons1 to cons10 (as given before)
    hex_colors = [
        "#0A7D82",  # cons1
        "#4AB0BF",  # cons2
        "#A6DBE6",  # cons3
        "#D6EFEF",  # cons4
        "#FFFFFF",  # cons5
        "#FAEAF5",  # cons6
        "#FAC7DB",  # cons7
        "#F07DAB",  # cons8
        "#A02960",  # cons9
        "#FFFF96",  # cons10
    ]
    # Create discrete colormap
    cmap = ListedColormap(hex_colors)

    # Plot with the custom discrete colormap
    ax.imshow(data, cmap=cmap, aspect='auto', interpolation='none')
    ax.set_xlim(get_xlim)
    ax.set_xticks([])
    ax.set_yticks([])
    # Set label text and rotation
    ax.set_ylabel('ConSurf', fontsize=15, rotation=0, labelpad=5)
    bbox = ax.get_position()
    ax.yaxis.set_label_coords(bbox.x0-0.15, 0)  # y=0.5 is always vertical center
    
def plot_ax_dssp(ax, data, get_xlim=None):
    x = 0
    i = 0
    while i < len(data):
        s = data[i]
        # s nan continue
        if np.isnan(s):
            i += 1
            continue
        # Group consecutive H, E, or -
        start = i
        current = s
        while i < len(data) and data[i] == current:
            i += 1
        length = i - start
        if current == 0.5:
            # # Draw helix as a sine wave
            # xs = np.linspace(x, x + length, 100)
            # ys = 0.2 * np.sin(10 * np.linspace(0, 2 * np.pi, 100))
            # ax_dssp.plot(xs, ys, color='navy', linewidth=2)
            # Control frequency: N full sine cycles per residue
            cycles_per_residue = 1  # e.g., 1 wave per residue
            total_cycles = cycles_per_residue * length
            xs = np.linspace(x, x + length, 100)
            phase = np.linspace(0, 2 * np.pi * total_cycles, 100)
            ys = 0.2 * np.sin(phase)
            ax.plot(xs, ys, color='navy', linewidth=2)
            
        elif current == 0:
            # Draw sheet as a single yellow arrow
            rect = patches.FancyArrow(
                x, 0,
                length, 0,
                width=0.5,
                head_width=1,
                head_length=1,
                length_includes_head=True,
                color='darkblue',
                linewidth=0
            )
            ax.add_patch(rect)
        else:
            # Draw loop as straight black line
            ax.plot([x, x + length], [0, 0], color='black', linewidth=1)
        x += length

    ax.set_xlim(get_xlim)
    ax.set_yticks([])
    # # Set label text and rotation
    ax.set_ylabel('Sec. Str.', fontsize=15, rotation=0, labelpad=5)
    bbox = ax.get_position()
    ax.yaxis.set_label_coords(bbox.x0-0.15, 0)  # y=0.5 is always vertical center
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # # Domain information
    cx26_domains_dict = {
        (1, 13): {"name": "NTH", "type": "Intramembrane"},
        # (14, 20): {"name": "TD1", "type": "Topological domain", "location": "Cytoplasmic"},
        (14, 20): {"name": "CL1", "type": "Topological domain", "location": "Cytoplasmic"},
        (21, 40): {"name": "TM1", "type": "Transmembrane", "structure": "Helical"},
        (41, 73): {"name": "EL1", "type": "Topological domain", "location": "Extracellular"},
        (74, 94): {"name": "TM2", "type": "Transmembrane", "structure": "Helical"},
        (95, 135): {"name": "CL2", "type": "Topological domain", "location": "Cytoplasmic"},
        (136, 156): {"name": "TM3", "type": "Transmembrane", "structure": "Helical"},
        (157, 189): {"name": "EL2", "type": "Topological domain", "location": "Extracellular"},
        (190, 210): {"name": "TM4", "type": "Transmembrane", "structure": "Helical"},
        (211, 226): {"name": "CT", "type": "Topological domain", "location": "Cytoplasmic"},
    }
    # Separate the domains by drawing rectangles
    for (start, end), domain_info in cx26_domains_dict.items():
        # Plot the name in the center of the rectangle
        # Plot a vertical line at the start and end of the domain
        if end < get_xlim[0]:
            continue
        # if start >= get_xlim[1]+0.5:
        #     continue
        ax.axvline(start-1.0, color='black', linewidth=0.5)
        if end == data.shape[0]:
            ax.axvline(end-1, color='black', linewidth=0.5)
        ax.text(
            (start + end) / 2 - 0.5, 1.3,
            domain_info["name"],
            ha='center', va='center',
            fontsize=12,
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )
        # 94.5
    xticks = np.array([1, 14, 21, 41, 74, 95, 136, 157, 190, 211, 226, 27, 34, 197, 203 ])
    show_xticks = xticks[(xticks-0.5 >= get_xlim[0]) & (xticks-0.5 <= get_xlim[1])]

    xticks = np.array(show_xticks)
    ax.set_xticks(xticks-1)
    ax.set_xticklabels(xticks, rotation=0, fontsize=15)
    # ax x title
    ax.set_xlabel('Residue number', fontsize=20) # 
    # Shift x label upper
    # bbox = ax.get_position()
    # ax.xaxis.set_label_coords(bbox.x0+0.5, -0.15)  # x=0.5 is always horizontal center
    # ax.xaxis.set_label_coords(bbox.x0+0.5, -0.15)  # x=0.5 is always horizontal center
    
df_SAVs = pd.read_csv(f'{ROOT_DIR}/data/GJB2/SAVs.csv')
df_knwSAVs = df_SAVs[~df_SAVs['labels'].isna()]

SAV_coords = df_knwSAVs['SAV_coords'].values
resids, wt_aas, mut_aas = zip(*[ele.split()[1:4] for ele in SAV_coords])
labels = df_knwSAVs['labels'].values
resids = np.array(resids).astype(int) - 1

aa_list = 'ACDEFGHIKLMNPQRSTVWY'
mut_aa_indices = np.array([aa_list.index(aa) for aa in mut_aas])

# # AlphaMissense data
# avg_aln_preds, masked_aln_preds
# # TANDEM_GJB2 data
# masked_td_GJB2_preds, avg_td_GJB2_preds, min_td_GJB2_preds, max_td_GJB2_preds
fig_height=8
fig_width=25
dpi=300
fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
# Create GridSpec with custom row and column size ratios
gs = gridspec.GridSpec(
    # nrows=7, ncols=2, 
    # nrows=7, ncols=1, 
    nrows=6, ncols=1, 
    height_ratios=[7.8, 0.02, 2, 0.3, 0.3, 0.3], 
    # height_ratios=[7.8, 0.02, 2, 0.3, 0.3, 0.1, 0.3], 
    # width_ratios=[9.9, 0.15],
    hspace=0.1,  # global hspace for others
    # wspace=0.02
)

# Create axes individually
ax_patho = fig.add_subplot(gs[0, 0])     # Heatmap
ax_spacer = fig.add_subplot(gs[1, 0])       # Spacer for bottom
ax_spacer.axis('off')  # Hide the spacer axis
ax_avg = fig.add_subplot(gs[2, 0])       # Line plot (avg, sem)
ax_consurf = fig.add_subplot(gs[3, 0])   # ConSurf
ax_spacer = fig.add_subplot(gs[4, 0])       # Spacer for bottom
ax_spacer.axis('off')  # Hide the spacer axis
ax_dssp = fig.add_subplot(gs[5, 0])      # Secondary Structure
# # Pathogenicity heatmap
im = plot_ax_patho(ax_patho, masked_td_GJB2_preds, xlim=(-0.5, 225+8))

# xlim1 = (-0.5, 94+0.5)
# xlim2 = (95+0.5, 225+8)
# ax_patho.set_xlim(-0.5, 94+0.5)
# ax_patho.set_xlim(xlim2[0], xlim2[1])
# ax_patho.set_xlim(xlim1[0], xlim1[1])

# Draw first so layout settles
plt.draw()

# Get ax_patho position to calculate where to place colorbar
x0, y0, width, height = ax_patho.get_position().bounds

# Now place a new inset Axes inside that region for colorbar
cbar_ax = fig.add_axes([
    x0 + width - 0.035,  # push a bit inward from right edge
    y0 + 0.095 * height,  # start from ~15% up
    0.012,               # narrow width
    0.7 * height         # about 70% of the heatmap height
])

# Now draw the colorbar in the inset axes
cbar = fig.colorbar(im, cax=cbar_ax)
# Customize appearance
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels([0, 0.5, 1], fontsize=15)
cbar.ax.text(0.5, 0.25, 'B', ha='center', va='center', fontsize=15)
cbar.ax.text(0.5, 0.75, 'P', ha='center', va='center', fontsize=15)
cbar.ax.axhline(0.5, color='black', linestyle='--')


# Mark known SAVs on the heaxtmap
mark_heatmap(ax_patho, resids, mut_aa_indices, labels)
# Add legend to the heatmap
# Define handles
red_rect = mpatches.Rectangle((0, 0), 1, 1, edgecolor='red', facecolor='none', linewidth=2, label='knwP')
blue_rect = mpatches.Rectangle((0, 0), 1, 1, edgecolor='blue', facecolor='none', linewidth=2, label='knwB')
white_rect = mpatches.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='none', linewidth=2, label='WT')
# Add legend with custom handler
ax_patho.legend(
    handles=[red_rect, blue_rect, white_rect],
    handler_map={mpatches.Rectangle: TallRectangleHandler()},
    bbox_to_anchor=(0.94, 1),
    # loc='upper right',
    frameon=False,
    handletextpad=0.1,
    fontsize=12,
    labelspacing=0.5
)
# Plot avg
# plot_ax_avg(ax_avg, masked_td_GJB2_preds, label=r'TANDEM$_{GJB2}$', color='red', linewidth=2, linestyle='solid', min_max=True, setup=True, get_xlim=ax_patho.get_xlim(), plot_peaks=True)
plot_ax_avg(ax_avg, masked_td_GJB2_preds, label=r'TANDEM-DIMPLE for GJB2', color='red', linewidth=2, linestyle='solid', min_max=True, setup=True, get_xlim=ax_patho.get_xlim(), plot_peaks=True)
# Plot avg for AlphaMissense
plot_ax_avg(ax_avg, masked_aln_preds, label='AlphaMissense', color='darkgreen', linewidth=2, linestyle=(5, (10,3)), min_max=False, setup=False, get_xlim=ax_patho.get_xlim(), plot_peaks=False)
ax_avg.legend(
    loc='upper right', 
    fontsize=15,
    bbox_to_anchor=(0.4, 0.41),
    bbox_transform=ax_avg.transAxes,
    frameon=False,
    handlelength=1.5,
    handletextpad=0.5,
    borderpad=0.5,
    borderaxespad=0.5,
    labelspacing=0.5,
    ncol=2,
    # ncol=1,
)
# Plot V1
# plot_ax_V1(ax_V1, V1_normalized, get_xlim=ax_patho.get_xlim())
# Plot ConSurf
plot_ax_consurf(ax_consurf, consurf_color, get_xlim=ax_patho.get_xlim())
# Plot DSSP
dssp = df_feat.dssp.values[:-1:19]
dssp[-5:-1] = 1
# plot_ax_dssp(ax_dssp, dssp, get_xlim=(-0.5, 225))
plot_ax_dssp(ax_dssp, dssp, get_xlim=ax_patho.get_xlim())

########################## DSSP and ConSurf legends ##########################
from matplotlib import colors as mcolors

# anchor from ax_dssp
dx0, dy0, dwidth, dheight = ax_dssp.get_position().bounds

# shared geometry controls
bar_y = dy0 - 0.055          # both bars at same vertical level
bar_h = 0.020                # same bar height
title_y = -0.70              # same title offset below each bar

# ConSurf bar axis
consurf_cax = fig.add_axes([
    dx0 + 0.05 * dwidth,     # x
    bar_y,                   # y (shared)
    0.10 * dwidth,           # width
    bar_h                    # height (shared)
])
consurf_cax.set_axis_off()
# DSSP legend axis
dssp_lax = fig.add_axes([
    dx0 + 0.20 * dwidth,     # x (choose spacing you want)
    bar_y,                   # y (shared)
    0.22 * dwidth,           # width
    bar_h                    # height (shared)
])
dssp_lax.set_axis_off()

# titles aligned
consurf_cax.text(0.5, title_y, "ConSurf", ha='center', va='center', transform=consurf_cax.transAxes, fontsize=14)
dssp_lax.text(0.5, title_y, "Secondary Structure", ha='center', va='center', transform=dssp_lax.transAxes, fontsize=14)

# ConSurf colorbar (compact, centered under ax_dssp)
consurf_bar_colors = [
    [0.039215686, 0.490196078, 0.509803922],
    [0.294117647, 0.68627451, 0.745098039],
    [0.647058824, 0.862745098, 0.901960784],
    [0.843137255, 0.941176471, 0.941176471],
    [1, 1, 1],
    [0.980392157, 0.921568627, 0.960784314],
    [0.980392157, 0.784313725, 0.862745098],
    [0.941176471, 0.490196078, 0.666666667],
    [0.62745098, 0.156862745, 0.37254902],
]
consurf_cmap = mcolors.LinearSegmentedColormap.from_list("consurf_bar", consurf_bar_colors)
consurf_cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=100), cmap=consurf_cmap), cax=consurf_cax, orientation='horizontal')
consurf_cb.set_ticks([])
x_var = -0.18
x_cons = 1.23
x_mid = (x_var + x_cons) / 2

consurf_cax.text(x_var, 0.5, "Variable", ha='center', va='center', transform=consurf_cax.transAxes, fontsize=12, color='black')
consurf_cax.text(x_cons, 0.5, "Conserved", ha='center', va='center', transform=consurf_cax.transAxes, fontsize=12, color='black')

# --- proxy handle classes ---
class ArrowHandle:
    def __init__(self, color='darkblue'):
        self.color = color

class WaveHandle:
    def __init__(self, color='navy', lw=2):
        self.color = color
        self.lw = lw

# --- custom legend drawing ---
class HandlerArrow(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y = ydescent + 0.5 * height
        arr = FancyArrow(
            xdescent, y, width, 0,
            width=0.55 * height, head_width=1.1 * height, head_length=0.22 * width,
            length_includes_head=True, color=orig_handle.color, linewidth=0
        )
        arr.set_transform(trans)
        return [arr]

class HandlerWave(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x = np.linspace(xdescent, xdescent + width, 120)
        y = ydescent + 0.5 * height + 0.25 * height * np.sin(np.linspace(0, 10*np.pi, 120))
        wave = Line2D(x, y, color=orig_handle.color, lw=orig_handle.lw)
        wave.set_transform(trans)
        return [wave]

# handles in desired order: arrow, line, wave
h_arrow = ArrowHandle('darkblue')
h_line  = Line2D([0], [0], color='black', lw=1)
h_wave  = WaveHandle('navy', lw=2)
# legend drawn inside the separate axis
dssp_lax.legend(
    handles=[h_arrow, h_line, h_wave],
    labels=['Beta sheet', 'Loop', 'Helix'],
    handler_map={ArrowHandle: HandlerArrow(), WaveHandle: HandlerWave()},
    loc='center', frameon=False, fontsize=12, ncols=3, handlelength=2.2, handletextpad=0.5, columnspacing=1.2, borderpad=0.0
)
########################## DSSP and ConSurf legends ##########################


arrow_style = dict(shrink=0.01, width=1, headwidth=8, headlength=5)
dy = 0.1

def annotate_vertical_arrows(ax, positions, color, y_base, dy=0.1):
    x0, x1 = ax.get_xlim()
    for pos in positions:
        if not (x0 <= pos - 0.5 <= x1):
            continue
        pos -= 1
        ax.annotate(
            '',
            xy=(pos, y_base + dy),
            xytext=(pos, y_base - dy),
            arrowprops={**arrow_style, 'facecolor': color, 'edgecolor': color},
            annotation_clip=False
        )

annotate_vertical_arrows(
    ax_patho,
    [34, 37, 44, 44, 50, 59, 75, 75, 84, 90, 95, 143, 143, 161, 163, 179, 184, 195, 197, 202, 205, 206],
    color='red',
    y_base=ax_patho.get_ylim()[1] - 0.2,
    dy=dy,
)
annotate_vertical_arrows(
    ax_patho,
    [217, 215, 214, 210, 203, 197, 170, 170, 168, 156, 153, 127, 123, 121, 115, 114, 111, 107, 100, 83, 27, 16, 4, 4, 165],
    color='blue',
    y_base=ax_patho.get_ylim()[1] - 0.2,
    dy=dy,
)
    
x_pos = [27, 34, 197, 203]
x_pos = [v for v in x_pos if v >= ax_patho.get_xlim()[0] and v <= ax_patho.get_xlim()[1]]
for pos in x_pos:
    pos -= 1
    y_base = ax_patho.get_ylim()[0] + 0.2
    ax_patho.annotate(
        '',
        xy=(pos, y_base - dy),        # arrow tip
        xytext=(pos, y_base + dy),    # arrow start (a bit above)
        arrowprops={**arrow_style, 'facecolor': 'black', 'edgecolor': 'black'},
        annotation_clip=False
    )
    y_pos = ax_dssp.get_ylim()[1]-0.2  # top of the y-axis
    ax_dssp.annotate(
        '',
        xy=(pos, y_pos -0.1),        # arrow tip
        xytext=(pos, y_pos +0.1),    # arrow start (a bit above)
        arrowprops={**arrow_style, 'facecolor': 'black', 'edgecolor': 'black'},
        annotation_clip=False
    )
plt.subplots_adjust(top=0.9, bottom=0.14)  # leave room for ConSurf color bar
fig.savefig(
    os.path.join(FIGURE_OUTDIR, 'figure4_saturation_mutagenesis_GJB2.png'),
    dpi=300,
    bbox_inches='tight',
)
# plt.close()
plt.show()
