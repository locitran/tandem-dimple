import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, ks_2samp
import pandas as pd 
from scipy.stats import wasserstein_distance

dynamics_feat = {
    'GNM_Ventropy_full': r'Entropy$_v$', 'GNM_rmsf_overall_full': r'RMSF$_{all}$', 
    'GNM_Eigval1_full': r'$\lambda_1$', 'GNM_Eigval2_full': r'$\lambda_2$',
    'GNM_Eigval5_1_full': r'$\lambda_{5-1}$', 'GNM_SEall_full': r'SE$_{all}$',
    'GNM_SE20_full': r'SE$_{20}$', 'GNM_V1_full': r'$‖V_{1,i}‖$',
    'GNM_rankV1_full': r'rank ($‖V_{1,i}‖$)', 'GNM_V2_full': r'$‖V_{2,i}‖$',
    'GNM_rankV2_full': r'rank ($‖V_{2,i}‖$)', 'GNM_co_rank_full': r'rank ($‖C_{i,i}‖$)',
    'GNM_displacement_full': r'$‖C_{i,i}‖$', 'GNM_MC1_full': r'MC${_1}$',
    'GNM_MC2_full': r'MC${_2}$', 'ANM_effectiveness_chain': r'Effectiveness$^*$',
    'ANM_sensitivity_reduced': r'Sensitivity$^*$',
    'ANM_stiffness_chain': r'Stiffness$^*$',
    'wtBJCE': r'wtBJCE', 'deltaBJCE': r'$\Delta$BJCE'
}
structure_feat = {
    'protein_length': r'Protein Size', 
    'Rg': r'R$_g$', 'DELTA_Rg': r'$\Delta$R$_g$',
    'AG1': r'AG$_1$', 'AG3': r'AG$_3$', 'AG5': r'AG$_5$', 
    'ACR': r'ACR', 'DELTA_ACR': r'$\Delta$ACR',
    'SF1': r'SF$_1$', 'SF2': r'SF$_2$', 'SF3': r'SF$_3$',
    'loop_percent': r'%Loop', 'helix_percent': r'%Helix', 'sheet_percent': r'%Sheet',
    'Lside': r'L$_{side}$', 'deltaLside': r'$\Delta$L$_{side}$', # not availabel yet
    'IDRs': r'Disorderliness', 'dssp': r'DSSP', 'Dcom': r'D$_{com}$', 
    'SASA': r'SASA', 'DELTA_SASA': r'$\Delta$SASA', 
    'Hbond': r'N$_{H-bond}$', 'DELTA_Hbond': r'$\Delta$N$_{H-bond}$',
    'SSbond': r'N$_{SS-bond}$', 'DELTA_DSS': r'$\Delta$N$_{SS-bond}$',
}
seq_feat = {
    'wtPSIC': r'wtPSIC$^*$', 'deltaPSIC': r'$\Delta$PSIC$^*$', 
    'BLOSUM': r'BLOSUM$^*$', 'entropy': r'Entropy$^*$', 'ranked_MI': r'Ranked MI$^*$',
    'consurf': r'ConSurf', 'ACNR': r'ACNR',
    'phobic_percent': r'%Hydrophobic', 'delta_phobic_percent': r'$\Delta$%Hydrophobic',
    'charge': r'Charge', 'deltaCharge': r'$\Delta$Charge', 
    'polarity': r'Polarity', 'deltaPolarity': r'$\Delta$Polarity', 
    'charge_pH7': r'Charge$_{pH7}$', 'DELTA_charge_pH7': r'$\Delta$Charge$_{pH7}$',
}