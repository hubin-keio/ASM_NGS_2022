# Statistical analysis of the binding data from Starr et al. The input data
# is generated from DMS mutation data with all synonymous mutations removed
# and nonsynonymous mutations and wildtype kd averages.

# It is generated by this command:
# 
# grep '^>' ASM_NGS_2022/seqs/binding_Kds.fasta \
# |awk '{print substr($0, 2)}'|sed -e "s/ /, /g" \
# > ASM_NGS_2022/datasets/mutation_binding_Kds.csv
#
# It has this format:
# A22C_R127G_E141D_L188V, log10Ka=8.720000267028809, ka_sd=nan
# N13F, log10Ka=10.358182907104492, ka_sd=0.05153989791870117
# V71K_P149L_N157T, log10Ka=6.0, ka_sd=nan
# A18V_T148S_H189Y, log10Ka=10.135000228881836, ka_sd=0.02121301367878914
# T63D_A89N, log10Ka=9.609999656677246, ka_sd=nan
# P82S_P96L_T148D_F160R, log10Ka=6.0, ka_sd=nan
# K48G_L60S, log10Ka=8.569999694824219, ka_sd=nan
# R16C_R27G_K94D_F99W_N107M, log10Ka=6.960000038146973, ka_sd=1.357645034790039
# wildtype, log10Ka=10.792753219604492, ka_sd=0.11116741597652435
# P7L_D90F, log10Ka=10.029999732971191, ka_sd=nan


import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def load_data(csv: str) -> pd.DataFrame:
    """ Load binding csv file and return a dataframe
    """
    try:
        data = pd.read_csv(csv, sep=',',
                           names=['mutation', 'log10Ka', 'log10Ka_SD'])
    except FileNotFoundError as err:
        print(f'File not found error: {csv}.', file=sys.stderr)
        sys.exit()

    data['log10Ka'] = data['log10Ka'].str.slice(start=len('log10Ka=')+1)
    data['log10Ka_SD'] = data['log10Ka_SD'].str.slice(start=len('ka_sd=')+1)

    # data['log10Ka'] = pd.to_numeric(data['log10Ka'])
    # data['log10Ka_SD'] = pd.to_numeric(data['log10Ka_SD'])

    return data.astype({'mutation': str,
                        'log10Ka': np.float32,
                        'log10Ka_SD': np.float32})

def plot_stats(df: pd.DataFrame, colname, ref_point, save_under) -> None:
    """Take a dataframe and plot histogram

    colname: name of the column in dataframe for plotting

    ref_point: reference value from wildtype, which will be used to draw
    a vertical line with a label indicating its value.

    """
    sns.set_theme()
    sns.set_context('talk')
    
    plt.figure()
    g = sns.histplot(data=data, x=colname, stat='percent', bins=10, kde=True)
    g.axvline(ref_point)

    # percentage of mutations with increased
    increased_pct = data[data[colname] > wt_log10Ka].shape[0]/data.shape[0] * 100
    g.text(7, 20, f'{increased_pct:.2f}%: increased Ka', fontsize=20)

    try:
        plt.savefig(save_under)
    except:
        print(f'Cannot save file {save_under}.', file=sys.stderr)
        sys.exit()

def get_increased_kds(df: pd.DataFrame, colname: str, ref: float) -> list:
    """Get the mutant labels for all mutations showing increased Kd.
    
    df: dataframe with mutant and kd
    colname: column name of interest
    ref: reference (wildtype) kd
    """
    mut = data[data[colname] > wt_log10Ka]
    mut = list(mut['mutation'])

    # TODO: get kmer analysis for increased kd.
    # def build_
    



if __name__ == '__main__':
    # Directories and inputput output file settings
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    DATA_DIR = os.path.join(ROOT_DIR, 'datasets')
    PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
    
    binding_csv = os.path.join(DATA_DIR, 'mutation_binding_Kds.csv')
    hist_plot_file = os.path.join(PLOTS_DIR, 'mutation_binding_Kds.png')
    wt_log10Ka=10.792753219604492
    
    data = load_data(binding_csv)
    plot_stats(data, 'log10Ka', wt_log10Ka, hist_plot_file)

    mut_inc_kd  = get_increased_kds(data, 'log10Ka', wt_log10Ka)
    
    

