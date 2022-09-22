import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_tsne(vector_txt: str, vec_size: int, save_as: str):
    """Load vacabulary vectors from txt file and plot in tsne

    vector_txt: text file containing the vector.
    vec_size: dimension size of the vector.
    save_as: file to save the plot.

    One line of the vacabulary vector text looks like this:

    PATVCG 1.289958 -1.296375 0.495700 3.374617 -1.649096 0.082823 0.521626 -1.429745 -0.277816 -0.232896 -0.407372 1.150546 0.446664 -0.193387 -0.406607 0.663098 -1.034090 0.376743 -0.785020 -0.990868 -0.007567 -0.484719 0.812402 0.203502 -0.437623 -0.416958 -0.140471 1.397378 0.427073 -0.479609 -0.834507 -1.248022 -0.594637 -0.384299 0.255136 -0.594307 0.661710 0.027789 1.501607 -0.171032 0.242372 1.255432 -0.891688 -1.059083 0.338175 0.903765 1.459189 1.051903 0.357112 -0.314426
    """
    try:
        usecols = [x for x in range(1, vec_size+1)]
        vecs = np.loadtxt(vector_txt, delimiter=' ', usecols=usecols)
    except:
        print(f'Cannot load data from {vector_txt}.', file=sys.stderr)
        sys.exit(1)

    model = TSNE(n_components=2,
                 init='pca',
                 learning_rate='auto',
                 random_state=0)
    vecs_2d = model.fit_transform(vecs)

    fig = plt.figure(figsize=(8, 6))    
    sns.scatterplot(x=vecs_2d[:,0], y=vecs_2d[:,1], marker='.')
    try:
        fig.savefig(save_as)
    except:
        print(f'Cannot save file {save_as}.', file=sys.stderr)
        sys.exit()

def plot_voc(voc_txt: str, save_as: str):
    """Plot vocabulary usage frequency

    voc_txt: two column vocab file.
    save_as: file to save the plot.
    """
    try:
        voc = np.loadtxt(voc_txt, delimiter=' ', usecols=1)
    except:
        print(f'Cannot load data from {voc_txt}.', file=sys.stderr)
        sys.exit(1)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(voc)
    plt.yscale('log')
    plt.ylabel('Vocabulary Frequency')

    try:
        gig.savefig(save_as)
    except:
        print(f'Cannot save file {save_as}.', file=sys.stderr)
        sys.exit()
    
    

    
    
if __name__ == '__main__':
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')

    sns.set_theme()
    sns.set_context('talk')

    vector_txt = os.path.join(ROOT_DIR, 'glove/glove_dms_k6_s2.vector.txt')
    tsne_plot_file = os.path.join(PLOTS_DIR, 'glove_tsne.png')
    vec_size = 50
    plot_tsne(vector_txt, vec_size, tsne_plot_file)
    
    voc_txt = os.path.join(ROOT_DIR, 'glove/glove_dms_k6_s2.vocab')
    voc_plot_file = os.path.join(PLOTS_DIR, 'glove_voc_frequency.png')    
    plot_voc(voc_txt, voc_plot_file)
