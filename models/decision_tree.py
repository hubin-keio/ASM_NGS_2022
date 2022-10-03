""" Random forest and decision tree models

Model MSE=2.8790654158926317
"""
import os
import re
import sys
import pickle
from datetime import date

import torch
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import Dataset
from torch.utils.data import random_split


class Transformer:
    """Transforms input sequence to features"""

    def __init__(self, method: str, **kwargs):
        if method == 'glove_kmer':
            assert 'glove_csv' in kwargs.keys()
            self.glove_csv = kwargs['glove_csv']
            self.method = 'glove_kmer'
            self._load_glove_vect()

        elif method == 'one_hot':
            self.method = 'one_hot'

        else:
            print('unimplemented transform method', file=sys.stderr)
            sys.exit(1)

    def _load_glove_vect(self):
        """load Glove word vector file"""
        self.glove_kmer_dict = {}
        with open(self.glove_csv, 'r', encoding='utf-8') as fh:
            for k_line in fh.readlines():
                k_line = k_line.rstrip().split()
                kmer = k_line[0]
                vects = k_line[1:]
                vects = torch.tensor([float(x) for x in vects], requires_grad=True)
                self.glove_kmer_dict.update({kmer: vects})
            self.glove_vec_size = len(vects)

    @staticmethod
    def get_kmer(seq: str, k_size: int=6, stride: int=2) -> list:
        """ Transform seq to kmers

        seq: input sequence
        k_size: kmer size, default to 6
        stride: stride for sliding widow, default to 2
        """
        seq_len = len(seq)
        kmers = []

        for i in range(0, seq_len, stride):
            if i + k_size >= seq_len+1:
                break
            kmers.append(seq[i:i + k_size])
        return kmers

    def embed(self, seq) -> torch.Tensor:
        """Embed sequence feature using a defind method.

        seq: sequence of input.
        """
        if self.method == 'glove_kmer':
            return self._embed_glove(seq)

        print(f'Undefined embedding method: {self.method}', file=sys.stderr)
        sys.exit(1)

    def _embed_one_hot(self, seq: str) -> torch.Tensor:
        """Embed sequence feature using one-hot."""
        kmers = self.get_kmer(seq)
        embedding = torch.zeros(len(kmers), self.glove_vec_size)
        for idx, kmer in enumerate(kmers):
            try:
                vec = self.glove_kmer_dict[kmer]
                embedding[idx] = vec
            except KeyError:
                print(f'Unknown kmer: {kmer}', file=sys.stderr)
                sys.exit(1)
        return embedding

    def _embed_glove(self, seq: str) -> torch.Tensor:
        """Embed sequence feature using Glove vectors."""
        kmers = self.get_kmer(seq)
        embedding = torch.zeros(len(kmers), self.glove_vec_size)
        for idx, kmer in enumerate(kmers):
            try:
                vec = self.glove_kmer_dict[kmer]
                embedding[idx] = vec
            except KeyError:
                print(f'Unknown kmer: {kmer}', file=sys.stderr)
                sys.exit(1)
        return embedding


class BindignDataset(Dataset):
    """Binding dataset."""
    def __init__(self, csv_file: str, refseq: str, transformer: Transformer):
        """
        Load sequence label and binding data from csv file and generate full
        sequence using the label and refseq.

        csv_file: a csv file with sequence label and kinetic data.
        refseq: reference sequence (wild type sequence)
        transformer: a Transfomer object for embedding features.

        The csv file has this format (notice that ka_sd is not always a number):

        A22C_R127G_E141D_L188V, log10Ka=8.720000267028809, ka_sd=nan
        N13F, log10Ka=10.358182907104492, ka_sd=0.05153989791870117
        V71K_P149L_N157T, log10Ka=6.0, ka_sd=nan
        """
        def _load_csv():
            labels = []
            log10_ka = []
            try:
                with open(csv_file, 'r') as fh:
                    for line in fh. readlines():
                        [label, affinity, _] = line.split(',')
                        affinity = np.float32(affinity.split('=')[1])
                        labels.append(label)
                        log10_ka.append(affinity)
            except FileNotFoundError:
                print(f'File not found error: {csv_file}.', file=sys.stderr)
                sys.exit(1)
            return labels, log10_ka

        self.csv_file = csv_file
        self.refseq = list(refseq)
        self.labels, self.log10_ka = _load_csv()
        self.transformer = transformer

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            label = self.labels[idx]
            seq = self._label_to_seq(label)
            features = self.transformer.embed(seq)

            return label, features, self.log10_ka[idx]
        except IndexError:
            print(f'List index out of range: {idx}, length: {len(self.labels)}.',
                  file=sys.stderr)
            sys.exit(1)

    def _label_to_seq(self, label: str) -> str:
        """Genreate sequence based on reference sequence and mutation label."""
        seq = self.refseq.copy()
        p = '([0-9]+)'
        if '_' in label:
            for mutcode in label.split('_'):
                [ori, pos, mut] = re.split(p, mutcode)
                pos = int(pos)-1    # use 0-based counting
                assert self.refseq[pos].upper() == ori
                seq[pos] = mut.upper()
            seq = ''.join(seq)
            return seq

        if label=='wildtype':
            return ''.join(seq)

        [ori, pos, mut] = re.split(p, label)
        pos = int(pos)-1    # use 0-based counting
        assert self.refseq[pos] == ori
        seq[pos] = mut.upper()
        seq = ''.join(seq)
        return seq

def prep_data(dataset: BindignDataset, num_features: int, frac: float=1.0):
    """Transform a BindingDataSet to numpy.ndarray

    Args:
    dataset: a BindignDataset object
    num_features: number of features.
    frac: fraction of the dataset being transformed. Default to 1.0 or all the data.

    Returns:
    L: labels of input
    X: input features
    y: output
    """
    total_entries = int(np.floor(len(dataset)* frac))
    labels = []

    X = np.ndarray((total_entries, num_features), dtype=float )
    y = np.ndarray(total_entries)

    for i in range(total_entries):
        (label, feature, y[i]) = dataset[i]
        labels.append(label)
        feature = feature.detach().numpy()
        X[i]= feature.ravel()

    return labels, X, y

def train_decision_tree(max_depth: int,
                        train_X: np.ndarray, train_y: np.ndarray,
                        test_X: np.ndarray, test_y: np.ndarray) -> np.ndarray:
    """Run decision tree regressor.

    Args:
    max_depth: maximmum depth of the tree.
    train_X: training features in ndarray.
    train_y: training output.
    test_X: testing features in ndarray.
    test_y: testing output.

    Return:
    regr: a trained DecisionTreeRegressor object
    se: a numpy ndarray object of squared error of test data set.
    """
    regr = DecisionTreeRegressor(max_depth=max_depth)
    regr.fit(train_X, train_y)
    pred = regr.predict(test_X)
    squared_error = (pred - test_y)**2
    return regr, squared_error

def save_model(model: DecisionTreeRegressor, save_as: str):
    """Save a decision tree model using Pickle.

    model: a DecisionTreeRegressor object
    save_as: file name for saveing the model.
    """
    with open(save_as, 'wb') as fh:
        pickle.dump(model, fh)



if __name__=='__main__':
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    ROOT_DIR = os.path.abspath(ROOT_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, 'datasets')


    KD_CSV = os.path.join(DATA_DIR, 'mutation_binding_Kds.csv')
    REFSEQ = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
    GLOVE_CSV = os.path.join(ROOT_DIR, 'glove/glove_dms_k6_s2.vector.txt')

    # Dataset split and dataloader
    glove_transformer = Transformer('glove_kmer', glove_csv=GLOVE_CSV)
    data_set = BindignDataset(KD_CSV, REFSEQ, glove_transformer)
    TRAIN_SIZE = int(0.8 * len(data_set))  # 80% goes to training.
    TEST_SIZE = len(data_set) - TRAIN_SIZE
    train_set, test_set = random_split(data_set, (TRAIN_SIZE, TEST_SIZE))

    NUM_FEATURES = train_set[0][1].shape[0] * train_set[0][1].shape[1]  # 98*50 = 4900
    FRAC = 1                    # Fraction of data goes to model. 1 for 100%, smaller number for debugging.
    train_labels, train_X, train_y = prep_data(train_set, NUM_FEATURES, FRAC)
    test_labels, test_X, test_y = prep_data(test_set, NUM_FEATURES, FRAC)
    tree_regr, se = train_decision_tree(5, train_X, train_y, test_X, test_y)
    mse = se.mean()


    model_file = f'decision_tree_train_{int(np.floor(TRAIN_SIZE*FRAC))}_test_{int(np.floor(TEST_SIZE*FRAC))}_{date.today()}'
    model_file = os.path.join(ROOT_DIR, f'models/{model_file}')
    print(f'Model MSE={mse}')

    save_model(tree_regr, model_file)
