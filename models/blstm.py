""" Bidirectional LSTM model
"""
import os
import re
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class Transformer:
    """Transforms input sequence to features"""

    def __init__(self, method: str, **kwargs):
        if method == 'glove_kmer':
            assert 'glove_csv' in kwargs.keys()
            self.glove_csv = kwargs['glove_csv']
            self.method = 'glove_kmer'
            self._load_glove_vect()
        else:
            print('unimplemented transform method', file=sys.stderr)
            sys.exit(1)

    def _load_glove_vect(self):
        """load Glove word vector file"""
        self.glove_kmer_dict = {}
        with open(self.glove_csv, 'r') as fh:
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
            # kmers = self._get_kmers(seq)
            # return kmers, self.log10_ka[idx]

            return label, features, self.log10_ka[idx]
        except IndexError:
            print(f'List index out of range: {idx}, length: {len(self.labels)}.',
                  file=sys.stderr)
            sys.exit(1)

    # def __repr__(self):
    #     repr = f'Dataset object using {self.transformer.method} with {len(self)} entries.'
    #     return repr

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


class BLSTM(nn.Module):
    """Bidirectional LSTM
    """
    def __init__(self,
                 batch_size,         # Batch size of the tensor
                 lstm_input_size,    # The number of expected features. For GloVe, it is 50.
                 lstm_hidden_size,   # The number of features in hidden state h.
                 lstm_num_layers,    # Number of recurrent layers in LSTM.
                 lstm_bidirectional, # Bidrectional LSTM.
                 fcn_hidden_size,    # The number of features in hidden layer of CN.
                 device):            # Device ('cpu' or 'cuda')
        super().__init__()
        self.batch_size = batch_size
        self.device = device


        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            batch_first=True)               # batch first

        # FCN fcn layer
        if lstm_bidirectional:
            self.fcn = nn.Linear(2 * lstm_hidden_size, fcn_hidden_size)
        else:
            self.fcn = nn.Linear(lstm_hidden_size, fcn_hidden_size)

        # FCN output layer
        self.out = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        # Initialize hidden and cell states to zeros.
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(num_directions * self.lstm.num_layers,
                          x.size(0),
                          self.lstm.hidden_size).to(self.device)
        c_0 = torch.zeros(num_directions * self.lstm.num_layers,
                          x.size(0),
                          self.lstm.hidden_size).to(self.device)

        # call lstm with input, hidden state, and internal state
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n.detach()
        c_n.detach()
        lstm_final_out = lstm_out[:,-1,:]  # last hidden state from every batch. size: N*H_cell
        lstm_final_state = lstm_final_out.to(self.device)
        fcn_out = self.fcn(lstm_final_state)
        prediction = self.out(fcn_out)
        return prediction


def run_lstm(model: BLSTM,
             train_loader: DataLoader,
             test_loader: DataLoader,
             n_epochs: int,
             batch_size: int,
             device: str):
    """Run LSTM model

    model: BLSTM,
    train_loader: training set dataLoader,
    test_loader: test det dataLoader,
    n_epochs: number of epochs
    batch_size: batch size
    device: 'gpu' or 'cpu'
    """

    L_RATE = 1e-5               # learning rate
    model = model.to(device)

    loss_fn = nn.MSELoss(reduction='sum').to(device)  # MSE loss with sum
    optimizer = torch.optim.SGD(model.parameters(), L_RATE)  # SGD optimizer

    train_loss_history = []
    test_loss_history = []
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        test_loss = 0

        for batch, (label, feature, target) in enumerate(train_loader):
            optimizer.zero_grad()
            feature, target = feature.to(device), target.to(device)
            pred = model(feature).flatten()
            batch_loss = loss_fn(pred, target)        # MSE loss at batch level
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()


        for batch, (label, feature, target) in enumerate(test_loader):
            feature, target = feature.to(device), target.to(device)
            with torch.no_grad():
                pred = model(feature).flatten()
                batch_loss = loss_fn(pred, target)
                test_loss += batch_loss.item()

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        if epoch < 11:
            print(f'Epoch {epoch}, Train MSE: {train_loss}, Test MSE: {test_loss}')
        elif epoch%10 == 0:
            print(f'Epoch {epoch}, Train MSE: {train_loss}, Test MSE: {test_loss}')

    return train_loss_history, test_loss_history


def plot_history(train_losses: list, n_train: int, test_losses: list,
                 n_test: int, save_as: str):
    """Plot training and testing hisotry per epoch

    train_losses: a list of per epoch error from the training set
    n_train: number of items in the training set
    test_losses: a list of per epoch error from the test set
    n_test: number of items in the test set
    """
    history_df = pd.DataFrame(list(zip(train_losses, test_losses)),
                              columns = ['training','testing'])

    history_df['training'] = history_df['training']/n_train  # average error per item
    history_df['testing'] = history_df['testing']/n_test

    print(history_df)

    sns.set_theme()
    sns.set_context('talk')

    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=history_df, x=history_df.index, y='training', label='training')
    sns.scatterplot(data=history_df, x=history_df.index, y='testing', label='tesing')
    fig.savefig(save_as + '.png')
    history_df.to_csv(save_as + '.csv')




if __name__=='__main__':

    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    ROOT_DIR = os.path.abspath(ROOT_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, 'datasets')


    KD_CSV = os.path.join(DATA_DIR, 'mutation_binding_Kds.csv')
    REFSEQ = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
    GLOVE_CSV = os.path.join(ROOT_DIR, 'glove/glove_dms_k6_s2.vector.txt')




    # Run setup
    DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
    BATCH_SIZE = 20
    N_EPOCHS = 100
    LSTM_INPUT_SIZE = 50        # lstm_input_size
    LSTM_HIDDEN_SIZE = 50       # lstm_hidden_size
    LSTM_NUM_LAYERS = 1         # lstm_num_layers
    LSTM_BIDIRECTIONAL = True   # lstm_bidrectional
    FCN_HIDDEN_SIZE = 20        # fcn_hidden_size

    # Dataset split and dataloader
    glove_transformer = Transformer('glove_kmer', glove_csv=GLOVE_CSV)
    data_set = BindignDataset(KD_CSV, REFSEQ, glove_transformer)
    TRAIN_SIZE = int(0.8 * len(data_set))  # 80% goes to training.
    TEST_SIZE = len(data_set) - TRAIN_SIZE
    train_set, test_set = random_split(data_set, (TRAIN_SIZE, TEST_SIZE))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    labels, train_features , train_targets = next(iter(train_loader))

    model = BLSTM(BATCH_SIZE,
                  LSTM_INPUT_SIZE,
                  LSTM_HIDDEN_SIZE,
                  LSTM_NUM_LAYERS,
                  LSTM_BIDIRECTIONAL,
                  FCN_HIDDEN_SIZE,
                  DEVICE)
    train_losses, test_losses = run_lstm(model, train_loader, test_loader, N_EPOCHS, BATCH_SIZE, DEVICE)
    model_result = f'blstm_train_{TRAIN_SIZE}_test_{TEST_SIZE}_{date.today()}'
    model_result = os.path.join(ROOT_DIR, f'plots/{model_result}')
    plot_history(train_losses, TRAIN_SIZE, test_losses, TEST_SIZE, model_result)
