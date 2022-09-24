""" Bidirectional LSTM model
"""
import os
import re
import sys
import numpy as np
import torch
# from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# class GloveKmer:
#     """Transform a sequence to Glove kmer features.
#     """
#     def __init__(self, kmer_csv: str):
#         self.csv = kmer_csv


#     def __call__(self, seq: str):
#         return seq

class Transformer:
    """Transforms input sequence to features"""


    def __init__(self, method: str, **kwargs):
        if method == 'glove_kmer':
            assert 'glove_csv' in kwargs.keys()
            self.glove_csv = kwargs['glove_csv']
            self.method = 'glove_kmer'
            self._load_glove_vect()

        else:
            print('unimplemented transform method', file=sys.pystderr)
            sys.exit(1)
        self._load_glove_vect()

    # def __call__(self, seq: str):
    #     if self.method == 'glove_kmer':
    #         glove_kmer(seq)

    def _load_glove_vect(self):
        """load Glove word vector file"""
        self.glove_kmer_dict = {}
        with open(self.glove_csv, 'r') as fh:
            for k_line in fh.readlines():
                k_line = k_line.rstrip().split()
                kmer = k_line[0]
                vects = k_line[1:]
                vects = torch.FloatTensor([float(x) for x in vects])
                self.glove_kmer_dict.update({kmer: vects})
            self.glove_vec_size = len(vects)

    def _get_kmer(self, seq: str, k_size: int=6, stride: int=2) -> list:
        """ Transform seq to tensor using glove vectors

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

    def _get_glove_embedding(self, seq: str) -> torch.Tensor:
        """Get the tensor representation of a sequence using Glove vectors."""
        kmers = self._get_kmer(seq)
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
    """Binding dataset
    """
    def __init__(self, csv_file: str, refseq: str, ):
        """
        Load sequence label and binding data from csv file and generate full
        sequence using the label and refseq.

        csv_file: a csv file with sequence label and kinetic data.
        refseq: reference sequence (wild type sequence)

        The csv file has this format (notice that ka_sd :

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

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            seq = self._label_to_seq(self.labels[idx])
            # kmers = self._get_kmers(seq)
            # return kmers, self.log10_ka[idx]

            return GloveKmer(seq), self.log10_ka[idx]
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

    # def _get_kmers(self, seq: str, k_size: int=6, stride: int=2) -> list:
    #     """Get Kmers of a sequence with kmer and stride length defined in class.

    #     seq: input sequence
    #     k_size: kmer size, default to 6
    #     stride: stride for sliding widow, default to 2
    #     """

    #     seq_len = len(seq)
    #     kmers = []

    #     for i in range(0, seq_len, stride):
    #         if i + k_size >= seq_len+1:
    #             break
    #         kmers.append(seq[i:i + k_size])
    #     return kmers

# class BLSTM(nn.Module):
#     """Bidirectional LSTM

#     """
#     def __init__(self, input_size, hidden_size, hidden_size_2, num_layers):
#         super(RBDLSTM, self).__init__()
#         self.input_size = input_size  # input size
#         self.hidden_size = hidden_size  # hidden state
#         self.num_layers = num_layers  # number of layers

#         self.lstm = nn.LSTM(input_size=input_size,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True)
#         self.dense1 = nn.Linear(hidden_size, hidden_size_2)  # first connected layer
#         self.out = nn.Linear(hidden_size_2, 1)  # connected last layer

#         self.relu = nn.ReLU()  # set activation layer ============> Is this needed? BH 1/29/22

#     def forward(self, x):
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # hidden_state
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # internal state

#         # call lstm with input, hidden state, and internal state
#         output, (hn, cn) = self.lstm(x, (h_0, c_0))
#         # reshape hdden layer for activate layer for out.shape = (seq_len, hidden_size)
#         out = output.view(-1, self.hidden_size).to(device)
#         logits = self.relu(out)
#         logits = self.dense1(logits)
#         logits = self.relu(logits)
#         prediction = self.out(logits)
#         return prediction


# def run_lstm(train_loader, test_loader, model, n_epoch: int=10):


if __name__=='__main__':

    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    ROOT_DIR = os.path.abspath(ROOT_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

    kd_csv = os.path.join(DATA_DIR, 'mutation_binding_Kds.csv')
    REFSEQ = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
    glove_csv = os.path.join(ROOT_DIR, 'glove/glove_dms_k6_s2.vector.txt')


    # Run setup
    DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
    BATCH_SIZE = 64

    # Dataset split and dataloader
    data_set = BindignDataset(kd_csv, REFSEQ)
    TRAIN_SIZE = int(0.8 * len(data_set))  # 80% goes to training.
    TEST_SIZE = len(data_set) - TRAIN_SIZE
    train_set, test_set = random_split(data_set, (TRAIN_SIZE, TEST_SIZE))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    transformer = Transformer('glove_kmer', glove_csv=glove_csv)
