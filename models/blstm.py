""" Bidirectional LSTM model
"""
import os
import re
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

class BindignDataset(Dataset):
    """Binding dataset
    """
    def __init__(self, csv_file: str, refseq: str):
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
            kmers = self._get_kmers(seq)
            return kmers, self.log10_ka[idx]
        except IndexError:
            print(f'List index out of range: {idx}, length: {len(self.labels)}.',
                  file=sys.stderr)
            sys.exit(1)

    def _label_to_seq(self, label: str) -> str:
        """Genreate sequence based on reference sequence and mutation label."""
        seq = self.refseq
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

    def _get_kmers(self, seq: str, k_size: int=6, stride: int=2) -> list:
        """Get Kmers of a sequence with kmer and stride length defined in class.

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




if __name__=='__main__':

    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    ROOT_DIR = os.path.join(ROOT_DIR, '..')
    DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

    DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'


    csv_file = os.path.join(DATA_DIR, 'mutation_binding_Kds.csv')
    REFSEQ = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
    tests = BindignDataset(csv_file, REFSEQ)
