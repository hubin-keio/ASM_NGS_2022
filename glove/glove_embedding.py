# GloVe embedding for DMSML

# This script utilize the GloVe package (https://nlp.stanford.edu/projects/glove/) to
# collect the co-occurance statistics of k-mers used by mutated sequences in a DMS study.
#
# 1. Generate full sequences of the mutated RBD and save it to fasta file.
# 2. Generate kmer embeddings from all the mutated RBD sequences using GloVe.
import os
import re
import sys
import shlex
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

def df_to_fasta(df: pd.DataFrame, ref_seq: str, output_fa: str, msg=True) -> None:
    """Convert a dms dataframe loaded from CSV to a fasta file. 

    Keyword argguments:
    df      -- data frame of dms data with columns:'variant_class', 'as_substitutions', 'log10Ka'
    ref_seq -- reference sequence
    output  -- output fasta file name
    msg     -- print summary after execution.

    Example fasta file (only two entries are shown):
    >A22C_R127G_E141D_L188V log10Ka=8.720000267028809
    NITNLCPFGEVFNATRFASVYCWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDL
    CFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYN
    YLYRLFGKSNLKPFERDISTDIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRV
    VVLSFELVHAPATVCGPKKST

    >V71K_P149L_N157T log10Ka=6.0
    NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDL
    CFTNVYADSFKIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYN
    YLYRLFRKSNLKPFERDISTEIYQAGSTLCNGVEGFTCYFPLQSYGFQPTNGVGYQPYRV
    VVLSFELLHAPATVCGPKKST
    """

    def label_to_seq(ref_seq: str, label: str) -> str:
        if label=='wildtype':
            return ref_seq
        else:
            p = '([0-9]+)'
            seq = list(ref_seq)
            for mutcode in label.split():
                [ori, pos, mut] = re.split(p, mutcode)
                pos = int(pos)-1    # use 0-based counting
                assert ref_seq[pos].upper() == ori
                seq[pos] = mut.upper()
            seq = ''.join(seq)
            return seq

    def print_seq_pos(seq: str):
        seq_len = len(seq)
        for i in range(0, len(seq)//10):
            print(i*10+1, '.'*8, (i+1)*10, seq[i*10:(i+1)*10])
        if seq_len % 10 != 0:
            print((i+1)*10 + 1, ':', seq_len, seq[(i+1)*10:])

    # df.reset_index()

    try:
        fa_h = open(output_fa, 'w')
    except OSError:
        print(f'Could not write to {output_fa}.', file=sys.stderr)
        sys.exit()
    with fa_h:
        for idx, row in df.iterrows():
            seq_id = row['aa_substitutions']
            seq = label_to_seq(ref_seq, seq_id)
            if seq_id.count(' ')>0:
                seq_id = '_'.join(seq_id.split())
            record = SeqRecord(Seq(seq),
                               id=seq_id,
                               description=f"log10Ka={row['log10Ka']} ka_sd={row['ka_sd']}")
            SeqIO.write(record, fa_h, 'fasta')
            if msg:
                print(idx, seq_id)
                print_seq_pos(seq)
    return None

def dms_to_df(csv: str, msg=True) -> pd.DataFrame:
    """Read a csv file with a column of mutation labels, and write all nonsynonymous
    mutations with complete sequence (e.g. exclude stop mutation) to a fasta file. All
    mutations with missing kinetic meassurements (NAs) are removed during the process.
    When multilple measurements were taken for the same mutations, the average values
    are assigned.

    Keyword argguments:
    csv     -- csv input file name
    ref_seq -- reference sequence
    output  -- output fasta file name
    msg     -- print summary after execution.
    """
    columns = ['log10Ka', 'variant_class', 'aa_substitutions']
    
    data = pd.read_csv(csv, usecols=columns)
    data = data[data['log10Ka'].notna()]  # remove rows with no log10Ka

    # use only nonsynonymous mutations or wildtype
    data = data.loc[data['variant_class'].isin(['1 nonsynonymous',
                                                '>1 nonsynonymous',
                                                'wildtype'])]

    data = data.astype({'variant_class': str,
                        'aa_substitutions': str,
                        'log10Ka': np.float32})

    # use the average of repeated strains
    unique_aa = data['aa_substitutions'].unique()
    ka_df = pd.DataFrame(columns=columns)
    ka_df.insert(1, 'ka_sd', np.NAN)  # add standard error column

    for uaa in tqdm(unique_aa, desc=f'Loading {len(unique_aa)} unique mutations: '):
        uaa_rows = data.loc[data['aa_substitutions'] == uaa]
        if uaa_rows.shape[0] == 1:  # one measurement only
            ka_df = ka_df.append(uaa_rows, ignore_index=True)
        else:                   # multiple measurements
            tmp_df = pd.DataFrame({'log10Ka': uaa_rows['log10Ka'].mean(),
                                   'ka_sd': uaa_rows['log10Ka'].std(),
                                   'variant_class': uaa_rows['variant_class'].values[0],
                                   'aa_substitutions': uaa_rows['aa_substitutions'].values[0]},
                                  index=[0])
            ka_df = pd.concat([ka_df, tmp_df], ignore_index=True)                                  
                                  
        data.drop(uaa_rows.index, inplace=True)

    ka_df.replace({'aa_substitutions': 'nan'}, 'wildtype', inplace=True)

    return ka_df


class GloveEncoder(object):
    'A callable GloVE word embedder class.'

    def __init__(self, fa: str, k_size: int, stride: int, output_base: str):
        """
        fa: fasta file 
        k_size: k-mer size
        stride: stride when slicing k-mers
        output_base: output file name base
        """
        assert os.access(fa, os.R_OK)
        self.fa = fa
        self.k_size = k_size
        self.stride = stride
        self.output_base = output_base
        self.corpus_f = '_'.join([self.output_base,
                                f'k{self.k_size}',
                                f's{self.stride}']) + '.corpus'
        self.vocab_f = self.corpus_f.replace('.corpus', '.vocab')
        self.cooccur_f = self.corpus_f.replace('.corpus', '_cooccurance.bin')
        self.shuffle_f = self.corpus_f.replace('.corpus', '_cooccurance_shuffled.bin')
        self.vector_f = self.corpus_f.replace('.corpus', '.vector')

    def getKmers(self, seq: str) -> list:
        seq_len = len(seq)
        kmers = []

        for i in range(0, seq_len, self.stride):
            if i + self.k_size >= seq_len+1:
                break
            kmers.append(seq[i:i+self.k_size])
        return kmers


    def write_corpus(self, msg=True) -> None:
        records = SeqIO.parse(self.fa, 'fasta')
        try:
            output = open(self.corpus_f, 'w')
        except OSError:
            print(f'Could not open file {corpus} to write.', file=sys.stderr)
            sys.exit()

        with output:
            for rec in records:
                tag = rec.id
                seq = str(rec.seq).upper()  # only use upper cases
                kmers = self.getKmers(seq)
                output.write(' '.join(kmers))
                output.write(' ')   # space between records.
        if msg:
            print(f'Wrote corpus to {self.corpus_f}.')
        return None
        

    # GloVe step 1: Count vocabulary
    def count_vocab(self, msg=True) -> None:
        cmd = f'vocab_count -min-count 1 -verbose 2'
        try:
            corpus = open(self.corpus_f, 'r')
            vocab = open(self.vocab_f, 'w')
        except OSError:
            print(f'Cannot access {self.corpus_f} or {self.vocab_f}.', file=sys.stderr)
            sys.exit()

        with corpus:
            with vocab:
                try:
                    p = subprocess.run(shlex.split(cmd),
                                       stdin=corpus,
                                       stdout=vocab,
                                       stderr=subprocess.PIPE)
                except FileNotFoundError:
                    print(f'Cannot find vocab_count in the GloVe package.', file=sys.stderr)
        if msg:
            msg = p.stderr.decode().split('\x1b[0G')[1]
            print(msg)
        return None            

    # GlovE step 2: calculate co-occurance matrix
    def calc_cooccur(self, msg=True) -> None:
        'Run cooccur from the GloVe package.'
        
        cmd = f'cooccur -memory 4.0 -vocab-file {self.vocab_f} -verbose 2 -window-size 15'
        try:
            corpus_f = open(self.corpus_f, 'r')
            cooccur_f = open(self.cooccur_f, 'w')
        except OSError:
            print('Cannot access {self.corpus_f} or {self.cooccur_f}.', file=sys.stderr)
            sys.exit()
        with corpus_f:
            with cooccur_f:
                try:
                    p = subprocess.run(shlex.split(cmd),
                                       stdin=corpus_f,
                                       stdout=cooccur_f,
                                       stderr=subprocess.PIPE)
                except FileNotFoundError:
                    print(f'Cannot find coocur in the GloVe package.', file=sys.stderr)
        if msg:
            msg = p.stderr.decode()
            print(msg)
        return None


    # GloVe step 3: Shuffle sequences
    def shuffle(self, msg=True) -> None:
        'Run shuffle from the GloVe package.'
        
        cmd = f'shuffle -memory 16.0 -verbose 2'
        try:
            cooccur_f = open(self.cooccur_f, 'r')
            shuffle_f = open(self.shuffle_f, 'w')
        except OSError:
            print(f'Cannot access {self.cooccur_f} or {self.shuffle_f}.', file=sys.stderr)
            sys.exit()
        with cooccur_f:
            with shuffle_f:
                try:
                    p = subprocess.run(shlex.split(cmd),
                                       stdin=cooccur_f,
                                       stdout=shuffle_f,
                                       stderr=subprocess.PIPE)
                except FileNotFoundError:
                    print(f'Cannot find shuffle in the GloVe package.', file=sys.stderr)
        if msg:
            msg = p.stderr.decode()
            print(msg)
        return None


    # GloVe step 4: GloVE training and embedding generation
    def glove(self, msg=True) -> None:
        'Run glove with 100 iterations with a vector size of 50.'
        
        cmd = f'glove -save-file {self.vector_f} -threads 10 -input-file {self.shuffle_f} -x-max 10 -iter 100 -vector-size 50 -binary 2  -vocab-file {self.vocab_f} -verbose 2'
        try:
            open(self.vector_f, 'w')
            open(self.shuffle_f, 'r')
            open(self.vocab_f, 'r')
        except OSError:
            print(f'Cannot access {self.vector_f}, or {self.shuffle_f}, or {self.vocab_f}.', 
                 file=sys.stderr)
        p = subprocess.run(shlex.split(cmd),
                           stderr=subprocess.PIPE)
        if msg:
            print(p.stderr.decode())
        return None


    def __call__(self):
        self.write_corpus()
        self.count_vocab()
        self.calc_cooccur()
        self.shuffle()
        self.glove()


if __name__ == '__main__':
    # Directories and inputput output file settings
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    ROOT_DIR = os.path.join(ROOT_DIR, '..')
    SEQ_DIR =  os.path.join(ROOT_DIR, 'seqs')
    DATA_DIR = os.path.join(ROOT_DIR, 'datasets')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'emeddings')
    
    binding_csv = os.path.join(DATA_DIR, 'binding_Kds.csv')
    output_fasta = os.path.join(SEQ_DIR, 'binding_Kds.fasta')
    

    # Reference sequence (wildtype, Wuhan-Hu-1)
    REF_SEQ = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
    
    # Run GloveEncoder
    df = dms_to_df(binding_csv)
    df_to_fasta(df, REF_SEQ, output_fasta, msg=False)
    encoder = GloveEncoder(output_fasta, 6, 2, 'glove_dms')
    encoder()


