import numpy as np
import pandas as pd

def seq2int(
    sequences, 
    seq_column=None,
    amino_acids='-ACDEFGHIKLMNPQRSTVWY',
    pad_side='right',
    check_aa_z=True,
    batch_first=True):

    '''
    Convert amino acid sequences to int array for transformers and other models.
    Output size: [seq_len, batch_size] or [batch_size, seq_len] if batch_first==True
    '''

    # Convert input sequences to a list of string
    if   type(sequences) == list: seqs = sequences.copy()
    elif type(sequences) == pd.Series: seqs = sequences.copy().values
    elif type(sequences) == pd.DataFrame:
        if seq_column == None:
            raise AssertionError("Missing name for the column of the sequences.")
        seqs = sequences[seq_column].copy().values
    else: 
        raise TypeError("Unable to convert input to a list of string.")

    def check_string(s):
        if not set(s).issubset(set(amino_acids)):
            raise AssertionError(f"String '{s}' contains unexpected characters.")
    if check_aa_z: [check_string(s) for s in seqs]

    aa_to_int_dic = dict((c, i) for i, c in enumerate(amino_acids))
    max_length = max(len(seq) for seq in seqs)

    integer_array = []

    for seq in seqs:
        if   pad_side == 'right': seq = seq.ljust(max_length, '-')
        elif pad_side == 'left' : seq = seq.rjust(max_length, '-')
        else:
            raise ValueError("Invalid padd_side value. It should be either 'right' or 'left'.")

        # Replace each character in the sequence with its corresponding integer
        integer_encoded = [aa_to_int_dic[aa] for aa in seq]
        integer_array.append(integer_encoded)

    integer_array = np.array(integer_array)

    if batch_first:
        return integer_array
    else:
        return integer_array.T
        

def one_hot_encode(
    sequences, 
    seq_column=None,
    aa_list='-ACDEFGHIKLMNPQRSTVWY',
    pad_side='right',
    check_aa_z=True,
    seq_len_last=False # For nn.conv1D input
    ):

    '''
    Convert amino acid sequences to one-hot encoding.
    Output size: [batch_size, seq_len, 21] or [batch_size, 21, seq_len] if batch_first=True
    '''
    
    batch_first = True
    args = seq_column, aa_list, pad_side, check_aa_z, batch_first
    integer_encoded = seq2int(sequences, *args)

    one_hot_encoded = []
        
    for seq in integer_encoded:
        # Convert to one-hot with np.eye
        one_hot = np.eye(len(aa_list))[seq]
        one_hot_encoded.append(one_hot)

    one_hot_encoded = np.array(one_hot_encoded)
    
    if seq_len_last:
        return np.transpose(one_hot_encoded, (0, 2, 1)) 
    else:
        return one_hot_encoded
