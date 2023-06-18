import numpy as np
import pandas as pd

def one_hot_encode(sequences, unaligned = True):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY-' 
    aa_to_int = dict((c, i) for i, c in enumerate(amino_acids))
    
    max_length = max(len(seq) for seq in sequences)  # Find length of longest sequence

    one_hot_encoded = []

    for sequence in sequences:
        if unaligned == True:
            sequence = sequence.ljust(max_length, '-')

        # Replace each character in the sequence with its corresponding integer
        integer_encoded = [aa_to_int[aa] for aa in sequence]

        # Convert to one-hot with np.eye function
        one_hot = np.eye(len(amino_acids))[integer_encoded]
        one_hot_encoded.append(one_hot)

    return np.array(one_hot_encoded)
