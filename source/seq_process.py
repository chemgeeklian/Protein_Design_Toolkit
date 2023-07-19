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

    integer_array = np.empty([len(seqs), max_length])

    for i, seq in enumerate(seqs):
        if   pad_side == 'right': seq = seq.ljust(max_length, '-')
        elif pad_side == 'left' : seq = seq.rjust(max_length, '-')
        else:
            raise ValueError("Invalid padd_side value. It should be either 'right' or 'left'.")

        # Replace each character in the sequence with its corresponding integer
        integer_array[i, :] = np.array([aa_to_int_dic[aa] for aa in seq])

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
    seq_len_last=False, # For nn.conv1D input
    input_logits = False
    ):

    '''
    Convert amino acid sequences to one-hot encoding.
    Output size: [batch_size, seq_len, 21] or [batch_size, 21, seq_len] if batch_first=True
    '''
    
    batch_first = True
    args = seq_column, aa_list, pad_side, check_aa_z, batch_first

    if input_logits:
        if isinstance(sequences, torch.Tensor):
            integer_encoded = sequences.detach().cpu().numpy().copy()
        else: 
            integer_encoded = sequences.copy()
    else:
        integer_encoded = seq2int(sequences, *args)

    one_hot_encoded = np.empty([integer_encoded.shape[0], 
                                integer_encoded.shape[1], 
                                21])
        
    for i, seq in enumerate(integer_encoded):
        # Convert to one-hot with np.eye
        one_hot_encoded[i, :, :] = np.eye(len(aa_list))[seq]
    
    if seq_len_last:
        return np.transpose(one_hot_encoded, (0, 2, 1)) 
    else:
        return one_hot_encoded


class one_hot_dataloader:
    def __init__(self, 
                 sequence_data, 
                 label_data=None, 
                 test_size=0.15, 
                 batch_size=32,
                 normalize_label = True,
                 device=None):
        self.sequence_data = sequence_data
        self.label_data = label_data
        self.test_size = test_size
        self.batch_size = batch_size
        self.normalize_label = normalize_label
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def normalize_y(self, y):
        if isinstance(y, pd.DataFrame):
            return y.apply(lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), axis=0)
        elif isinstance(y, pd.Series):
            return (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
        elif isinstance(y, np.ndarray):
            return (y - np.nanmin(y, axis=0)) / (np.nanmax(y, axis=0) - np.nanmin(y, axis=0))
        else:
            raise ValueError("Input y must be a pandas DataFrame, Series, or a numpy array.")

    def split_data(self, X, y=None):
        indices = np.arange(X.shape[0]) # assuming X is a numpy array or pandas dataframe
        if y is None:
            X_train, X_val, train_id, val_id = train_test_split(X, indices, test_size=self.test_size, random_state=0)
            return X_train, X_val, train_id, val_id
        else:
            X_train, X_val, y_train, y_val, train_id, val_id = train_test_split(X, y, indices, test_size=self.test_size, random_state=0)
            return X_train, X_val, y_train, y_val, train_id, val_id

    def np_data(self):
        X = one_hot_encode(self.sequence_data)
        
        if self.label_data is not None:
            y = self.label_data
            if self.normalize_label:
                y = self.normalize_y(y).values
            return X, y
        else:
            return X

    def load_data(self):
        if self.label_data is not None:
            X, y = self.np_data()
            X = torch.FloatTensor(X).to(self.device)
            y = torch.FloatTensor(y).to(self.device)

            X_train, X_val, y_train, y_val, train_id, val_id = self.split_data(X, y)
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
        else:
            X = self.np_data()
            X = torch.FloatTensor(X).to(self.device)
            train_dataset, val_dataset, train_id, val_id = self.split_data(X)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return self.train_loader, self.val_loader, train_id, val_id
