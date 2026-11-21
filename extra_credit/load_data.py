import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from sql_tokenizer import load_tokenizer     # adjusted
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.data_folder = data_folder
        self.split = split

        tokenizer = load_tokenizer()       # adjusted
        self.bos_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.examples = self.process_data(data_folder, split, tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        
        questions = load_lines(f"{data_folder}/{split}.nl")
        question_ids = tokenizer(questions, add_special_tokens=True)["input_ids"]

        if split == "test":
            return [(ids, None, self.bos_id) for ids in question_ids]

        queries = load_lines(f"{data_folder}/{split}.sql")

        if len(questions) != len(queries):
            raise ValueError(f"{split}: questions and SQL queries must pair by line")

        query_ids = tokenizer(queries, add_special_tokens=True)["input_ids"]

        return [(encoder_ids, target_ids, self.bos_id) for encoder_ids, target_ids in zip(question_ids, query_ids)]
    
    def __len__(self):
        
        return len(self.examples)

    def __getitem__(self, idx):
        
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_sequences = []
    decoder_input_sequences = []
    decoder_target_sequences = []
    decoder_starts = []

    for source_ids, target_ids, bos_id in batch:
        encoder_sequences.append(torch.tensor(source_ids, dtype=torch.long))
        decoder_input_sequences.append(
            torch.tensor([bos_id] + target_ids[:-1], dtype=torch.long)
        )
        decoder_target_sequences.append(torch.tensor(target_ids, dtype=torch.long))
        decoder_starts.append([bos_id])

    encoder_ids = pad_sequence(
        encoder_sequences, batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_inputs = pad_sequence(
        decoder_input_sequences, batch_first=True, padding_value=PAD_IDX
    )
    decoder_targets = pad_sequence(
        decoder_target_sequences, batch_first=True, padding_value=PAD_IDX
    )
    initial_decoder_inputs = torch.tensor(decoder_starts, dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_sequences = []
    decoder_starts = []
    for source_ids, _, bos_id in batch:
        encoder_sequences.append(torch.tensor(source_ids, dtype=torch.long))
        decoder_starts.append([bos_id])
        
    encoder_ids = pad_sequence(encoder_sequences, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.tensor(decoder_starts, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(f"{data_folder}/train.nl")
    train_y = load_lines(f"{data_folder}/train.sql")
    dev_x = load_lines(f"{data_folder}/dev.nl")
    dev_y = load_lines(f"{data_folder}/dev.sql")
    test_x = load_lines(f"{data_folder}/test.nl")
    return train_x, train_y, dev_x, dev_y, test_x