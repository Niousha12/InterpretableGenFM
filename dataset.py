import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

from transformers import AutoTokenizer


class Passcode(Dataset):
    def __init__(self, csv_file_path, split='train', max_tokenize_length=512):
        self.df = pd.read_csv(csv_file_path)

        self.classes = sorted(self.df.loc[self.df.split == "train", "label"].astype(str).unique())
        self.label2id = {c: i for i, c in enumerate(self.classes)}
        self.id2label = {i: c for c, i in self.label2id.items()}

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.split = split
        self.max_length = max_tokenize_length

        # apply to all splits
        self.df["label_id"] = self.df["label"].astype(str).map(self.label2id)

        # safety: ensure a pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tok_ref = self.tokenizer(
            row["ref"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        tok_alt = self.tokenizer(
            row["alt"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        ref = tok_ref['input_ids'].squeeze(0)
        alt = tok_alt['input_ids'].squeeze(0)
        ref_att = tok_ref['attention_mask'].squeeze(0)
        alt_att = tok_alt['attention_mask'].squeeze(0)
        label = row["label_id"]

        return {'ref': ref, 'alt': alt, 'ref_att': ref_att, 'alt_att': alt_att, 'labels': label}

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    train_dataset = Passcode(csv_file_path='dataset/cell_passport.csv', split='train')
    test_dataset = Passcode(csv_file_path='dataset/cell_passport.csv', split='test')

    print(len(train_dataset))
    print(len(test_dataset))
    labels = []
    for i in tqdm(range(len(train_dataset))):
        labels.append(train_dataset[i]['labels'])
    print(np.unique(labels, return_counts=True))
