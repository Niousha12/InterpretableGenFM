import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

PRETRAINED_MODEL_NAME = "zhihan1996/DNABERT-2-117M"


class DnaPassportDelta(Dataset):
    def __init__(self, csv_file_path, split='train', anomaly_detection=False, max_tokenize_length=512):
        self.df = pd.read_csv(csv_file_path)

        self.classes = sorted(self.df.loc[self.df.split == "train", "label"].astype(str).unique(), reverse=True)
        self.label2id = {c: i for i, c in enumerate(self.classes)}
        self.id2label = {i: c for c, i in self.label2id.items()}

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, trust_remote_code=True)
        self.split = split
        self.max_length = max_tokenize_length

        # apply to all splits
        self.df["label_id"] = self.df["label"].astype(str).map(self.label2id)
        if split == "train" and anomaly_detection:
            self.df = self.df[self.df["label_id"] == 0].reset_index(drop=True)
        # safety: ensure a pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tok_ref = self.tokenizer(row["ref"], return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.max_length, )
        tok_alt = self.tokenizer(row["alt"], return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.max_length, )

        ref = tok_ref['input_ids'].squeeze(0)
        alt = tok_alt['input_ids'].squeeze(0)
        ref_att = tok_ref['attention_mask'].squeeze(0)
        alt_att = tok_alt['attention_mask'].squeeze(0)
        label = row["label_id"]
        return {'ref': ref, 'alt': alt, 'ref_att': ref_att, 'alt_att': alt_att, 'labels': label}

    def __len__(self):
        return len(self.df)


class DnaPassportSequential(Dataset):
    def __init__(self, csv_file_path, split='train', max_tokenize_length=512):
        self.df = pd.read_csv(csv_file_path)

        self.classes = sorted(self.df.loc[self.df.split == "train", "label"].astype(str).unique(), reverse=True)
        self.label2id = {c: i for i, c in enumerate(self.classes)}
        self.id2label = {i: c for c, i in self.label2id.items()}

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, trust_remote_code=True)
        self.split = split
        self.max_length = max_tokenize_length

        # apply to all splits
        self.df["label_id"] = self.df["label"].astype(str).map(self.label2id)

        # safety: ensure a pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tok = self.tokenizer(row["ref"], row["alt"],  # second sequence
            return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length, )

        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)
        token_type_ids = tok["token_type_ids"].squeeze(0) if "token_type_ids" in tok else None

        label = row["label_id"]

        item = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label, }
        if token_type_ids is not None:
            item["token_type_ids"] = token_type_ids

        return item

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    train_dataset = DnaPassportDelta(csv_file_path='../dataset/cell_passport_sampled.csv', split='train')
    test_dataset = DnaPassportDelta(csv_file_path='../dataset/cell_passport_sampled.csv', split='test')

    distil_dataset = DnaPassportSequential(csv_file_path='../dataset/cell_passport_sampled.csv', split='train')
    print(len(distil_dataset))
    print(distil_dataset.__getitem__(0))
