from torch.utils.data import Dataset
from tqdm import tqdm

import csv
import torch


class BertweetDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_maps, batch_size=32, max_length=128):
        self.file_path = file_path
        self.bz = batch_size
        self.max_len = max_length
        self.label_maps = label_maps
        self.examples = []
        self.create_examples(tokenizer)

    @staticmethod
    def read_tsv(file_path):
        samples = []
        with open(file_path, 'r') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            next(tsv_reader)
            for row in tsv_reader:
                samples.append((row[1].strip(), row[2].strip()))
        return samples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        token_ids = torch.tensor(example[0], dtype=torch.long)
        token_type_ids = torch.tensor(example[1], dtype=torch.long)
        attention_masks = torch.tensor(example[2], dtype=torch.long)
        labels_id = torch.tensor(example[3], dtype=torch.long)
        return token_ids, token_type_ids, attention_masks, labels_id

    def create_examples(self, tokenizer):
        samples = self.read_tsv(self.file_path)
        sidx, eidx = 0, self.bz
        pbar = tqdm(total=len(samples), position=0)
        while sidx <= len(samples):
            batch_samples = samples[sidx: eidx]
            if len(batch_samples) == 0:
                break
            texts, labels = list(zip(*batch_samples))
            label_ids = [self.label_maps[label] for label in labels]
            encoded_inputs = tokenizer(text=texts,
                                       max_length=self.max_len,
                                       padding='max_length',
                                       truncation='longest_first')
            encoded_inputs["label_ids"] = label_ids
            self.examples.extend(list(zip(*encoded_inputs.values())))
            sidx += self.bz
            eidx += self.bz
            pbar.update(len(batch_samples))


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    file_path = "./dataset/HASOC/english_dataset.tsv"
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    label_maps = {"NOT": 0, "HOF": 1}
    bertdata = BertweetDataset(file_path, tokenizer, label_maps, batch_size=32, max_length=128)
    dataiter = DataLoader(bertdata, batch_size=32, shuffle=True)
    for batch in dataiter:
        continue