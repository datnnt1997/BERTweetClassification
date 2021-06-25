from torch.utils.data import Dataset
from tqdm import tqdm

import csv
import torch


class BertweetDataset(Dataset):
    def __init__(self,
                 data_file_path,
                 bert_tokenizer,
                 label_map,
                 text_idx,
                 class_idx,
                 delimiter='\t',
                 batch_size=32,
                 max_length=128):
        self.file_path = data_file_path
        self.text_idx = text_idx
        self.class_idx = class_idx
        self.delimiter = delimiter
        self.bz = batch_size
        self.max_len = max_length
        self.label_map = label_map
        self.examples = []
        self.create_examples(bert_tokenizer)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        token_ids = torch.tensor(example[0], dtype=torch.long)
        token_type_ids = torch.tensor(example[1], dtype=torch.long)
        attention_masks = torch.tensor(example[2], dtype=torch.long)
        labels_id = torch.tensor(example[3], dtype=torch.long)
        return token_ids, token_type_ids, attention_masks, labels_id

    @staticmethod
    def read_tsv(file_path,  text_idx, class_idx, delimiter='\t'):
        samples = []
        with open(file_path, 'r') as f:
            tsv_reader = csv.reader(f, delimiter=delimiter)
            next(tsv_reader)
            for row in tsv_reader:
                samples.append((row[text_idx].strip(), row[class_idx].strip()))
        return samples

    def create_examples(self, bert_tokenizer):
        samples = self.read_tsv(self.file_path, self.text_idx, self.class_idx, self.delimiter)
        sidx, eidx = 0, self.bz
        pbar = tqdm(total=len(samples), position=0)
        while sidx <= len(samples):
            batch_samples = samples[sidx: eidx]
            if len(batch_samples) == 0:
                break
            texts, labels = list(zip(*batch_samples))
            label_ids = [self.label_map[label] for label in labels]
            encoded_inputs = bert_tokenizer(text=texts,
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
    file_path = "./dataset/Davidson2017/labeled_data.csv"
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    label_maps = {"0": 1, "1": 1, "2": 0}
    bertdata = BertweetDataset(file_path, tokenizer, label_maps,
                               text_idx=6,
                               class_idx=5,
                               delimiter=',',
                               batch_size=32,
                               max_length=128)
    dataiter = DataLoader(bertdata, batch_size=32, shuffle=True)
    pbar = tqdm(dataiter, total=len(dataiter), position=0)
    for batch in pbar:
        continue