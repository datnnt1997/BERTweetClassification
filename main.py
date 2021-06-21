from dataloader import BertweetDataset
from utils import caculate_score, get_total_time, plot_loss
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import logging
import torch
import yaml
import time

logging.basicConfig(format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    with open("finetuning_config.yaml", 'r') as stream:
        try:
            opts = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    device = 'cuda' if torch.cuda.is_available() and opts["use_gpu"] else 'cpu'

    model_config = AutoConfig.from_pretrained(opts["model_name"],
                                              num_labels=len(opts["labels"]),
                                              finetuning_task=opts["task_name"])
    tokenizer = AutoTokenizer.from_pretrained(opts["model_name"],
                                              normalization=True)
    label_maps = {label: idx for idx, label in enumerate(opts["labels"])}
    logger.info("Load dataset")
    train_dataset = BertweetDataset(opts["train_file_path"], tokenizer, label_maps,
                                    batch_size=opts["train_batch_size"],
                                    max_length=opts["max_length"])
    test_dataset = BertweetDataset(opts["test_file_path"], tokenizer, label_maps,
                                   batch_size=opts["test_batch_size"],
                                   max_length=opts["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(opts["model_name"], config=model_config)
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": opts["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opts["learning_rate"])

    train_iter = DataLoader(train_dataset, batch_size=opts["train_batch_size"], shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=opts["test_batch_size"], shuffle=True)
    history = {"train_loss": [], "test_loss": []}
    for epoch in range(opts["num_epochs"]):
        logger.info(f"Epoch: {epoch+1}/{opts['num_epochs']}")
        model.train()
        train_bar = tqdm(train_iter, total=len(train_iter), desc='\tTRAIN:', position=0, leave=True)
        train_loss = 0
        train_preds, train_golds = [], []
        start_time = time.time()
        for batch in train_bar:
            input_ids, token_type_ids, att_masks, label_ids = batch
            if device == "cuda":
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                att_masks = att_masks.to(device)
                label_ids = label_ids.to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=att_masks,
                            token_type_ids=token_type_ids,
                            labels=label_ids)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += outputs.loss.item()
            train_preds += [y.argmax().item() for y in outputs.logits]
            train_golds += label_ids.tolist()
        acc_score, f1_macro, f1_weighted = caculate_score(train_golds, train_preds)
        train_loss = train_loss / len(train_iter)
        logger.info(f"\tTRAIN  - Time: {get_total_time(start_time)}; AVG Loss: {train_loss:.6f}; Accurancy: {acc_score:.4f}; F1_maro: {f1_macro:.4f}; F1_weighted: {f1_weighted:.4f}")

        model.eval()
        eval_bar = tqdm(test_iter, total=len(test_iter), desc='\tEVAL:', position=0, leave=True)
        eval_loss = 0
        eval_preds, eval_golds = [], []
        start_time = time.time()
        for batch in eval_bar:
            input_ids, token_type_ids, att_masks, label_ids = batch
            if device == "cuda":
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                att_masks = att_masks.to(device)
                label_ids = label_ids.to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=att_masks,
                            token_type_ids=token_type_ids,
                            labels=label_ids)
            eval_loss += outputs.loss.item()
            eval_preds += [y.argmax().item() for y in outputs.logits]
            eval_golds += label_ids.tolist()
        acc_score, f1_macro, f1_weighted = caculate_score(eval_golds, eval_preds)
        eval_loss = eval_loss / len(test_iter)
        logger.info(f"\tEVAL  - Time: {get_total_time(start_time)}; AVG Loss: {eval_loss:.6f}; Accurancy: {acc_score:.4f}; F1_maro: {f1_macro:.4f}; F1_weighted: {f1_weighted:.4f}")
        history["train_loss"].append(train_loss)
        history["test_loss"].append(eval_loss)
    plot_loss(history["train_loss"], history["test_loss"])


if __name__ == "__main__":
    main()