from dataloader import BertweetDataset
from utils import caculate_score, get_total_time, plot_loss
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

import logging
import torch
import yaml
import copy
import time
import argparse


logging.basicConfig(format='%(message)s', level=logging.INFO, filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def build_model(opts, model_config):
    model = AutoModelForSequenceClassification.from_pretrained(opts["model_name"], config=model_config)
    model.to(opts["device"])
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
    return model, optimizer


def train(opts, model, optimizer, data_iter):
    model.train()
    train_bar = tqdm(data_iter, total=len(data_iter), desc='\tTRAIN:', position=0, leave=True)
    train_loss = 0
    train_preds, train_golds = [], []
    start_time = time.time()
    for batch in train_bar:
        input_ids, token_type_ids, att_masks, label_ids = batch
        if opts["device"] == "cuda":
            input_ids = input_ids.to(opts["device"])
            token_type_ids = token_type_ids.to(opts["device"])
            att_masks = att_masks.to(opts["device"])
            label_ids = label_ids.to(opts["device"])
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
    train_loss = train_loss / len(data_iter)
    return model, train_loss, (acc_score, f1_macro, f1_weighted), get_total_time(start_time)


def eval(opts, model, data_iter):
    model.eval()
    eval_bar = tqdm(data_iter, total=len(data_iter), desc='\tEVAL:', position=0, leave=True)
    eval_loss = 0
    eval_preds, eval_golds = [], []
    start_time = time.time()
    for batch in eval_bar:
        input_ids, token_type_ids, att_masks, label_ids = batch
        if opts["device"] == "cuda":
            input_ids = input_ids.to(opts["device"])
            token_type_ids = token_type_ids.to(opts["device"])
            att_masks = att_masks.to(opts["device"])
            label_ids = label_ids.to(opts["device"])
        outputs = model(input_ids=input_ids,
                        attention_mask=att_masks,
                        token_type_ids=token_type_ids,
                        labels=label_ids)
        eval_loss += outputs.loss.item()
        eval_preds += [y.argmax().item() for y in outputs.logits]
        eval_golds += label_ids.tolist()
    acc_score, f1_macro, f1_weighted = caculate_score(eval_golds, eval_preds)
    eval_loss = eval_loss / len(data_iter)
    return eval_loss, (acc_score, f1_macro, f1_weighted), get_total_time(start_time)


def main(config_file):
    with open(config_file, 'r') as stream:
        try:
            opts = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    opts["device"] = 'cuda' if torch.cuda.is_available() and opts["use_gpu"] else 'cpu'
    model_config = AutoConfig.from_pretrained(opts["model_name"],
                                              num_labels=len(opts["label_maps"]),
                                              finetuning_task=opts["task_name"])
    tokenizer = AutoTokenizer.from_pretrained(opts["model_name"],
                                              normalization=True)
    label_maps = opts["label_maps"]
    logger.info("Load dataset")
    train_dataset = BertweetDataset(opts["train_file_path"], tokenizer, label_maps,
                                    text_idx=opts["text_idx"],
                                    class_idx=opts["class_idx"],
                                    delimiter=opts["delimiter"],
                                    batch_size=opts["train_batch_size"],
                                    max_length=opts["max_length"])
    test_dataset = None
    if opts["test_file_path"] is not None and opts["test_file_path"] != '':
        test_dataset = BertweetDataset(opts["test_file_path"], tokenizer, label_maps,
                                       text_idx=opts["text_idx"],
                                       class_idx=opts["class_idx"],
                                       delimiter=opts["delimiter"],
                                       batch_size=opts["test_batch_size"],
                                       max_length=opts["max_length"])
        test_iter = DataLoader(test_dataset, batch_size=opts["test_batch_size"], shuffle=True)

    if test_dataset is None:
        data_idxs = list(range(train_dataset.__len__()))
        data_idxs, test_idxs = train_test_split(data_idxs, test_size=opts["test_split"], shuffle=True)
        test_iter = DataLoader(train_dataset, batch_size=opts["test_batch_size"],
                               sampler=SubsetRandomSampler(test_idxs))
    else:
        data_idxs = list(range(train_dataset.__len__()))
    kf = KFold(n_splits=opts["kfold"], random_state=opts["random_state"], shuffle=True)
    avg_scores = {"acc": [], "f1_macro": [], "f1_weighted": []}
    for idx, (train_idx, eval_idx) in enumerate(kf.split(data_idxs)):
        model, optimizer = build_model(opts, model_config)
        best_model = copy.deepcopy(model)
        best_epoch, best_loss, best_score = 0, float("inf"), 0
        counter = 0
        train_iter = DataLoader(train_dataset,
                                batch_size=opts["train_batch_size"],
                                sampler=SubsetRandomSampler(train_idx))
        eval_iter = DataLoader(train_dataset,
                               batch_size=opts["test_batch_size"],
                               sampler=SubsetRandomSampler(eval_idx))
        history = {"train_loss": [], "eval_loss": []}
        for epoch in range(opts["num_epochs"]):
            logger.info(f"Fold: {idx} - Epoch: {epoch+1}/{opts['num_epochs']}")
            model, train_loss, train_score, train_time = train(opts, model, optimizer, train_iter)
            logger.info(f'\tTRAIN  - Time: {train_time}; AVG Loss: {train_loss:.6f}; Accurancy: {train_score[0]:.4f}; '
                        f'F1_maro: {train_score[1]:.4f}; F1_weighted: {train_score[2]:.4f}')
            eval_loss, eval_score, eval_time = eval(opts, model, eval_iter)
            logger.info(f"\tEVAL  - Time: {eval_time}; AVG Loss: {eval_loss:.6f}; Accurancy: {eval_score[0]:.4f}; "
                        f"F1_maro: {eval_score[1]:.4f}; F1_weighted: {eval_score[2]:.4f}")
            history["train_loss"].append(train_loss)
            history["eval_loss"].append(eval_loss)
            if best_score <= eval_score[1]:
                best_model = copy.deepcopy(model)
                best_score = eval_score[1]
                best_epoch = epoch

            if best_loss >= eval_loss:
                best_loss = eval_loss
                counter = 0
            else:
                counter += 1
            if counter >= opts["early_stop"]:
                break

        logger.info(f"Test at epoch {best_epoch+1}:")
        test_loss, test_score, test_time = eval(opts, best_model, test_iter)
        logger.info(f"\tTEST  - Time: {test_time}; AVG Loss: {test_loss:.6f}; Accurancy: {test_score[0]:.4f}; "
                    f"F1_maro: {test_score[1]:.4f}; F1_weighted: {test_score[2]:.4f}")
        avg_scores["acc"].append(test_score[0])
        avg_scores["f1_macro"].append(test_score[1])
        avg_scores["f1_weighted"].append(test_score[2])
        plot_loss(history["train_loss"], history["eval_loss"], opts["output_path"])

    logger.info("Summary:")
    logger.info(f"\tAccurancy: {(sum(avg_scores['acc'])/opts['kfold']):.4f}")
    logger.info(f"\tF1 Macro: {(sum(avg_scores['f1_macro'])/opts['kfold']):.4f}")
    logger.info(f"\tF1 Weighted: {(sum(avg_scores['f1_weighted'])/opts['kfold']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None, type=str, required=True,
                        help="Config file path.")
    args = parser.parse_args()
    main(args.config_file)
