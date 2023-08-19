import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from classifier import BertForMaskClassification
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
import pandas as pd
import numpy as np
import jieba
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

label_list = ["O","COMMA","PERIOD","COLON"]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    for prediction, label in zip(predictions, labels):
        for (p, l) in zip(prediction, label):
            if l != -100:
                true_predictions.append(label_list[p])

    true_labels = []
    for prediction, label in zip(predictions, labels):
        for (p, l) in zip(prediction, label):
            if l != -100:
                true_labels.append(label_list[l])

    labels = [ l for l in label_list if l!= "O"]
    overall = precision_recall_fscore_support(true_labels, true_predictions, average='macro', labels = labels )

    return {
        "precision": overall[0],
        "recall": overall[1],
        "f1": overall[2]
    }


def eval_testset(test_dataset, test_results):
    predictions, labels = test_results.predictions, test_results.label_ids
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    tags = test_dataset["tags"]
    mask_tags = test_dataset["mask_tags"]
    
    predict_tags = []
    for mask_tag in mask_tags:
        i = 0
        tmp = []
        mask_tag.append('PAD')
        while i < len(mask_tag) - 1:
            if mask_tag[i+1] == 'PAD':
                tmp.append(mask_tag[i])
            i+=1
        predict_tags.append(tmp)

    for i,j in enumerate(predict_tags):
        idx = 0
        for k,v in enumerate(j):
            if v != 'PAD':
                j[k] = true_predictions[i][idx]
                idx += 1

    tags = [ j for tag in tags for j in tag]
    predict_tags = [ j if j !='PAD' else 'O' for new_tag in predict_tags for j in new_tag]

    labels = [ l for l in label_list if l!= "O"]
    precision, recall, f1, _  = precision_recall_fscore_support(tags, predict_tags, average=None, labels = labels)
    overall = precision_recall_fscore_support(tags, predict_tags, average='macro', labels = labels )

    eval_metric = pd.DataFrame(
        np.array([precision, recall, f1]),
        columns=labels,
        index=['Precision', 'Recall', 'F1'])
    
    eval_metric['OVERALL'] = overall[:3]
    print(eval_metric)
    return {
        "precision": overall[0],
        "recall": overall[1],
        "f1": overall[2],
        "eval_metric": eval_metric.to_dict(orient='list')
    }


def main():
    global tokenizer
    # global ltp_tokenizer
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument('--data_dir', default='data/iwslt', type=str)
    parser.add_argument('--cache_dir', default='cache_dir', type=str)
    parser.add_argument('--model_name_or_path', default="./distill/bert/h312_wiki", type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument('--max_seq_length', default=512, type=int)
    parser.add_argument("--per_device_train_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed(args.seed)

    label2id = {j:i for i,j in enumerate(label_list)}
    id2label = {i:j for i,j in enumerate(label_list)}
    num_labels = len(label_list)

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_config.classifier_dropout = 0.1
    model_config.num_labels = num_labels
    model = BertForMaskClassification.from_pretrained(args.model_name_or_path, config=model_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # ltp_tokenizer = LTP(args.ltp_path)
    # if torch.cuda.is_available():
    #     ltp_tokenizer = ltp_tokenizer.to("cuda")

    data_files = {}
    data_files["train"] = os.path.join(args.data_dir,'train.txt')
    data_files["dev"] = os.path.join(args.data_dir,'dev.txt')
    data_files["test"] = os.path.join(args.data_dir,'test.txt')
    raw_datasets = load_dataset(
        "text",
        data_files=data_files,
        cache_dir=args.cache_dir
    )

    def add_mask_into_sequence(examples):
        model_inputs = {
            "tokens": [],
            "tags": [],
            "mask_tokens": [],
            "mask_tags": []
        }
        tokens = [ example.split("\t")[0] for example in examples["text"]]
        tags = [ example.split("\t")[1] for example in examples["text"]]
        tokenize_words = jieba.lcut(''.join(tokens))
        # tokenize_words = ltp_tokenizer.pipeline(''.join(tokens), tasks=["cws"]).cws
        mask_tags = []
        mask_tokens = []
        idx = 0
        for word in tokenize_words:
            mask_tags.extend(["PAD"]*len(word))
            mask_tags.append(tags[idx + len(word)-1])
            idx += len(word)
            mask_tokens.extend(word)
            mask_tokens.append("[MASK]")
        
        model_inputs["mask_tokens"].append(mask_tokens)
        model_inputs["mask_tags"].append(mask_tags)
        model_inputs["tokens"].append(tokens)
        model_inputs["tags"].append(tags)
        return model_inputs
    
    raw_datasets = raw_datasets.map(add_mask_into_sequence,
                                   batched=True, 
                                   batch_size=int(args.max_seq_length/2),
                                   drop_last_batch=True,
                                #    load_from_cache_file=True,
                                   remove_columns=["text"])

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["mask_tokens"],is_split_into_words=True)
        label_ids = [ [label2id[j] if j in label_list else -100 for j in i ] for i in examples["mask_tags"]]

        labels = []
        for i, label in enumerate(label_ids):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = raw_datasets.map(tokenize_and_align_labels, 
                                     batched=True, 
                                     remove_columns=["mask_tokens","tokens"],
                                    #  load_from_cache_file=True,
                                     desc="Running tokenizer on dataset")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_strategy = "epoch", 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        seed=args.seed, 
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    train_dataset = tokenized_dataset["train"]
    dev_dataset = tokenized_dataset["dev"]
    test_dataset = tokenized_dataset["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    test_results = trainer.predict(test_dataset)
    test_results = eval_testset(test_dataset, test_results)["eval_metric"]

    test_results = pd.DataFrame(test_results)

    output_eval_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write("Dataset: {} \n".format(args.data_dir.split("/")[-1]))
        writer.write("Model: {} \n".format(args.model_name_or_path.split("/")[-1]))
        writer.write("Seed: {} \n".format(args.seed))
        writer.write("{} \n".format(test_results))

if __name__ == '__main__':
    main() 