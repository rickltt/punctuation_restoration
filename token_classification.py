import argparse
import torch
import numpy as np
import json
import pandas as pd
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
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

    # true_predictions = [
    #     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    # true_labels = [
    #     [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    labels = [ l for l in label_list if l!= "O"]
    precision, recall, f1, _  = precision_recall_fscore_support(true_labels, true_predictions, average=None, labels = labels)
    overall = precision_recall_fscore_support(true_labels, true_predictions, average='macro', labels = labels )

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

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--data_dir', default='data/cmrpt', type=str)
    parser.add_argument('--cache_dir', default='cache_dir', type=str)
    parser.add_argument('--model_name_or_path', default="/disc1/models/chinese-roberta-wwm-ext", type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument('--max_seq_length', default=256, type=int)
    parser.add_argument("--per_device_train_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    set_seed(args.seed)

    data_files = {}
    data_files["train"] = os.path.join(args.data_dir,'train.txt')
    data_files["dev"] = os.path.join(args.data_dir,'dev.txt')
    data_files["test"] = os.path.join(args.data_dir,'test.txt')
    raw_datasets = load_dataset(
        "text",
        data_files=data_files,
        cache_dir=args.cache_dir
    )

    label2id = {j:i for i,j in enumerate(label_list)}
    id2label = {i:j for i,j in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    def split_into_sequence(examples):
        model_inputs = {
            "tokens": [],
            "tags": []
        }
        tokens = [ example.split("\t")[0] for example in examples["text"]]
        tags = [ example.split("\t")[1] for example in examples["text"]]
        model_inputs["tokens"].append(tokens)
        model_inputs["tags"].append(tags)
        return model_inputs
    
    sequence_length = args.max_seq_length - 2
    raw_datasets = raw_datasets.map(split_into_sequence,
                                   batched=True, 
                                   batch_size=sequence_length,
                                   drop_last_batch=True,
                                   remove_columns=["text"])
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"],is_split_into_words=True)
        label_ids = [ [label2id[j] for j in i ] for i in examples["tags"]]

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
                                     remove_columns=["tokens","tags"],
                                     load_from_cache_file=True,
                                     desc="Running tokenizer on dataset")
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

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
    test_results = trainer.predict(test_dataset).metrics["test_eval_metric"]
    test_results = pd.DataFrame(test_results)

    output_eval_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write("Dataset: {} \n".format(args.data_dir.split("/")[-1]))
        writer.write("Model: {} \n".format(args.model_name_or_path.split("/")[-1]))
        writer.write("Seed: {} \n".format(args.seed))
        writer.write("{} \n".format(test_results))

if __name__ == '__main__':
    main()