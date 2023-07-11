import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from transformers import (
    BertTokenizer,
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed
)
from data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset
from mlm_bert import BertForMaskedLM
from torch.utils.data import DataLoader, RandomSampler
from textbrewer import (
    GeneralDistiller,
    TrainingConfig,
    DistillationConfig
)
from distill_utils import *
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",default="output",type=str)
    parser.add_argument("--teacher_name_or_path",default="/disc1/models/bert-base-uncased",type=str)
    parser.add_argument("--student_config",default="./distill_configs/h312_bert.json",type=str)
    parser.add_argument("--train_file",default="train.txt",type=str)
    parser.add_argument("--dataset_name",default="wikitext",type=str)
    parser.add_argument("--dataset_config_name",default="wikitext-2-raw-v1",type=str)
    parser.add_argument("--cache_dir",default="cache_dir",type=str)
    parser.add_argument("--train_batch_size",default=128,type=int)
    parser.add_argument("--max_seq_length",default=512,type=int)
    parser.add_argument("--learning_rate",default=4e-4,type=float)
    parser.add_argument("--ckpt_steps",default=20,type=int)
    parser.add_argument("--num_train_steps",default=100,type=int)
    parser.add_argument("--warmup_proportion",default=0.1,type=float)
    parser.add_argument("--gradient_accumulation_steps",default=1,type=int)
    parser.add_argument("--mlm_probability",default=0.15,type=float)
    parser.add_argument("--temperature",default=8,type=int)
    parser.add_argument("--lambda_value",default=0.01,type=float)
    parser.add_argument("--seed",default=42,type=int)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger("Main")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    # DATA LOADER #
    if args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.dataset_name+"_cache_dir",
        )
        punctuation_dict = {'O': 0, ',': 1, '.': 2, '?': 3}
    else:
        data_files = {}
        data_files["train"] = args.train_file
        extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=args.cache_dir
        )
        punctuation_dict = {'O': 0, '，': 1, '。': 2, '：': 3}
        set_seed(args.seed)

    forward_batch_size = int(args.train_batch_size /
                             args.gradient_accumulation_steps)
    # TEACHER #  
    tokenizer = BertTokenizer.from_pretrained(args.teacher_name_or_path)  
    config_kwargs = {
        "tokenizer":tokenizer,
        "punctuation_dict": punctuation_dict,
        "lambda_value": args.lambda_value
    }
    model_T = BertForMaskedLM.from_pretrained(
        args.teacher_name_or_path, output_hidden_states=True, ignore_mismatched_sizes=True, **config_kwargs)
    logger.info("model_T loaded.")

    stu_config = BertConfig.from_pretrained(args.student_config)
    stu_config.output_hidden_states = True
    model_S = BertForMaskedLM(stu_config,**config_kwargs)
    # model_S = BertForMaskedLM.from_pretrained(args.teacher_name_or_path, config=stu_config, ignore_mismatched_sizes=True, **config_kwargs)
    logger.info(stu_config)
    logger.info("------total Number of parameters model_S: %i M" %
            (sum(p.numel() for p in model_S.parameters())//1000000))
    logger.info("model_S created.")

    # init checkpoint from teacher model
    num_layers = stu_config.num_hidden_layers
    compressed_sd = init_checkpoint(model_T,num_layers)
    for k in compressed_sd.keys():
        if k in model_S.state_dict().keys():
            model_S.state_dict()[k] = compressed_sd[k]

    model_T.to(device)
    model_S.to(device)

    params = list(model_S.named_parameters())
    all_trainable_params = divide_parameters(
        params, lr=args.learning_rate)
    logger.info("Length of all_trainable_params: %d",
                len(all_trainable_params))
    
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    padding = "max_length"
    max_seq_length = args.max_seq_length

    def tokenize_function(examples):
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column_name],
    )
    train_dataset = tokenized_datasets["train"]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm_probability=args.mlm_probability)
    
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=forward_batch_size, drop_last=True, collate_fn=data_collator)

    logger.info(f"Loading student config from {args.student_config}")
    optimizer = AdamW(all_trainable_params,lr=args.learning_rate, correct_bias=False)

    scheduler_class = get_linear_schedule_with_warmup

    scheduler_args = {'num_warmup_steps': int(args.warmup_proportion*args.num_train_steps), 'num_training_steps': args.num_train_steps}

    logger.warning("***** Running training *****")
    logger.warning("Num split examples = %d", len(train_dataset))
    logger.warning("Forward batch size = %d", forward_batch_size)
    logger.warning("Num backward steps = %d", args.num_train_steps)

    ########### DISTILLATION ###########
    train_config = TrainingConfig(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ckpt_steps=args.ckpt_steps,
        log_dir=os.path.join(args.output_dir,"logs"),
        output_dir=args.output_dir,
        device=device
        )
    
    logger.info(f"{train_config}")

    intermediate_matches = matches[args.student_config.split("/")[-1].split(".")[0]]
    distill_config = DistillationConfig(
        temperature=args.temperature,
        intermediate_matches=intermediate_matches)

    adaptor_T = MlmAdaptorWithLogitsMask
    adaptor_S = MlmAdaptorWithLogitsMask

    distiller = GeneralDistiller(train_config=train_config,
                                 distill_config=distill_config,
                                 model_T=model_T, model_S=model_S,
                                 adaptor_T=adaptor_T,
                                 adaptor_S=adaptor_S)
    
    def proc_fn(batch):
        return {'input_ids':batch['input_ids'],
                'token_type_ids':batch['token_type_ids'],
                'attention_mask':batch['attention_mask'],
                'labels':batch['labels']}

    with distiller:
        distiller.train(optimizer,
                        dataloader=train_dataloader,
                        # num_epochs=args.num_train_epochs,
                        scheduler_class=scheduler_class,
                        scheduler_args=scheduler_args,
                        callback=None,
                        max_grad_norm=1.0,
                        num_steps=args.num_train_steps,
                        batch_postprocessor=proc_fn
                        )
        
if __name__ == "__main__":
    main()
