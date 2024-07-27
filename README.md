# 中文医疗标点恢复

主要解决医疗领域，自动语音识别（ASR）的后处理步骤，给没有标点的序列添加标点符号。

论文：[A Small and Fast BERT for Chinese Medical Punctuation Restoration](https://arxiv.org/pdf/2308.12568) (**INTERSPEECH** 2024).

## 环境

```bash
conda create -n punct python==3.8
conda activate punct
pip install -r requirements.txt
```

## 预训练

完整的训练代码位于[distill](distill)目录下。

使用了[TextBrewer](https://github.com/airaria/TextBrewer)工具包实现知识蒸馏预训练过程，引入了PMP任务，对比学习，知识蒸馏。

### 预训练权重下载

预训练数据和模型权重
- [百度网盘](https://pan.baidu.com/s/1zC-x2B0Q7lVK1Sm2cqnFBw)，提取码: q12e。
- [Google Drive](https://drive.google.com/drive/folders/1O5rnKVblcOeHh3XgGU0ow9kDcoXKFRqI?usp=drive_link)。


## 微调

完整代码位于[classifier_run.py](classifier_run.py)。

🤗 [HF Repo](https://huggingface.co/rickltt).

## Demo

notebook: [inference.ipynb](inference.ipynb)

```python
import torch
import jieba
import numpy as np
from classifier import BertForMaskClassification
from transformers import AutoTokenizer, AutoConfig, BertForTokenClassification

label_list = ["O","COMMA","PERIOD","COLON"]

label2punct = {
    "COMMA": "，",
    "PERIOD": "。",
    "COLON":"：",
}

model_name_or_path = "pmp-h256"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = BertForMaskClassification.from_pretrained(model_name_or_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def punct(text):

    tokenize_words = jieba.lcut(''.join(text))
    mask_tokens = []
    for word in tokenize_words:
        mask_tokens.extend(word)
        mask_tokens.append("[MASK]")
    tokenized_inputs = tokenizer(mask_tokens,is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():   
        logits = model(**tokenized_inputs).logits
    predictions = logits.argmax(-1).tolist()
    predictions = predictions[0]
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])

    result =[]
    print(tokens)
    print(predictions)
    for token, prediction in zip(tokens, predictions):
        if token =="[CLS]" or token =="[SEP]":
            continue
        if token == "[MASK]":
            label = label_list[prediction]
            if label != "O":
                punct = label2punct[label]
                result.append(punct)
        else:
            result.append(token)

    return "".join(result)
```

### Inference

```python
>>> text1 = '对于腰痛治疗专家建议谨慎手术治疗推荐物理康复治疗与心理治疗相结合'
['[CLS]', '对', '于', '[MASK]', '腰', '痛', '[MASK]', '治', '疗', '[MASK]', '专', '家', '建', '议', '[MASK]', '谨', '慎', '[MASK]', '手', '术', '[MASK]', '治', '疗', '[MASK]', '推', '荐', '[MASK]', '物', '理', '[MASK]', '康', '复', '[MASK]', '治', '疗', '[MASK]', '与', '[MASK]', '心', '理', '治', '疗', '[MASK]', '相', '结', '合', '[MASK]', '[SEP]']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
对于腰痛治疗。专家建议，谨慎手术治疗，推荐物理康复治疗与心理治疗相结合。


>>> text2 = '全腹未触及包块肝脾肋下未触及胆囊未触及Murphy征阴性肾脏未触及'
['[CLS]', '全', '腹', '[MASK]', '未', '[MASK]', '触', '及', '[MASK]', '包', '块', '[MASK]', '肝', '[MASK]', '脾', '[MASK]', '肋', '[MASK]', '下', '[MASK]', '未', '[MASK]', '触', '及', '[MASK]', '胆', '囊', '[MASK]', '未', '[MASK]', '触', '及', '[MASK]', 'm', 'u', 'r', 'p', 'h', 'y', '[MASK]', '征', '[MASK]', '阴', '性', '[MASK]', '肾', '脏', '[MASK]', '未', '[MASK]', '触', '及', '[MASK]', '[SEP]']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0]
全腹未触及包块，肝脾肋下未触及，胆囊未触及，murphy征阴性，肾脏未触及。

>>> text3 = '肝浊音界正常肝上界位于锁骨中线第五肋间移动浊音阴性肾区无叩痛'
['[CLS]', '肝', '[MASK]', '浊', '音', '[MASK]', '界', '[MASK]', '正', '常', '[MASK]', '肝', '上', '界', '[MASK]', '位', '于', '[MASK]', '锁', '骨', '[MASK]', '中', '线', '[MASK]', '第', '五', '[MASK]', '肋', '间', '[MASK]', '移', '动', '[MASK]', '浊', '音', '[MASK]', '阴', '性', '[MASK]', '肾', '区', '[MASK]', '无', '[MASK]', '叩', '痛', '[MASK]', '[SEP]']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0]
肝浊音界正常，肝上界位于锁骨中线第五肋间，移动浊音阴性，肾区无叩痛。
```