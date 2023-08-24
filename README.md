# 中文医疗标点恢复

主要解决医疗领域，自动语音识别（ASR）的后处理步骤，给没有标点的序列添加标点符号。

## 预训练

完整的训练代码位于[distill](distill)目录下。

使用了[TextBrewer](https://github.com/airaria/TextBrewer)工具包实现知识蒸馏预训练过程，引入了PMP任务，对比学习，知识蒸馏。

## 微调

完整代码位于[classifier_run.py](classifier_run.py)。

## 下载

预训练数据和模型权重[下载](https://pan.baidu.com/s/1zC-x2B0Q7lVK1Sm2cqnFBw)。

提取码: q12e