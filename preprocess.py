# import os
# for mode in ["train","dev","test"]:
#     df = {
#         "O":0,
#         "COMMA":0,
#         "PERIOD":0,
#         "COLON":0
#     }
#     with open(os.path.join("data/cmrpt/", mode+".txt"),"r",encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in lines:
#             df[line.strip().split("\t")[-1]] += 1
#         print(mode)
#         print(df)
import numpy as np
result = {
    "TinyBERT_4L_zh": { "comma": {"p":[],"r":[],"f1":[]}, "period": {"p":[],"r":[],"f1":[]}, "colon": {"p":[],"r":[],"f1":[]}, "overall": {"p":[],"r":[],"f1":[]} },
    "minirbt-h256":{ "comma": {"p":[],"r":[],"f1":[]}, "period": {"p":[],"r":[],"f1":[]}, "colon": {"p":[],"r":[],"f1":[]}, "overall": {"p":[],"r":[],"f1":[]} },
    "chinese-roberta-wwm-ext":{ "comma": {"p":[],"r":[],"f1":[]}, "period": {"p":[],"r":[],"f1":[]}, "colon": {"p":[],"r":[],"f1":[]}, "overall": {"p":[],"r":[],"f1":[]} },
    "albert-base-chinese-cluecorpussmall":{ "comma": {"p":[],"r":[],"f1":[]}, "period": {"p":[],"r":[],"f1":[]}, "colon": {"p":[],"r":[],"f1":[]}, "overall": {"p":[],"r":[],"f1":[]} },
    "rbt6":{ "comma": {"p":[],"r":[],"f1":[]}, "period": {"p":[],"r":[],"f1":[]}, "colon": {"p":[],"r":[],"f1":[]}, "overall": {"p":[],"r":[],"f1":[]} },
    "h256":{ "comma": {"p":[],"r":[],"f1":[]}, "period": {"p":[],"r":[],"f1":[]}, "colon": {"p":[],"r":[],"f1":[]}, "overall": {"p":[],"r":[],"f1":[]} },
    "h312":{ "comma": {"p":[],"r":[],"f1":[]}, "period": {"p":[],"r":[],"f1":[]}, "colon": {"p":[],"r":[],"f1":[]}, "overall": {"p":[],"r":[],"f1":[]} },
    "h768":{ "comma": {"p":[],"r":[],"f1":[]}, "period": {"p":[],"r":[],"f1":[]}, "colon": {"p":[],"r":[],"f1":[]}, "overall": {"p":[],"r":[],"f1":[]} },
}
marks = ["comma","period","colon","overall"]
lines = open("result.txt","r").readlines()
for i in range(0,len(lines),7):
    sample = lines[i:i+7]
    model_name = sample[1].strip().split(":")[-1].strip()
    p = sample[4].strip().split("  ")[1:]
    r = sample[5].strip().split("  ")[1:]
    f1 = sample[6].strip().split("  ")[1:]

    for idx,mark in enumerate(marks):
        result[model_name][mark]["p"].append(p[idx])
        result[model_name][mark]["r"].append(r[idx])
        result[model_name][mark]["f1"].append(f1[idx])
    
for k,v in result.items():
    print(k)
    for i,j in v.items():
        print(i)
        for a,b in j.items():
            # print(a)
            print(a,round(np.mean(np.array(b,dtype=float),axis=0)*100,2))
            # print(np.array(b).mean())

# print(result)