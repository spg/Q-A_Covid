import glob
import json
import os
import re

import numpy as np
import torch
from rich.progress import track
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer, CamembertForQuestionAnswering,
                          pipeline)

# LANG = "fr"
LANG = "en"

try:
    os.remove(f"./all_{LANG}/content_{LANG}.txt")
except FileNotFoundError:
    pass

for file in glob.iglob(f"./{LANG}/*"):
    os.remove(file)

# model = "illuin/camembert-large-fquad"
model = "camembert-base"
# model = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model)
# bertizer = AutoModelForQuestionAnswering.from_pretrained(model)
bertizer = AutoModel.from_pretrained(model)

with open("covid_raw.json", "r") as file:
    dico = json.load(file)

dico_splitted = {}
for source, sub_dic in track(dico.items(), description="Entries..."):
    try:
        raw_text_fr = sub_dic["content_fr"]
        title_fr = sub_dic["title_fr"]
    except KeyError:
        continue

    splited_words_fr = np.array(raw_text_fr.split(" "))
    splitted_chunk_words_fr = np.array_split(splited_words_fr,
                                             (len(splited_words_fr)//200)+1)
    chunk_sentence_fr = [" ".join(s) for s in splitted_chunk_words_fr]

    for i, chunk in enumerate(chunk_sentence_fr):
        new_sub_dic = sub_dic.copy()
        new_sub_dic.pop("content_en", None)
        new_sub_dic.pop("url_en", None)
        new_sub_dic.pop("title_en", None)
        new_sub_dic.pop("breadcrumb_en", None)
        new_sub_dic.pop("page_title_en", None)
        new_sub_dic["content_fr"] = chunk
        dico_splitted[f"{source}__{i}"] = new_sub_dic

with open(f"covid_splited_raw.json", "w+") as file:
    json.dump(dico_splitted, file)
