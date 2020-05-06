from sanic import Sanic
from sanic.response import json
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer, CamembertForQuestionAnswering,
                          pipeline)
import torch
import re
import numpy as np
from rich import print

app = Sanic("Covid_NLU")

QA_MODEL_NAME_FR = "illuin/camembert-large-fquad"
QA_TOK_FR = AutoTokenizer.from_pretrained(QA_MODEL_NAME_FR)
QA_MODEL_FR = CamembertForQuestionAnswering.from_pretrained(QA_MODEL_NAME_FR)
QA_FR = pipeline('question-answering', model=QA_MODEL_FR, tokenizer=QA_TOK_FR)

EMB_MODEL_NAME_FR = "camembert-base"
EMB_TOK_FR = AutoTokenizer.from_pretrained(EMB_MODEL_NAME_FR)
EMB_FR = AutoModel.from_pretrained(EMB_MODEL_NAME_FR)

# QA_MODEL_NAME_EN = "bert-large-uncased-whole-word-masking-finetuned-squad"
# QA_TOK_EN = AutoTokenizer.from_pretrained(QA_MODEL_NAME_EN)
# QA_MODEL_EN = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME_EN)
# QA_EN = pipeline('question-answering', model=QA_MODEL_EN, tokenizer=QA_TOK_EN)

# EMB_MODEL_NAME_EN = "bert-large-uncased"
# EMB_TOK_EN = AutoTokenizer.from_pretrained(EMB_MODEL_NAME_EN)
# EMB_EN = AutoModel.from_pretrained(EMB_MODEL_NAME_EN)

print(":floppy_disk: [green]Model loaded[/green] :floppy_disk:")


@app.post("/get_embedding")
async def get_embedding(request):
    data = request.json
    LANG = data["lang"]
    text = data["utterances"]
    if LANG == "fr":
        tokenizer = EMB_TOK_FR
        embedder = EMB_FR
    elif LANG == "en":
        tokenizer = EMB_TOK_EN
        embedder = EMB_EN

    text_clean = re.sub(
        r'(((http|ftp|https):\/\/)|(www\.))([\wàâçéèêëîïôûùüÿñæœ.,@?^=%&:\\\/~+#-]*[\w@?^=%&\/~+#-])?',
        " ",
        text)
    splited_text = np.array(text_clean.split(" "))
    splitted_chunk_text = np.array_split(splited_text,
                                         (len(splited_text)//200)+1)
    chunk_text = [" ".join(s) for s in splitted_chunk_text]
    with torch.no_grad():
        input_tensor = tokenizer.batch_encode_plus(chunk_text,
                                                   pad_to_max_length=True,
                                                   return_tensors="pt")
        last_hidden, pool = embedder(input_tensor["input_ids"],
                                     input_tensor["attention_mask"])
        emb_text = torch.mean(torch.mean(last_hidden, axis=1), axis=0)
        emb_text = emb_text.squeeze().detach().cpu().data.numpy().tolist()
    return json({"embeddings": emb_text})


@app.post("/get_answer")
async def get_answer(request):
    data = request.json
    LANG = data["lang"]
    question = data["question"]
    documents = data["docs"]
    if LANG == "fr":
        q_a_pipeline = QA_FR
    elif LANG == "en":
        q_a_pipeline = QA_EN
    resultats = []
    for doc in documents:
        res = q_a_pipeline({'question': question, 'context': doc})
        big_left = max(0, res["start"]-500)
        big_right = min(res["end"]+500, len(doc))
        dic_res = {}
        dic_res["score"] = res["score"]
        dic_res["ctx"] = doc[big_left:big_right]
        dic_res["answer"] = res["answer"]
        dic_res["start"] = res["start"]-big_left
        dic_res["end"] = res["end"]-big_left
        resultats.append(dic_res)
    return json({"answers": resultats})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, access_log=True)
