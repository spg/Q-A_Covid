

import torch
import numpy as np
from rich import print
from typing import TypedDict, List


class Result(TypedDict):
    score: float
    ctx: str
    answer: str
    start: float
    end: float


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1-np.dot(a, b)/((np.linalg.norm(a)*np.linalg.norm(b)))


def get_answer(question: str, dico, tokenizer, embedder, q_a_pipeline) -> List[Result]:
    with torch.no_grad():
        input_tensor = tokenizer.batch_encode_plus([question],
                                                   pad_to_max_length=True,
                                                   return_tensors="pt")
        last_hidden_question, pool = embedder(input_tensor["input_ids"],
                                              input_tensor["attention_mask"])
        emb_q = torch.mean(last_hidden_question, axis=1)
        emb_q = emb_q.squeeze().detach().cpu().data.numpy()

    embs = [(np.linalg.norm(emb_q-sub_dic["embedding_title_fr"]), source) for
            source, sub_dic in dico.items()
            if sub_dic.get("embedding_title_fr", None) is not None]

    # embs = [(cosine_distance(emb_q, sub_dic["embedding_title_fr"]), source)
    #         for source, sub_dic in dico.items()
    #         if sub_dic.get("embedding_title_fr", None) is not None]

    top_3 = sorted(embs)[:3]

    resultats: List[Result] = []
    for i, (score, source) in enumerate(top_3):
        try:
            ctx = dico[source]["content_fr"]
        except KeyError:
            print(source)
        res = q_a_pipeline({'question': question, 'context': ctx})

        big_left = max(0, res["start"]-500)
        big_right = min(res["end"]+500, len(ctx))
        dic_res: Result = {}
        dic_res["score"] = res["score"]
        dic_res["ctx"] = ctx[big_left:big_right]
        dic_res["answer"] = res["answer"]
        dic_res["start"] = res["start"]-big_left
        dic_res["end"] = res["end"]-big_left
        resultats.append(dic_res)
    return resultats


def print_results(question, top_3):
    print("\n\n[green underline] Question :[/green underline]", end=" ")
    print(f"[light_cyan3 underline]{question}[/light_cyan3 underline]")
    for data in top_3:
        start_answer, end_answer = data["start"], data["end"]
        print("\tContexte:", data['ctx'][:start_answer], end=" ")
        print(
            f"[yellow]{data['ctx'][start_answer:end_answer]}[/yellow]", end=" ")
        print(data['ctx'][end_answer:])
        print("\tAnswer", round(data["score"], 4), data["answer"])
        print("\n\n\n")
