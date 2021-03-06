from sanic import Sanic
from sanic.exceptions import abort
from sanic.response import json
import torch
import numpy as np
from unicodedata import normalize
from model_loader import (
    preload_weights, get_loading_status, get_models_for_lang, load_models)

app = Sanic("Covid_NLU")
preload_weights()


@app.get('status')
async def get_info(req):
    return json({'status': get_loading_status()})


@app.post("/embeddings")
async def get_embedding(request):
    lang = request.json.get('lang')
    text = request.json.get('text')
    try:
        tokenizer, embedder, _ = get_models_for_lang(lang)
    except RuntimeError:
        abort(400, 'Model not loaded')

    splited_text = np.array(text.split(" "))
    splitted_chunk_text = np.array_split(splited_text,
                                         (len(splited_text)//200)+1)
    chunk_text = [" ".join(s) for s in splitted_chunk_text]
    try:
        with torch.no_grad():
            input_tensor = tokenizer.batch_encode_plus(chunk_text,
                                                       pad_to_max_length=True,
                                                       return_tensors="pt")
            last_hidden, pool = embedder(input_tensor["input_ids"],
                                         input_tensor["attention_mask"])
            emb_text = torch.mean(torch.mean(last_hidden, axis=1), axis=0)
            emb_text = emb_text.squeeze().detach().cpu().data.numpy().tolist()
    except RuntimeError as e:
        return json({"error": f"Be careful, special strings will be tokenized in many pieces and the model will not be able to fit : {e}"})
    return json({"embeddings": emb_text})


@app.post("/answers")
async def get_answer(request):
    lang = request.json.get('lang')
    question = normalize("NFC", request.json.get('question'))
    documents = [normalize("NFC", d) for d in request.json.get('docs')]
    try:
        _, __, q_a_pipeline = get_models_for_lang(lang)
    except RuntimeError:
        abort(400, 'Model not loaded')
    else:
        q_a_pipeline({'question': "À quel âge pouvous-nous développer des complications ?", 'context': """Quelques jours après leur avoir dit de rester à la maison, le gouvernement Legault tente maintenant de rassurer les sexagénaires en affirmant qu'il est sécuritaire pour eux de retourner travailler.Plusieurs enseignants et éducateurs sont inquiets pour leur santé et ça se comprend, a indiqué la vice-première ministre Geneviève Guilbault, mercredi après-midi, alors qu'elle remplaçait François Legault pour la conférence de presse quotidienne du gouvernement du Québec. C'est compréhensible que des gens puissent avoir des inquiétudes.Tout indique pourtant que les travailleurs âgés de 60 à 69 ans ne pourront pas invoquer leur âge pour éviter de rentrer au travail, selon le plan de déconfinement des écoles primaires et des services de garde présenté la semaine dernière, qui prévoit une réouverture graduelle des établissements à compter de lundi.La santé publique a établi le facteur de risque à 70 ans et non à 60 ans, a indiqué Mme Guilbault. À partir de 70 ans, le risque de développer des complications est plus important, mais en bas de 70 ans, les gens peuvent retourner travailler , à condition de respecter les consignes de la santé publique en matière de distanciation et d'hygiène.Plus spécifiquement, la vice-première ministre a affirmé que ce sera possible pour les éducateurs et les enseignants âgés entre 60 et 69 ans de reprendre le travail dès la semaine prochaine s'ils respectent ces mesures."""})
        print("OKAY")
        results = [q_a_pipeline({'question': question, 'context': doc})
                   for doc in documents]

        return json({"answers": results})

load_models()

if __name__ == "__main__":
    # TODO load models in an async coro
    app.run(host="0.0.0.0", port=8000, workers=1)
