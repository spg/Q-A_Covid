#!/bin/sh

bert_complete () {
    if [[ -f server/weights/Bert/config.json && -f server/weights/Bert/pytorch_model.bin && -f server/weights/Bert/special_tokens_map.json && -f server/weights/Bert/tokenizer_config.json && -f server/weights/Bert/vocab.txt ]]; then
        return 1
    fi

    return 0
}

bert_qa_complete () {
    if [[ -f server/weights/Bert_Q_A/config.json && -f server/weights/Bert_Q_A/pytorch_model.bin && -f server/weights/Bert_Q_A/special_tokens_map.json && -f server/weights/Bert_Q_A/tokenizer_config.json && -f server/weights/Bert_Q_A/vocab.txt ]]; then
        return 1
    fi

    return 0
}

camember_complete () {
    if [[ -f server/weights/Camembert/config.json && -f server/weights/Camembert/pytorch_model.bin && -f server/weights/Camembert/special_tokens_map.json && -f server/weights/Camembert/tokenizer_config.json && -f server/weights/Camembert/sentencepiece.bpe.model ]]; then
        return 1
    fi

    return 0
}

camember_qa_complete () {
    if [[ -f server/weights/Camembert_Q_A/config.json && -f server/weights/Camembert_Q_A/pytorch_model.bin && -f server/weights/Camembert_Q_A/special_tokens_map.json && -f server/weights/Camembert_Q_A/tokenizer_config.json && -f server/weights/Camembert_Q_A/sentencepiece.bpe.model ]]; then
        return 1
    fi

    return 0
}


all_complete () {
    bert_complete
    is_bert_complete=$?

    bert_qa_complete
    is_bert_qa_complete=$?

    camember_complete
    is_camember_complete=$?

    camember_qa_complete
    is_camember_qa_complete=$?

    if [[ "$is_bert_complete" == "1" && "$is_bert_qa_complete" == "1" && "$is_camember_complete" == "1" && "$is_camember_qa_complete" == "1" ]]; then
        echo "All complete!"
        return 1
    else
        echo "Not complete..."
        return 0
    fi
}

complete="0"
while [ "$complete" == "0" ]
do
all_complete
complete=$?
sleep 10
done
