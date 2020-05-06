import axios from 'axios'

export async function getEmbedding(utterances: string, lang: string) {
    const { data } = await axios.post('http://localhost:8000/get_embedding', { utterances, lang })
    return data
}

export async function answerQuestion(utterances: string, doc: string, lang: string) {
    const { data } = await axios.post('http://localhost:8000/get_answer', { utterances, doc, lang })
    return data
}