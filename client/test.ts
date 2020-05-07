import axios from 'axios'

export async function getEmbedding(utterances: string, lang: string) {
    const { data } = await axios.post('http://localhost:8000/get_embedding', { utterances, lang })
    console.log(data)
    return data
}

export async function answerQuestion(question: string, docs: string[], lang: string) {
    const { data } = await axios.post('http://localhost:8000/get_answer', { question, docs, lang })
    console.log(data)
    return data
}