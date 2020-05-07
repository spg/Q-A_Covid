import axios from "axios";

export async function getEmbedding(text: string, lang: string) {
  const { data } = await axios.post("http://localhost:8000/embeddings", {
    text,
    lang
  });
  console.log(data);
  return data;
}

export async function answerQuestion(
  question: string,
  docs: string[],
  lang: string
) {
  const { data } = await axios.post("http://localhost:8000/answers", {
    question,
    docs,
    lang
  });
  console.log(data);
  return data;
}
