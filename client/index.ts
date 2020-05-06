import { getEmbedding, answerQuestion } from "./test"

getEmbedding("Coucou je suis un troululu et j'aime boire du tesguino",
    "fr").then(() => console.log("all done"))