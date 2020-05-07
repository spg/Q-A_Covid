import { getEmbedding, answerQuestion } from "./test"

const ctx: string[] = [`C'est une cloison issue de la face interne de la dure-mère crânienne (tout comme la tente du cervelet et la Faux du cerveau). 
Horizontale, elle est tendue au dessus de la Selle turcique. Elle s'insert au bord supérieur de la lame quadrilatère de l'os sphénoïde en arrière, à la lèvre postérieure de la gouttière optique et aux 4 apophyses clinoïdes (antérieures et postérieures) en avant. 
Sur son trajet, elle s'unit au Sinus caverneux.
Elle présente deux feuillets :
* Un feuillet superficiel qui n'est autre que la tente de l'hypophyse
* Un feuillet profond qui tapisse la selle turcique et rejoint le superficiel au niveau de la gouttière optique
La tente recouvre l'hypophyse et est percée d'un orifice pour le passage de la Tige pituitaire et contient également le sinus coronaire.`]

const question: string = `Ou s'insert-elle ?`
const phrase: string = `Coucou je suis un troululu et j'aime boire du tesguino`

getEmbedding(phrase, "fr").then(() => console.log("all done"))
getEmbedding(ctx[0], "fr").then(() => console.log("all done"))
answerQuestion(question, ctx, "fr").then(() => console.log("all done"))