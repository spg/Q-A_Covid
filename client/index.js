"use strict";
exports.__esModule = true;
var test_1 = require("./test");
var ctx = ["C'est une cloison issue de la face interne de la dure-m\u00E8re cr\u00E2nienne (tout comme la tente du cervelet et la Faux du cerveau). \nHorizontale, elle est tendue au dessus de la Selle turcique. Elle s'insert au bord sup\u00E9rieur de la lame quadrilat\u00E8re de l'os sph\u00E9no\u00EFde en arri\u00E8re, \u00E0 la l\u00E8vre post\u00E9rieure de la goutti\u00E8re optique et aux 4 apophyses clino\u00EFdes (ant\u00E9rieures et post\u00E9rieures) en avant. \nSur son trajet, elle s'unit au Sinus caverneux.\nElle pr\u00E9sente deux feuillets :\n* Un feuillet superficiel qui n'est autre que la tente de l'hypophyse\n* Un feuillet profond qui tapisse la selle turcique et rejoint le superficiel au niveau de la goutti\u00E8re optique\nLa tente recouvre l'hypophyse et est perc\u00E9e d'un orifice pour le passage de la Tige pituitaire et contient \u00E9galement le sinus coronaire."];
var question = "Ou s'insert-elle ?";
var phrase = "Coucou je suis un troululu et j'aime boire du tesguino";
test_1.getEmbedding(phrase, "fr").then(function () { return console.log("all done"); });
test_1.getEmbedding(ctx[0], "fr").then(function () { return console.log("all done"); });
test_1.answerQuestion(question, ctx, "fr").then(function () { return console.log("all done"); });
