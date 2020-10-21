import nltk
from nltk.tokenize import word_tokenize

sentence = "Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) language, in particular how to program computers to process and analyze large amounts of natural language data."

print(word_tokenize(sentence))