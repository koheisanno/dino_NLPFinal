# Given a file with sentence pairs, this script removes pairs where the source and target languages are the same.

# Read in input file from DINO_NLPFINAL/datasets/trans-sts-gpt3.5.jsonl
import json
import argparse
from collections import defaultdict
import random
from typing import List
# from spacy_langdetect import LanguageDetector
import spacy
import spacy_fastlang

def read_jsonl(file_path: str) -> List[dict]:
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# pairs = read_jsonl('../../datasets/trans-sts-gpt3.5.jsonl')
pairs = read_jsonl('../../datasets/trans-sts-gpt3.5-dataset.jsonl')

print('Original number of pairs:', len(pairs))

# nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector")

# Remove pairs where the source and target languages are not english and french, respectively
def remove_pairs_with_wrong_language(pairs: List[dict]) -> List[dict]:
    new_pairs = []
    faulty_pairs = []

    for i, pair in enumerate(pairs):
        if i % 1000 == 0:
            print(f'Processing pair {i}...')
        
        doc_text_a = nlp(pair['text_a'])
        first_sentence_language = doc_text_a._.language
        
        doc_text_b = nlp(pair['text_b'])
        second_sentence_language = doc_text_b._.language

        if first_sentence_language == 'en' and second_sentence_language == 'fr':
            new_pairs.append(pair)
        else:
            faulty_pairs.append(pair)
    return new_pairs, faulty_pairs

new_pairs, faulty_pairs = remove_pairs_with_wrong_language(pairs)
print('Number of pairs after removing pairs with wrong languages:', len(new_pairs))
print('Number of pairs removed:', len(faulty_pairs))

print(faulty_pairs)

# Write new pairs to file
def write_jsonl(file_path: str, data: List[dict]):
    with open(file_path, 'w') as f:
        for pair in data:
            f.write(json.dumps(pair) + '\n')

write_jsonl('../../datasets/trans-sts-gpt3.5-dataset-en-fr.jsonl', new_pairs)
