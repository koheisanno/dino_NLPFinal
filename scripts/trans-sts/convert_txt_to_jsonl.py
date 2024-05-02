# Convert dataset_so_far.txt to dataset_so_far.jsonl

import json
import argparse
from collections import defaultdict
import random
from typing import List
import sys

# in ../../datasets/dataset_so_far.txt

# Read lines in utf-8 format
with open('../../datasets/dataset_so_far.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Each line in text file is in this format:
# DatasetEntry(text_a="Senate confirms Yellen as Fed head", text_b="Le Sénat confirme Yellen en tant que chef de la Fed.", label=0.5)

# Need line in this format for jsonl
# {"text_a": "Senate confirms Yellen as Fed head", "text_b": "Le Sénat confirme Yellen en tant que chef de la Fed.", "label": "0"}

def convert_txt_to_jsonl(lines: List[str]) -> List[dict]:
    new_lines = []
    for line in lines:
        text_a = line.split('text_a="')[1].split('",')[0]
        text_b = line.split('text_b="')[1].split('",')[0]
        label = line.split('label=')[1].split(')')[0]
        new_lines.append({"text_a": text_a, "text_b": text_b, "label": label})
    return new_lines

new_lines = convert_txt_to_jsonl(lines)

with open('../../datasets/dataset_so_far.jsonl', 'w') as f:
    for line in new_lines:
        f.write(json.dumps(line) + '\n')

print('Conversion complete.')
