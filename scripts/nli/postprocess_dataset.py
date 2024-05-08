import argparse
import random
from collections import defaultdict
from typing import List

import sys
sys.path.append('/content/dino_NLPFinal/')

from utils import DatasetEntry


def postprocess_dataset(
        dataset: List[DatasetEntry],
        remove_identical_pairs: bool = True,
        remove_duplicates: bool = True,
        seed: int = 42
) -> List[DatasetEntry]:
    """
    Apply various postprocessing steps to a NLI dataset.
    :param dataset: The dataset to postprocess.
    :param remove_identical_pairs: If set to true, we remove all pairs (x1, x2, y) where x1 == x2 as a bi-encoder cannot learn from them.
    :param remove_duplicates:  If set to true, if there are pairs (x1, x2, y) and (x1', x2', y') with x1 == x1', x2 == x2', y == y', we
           only keep one of them.
    :param add_sampled_pairs: If set to true, we add pairs of randomly sampled x1's and x2's and similarity 0 to the dataset.
    :param max_num_text_b_for_text_a_and_label: We keep at most this many examples for each pair of text_a and similarity label.
    :param seed: The seed for the random number generator used to shuffle the dataset and for adding sampled pairs.
    :return: The postprocessed dataset.
    """
    postprocessed_dataset = []

    rng = random.Random(seed)
    rng.shuffle(dataset)

    if remove_duplicates:
        dataset = list(set(dataset))

    for example in dataset:
        if remove_identical_pairs and example.text_a == example.text_b:
            continue

        example.text_b = example.text_b.replace('\u00a0', '').replace('_', '')

        postprocessed_dataset.append(example)

    return postprocessed_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_file", type=str, required=True,
                        help="The input file which contains the NLI dataset")
    parser.add_argument("--output_file", type=str, required=True,
                        help="The output file to which the postprocessed NLI dataset is saved")

    args = parser.parse_args()

    ds = DatasetEntry.read_list(args.input_file)
    ds_pp = postprocess_dataset(ds)
    DatasetEntry.save_list(ds_pp, args.output_file)
