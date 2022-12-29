import os
import re


def get_sorted_seed_numbers(full_model_path):
    file_names = os.listdir(full_model_path)
    seeds = map(lambda name: int(re.search(r'\d+', name).group()), file_names)
    return sorted(seeds)


