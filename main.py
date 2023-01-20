from load_models import load_models
from get_input_and_bounds import get_aurora_input_and_bounds
from utils import *
from attack import PGD
import torch
import os
import csv

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    torch.set_default_dtype(torch.float64)
    mod_type = "short"  # short/long
    full_model_path = f"Aurora/{mod_type}_training"

    seeds = get_sorted_seed_numbers(full_model_path)
    with open(f'{os.path.basename(full_model_path)}_{mod_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['seed_1', 'seed_2', 'm1(x)', 'm2(x)', 'x', 'diff'])
        for i, seed_1 in enumerate(seeds[:-1]):
            for seed_2 in seeds[i + 1:]:
                model_1, model_2 = load_models(seed_1, seed_2, full_model_path)
                initial_input, min_bounds, max_bounds = get_aurora_input_and_bounds()

                x = PGD(model_1, model_2, initial_input, min_bounds, max_bounds, 100)

                m1_x = model_1(x)
                m2_x = model_2(x)

                diff = torch.abs(m1_x - m2_x).item()

                writer.writerow([seed_1, seed_2, m1_x.item(), m2_x.item(), x.tolist(), diff])
                csvfile.flush()

                print(f"({seed_1}, {seed_2}), ({m1_x.item()}, {m2_x.item()}), {diff}")
