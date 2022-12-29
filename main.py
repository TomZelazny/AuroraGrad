from load_models import load_models
from get_input_and_bounds import get_aurora_input_and_bounds, get_cartpole_input_and_bounds
from utils import *
import torch
import os
import csv

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    full_model_path = f"Cartpole"
    mod_type = "G"  # cartpole: F/G, aurora: short/long
    # full_model_path = f"Aurora/{mod_type}_training"

    seeds = get_sorted_seed_numbers(full_model_path)
    with open(f'{os.path.basename(full_model_path)}_{mod_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['seed_1', 'seed_2', 'diff'])
        for i, seed_1 in enumerate(seeds[:-1]):
            for seed_2 in seeds[i + 1:]:
                model_1, model_2 = load_models(seed_1, seed_2, full_model_path)

                if full_model_path == "Cartpole":
                    initial_input, min_bounds, max_bounds = get_cartpole_input_and_bounds(mod_type)
                else:
                    initial_input, min_bounds, max_bounds = get_aurora_input_and_bounds()
                step_size = max_bounds - min_bounds

                model_1.zero_grad()
                model_2.zero_grad()
                y = torch.abs(model_1(initial_input) - model_2(initial_input))
                y.backward()

                # Collect the element-wise sign of the data gradient
                sign_data_grad = initial_input.grad.data.sign()
                # Create the perturbed image by adjusting each pixel of the input image
                perturbed_input = initial_input + step_size * sign_data_grad
                # Adding clipping to maintain bounds
                perturbed_input = torch.clamp(perturbed_input, min_bounds, max_bounds)

                diff = torch.abs(model_1(perturbed_input) - model_2(perturbed_input)).item()
                writer.writerow([seed_1, seed_2, diff])
                csvfile.flush()
                print(seed_1, seed_2, diff)
                # assert(torch.all(torch.le(perturbed_input, max_bounds)))
                # assert(torch.all(torch.ge(perturbed_input, min_bounds)))
