import torch


def get_aurora_input_and_bounds():
    min_bounds = torch.tensor([
        -0.007, 1.0, 0.7,
        -0.007, 1.0, 0.7,
        -0.007, 1.0, 0.7,
        -0.007, 1.0, 0.7,
        -0.007, 1.0, 0.7,
        -0.007, 1.0, 0.7,
        -0.007, 1.0, 0.7,
        -0.007, 1.0, 0.7,
        -0.007, 1.0, 0.7,
        -0.007, 1.0, 0.7
    ], requires_grad=False)

    max_bounds = torch.tensor([
        0.007, 1.04, 8,
        0.007, 1.04, 8,
        0.007, 1.04, 8,
        0.007, 1.04, 8,
        0.007, 1.04, 8,
        0.007, 1.04, 8,
        0.007, 1.04, 8,
        0.007, 1.04, 8,
        0.007, 1.04, 8,
        0.007, 1.04, 8
    ], requires_grad=False)

    initial_input = (min_bounds + max_bounds) / 2
    initial_input.requires_grad = True

    return initial_input, min_bounds, max_bounds
