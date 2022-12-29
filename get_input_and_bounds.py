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


def get_cartpole_input_and_bounds(experiment):
    if experiment == "F":
        min_bounds = torch.tensor([-10, -2.18, -0.23, -1.3], requires_grad=False)
        max_bounds = torch.tensor([-2.4, 2.66, 0.23, 1.22], requires_grad=False)
    else:
        min_bounds = torch.tensor([2.4, -2.18, -0.23, -1.3], requires_grad=False)
        max_bounds = torch.tensor([10, 2.66, 0.23, 1.22], requires_grad=False)

    initial_input = (min_bounds + max_bounds) / 2
    initial_input.requires_grad = True

    return initial_input, min_bounds, max_bounds