import os
import torch
import torch.nn as nn


def get_aurora_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(30, 32),  # input layer
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )


def load_policy_network(state_dict_path: str, model_type: str) -> nn.Module:
    model = get_aurora_model()
    model.load_state_dict(torch.load(state_dict_path))
    return model


def load_models(first_seed=1, second_seed=2, full_model_path=f"Aurora"):
    model_1_state_dict_path = os.path.join(full_model_path, f"model_{first_seed}.pt")
    model_1 = load_policy_network(model_1_state_dict_path, full_model_path)
    model_1.eval()
    for param in model_1.parameters():
        param.requires_grad = False

    model_2_state_dict_path = os.path.join(full_model_path, f"model_{second_seed}.pt")
    model_2 = load_policy_network(model_2_state_dict_path, full_model_path)
    model_2.eval()
    for param in model_2.parameters():
        param.requires_grad = False

    return model_1, model_2
