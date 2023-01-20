import torch
import torch.nn as nn


def PGD(m1, m2, initial_input, input_min, input_max, iterations):
    x = initial_input
    step_size = (input_max - input_min) / iterations
    for i in range(iterations):
        m1.zero_grad()
        m2.zero_grad()
        x.requires_grad = True
        loss = torch.abs(m1(x) - m2(x))
        loss.backward()
        x = x + step_size * x.grad.data
        x = torch.clamp(x, input_min, input_max).detach_()
    return x
