import torch


def spectral_convergence(input, target):
    return 10 * ((input - target).norm().log10() - target.norm().log10())


def SNR(input, target):
    return -10 * (input / input.norm() - target / target.norm()).pow(2).sum().log10()
