import torch


def spectral_convergence(input, target):
    return 20 * ((input - target).norm().log10() - target.norm().log10())


def SNR(input, target):
    return -10 * (input / input.norm() - target / target.norm()).pow(2).sum().log10()


def SER(input, target):
    return 10 * (input.pow(2).sum().log10() - (input - target).pow(2).sum().log10())
