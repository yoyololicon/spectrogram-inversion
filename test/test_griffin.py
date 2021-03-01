import torch
from torch_specinv.methods import griffin_lim


def test_doable():
    x = torch.randn(8, 44100).cuda()
    spec = torch.stft(x, 1024, return_complex=True)
    y = griffin_lim(spec.abs(), max_iter=50, return_complex=True)
    assert x.numel() >= y.numel()
    return
