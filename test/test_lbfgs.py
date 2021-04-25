import torch
import torch.nn.functional as F
import pytest
from torch_specinv.methods import L_BFGS

from .consts import nfft_list


@pytest.mark.parametrize("x_sizes", [(4410,), (2, 4410), (1, 4410)])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nfft", nfft_list)
def test_empty_args(x_sizes, device, dtype, nfft):
    x = torch.randn(*x_sizes, device=device, dtype=dtype)

    def trsfn(x):
        return torch.stft(x, nfft, return_complex=True).abs()
    
    spec = trsfn(x)

    y = L_BFGS(spec, trsfn, max_iter=4)
    assert len(y.shape) == len(x.shape)
    if len(y.shape) > 1:
        assert y.shape[0] == x.shape[0]
        assert y.shape[1] <= x.shape[1]
    return


@pytest.mark.parametrize("win_length, window", [(None, None),
                                                (300, None),
                                                (300, torch.hann_window(300))])
@pytest.mark.parametrize("hop_length", [None, 128])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("onesided", [False, True])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant", "replicate", "circular"])
@pytest.mark.parametrize("return_complex", [True, False])
def test_stft_args(
        win_length,
        window,
        hop_length,
        center,
        normalized,
        onesided,
        pad_mode,
        return_complex):
    x = torch.randn(4410)
    n_fft = 512
    spec = torch.stft(x, n_fft,
                      hop_length=hop_length,
                      win_length=win_length,
                      window=window,
                      center=center,
                      pad_mode=pad_mode,
                      normalized=normalized,
                      onesided=onesided,
                      return_complex=True).abs()

    spec.requires_grad = True
    y = L_BFGS(spec, max_iter=2,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=center,
                    pad_mode=pad_mode,
                    normalized=normalized,
                    onesided=onesided,
                    return_complex=return_complex)

    loss = F.mse_loss(x[:y.shape[0]], y)
    loss.backward()
    assert hasattr(spec, "grad")
    return
