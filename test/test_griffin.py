import torch
import torch.nn.functional as F
import pytest
from torch_specinv.methods import griffin_lim

nfft_list = [
    256, 512, 1024, 2048, 4096
]


@pytest.mark.parametrize("x_sizes", [(44100 * 3,), (2, 44100 * 3,), (1, 44100 * 3)])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nfft", [256, 512, 1024, 2048, 4096])
def test_empty_args(x_sizes, device, dtype, nfft):
    x = torch.randn(*x_sizes, device=device, dtype=dtype)
    spec = torch.stft(x, nfft, return_complex=True)
    y = griffin_lim(spec.abs(), max_iter=4)
    assert len(y.shape) == len(x.shape)
    if len(y.shape) > 1:
        assert y.shape[0] == x.shape[0]
        assert y.shape[1] <= x.shape[1]
    return


@pytest.mark.parametrize("win_length, window", [(None, None),
                                                (1200, None),
                                                (1200, torch.hann_window(1200))])
@pytest.mark.parametrize("hop_length", [None, 256, 800])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("onesided", [None, False, True])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant", "replicate", "circular"])
@pytest.mark.parametrize("return_complex", [None, True, False])
def test_stft_args(
        win_length,
        window,
        hop_length,
        center,
        normalized,
        onesided,
        pad_mode,
        return_complex):
    x = torch.randn(2, 44100 * 2)
    n_fft = 2048
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
    y = griffin_lim(spec, max_iter=2,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=center,
                    pad_mode=pad_mode,
                    normalized=normalized,
                    onesided=onesided,
                    return_complex=return_complex)

    loss = F.mse_loss(x[:, :y.shape[1]], y)
    loss.backward()
    assert hasattr(spec, "grad")
    return
