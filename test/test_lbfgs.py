import torch
import torch.nn.functional as F
import pytest
from torch_specinv.methods import L_BFGS

from .consts import nfft_list


@pytest.mark.parametrize("x_sizes", [(4410,), (2, 4410), (1, 4410)])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nfft", nfft_list)
@pytest.mark.parametrize("metric", ['sc', 'snr', 'ser'])
def test_args(x_sizes, device, dtype, nfft, metric):
    x = torch.randn(*x_sizes, device=device, dtype=dtype)

    def trsfn(x):
        return torch.stft(x, nfft, return_complex=True).abs()

    spec = trsfn(x)

    y = L_BFGS(spec, trsfn, samples=x.shape, max_iter=10, metric=metric, eva_iter=3)
    assert len(y.shape) == len(x.shape)
    return
