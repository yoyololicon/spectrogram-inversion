import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lbfgs import LBFGS
from tqdm import tqdm
from functools import partial
import math
import matplotlib.pyplot as plt

pi2 = 2 * math.pi

from .metrics import spectral_convergence, SNR, SER


def _args_helper(spec, **stft_kwargs) -> (int, dict):
    """A helper function to get stft arguments from the provided kwargs.

    Args:
        spec: The magnitude spectrum of size (freq x time).
        **stft_kwargs: Keyword arguments that computed spec from 'torch.stft'.
        See `torch.stft` for details.

    Returns:
        n_fft: FFT size of the spectrum.
        processed_kwargs: Dict object that stored the processed keyword arguments.

    """
    device = spec.device
    dtype = spec.dtype
    args_dict = {'win_length': None, 'window': None, 'hop_length': None, 'center': True, 'normalized': False,
                 'onesided': True}
    for key, item in args_dict.items():
        try:
            args_dict[key] = stft_kwargs[key]
        except:
            pass
    win_length, window, hop_length, center, normalized, onesided = tuple(args_dict.values())

    if onesided:
        n_fft = (spec.shape[1] - 1) * 2
    else:
        n_fft = spec.shape[1]

    if not win_length:
        win_length = n_fft

    if not hop_length:
        hop_length = n_fft // 4

    if window is None:
        coeff = hop_length / win_length
        conv_weight = torch.eye(win_length, dtype=dtype, device=device).unsqueeze(1)
    else:
        coeff = hop_length / window.pow(2).sum()
        conv_weight = torch.diag(window).unsqueeze(1)

    offset = (n_fft - win_length) // 2
    args_dict['win_length'] = win_length
    args_dict['hop_length'] = hop_length
    args_dict['ola_weight'] = conv_weight
    args_dict['offset'] = offset
    return n_fft, args_dict


def _ola(x, hop_length, weight, norm_envelope=None):
    """A helper function to do overlap-and-add.

    Args:
        x: input tensor of size :math: '(window_size, time)'.
        hop_length: The distance between neighboring sliding window frames.
        weight: An identity matrix of size (win_length x win_length) .
        norm_envelope: The normalized coefficient apply on synthesis window.

    Returns:
        A 1d tensor containing the overlap-and-add result.

    """
    ola_x = F.conv_transpose1d(x, weight, stride=hop_length).squeeze(1)
    if norm_envelope is None:
        norm_envelope = F.conv_transpose1d(torch.ones_like(x[:1]), weight.pow(2), stride=hop_length).squeeze(1)
    return ola_x / norm_envelope, norm_envelope


def _istft(x, n_fft, win_length, window, hop_length, center, normalized, onesided, offset, ola_weight,
           norm_envelope=None):
    """
    A helper function to do istft.
    """
    x = torch.irfft(x.transpose(1, 2), 1, normalized=normalized, onesided=onesided,
                    signal_sizes=[n_fft] if onesided else None)[..., offset:offset + win_length]

    x = x.transpose(1, 2)
    x, norm_envelope = _ola(x, hop_length, ola_weight, norm_envelope)
    if center:
        x = x[:, win_length // 2:-win_length // 2]
    return x, norm_envelope


def griffin_lim(spec, maxiter=200, tol=1e-6, alpha=0.99, verbose=True, evaiter=10, metric='sc', **stft_kwargs):
    r"""Reconstruct spectrogram phase using the will known `Griffin-Lim`_ algorithm and its variation, `Fast Griffin-Lim`_.


    .. _`Griffin-Lim`: https://pdfs.semanticscholar.org/14bc/876fae55faf5669beb01667a4f3bd324a4f1.pdf
    .. _`Fast Griffin-Lim`: https://perraudin.info/publications/perraudin-note-002.pdf

    Args:
        spec (Tensor): the input tensor of size :math:`(N \times T)` (magnitude) or :math:`(N \times T \times 2)`
            (complex input). If a magnitude spectrogram is given, the phase will first be intialized using
            :func:`torch_specinv.methods.phase_init`; otherwise start from the complex input.
        maxiter (int): maximum number of iterations before timing out.
        tol (float): tolerance of the stopping condition base on L2 loss. Default: ``1e-6``
        alpha (float): speedup parameter used in `Fast Griffin-Lim`_, set it to zero will disable it. Default: ``0``
        verbose (bool): whether to be verbose. Default: :obj:`True`
        evaiter (int): steps size for evaluation. After each step, the function defined in `metric` will evaluate. Default: ``10``
        metric (str): evaluation function. Currently available functions: ``'sc'`` (spectral convergence), ``'snr'`` or ``'ser'``. Default: ``'sc'``
        **stft_kwargs: other arguments that pass to :func:`torch.stft`

    Returns:
        A 1d tensor converted from the given spectrogram

    """
    if spec.shape[-1] != 2:
        cmplx_spec = phase_init(spec, **stft_kwargs)
        if len(cmplx_spec.shape) == 3:
            target_spec = spec.unsqueeze(0)
            cmplx_spec = cmplx_spec.unsqueeze(0)
        else:
            target_spec = spec
    else:
        if len(spec.shape) == 3:
            cmplx_spec = spec.unsqueeze(0)
        else:
            cmplx_spec = spec
        target_spec = cmplx_spec.pow(2).sum(-1).sqrt()

    n_fft, proccessed_args = _args_helper(target_spec, **stft_kwargs)
    istft = partial(_istft, n_fft=n_fft, **proccessed_args)
    pre_spec = cmplx_spec.clone()
    x, norm_envelope = istft(cmplx_spec)

    criterion = nn.MSELoss()
    init_loss = None
    bar_dict = {}
    if metric == 'snr':
        metric_func = SNR
        bar_dict['SNR'] = 0
        metric = metric.upper()
    elif metric == 'ser':
        metric_func = SER
        bar_dict['SER'] = 0
        metric = metric.upper()
    else:
        metric_func = spectral_convergence
        bar_dict['spectral_convergence'] = 0
        metric = 'spectral_convergence'

    lr = alpha / (1 + alpha)

    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            new_spec = torch.stft(x, n_fft, **stft_kwargs)
            if i % evaiter == evaiter - 1:
                mag = new_spec.pow(2).sum(-1).sqrt()
                bar_dict[metric] = metric_func(mag, target_spec).item()
                l2_loss = criterion(mag, target_spec).item()
                pbar.set_postfix(**bar_dict, loss=l2_loss)
                pbar.update(evaiter)

                if not init_loss:
                    init_loss = l2_loss
                elif (previous_loss - l2_loss) / init_loss < tol * evaiter and previous_loss > l2_loss:
                    break
                previous_loss = l2_loss

            new_spec = new_spec - pre_spec * lr
            # new_spec = new_spec + alpha * (new_spec - pre_spec)
            pre_spec = new_spec

            norm = new_spec.pow(2).sum(-1).sqrt() + 1e-16
            new_spec = new_spec * (target_spec / norm).unsqueeze(-1)
            x, _ = istft(new_spec, norm_envelope=norm_envelope)

    return x.squeeze(0)


def RTISI_LA(spec, look_ahead=-1, asymmetric_window=False, maxiter=25, alpha=0.99, verbose=1, **stft_kwargs):
    r"""
    Reconstruct spectrogram phase using `Real-Time Iterative Spectrogram Inversion with Look Ahead`_ (RTISI-LA).

    .. _`Real-Time Iterative Spectrogram Inversion with Look Ahead`:
        https://lonce.org/home/Publications/publications/2007_RealtimeSignalReconstruction.pdf


    Args:
        spec (Tensor): the input tensor of size :math:`(N \times T)` (magnitude).
        look_ahead (int): how many future frames will be consider. ``-1`` will set it to ``(win_length - 1) / hop_length``,
            ``0`` will disable look-ahead strategy and fall back to original RTISI algorithm. Default: ``-1``
        asymmetric_window (bool): whether to apply asymmetric window on the first iteration for new coming frame.
        maxiter (int): number of iterations for each step.
        alpha (float): speedup parameter used in `Fast Griffin-Lim`_, set it to zero will disable it. Default: ``0``
        verbose (bool): whether to be verbose. Default: :obj:`True`
        **stft_kwargs: other arguments that pass to :func:`torch.stft`.

    Returns:
        A 1d tensor converted from the given spectrogram

    """
    if len(spec.shape) == 2:
        target_spec = spec.unsqueeze(0)
    else:
        target_spec = spec

    n_fft, proccessed_args = _args_helper(target_spec, **stft_kwargs)
    copyed_kwargs = stft_kwargs.copy()
    copyed_kwargs['center'] = False

    win_length = proccessed_args['win_length']
    hop_length = proccessed_args['hop_length']
    offset = proccessed_args['offset']
    onesided = proccessed_args['onesided']
    normalized = proccessed_args['normalized']
    ola_weight = proccessed_args['ola_weight']
    # ola_weight_straight = torch.eye(win_length, dtype=target_spec.dtype, device=target_spec.device).unsqueeze(1)
    window = proccessed_args['window']
    if window is None:
        window = torch.diagonal(ola_weight.squeeze())
        synth_coeff = hop_length / win_length
    else:
        synth_coeff = hop_length / window.pow(2).sum()

    #ola_weight = ola_weight * synth_coeff

    num_keep = (win_length - 1) // hop_length
    if look_ahead < 0:
        look_ahead = num_keep

    asym_window1 = target_spec.new_zeros(win_length)
    for i in range(num_keep):
        asym_window1[(i + 1) * hop_length:] += window.flip(0)[:-(i + 1) * hop_length:]
    asym_window1 *= synth_coeff
    # asym_window1 *= hop_length / (asym_window1 * window).sum() / synth_coef

    asym_window2 = target_spec.new_zeros(win_length)
    for i in range(num_keep + 1):
        asym_window2[i * hop_length:] += window.flip(0)[:-i * hop_length if i else None]
    asym_window2 *= synth_coeff

    steps = target_spec.shape[2]
    xt = target_spec.new_zeros(target_spec.shape[0], steps + num_keep + 2 * look_ahead, n_fft)
    xt_winview = xt[..., offset:offset + win_length]
    target_spec = F.pad(target_spec, [look_ahead, look_ahead])

    def irfft(x):
        return torch.irfft(x, 1, normalized=normalized, onesided=onesided, signal_sizes=[n_fft] if onesided else None)

    def rfft(x):
        return torch.rfft(x, 1, normalized=normalized, onesided=onesided)

    # initialize first frame with zero phase
    first_frame = target_spec[..., look_ahead]
    xt_winview[:, num_keep + look_ahead] = irfft(torch.stack((first_frame, torch.zeros_like(first_frame)), -1))[:,
                                           offset:offset + win_length]

    lr = alpha / (1 + alpha)
    with tqdm(total=steps + look_ahead, disable=not verbose) as pbar:
        for i in range(steps + look_ahead):
            for j in range(maxiter):
                x, _ = _ola(xt_winview[:, i:i + num_keep + look_ahead + 1].transpose(1, 2), hop_length,
                            ola_weight * synth_coeff, norm_envelope=1.)
                if asymmetric_window:
                    xt_winview[:, i + num_keep:i + num_keep + look_ahead + 1] = \
                        x.unfold(1, win_length, hop_length)[:, num_keep:]

                    xt_winview[:, i + num_keep:i + num_keep + look_ahead] *= window
                    if j:
                        xt_winview[:, i + num_keep + look_ahead] *= asym_window2
                    else:
                        xt_winview[:, i + num_keep + look_ahead] *= asym_window1

                    new_spec = rfft(xt[:, i + num_keep:i + num_keep + look_ahead + 1]).transpose(1, 2)
                else:
                    new_spec = torch.stft(F.pad(x[:, num_keep * hop_length - offset:], [0, offset]),
                                          n_fft=n_fft, **copyed_kwargs)

                if j:
                    new_spec -= lr * pre_spec
                    pre_spec.copy_(new_spec)
                elif i:
                    new_spec[:, :, :-1] -= lr * pre_spec[:, :, 1:]
                    pre_spec.copy_(new_spec)
                else:
                    pre_spec = new_spec.clone()

                norm = new_spec.pow(2).sum(-1).sqrt() + 1e-16
                new_spec *= (target_spec[..., i:i + look_ahead + 1] / norm).unsqueeze(-1)

                xt_winview[:, i + num_keep:i + num_keep + look_ahead + 1] = irfft(new_spec.transpose(1, 2))[...,
                                                                            offset:offset + win_length]

            pbar.update()

    x, _ = _ola(xt_winview[:, num_keep + look_ahead:-look_ahead if look_ahead else None].transpose(1, 2), hop_length,
                ola_weight)
    if proccessed_args['center']:
        x = x[:, win_length // 2:-win_length // 2]
    else:
        x = F.pad(x, [offset, offset])

    return x.squeeze(0)


def ADMM(spec, maxiter=1000, tol=1e-6, rho=0.1, verbose=1, evaiter=10, metric='sc', **stft_kwargs):
    r"""
    Reconstruct spectrogram phase using `Griffin–Lim Like Phase Recovery via Alternating Direction Method of Multipliers`_ .

    .. _`Griffin–Lim Like Phase Recovery via Alternating Direction Method of Multipliers`:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8552369

    Args:
        spec (Tensor): the input tensor of size :math:`(N \times T)` (magnitude) or :math:`(N \times T \times 2)`
            (complex input). If a magnitude spectrogram is given, the phase will first be intialized using
            :func:`torch_specinv.methods.phase_init`; otherwise start from the complex input.
        maxiter (int): maximum number of iterations before timing out.
        tol (float): tolerance of the stopping condition base on L2 loss. Default: ``1e-6``
        rho (float): non-negative speedup parameter. Small value is preferable when the input spectrogram is noisy (inperfect);
            set it to 1 will behave similar to ``griffin_lim``.  Default: ``0.1``
        verbose (bool): whether to be verbose. Default: :obj:`True`
        evaiter (int): steps size for evaluation. After each step, the function defined in ``metric`` will evaluate. Default: ``10``
        metric (str): evaluation function. Currently available functions: ``'sc'`` (spectral convergence), ``'snr'`` or ``'ser'``. Default: ``'sc'``
        **stft_kwargs: other arguments that pass to :func:`torch.stft`.


    Returns:
        A 1d tensor converted from the given spectrogram

    """
    n_fft, proccessed_args = _args_helper(spec, **stft_kwargs)

    istft = partial(_istft, n_fft=n_fft, **proccessed_args)

    if len(spec.shape) == 2:
        X = phase_init(spec, **stft_kwargs)
    else:
        X = torch.stack((spec, torch.zeros_like(spec)), 2)

    x = istft(X)
    Z = X.clone()
    Y = X.clone()
    U = torch.zeros_like(X)

    criterion = nn.MSELoss()
    init_loss = None
    bar_dict = {}
    if metric == 'snr':
        metric_func = SNR
        bar_dict['SNR'] = 0
        metric = metric.upper()
    elif metric == 'ser':
        metric_func = SER
        bar_dict['SER'] = 0
        metric = metric.upper()
    else:
        metric_func = spectral_convergence
        bar_dict['spectral_convergence'] = 0
        metric = 'spectral_convergence'

    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            # Pc2
            X[:] = Z - U
            mag = X.pow(2).sum(2).sqrt()
            X *= (spec / F.threshold_(mag, 1e-7, 1e-7)).unsqueeze(-1)

            Y[:] = X + U
            # Pc1
            x[:] = istft(Y)
            reconstruted = torch.stft(x, n_fft, **stft_kwargs)

            Z[:] = (rho * Y + reconstruted) / (1 + rho)
            U += X - Z

            if i % evaiter == evaiter - 1:
                mag = reconstruted.pow(2).sum(2).sqrt()
                bar_dict[metric] = metric_func(mag, spec).item()
                l2_loss = criterion(mag, spec).item()
                pbar.set_postfix(**bar_dict, loss=l2_loss)
                pbar.update(evaiter)

                if not init_loss:
                    init_loss = l2_loss
                elif (previous_loss - l2_loss) / init_loss < tol * evaiter and previous_loss > l2_loss:
                    break
                previous_loss = l2_loss

    return istft(X)


def L_BFGS(spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, metric='sc',
           **kwargs):
    r"""

    Reconstruct spectrogram phase using `Inversion of Auditory Spectrograms, Traditional Spectrograms, and Other
    Envelope Representations`_, where I directly use the :class:`torch.optim.LBFGS` optimizer provided in PyTorch.
    This method doesn't restrict to traditional short-time Fourier Transform, but any kinds of presentation (ex: Mel-scaled Spectrogram) as
    long as the transform function is differentiable.

    .. _`Inversion of Auditory Spectrograms, Traditional Spectrograms, and Other Envelope Representations`:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6949659

    Args:
        spec (Tensor): the input presentation.
        transform_fn: a function that has the form ``spec = transform_fn(x)`` where x is an 1d tensor.
        samples (int, optional): number of samples in time domain. Default: :obj:`None`
        init_x0 (Tensor, optional): an 1d tensor that make use as initial time domain samples. If not provided, will use random
            value tensor with length equal to ``samples``.
        maxiter (int): maximum number of iterations before timing out.
        tol (float): tolerance of the stopping condition base on L2 loss. Default: ``1e-6``.
        verbose (bool): whether to be verbose. Default: :obj:`True`
        evaiter (int): steps size for evaluation. After each step, the function defined in ``metric`` will evaluate. Default: ``10``
        metric (str): evaluation function. Currently available functions: ``'sc'`` (spectral convergence), ``'snr'`` or ``'ser'``. Default: ``'sc'``
        **kwargs: other arguments that pass to :class:`torch.optim.LBFGS`.

    Returns:
        A 1d tensor converted from the given presentation
    """
    if init_x0 is None:
        init_x0 = spec.new_empty(samples).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.MSELoss()
    optimizer = LBFGS([x], **kwargs)

    def closure():
        optimizer.zero_grad()
        V = transform_fn(x)
        loss = criterion(V, T)
        loss.backward()
        return loss

    bar_dict = {}
    if metric == 'snr':
        metric_func = SNR
        bar_dict['SNR'] = 0
        metric = metric.upper()
    elif metric == 'ser':
        metric_func = SER
        bar_dict['SER'] = 0
        metric = metric.upper()
    else:
        metric_func = spectral_convergence
        bar_dict['spectral_convergence'] = 0
        metric = 'spectral_convergence'

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            optimizer.step(closure)

            if i % evaiter == evaiter - 1:
                with torch.no_grad():
                    V = transform_fn(x)
                    bar_dict[metric] = metric_func(V, spec).item()
                    l2_loss = criterion(V, spec).item()
                    pbar.set_postfix(**bar_dict, loss=l2_loss)
                    pbar.update(evaiter)

                    if not init_loss:
                        init_loss = l2_loss
                    elif (previous_loss - l2_loss) / init_loss < tol * evaiter:
                        break
                    previous_loss = l2_loss

    return x.detach()


def phase_init(spec, **stft_kwargs):
    r"""
    A phase initialize function that can be seen as a simplified version of `Single Pass Spectrogram Inversion`_.

    .. _`Single Pass Spectrogram Inversion`:
        https://ieeexplore.ieee.org/document/7251907

    Args:
        spec (Tensor): the input tensor of size :math:`(* \times N \times T)` (magnitude).
        **stft_kwargs: other arguments that pass to :func:`torch.stft`

    Returns:
        The estimated complex value spectrogram of size :math:`(N \times T \times 2)`
    """
    n_fft, proccessed_args = _args_helper(spec, **stft_kwargs)
    hop_length = proccessed_args['hop_length']

    if len(spec.shape) == 2:
        spec = spec.unsqueeze(0)

    phase = torch.zeros_like(spec)

    def peak_picking(x):
        mask = (x[:, 1:-1] > x[:, 2:]) & (x[:, 1:-1] > x[:, :-2])
        return F.pad(mask, [0, 0, 1, 1])

    mask = peak_picking(spec)
    b = torch.masked_select(spec, mask)
    a = torch.masked_select(spec[:, :-1], mask[:, 1:])
    r = torch.masked_select(spec[:, 1:], mask[:, :-1])
    idx1, idx2, idx3 = torch.nonzero(mask).t().unbind()
    p = 0.5 * (a - r) / (a - 2 * b + r)
    omega = pi2 * (idx2.float() + p) / n_fft * hop_length

    phase[idx1, idx2, idx3] = omega
    phase[idx1, idx2 - 1, idx3] = omega
    phase[idx1, idx2 + 1, idx3] = omega

    phase = torch.cumsum(phase, 2, out=phase)
    x = torch.stack((torch.cos(phase), torch.sin(phase)), -1) * spec.unsqueeze(-1)
    return x.squeeze(0)
