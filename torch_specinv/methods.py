from .metrics import *
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.optim.lbfgs import LBFGS
from tqdm import tqdm
from functools import partial
from typing import Tuple
import math

pi2 = 2 * math.pi

_func_mapper = {
    'SC': sc,
    'SNR': snr,
    'SER': ser
}


def _args_helper(spec, **stft_kwargs):
    """A helper function to get stft arguments from the provided kwargs.

    Args:
        spec: The magnitude spectrum of size (*, freq, time).
        **stft_kwargs: Keyword arguments that computed spec from 'torch.stft'.
        See `torch.stft` for details.

    Returns:
        n_fft: FFT size of the spectrum.
        processed_kwargs: Dict object that stored the processed keyword arguments.

    """
    args_dict = {'win_length': None,
                 'window': None,
                 'hop_length': None,
                 'center': True,
                 'pad_mode': 'reflect',
                 'normalized': False,
                 'onesided': None,
                 'return_complex': None}
    for key, item in args_dict.items():
        try:
            args_dict[key] = stft_kwargs[key]
        except:
            pass
    win_length, window, hop_length, center, pad_mode, normalized, onesided, return_complex = tuple(
        args_dict.values())

    device = spec.device
    dtype = spec.dtype
    if dtype == torch.complex32:
        dtype = torch.float16
    elif dtype == torch.complex64:
        dtype = torch.float32
    elif dtype == torch.complex128:
        dtype = torch.float64

    if onesided is None:
        if window is not None and window.is_complex():
            onesided = False
        else:
            onesided = True

    if onesided:
        n_fft = (spec.shape[-2] - 1) * 2
    else:
        n_fft = spec.shape[-2]

    if not win_length:
        win_length = n_fft

    if not hop_length:
        hop_length = n_fft // 4

    if window is None:
        window = torch.ones(win_length, dtype=dtype, device=device)

    assert n_fft >= win_length
    if n_fft > win_length:
        window = F.pad(window, [(n_fft - win_length) // 2, (n_fft - win_length + 1) // 2])
        win_length = n_fft

    args_dict['win_length'] = win_length
    args_dict['hop_length'] = hop_length
    args_dict['window'] = window
    args_dict['return_complex'] = True
    args_dict['onesided'] = onesided

    return n_fft, args_dict


def _get_ola_weight(window):
    ola_weight = torch.diag(window).unsqueeze(1)
    return ola_weight


def _spec_formatter(spec, **stft_kwargs):
    shape = spec.shape
    assert 4 > len(shape) > 1
    if len(shape) == 2:
        spec = spec.unsqueeze(0)

    if not spec.is_complex():
        cmplx_spec = phase_init(spec, **stft_kwargs)
        target_spec = spec
    else:
        cmplx_spec = spec
        target_spec = spec.abs()
    return cmplx_spec, target_spec


def _ola(x, hop_length, weight, padding, norm_envelope=None):
    """A helper function to do overlap-and-add.

    Args:
        x: input tensor of size :math: '(batch, window_size, time)'.
        hop_length: The distance between neighboring sliding window frames.
        weight: An identity matrix of size (win_length x win_length) .
        norm_envelope: The normalized coefficient apply on synthesis window.

    Returns:
        A 1d tensor containing the overlap-and-add result.

    """
    ola_x = F.conv_transpose1d(x, weight, stride=hop_length, padding=padding).squeeze(1)
    if norm_envelope is None:
        norm_envelope = F.conv_transpose1d(torch.ones_like(
            x[:1]), weight * weight, stride=hop_length, padding=padding).squeeze()
    return ola_x / norm_envelope, norm_envelope


def _istft(x, n_fft, ola_weight,
           win_length, window, hop_length, center, normalized, onesided, pad_mode, return_complex,
           norm_envelope=None):
    """
    A helper function to do istft.
    """
    if onesided:
        x = fft.irfft(x, n=n_fft, dim=-2, norm='ortho' if normalized else 'backward')
    else:
        x = fft.ifft(x, n=n_fft, dim=-2, norm='ortho' if normalized else 'backward').real

    x, norm_envelope = _ola(x, hop_length, ola_weight, padding=n_fft // 2 if center else 0,
                            norm_envelope=norm_envelope)
    return x, norm_envelope


def _training_loop(
        closure,
        status_dict,
        target,
        max_iter,
        tol,
        verbose,
        eva_iter,
        metric,
):
    assert eva_iter > 0
    assert max_iter > 0
    assert tol >= 0

    metric = metric.upper()
    assert metric.upper() in _func_mapper.keys()

    bar_dict = {}
    bar_dict[metric] = 0
    metric_func = _func_mapper[metric]

    criterion = F.mse_loss
    init_loss = None

    with tqdm(total=max_iter, disable=not verbose) as pbar:
        for i in range(max_iter):
            output = closure(status_dict)
            if i % eva_iter == eva_iter - 1:
                bar_dict[metric] = metric_func(output, target).item()
                l2_loss = criterion(output, target).item()
                pbar.set_postfix(**bar_dict, loss=l2_loss)
                pbar.update(eva_iter)

                if not init_loss:
                    init_loss = l2_loss
                elif (previous_loss - l2_loss) / init_loss < tol and previous_loss > l2_loss:
                    break
                previous_loss = l2_loss


def griffin_lim(spec,
                max_iter=200,
                tol=1e-6,
                alpha=0.99,
                verbose=True,
                eva_iter=10,
                metric='sc',
                **stft_kwargs):
    r"""Reconstruct spectrogram phase using the will known `Griffin-Lim`_ algorithm and its variation, `Fast Griffin-Lim`_.


    .. _`Griffin-Lim`: https://pdfs.semanticscholar.org/14bc/876fae55faf5669beb01667a4f3bd324a4f1.pdf
    .. _`Fast Griffin-Lim`: https://perraudin.info/publications/perraudin-note-002.pdf

    Args:
        spec (Tensor): the input tensor of size :math:`(N \times T)` (magnitude) or :math:`(N \times T \times 2)`
            (complex input). If a magnitude spectrogram is given, the phase will first be intialized using
            :func:`torch_specinv.methods.phase_init`; otherwise start from the complex input.
        max_iter (int): maximum number of iterations before timing out.
        tol (float): tolerance of the stopping condition base on L2 loss. Default: ``1e-6``
        alpha (float): speedup parameter used in `Fast Griffin-Lim`_, set it to zero will disable it. Default: ``0``
        verbose (bool): whether to be verbose. Default: :obj:`True`
        eva_iter (int): steps size for evaluation. After each step, the function defined in `metric` will evaluate. Default: ``10``
        metric (str): evaluation function. Currently available functions: ``'sc'`` (spectral convergence), ``'snr'`` or ``'ser'``. Default: ``'sc'``
        **stft_kwargs: other arguments that pass to :func:`torch.stft`

    Returns:
        A 1d tensor converted from the given spectrogram

    """
    assert alpha >= 0

    cmplx_spec, target_spec = _spec_formatter(spec, **stft_kwargs)
    n_fft, processed_args = _args_helper(target_spec, **stft_kwargs)
    ola_weight = _get_ola_weight(processed_args['window'])

    istft = partial(_istft, n_fft=n_fft, ola_weight=ola_weight,
                    **processed_args)

    pre_spec = cmplx_spec.clone()
    x, norm_envelope = istft(cmplx_spec)

    lr = alpha / (1 + alpha)

    def closure(status_dict):
        x = status_dict['x']
        pre_spec = status_dict['pre_spec']

        new_spec = torch.stft(x, n_fft, **processed_args)
        output = new_spec.abs()
        new_spec = new_spec - pre_spec * lr
        status_dict['pre_spec'] = new_spec

        norm = new_spec.abs().add_(1e-16)
        new_spec = new_spec * target_spec / norm
        x, _ = istft(new_spec, norm_envelope=norm_envelope)
        status_dict['x'] = x
        return output

    stats = {
        'x': x,
        'pre_spec': pre_spec
    }

    _training_loop(
        closure,
        stats,
        target_spec,
        max_iter,
        tol,
        verbose,
        eva_iter,
        metric
    )
    x = stats['x']
    if not (spec.shape[0] == 1 and len(spec.shape) == 3):
        x = x.squeeze(0)
    return x


def RTISI_LA(spec, look_ahead=-1, asymmetric_window=False, max_iter=25, alpha=0.99, verbose=1, **stft_kwargs):
    r"""
    Reconstruct spectrogram phase using `Real-Time Iterative Spectrogram Inversion with Look Ahead`_ (RTISI-LA).

    .. _`Real-Time Iterative Spectrogram Inversion with Look Ahead`:
        https://lonce.org/home/Publications/publications/2007_RealtimeSignalReconstruction.pdf


    Args:
        spec (Tensor): the input tensor of size :math:`(N \times T)` (magnitude).
        look_ahead (int): how many future frames will be consider. ``-1`` will set it to ``(win_length - 1) / hop_length``,
            ``0`` will disable look-ahead strategy and fall back to original RTISI algorithm. Default: ``-1``
        asymmetric_window (bool): whether to apply asymmetric window on the first iteration for new coming frame.
        max_iter (int): number of iterations for each step.
        alpha (float): speedup parameter used in `Fast Griffin-Lim`_, set it to zero will disable it. Default: ``0``
        verbose (bool): whether to be verbose. Default: :obj:`True`
        **stft_kwargs: other arguments that pass to :func:`torch.stft`.

    Returns:
        A 1d tensor converted from the given spectrogram

    """
    assert max_iter > 0
    assert alpha >= 0
    assert not spec.is_complex()

    shape = spec.shape
    assert 4 > len(shape) > 1
    target_spec = spec
    if len(shape) == 2:
        target_spec = target_spec.unsqueeze(0)

    n_fft, processed_args = _args_helper(target_spec, **stft_kwargs)
    ola_weight = _get_ola_weight(processed_args['window'])

    copyed_kwargs = stft_kwargs.copy()
    copyed_kwargs['center'] = False

    win_length = processed_args['win_length']
    hop_length = processed_args['hop_length']
    onesided = processed_args['onesided']
    normalized = processed_args['normalized']

    window = processed_args['window']
    synth_coeff = hop_length / (window @ window)

    # ola_weight = ola_weight * synth_coeff

    num_keep = (win_length - 1) // hop_length
    if look_ahead < 0:
        look_ahead = num_keep

    asym_window1 = target_spec.new_zeros(win_length)
    for i in range(num_keep):
        asym_window1[(i + 1) * hop_length:] += window.flip(0)[:-(i + 1) * hop_length]
    asym_window1 *= synth_coeff

    asym_window2 = target_spec.new_zeros(win_length)
    for i in range(num_keep + 1):
        asym_window2[i * hop_length:] += window.flip(0)[:-i * hop_length if i else None]
    asym_window2 *= synth_coeff

    steps = target_spec.shape[2]
    target_spec = F.pad(target_spec, [look_ahead, look_ahead])

    if onesided:
        irfft = partial(fft.irfft, n=n_fft, dim=-2, norm='ortho' if normalized else 'backward')
        rfft = partial(fft.rfft, n=n_fft, dim=-2, norm='ortho' if normalized else 'backward')
    else:
        irfft = lambda x: fft.ifft(x, n=n_fft, dim=-2, norm='ortho' if normalized else 'backward').real
        rfft = lambda x: fft.fft(x, n=n_fft, dim=-2, norm='ortho' if normalized else 'backward')

    # initialize first frame with zero phase
    first_frame = target_spec[..., look_ahead]
    keeped_chunk = target_spec.new_zeros(target_spec.shape[0], n_fft, num_keep)
    update_chunk = target_spec.new_zeros(target_spec.shape[0], n_fft, look_ahead)
    update_chunk = torch.cat((update_chunk,
                              irfft(first_frame * (1 + 0j)).unsqueeze(2)), 2)

    lr = alpha / (1 + alpha)
    output_xt_list = []
    with tqdm(total=steps + look_ahead, disable=not verbose) as pbar:
        for i in range(steps + look_ahead):
            for j in range(max_iter):
                x, _ = _ola(torch.cat((keeped_chunk,
                                       update_chunk), 2),
                            hop_length,
                            ola_weight * synth_coeff, padding=0, norm_envelope=1)

                x = x[:, num_keep * hop_length:]
                if asymmetric_window:
                    xt_winview = x.unfold(1, win_length, hop_length)
                    xt_norm_wind = xt_winview[:, :, :-1] * window[:, None]
                    if j:
                        xt_asym_wind = xt_winview[:, :, -1:] * asym_window2[:, None]
                    else:
                        xt_asym_wind = xt_winview[:, :, -1:] * asym_window1[:, None]

                    xt = torch.cat((xt_norm_wind, xt_asym_wind), 2)
                    new_spec = rfft(xt)
                else:
                    new_spec = torch.stft(x, n_fft=n_fft, **copyed_kwargs)

                if j:
                    new_spec = new_spec - lr * pre_spec
                elif i:
                    new_spec = torch.cat(
                        (new_spec[:, :, :-1] - lr * pre_spec[:, :, 1:], new_spec[:, :, -1:]), 2)
                pre_spec = new_spec

                norm = new_spec.abs() + 1e-16
                new_spec = new_spec * target_spec[..., i:i + look_ahead + 1] / norm

                update_chunk = irfft(new_spec)

            pbar.update()
            output_xt_list.append(update_chunk[:, :, 0])
            keeped_chunk = torch.cat((keeped_chunk[:, :, 1:], update_chunk[:, :, :1]), 2)
            update_chunk = F.pad(update_chunk[:, :, 1:], [0, 1])

    all_xt = torch.stack(output_xt_list[look_ahead if look_ahead else 0:], 2)
    x, _ = _ola(all_xt, hop_length, ola_weight, padding=win_length // 2 if processed_args['center'] else 0)

    if not (spec.shape[0] == 1 and len(spec.shape) == 3):
        x = x.squeeze(0)
    return x


def ADMM(spec, max_iter=1000, tol=1e-6, rho=0.1, verbose=1, eva_iter=10, metric='sc', **stft_kwargs):
    r"""
    Reconstruct spectrogram phase using `Griffin–Lim Like Phase Recovery via Alternating Direction Method of Multipliers`_ .

    .. _`Griffin–Lim Like Phase Recovery via Alternating Direction Method of Multipliers`:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8552369

    Args:
        spec (Tensor): the input tensor of size :math:`(N \times T)` (magnitude) or :math:`(N \times T \times 2)`
            (complex input). If a magnitude spectrogram is given, the phase will first be intialized using
            :func:`torch_specinv.methods.phase_init`; otherwise start from the complex input.
        max_iter (int): maximum number of iterations before timing out.
        tol (float): tolerance of the stopping condition base on L2 loss. Default: ``1e-6``
        rho (float): non-negative speedup parameter. Small value is preferable when the input spectrogram is noisy (inperfect);
            set it to 1 will behave similar to ``griffin_lim``.  Default: ``0.1``
        verbose (bool): whether to be verbose. Default: :obj:`True`
        eva_iter (int): steps size for evaluation. After each step, the function defined in ``metric`` will evaluate. Default: ``10``
        metric (str): evaluation function. Currently available functions: ``'sc'`` (spectral convergence), ``'snr'`` or ``'ser'``. Default: ``'sc'``
        **stft_kwargs: other arguments that pass to :func:`torch.stft`.


    Returns:
        A 1d tensor converted from the given spectrogram

    """
    assert eva_iter > 0
    assert max_iter > 0
    assert tol >= 0
    assert metric.upper() in list(_func_mapper.keys())

    cmplx_spec, target_spec = _spec_formatter(spec, **stft_kwargs)
    n_fft, processed_args = _args_helper(target_spec, **stft_kwargs)
    ola_weight = _get_ola_weight(processed_args['window'])

    istft = partial(_istft, n_fft=n_fft, ola_weight=ola_weight,
                    **processed_args)

    X = cmplx_spec
    x, norm_envelope = istft(X)
    Z = X.clone()
    Y = X.clone()
    U = torch.zeros_like(X)

    def closure(status_dict):
        X = status_dict['X']
        Y = status_dict['Y']
        U = status_dict['U']
        x = status_dict['x']

        reconstruted = torch.stft(x, n_fft, **processed_args)
        output = reconstruted.abs()

        Z = (rho * Y + reconstruted) / (1 + rho)
        U = U + X - Z

        # Pc2
        X = Z - U
        norm = X.abs() + 1e-16
        X = X * target_spec / norm

        Y = X + U
        # Pc1
        x, _ = istft(Y, norm_envelope=norm_envelope)

        status_dict['Y'] = Y
        status_dict['X'] = X
        status_dict['U'] = U
        status_dict['x'] = x
        return output

    stats = {
        'Y': Y,
        'U': U,
        'X': X,
        'x': x
    }

    _training_loop(
        closure,
        stats,
        target_spec,
        max_iter,
        tol,
        verbose,
        eva_iter,
        metric
    )

    x = stats['x']
    return x.squeeze(0)


def L_BFGS(spec, transform_fn, samples=None, init_x0=None, max_iter=1000, tol=1e-6, verbose=1, eva_iter=10, metric='sc',
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
        max_iter (int): maximum number of iterations before timing out.
        tol (float): tolerance of the stopping condition base on L2 loss. Default: ``1e-6``.
        verbose (bool): whether to be verbose. Default: :obj:`True`
        eva_iter (int): steps size for evaluation. After each step, the function defined in ``metric`` will evaluate. Default: ``10``
        metric (str): evaluation function. Currently available functions: ``'sc'`` (spectral convergence), ``'snr'`` or ``'ser'``. Default: ``'sc'``
        **kwargs: other arguments that pass to :class:`torch.optim.LBFGS`.

    Returns:
        A 1d tensor converted from the given presentation
    """
    if init_x0 is None:
        init_x0 = spec.new_empty(*samples).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.MSELoss()
    optimizer = LBFGS([x], **kwargs)

    def inner_closure():
        optimizer.zero_grad()
        V = transform_fn(x)
        loss = criterion(V, T)
        loss.backward()
        return loss

    def outer_closure(status_dict):
        optimizer.step(inner_closure)
        with torch.no_grad():
            V = transform_fn(x)
        return V

    _training_loop(
        outer_closure,
        {},
        T,
        max_iter,
        tol,
        verbose,
        eva_iter,
        metric
    )

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
    assert not spec.is_complex()
    shape = spec.shape
    if len(spec.shape) == 2:
        spec = spec.unsqueeze(0)
    assert len(spec.shape) == 3

    n_fft, processed_args = _args_helper(spec, **stft_kwargs)
    hop_length = processed_args['hop_length']

    phase = torch.zeros_like(spec)

    def peak_picking(x):
        mask = (x[:, 1:-1] > x[:, 2:]) & (x[:, 1:-1] > x[:, :-2])
        return F.pad(mask, [0, 0, 1, 1])

    with torch.no_grad():
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

        phase = torch.cumsum(phase, 2)
        angle = torch.exp(phase * 1j)

    spec = spec * angle
    return spec.view(shape)
