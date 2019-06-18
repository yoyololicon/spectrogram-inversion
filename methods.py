import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lbfgs import LBFGS
from tqdm import tqdm
from functools import partial
import numpy as np

pi2 = 2 * np.pi

from metrics import spectral_convergence, SNR, SER


def L_BFGS(spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, **kwargs):
    """
    Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6949659

    Parameters
    ----------
    spec
    transform_fn
    samples
    init_x0
    maxiter
    tol
    verbose
    evaiter
    kwargs

    Returns
    -------

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

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            optimizer.step(closure)

            if i % evaiter == evaiter - 1:
                with torch.no_grad():
                    V = transform_fn(x)
                    c = spectral_convergence(V, T).item()
                    l2_loss = criterion(V, T).item()
                    snr = SNR(V, T).item()
                    pbar.set_postfix(snr=snr, spectral_convergence=c, loss=l2_loss)
                    pbar.update(evaiter)

                    if not init_loss:
                        init_loss = l2_loss
                    elif (previous_loss - l2_loss) / init_loss < tol * evaiter:
                        break
                    previous_loss = l2_loss

    return x.detach()


@torch.no_grad()
def SPSI(spec, **kwargs):
    device = spec.device
    dtype = spec.dtype
    internal_dict = {'win_length': None, 'window': None, 'hop_length': None, 'center': True, 'normalized': False,
                     'onesided': True}
    for key, item in internal_dict.items():
        try:
            internal_dict[key] = kwargs[key]
        except:
            pass
    win_length, window, hop_length, center, normalized, onesided = tuple(internal_dict.values())

    if onesided:
        n_fft = (spec.shape[0] - 1) * 2
    else:
        n_fft = spec.shape[0]

    if not win_length:
        win_length = n_fft

    if not hop_length:
        hop_length = n_fft // 4

    if window is None:
        coeff = hop_length / win_length
    else:
        coeff = hop_length / window.pow(2).sum()

    offset = (n_fft - win_length) // 2
    conv_weight = torch.eye(win_length, dtype=dtype, device=device).unsqueeze(1)

    phase = torch.zeros_like(spec)

    def peak_picking(x):
        mask = (x[1:-1] > x[2:]) & (x[1:-1] > x[:-2])
        return F.pad(mask, [0, 0, 1, 1])

    mask = peak_picking(spec)
    b = torch.masked_select(spec, mask)
    a = torch.masked_select(spec[:-1], mask[1:])
    r = torch.masked_select(spec[1:], mask[:-1])
    b_peaks = torch.nonzero(mask).t()
    p = 0.5 * (a - r) / (a - 2 * b + r)
    omega = pi2 * (b_peaks[0].float() + p) / n_fft * hop_length
    pi_shift = p > 0

    def unwarp(x):
        x %= pi2
        return torch.where(x > np.pi, x - pi2, x)

    idx1, idx2 = b_peaks.unbind()
    phase[idx1, idx2] = omega
    phase[idx1 - 1, idx2] = omega
    phase[idx1 + 1, idx2] = omega

    phase = torch.cumsum(phase, 1, out=phase)
    # phase = unwarp(phase)

    x = torch.stack((torch.cos(phase), torch.sin(phase)), 2) * spec.unsqueeze(2)
    x = torch.irfft(x.transpose(0, 1), 1, normalized=normalized, onesided=onesided,
                    signal_sizes=[n_fft] if onesided else None)[:, offset:offset + win_length] * coeff
    if window is not None:
        x *= window
    x = x.t().unsqueeze(0)
    x = F.conv_transpose1d(x, conv_weight, stride=hop_length).view(-1)
    if center:
        x = x[win_length // 2:-win_length // 2]

    return x


@torch.no_grad()
def ADMM(spec, maxiter=1000, tol=1e-6, rho=0.1, decay=False, gamma=1.3, verbose=1, evaiter=10, **kwargs):
    """
    Paper: https://pdfs.semanticscholar.org/14bc/876fae55faf5669beb01667a4f3bd324a4f1.pdf

    Parameters
    ----------
    spec
    maxiter
    tol
    verbose
    evaiter
    hop_length
    win_length
    window
    center
    pad_mode
    normalized
    onesided

    Returns
    -------

    """
    device = spec.device
    dtype = spec.dtype
    internal_dict = {'win_length': None, 'window': None, 'hop_length': None, 'center': True, 'normalized': False,
                     'onesided': True}
    for key, item in internal_dict.items():
        try:
            internal_dict[key] = kwargs[key]
        except:
            pass
    win_length, window, hop_length, center, normalized, onesided = tuple(internal_dict.values())

    if onesided:
        n_fft = (spec.shape[0] - 1) * 2
    else:
        n_fft = spec.shape[0]

    if not win_length:
        win_length = n_fft

    if not hop_length:
        hop_length = n_fft // 4

    if window is None:
        coeff = hop_length / win_length
    else:
        coeff = hop_length / window.pow(2).sum()

    offset = (n_fft - win_length) // 2
    conv_weight = torch.eye(win_length, dtype=dtype, device=device).unsqueeze(1)

    def istft(x):
        x = torch.irfft(x.transpose(0, 1), 1, normalized=normalized, onesided=onesided,
                        signal_sizes=[n_fft] if onesided else None)[:, offset:offset + win_length] * coeff
        if window is not None:
            x *= window
        x = x.t().unsqueeze(0)
        x = F.conv_transpose1d(x, conv_weight, stride=hop_length).view(-1)
        if center:
            x = x[win_length // 2:-win_length // 2]
        return x

    x = SPSI(spec, **kwargs)
    X = torch.stft(x, n_fft, **kwargs)
    #X = torch.stack((spec, torch.zeros_like(spec)), -1)
    #x = istft(X)
    Z = X.clone()
    Y = X.clone()
    U = torch.zeros_like(X)
    if decay:
        rho = torch.linspace(rho ** (1 / gamma), 1, maxiter) ** gamma
    else:
        rho = torch.ones(maxiter) * rho

    criterion = nn.MSELoss()
    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            # Pc2
            X[:] = Z - U
            mag = X.pow(2).sum(2).sqrt()
            X *= (spec / F.threshold_(mag, 1e-7, 1e-7)).unsqueeze(-1)

            Y[:] = X + U
            # Pc1
            x[:] = istft(Y)
            reconstruted = torch.stft(x, n_fft, **kwargs)

            Z[:] = (rho[i] * Y + reconstruted) / (1 + rho[i])
            U += X - Z

            if i % evaiter == evaiter - 1:
                mag = reconstruted.pow(2).sum(2).sqrt()
                c = spectral_convergence(mag, spec).item()
                l2_loss = criterion(mag, spec).item()
                snr = SNR(mag, spec).item()
                ser = SER(mag, spec).item()
                pbar.set_postfix(snr=snr, ser=ser, spectral_convergence=c, loss=l2_loss)
                pbar.update(evaiter)

                if not init_loss:
                    init_loss = l2_loss
                elif (previous_loss - l2_loss) / init_loss < tol * evaiter and previous_loss > l2_loss:
                    break
                previous_loss = l2_loss

    return istft(X)


@torch.no_grad()
def griffin_lim(spec, maxiter=1000, tol=1e-6, alpha=0.99, verbose=1, evaiter=10, **kwargs):
    """
    Paper: https://pdfs.semanticscholar.org/14bc/876fae55faf5669beb01667a4f3bd324a4f1.pdf

    Parameters
    ----------
    spec
    maxiter
    tol
    verbose
    evaiter
    hop_length
    win_length
    window
    center
    pad_mode
    normalized
    onesided

    Returns
    -------

    """
    device = spec.device
    dtype = spec.dtype
    internal_dict = {'win_length': None, 'window': None, 'hop_length': None, 'center': True, 'normalized': False,
                     'onesided': True}
    for key, item in internal_dict.items():
        try:
            internal_dict[key] = kwargs[key]
        except:
            pass
    win_length, window, hop_length, center, normalized, onesided = tuple(internal_dict.values())

    if onesided:
        n_fft = (spec.shape[0] - 1) * 2
    else:
        n_fft = spec.shape[0]

    if not win_length:
        win_length = n_fft

    if not hop_length:
        hop_length = n_fft // 4

    if window is None:
        coeff = hop_length / win_length
    else:
        coeff = hop_length / window.pow(2).sum()

    offset = (n_fft - win_length) // 2
    conv_weight = torch.eye(win_length, dtype=dtype, device=device).unsqueeze(1)

    def istft(x):
        x = torch.irfft(x.transpose(0, 1), 1, normalized=normalized, onesided=onesided,
                        signal_sizes=[n_fft] if onesided else None)[:, offset:offset + win_length] * coeff
        if window is not None:
            x *= window
        x = x.t().unsqueeze(0)
        x = F.conv_transpose1d(x, conv_weight, stride=hop_length).view(-1)
        if center:
            x = x[win_length // 2:-win_length // 2]
        return x

    new_spec = torch.stack((spec, torch.zeros_like(spec)), -1)
    pre_spec = new_spec.clone()
    # x = istft(new_spec)
    x = SPSI(spec, **kwargs)

    criterion = nn.MSELoss()
    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            new_spec[:] = torch.stft(x, n_fft, **kwargs)
            new_spec += alpha * (new_spec - pre_spec)
            pre_spec.copy_(new_spec)

            mag = new_spec.pow(2).sum(2).sqrt()
            new_spec *= (spec / F.threshold_(mag, 1e-7, 1e-7)).unsqueeze(-1)
            x[:] = istft(new_spec)

            if i % evaiter == evaiter - 1:
                c = spectral_convergence(mag, spec).item()
                l2_loss = criterion(mag, spec).item()
                snr = SNR(mag, spec).item()
                ser = SER(mag, spec).item()
                pbar.set_postfix(snr=snr, ser=ser, spectral_convergence=c, loss=l2_loss)
                pbar.update(evaiter)

                if not init_loss:
                    init_loss = l2_loss
                elif (previous_loss - l2_loss) / init_loss < tol * evaiter and previous_loss > l2_loss:
                    break
                previous_loss = l2_loss

    return x


@torch.no_grad()
def RTISI_LA(spec, look_ahead=-1, asymmetric_window=False, maxiter=25, alpha=0.9, verbose=1, hop_length=None,
             win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True):
    """
    Paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.420.3256&rep=rep1&type=pdf

    Parameters
    ----------
    spec
    look_ahead
    maxiter
    verbose
    hop_length
    win_length
    window
    center
    pad_mode
    normalized
    onesided

    Returns
    -------

    """
    device = spec.device
    dtype = spec.dtype

    if onesided:
        n_fft = (spec.shape[0] - 1) * 2
    else:
        n_fft = spec.shape[0]

    if not win_length:
        win_length = n_fft

    if not hop_length:
        hop_length = n_fft // 4

    if window is None:
        coeff = hop_length / win_length
    else:
        coeff = hop_length / window.pow(2).sum()

    offset = (n_fft - win_length) // 2
    conv_weight = torch.eye(win_length, dtype=dtype, device=device).unsqueeze(1)
    num_keep = (win_length - 1) // hop_length
    if look_ahead < 0:
        look_ahead = num_keep

    asym_window1 = spec.new_zeros(win_length)
    for i in range(num_keep):
        asym_window1[(i + 1) * hop_length:] += window.flip(0)[:-(i + 1) * hop_length:]
    asym_window1 *= hop_length / (asym_window1 * window).sum() / coeff

    asym_window2 = spec.new_zeros(win_length)
    for i in range(num_keep + 1):
        asym_window2[i * hop_length:] += window.flip(0)[:-i * hop_length if i else None]
    asym_window2 *= hop_length / (asym_window2 * window).sum() / coeff

    steps = spec.shape[1]
    xt = spec.new_zeros(steps + num_keep + 2 * look_ahead, n_fft)
    xt_winview = xt[:, offset:offset + win_length]
    spec = F.pad(spec, [look_ahead, look_ahead])

    def irfft(x):
        return torch.irfft(x, 1, normalized=normalized, onesided=onesided, signal_sizes=[n_fft] if onesided else None)

    def rfft(x):
        return torch.rfft(x, 1, normalized=normalized, onesided=onesided)

    def ola(x):
        if window is not None:
            x = x * window.unsqueeze(-1)
        return F.conv_transpose1d((x * coeff).unsqueeze(0), conv_weight, stride=hop_length).view(-1)

    # initialize first frame with zero phase
    first_frame = spec[:, look_ahead]
    xt_winview[num_keep + look_ahead] = irfft(torch.stack((first_frame, torch.zeros_like(first_frame)), -1))[
                                        offset:offset + win_length]

    with tqdm(total=steps + look_ahead, disable=not verbose) as pbar:
        for i in range(steps + look_ahead):
            for j in range(maxiter):
                x = ola(xt_winview[i:i + num_keep + look_ahead + 1].t())
                if asymmetric_window:
                    xt_winview[i + num_keep:i + num_keep + look_ahead + 1] = \
                        x.unfold(0, win_length, hop_length)[num_keep:]

                    xt_winview[i + num_keep:i + num_keep + look_ahead] *= window
                    if j:
                        xt_winview[i + num_keep + look_ahead] *= asym_window2
                    else:
                        xt_winview[i + num_keep + look_ahead] *= asym_window1

                    new_spec = rfft(xt[i + num_keep:i + num_keep + look_ahead + 1]).transpose(0, 1)
                else:
                    new_spec = torch.stft(F.pad(x[num_keep * hop_length - offset:], [0, offset]),
                                          n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                                          center=False, pad_mode=pad_mode, normalized=normalized, onesided=onesided)

                if j or j:
                    new_spec += alpha * (new_spec - pre_spec)
                    pre_spec.copy_(new_spec)
                else:
                    pre_spec = new_spec.clone()

                mag = F.threshold_(new_spec.pow(2).sum(2).sqrt(), 1e-7, 1e-7)
                new_spec *= (spec[:, i:i + look_ahead + 1] / mag).unsqueeze(-1)

                xt_winview[i + num_keep:i + num_keep + look_ahead + 1] = irfft(new_spec.transpose(0, 1))[:,
                                                                         offset:offset + win_length]

            pbar.update()

    x = ola(xt_winview[num_keep + look_ahead:-look_ahead if look_ahead else None].t())
    if center:
        x = x[win_length // 2:-win_length // 2]
    else:
        x = F.pad(x, [offset, offset])

    return x


if __name__ == '__main__':
    import librosa
    from librosa import display
    import matplotlib.pyplot as plt
    import numpy as np

    nfft = 1024
    winsize = 1024
    hopsize = 128

    y, sr = librosa.load(librosa.util.example_audio_file(), duration=30)
    # librosa.output.write_wav('origin.wav', y, sr)
    y = torch.Tensor(y).cuda()
    window = torch.hann_window(winsize).cuda()


    def spectrogram(x, *args, p=1, **kwargs):
        return torch.stft(x, *args, **kwargs).pow(2).sum(2).add_(1e-7).pow(p / 2)


    arg_dict = {
        'win_length': winsize,
        'window': window,
        'hop_length': hopsize,
        'pad_mode': 'reflect',
        'onesided': True,
        'normalized': False,
        'center': True
    }

    spec = spectrogram(y, nfft, **arg_dict)
    # func = partial(spectrogram, p=2 / 3, n_fft=1024, window=window)
    # mag = spec.pow(0.5).cpu().numpy()
    # phase = np.random.uniform(-np.pi, np.pi, mag.shape)
    # _, init_x = istft(mag * np.exp(1j * phase), noverlap=1024 - 256)

    # estimated = L_BFGS(spec, func, len(y), maxiter=50, lr=1, history_size=10, evaiter=5)
    estimated = griffin_lim(spec, maxiter=100, alpha=0.3, **arg_dict)
    #estimated = ADMM(spec, maxiter=1000, rho=0., tol=0, decay=True, gamma=3, **arg_dict)
    # arg_dict['hop_length'] = 333
    #estimated = RTISI_LA(spec, maxiter=4, look_ahead=3, asymmetric_window=True, **arg_dict)
    #estimated = SPSI(spec, **arg_dict)
    estimated_spec = spectrogram(estimated, nfft, **arg_dict)
    display.specshow(librosa.amplitude_to_db(estimated_spec.cpu().numpy(), ref=np.max), y_axis='log')
    plt.show()

    print(SNR(estimated_spec, spec).item(), SER(estimated_spec, spec).item(),
          spectral_convergence(estimated_spec, spec).item())

    librosa.output.write_wav('test.wav', estimated.cpu().numpy(), sr)
