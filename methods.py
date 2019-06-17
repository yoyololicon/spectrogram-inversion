import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lbfgs import LBFGS
from tqdm import tqdm
from functools import partial

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
def griffin_lim(spec, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, hop_length=None, win_length=None, window=None,
                center=True, pad_mode='reflect', normalized=False, onesided=True):
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
    if onesided:
        n_fft = (spec.shape[0] - 1) * 2
    else:
        n_fft = spec.shape[0]

    if not win_length:
        win_length = n_fft

    if not hop_length:
        hop_length = win_length // 4

    if window is None:
        coeff = hop_length / win_length
    else:
        coeff = hop_length / window.pow(2).sum()
        win_length = len(window)

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
    x = istft(new_spec)

    criterion = nn.MSELoss()
    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            new_spec[:] = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                                     center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided)

            mag = new_spec.pow(2).sum(2).sqrt()
            new_spec *= (spec / F.threshold_(mag, 1e-7, 1e-7)).unsqueeze(-1)
            x[:] = istft(new_spec)

            if i % evaiter == evaiter - 1:
                c = spectral_convergence(mag, spec).item()
                l2_loss = criterion(mag, spec).item()
                snr = SNR(mag, spec).item()
                pbar.set_postfix(snr=snr, spectral_convergence=c, loss=l2_loss)
                pbar.update(evaiter)

                if not init_loss:
                    init_loss = l2_loss
                elif (previous_loss - l2_loss) / init_loss < tol * evaiter:
                    break
                previous_loss = l2_loss

    return x


@torch.no_grad()
def RTISI_LA(spec, look_ahead=-1, asymmetric_window=False, maxiter=25, verbose=1, hop_length=None, win_length=None,
             window=None, center=True, pad_mode='reflect', normalized=False, onesided=True):
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
        hop_length = win_length // 4

    if window is None:
        coeff = hop_length / win_length
    else:
        coeff = hop_length / window.pow(2).sum()
        win_length = len(window)

    offset = (n_fft - win_length) // 2
    conv_weight = torch.eye(win_length, dtype=dtype, device=device).unsqueeze(1)
    num_keep = (win_length - 1) // hop_length
    if look_ahead < 0:
        look_ahead = num_keep

    asym_window1 = spec.new_zeros(win_length)
    for i in range(num_keep):
        asym_window1[(i + 1) * hop_length:] += window.flip(0)[:-(i + 1) * hop_length:]
    asym_window1 = asym_window1 / coeff

    asym_window2 = spec.new_zeros(win_length)
    for i in range(num_keep + 1):
        asym_window2[i * hop_length:] += window.flip(0)[:-i * hop_length if i else None]
    asym_window2 = asym_window2 / coeff

    steps = spec.shape[1]
    xt = spec.new_zeros(steps + num_keep + 2 * look_ahead, n_fft)
    xt_winview = xt[:, offset:offset + win_length]
    spec = F.pad(spec, [look_ahead, look_ahead])

    def irfft(x):
        return torch.irfft(x, 1, normalized=normalized, onesided=onesided, signal_sizes=[n_fft] if onesided else None)

    def rfft(x):
        return torch.rfft(x, 1, normalized=normalized, onesided=onesided)

    def transpose_conv(x):
        if window is not None:
            x = x * window.unsqueeze(-1)
        return F.conv_transpose1d((x * coeff).unsqueeze(0), conv_weight, stride=hop_length).view(-1)

    # initialize first frame with zero phase
    first_frame = spec[:, look_ahead]
    xt[num_keep + look_ahead] = irfft(torch.stack((first_frame, torch.zeros_like(first_frame)), -1))

    with tqdm(total=steps + look_ahead, disable=not verbose) as pbar:
        for i in range(steps + look_ahead):
            for j in range(maxiter):
                x = transpose_conv(xt_winview[i:i + num_keep + look_ahead + 1].t())
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
                    new_spec = torch.stft(x[num_keep * hop_length:(num_keep + look_ahead) * hop_length + win_length],
                                          n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                                          center=False, pad_mode=pad_mode, normalized=normalized, onesided=onesided)
                mag = F.threshold_(new_spec.pow(2).sum(2).sqrt(), 1e-7, 1e-7)
                new_spec *= (spec[:, i:i + look_ahead + 1] / mag).unsqueeze(-1)
                xt[i + num_keep:i + num_keep + look_ahead + 1] = irfft(new_spec.transpose(0, 1))

            pbar.update()

    x = transpose_conv(xt_winview[num_keep + look_ahead:-look_ahead if look_ahead else None].t())
    if center:
        x = x[win_length // 2:-win_length // 2]

    return x


if __name__ == '__main__':
    import librosa
    from librosa import display
    import matplotlib.pyplot as plt

    y, sr = librosa.load(librosa.util.example_audio_file(), duration=20)
    # librosa.output.write_wav('origin.wav', y, sr)
    y = torch.Tensor(y).cuda()
    window = torch.blackman_window(1024).cuda()


    def spectrogram(x, *args, p=1, **kwargs):
        return torch.stft(x, *args, **kwargs).pow(2).sum(2).add_(1e-7).pow(p / 2)


    func = partial(spectrogram, p=1, n_fft=1024, window=window)

    spec = func(y)
    # mag = spec.pow(0.5).cpu().numpy()
    # phase = np.random.uniform(-np.pi, np.pi, mag.shape)
    # _, init_x = istft(mag * np.exp(1j * phase), noverlap=1024 - 256)

    # estimated = L_BFGS(spec, func, len(y), maxiter=50, lr=1, history_size=10, evaiter=5)
    # estimated = griffin_lim(spec, maxiter=200, window=window)
    estimated = RTISI_LA(spec, maxiter=20, look_ahead=3, asymmetric_window=True, window=window)
    estimated_spec = func(estimated)
    display.specshow(estimated_spec.cpu().numpy(), y_axis='log')
    plt.show()

    print(SNR(estimated_spec, spec).item(), spectral_convergence(estimated_spec, spec).item(),
          SER(estimated_spec, spec).item())

    librosa.output.write_wav('test.wav', estimated.cpu().numpy(), sr)
