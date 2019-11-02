import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

import torch
from torch_specinv import *

if __name__ == '__main__':

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

    #spec = spectrogram(y, nfft, **arg_dict)
    func = partial(spectrogram, n_fft=nfft, **arg_dict)
    spec = func(y)
    # mag = spec.pow(0.5).cpu().numpy()
    # phase = np.random.uniform(-np.pi, np.pi, mag.shape)
    # _, init_x = istft(mag * np.exp(1j * phase), noverlap=1024 - 256)

    #estimated = L_BFGS(spec, func, len(y), maxiter=50, lr=1, history_size=10, evaiter=5)
    #estimated = griffin_lim(spec, maxiter=100, alpha=0.3, **arg_dict)
    #estimated = ADMM(spec, maxiter=100, rho=0.2, **arg_dict)
    # arg_dict['hop_length'] = 333
    estimated = RTISI_LA(spec, maxiter=4, look_ahead=3, asymmetric_window=True, **arg_dict)
    #estimated = SPSI(spec, **arg_dict)
    # arg_dict.pop('window')
    # estimated = PGHI(spec, **arg_dict)
    estimated_spec = func(estimated)
    #estimated_spec = estimated.pow(2).sum(2).sqrt()
    display.specshow(librosa.amplitude_to_db(estimated_spec.cpu().numpy(), ref=np.max), y_axis='log')
    plt.show()

    print(spectral_convergence(estimated_spec, spec))

    #librosa.output.write_wav('test.wav', estimated.cpu().numpy(), sr)