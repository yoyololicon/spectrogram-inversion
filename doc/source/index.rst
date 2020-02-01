.. torch_specinv documentation master file, created by
   sphinx-quickstart on Thu Oct 10 21:52:34 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/yoyololicon/spectrogram-inversion

PyTorch Spectrogram Inversion Documentation
============================================

A major direction of Deep Learning in audio, especially generative models, is using features in frequency domain because
directly model raw time signal is hard.
But this require an extra process to convert the predicted spectrogram (magnitude-only in most situation) back to time domain.

To help researcher no need to care this post-precessing step, this package provide some useful and classic spectrogram
inversion algorithms. These algorithms are selected base on their performance and high parallelizability, and can even
be integrated in your model training process.

We hope this tool can serve as a standard, making fair comparison of different audio generation models.

Installation
============

PyPi
~~~~

First `Install PyTorch <https://pytorch.org/get-started/locally/>`_ with the desired cpu/gpu support and version >= 0.4.1.
Then install via pip::

   pip install torch_specinv

or::

   pip install git+https://github.com/yoyololicon/spectrogram-inversion

to get the latest version.



Getting Started
===============
The following example estimated the time signal given only the magnitude information of an audio file.

.. code-block:: python

   import torch
   import librosa
   from torch_specinv import griffin_lim
   from torch_specinv.metrics import spectral_convergence as SC

   y, sr = librosa.load(librosa.util.example_audio_file())
   y = torch.from_numpy(y)
   windowsize = 2048
   window = torch.hann_window(windowsize)
   S = torch.stft(y, windowsize, window=window)

   # discard phase information
   mag = S.pow(2).sum(2).sqrt()

   # move to gpu memory for faster computation
   mag = mag.cuda()

   yhat = griffin_lim(mag, maxiter=100, alpha=0.3, window=window)

   # check convergence
   mag_hat = torch.stft(yhat, windowsize, window=window).pow(2).sum(2).sqrt()
   print(SC(mag_hat, mag))

Reconstruct from other spectral representation:

.. code-block:: python

   from librosa.filters import mel
   from torch_specinv import L_BFGS

   filter_banks = torch.from_numpy(mel(sr, windowsize)).cuda()

   def trsfn(x):
      S = torch.stft(x, windowsize, window=window).pow(2).sum(2).sqrt()
      mel_S = filter_banks @ S
      return torch.log1p(mel_S)

   y = y.cuda()
   mag = trsfn(y)
   yhat = L_BFGS(mag, trsfn, len(y))

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/methods
   modules/metrics

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`