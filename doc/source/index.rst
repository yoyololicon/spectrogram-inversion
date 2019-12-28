.. torch_specinv documentation master file, created by
   sphinx-quickstart on Thu Oct 10 21:52:34 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyTorch Spectrogram Inversion Package
=========================================

A major direction of Deep Learning in audio, especially generative models, is using features in frequency domain as
directly model raw time signal is hard.
But this require an extra process to convert the predicted spectrogram (magnitude-only in most situation) back to time domain.

To help researcher no need to care this post-precessing step, this package provide some useful and classic spectrogram
inversion algorithms. These algorithms are selected base on their performance and high parallelizability, and can even
be integrated in your model training process.

We hope this tools can serve as a standard, making comparison of different audio generation models more fairly.

Getting started
---------------
.. toctree::
    :maxdepth: 1

    tutorial

API documentation
-----------------

.. toctree::
    :maxdepth: 1

    methods
    metrics

* :ref:`genindex`