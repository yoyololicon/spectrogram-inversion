import torch


def sc(input, target):
    r"""
    The Spectral Convergence score is calculated as follow:

    .. math::
        \mathcal{C}(\mathbf{\hat{S}}, \mathbf{S})=\frac{\|\mathbf{\hat{S}}-\mathbf{S}\|_{Fro}}{\|\mathbf{S}\|_{Fro}}

    Returns:
        scalar output in db scale.
    """
    return 20 * ((input - target).norm().log10() - target.norm().log10())


def snr(input, target):
    r"""
    The Signal-to-Noise Ratio (SNR) is calculated as follow:

    .. math::
        SNR(\mathbf{\hat{S}}, \mathbf{S})=
        10\log_{10}\frac{1}{\sum (\frac{\hat{s}_i}{\|\mathbf{\hat{S}}\|_{Fro}} - \frac{s_i}{\|\mathbf{S}\|_{Fro}})^2}

    Returns:
        scalar output.
    """
    norm = target.norm()
    return -10 * (input / norm - target / norm).pow(2).sum().log10()


def ser(input, target):
    r"""
    The Signal-to-Error Ratio (SER) is calculated as follow:

    .. math::
        SER(\mathbf{\hat{S}}, \mathbf{S})=
        10\log_{10}\frac{\sum \hat{s}_i^2}{\sum (\hat{s}_i - s_i)^2}

    Returns:
        scalar output.
    """
    return 10 * (input.pow(2).sum().log10() - (input - target).pow(2).sum().log10())
