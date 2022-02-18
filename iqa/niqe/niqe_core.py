import os
import math

import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import gamma

from .utils import imresize


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NIQE_PRIS_PARAMS = np.load(os.path.join(ROOT_DIR, 'niqe_pris_params.npz'))


def estimate_aggd_param(block: torch.Tensor) -> tuple:
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Args:
        block (torch.Tensor): 2D image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3)
    )

    block = block.flatten()
    left_std = torch.sqrt(torch.mean(block[block < 0] ** 2)).item()
    right_std = torch.sqrt(torch.mean(block[block > 0] ** 2)).item()
    gammahat = left_std / right_std
    rhat = (torch.mean(torch.abs(block))) ** 2 / torch.mean(block ** 2)
    rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / (
        (gammahat ** 2 + 1) ** 2
    )

    array_position = np.argmin((r_gam - rhatnorm.item()) ** 2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return alpha, beta_l, beta_r


def compute_feature(block: torch.Tensor) -> list:
    """Compute features.

    Args:
        block (torch.Tensor): 2D image block.

    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = torch.roll(block, shifts[i], dims=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])

    return feat


def _apply_niqe(
    img: torch.Tensor, block_size_h: int = 96, block_size_w: int = 96
) -> float:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Ref: https://github.com/xinntao/BasicSR/blob/master/basicsr/metrics/niqe.py

    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.

    For good performance, it is advisable by the official implementation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.

    Args:
        img (torch.Tensor): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255].
        block_size_h (int, optional): Height of the blocks in to which image is divided.
            Defaults to 96 (the official recommended value).
        block_size_w (int, optional): Width of the blocks in to which image is divided.
            Defaults to 96 (the official recommended value).

    Returns:
        float: NIQE value.
    """
    mu_pris_param = NIQE_PRIS_PARAMS['mu_pris_param']
    cov_pris_param = NIQE_PRIS_PARAMS['cov_pris_param']
    gaussian_window = torch.from_numpy(NIQE_PRIS_PARAMS['gaussian_window'])
    gaussian_window = gaussian_window[None, None, ...].to(img.device)

    if img.dim() != 2:
        raise ValueError(
            'Input image must be a gray or Y (of YCbCr) image with shape (h, w).'
        )

    # convert to float64
    img = img.to(torch.float64)

    # crop image
    h, w = img.size()
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0 : num_block_h * block_size_h, 0 : num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        _img = img[None, None, ...]
        mu = F.conv2d(F.pad(_img, (3, 3, 3, 3), 'replicate'), gaussian_window)
        sigma = (
            F.conv2d(F.pad(_img ** 2, (3, 3, 3, 3), 'replicate'), gaussian_window)
            - mu ** 2
        ).abs() ** 0.5
        img_nomalized = (_img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[
                    0,
                    0,
                    idx_h * block_size_h // scale : (idx_h + 1) * block_size_h // scale,
                    idx_w * block_size_w // scale : (idx_w + 1) * block_size_w // scale,
                ]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        if scale == 1:
            img = imresize(img[None, ...] / 255.0, scale=0.5, antialiasing=True)
            img = img[0] * 255.0

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    # use nancov. ref: https://ww2.mathworks.cn/help/stats/nancov.html
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param),
        np.transpose(mu_pris_param - mu_distparam),
    )
    return float(quality ** 0.5)
