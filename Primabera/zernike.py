import numpy as np
import math
from math import factorial as fac
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

npix = 2049    # Change this if you change this parameter in beam modelling

# =====================================================================
# ------------------ ZERNIKE POLYNOMIAL UTILITIES ---------------------
# =====================================================================

def zernike_rad(m: int, n: int, rho: np.ndarray) -> np.ndarray:
    """
    Compute the radial part R_n^m(rho) of a Zernike polynomial.

    Parameters
    ----------
    m : int
        Azimuthal frequency (|m| ≤ n, same parity as n).
    n : int
        Radial degree (n ≥ 0).
    rho : ndarray
        Radial coordinate array, with values in [0, 1].

    Returns
    -------
    ndarray
        Radial polynomial evaluated at each rho.
    """
    m, n = int(m), int(n)

    # Parity condition: R_n^m is zero if (n - m) is odd
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)

    def term_coeff(k: int) -> float:
        """Compute coefficient of the k-th term in the expansion."""
        return ((-1) ** k) * fac(n - k) / (
            fac(k) * fac((n + m) // 2 - k) * fac((n - m) // 2 - k)
        )

    # Radial series expansion
    terms = [term_coeff(k) * rho ** (n - 2 * k) for k in range((n - m) // 2 + 1)]
    return sum(terms)


def zernike(m: int, n: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute the full Zernike polynomial Z_n^m(rho, phi).

    Z_n^m(rho, phi) = R_n^m(rho) * cos(m*phi) or sin(|m|*phi)

    Parameters
    ----------
    m : int
        Azimuthal order.
    n : int
        Radial order.
    rho, phi : ndarray
        Polar coordinate grids (0 ≤ rho ≤ 1, 0 ≤ phi < 2π).

    Returns
    -------
    ndarray
        Zernike polynomial evaluated on (rho, phi).
    """
    if m > 0:
        return zernike_rad(m, n, rho) * np.cos(m * phi)
    elif m < 0:
        return zernike_rad(-m, n, rho) * np.sin(-m * phi)
    else:
        return zernike_rad(0, n, rho)


def noll_to_zern(j: int):
    """
    Convert a Noll index (j) to Zernike indices (n, m).

    Reference: https://oeis.org/A176988

    Parameters
    ----------
    j : int
        Noll index (0-based).

    Returns
    -------
    (n, m) : tuple of int
        Radial (n) and azimuthal (m) indices.
    """
    j += 1  # convert to 1-based indexing
    n = 0
    j1 = j - 1

    while j1 > n:
        n += 1
        j1 -= n

    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return n, m


def zernikel(j: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute Zernike polynomial given a Noll index.

    Parameters
    ----------
    j : int
        Noll index (0-based).
    rho, phi : ndarray
        Polar coordinates.

    Returns
    -------
    ndarray
        Zernike polynomial for index j.
    """
    n, m = noll_to_zern(j)
    return zernike(m, n, rho, phi)


# =====================================================================
# -------------------- GRID / UNIT DISK CREATION ----------------------
# =====================================================================

def unit_disk(npix: int):
    """
    Create a polar coordinate grid (rho, phi, mask) within a unit disk.

    Parameters
    ----------
    npix : int
        Number of pixels along one dimension (grid is npix x npix).

    Returns
    -------
    grid_rho, grid_phi, grid_mask : ndarray
        Radial coordinates, azimuthal angles, and disk mask (rho ≤ 1).
    """
    grid = (np.indices((npix, npix), dtype=float) - npix / 2) / (npix / 2.0)
    grid_rho = np.sqrt(grid[0]**2 + grid[1]**2)
    grid_phi = np.arctan2(grid[0], grid[1])
    grid_mask = grid_rho <= 1
    return grid_rho, grid_phi, grid_mask


# =====================================================================
# ---------------- DECOMPOSITION AND RECONSTRUCTION -------------------
# =====================================================================

def decompose(img: np.ndarray, N: int) -> np.ndarray:
    """
    Decompose an image into its Zernike coefficients.

    Parameters
    ----------
    img : ndarray
        Input 2D image.
    N : int
        Number of Zernike modes to use.

    Returns
    -------
    coeffs : ndarray
        Zernike decomposition coefficients (length = N).
    """
    grid_rho, grid_phi, grid_mask = unit_disk(img.shape[0])
    basis = [zernikel(i, grid_rho, grid_phi) * grid_mask for i in range(N)]

    # Compute covariance between all basis functions
    cov_mat = np.array([[np.sum(zerni * zernj) for zerni in basis] for zernj in basis])
    cov_inv = np.linalg.pinv(cov_mat)

    # Compute inner products (projection of image onto basis)
    proj = np.array([np.sum(img * zerni) for zerni in basis])

    return cov_inv @ proj


def truncate(coeffs: np.ndarray, sortedindex: np.ndarray, thresh: int):
    """
    Retain only the largest-magnitude Zernike coefficients.

    Parameters
    ----------
    coeffs : ndarray
        All coefficients.
    sortedindex : ndarray
        Sorted coefficient indices (descending by magnitude).
    thresh : int
        Number of coefficients to keep.

    Returns
    -------
    recon_coeffs, selected_index : ndarray
        Truncated coefficients and their indices.
    """
    return coeffs[sortedindex[:thresh]], sortedindex[:thresh]


def reconstruct(dcom: np.ndarray, ind: np.ndarray, img: np.ndarray):
    """
    Reconstruct an image from truncated Zernike coefficients.

    Parameters
    ----------
    dcom : ndarray
        Selected coefficients.
    ind : ndarray
        Indices of selected coefficients.
    img : ndarray
        Reference image (for grid shape).

    Returns
    -------
    recon : ndarray
        Reconstructed image.
    """
    grid_rho, grid_phi, grid_mask = unit_disk(img.shape[0])
    recon = np.sum(val * zernikel(i, grid_rho, grid_phi) * grid_mask
                   for i, val in zip(ind, dcom))
    return recon


# =====================================================================
# ------------------- MATRIX DECOMP/RECON WRAPPERS --------------------
# =====================================================================

def decom_matrix(J: np.ndarray, modes: int = 200) -> np.ndarray:
    """
    Decompose a Jones matrix or image into Zernike modes.

    Parameters
    ----------
    J : ndarray
        Input image or 4D Jones matrix.
    modes : int
        Number of modes.

    Returns
    -------
    coef : ndarray
        Complex Zernike coefficients for each Jones component.
    """
    if J.ndim == 2:
        return np.array([decompose(J.real, modes), decompose(J.imag, modes)])

    coef = np.empty((2, 2, 2, modes))
    for i in range(2):
        for j in range(2):
            data = J[:, :, j, i]
            coef[i][j] = [decompose(data.real, modes), decompose(data.imag, modes)]
    return coef


def selected_coefs(coef: np.ndarray, sortind, thresh: int):
    """
    Select top coefficients for all Jones matrix components.

    Parameters
    ----------
    coef : ndarray
        Coefficient array (2D or 4D).
    sortind : list of arrays
        Sorted indices for truncation.
    thresh : int
        Number of coefficients to keep.

    Returns
    -------
    coef_compiled : ndarray
        Truncated coefficient arrays and their indices.
    """
    if coef.ndim == 2:
        recon_Cr, ind_Cr = truncate(coef[0], sortind[0], thresh)
        recon_Ci, ind_Ci = truncate(coef[1], sortind[1], thresh)
        return np.array([[ind_Cr, ind_Ci], [recon_Cr, recon_Ci]])

    coef_compiled = np.empty((2, 2, 2, 2, thresh))
    for i in range(2):
        for j in range(2):
            recon_Cr, ind_Cr = truncate(coef[i][j][0], sortind[i][j][0], thresh)
            recon_Ci, ind_Ci = truncate(coef[i][j][1], sortind[i][j][1], thresh)
            coef_compiled[i][j] = [[ind_Cr, ind_Ci], [recon_Cr, recon_Ci]]
    return coef_compiled


def recon_matrix(data: np.ndarray, coef: np.ndarray, sortind=None):
    """
    Reconstruct original data from Zernike coefficients.

    Parameters
    ----------
    data : ndarray
        Original image.
    coef : ndarray
        Truncated coefficients.
    sortind : list of arrays, optional
        Sorted indices.

    Returns
    -------
    Jz : list of ndarray
        [Original amplitude, reconstructed amplitude]
    """
    thresh = coef.shape[-1]

    if coef.ndim == 3:
        Mr = reconstruct(coef[1][0], coef[0][0], data)
        Mi = reconstruct(coef[1][1], coef[0][1], data)
    else:
        sortedindex_r = range(thresh) if sortind is None else sortind[0]
        sortedindex_i = range(thresh) if sortind is None else sortind[1]
        Mr = reconstruct(coef[0], sortedindex_r, data)
        Mi = reconstruct(coef[1], sortedindex_i, data)

    mod = np.abs(Mr + 1j * Mi)
    rho, phi, mask = unit_disk(mod.shape[0])
    dat = np.abs(data) * mask
    return [dat, mod]


# =====================================================================
# -------------------------- VISUALIZATION ----------------------------
# =====================================================================

def jones_plot(Jz, r, c, patch_deg=15):
    """
    Plot comparison between original, reconstructed, and residual Jones components.

    Parameters
    ----------
    Jz : list
        [Original, reconstructed] images.
    r, c : int
        Jones matrix indices.
    """
    res = np.abs(Jz[0] - Jz[1])
    imgs = 20 * np.log10(np.array([Jz[0], Jz[1], res]))
    labels = ['Hamaker', 'Zernike', 'Residual']

    fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    px = 180. / npix
    R = int(patch_deg / px)
    x = np.linspace(-patch_deg, patch_deg, R * 2)

    # Top: images with colorbars
    for i in range(3):
        ax[0, i].axis('off')
        im = ax[0, i].imshow(imgs[i, ...], vmin=-70, vmax=0)
        div = make_axes_locatable(ax[0, i])
        cax = div.append_axes("top", size="5%", pad=0.1)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.set_xlabel(f"{labels[i]} $J_{{{r}{c}}}$")
        cb.ax.xaxis.set_label_position('top')

    # Bottom: 1D profiles
    for i in range(3):
        ax[1, i].plot(x, imgs[i][R, :], label='Horizontal')
        ax[1, i].plot(x, imgs[i][:, R], label='Vertical')
        ax[1, i].plot(x, np.diagonal(imgs[i][:, :]), label='Diagonal')
        ax[1, i].set_title(f"$J_{{{r}{c}}}$, {labels[i]}")
        ax[1, i].set_xlabel('Distance from NCP [deg]')
        ax[1, i].set_ylim([-60, 1])
        ax[1, i].set_xlim([-15, 15])
        ax[1, i].grid(alpha=.6)
        ax[1, i].legend()

    plt.show()
