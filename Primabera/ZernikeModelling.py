"""
Beam Modeling and Zernike Decomposition Pipeline
------------------------------------------------
Automates the full beam modeling and compression workflow for LOFAR HBA.

Main Steps:
------------
1. Generate measurement sets (MS) if they do not exist.
2. Compute full-sky Jones beams for a specified frequency range.
3. Decompose beams into Zernike modes.
4. Truncate modes based on modal energy contribution.
5. Compress spectral variation of Zernike coefficients using polynomial + sinusoidal fitting.
6. (Optional) Reconstruct beams and compute NMRSE for validation.

Author: Masuk Ridwan
"""

import os
import numpy as np
import hamaker
import losito
import zernike
import PolySin
import psutil


# =====================================================
#  Helper Function: Load or Create Memmap Arrays
# =====================================================

def load_or_create_memmap(outfile, shape, dtype, force_recreate=False, min_finite_ratio=1):
    """
    Load or create a numpy memory-mapped (.npy) file with validity and integrity checks.
    
    Parameters
    ----------
    outfile : str
        Path to the memmap file (.npy recommended).
    shape : tuple
        Expected array shape.
    dtype : np.dtype or type
        Data type (e.g., np.complex128, np.float64).
    force_recreate : bool, optional
        If True, recreates the memmap even if it exists.
    min_finite_ratio : float, optional
        Minimum fraction of finite (non-NaN, non-inf) values required.

    Returns
    -------
    arr : np.memmap
        Memory-mapped array (writable view).
    created : bool
        True if a new file was created, False if an existing valid one was reused.
    """
    def create_new():
        """Create a new, empty memmap file initialized with zeros."""
        print(f"🆕 Creating new memmap: {outfile}")
        if os.path.exists(outfile):
            os.remove(outfile)
        arr = np.lib.format.open_memmap(outfile, mode="w+", dtype=dtype, shape=shape)
        return arr, True

    # Optionally force recreation
    if force_recreate:
        return create_new()

    # Try to reuse existing memmap
    if os.path.exists(outfile):
        try:
            arr = np.load(outfile, mmap_mode="r+")
            if arr.shape != shape or arr.dtype != np.dtype(dtype):
                print(f"⚠️  Mismatch detected — expected {shape}/{dtype}, got {arr.shape}/{arr.dtype}")
                return create_new()

            # Validate data (finite ratio check)
            print("🔍 Checking memmap data validity...")
            finite_ratio = np.isfinite(arr).sum() / arr.size
            if finite_ratio < min_finite_ratio:
                print(f"⚠️  Finite ratio {finite_ratio:.3f} below threshold {min_finite_ratio}")
                return create_new()

            print(f"✅ Loaded existing memmap: {outfile}")
            print(f"   Shape: {arr.shape}, Dtype: {arr.dtype}")
            return arr, False

        except Exception as e:
            print(f"❌ Error reading {outfile}: {e}")
            return create_new()

    # If file doesn't exist
    return create_new()


# =====================================================
#  Beam Generation
# =====================================================

def generate_beam(start_freq, end_freq, interval, ms_path, patch_deg=15, npix=2049, 
                  time=4.92183348e09, station=0, outfile="beams_memmap.npy"):
    """
    Compute and store all-sky Jones beams across a frequency range using memmap.

    Parameters
    ----------
    start_freq, end_freq : float
        Frequency range in MHz.
    interval : float
        Frequency step in MHz.
    ms_path : str
        Path to the Measurement Set (MS).
    patch_deg : float
        Angular radius (in degrees) of the extracted beam patch (default: 15°).
    npix : int
        Resolution of the all-sky beam grid (default: 2049 pixels).
    time : float
        Observation timestamp (default corresponds to fixed epoch).
    station : int
        LOFAR station ID.
    outfile : str
        Output memmap file for storing Jones matrices.

    Returns
    -------
    np.ndarray
        Array of Jones beams with shape (N_freqs, Ny, Nx, 2, 2).
    """
    freqs = np.arange(start_freq, end_freq + 1, interval)
    print("\n=== Beam Generation ===")
    print("Progress:")

    # --- Probe dimensions ---
    px = 180.0 / npix  # pixel scale in degrees
    r = int(patch_deg / px)
    ny, nx = 2 * r, 2 * r
    
    # --- Load or create memory-mapped storage ---
    J_comp, created = load_or_create_memmap(outfile, shape=(len(freqs), ny, nx, 2, 2),
                                            dtype=np.complex128)

    # If data already exists, skip regeneration
    if not created:
        print("✅ Beam data already available, skipping regeneration.")
        return np.array(J_comp)

    # --- Process beam for all frequencies ---
    for idx, f in enumerate(freqs):
        J = hamaker.allsky_beam(ms_path, time, f * 1e6, station_id=station, npix=npix)
        J[np.isnan(J)] = 0

        cy, cx = np.unravel_index(np.argmax(np.abs(J[:, :, 0, 0])), J[:, :, 0, 0].shape)
        Jz = J[cy - r:cy + r, cx - r:cx + r, :, :]

        J_comp[idx] = Jz
        J_comp.flush()
        print(f"{f} MHz", end=" | ", flush=True)

    print("\n=== Beam Generation Completed ===")
    return np.array(J_comp)


# =====================================================
#  Zernike Decomposition
# =====================================================

def generate_coeffs(J_comp, modes=200, outfile1="zernike_memmap.npy", outfile2="sortind_memmap.npy"):
    """
    Decompose beam matrices into Zernike coefficients and compute energy-based sorting.

    Parameters
    ----------
    J_comp : np.ndarray
        Jones beam data, shape (N_freq, Ny, Nx, 2, 2).
    modes : int, optional
        Number of Zernike modes to compute.
    outfile1, outfile2 : str
        Paths for memmap storage of coefficients and sorting indices.

    Returns
    -------
    coef_comp : np.ndarray
        Zernike coefficients per frequency.
    sortind_noll : np.ndarray
        Sorting indices of Zernike modes by modal energy.
    """
    num_freq = J_comp.shape[0]
    coef_comp, created1 = load_or_create_memmap(outfile1, shape=(num_freq, 2, 2, 2, modes), dtype=np.float64)
    sortind_noll, created2 = load_or_create_memmap(outfile2, shape=(2, 2, 2, modes), dtype=int)

    # If both already exist, skip computation
    if not created1 and not created2:
        print("✅ Using cached Zernike coefficients and sorting indices.")
        return np.array(coef_comp), np.array(sortind_noll)

    print(f"\n=== Zernike Decomposition (first {modes} modes) ===")
    print("Progress:")

    for f in range(num_freq):
        coef = zernike.decom_matrix(J_comp[f], modes)
        coef_comp[f] = coef
        coef_comp.flush()
        print(f"{f + 1}/{num_freq}", end=" | ", flush=True)

    # Compute average energy and sort indices
    avg_ener = np.mean(coef_comp**2, axis=0)
    for i in range(2):
        for j in range(2):
            sortind_noll[i, j, 0] = np.argsort(avg_ener[i, j, 0])[::-1]
            sortind_noll[i, j, 1] = np.argsort(avg_ener[i, j, 1])[::-1]
            sortind_noll.flush()

    print("\n=== Zernike Decomposition Completed ===")
    return np.array(coef_comp), np.array(sortind_noll)


def truncate_coeffs(coef_comp, sortind, thresh):
    """
    Select and retain dominant Zernike coefficients based on global threshold.
    """
    trun_coef_comp = []
    for coef in coef_comp:
        trun_coef = zernike.selected_coefs(coef, sortind, thresh)
        trun_coef_comp.append(trun_coef)
    return np.array(trun_coef_comp)


# =====================================================
#  Beam Reconstruction & Error
# =====================================================

def reconstruct_jones(J_comp, coef_comp, outfile="recon_beams_memmap.npy"):
    """
    Reconstruct Jones matrices from Zernike coefficients using memmap.
    """
    num_freq = J_comp.shape[0]
    Jz_comp = np.lib.format.open_memmap(outfile, mode="w+", dtype=np.float64, 
                                        shape=(num_freq, 2, *J_comp[0].shape))
    print("\n=== Reconstructing Jones Matrices ===")
    print("Progress:")
        
    for f in range(num_freq):
        J = J_comp[f]
        coef = coef_comp[f]
        Jz = np.empty((2, *J.shape))
        for i in range(2):
            for j in range(2):
                Jz[:, :, :, j, i] = zernike.recon_matrix(J[:, :, j, i], coef[i][j])
        Jz_comp[f] = Jz
        Jz_comp.flush()
        print(f"{f + 1}/{num_freq}", end=" | ", flush=True)

    print("\n=== Reconstruction Completed ===")
    return np.array(Jz_comp)


def calc_nmrse(J_comp):
    """
    Compute normalized mean root square error (NMRSE) between
    original and reconstructed Jones matrices.
    """
    num_freq = J_comp.shape[0]
    mrse_comp = []

    for f in range(num_freq):
        J = J_comp[f]
        data, model = J[0], J[1]
        mrse_polar = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                res = np.abs(data[:, :, j, i] - model[:, :, j, i])
                mrse = np.sqrt(np.mean(res ** 2)) / np.mean(np.abs(data[:, :, j, i]))
                mrse_polar[i, j] = mrse
        mrse_comp.append(mrse_polar)
    return np.array(mrse_comp)

# -----------------------------------------------------
# Spectral Compression and Reconstruction
# -----------------------------------------------------

def spectral_compression(freqs, coef, min_coverage=1):
    """
    Fit polynomial + sinusoidal models to the frequency dependence
    of Zernike coefficients.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    coef : np.ndarray
        Truncated Zernike coefficients.
    min_coverage : float
        Minimum variance coverage threshold.

    Returns
    -------
    coef_compressed : dict
        Dictionary containing compressed spectral models.
    """
    coef_compressed = {}

    for i in range(2):
        for j in range(2):
            real_ind = coef[:, i, j, 0, 0]
            real_coef = coef[:, i, j, 1, 0]
            img_ind = coef[:, i, j, 0, 1]
            img_coef = coef[:, i, j, 1, 1]

            print(f"\n=== Compressing J_{i}{j} ===")
            print("Real components:")
            real_results = PolySin.fit_zernike_with_indices(
                freqs, real_ind, real_coef, min_coverage, verbose=True
            )

            print("Imaginary components:")
            img_results = PolySin.fit_zernike_with_indices(
                freqs, img_ind, img_coef, min_coverage, verbose=True
            )

            coef_compressed[(i, j, 0)] = {
                int(idx): params for idx, params in zip(real_results['noll_indices'], real_results['fitted_params'])
            }
            coef_compressed[(i, j, 1)] = {
                int(idx): params for idx, params in zip(img_results['noll_indices'], img_results['fitted_params'])
            }

    return coef_compressed


def spectral_recon(freqs, coef, coef_compressed):
    """
    Reconstruct Zernike coefficients from compressed spectral models.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    coef : np.ndarray
        Original truncated coefficients (for indices).
    coef_compressed : dict
        Compressed model dictionary.

    Returns
    -------
    coef_recon : np.ndarray
        Reconstructed coefficient array.
    """
    coef_recon = np.empty((len(freqs), 2, 2, 2, 2, coef.shape[-1]))

    for i in range(2):
        for j in range(2):
            real_noll = np.array(list(coef_compressed[(i, j, 0)].keys()))
            real_params = np.array(list(coef_compressed[(i, j, 0)].values()))
            real_ind = coef[:, i, j, 0, 0]

            img_noll = np.array(list(coef_compressed[(i, j, 1)].keys()))
            img_params = np.array(list(coef_compressed[(i, j, 1)].values()))
            img_ind = coef[:, i, j, 0, 1]

            recon_real = PolySin.reconstruct_with_indices(real_noll, real_params, real_ind, freqs)
            recon_img = PolySin.reconstruct_with_indices(img_noll, img_params, img_ind, freqs)

            coef_recon[:, i, j, 0, 0] = real_ind
            coef_recon[:, i, j, 1, 0] = recon_real
            coef_recon[:, i, j, 0, 1] = img_ind
            coef_recon[:, i, j, 1, 1] = recon_img

    return coef_recon


# -----------------------------------------------------
# Main Compression Pipeline
# -----------------------------------------------------

def compressed_zernike(start_freq, end_freq, interval, modes=200, thresh=25, patch_deg=15):
    """
    Full pipeline: generate beams, decompose into Zernike modes,
    truncate, and spectrally compress coefficients.

    Parameters
    ----------
    start_freq : float
        Starting frequency (MHz).
    end_freq : float
        Ending frequency (MHz).
    interval : float
        Frequency step (MHz).
    modes : int
        Number of Zernike modes.
    thresh : int
        Number of strongest Zernike modes to be taken.
    patch_deg : float
        Angular distance of the rectangular patch.

    Returns
    -------
    coef_compressed : dict
        Compressed spectral models.
    trun_coef_comp : np.ndarray
        Truncated coefficients.
    J_comp : np.ndarray
        Jones beam data.
    """
    freqs = np.arange(start_freq, end_freq + 1, interval)
    file_name = f"synthms_hba_dec{losito.dec_deg}"
    files = [f for f in os.listdir('.') if f.startswith(file_name)]

    possible_paths = [
        "/usr/bin/synthms",
        "/usr/local/bin/synthms",
        os.path.expanduser("~/losito/bin/synthms"),
    ]
    synthms_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not files:
        print("🧰 Measurement Set not found — generating new one...")
        name = losito.generate_ms(synthms_path)
        if name is None:
            print("❌ Measurement Set generation failed. Aborting.")
            return
        files = [f for f in os.listdir('.') if f.startswith(file_name)]
        
    # Pipeline steps
    J_comp = generate_beam(start_freq, end_freq, interval, files[0], patch_deg=patch_deg)
    coef_comp, sortind = generate_coeffs(J_comp, modes=modes)
    trun_coef_comp = truncate_coeffs(coef_comp, sortind, thresh)
    coef_compressed = spectral_compression(freqs, trun_coef_comp)

    return coef_compressed, trun_coef_comp, J_comp
