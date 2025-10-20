import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional


class PolySinIndexAwareProcessor:
    """
    Fit a polynomial + sinusoidal model to time-series of Zernike coefficients
    where the Noll indices vary with frequency (i.e. coefficients appear/disappear).

    The model for a coefficient as a function of frequency f (MHz) is:

        y(f) = A0 + Ainv / f + A1*f + A2*f^2 + A_sin * sin(2π f / f0 + phi)

    Notes
    -----
    - The class keeps internal maps of fitted parameters, fit success flags and
      optional error messages for individual Noll indices.
    - Intended workflow:
        processor = PolySinIndexAwareProcessor(freqs)
        results = processor.fit_matrices(indices_matrix, coeffs_matrix)
        rec = processor.reconstruct_matrices(target_indices_matrix)
    """

    def __init__(self, freqs: np.ndarray):
        """
        Parameters
        ----------
        freqs : array_like
            1D array of frequency values (MHz). Typically shape (N_freqs,).
        """
        self.freqs = np.array(freqs, dtype=float)
        self.fitted_params: Dict[int, np.ndarray] = {}  # noll_index -> params (7,)
        self.fit_success: Dict[int, bool] = {}          # noll_index -> bool
        self.fit_errors: Dict[int, Optional[str]] = {}  # noll_index -> error message (or None)
        self.all_noll_indices = set()

    # --------------------- model ------------------------------------
    def poly_sin_model(self, freqs: np.ndarray, A0: float, Ainv: float,
                       A1: float, A2: float, A_sin: float,
                       f0: float, phi: float) -> np.ndarray:
        """
        Polynomial + sinusoidal model used for curve_fit.

        Small epsilon added to denominator to avoid division by zero.
        """
        f = np.array(freqs, dtype=float)
        return A0 + Ainv / (f + 1e-12) + A1 * f + A2 * f**2 + A_sin * np.sin(2 * np.pi * f / f0 + phi)

    # ----------------- data extraction ------------------------------
    def extract_coefficient_timeseries(self,
                                       indices_matrix: np.ndarray,
                                       coeffs_matrix: np.ndarray
                                       ) -> Dict[int, Dict[str, Any]]:
        """
        Build per-Noll-index time series from variable-index matrices.

        Parameters
        ----------
        indices_matrix : (N_freqs, N_positions) array
            Noll index at each frequency & grid position. May contain NaN for missing.
        coeffs_matrix : (N_freqs, N_positions) array
            Corresponding coefficient values (complex or real). If complex, user should pass real/imag separately.

        Returns
        -------
        processed_data : dict
            Mapping noll_index -> {
                'freqs': np.array of frequencies where it appears,
                'values': np.array of coefficient values,
                'freq_indices': np.array of integer frequency indices (0..N_freqs-1),
                'coverage': fraction of frequencies where the coefficient appears
            }
        """
        indices_matrix = np.array(indices_matrix)
        coeffs_matrix = np.array(coeffs_matrix)

        if indices_matrix.shape != coeffs_matrix.shape:
            raise ValueError("indices_matrix and coeffs_matrix must have the same shape")

        n_freqs = indices_matrix.shape[0]
        data_acc = defaultdict(lambda: {'freqs': [], 'values': [], 'freq_indices': []})

        for freq_idx in range(n_freqs):
            for pos_idx in range(indices_matrix.shape[1]):
                noll = indices_matrix[freq_idx, pos_idx]
                if np.isnan(noll):
                    continue
                noll_int = int(noll)
                data_acc[noll_int]['freqs'].append(self.freqs[freq_idx])
                data_acc[noll_int]['values'].append(coeffs_matrix[freq_idx, pos_idx])
                data_acc[noll_int]['freq_indices'].append(freq_idx)

        processed = {}
        for noll, d in data_acc.items():
            freqs_arr = np.array(d['freqs'], dtype=float)
            values_arr = np.array(d['values'], dtype=float)
            freq_indices_arr = np.array(d['freq_indices'], dtype=int)
            coverage = len(freqs_arr) / float(n_freqs)
            processed[noll] = {
                'freqs': freqs_arr,
                'values': values_arr,
                'freq_indices': freq_indices_arr,
                'coverage': coverage
            }
            self.all_noll_indices.add(noll)

        return processed

    # ----------------- single coefficient fit -----------------------
    def fit_single_coefficient(self,
                               freqs: np.ndarray,
                               values: np.ndarray,
                               noll_index: int,
                               max_attempts: int = 3
                               ) -> Tuple[np.ndarray, bool, Optional[str]]:
        """
        Fit the poly+sin model to a single coefficient time series.

        Returns:
            (params, success, error_message)
            params: array of length 7 (may be filled with NaNs if failed)
        """
        freqs = np.array(freqs, dtype=float)
        values = np.array(values, dtype=float)

        # require at least 7 data points for 7 parameters to avoid gross underdetermination
        if len(freqs) < 7:
            return np.full(7, np.nan), False, f"Insufficient data points ({len(freqs)})"

        # remove NaNs in values
        valid = ~np.isnan(values)
        if np.sum(valid) < 7:
            return np.full(7, np.nan), False, "Not enough valid (non-NaN) data points"

        x = freqs[valid]
        y = values[valid]

        # -------------------- initial guesses (multiple strategies) ------------------
        init_list: List[List[float]] = []

        A0_guess = float(np.median(y))
        Ainv_guess = 0.0
        A1_guess = float((y[-1] - y[0]) / (x[-1] - x[0])) if len(x) > 1 else 0.0
        A2_guess = 0.0
        A_sin_guess = 0.5 * float(np.std(y)) if np.std(y) > 0 else 1e-3
        # plausible period guess: 2x data span (so 1 full oscillation over the span)
        f0_guess = float(2 * max(x[-1] - x[0], 1.0))
        phi_guess = 0.0

        init_list.append([A0_guess, Ainv_guess, A1_guess, A2_guess, A_sin_guess, f0_guess, phi_guess])
        # conservative
        init_list.append([float(np.mean(y)), 0.0, 0.0, 0.0, 0.1 * A_sin_guess, max(30., f0_guess), 0.0])
        # alternative phasing / period
        init_list.append([A0_guess, Ainv_guess, A1_guess, A2_guess, A_sin_guess, max(30., f0_guess / 2.), np.pi / 2.])

        # -------------------- bounds ------------------
        val_range = max(np.max(y) - np.min(y), 1e-6)
        f_span = max(x[-1] - x[0], 1.0)

        lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -10 * val_range, max(1.0, f_span / 20.), -2 * np.pi]
        upper_bounds = [ np.inf,  np.inf,  np.inf,  np.inf,  10 * val_range, f_span * 10.,  2 * np.pi]

        best_params = None
        best_rmse = np.inf
        best_err: Optional[str] = None
        success = False

        # try different initializations up to max_attempts
        for p0 in init_list[:max_attempts]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _pcov = curve_fit(
                        self.poly_sin_model, x, y, p0=p0,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=20000, method='trf'
                    )

                y_pred = self.poly_sin_model(x, *popt)
                rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = popt.copy()
                    success = True
                    best_err = None
            except Exception as exc:
                # capture the last exception message in case all attempts fail
                best_err = str(exc)
                continue

        if success and best_params is not None:
            return np.array(best_params, dtype=float), True, None
        else:
            return np.full(7, np.nan), False, f"Fitting failed. Last error: {best_err}"

    # ----------------- matrix-level fitting -------------------------
    def fit_matrices(self,
                     indices_matrix: np.ndarray,
                     coeffs_matrix: np.ndarray,
                     min_coverage: float = 0.2,
                     verbose: bool = True
                     ) -> Dict[str, Any]:
        """
        Fit model for all Noll indices present in indices_matrix/coeffs_matrix.

        Parameters
        ----------
        indices_matrix : (N_freqs, N_positions) array
        coeffs_matrix : (N_freqs, N_positions) array
        min_coverage : float
            Minimum fraction of frequencies a coefficient must appear in to attempt a fit.
        verbose : bool
            If True, print progress.

        Returns
        -------
        dictionary with keys:
            - 'coefficient_data' : dict from extract_coefficient_timeseries
            - 'fitted_coefficients' : list of noll indices attempted and stored
            - 'n_successful' : int
            - 'n_total' : int (number attempted)
            - 'coverage_stats' : dict noll -> coverage
        """
        coeff_data = self.extract_coefficient_timeseries(indices_matrix, coeffs_matrix)

        # filter by coverage
        filtered = {n: d for n, d in coeff_data.items() if d['coverage'] >= min_coverage}

        if verbose:
            print(f"Unique Noll indices found: {len(coeff_data)}")
            print(f"Fitting indices with coverage ≥ {min_coverage:.0%}: {len(filtered)}")
            print("Progress:", end=" ", flush=True)

        n_total = len(filtered)
        n_success = 0

        # deterministic ordering for reproducibility
        for idx, (noll, data) in enumerate(sorted(filtered.items())):
            # simple progress indicator every 10% or for small sets print at each step
            if verbose and (n_total <= 20 or (idx + 1) % max(1, n_total // 10) == 0):
                print(f"{idx+1}/{n_total}", end=" ", flush=True)

            params, ok, err = self.fit_single_coefficient(data['freqs'], data['values'], noll)
            self.fitted_params[noll] = params
            self.fit_success[noll] = ok
            self.fit_errors[noll] = err

            if ok:
                n_success += 1

        if verbose:
            print(f"\nFitting completed: {n_success}/{n_total} succeeded")

        return {
            'coefficient_data': coeff_data,
            'fitted_coefficients': sorted(list(self.fitted_params.keys())),
            'n_successful': n_success,
            'n_total': n_total,
            'coverage_stats': {n: d['coverage'] for n, d in coeff_data.items()}
        }

    # ----------------- reconstruction --------------------------------
    def reconstruct_matrices(self,
                             target_indices_matrix: np.ndarray,
                             freqs: Optional[np.ndarray] = None
                             ) -> np.ndarray:
        """
        Reconstruct a coefficient matrix according to a target Noll-index layout.

        Parameters
        ----------
        target_indices_matrix : (M_freqs, N_positions) array
            Noll-index layout to reconstruct. May contain NaN where no coefficient exists.
        freqs : optional array of frequencies (length M_freqs). If None, uses the processor's original frequencies.

        Returns
        -------
        reconstructed : ndarray (M_freqs, N_positions)
            Reconstructed coefficient values. Positions with missing/unfitted indices are set to zero.
        """
        target_indices = np.array(target_indices_matrix)
        M, P = target_indices.shape
        freqs_used = np.array(freqs) if freqs is not None else self.freqs
        if len(freqs_used) != M:
            raise ValueError("Length of freqs must match number of rows in target_indices_matrix")

        reconstructed = np.full((M, P), np.nan, dtype=float)

        for fi in range(M):
            f_val = freqs_used[fi]
            for pj in range(P):
                noll = target_indices[fi, pj]
                if np.isnan(noll):
                    continue
                noll_int = int(noll)
                if noll_int in self.fitted_params and self.fit_success.get(noll_int, False):
                    params = self.fitted_params[noll_int]
                    if not np.all(np.isnan(params)):
                        # model expects array input, but we pass scalar frequency
                        reconstructed[fi, pj] = float(self.poly_sin_model(np.array([f_val]), *params)[0]

                        )
        # fill remaining NaNs with zeros for a clean numeric matrix
        reconstructed[np.isnan(reconstructed)] = 0.0
        return reconstructed

    # ----------------- export helpers --------------------------------
    def get_fitted_parameters_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return fitted parameters as structured arrays.

        Returns
        -------
        noll_indices : ndarray (N,)
        param_matrix : ndarray (N, 7)
        success_mask : ndarray (N,) boolean
        """
        successful_nolls = sorted([noll for noll, ok in self.fit_success.items() if ok])
        if len(successful_nolls) == 0:
            return np.array([], dtype=int), np.zeros((0, 7), dtype=float), np.array([], dtype=bool)

        param_matrix = np.array([self.fitted_params[n] for n in successful_nolls], dtype=float)
        success_mask = np.array([self.fit_success[n] for n in successful_nolls], dtype=bool)
        return np.array(successful_nolls, dtype=int), param_matrix, success_mask

    # ----------------- plotting -------------------------------------
    def plot_fit_examples(self, coeff_data: Dict[int, Dict[str, Any]], max_plots: int = 6):
        """
        Plot a selection of fitted coefficient timeseries and their model fits.

        Parameters
        ----------
        coeff_data : dict
            The dictionary returned by extract_coefficient_timeseries (or fit_matrices['coefficient_data']).
        max_plots : int
            Maximum number of example plots to draw.
        """
        # select only successfully fitted indices
        successful = [n for n, ok in self.fit_success.items() if ok]
        if not successful:
            print("No successful fits to plot.")
            return

        plot_nolls = sorted(successful)[:max_plots]
        n_plots = len(plot_nolls)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for i, noll in enumerate(plot_nolls):
            ax = axes_flat[i]
            data = coeff_data[noll]
            x = data['freqs']
            y = data['values']
            params = self.fitted_params[noll]

            ax.plot(x, y, 'ko', markersize=4, label='data')
            xfine = np.linspace(np.min(x), np.max(x), 200)
            yfit = self.poly_sin_model(xfine, *params)
            ax.plot(xfine, yfit, '-', linewidth=1.8, label='poly+sin fit')

            ypred = self.poly_sin_model(x, *params)
            rmse = np.sqrt(np.mean((y - ypred) ** 2))

            ax.set_title(f"Noll {noll} (cov: {data['coverage']:.0%}) RMSE={rmse:.2e}")
            ax.set_xlabel("Frequency [MHz]")
            ax.set_ylabel("Coefficient value")
            ax.grid(alpha=0.3)
            ax.legend()

        # hide unused axes
        for j in range(len(axes_flat)):
            if j >= n_plots:
                axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()


# --------------------- convenience functions --------------------------

def fit_zernike_with_indices(freqs: np.ndarray,
                             indices_matrix: np.ndarray,
                             coeffs_matrix: np.ndarray,
                             min_coverage: float = 0.3,
                             verbose: bool = True,
                             plot_examples: bool = False) -> Dict[str, Any]:
    """
    Convenience wrapper to create a processor, fit all coefficients and return a
    structured results dict plus the processor instance.

    Returns a dictionary with:
      - 'processor' : the PolySinIndexAwareProcessor instance
      - 'noll_indices', 'fitted_params', 'success_mask' : structured fitted params
      - 'coefficient_data' : original extracted timeseries
      - 'compression_info' : simple storage/compression stats
    """
    processor = PolySinIndexAwareProcessor(freqs)
    results = processor.fit_matrices(indices_matrix, coeffs_matrix, min_coverage, verbose=verbose)

    noll_indices, param_matrix, success_mask = processor.get_fitted_parameters_matrix()

    results.update({
        'processor': processor,
        'noll_indices': noll_indices,
        'fitted_params': param_matrix,
        'success_mask': success_mask,
        'compression_info': {
            'original_elements': int(indices_matrix.shape[0] * indices_matrix.shape[1]),
            'fitted_parameters': int(len(noll_indices) * 7),
            'compression_ratio': float((indices_matrix.shape[0] * indices_matrix.shape[1]) / (len(noll_indices) * 7)) if len(noll_indices) > 0 else 0.0
        }
    })

    if verbose:
        comp = results['compression_info']
        print(f"Compression ratio ~ {comp['compression_ratio']:.2f}:1 ({comp['original_elements']} -> {comp['fitted_parameters']} params)")

    if plot_examples and results['n_successful'] > 0:
        processor.plot_fit_examples(results['coefficient_data'], max_plots=min(6, len(noll_indices)))

    return results


def reconstruct_with_indices(noll_indices: np.ndarray,
                             fitted_params: np.ndarray,
                             target_indices_matrix: np.ndarray,
                             freqs_original: np.ndarray,
                             freqs_new: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruct coefficient matrix from stored fitted_params and a target index layout.

    Parameters
    ----------
    noll_indices : (N,) array of int
        Noll indices corresponding to rows of fitted_params
    fitted_params : (N, 7) array
        Parameter matrix returned by get_fitted_parameters_matrix()
    target_indices_matrix : (M, P) array
        Desired layout for reconstruction (may contain NaN)
    freqs_original : array_like
        Original frequency array used to fit the models (only used to init the processor)
    freqs_new : optional array_like
        Frequencies to reconstruct at (length M). If None, uses freqs_original truncated/padded as needed.

    Returns
    -------
    reconstructed : (M, P) ndarray
    """
    processor = PolySinIndexAwareProcessor(np.array(freqs_original))

    # load params
    for i, noll in enumerate(np.array(noll_indices, dtype=int)):
        processor.fitted_params[int(noll)] = np.array(fitted_params[i], dtype=float)
        processor.fit_success[int(noll)] = True

    # pass freqs_new to reconstruct_matrices; if None the processor will use its own freqs
    return processor.reconstruct_matrices(target_indices_matrix, freqs=freqs_new)
