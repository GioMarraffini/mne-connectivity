# Authors: Picnic Lab
#          Giovanni Marraffini <giovanni.marraffini@gmail.com>
#          Based on the work of Jean-Remy King, Jacobo Sitt and Federico Raimondo
#
# License: BSD (3-clause)

import math
from itertools import permutations

import numpy as np
import numba
from scipy.signal import butter, filtfilt

import mne
from mne import pick_types
from mne.epochs import BaseEpochs
from mne.preprocessing import compute_current_source_density
from mne.utils import logger, _time_mask, verbose
from mne.utils.docs import fill_doc
from mne.utils.check import _validate_type

from .base import EpochTemporalConnectivity



def _define_symbols(kernel):
    """Define all possible symbols for a given kernel size (original implementation)."""
    result_dict = dict()
    total_symbols = math.factorial(kernel)
    cursymbol = 0
    for perm in permutations(range(kernel)):
        order = ''.join(map(str, perm))
        if order not in result_dict:
            result_dict[order] = cursymbol
            cursymbol = cursymbol + 1
            result_dict[order[::-1]] = total_symbols - cursymbol
    result = []
    for v in range(total_symbols):
        for symbol, value in result_dict.items():
            if value == v:
                result += [symbol]
    return result

def _symb_python_optimized(data, kernel, tau):
    """Compute symbolic transform using original logic but optimized.
    
    This matches the original _symb_python exactly but with optimizations.
    """
    symbols = _define_symbols(kernel)
    dims = data.shape

    signal_sym_shape = list(dims)
    signal_sym_shape[1] = data.shape[1] - tau * (kernel - 1)
    signal_sym = np.zeros(signal_sym_shape, np.int32)

    count_shape = list(dims)
    count_shape[1] = len(symbols)
    count = np.zeros(count_shape, np.int32)

    # Create a dict for fast lookup (instead of symbols.index which is O(n))
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    
    for k in range(signal_sym_shape[1]):
        subsamples = range(k, k + kernel * tau, tau)
        ind = np.argsort(data[:, subsamples], 1)

        # Process each channel and epoch
        for ch in range(data.shape[0]):
            for ep in range(data.shape[2]):
                symbol_str = ''.join(map(str, ind[ch, :, ep]))
                signal_sym[ch, k, ep] = symbol_to_idx[symbol_str]

    count = np.double(np.apply_along_axis(
        lambda x: np.bincount(x, minlength=len(symbols)), 1, signal_sym))

    return signal_sym, (count / signal_sym_shape[1])

def _get_weights_matrix(nsym):
    """Aux function (original implementation)"""
    wts = np.ones((nsym, nsym))
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    return wts

@numba.njit(parallel=True) # Enabled parallel execution
def _wsmi_python_jitted(data_sym, counts, wts_matrix):
    """Compute raw wSMI and SMI from symbolic data (Numba-jitted)."""
    nchannels, nsamples_after_symb, ntrials = data_sym.shape
    n_unique_symbols = counts.shape[1]

    smi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    wsmi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)

    epsilon = 1e-15 
    log_counts = np.log(counts + epsilon)

    for trial_idx in numba.prange(ntrials):
        for ch1_idx in range(nchannels):
            for ch2_idx in range(ch1_idx + 1, nchannels):
                pxy = np.zeros((n_unique_symbols, n_unique_symbols), dtype=np.double)
                for sample_idx in range(nsamples_after_symb):
                    sym1 = data_sym[ch1_idx, sample_idx, trial_idx]
                    sym2 = data_sym[ch2_idx, sample_idx, trial_idx]
                    
                    pxy[sym1, sym2] += 1
                
                if nsamples_after_symb > 0:
                    pxy /= nsamples_after_symb

                current_smi_val = 0.0
                current_wsmi_val = 0.0
                
                # Compute MI terms manually to avoid broadcasting issues in Numba
                for r_idx in range(n_unique_symbols):
                    for c_idx in range(n_unique_symbols):
                        if pxy[r_idx, c_idx] > epsilon:
                            log_pxy_val = np.log(pxy[r_idx, c_idx])
                            log_px_val = log_counts[ch1_idx, r_idx, trial_idx]
                            log_py_val = log_counts[ch2_idx, c_idx, trial_idx]
                            
                            mi_term = pxy[r_idx, c_idx] * (log_pxy_val - log_px_val - log_py_val)
                            current_smi_val += mi_term
                            current_wsmi_val += wts_matrix[r_idx, c_idx] * mi_term
              
                smi[ch1_idx, ch2_idx, trial_idx] = current_smi_val
                wsmi[ch1_idx, ch2_idx, trial_idx] = current_wsmi_val

    if n_unique_symbols > 1:
        norm_factor = np.log(n_unique_symbols)
        if norm_factor > epsilon: 
            smi /= norm_factor
            wsmi /= norm_factor
    else: 
        smi_fill_val = 0.0 
        wsmi_fill_val = 0.0
        smi[:,:,:] = smi_fill_val 
        wsmi[:,:,:] = wsmi_fill_val

    # Note: Original implementation only fills upper triangle, so we match that behavior
    # No mirroring to lower triangle to match original exactly
                
    return wsmi, smi


@fill_doc
@verbose
def wsmi(epochs, kernel, tau, tmin=None, tmax=None,
         method_params=None, n_jobs=1, verbose=None):
    """Compute weighted symbolic mutual information (wSMI).

    Parameters
    ----------
    epochs : instance of BaseEpochs
        The data from which to compute connectivity.
    kernel : int
        Pattern length (symbol dimension) for symbolic analysis.
        Must be > 1. Values > 7 may require significant memory.
    tau : int  
        Time delay (lag) between consecutive pattern elements.
        Must be > 0.
    tmin : float | None
        Time to start connectivity estimation. If None, uses beginning
        of epoch.
    tmax : float | None
        Time to end connectivity estimation. If None, uses end of epoch.
    method_params : dict | None
        Additional parameters for the method. Supported keys:
        
        - 'filter_freq' : float | None
            Low-pass filter frequency in Hz. If None, defaults to
            sfreq / (kernel * tau).
        - 'bypass_csd' : bool
            Whether to bypass Current Source Density (CSD) computation
            for EEG channels. Default is False.
    n_jobs : int
        Number of parallel jobs. Currently not used.
    %(verbose)s

    Returns
    -------
    conn : instance of EpochTemporalConnectivity
        Computed wSMI connectivity measures. The connectivity object contains
        the weighted symbolic mutual information values between all channel pairs.

    Notes
    -----
    The weighted Symbolic Mutual Information (wSMI) is a connectivity measure
    that quantifies non-linear statistical dependencies between time series
    based on symbolic dynamics :footcite:`KingEtAl2013`.

    The method involves:
    1. Symbolic transformation of time series using ordinal patterns
    2. Computation of mutual information between symbolic sequences  
    3. Weighting based on pattern distance for enhanced sensitivity

    References
    ----------
    .. footbibliography::
    """
    _validate_type(epochs, BaseEpochs, 'epochs')
    _validate_type(kernel, 'int', 'kernel')
    _validate_type(tau, 'int', 'tau')
    
    if kernel <= 1:
        raise ValueError(f"kernel (pattern length) must be > 1, got {kernel}")
    if tau <= 0:
        raise ValueError(f"tau (delay) must be > 0, got {tau}")

    # --- Memory check for large kernels ---
    if kernel > 7:  # Check based on actual kernel, factorial can be huge
        actual_symbols_for_mem_check = math.factorial(kernel)  # For theoretical worst case
        memory_gb = (actual_symbols_for_mem_check ** 2 * 8) / (1024 ** 3)
        if memory_gb > 1.0:  # Example threshold, adjust as needed
            raise ValueError(
                f"kernel={kernel} (factorial={actual_symbols_for_mem_check}) could theoretically require "
                f"{memory_gb:.1f} GB for matrices. Consider kernel <= 7.")

    if method_params is None:
        method_params = {}

    ch_names_original = epochs.ch_names
    sfreq = epochs.info['sfreq']
    events = epochs.events
    event_id = epochs.event_id
    metadata = epochs.metadata

    # --- 1. Preprocessing (CSD) ---
    # (User's CSD and picking logic largely preserved)
    bypass_csd = method_params.get('bypass_csd', False)
    
    # Determine relevant info object based on CSD application
    epochs_to_get_data_from = epochs
    if not bypass_csd and 'eeg' in epochs and pick_types(epochs.info, meg=False, eeg=True).size > 0:
        logger.info('Computing Current Source Density (CSD) for EEG channels.')
        epochs_for_csd = epochs.copy()
        if epochs_for_csd.info['bads']:
            logger.info(f"Interpolating {len(epochs_for_csd.info['bads'])} bad EEG channels for CSD computation.")
            epochs_for_csd.interpolate_bads(reset_bads=True) # Interpolate bads before CSD
        
        csd_epochs = compute_current_source_density(epochs_for_csd, lambda2=1e-5)
            # Check if CSD actually produced CSD channels
        if pick_types(csd_epochs.info, csd=True).size > 0:
            epochs_to_get_data_from = csd_epochs
        else:
            logger.warning('CSD computation did not result in any CSD channels. Using original EEG data for EEG channels.')
            # Fallback to original epochs, CSD effectively skipped for data picking.
    
    # Pick data channels for connectivity computation from the (potentially CSD-transformed) info
    # MEG, EEG, CSD, SEEG, ECoG are typical data channels. Exclude bads.
    picks = pick_types(
        epochs_to_get_data_from.info, meg=True, eeg=True, csd=True, seeg=True, ecog=True, ref_meg=False, exclude="bads"
    )

    if len(picks) == 0:
        raise ValueError(
            "No suitable channels (MEG, EEG, CSD, SEEG, ECoG) found after picking logic. "
            "Check channel types and 'bads'."
        )
    
    data_for_comp = epochs_to_get_data_from.get_data(picks=picks)
    picked_ch_names = [epochs_to_get_data_from.ch_names[i] for i in picks]
    n_epochs, n_channels_picked, n_times_epoch = data_for_comp.shape
    
    if n_channels_picked == 0: # Should be caught by len(picks) == 0
        raise ValueError("No channels selected for wSMI computation after picking.")
    logger.info(
        f"Processing {n_epochs} epochs, {n_channels_picked} channels "
        f"({picked_ch_names}), {n_times_epoch} time points per epoch."
    )

    # --- 2. Filtering (match original exactly) ---
    filter_freq_param = method_params.get('filter_freq', None)
    if filter_freq_param is None:
        if kernel <= 0 or tau <= 0:  # prevent division by zero if params are bad
            raise ValueError(f"kernel ({kernel}) and tau ({tau}) must be positive for default filter_freq.")
        filter_freq = np.double(sfreq) / kernel / tau  # Use np.double like original
    else:
        _validate_type(filter_freq_param, 'numeric', 'method_params["filter_freq"]')
        filter_freq = float(filter_freq_param)
        
    # Validate filter frequency
    nyquist = sfreq / 2.0
    if filter_freq <= 0 or filter_freq >= nyquist:
        raise ValueError(f"filter_freq ({filter_freq:.2f}) must be > 0 and < Nyquist "
                       f"frequency ({nyquist:.2f} Hz)")
    logger.info(f'Filtering  at {filter_freq:.2f} Hz')  # Match original message format
    
    # Match original exactly: concatenate epochs, filter, then split back
    b, a = butter(6, 2.0 * filter_freq / np.double(sfreq), 'lowpass')
    data_concatenated = np.hstack(data_for_comp)  # Concatenate epochs horizontally like original
    
    # Filter the concatenated data
    fdata_concatenated = filtfilt(b, a, data_concatenated)
    
    # Split back into epochs and transpose to match original format
    fdata = np.transpose(np.array(
        np.split(fdata_concatenated, n_epochs, axis=1)), [1, 2, 0])

    # --- Time masking (match original exactly) ---
    time_mask = _time_mask(epochs.times, tmin, tmax)
    fdata_masked = fdata[:, time_mask, :]
    
    # Check if time masking resulted in too few samples for symbolization
    min_samples_needed_for_one_symbol = tau * (kernel - 1) + 1
    if fdata_masked.shape[1] < min_samples_needed_for_one_symbol:
        raise ValueError(
            f"After time masking ({tmin}-{tmax}s), data has {fdata_masked.shape[1]} samples per epoch, "
            f"but at least {min_samples_needed_for_one_symbol} are needed for kernel={kernel}, tau={tau}. "
            f"Adjust tmin/tmax or check epoch length.")
            
    # Data is already in the right format for symbolic transformation: (n_channels_picked, n_times, n_epochs)
    fdata_for_symb = fdata_masked

    # --- 3. Symbolic Transformation ---
    logger.info("Performing symbolic transformation...")
    try:
        sym, count = _symb_python_optimized(fdata_for_symb, kernel, tau)
    except Exception as e:
        logger.error(f"Error during symbolic transformation: {e}")
        raise
        
    n_unique_symbols = count.shape[1]
    if sym.shape[0] != n_channels_picked or sym.shape[2] != n_epochs or \
       count.shape[0] != n_channels_picked or count.shape[2] != n_epochs:
        raise ValueError(
            f"Symbolic transformation output has unexpected shape. Got sym: {sym.shape}, "
            f"count: {count.shape}. Expected channels: {n_channels_picked}, epochs: {n_epochs}.")

    wts = _get_weights_matrix(n_unique_symbols)


    # --- 4. wSMI/SMI Computation (Jitted) ---
    logger.info(f"Computing wSMI and SMI for {n_unique_symbols} unique symbols (Numba-jitted)...")
    wsmi_val, smi_val = _wsmi_python_jitted(sym, count, wts)
    # wsmi_val, smi_val are (n_channels_picked, n_channels_picked, n_epochs)
    
    wsmi_val_epoched = wsmi_val.transpose(2, 0, 1)
    smi_val_epoched = smi_val.transpose(2, 0, 1)

    # --- Packaging results ---
    if n_channels_picked > 1:
        # Extract upper triangular part (excluding diagonal) for connectivity object
        triu_inds = np.triu_indices(n_channels_picked, k=1)
        indices_list = list(zip(triu_inds[0], triu_inds[1]))
        
        # Extract connectivity data for upper triangular connections only
        wsmi_conn_data = np.zeros((n_epochs, len(indices_list), 1))
        for epoch_idx in range(n_epochs):
            for conn_idx, (i, j) in enumerate(indices_list):
                wsmi_conn_data[epoch_idx, conn_idx, 0] = wsmi_val_epoched[epoch_idx, i, j]
        
        wsmi_connectivity = EpochTemporalConnectivity(
            data=wsmi_conn_data,
            names=picked_ch_names,
            times=None, 
            method="wSMI",
            indices=indices_list,
            n_epochs_used=n_epochs,
            n_nodes=n_channels_picked,
            sfreq=sfreq,
            events=events,
            event_id=event_id,
            metadata=metadata,
        )
    else:
        # For single channel or no channels, create empty connectivity
        logger.info("Only 1 channel or fewer selected, wSMI connectivity will be empty.")
        wsmi_connectivity = EpochTemporalConnectivity(
            data=np.empty((n_epochs, 0, 1)),
            names=picked_ch_names,
            times=None, 
            method="wSMI",
            indices=[],
            n_epochs_used=n_epochs,
            n_nodes=n_channels_picked,
            sfreq=sfreq,
            events=events,
            event_id=event_id,
            metadata=metadata,
        )
    
    logger.info("wSMI computation finished.")
    
    return wsmi_connectivity

# User's original footbibliography comment can remain here or at the end of the file.
# .. [1] King, J. R., Sitt, J. D., Faugeras, F., Rohaut, B., El Karoui, I.,
#        Cohen, L., ... & Dehaene, S. (2013). Information sharing in the
#        brain indexes consciousness in noncommunicative patients. Current
#        biology, 23(19), 1914-1919. 