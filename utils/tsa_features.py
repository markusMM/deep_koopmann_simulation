from typing import Any
import numpy as np
import pandas as pd
import pywt
import multiprocessing as mp
import torch
import torch.fft as fft


def torch_cwt_batch(signals, scales, wavelet, fs=1.0) -> tuple[torch.Tensor, Any]:
    """Perform batched Continuous Wavelet Transform on several signals."""
    n = signals.shape[-1]
    freqs = torch.tensor(pywt.scale2frequency(wavelet, scales) / fs, dtype=torch.float32)
    t = torch.arange(n, dtype=torch.float32, device=signals.device)
    # Create morlet in time or precompute in frequency domain
    # Wavelet FFT; here simplified analytic Morlet version
    wavelet_fft = []
    for s in scales:
        sigma = s / fs
        w = torch.exp(-0.5 * ((t - n/2)/sigma)**2)
        w = w / w.abs().sum()
        wavelet_fft.append(fft.fft(w, n=n))
    wavelet_fft = torch.stack(wavelet_fft)

    sig_fft = fft.fft(signals, n=n)
    coeffs = fft.ifft(sig_fft.unsqueeze(1) * wavelet_fft.unsqueeze(0), n=n)
    power = coeffs.abs() ** 2
    avg_power = power.mean(dim=-1)           # average across time axis
    dom_idx = avg_power.argmax(dim=-1)
    dom_freq = freqs[dom_idx]
    ent = - (avg_power/avg_power.sum(dim=-1, keepdim=True) * 
            (avg_power/avg_power.sum(dim=-1, keepdim=True)+1e-12).log()).sum(dim=-1)
    return dom_freq, ent


def wavelet_features(series, wavelet="cmor1.5-1.0", scales=np.geomspace(1,512,32), fs=1.0):
    data = (series - np.mean(series)) / np.std(series)
    coeffs, freqs = pywt.cwt(data, scales, wavelet, sampling_period=1/fs)

    power = np.abs(coeffs) ** 2                  # time–frequency energy
    avg_power = power.mean(axis=1)               # mean energy per scale
    dom_scale = scales[np.argmax(avg_power)]   # scale of peak energy
    spectral_entropy = -(avg_power/avg_power.sum() * np.log(avg_power/avg_power.sum()+1e-12)).sum()

    return {
        "dom_freq": pywt.scale2frequency(wavelet, dom_scale)/fs,
        "entropy": spectral_entropy,
        "energy_low": avg_power[:len(scales)//3].sum(),
        "energy_mid": avg_power[len(scales)//3:2*len(scales)//3].sum(),
        "energy_high": avg_power[2*len(scales)//3:].sum()
    }

# Sliding application
def wavelet_feature_series(data, window=24*30, step=24*7, fs=1.0) -> pd.DataFrame:
    starts = range(0, len(data)-window, step)
    params = [(data, s, window, fs) for s in starts]
    with mp.Pool() as pool:
        results = pool.map(process_window, params)

    return pd.DataFrame(results)


def wavelet_feature_variables(
    df: pd.DataFrame,
    variables: list[str],
    keys: list[str] = ['time', 'lat', 'lon'],
    group_keys: list[str] = ['lat', 'lon'],
    window: int = 24*30,
    step: int = 24*7,
    fs: float = 1.0
) -> pd.DataFrame:
    # reindex
    df = df.set_index(keys)

    # group by spatial coords
    grouped = df.groupby(level=group_keys)

    dfs_wav_all = []
    for (lat, lon), group in grouped:
        dfs_wav = []
        for v in variables:
            series = group[v].droplevel(group_keys)  # type: ignore
            df_wav = wavelet_feature_series(series, window, step, fs)
            df_wav = df_wav.rename(columns={col: f"{v}__{col}" for col in df_wav.columns})

            # Add spatial coords as columns for merging later
            df_wav['lat'] = lat
            df_wav['lon'] = lon

            dfs_wav.append(df_wav)
        # Concat variables horizontally for this spatial location
        spatial_features = pd.concat(dfs_wav, axis=1)

        # Remove duplicate lat/lon columns if any
        spatial_features = spatial_features.loc[:,~spatial_features.columns.duplicated()]

        dfs_wav_all.append(spatial_features)

    # Combine all spatial locations vertically
    result = pd.concat(dfs_wav_all, ignore_index=True, axis=0)

    return result


def process_window(args) -> dict[str, Any]:
    data, start, window, fs = args
    return wavelet_features(data[start:start+window], fs=fs)
