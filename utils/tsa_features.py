from typing import Any
import numpy as np
import pandas as pd
import pywt
import multiprocessing as mp
import torch
import torch.fft as fft
from statsmodels.tsa.seasonal import seasonal_decompose
from skopt.space import Integer
from skopt import gp_minimize
from common import era5_variables, NCPU


def torch_cwt_batch(
    signals, wavelet="cmor1.5-1.0", 
    scales=np.geomspace(1,512,32), 
    fs=1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            (avg_power/avg_power.sum(dim=-1, keepdim=True) + 1e-12).log()).sum(dim=-1)
    return avg_power, dom_freq, ent, freqs


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


def opt_decomp(
    data: pd.DataFrame, 
    variables: list[str] = list(era5_variables.keys()),
    p_low: int = 18,
    p_high: int = 24,
    p_dt: int = 1,
    gp_n_iter: int = 15
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    # Define search space — e.g., range of candidate periods 
    # hourly → `days_low` days to `days_high` days 
    periods = [Integer(
        p_low, 
        p_high, 
        name="period"
    )]
    # Loop and plot results for each variable
    results = []
    for var in variables:
        series = data[var].dropna()

        # Define objective function to minimize residual error
        def objective(period) -> (Any | float):
            period = p_dt * period[0]
            try:
                result = seasonal_decompose(series, model="additive", period=int(period))
                resid = result.resid.dropna()
                # compute sum of squared residual
                score = np.mean(resid.values ** 2)
            except Exception as e:
                err_tuple = f"\
                    var: {var};  date-range:  {data['date'].min()} - {data['date'].max()}\n\
                    lat: {data['lat'].values};  lon: {data['lon'].values}\
                    period: {period}"
                print(f'Cannot decompose {err_tuple} {e}')
                score = 1000000  # invalid period or failed decomposition
            return score

        # Run Bayesian optimization
        try:
            res = gp_minimize(
                objective,
                dimensions=periods,
                n_calls=gp_n_iter,
                n_random_starts=4,
                random_state=42,
                n_jobs=NCPU
            )
        except Exception as e:
            err_tuple = f"\
                var: {var};  date-range:  {data['date'].min()} - {data['date'].max()}\n\
                lat: {data['lat'].values};  lon: {data['lon'].values}"
            print(f'Cannot decompose {err_tuple}: {e}')
            

        # parse "optimal" period
        period = res.x[0]  # type: ignore
        
        # get final decomposition
        result = seasonal_decompose(series, model="additive", period=period)

        # overload decomposition
        data[f'{var}_season'] = result.seasonal
        data[f'{var}_trend'] = result.trend
        data[f'{var}_resid'] = result.resid

        results += [{
            'time': data['time'].unique(),
            'lon': data['lon'].unique()[0],
            'lat': data['lat'].unique()[0],
            'decomp': result,
            'period': period
        }]

    return data, results
