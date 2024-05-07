import pandas as pd
import signal_quality as sq
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def read_hbf(file_path):
    # reads in a .hbf file and returns a pandas df
    if file_path is not None:
        data = pd.read_csv(file_path, delim_whitespace=True, header=None, usecols=[1,2,3])
        data.columns = ['a', 'b', 'c']
    
    return data

def process_signal_quality(data, file_path=None, winlen=3, use_ensemble=True):
    if file_path is not None:
        data = read_hbf(file_path)
    
    if data is None:
        raise ValueError("No data provided. Please provide a DataFrame, an array, or a file path.")
    
    fs = 500
    # window length in seconds
    window_size = winlen*fs
    # step size in seconds
    step_size = 1*fs

    overlap = window_size - step_size
    num_windows = (len(data) - overlap) // step_size

    signal_quality_processor = sq.SignalQuality(use_ensemble=use_ensemble)

    noise_df = pd.DataFrame()
    

    for col in data.columns:
        ecg = data[col]  
        noise_signal = []
        timestamp = []    

        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            ecg_segment = ecg[start_idx:end_idx]

            signal_quality_estimate = signal_quality_processor.process_ecg_window(ecg_segment, winlen)

            noise_signal.append(signal_quality_estimate[0])

            timestamp.append(start_idx + window_size // 2)
        
        noise_df[col] = noise_signal

    return noise_df, timestamp

def plot_signal_quality(ecg_df, noise_df, timestamp):

    for col in ecg_df.columns:
        ecg = ecg_df[col]
        noise_signal = noise_df[col]

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        time_axis = np.arange(len(ecg)) / 500

        axs[0].plot(time_axis, ecg, label=f'Original ECG')
        axs[0].set_title(f'Unfiltered ECG')
        axs[0].set_xlabel('Time')

        # filter the ECG to plot
        b, a = butter(N=5, Wn=[0.35, 45], btype='bandpass', fs=500)
        ecg_filtered = filtfilt(b,a,ecg)
        axs[1].plot(time_axis, ecg_filtered)
        axs[1].set_title(f'Filtered ECG')
        axs[1].set_xlabel('Time')

        axs[2].step(np.array(timestamp)/500, noise_signal, label=f'Quality')
        axs[2].set_title(f'Signal Quality')
        axs[2].set_xlabel('Time')

        plt.tight_layout()
        plt.show()

def summarize_signal_quality(noise_df):

    result = []
    for col in noise_df.columns:
        noise_signal = noise_df[col]

        segment_length_greater_10s = check_consecutive_ones_np(noise_signal)
        result.append(segment_length_greater_10s)

    overall_result = np.min(result)

    return overall_result, result

def check_consecutive_ones_np(data):
    
    padded_data = np.pad(data, (1, 1), mode='constant', constant_values=0)
    edges = np.diff(padded_data)
    start_indices = np.where(edges == 1)[0]
    end_indices = np.where(edges == -1)[0]
    lengths = end_indices - start_indices
    if any(lengths > 10):
        return np.max(lengths)
    else:
        return 0
