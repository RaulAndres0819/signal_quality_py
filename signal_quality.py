import pickle
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis, iqr, entropy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class SignalQuality():
    def __init__(self, use_ensemble=False):
        self.use_ensemble = use_ensemble
        
        self.load_models()
                
        filter_params = {
            'order': 5,
            'cutoff_low': 0.35,  
            'cutoff_high': 45,
            'type': 'bandpass',  
            'design': 'butter'  
        }
        
        self.iir_b, self.iir_a = butter(N=filter_params['order'], Wn=[filter_params['cutoff_low'], filter_params['cutoff_high']], 
                      btype=filter_params['type'], fs=500) 
        
    def load_models(self):
        # 3s model1
        with open('model1_4_19_24.pkl', 'rb') as file:
            self.model1_3s = pickle.load(file)
            print("==== model1_4_19_24.pkl ===== ")
            print(self.model1_3s)

        #5s model1
        with open('model5s_1_4_19_24.pkl', 'rb') as file:
            self.model1_5s = pickle.load(file)
            print("==== model5s_1_4_19_24.pkl ===== ")
            print(self.model1_5s)

        if self.use_ensemble:
            # 3s model2
            with open('model3s_2_4_19_24.pkl', 'rb') as file:
                self.model2_3s = pickle.load(file)
                print("==== model3s_2_4_19_24.pkl ===== ")
                print(self.model2_3s)
            # 3s model3
            with open('model3s_3_4_19_24.pkl', 'rb') as file:
                self.model2_5s = pickle.load(file)
                print("==== model3s_3_4_19_24.pkl ===== ")
                print(self.model2_5s)

            # 5s model2
            with open('model5s_2_4_19_24.pkl', 'rb') as file:
                self.model3_3s = pickle.load(file)
                print("==== model5s_2_4_19_24.pkl ===== ")
                print(self.model3_3s)
            # 5s model3
            with open('model5s_3_4_19_24.pkl', 'rb') as file:
                self.model3_5s = pickle.load(file)
                print("==== model5s_3_4_19_24.pkl ===== ")
                print(self.model3_5s)

            

        
        self.feature_names = ['std_dev max', 'skewness max',
       'kurtosis min', 'range_of_data max', 'iqr_value max', 'data_entropy max', 'p95 max', 'hf_energy max']

    def filter_ecg(self, ecg):
        self.ecg_filtered = filtfilt(self.iir_b, self.iir_a, ecg)
            
    
    def process_ecg_window(self, ecg, win_len):
        # win_len in s to know which model to use
        self.win_len = win_len
        # filter the ecg
        self.filter_ecg(ecg)
        # Extract distribution features
        dist_features = self.extract_distribution_features()
        # Extract frequency (emg) features
        hf_energy = self.extract_frequency_features()
        # Group features
        features = dist_features + [hf_energy]
        features_df = pd.DataFrame([features], columns=self.feature_names)
        # classify
        
        if self.use_ensemble:
            signal_quality_estimate = self.ensemble_predict(features_df)
        else:
            if self.win_len == 3:
                signal_quality_estimate = self.model1_3s.predict(features_df)
            else:
                signal_quality_estimate = self.model1_5s.predict(features_df)
        return signal_quality_estimate
        

    def extract_distribution_features(self):
        # Normalize
        data = self.ecg_filtered
        data =  (data - np.mean(data)) / (np.std(data))

        # features
        dist_features = [np.std(data), skew(data), kurtosis(data), np.ptp(data),
            iqr(data), entropy(np.histogram(data, bins=30)[0]), np.percentile(data, 95)]
        
        return dist_features
    
    def extract_frequency_features(self):
        # Pre-process to enhance R-waves and shift noise freq up
        signal = np.square(np.diff(self.ecg_filtered))  
        # Get spectrum  (win_len > 3 or 5s)                        
        freq, Pxx = welch(signal, fs=500, nperseg=len(signal), detrend='linear')

         # Energy in HF band > 100Hz
        f100_index = np.searchsorted(freq, 100, side='left')
        hf_energy = np.sum(signal[f100_index:])

        return hf_energy
    
    def ensemble_predict(self, features_df):
        if self.win_len == 3:
            model1_result = self.model1_3s.predict(features_df)
            model2_result = self.model2_3s.predict(features_df)
            model3_result = self.model3_3s.predict(features_df)

            majority_vote = ((model1_result + model2_result + model3_result) >= 2).astype(int)   

        if self.win_len == 5:
            model1_result = self.model1_5s.predict(features_df)
            model2_result = self.model2_5s.predict(features_df)
            model3_result = self.model3_5s.predict(features_df)

            majority_vote = ((model1_result + model2_result + model3_result) >= 2).astype(int)   
        
        return majority_vote


signalQuality = SignalQuality()


        
