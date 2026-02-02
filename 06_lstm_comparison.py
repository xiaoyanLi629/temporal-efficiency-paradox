"""
Step 6: LSTM vs Ridge Encoding Comparison
LSTM与Ridge编码模型对比分析

This script compares encoding performance between:
1. Ridge Regression (no temporal memory)
2. LSTM (captures long-range temporal dependencies)

Key Hypothesis: If DMN requires long-range temporal integration,
LSTM should improve DMN encoding more than Visual network.
"""

import os
import sys
import numpy as np
import h5py
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Schaefer 1000 parcellation network indices
SCHAEFER_NETWORK_INDICES = {
    'Visual': list(range(0, 60)) + list(range(500, 560)),
    'Somatomotor': list(range(60, 130)) + list(range(560, 630)),
    'DorsalAttention': list(range(130, 175)) + list(range(630, 685)),
    'VentralAttention': list(range(175, 220)) + list(range(685, 740)),
    'Limbic': list(range(220, 250)) + list(range(740, 780)),
    'Frontoparietal': list(range(250, 330)) + list(range(780, 870)),
    'Default': list(range(330, 500)) + list(range(870, 1000))
}

NETWORK_ORDER = ['Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention', 
                 'Limbic', 'Frontoparietal', 'Default']


class LSTMEncoder(nn.Module):
    """LSTM-based encoding model for fMRI prediction."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.3):
        super(LSTMEncoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class LSTMRidgeComparison:
    """Compare LSTM and Ridge encoding models across brain networks."""
    
    def __init__(self, data_dir, output_dir, device='cuda'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
        self.num_regions = 1000
        
        # Model parameters
        self.seq_length = 20  # ~30 seconds of context
        self.hidden_dim = 64
        self.num_layers = 2
        self.batch_size = 128
        self.epochs = 30
        self.lr = 0.001
        
        # Ridge parameters
        self.alpha = 1000
        
    def load_fmri_data(self, subject, max_samples=50000):
        """Load fMRI time series data."""
        fmri_path = os.path.join(
            self.data_dir, 'fmri', subject, 'func',
            f'{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
        )
        
        if not os.path.exists(fmri_path):
            return None
            
        with h5py.File(fmri_path, 'r') as f:
            keys = list(f.keys())[:4]  # Use first 4 episodes for speed
            data_list = [f[key][:] for key in sorted(keys)]
            fmri_data = np.concatenate(data_list, axis=0)
        
        # Limit samples for computational efficiency
        if len(fmri_data) > max_samples:
            fmri_data = fmri_data[:max_samples]
            
        return fmri_data
    
    def load_features(self):
        """Load and combine stimulus features."""
        feature_dir = os.path.join(self.data_dir, 'features', 'official_stimulus_features', 'pca', 'friends_movie10')
        
        features_list = []
        for modality in ['visual', 'audio', 'language']:
            feature_path = os.path.join(feature_dir, modality, 'features_train.npy')
            if os.path.exists(feature_path):
                feat = np.load(feature_path, allow_pickle=True)
                if isinstance(feat, np.ndarray) and feat.dtype == object:
                    feat = feat.item()
                    # Concatenate episodes (use first 4 for speed)
                    sorted_keys = sorted(feat.keys())[:4]
                    feat = np.concatenate([feat[k] for k in sorted_keys], axis=0)
                features_list.append(feat)
                print(f"  ✓ Loaded {modality}: {feat.shape}")
        
        if features_list:
            # Stack features horizontally
            combined = np.hstack(features_list)
            print(f"  ✓ Combined features: {combined.shape}")
            return combined
        return None
    
    def create_sequences(self, features, fmri, region_idx):
        """Create sequences for LSTM training."""
        n_samples = min(len(features), len(fmri))
        
        X_seq = []
        y_seq = []
        
        for i in range(self.seq_length, n_samples):
            X_seq.append(features[i-self.seq_length:i])
            y_seq.append(fmri[i, region_idx])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_ridge(self, X, y):
        """Train Ridge regression model."""
        # Flatten temporal dimension for Ridge
        X_flat = X.reshape(X.shape[0], -1)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        predictions = np.zeros(len(y))
        
        for train_idx, test_idx in kf.split(X_flat):
            model = Ridge(alpha=self.alpha)
            model.fit(X_flat[train_idx], y[train_idx])
            predictions[test_idx] = model.predict(X_flat[test_idx])
        
        # Compute correlation
        corr, _ = pearsonr(predictions, y)
        return max(0, corr)
    
    def train_lstm(self, X, y, save_path=None):
        """Train LSTM model and optionally save weights."""
        # Normalize
        X = zscore(X, axis=0)
        y = zscore(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Create dataset and split
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        
        train_dataset = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
        test_dataset = TensorDataset(X_tensor[n_train:], y_tensor[n_train:])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X.shape[2]
        model = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=1
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Training
        model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test = X_tensor[n_train:].to(self.device)
            y_test = y_tensor[n_train:].numpy().flatten()
            
            predictions = model(X_test).cpu().numpy().flatten()
            
            corr, _ = pearsonr(predictions, y_test)
        
        # Save model weights if path provided
        if save_path is not None:
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dim': self.hidden_dim,
                    'num_layers': self.num_layers,
                    'output_dim': 1
                },
                'correlation': float(max(0, corr))
            }, save_path)
        
        return max(0, corr), model
    
    def analyze_network(self, features, fmri, network_name, subject_name=None):
        """Analyze a single network with both models and save best LSTM model."""
        region_indices = SCHAEFER_NETWORK_INDICES[network_name]
        
        # Sample regions for efficiency
        sample_regions = region_indices[::max(1, len(region_indices)//10)][:10]
        
        ridge_scores = []
        lstm_scores = []
        best_lstm_score = -1
        best_lstm_model = None
        best_region_idx = None
        
        for region_idx in sample_regions:
            X_seq, y_seq = self.create_sequences(features, fmri, region_idx)
            
            if len(X_seq) < 1000:
                continue
            
            # Subsample for speed
            idx = np.random.choice(len(X_seq), min(5000, len(X_seq)), replace=False)
            X_sub = X_seq[idx]
            y_sub = y_seq[idx]
            
            # Train both models
            ridge_score = self.train_ridge(X_sub, y_sub)
            lstm_score, lstm_model = self.train_lstm(X_sub, y_sub)
            
            ridge_scores.append(ridge_score)
            lstm_scores.append(lstm_score)
            
            # Track best LSTM model for this network
            if lstm_score > best_lstm_score:
                best_lstm_score = lstm_score
                best_lstm_model = lstm_model
                best_region_idx = region_idx
        
        # Save the best LSTM model for this network
        if best_lstm_model is not None and subject_name is not None:
            models_dir = os.path.join(self.output_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f'{subject_name}_{network_name}_lstm.pt')
            torch.save({
                'model_state_dict': best_lstm_model.state_dict(),
                'model_config': {
                    'input_dim': best_lstm_model.lstm.input_size,
                    'hidden_dim': self.hidden_dim,
                    'num_layers': self.num_layers,
                    'output_dim': 1
                },
                'network': network_name,
                'subject': subject_name,
                'best_region_idx': int(best_region_idx),
                'correlation': float(best_lstm_score)
            }, model_path)
        
        return {
            'ridge_mean': np.mean(ridge_scores) if ridge_scores else 0,
            'ridge_std': np.std(ridge_scores) if ridge_scores else 0,
            'lstm_mean': np.mean(lstm_scores) if lstm_scores else 0,
            'lstm_std': np.std(lstm_scores) if lstm_scores else 0,
            'improvement': np.mean(lstm_scores) - np.mean(ridge_scores) if lstm_scores else 0
        }
    
    def run_comparison(self):
        """Run full comparison analysis."""
        print("=" * 60)
        print("LSTM vs Ridge Encoding Comparison")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        all_results = {}
        
        for subject in self.subjects:
            print(f"\n{'='*40}")
            print(f"Processing {subject}")
            print('='*40)
            
            # Load data
            fmri = self.load_fmri_data(subject)
            if fmri is None:
                print(f"  ⚠ Could not load fMRI for {subject}")
                continue
                
            features = self.load_features()
            if features is None:
                print(f"  ⚠ Could not load features")
                continue
            
            # Align lengths
            n_samples = min(len(features), len(fmri))
            features = features[:n_samples]
            fmri = fmri[:n_samples]
            
            print(f"  Data shape: features {features.shape}, fMRI {fmri.shape}")
            
            subject_results = {}
            
            for network in NETWORK_ORDER:
                print(f"\n  Analyzing {network}...")
                results = self.analyze_network(features, fmri, network, subject_name=subject)
                subject_results[network] = results
                
                print(f"    Ridge: {results['ridge_mean']:.4f} ± {results['ridge_std']:.4f}")
                print(f"    LSTM:  {results['lstm_mean']:.4f} ± {results['lstm_std']:.4f}")
                print(f"    Improvement: {results['improvement']*100:.2f}%")
            
            all_results[subject] = subject_results
        
        # Print saved models info
        models_dir = os.path.join(self.output_dir, 'models')
        if os.path.exists(models_dir):
            saved_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
            print(f"\n✓ Saved {len(saved_models)} LSTM model weights to: {models_dir}")
        
        # Aggregate across subjects
        aggregated = {}
        for network in NETWORK_ORDER:
            ridge_means = [all_results[s][network]['ridge_mean'] 
                         for s in all_results if network in all_results[s]]
            lstm_means = [all_results[s][network]['lstm_mean'] 
                        for s in all_results if network in all_results[s]]
            
            aggregated[network] = {
                'ridge_mean': np.mean(ridge_means),
                'ridge_std': np.std(ridge_means),
                'lstm_mean': np.mean(lstm_means),
                'lstm_std': np.std(lstm_means),
                'improvement': np.mean(lstm_means) - np.mean(ridge_means),
                'improvement_percent': (np.mean(lstm_means) - np.mean(ridge_means)) / (np.mean(ridge_means) + 1e-10) * 100
            }
        
        # Save results
        results_data = {
            'subject_results': all_results,
            'aggregated': aggregated,
            'parameters': {
                'seq_length': self.seq_length,
                'hidden_dim': self.hidden_dim,
                'epochs': self.epochs,
                'alpha': self.alpha
            }
        }
        
        results_path = os.path.join(self.output_dir, 'lstm_ridge_comparison.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY: LSTM Improvement Over Ridge")
        print("=" * 60)
        print(f"{'Network':<20} {'Ridge':<12} {'LSTM':<12} {'Improvement':<12}")
        print("-" * 60)
        
        for network in NETWORK_ORDER:
            r = aggregated[network]
            print(f"{network:<20} {r['ridge_mean']:.4f}       {r['lstm_mean']:.4f}       {r['improvement_percent']:+.2f}%")
        
        print("\n" + "=" * 60)
        print(f"Results saved to: {results_path}")
        
        return results_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/root/autodl-fs/CCN_Competition/data')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        # 查找最新的运行目录
        project_dir = os.path.dirname(os.path.abspath(__file__))
        runs_dir = os.path.join(project_dir, 'runs')
        if os.path.exists(runs_dir):
            existing_runs = [d for d in os.listdir(runs_dir) if d[0].isdigit()]
            if existing_runs:
                latest_run = sorted(existing_runs)[-1]
                args.output_dir = os.path.join(runs_dir, latest_run, 'lstm_ridge')
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.output_dir = os.path.join(runs_dir, timestamp, 'lstm_ridge')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = os.path.join(project_dir, 'runs', timestamp, 'lstm_ridge')
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    analyzer = LSTMRidgeComparison(args.data_dir, args.output_dir)
    analyzer.run_comparison()


if __name__ == "__main__":
    main()

