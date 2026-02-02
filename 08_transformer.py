"""
Step 8: Transformer Temporal Attention Analysis
Transformer时间注意力分析 - 直接可视化不同网络的时间整合范围

This script uses a Transformer model to:
1. Train encoding models with self-attention mechanism
2. Extract and visualize temporal attention patterns
3. Compare attention span across networks

Key Hypothesis: DMN should show longer-range attention patterns
(attending to distant time points), while Visual should show
short-range attention (focusing on recent time points).
"""

import os
import sys
import numpy as np
import h5py
from scipy.stats import zscore, pearsonr
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

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


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer-based temporal encoder that outputs attention weights.
    """
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=128, dropout=0.1, seq_length=30):
        super(TemporalTransformerEncoder, self).__init__()
        
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.1)
        
        # Transformer encoder layer with attention output
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Custom attention layer for extracting weights
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.attention_weights = None
        
    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Apply transformer
        x_transformed = self.transformer(x)
        
        # Get attention weights using the last layer
        # Query from last position, keys/values from all positions
        query = x_transformed[:, -1:, :]  # (batch, 1, d_model)
        key = x_transformed
        value = x_transformed
        
        attn_output, attn_weights = self.attention_layer(query, key, value)
        
        self.attention_weights = attn_weights  # Store for visualization
        
        # Use attended output for prediction
        output = self.output_proj(attn_output.squeeze(1))
        
        if return_attention:
            return output, attn_weights
        return output


class TransformerAttentionAnalyzer:
    """Analyze temporal attention patterns across brain networks."""
    
    def __init__(self, data_dir, output_dir, device='cuda'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
        self.num_regions = 1000
        
        # Model parameters
        self.seq_length = 30  # ~45 seconds
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 2
        self.batch_size = 64
        self.epochs = 20
        self.lr = 0.001
        
    def load_fmri_data(self, subject, max_samples=30000):
        """Load fMRI time series data."""
        fmri_path = os.path.join(
            self.data_dir, 'fmri', subject, 'func',
            f'{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
        )
        
        if not os.path.exists(fmri_path):
            return None
            
        with h5py.File(fmri_path, 'r') as f:
            keys = list(f.keys())[:4]
            data_list = [f[key][:] for key in sorted(keys)]
            fmri_data = np.concatenate(data_list, axis=0)
        
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
            combined = np.hstack(features_list)
            print(f"  ✓ Combined features: {combined.shape}")
            return combined
        return None
    
    def create_sequences(self, features, fmri, region_idx):
        """Create sequences for Transformer training."""
        n_samples = min(len(features), len(fmri))
        
        X_seq = []
        y_seq = []
        
        for i in range(self.seq_length, n_samples):
            X_seq.append(features[i-self.seq_length:i])
            y_seq.append(fmri[i, region_idx])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_and_extract_attention(self, X, y, save_path=None):
        """Train Transformer and extract attention patterns, optionally save model."""
        # Normalize
        X = zscore(X.reshape(-1, X.shape[-1]), axis=0).reshape(X.shape)
        y = zscore(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Split
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        
        train_dataset = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X.shape[2]
        model = TemporalTransformerEncoder(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            seq_length=self.seq_length
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
        
        # Extract attention patterns on test set
        model.eval()
        attention_patterns = []
        
        with torch.no_grad():
            X_test = X_tensor[n_train:].to(self.device)
            
            # Process in batches
            for i in range(0, len(X_test), self.batch_size):
                batch = X_test[i:i+self.batch_size]
                _, attn = model(batch, return_attention=True)
                attention_patterns.append(attn.cpu().numpy())
        
        attention_patterns = np.concatenate(attention_patterns, axis=0)
        
        # Average attention pattern: (batch, 1, seq_len) -> (seq_len,)
        avg_attention = np.mean(attention_patterns.squeeze(1), axis=0)
        
        # Compute attention span metrics
        # 1. Effective attention span (weighted average position)
        positions = np.arange(self.seq_length)
        attention_span = np.sum(positions * avg_attention) / np.sum(avg_attention)
        
        # 2. Attention entropy (higher = more distributed)
        attention_entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-10))
        
        # 3. Recent vs distant attention ratio
        midpoint = self.seq_length // 2
        recent_attention = np.sum(avg_attention[midpoint:])
        distant_attention = np.sum(avg_attention[:midpoint])
        temporal_ratio = distant_attention / (recent_attention + 1e-10)
        
        # Save model weights if path provided
        if save_path is not None:
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': input_dim,
                    'd_model': self.d_model,
                    'nhead': self.nhead,
                    'num_layers': self.num_layers,
                    'seq_length': self.seq_length
                },
                'attention_span': float(attention_span),
                'attention_entropy': float(attention_entropy),
                'temporal_ratio': float(temporal_ratio)
            }, save_path)
        
        return {
            'avg_attention_pattern': avg_attention.tolist(),
            'attention_span': float(attention_span),
            'attention_entropy': float(attention_entropy),
            'temporal_ratio': float(temporal_ratio),  # >1 means more distant attention
            'recent_attention': float(recent_attention),
            'distant_attention': float(distant_attention)
        }, model
    
    def analyze_network(self, features, fmri, network_name, subject_name=None):
        """Analyze attention patterns for a network and save best model."""
        region_indices = SCHAEFER_NETWORK_INDICES[network_name]
        
        # Sample regions
        sample_regions = region_indices[::max(1, len(region_indices)//5)][:5]
        
        all_patterns = []
        all_metrics = []
        best_attention_span = -1
        best_model = None
        best_region_idx = None
        
        for region_idx in sample_regions:
            X_seq, y_seq = self.create_sequences(features, fmri, region_idx)
            
            if len(X_seq) < 1000:
                continue
            
            # Subsample
            idx = np.random.choice(len(X_seq), min(3000, len(X_seq)), replace=False)
            X_sub = X_seq[idx]
            y_sub = y_seq[idx]
            
            results, model = self.train_and_extract_attention(X_sub, y_sub)
            all_patterns.append(results['avg_attention_pattern'])
            all_metrics.append({
                'attention_span': results['attention_span'],
                'attention_entropy': results['attention_entropy'],
                'temporal_ratio': results['temporal_ratio']
            })
            
            # Track model with best attention span (most interesting pattern)
            if results['attention_span'] > best_attention_span:
                best_attention_span = results['attention_span']
                best_model = model
                best_region_idx = region_idx
        
        # Save the best Transformer model for this network
        if best_model is not None and subject_name is not None:
            models_dir = os.path.join(self.output_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f'{subject_name}_{network_name}_transformer.pt')
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'model_config': {
                    'input_dim': best_model.input_proj.in_features,
                    'd_model': self.d_model,
                    'nhead': self.nhead,
                    'num_layers': self.num_layers,
                    'seq_length': self.seq_length
                },
                'network': network_name,
                'subject': subject_name,
                'best_region_idx': int(best_region_idx),
                'attention_span': float(best_attention_span)
            }, model_path)
        
        # Average across regions
        if all_patterns:
            avg_pattern = np.mean(all_patterns, axis=0)
            avg_metrics = {
                'attention_span': np.mean([m['attention_span'] for m in all_metrics]),
                'attention_entropy': np.mean([m['attention_entropy'] for m in all_metrics]),
                'temporal_ratio': np.mean([m['temporal_ratio'] for m in all_metrics])
            }
        else:
            avg_pattern = np.zeros(self.seq_length)
            avg_metrics = {'attention_span': 0, 'attention_entropy': 0, 'temporal_ratio': 0}
        
        return {
            'avg_attention_pattern': avg_pattern.tolist(),
            'metrics': avg_metrics
        }
    
    def run_analysis(self):
        """Run full attention analysis."""
        print("=" * 60)
        print("Transformer Temporal Attention Analysis")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        all_results = {}
        
        for subject in self.subjects[:2]:  # Use 2 subjects for speed
            print(f"\n{'='*40}")
            print(f"Processing {subject}")
            print('='*40)
            
            # Load data
            fmri = self.load_fmri_data(subject)
            if fmri is None:
                continue
                
            features = self.load_features()
            if features is None:
                continue
            
            # Align
            n_samples = min(len(features), len(fmri))
            features = features[:n_samples]
            fmri = fmri[:n_samples]
            
            print(f"  Data shape: features {features.shape}, fMRI {fmri.shape}")
            
            subject_results = {}
            
            for network in NETWORK_ORDER:
                print(f"\n  Analyzing {network}...")
                results = self.analyze_network(features, fmri, network, subject_name=subject)
                subject_results[network] = results
                
                m = results['metrics']
                print(f"    Attention Span: {m['attention_span']:.2f}")
                print(f"    Temporal Ratio (distant/recent): {m['temporal_ratio']:.3f}")
            
            all_results[subject] = subject_results
        
        # Print saved models info
        models_dir = os.path.join(self.output_dir, 'models')
        if os.path.exists(models_dir):
            saved_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
            print(f"\n✓ Saved {len(saved_models)} Transformer model weights to: {models_dir}")
        
        # Aggregate
        aggregated = {}
        for network in NETWORK_ORDER:
            patterns = [all_results[s][network]['avg_attention_pattern'] 
                       for s in all_results if network in all_results[s]]
            metrics = [all_results[s][network]['metrics'] 
                      for s in all_results if network in all_results[s]]
            
            if patterns:
                aggregated[network] = {
                    'avg_attention_pattern': np.mean(patterns, axis=0).tolist(),
                    'attention_span': np.mean([m['attention_span'] for m in metrics]),
                    'attention_entropy': np.mean([m['attention_entropy'] for m in metrics]),
                    'temporal_ratio': np.mean([m['temporal_ratio'] for m in metrics])
                }
        
        # Save results
        results_data = {
            'subject_results': all_results,
            'aggregated': aggregated,
            'parameters': {
                'seq_length': self.seq_length,
                'd_model': self.d_model,
                'nhead': self.nhead
            }
        }
        
        results_path = os.path.join(self.output_dir, 'transformer_attention_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY: Temporal Attention Patterns")
        print("=" * 60)
        print(f"{'Network':<18} {'Attn Span':<12} {'Entropy':<12} {'Distant/Recent':<15}")
        print("-" * 60)
        
        for network in NETWORK_ORDER:
            if network in aggregated:
                r = aggregated[network]
                print(f"{network:<18} {r['attention_span']:.2f}         {r['attention_entropy']:.3f}        {r['temporal_ratio']:.3f}")
        
        print("\n" + "=" * 60)
        print("KEY FINDING:")
        print("-" * 60)
        
        # Find network with longest attention span
        max_span_network = max(aggregated.keys(), key=lambda x: aggregated[x]['attention_span'])
        min_span_network = min(aggregated.keys(), key=lambda x: aggregated[x]['attention_span'])
        
        print(f"Longest Attention Span: {max_span_network} ({aggregated[max_span_network]['attention_span']:.2f})")
        print(f"Shortest Attention Span: {min_span_network} ({aggregated[min_span_network]['attention_span']:.2f})")
        print(f"\n→ {max_span_network} attends to more distant time points")
        print(f"→ {min_span_network} focuses on recent time points")
        print(f"\nThis directly visualizes WHY slow networks benefit from longer windows!")
        
        print(f"\nResults saved to: {results_path}")
        
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
                args.output_dir = os.path.join(runs_dir, latest_run, 'transformer_attention')
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.output_dir = os.path.join(runs_dir, timestamp, 'transformer_attention')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = os.path.join(project_dir, 'runs', timestamp, 'transformer_attention')
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    analyzer = TransformerAttentionAnalyzer(args.data_dir, args.output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

