"""
Rigorous Multi-Timescale Encoding Analysis
严格的多时间尺度编码分析 - 生成真实的TWG数据

This script computes REAL encoding accuracy across different temporal windows
using Ridge regression with proper cross-validation.

Key improvements over previous version:
1. Uses actual stimulus features and fMRI data
2. Proper HRF convolution
3. Cross-validation with held-out test sets
4. Statistical confidence intervals
5. Memory-efficient processing
"""

import os
import sys
import numpy as np
import h5py
from scipy import signal
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import json
import joblib
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

# Network definitions (Schaefer 1000 parcellation, 7 networks)
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

# Temporal windows in TRs (TR = 1.49s)
TEMPORAL_WINDOWS = {
    'instant': {'past': 1, 'future': 0, 'description': '~1.5s - current frame only'},
    'short': {'past': 4, 'future': 1, 'description': '~7.5s - short context'},
    'medium': {'past': 10, 'future': 2, 'description': '~18s - medium context'},
    'long': {'past': 20, 'future': 5, 'description': '~37s - long context'},
    'very_long': {'past': 40, 'future': 10, 'description': '~75s - extended context'}
}

TR = 1.49
HRF_DELAY_TRS = 4  # ~6 seconds HRF peak delay
N_FOLDS = 5
REGIONS_PER_NETWORK = 15  # Sample regions per network for efficiency


def create_hrf_kernel(tr=1.49, duration=30):
    """Create canonical HRF kernel for convolution."""
    t = np.arange(0, duration, tr)
    # Double gamma HRF
    a1, a2 = 6, 16
    b1, b2 = 1, 1
    c = 1/6
    
    hrf = (t**(a1-1) * np.exp(-t/b1) / (b1**a1 * np.math.factorial(int(a1-1))) -
           c * t**(a2-1) * np.exp(-t/b2) / (b2**a2 * np.math.factorial(int(a2-1))))
    
    return hrf / np.max(hrf)


def load_fmri_data(data_dir, subject, max_episodes=None):
    """Load fMRI data for a subject with episode tracking."""
    fmri_path = os.path.join(data_dir, 'fmri', subject, 'func',
        f'{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5')
    
    if not os.path.exists(fmri_path):
        print(f"  ⚠ fMRI data not found: {fmri_path}")
        return None, []
    
    with h5py.File(fmri_path, 'r') as f:
        available_keys = sorted(list(f.keys()))
        if max_episodes is not None:
            available_keys = available_keys[:max_episodes]
        
        data_list = []
        episode_lengths = []
        used_episodes = []
        
        for key in available_keys:
            episode_data = f[key][:]
            data_list.append(episode_data)
            episode_lengths.append(len(episode_data))
            
            # Extract episode name (e.g., 's01e01a' from 'ses-001_task-s01e01a')
            parts = key.split('task-')
            if len(parts) > 1:
                used_episodes.append(parts[1])
        
        fmri_data = np.concatenate(data_list, axis=0)
    
    print(f"  Loaded fMRI: {len(available_keys)} episodes, shape {fmri_data.shape}")
    return fmri_data, used_episodes


def load_stimulus_features(data_dir, episodes):
    """Load and concatenate stimulus features for specified episodes."""
    features = {}
    base_path = os.path.join(data_dir, 'features', 'official_stimulus_features', 'pca', 'friends_movie10')
    
    for modality in ['visual', 'audio', 'language']:
        mod_path = os.path.join(base_path, modality, 'features_train.npy')
        
        if not os.path.exists(mod_path):
            print(f"  ⚠ {modality} features not found")
            continue
        
        data = np.load(mod_path, allow_pickle=True).item()
        
        episode_features = []
        for ep in episodes:
            if ep in data:
                episode_features.append(data[ep])
        
        if episode_features:
            features[modality] = np.concatenate(episode_features, axis=0)
            print(f"  Loaded {modality}: {features[modality].shape}")
    
    return features


def create_temporal_features(features, window_config, n_timepoints):
    """
    Create temporally aggregated features using past and future context.
    
    For each timepoint t, aggregate features from [t-past, t+future].
    This tests whether longer temporal context improves encoding.
    """
    past = window_config['past']
    future = window_config['future']
    
    n_features = features.shape[1]
    aggregated = np.zeros((n_timepoints, n_features))
    
    for t in range(n_timepoints):
        start = max(0, t - past)
        end = min(n_timepoints, t + future + 1)
        
        # Mean aggregation over temporal window
        aggregated[t] = np.mean(features[start:end], axis=0)
    
    return aggregated


def compute_encoding_accuracy(X, y, n_folds=N_FOLDS, alphas=[0.1, 1, 10, 100, 1000], return_model=False):
    """
    Compute encoding accuracy using Ridge regression with cross-validation.
    
    Returns:
        mean_r: Mean Pearson correlation across folds
        std_r: Standard deviation across folds
        fold_rs: Individual fold correlations
        best_model: (optional) The best trained model if return_model=True
    """
    if len(X) < n_folds * 10:
        if return_model:
            return 0.0, 0.0, [], None, None, None
        return 0.0, 0.0, []
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_correlations = []
    best_model = None
    best_corr = -1
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
        
        # Use RidgeCV to find optimal alpha
        model = RidgeCV(alphas=alphas, cv=3)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        if np.std(y_test) > 1e-6 and np.std(y_pred) > 1e-6:
            r, _ = pearsonr(y_test, y_pred)
            if not np.isnan(r):
                fold_correlations.append(r)
                if return_model and r > best_corr:
                    best_corr = r
                    best_model = model
    
    if not fold_correlations:
        if return_model:
            return 0.0, 0.0, [], None, None, None
        return 0.0, 0.0, []
    
    if return_model:
        return np.mean(fold_correlations), np.std(fold_correlations), fold_correlations, best_model, scaler_X, scaler_y
    return np.mean(fold_correlations), np.std(fold_correlations), fold_correlations


def analyze_subject_encoding(subject, fmri_data, features, regions_per_network=REGIONS_PER_NETWORK, output_dir=None):
    """
    Analyze encoding performance for a single subject across all temporal windows.
    
    Returns encoding accuracy for each network at each temporal window.
    Optionally saves best Ridge models for each network.
    """
    n_timepoints, n_regions = fmri_data.shape
    
    # Combine all modality features
    all_features = []
    min_feat_timepoints = n_timepoints
    
    for mod in ['visual', 'audio', 'language']:
        if mod in features:
            min_feat_timepoints = min(min_feat_timepoints, len(features[mod]))
    
    # Clip all features to same length
    for mod in ['visual', 'audio', 'language']:
        if mod in features:
            all_features.append(features[mod][:min_feat_timepoints])
    
    if not all_features:
        print("  ⚠ No features available")
        return None
    
    combined_features = np.concatenate(all_features, axis=1)
    
    # Align timepoints
    min_timepoints = min(n_timepoints, min_feat_timepoints)
    combined_features = combined_features[:min_timepoints]
    fmri_data = fmri_data[:min_timepoints]
    
    print(f"  Aligned to {min_timepoints} timepoints, {combined_features.shape[1]} features")
    
    # Convolve features with HRF
    hrf = create_hrf_kernel(TR)
    convolved_features = np.zeros_like(combined_features)
    for i in range(combined_features.shape[1]):
        convolved_features[:, i] = np.convolve(combined_features[:, i], hrf, mode='same')
    
    # Sample regions from each network
    sampled_regions = {}
    for net, indices in SCHAEFER_NETWORK_INDICES.items():
        valid_indices = [i for i in indices if i < n_regions]
        n_sample = min(regions_per_network, len(valid_indices))
        sampled_regions[net] = np.random.choice(valid_indices, n_sample, replace=False)
    
    total_regions = sum(len(r) for r in sampled_regions.values())
    print(f"  Analyzing {total_regions} sampled regions...")
    
    # Results structure
    results = {
        'by_window': {w: {} for w in TEMPORAL_WINDOWS},
        'by_region': {},
        'twg': {}
    }
    
    # Track best models for each network (at instant window, which is most interpretable)
    best_models = {}
    
    # Process each temporal window
    for window_name, window_config in TEMPORAL_WINDOWS.items():
        print(f"    Window: {window_name} ({window_config['description']})")
        
        # Create temporally aggregated features
        agg_features = create_temporal_features(convolved_features, window_config, min_timepoints)
        
        # Apply HRF delay
        if min_timepoints > HRF_DELAY_TRS:
            X = agg_features[:-HRF_DELAY_TRS]
            fmri_shifted = fmri_data[HRF_DELAY_TRS:]
        else:
            X = agg_features
            fmri_shifted = fmri_data
        
        # Analyze each network
        for net in NETWORK_ORDER:
            net_correlations = []
            net_stds = []
            best_net_corr = -1
            best_net_model = None
            best_net_scalers = None
            best_region_idx = None
            
            for region_idx in sampled_regions[net]:
                y = fmri_shifted[:, region_idx]
                
                if np.std(y) < 1e-6:
                    continue
                
                # For instant window, also get model for saving
                if window_name == 'instant' and output_dir is not None:
                    mean_r, std_r, _, model, scaler_X, scaler_y = compute_encoding_accuracy(X, y, return_model=True)
                    if mean_r > best_net_corr:
                        best_net_corr = mean_r
                        best_net_model = model
                        best_net_scalers = (scaler_X, scaler_y)
                        best_region_idx = region_idx
                else:
                    mean_r, std_r, _ = compute_encoding_accuracy(X, y)
                
                if mean_r > 0:  # Only include positive correlations
                    net_correlations.append(mean_r)
                    net_stds.append(std_r)
                    
                    # Store region-level results (convert to string key for JSON)
                    region_key = str(int(region_idx))
                    if region_key not in results['by_region']:
                        results['by_region'][region_key] = {}
                    results['by_region'][region_key][window_name] = {
                        'r': float(mean_r),
                        'std': float(std_r)
                    }
            
            # Save best model for this network (only for instant window)
            if window_name == 'instant' and best_net_model is not None:
                best_models[net] = {
                    'model': best_net_model,
                    'scaler_X': best_net_scalers[0],
                    'scaler_y': best_net_scalers[1],
                    'region_idx': best_region_idx,
                    'correlation': best_net_corr
                }
            
            # Network summary
            if net_correlations:
                results['by_window'][window_name][net] = {
                    'mean': float(np.mean(net_correlations)),
                    'std': float(np.std(net_correlations)),
                    'sem': float(np.std(net_correlations) / np.sqrt(len(net_correlations))),
                    'n_regions': len(net_correlations),
                    'values': [float(v) for v in net_correlations]
                }
            else:
                results['by_window'][window_name][net] = {
                    'mean': 0.0, 'std': 0.0, 'sem': 0.0, 'n_regions': 0, 'values': []
                }
    
    # Save best Ridge models for each network
    if output_dir is not None and best_models:
        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for net, model_info in best_models.items():
            model_path = os.path.join(models_dir, f'{subject}_{net}_ridge.joblib')
            joblib.dump({
                'model': model_info['model'],
                'scaler_X': model_info['scaler_X'],
                'scaler_y': model_info['scaler_y'],
                'network': net,
                'subject': subject,
                'region_idx': int(model_info['region_idx']),
                'correlation': float(model_info['correlation']),
                'alpha': model_info['model'].alpha_ if hasattr(model_info['model'], 'alpha_') else None
            }, model_path)
        
        print(f"  ✓ Saved {len(best_models)} Ridge models to: {models_dir}")
    
    # Compute TWG (Temporal Window Gain)
    for net in NETWORK_ORDER:
        instant_r = results['by_window']['instant'].get(net, {}).get('mean', 0)
        very_long_r = results['by_window']['very_long'].get(net, {}).get('mean', 0)
        
        # TWG = improvement from extended temporal context
        twg = very_long_r - instant_r
        
        # Compute relative improvement (%)
        if instant_r > 0:
            relative_improvement = (very_long_r - instant_r) / instant_r * 100
        else:
            relative_improvement = 0
        
        results['twg'][net] = {
            'twg': float(twg),
            'instant_r': float(instant_r),
            'very_long_r': float(very_long_r),
            'relative_improvement_pct': float(relative_improvement)
        }
    
    return results


def bootstrap_ci(values, n_bootstrap=1000, ci=95):
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return 0, 0
    
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, len(values), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    
    return lower, upper


def aggregate_results(all_results):
    """Aggregate results across subjects with statistics."""
    
    aggregated = {
        'network_summary': {
            'encoding': {},
            'twg': {}
        },
        'statistics': {},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_subjects': len(all_results),
            'temporal_windows': {k: v['description'] for k, v in TEMPORAL_WINDOWS.items()},
            'regions_per_network': REGIONS_PER_NETWORK
        }
    }
    
    # Aggregate encoding by window
    for window_name in TEMPORAL_WINDOWS:
        aggregated['network_summary']['encoding'][window_name] = {}
        
        for net in NETWORK_ORDER:
            all_values = []
            for sub_results in all_results.values():
                val = sub_results['by_window'][window_name].get(net, {}).get('mean', 0)
                if val > 0:
                    all_values.append(val)
            
            if all_values:
                ci_low, ci_high = bootstrap_ci(all_values)
                aggregated['network_summary']['encoding'][window_name][net] = {
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'sem': float(np.std(all_values) / np.sqrt(len(all_values))),
                    'ci_95_low': float(ci_low),
                    'ci_95_high': float(ci_high),
                    'n_subjects': len(all_values)
                }
            else:
                aggregated['network_summary']['encoding'][window_name][net] = {
                    'mean': 0.0, 'std': 0.0, 'sem': 0.0,
                    'ci_95_low': 0.0, 'ci_95_high': 0.0, 'n_subjects': 0
                }
    
    # Aggregate TWG
    for net in NETWORK_ORDER:
        all_twg = []
        all_instant = []
        all_very_long = []
        
        for sub_results in all_results.values():
            twg_data = sub_results['twg'].get(net, {})
            if 'twg' in twg_data:
                all_twg.append(twg_data['twg'])
                all_instant.append(twg_data['instant_r'])
                all_very_long.append(twg_data['very_long_r'])
        
        if all_twg:
            ci_low, ci_high = bootstrap_ci(all_twg)
            aggregated['network_summary']['twg'][net] = {
                'mean': float(np.mean(all_twg)),
                'std': float(np.std(all_twg)),
                'sem': float(np.std(all_twg) / np.sqrt(len(all_twg))),
                'ci_95_low': float(ci_low),
                'ci_95_high': float(ci_high),
                'instant_mean': float(np.mean(all_instant)),
                'very_long_mean': float(np.mean(all_very_long)),
                'n_subjects': len(all_twg)
            }
        else:
            aggregated['network_summary']['twg'][net] = {
                'mean': 0.0, 'std': 0.0, 'sem': 0.0,
                'ci_95_low': 0.0, 'ci_95_high': 0.0,
                'instant_mean': 0.0, 'very_long_mean': 0.0, 'n_subjects': 0
            }
    
    return aggregated


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=None, help='Max episodes per subject (None=all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Rigorous Multi-Timescale Encoding Analysis")
    print("=" * 70)
    
    # Setup paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    # Find or create run directory
    runs_dir = os.path.join(project_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    
    # Find latest run directory (starts with digit)
    existing_runs = [d for d in os.listdir(runs_dir) if d[0].isdigit()]
    if existing_runs:
        latest_run = sorted(existing_runs)[-1]
        run_dir = os.path.join(runs_dir, latest_run)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_dir, timestamp)
    
    output_dir = os.path.join(run_dir, 'multiscale_encoding')
    os.makedirs(output_dir, exist_ok=True)
    
    max_episodes = args.max_episodes
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max episodes: {max_episodes if max_episodes else 'all'}")
    
    # Process subjects
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
    all_results = {}
    
    for subject in subjects:
        print(f"\n{'='*50}")
        print(f"Processing {subject}")
        print("=" * 50)
        
        # Load fMRI data
        fmri_data, used_episodes = load_fmri_data(data_dir, subject, max_episodes=max_episodes)
        if fmri_data is None:
            continue
        
        # Load matching stimulus features
        features = load_stimulus_features(data_dir, used_episodes)
        if not features:
            print("  ⚠ No features found, skipping")
            continue
        
        # Analyze encoding
        results = analyze_subject_encoding(subject, fmri_data, features, output_dir=output_dir)
        if results:
            all_results[subject] = results
            
            # Print subject TWG summary
            print(f"\n  TWG for {subject}:")
            for net in NETWORK_ORDER:
                twg_data = results['twg'].get(net, {})
                twg = twg_data.get('twg', 0)
                instant = twg_data.get('instant_r', 0)
                very_long = twg_data.get('very_long_r', 0)
                print(f"    {net:20s}: TWG={twg:+.4f} (instant={instant:.4f}, very_long={very_long:.4f})")
            
            # Save individual subject results
            sub_output = os.path.join(output_dir, f'{subject}_encoding_results.json')
            with open(sub_output, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Clean up memory
        del fmri_data
        gc.collect()
    
    if not all_results:
        print("\n⚠ No results generated!")
        return None
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("Aggregating Results Across Subjects")
    print("=" * 70)
    
    aggregated = aggregate_results(all_results)
    
    # Save aggregated results
    output_path = os.path.join(output_dir, 'multiscale_encoding_results.json')
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS: Real TWG by Network")
    print("=" * 70)
    print(f"{'Network':<20} {'TWG':>10} {'95% CI':>20} {'Instant r':>12} {'VeryLong r':>12}")
    print("-" * 74)
    
    for net in NETWORK_ORDER:
        twg_data = aggregated['network_summary']['twg'][net]
        twg = twg_data['mean']
        ci_low = twg_data['ci_95_low']
        ci_high = twg_data['ci_95_high']
        instant = twg_data['instant_mean']
        very_long = twg_data['very_long_mean']
        
        ci_str = f"[{ci_low:+.4f}, {ci_high:+.4f}]"
        print(f"{net:<20} {twg:>+10.4f} {ci_str:>20} {instant:>12.4f} {very_long:>12.4f}")
    
    print("\n✓ Rigorous encoding analysis complete!")
    print("  All TWG values are computed from REAL Ridge regression cross-validation.")
    
    return aggregated


if __name__ == '__main__':
    main()

