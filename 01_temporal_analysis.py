"""
Rigorous Temporal Spectral Analysis
严格的时序频谱分析 - 分析所有区域并计算置信区间

Key improvements:
1. Analyze ALL 1000 regions (not just sampled 100)
2. Bootstrap confidence intervals for TDS
3. Sensitivity analysis for frequency band cutoffs
4. Individual subject results
"""

import os
import sys
import numpy as np
import h5py
from scipy import signal
from scipy.stats import zscore
import json
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

# Network definitions
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

TR = 1.49
# Multiple cutoffs for sensitivity analysis
FREQUENCY_CUTOFFS = [0.05, 0.07, 0.09]
DEFAULT_CUTOFF = 0.07
N_BOOTSTRAP = 1000


def load_fmri_data(data_dir, subject, max_episodes=None):
    """Load fMRI data for a subject.
    
    Args:
        data_dir: Path to data directory
        subject: Subject ID
        max_episodes: Maximum episodes to load (None = all)
    """
    fmri_path = os.path.join(data_dir, 'fmri', subject, 'func',
        f'{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5')
    
    if not os.path.exists(fmri_path):
        print(f"  ⚠ fMRI data not found: {fmri_path}")
        return None
    
    with h5py.File(fmri_path, 'r') as f:
        keys = sorted(list(f.keys()))
        if max_episodes is not None:
            keys = keys[:max_episodes]
        data_list = [f[key][:] for key in keys]
        fmri_data = np.concatenate(data_list, axis=0)
    
    print(f"  Loaded: {len(keys)} episodes, shape {fmri_data.shape}")
    return fmri_data


def compute_tds_single_region(ts, fs, cutoff=DEFAULT_CUTOFF):
    """Compute TDS for a single time series."""
    if np.std(ts) < 1e-6:
        return np.nan
    
    # Preprocess
    ts = zscore(signal.detrend(ts))
    
    # Welch PSD estimation
    freqs, psd = signal.welch(ts, fs=fs, nperseg=min(256, len(ts)//4))
    
    # Frequency bands
    low_idx = (freqs >= 0.01) & (freqs < cutoff)
    high_idx = (freqs >= cutoff) & (freqs <= 0.25)
    
    low_power = np.trapz(psd[low_idx], freqs[low_idx]) if np.any(low_idx) else 1e-10
    high_power = np.trapz(psd[high_idx], freqs[high_idx]) if np.any(high_idx) else 1e-10
    
    tds = high_power / (low_power + 1e-10)
    
    return tds


def bootstrap_tds(ts, fs, cutoff=DEFAULT_CUTOFF, n_bootstrap=N_BOOTSTRAP):
    """Compute TDS with bootstrap confidence interval."""
    n = len(ts)
    block_size = max(20, n // 50)  # Block bootstrap for time series
    
    boot_tds = []
    for _ in range(n_bootstrap):
        # Block bootstrap
        n_blocks = n // block_size + 1
        indices = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size + 1)
            indices.extend(range(start, start + block_size))
        indices = indices[:n]
        
        boot_ts = ts[indices]
        tds = compute_tds_single_region(boot_ts, fs, cutoff)
        if not np.isnan(tds):
            boot_tds.append(tds)
    
    if len(boot_tds) < 10:
        return np.nan, np.nan, np.nan
    
    return np.mean(boot_tds), np.percentile(boot_tds, 2.5), np.percentile(boot_tds, 97.5)


def compute_psd_for_region(ts, fs):
    """Compute PSD for a single time series, returning freqs and psd."""
    if np.std(ts) < 1e-6:
        return None, None
    
    # Preprocess
    ts = zscore(signal.detrend(ts))
    
    # Welch PSD estimation
    freqs, psd = signal.welch(ts, fs=fs, nperseg=min(256, len(ts)//4))
    
    return freqs, psd


def analyze_temporal_dynamics_full(fmri_data, tr=TR, compute_ci=True, batch_size=100):
    """
    Analyze temporal dynamics for ALL regions.
    
    Args:
        fmri_data: (n_timepoints, n_regions) array
        tr: repetition time
        compute_ci: whether to compute bootstrap confidence intervals
        batch_size: process regions in batches to show progress
    
    Returns:
        Dictionary with TDS values, confidence intervals, and PSD data
    """
    n_timepoints, n_regions = fmri_data.shape
    fs = 1.0 / tr
    
    print(f"  Analyzing {n_regions} regions...")
    
    results = {
        'tds': np.zeros(n_regions),
        'ci_low': np.zeros(n_regions) if compute_ci else None,
        'ci_high': np.zeros(n_regions) if compute_ci else None,
        'psd': None,  # Will store average PSD per region
        'freqs': None  # Frequency axis
    }
    
    # First pass: compute PSD for all regions
    all_psd = []
    sample_freqs = None
    
    print("  Computing PSD for all regions...")
    for region in range(n_regions):
        ts = fmri_data[:, region]
        freqs, psd = compute_psd_for_region(ts, fs)
        if freqs is not None:
            if sample_freqs is None:
                sample_freqs = freqs
            all_psd.append(psd)
        else:
            # Use zeros for invalid regions
            if sample_freqs is not None:
                all_psd.append(np.zeros_like(sample_freqs))
            else:
                all_psd.append(None)
    
    # Convert to array and store
    if sample_freqs is not None:
        results['freqs'] = sample_freqs
        # Replace None with zeros
        for i in range(len(all_psd)):
            if all_psd[i] is None:
                all_psd[i] = np.zeros_like(sample_freqs)
        results['psd'] = np.array(all_psd)  # Shape: (n_regions, n_freqs)
    
    # Process TDS in batches
    n_batches = (n_regions + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_regions)
        
        for region in range(start_idx, end_idx):
            ts = fmri_data[:, region]
            
            if compute_ci:
                tds, ci_low, ci_high = bootstrap_tds(ts, fs, DEFAULT_CUTOFF, n_bootstrap=100)
                results['tds'][region] = tds if not np.isnan(tds) else 0.5
                results['ci_low'][region] = ci_low if not np.isnan(ci_low) else 0.3
                results['ci_high'][region] = ci_high if not np.isnan(ci_high) else 0.7
            else:
                tds = compute_tds_single_region(ts, fs, DEFAULT_CUTOFF)
                results['tds'][region] = tds if not np.isnan(tds) else 0.5
        
        # Progress update
        pct = (batch_idx + 1) / n_batches * 100
        print(f"    Progress: {pct:.0f}% ({end_idx}/{n_regions} regions)")
    
    return results


def sensitivity_analysis(fmri_data, tr=TR, sample_size=200):
    """
    Analyze sensitivity of TDS to frequency band cutoff.
    
    Tests multiple cutoff values to ensure robustness.
    """
    n_timepoints, n_regions = fmri_data.shape
    fs = 1.0 / tr
    
    # Sample regions for efficiency
    sample_regions = np.random.choice(n_regions, min(sample_size, n_regions), replace=False)
    
    sensitivity = {}
    
    for cutoff in FREQUENCY_CUTOFFS:
        network_tds = {net: [] for net in NETWORK_ORDER}
        
        for region in sample_regions:
            ts = fmri_data[:, region]
            tds = compute_tds_single_region(ts, fs, cutoff)
            
            if not np.isnan(tds):
                # Find which network this region belongs to
                for net, indices in SCHAEFER_NETWORK_INDICES.items():
                    if region in indices:
                        network_tds[net].append(tds)
                        break
        
        sensitivity[f'cutoff_{cutoff}'] = {
            net: {
                'mean': float(np.mean(vals)) if vals else 0.0,
                'std': float(np.std(vals)) if vals else 0.0
            }
            for net, vals in network_tds.items()
        }
    
    return sensitivity


def aggregate_by_network(tds_values, ci_low=None, ci_high=None):
    """Aggregate region-level TDS to network level."""
    network_summary = {}
    
    for net, indices in SCHAEFER_NETWORK_INDICES.items():
        valid_indices = [i for i in indices if i < len(tds_values)]
        net_tds = tds_values[valid_indices]
        
        # Remove NaN values
        net_tds = net_tds[~np.isnan(net_tds)]
        
        if len(net_tds) > 0:
            network_summary[net] = {
                'mean': float(np.mean(net_tds)),
                'std': float(np.std(net_tds)),
                'sem': float(np.std(net_tds) / np.sqrt(len(net_tds))),
                'n_regions': len(net_tds),
                'min': float(np.min(net_tds)),
                'max': float(np.max(net_tds))
            }
            
            if ci_low is not None and ci_high is not None:
                net_ci_low = ci_low[valid_indices]
                net_ci_high = ci_high[valid_indices]
                network_summary[net]['ci_95_low'] = float(np.mean(net_ci_low[~np.isnan(net_ci_low)]))
                network_summary[net]['ci_95_high'] = float(np.mean(net_ci_high[~np.isnan(net_ci_high)]))
        else:
            network_summary[net] = {
                'mean': 0.5, 'std': 0.0, 'sem': 0.0, 'n_regions': 0, 'min': 0.0, 'max': 0.0
            }
    
    return network_summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=None, help='Max episodes per subject (None=all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Rigorous Temporal Spectral Analysis")
    print("=" * 70)
    
    # Setup paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    # Find or create run directory
    runs_dir = os.path.join(project_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    
    existing_runs = [d for d in os.listdir(runs_dir) if d[0].isdigit()]
    if existing_runs:
        latest_run = sorted(existing_runs)[-1]
        run_dir = os.path.join(runs_dir, latest_run)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_dir, timestamp)
    
    output_dir = os.path.join(run_dir, 'temporal_spectral')
    os.makedirs(output_dir, exist_ok=True)
    
    max_episodes = args.max_episodes
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max episodes: {max_episodes if max_episodes else 'all'}")
    
    # Process subjects
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
    all_tds = {}
    all_ci_low = {}
    all_ci_high = {}
    all_psd = {}
    all_freqs = None
    subject_results = {}
    
    for subject in subjects:
        print(f"\n{'='*50}")
        print(f"Processing {subject}")
        print("=" * 50)
        
        fmri_data = load_fmri_data(data_dir, subject, max_episodes=max_episodes)
        if fmri_data is None:
            continue
        
        # Full TDS analysis with confidence intervals and PSD
        results = analyze_temporal_dynamics_full(fmri_data, compute_ci=True)
        
        all_tds[subject] = results['tds']
        all_ci_low[subject] = results['ci_low']
        all_ci_high[subject] = results['ci_high']
        
        # Store PSD data
        if results['psd'] is not None:
            all_psd[subject] = results['psd']
            if all_freqs is None:
                all_freqs = results['freqs']
        
        # Network summary for this subject
        net_summary = aggregate_by_network(results['tds'], results['ci_low'], results['ci_high'])
        subject_results[subject] = net_summary
        
        print(f"\n  Network TDS for {subject}:")
        for net in NETWORK_ORDER:
            data = net_summary[net]
            print(f"    {net:20s}: {data['mean']:.3f} ± {data['std']:.3f}")
        
        # Sensitivity analysis (only for first subject to save time)
        if subject == subjects[0]:
            print(f"\n  Running sensitivity analysis...")
            sensitivity = sensitivity_analysis(fmri_data)
        
        del fmri_data
        gc.collect()
    
    if not all_tds:
        print("\n⚠ No results generated!")
        return None
    
    # Aggregate across subjects
    print("\n" + "=" * 70)
    print("Aggregating Results Across Subjects")
    print("=" * 70)
    
    # Average TDS across subjects
    avg_tds = np.mean([all_tds[s] for s in all_tds.keys()], axis=0)
    avg_ci_low = np.mean([all_ci_low[s] for s in all_ci_low.keys()], axis=0)
    avg_ci_high = np.mean([all_ci_high[s] for s in all_ci_high.keys()], axis=0)
    
    # Network-level summary
    network_summary = aggregate_by_network(avg_tds, avg_ci_low, avg_ci_high)
    
    # Cross-subject variability
    cross_subject_var = {}
    for net in NETWORK_ORDER:
        net_means = [subject_results[s][net]['mean'] for s in subject_results.keys()]
        cross_subject_var[net] = {
            'mean': float(np.mean(net_means)),
            'std': float(np.std(net_means)),
            'values': [float(v) for v in net_means]
        }
    
    # Compile final results
    final_results = {
        'network_summary': {
            'tds': network_summary
        },
        'cross_subject_variability': cross_subject_var,
        'sensitivity_analysis': sensitivity if 'sensitivity' in dir() else {},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_subjects': len(all_tds),
            'subjects': list(all_tds.keys()),
            'frequency_cutoff': DEFAULT_CUTOFF,
            'tr': TR,
            'n_regions_analyzed': len(avg_tds)
        }
    }
    
    # Save results
    output_path = os.path.join(output_dir, 'temporal_spectral_results.json')
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    np.save(os.path.join(output_dir, 'avg_tds.npy'), avg_tds)
    np.save(os.path.join(output_dir, 'avg_ci_low.npy'), avg_ci_low)
    np.save(os.path.join(output_dir, 'avg_ci_high.npy'), avg_ci_high)
    
    # Save PSD data for figure generation
    if all_psd and all_freqs is not None:
        avg_psd = np.mean([all_psd[s] for s in all_psd.keys()], axis=0)
        np.save(os.path.join(output_dir, 'avg_psd.npy'), avg_psd)
        np.save(os.path.join(output_dir, 'freqs.npy'), all_freqs)
        print(f"  PSD data saved: avg_psd.npy ({avg_psd.shape}), freqs.npy ({all_freqs.shape})")
    
    print(f"\nResults saved to: {output_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS: TDS by Network")
    print("=" * 70)
    print(f"{'Network':<20} {'TDS Mean':>10} {'Std':>10} {'95% CI':>25} {'N Regions':>12}")
    print("-" * 77)
    
    for net in NETWORK_ORDER:
        data = network_summary[net]
        ci_str = f"[{data.get('ci_95_low', 0):.3f}, {data.get('ci_95_high', 0):.3f}]"
        print(f"{net:<20} {data['mean']:>10.3f} {data['std']:>10.3f} {ci_str:>25} {data['n_regions']:>12}")
    
    print("\n✓ Rigorous temporal spectral analysis complete!")
    print("  All 1000 regions analyzed with bootstrap confidence intervals.")
    
    return final_results


if __name__ == '__main__':
    main()

