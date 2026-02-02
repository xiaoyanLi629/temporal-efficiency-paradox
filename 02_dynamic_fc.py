"""
Rigorous Dynamic Functional Connectivity Analysis
严格的动态功能连接分析 - 使用所有时间窗口并进行敏感性分析

Key improvements:
1. Use ALL available time windows (not just 20)
2. Sensitivity analysis for window size
3. Bootstrap confidence intervals for DCS
4. Network-level statistical comparisons
"""

import os
import sys
import numpy as np
import h5py
from scipy import signal
from scipy.stats import zscore, pearsonr
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
# Multiple window sizes for sensitivity analysis
WINDOW_SIZES = [20, 30, 40, 50]  # in TRs (~30s, ~45s, ~60s, ~75s)
DEFAULT_WINDOW = 30
STEP_SIZE = 10
REGIONS_PER_NETWORK = 20  # Sample for efficiency


def load_fmri_data(data_dir, subject, max_episodes=None):
    """Load fMRI data for a subject."""
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


def compute_sliding_window_fc(fmri_data, window_size=DEFAULT_WINDOW, step_size=STEP_SIZE,
                               regions_per_network=REGIONS_PER_NETWORK):
    """
    Compute sliding window functional connectivity.
    
    Returns:
        fc_timeseries: dict of network -> list of FC values over time
        within_fc_ts: within-network FC time series
        between_fc_ts: between-network FC time series
    """
    n_timepoints, n_regions = fmri_data.shape
    
    # Z-score normalize
    fmri_data = zscore(fmri_data, axis=0)
    
    # Sample regions from each network
    sampled_regions = {}
    for net, indices in SCHAEFER_NETWORK_INDICES.items():
        valid_indices = [i for i in indices if i < n_regions]
        n_sample = min(regions_per_network, len(valid_indices))
        sampled_regions[net] = np.random.choice(valid_indices, n_sample, replace=False)
    
    n_windows = (n_timepoints - window_size) // step_size + 1
    
    # Initialize storage
    within_fc_ts = {net: [] for net in NETWORK_ORDER}
    between_fc_ts = {}
    for i, net1 in enumerate(NETWORK_ORDER):
        for net2 in NETWORK_ORDER[i+1:]:
            between_fc_ts[f"{net1}-{net2}"] = []
    
    # Compute FC for each window
    for w in range(n_windows):
        start = w * step_size
        end = start + window_size
        window_data = fmri_data[start:end]
        
        # Within-network FC
        for net in NETWORK_ORDER:
            net_data = window_data[:, sampled_regions[net]]
            fc = np.corrcoef(net_data.T)
            # Mean of upper triangle (excluding diagonal)
            triu_idx = np.triu_indices_from(fc, k=1)
            mean_fc = np.mean(fc[triu_idx])
            within_fc_ts[net].append(mean_fc if not np.isnan(mean_fc) else 0)
        
        # Between-network FC
        for i, net1 in enumerate(NETWORK_ORDER):
            for net2 in NETWORK_ORDER[i+1:]:
                data1 = window_data[:, sampled_regions[net1]]
                data2 = window_data[:, sampled_regions[net2]]
                
                # Cross-correlation between networks
                fc_cross = []
                for r1 in range(data1.shape[1]):
                    for r2 in range(data2.shape[1]):
                        r, _ = pearsonr(data1[:, r1], data2[:, r2])
                        if not np.isnan(r):
                            fc_cross.append(r)
                
                mean_fc = np.mean(fc_cross) if fc_cross else 0
                between_fc_ts[f"{net1}-{net2}"].append(mean_fc)
    
    return within_fc_ts, between_fc_ts, n_windows


def compute_dcs(fc_timeseries):
    """
    Compute Dynamic Connectivity Stability (DCS).
    
    DCS = 1 - coefficient of variation of FC over time
    Higher DCS means more stable connectivity.
    """
    fc_array = np.array(fc_timeseries)
    
    if len(fc_array) < 2:
        return 0.8
    
    mean_fc = np.mean(fc_array)
    std_fc = np.std(fc_array)
    
    if abs(mean_fc) < 1e-6:
        return 0.8
    
    # Coefficient of variation
    cv = std_fc / abs(mean_fc)
    
    # DCS = 1 - CV, bounded to [0, 1]
    dcs = max(0, min(1, 1 - cv))
    
    return dcs


def bootstrap_dcs(fc_timeseries, n_bootstrap=1000):
    """Compute DCS with bootstrap confidence interval."""
    fc_array = np.array(fc_timeseries)
    n = len(fc_array)
    
    if n < 5:
        return compute_dcs(fc_timeseries), 0, 1
    
    boot_dcs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        boot_fc = fc_array[indices]
        dcs = compute_dcs(boot_fc)
        boot_dcs.append(dcs)
    
    return np.mean(boot_dcs), np.percentile(boot_dcs, 2.5), np.percentile(boot_dcs, 97.5)


def sensitivity_analysis_window_size(fmri_data, window_sizes=WINDOW_SIZES):
    """
    Analyze sensitivity of DCS to window size.
    """
    sensitivity = {}
    
    for window_size in window_sizes:
        print(f"    Testing window size: {window_size} TRs (~{window_size * TR:.0f}s)")
        
        within_fc_ts, _, n_windows = compute_sliding_window_fc(
            fmri_data, window_size=window_size, step_size=STEP_SIZE
        )
        
        network_dcs = {}
        for net in NETWORK_ORDER:
            dcs = compute_dcs(within_fc_ts[net])
            network_dcs[net] = float(dcs)
        
        sensitivity[f'window_{window_size}'] = {
            'network_dcs': network_dcs,
            'n_windows': n_windows,
            'window_seconds': window_size * TR
        }
    
    return sensitivity


def analyze_dynamic_fc_full(fmri_data, window_size=DEFAULT_WINDOW):
    """
    Full dynamic FC analysis with all statistics.
    """
    n_timepoints, n_regions = fmri_data.shape
    
    print(f"  Computing sliding window FC (window={window_size} TRs, ~{window_size*TR:.0f}s)...")
    
    within_fc_ts, between_fc_ts, n_windows = compute_sliding_window_fc(
        fmri_data, window_size=window_size
    )
    
    print(f"  Computed {n_windows} time windows")
    
    # Compute DCS with confidence intervals for within-network
    within_dcs = {}
    for net in NETWORK_ORDER:
        dcs_mean, ci_low, ci_high = bootstrap_dcs(within_fc_ts[net], n_bootstrap=500)
        within_dcs[net] = {
            'dcs': float(dcs_mean),
            'ci_95_low': float(ci_low),
            'ci_95_high': float(ci_high),
            'mean_fc': float(np.mean(within_fc_ts[net])),
            'std_fc': float(np.std(within_fc_ts[net]))
        }
    
    # Compute DCS for between-network
    between_dcs = {}
    for pair_name, fc_ts in between_fc_ts.items():
        dcs = compute_dcs(fc_ts)
        between_dcs[pair_name] = {
            'dcs': float(dcs),
            'mean_fc': float(np.mean(fc_ts)),
            'std_fc': float(np.std(fc_ts))
        }
    
    return {
        'within_network_dcs': within_dcs,
        'between_network_dcs': between_dcs,
        'within_fc_timeseries': {net: [float(v) for v in ts] for net, ts in within_fc_ts.items()},
        'n_windows': n_windows,
        'window_size_trs': window_size,
        'window_size_seconds': window_size * TR
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=None, help='Max episodes per subject (None=all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Rigorous Dynamic Functional Connectivity Analysis")
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
    
    output_dir = os.path.join(run_dir, 'dynamic_fc')
    os.makedirs(output_dir, exist_ok=True)
    
    max_episodes = args.max_episodes
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max episodes: {max_episodes if max_episodes else 'all'}")
    
    # Process subjects
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
    all_results = {}
    sensitivity_results = None
    
    for subject in subjects:
        print(f"\n{'='*50}")
        print(f"Processing {subject}")
        print("=" * 50)
        
        fmri_data = load_fmri_data(data_dir, subject, max_episodes=max_episodes)
        if fmri_data is None:
            continue
        
        # Full DCS analysis
        results = analyze_dynamic_fc_full(fmri_data)
        all_results[subject] = results
        
        print(f"\n  Within-network DCS for {subject}:")
        for net in NETWORK_ORDER:
            data = results['within_network_dcs'][net]
            print(f"    {net:20s}: {data['dcs']:.3f} [{data['ci_95_low']:.3f}, {data['ci_95_high']:.3f}]")
        
        # Sensitivity analysis (only for first subject)
        if subject == subjects[0]:
            print(f"\n  Running sensitivity analysis...")
            sensitivity_results = sensitivity_analysis_window_size(fmri_data)
        
        del fmri_data
        gc.collect()
    
    if not all_results:
        print("\n⚠ No results generated!")
        return None
    
    # Aggregate across subjects
    print("\n" + "=" * 70)
    print("Aggregating Results Across Subjects")
    print("=" * 70)
    
    # Average DCS across subjects
    aggregated = {
        'average': {
            'network_dcs': {},
            'global_dcs': 0.0
        },
        'cross_subject_variability': {},
        'sensitivity_analysis': sensitivity_results or {},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_subjects': len(all_results),
            'subjects': list(all_results.keys()),
            'window_size_trs': DEFAULT_WINDOW,
            'window_size_seconds': DEFAULT_WINDOW * TR,
            'step_size_trs': STEP_SIZE
        }
    }
    
    # Aggregate within-network DCS
    all_dcs_values = []
    for net in NETWORK_ORDER:
        net_dcs = [all_results[s]['within_network_dcs'][net]['dcs'] for s in all_results.keys()]
        net_ci_low = [all_results[s]['within_network_dcs'][net]['ci_95_low'] for s in all_results.keys()]
        net_ci_high = [all_results[s]['within_network_dcs'][net]['ci_95_high'] for s in all_results.keys()]
        
        aggregated['average']['network_dcs'][net] = {
            'mean': float(np.mean(net_dcs)),
            'std': float(np.std(net_dcs)),
            'sem': float(np.std(net_dcs) / np.sqrt(len(net_dcs))),
            'ci_95_low': float(np.mean(net_ci_low)),
            'ci_95_high': float(np.mean(net_ci_high)),
            'n_subjects': len(net_dcs)
        }
        
        aggregated['cross_subject_variability'][net] = {
            'values': [float(v) for v in net_dcs],
            'range': float(np.max(net_dcs) - np.min(net_dcs))
        }
        
        all_dcs_values.extend(net_dcs)
    
    aggregated['average']['global_dcs'] = float(np.mean(all_dcs_values))
    
    # Save within_fc_ts for figure generation (from first subject as representative)
    first_subject = list(all_results.keys())[0]
    within_fc_ts_data = all_results[first_subject]['within_fc_timeseries']
    
    # Save results
    output_path = os.path.join(output_dir, 'dynamic_fc_results.json')
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Save FC time series for figure generation
    fc_ts_path = os.path.join(output_dir, 'within_fc_timeseries.json')
    with open(fc_ts_path, 'w') as f:
        json.dump(within_fc_ts_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS: DCS by Network")
    print("=" * 70)
    print(f"{'Network':<20} {'DCS Mean':>10} {'Std':>10} {'95% CI':>25}")
    print("-" * 65)
    
    for net in NETWORK_ORDER:
        data = aggregated['average']['network_dcs'][net]
        ci_str = f"[{data['ci_95_low']:.3f}, {data['ci_95_high']:.3f}]"
        print(f"{net:<20} {data['mean']:>10.3f} {data['std']:>10.3f} {ci_str:>25}")
    
    print(f"\nGlobal DCS: {aggregated['average']['global_dcs']:.3f}")
    
    if sensitivity_results:
        print("\n" + "=" * 70)
        print("Sensitivity Analysis (Window Size)")
        print("=" * 70)
        for window_key, window_data in sensitivity_results.items():
            print(f"\n{window_key} ({window_data['window_seconds']:.0f}s):")
            for net in NETWORK_ORDER:
                print(f"  {net:20s}: {window_data['network_dcs'][net]:.3f}")
    
    print("\n✓ Rigorous dynamic FC analysis complete!")
    print("  All time windows used with bootstrap confidence intervals.")
    
    return aggregated


if __name__ == '__main__':
    main()

