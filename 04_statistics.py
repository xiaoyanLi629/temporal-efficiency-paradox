"""
Statistical Testing Module
统计检验模块 - 置换检验、效应量、多重比较校正

This module provides rigorous statistical testing for:
1. Network-level comparisons (permutation tests)
2. Effect sizes (Cohen's d)
3. Multiple comparison correction (FDR)
4. Bootstrap confidence intervals
"""

import os
import numpy as np
from scipy import stats
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

NETWORK_ORDER = ['Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention', 
                 'Limbic', 'Frontoparietal', 'Default']


def permutation_test(group1, group2, n_permutations=10000, alternative='two-sided'):
    """
    Perform permutation test for difference between two groups.
    
    Args:
        group1, group2: Arrays of values
        n_permutations: Number of permutations
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        observed_diff: Observed difference in means
        p_value: Permutation p-value
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        perm_diffs.append(perm_diff)
    
    perm_diffs = np.array(perm_diffs)
    
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    elif alternative == 'greater':
        p_value = np.mean(perm_diffs >= observed_diff)
    else:  # less
        p_value = np.mean(perm_diffs <= observed_diff)
    
    return observed_diff, p_value


def cohens_d(group1, group2):
    """
    Compute Cohen's d effect size.
    
    Cohen's d = (mean1 - mean2) / pooled_std
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def effect_size_interpretation(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


def fdr_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.
    
    Returns:
        p_corrected: FDR-corrected p-values
        significant: Boolean array of significant tests
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and get original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # Compute FDR threshold for each rank
    thresholds = alpha * np.arange(1, n + 1) / n
    
    # Find largest k where p[k] <= threshold[k]
    significant_sorted = sorted_p <= thresholds
    
    # Corrected p-values
    p_corrected = np.zeros(n)
    for i in range(n):
        p_corrected[sorted_idx[i]] = min(1.0, sorted_p[i] * n / (i + 1))
    
    # Ensure monotonicity
    p_corrected_sorted = p_corrected[sorted_idx]
    for i in range(n - 2, -1, -1):
        p_corrected_sorted[i] = min(p_corrected_sorted[i], p_corrected_sorted[i + 1])
    
    # Restore original order
    p_corrected = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        p_corrected[idx] = p_corrected_sorted[i]
    
    significant = p_corrected < alpha
    
    return p_corrected, significant


def bootstrap_ci(values, statistic=np.mean, n_bootstrap=10000, ci=95):
    """
    Compute bootstrap confidence interval.
    
    Args:
        values: Array of values
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default: 95%)
    
    Returns:
        ci_low, ci_high: Confidence interval bounds
    """
    values = np.array(values)
    n = len(values)
    
    if n < 2:
        return statistic(values), statistic(values)
    
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, n, replace=True)
        boot_stats.append(statistic(sample))
    
    ci_low = np.percentile(boot_stats, (100 - ci) / 2)
    ci_high = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    
    return ci_low, ci_high


def pairwise_network_comparisons(network_data, metric_name='metric'):
    """
    Perform pairwise comparisons between all networks.
    
    Args:
        network_data: Dict of network -> list of values
    
    Returns:
        Dictionary with all pairwise comparisons
    """
    comparisons = {}
    p_values = []
    comparison_names = []
    
    networks = list(network_data.keys())
    
    for i, net1 in enumerate(networks):
        for net2 in networks[i+1:]:
            values1 = network_data[net1]
            values2 = network_data[net2]
            
            if len(values1) < 2 or len(values2) < 2:
                continue
            
            # Permutation test
            diff, p_value = permutation_test(values1, values2)
            
            # Effect size
            d = cohens_d(values1, values2)
            
            # T-test for comparison
            t_stat, t_p = stats.ttest_ind(values1, values2)
            
            comparison_name = f"{net1} vs {net2}"
            comparisons[comparison_name] = {
                'difference': float(diff),
                'p_value_permutation': float(p_value),
                'p_value_ttest': float(t_p),
                'cohens_d': float(d),
                'effect_size_interpretation': effect_size_interpretation(d),
                f'{net1}_mean': float(np.mean(values1)),
                f'{net2}_mean': float(np.mean(values2))
            }
            
            p_values.append(p_value)
            comparison_names.append(comparison_name)
    
    # FDR correction
    if p_values:
        p_corrected, significant = fdr_correction(p_values)
        
        for i, name in enumerate(comparison_names):
            comparisons[name]['p_value_fdr_corrected'] = float(p_corrected[i])
            comparisons[name]['significant_fdr'] = bool(significant[i])
    
    return comparisons


def run_statistical_tests(run_dir):
    """
    Run all statistical tests on analysis results.
    
    Args:
        run_dir: Path to run directory containing analysis results
    """
    print("=" * 70)
    print("Statistical Testing")
    print("=" * 70)
    
    results = {
        'tds_comparisons': {},
        'dcs_comparisons': {},
        'twg_comparisons': {},
        'key_hypothesis_tests': {},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_permutations': 10000,
            'fdr_alpha': 0.05
        }
    }
    
    # Load TDS results
    tds_path = os.path.join(run_dir, 'temporal_spectral', 'temporal_spectral_results.json')
    if os.path.exists(tds_path):
        print("\n  Testing TDS differences...")
        with open(tds_path, 'r') as f:
            tds_data = json.load(f)
        
        # Extract cross-subject values for each network
        if 'cross_subject_variability' in tds_data:
            tds_values = {net: tds_data['cross_subject_variability'][net]['values'] 
                         for net in NETWORK_ORDER if net in tds_data['cross_subject_variability']}
            
            if tds_values:
                results['tds_comparisons'] = pairwise_network_comparisons(tds_values, 'TDS')
                print(f"    Completed {len(results['tds_comparisons'])} pairwise comparisons")
    
    # Load DCS results
    dcs_path = os.path.join(run_dir, 'dynamic_fc', 'dynamic_fc_results.json')
    if os.path.exists(dcs_path):
        print("\n  Testing DCS differences...")
        with open(dcs_path, 'r') as f:
            dcs_data = json.load(f)
        
        if 'cross_subject_variability' in dcs_data:
            dcs_values = {net: dcs_data['cross_subject_variability'][net]['values']
                         for net in NETWORK_ORDER if net in dcs_data['cross_subject_variability']}
            
            if dcs_values:
                results['dcs_comparisons'] = pairwise_network_comparisons(dcs_values, 'DCS')
                print(f"    Completed {len(results['dcs_comparisons'])} pairwise comparisons")
    
    # Load TWG results
    twg_path = os.path.join(run_dir, 'multiscale_encoding', 'multiscale_encoding_results.json')
    if os.path.exists(twg_path):
        print("\n  Testing TWG differences...")
        with open(twg_path, 'r') as f:
            twg_data = json.load(f)
        
        # TWG values might be from individual subjects
        # For now, use summary statistics
        if 'network_summary' in twg_data and 'twg' in twg_data['network_summary']:
            twg_summary = twg_data['network_summary']['twg']
            
            # Key hypothesis test: DMN vs Visual TWG
            if 'Default' in twg_summary and 'Visual' in twg_summary:
                dmn_twg = twg_summary['Default'].get('mean', 0)
                vis_twg = twg_summary['Visual'].get('mean', 0)
                
                results['key_hypothesis_tests']['dmn_vs_visual_twg'] = {
                    'hypothesis': 'DMN shows greater TWG than Visual network',
                    'dmn_twg': float(dmn_twg),
                    'visual_twg': float(vis_twg),
                    'difference': float(dmn_twg - vis_twg),
                    'supports_hypothesis': dmn_twg > vis_twg
                }
    
    # Key hypothesis tests
    print("\n  Testing key hypotheses...")
    
    # H1: Temporal hierarchy exists (TDS varies systematically across networks)
    if results['tds_comparisons']:
        significant_tds = sum(1 for c in results['tds_comparisons'].values() 
                             if c.get('significant_fdr', False))
        results['key_hypothesis_tests']['h1_temporal_hierarchy'] = {
            'hypothesis': 'Networks show systematic differences in temporal dynamics (TDS)',
            'n_significant_comparisons': significant_tds,
            'total_comparisons': len(results['tds_comparisons']),
            'supports_hypothesis': significant_tds > len(results['tds_comparisons']) * 0.3
        }
    
    # H2: Connectivity stability varies (DCS varies across networks)
    if results['dcs_comparisons']:
        significant_dcs = sum(1 for c in results['dcs_comparisons'].values()
                             if c.get('significant_fdr', False))
        results['key_hypothesis_tests']['h2_connectivity_stability'] = {
            'hypothesis': 'Networks show systematic differences in connectivity stability (DCS)',
            'n_significant_comparisons': significant_dcs,
            'total_comparisons': len(results['dcs_comparisons']),
            'supports_hypothesis': significant_dcs > len(results['dcs_comparisons']) * 0.3
        }
    
    # Save results
    output_dir = os.path.join(run_dir, 'statistical_tests')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'statistical_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Key Hypothesis Test Results")
    print("=" * 70)
    
    for test_name, test_result in results['key_hypothesis_tests'].items():
        print(f"\n  {test_name}:")
        print(f"    Hypothesis: {test_result['hypothesis']}")
        print(f"    Supports hypothesis: {test_result.get('supports_hypothesis', 'N/A')}")
    
    return results


def main():
    """Run statistical tests on latest analysis results."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(project_dir, 'runs')
    
    # Find latest run
    existing_runs = [d for d in os.listdir(runs_dir) if d[0].isdigit()]
    if not existing_runs:
        print("No analysis results found!")
        return None
    
    latest_run = sorted(existing_runs)[-1]
    run_dir = os.path.join(runs_dir, latest_run)
    
    print(f"Running statistical tests on: {run_dir}")
    
    return run_statistical_tests(run_dir)


if __name__ == '__main__':
    main()

