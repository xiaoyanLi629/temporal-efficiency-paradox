"""
Individual Difference Analysis
个体差异分析 - ICC、个体变异、异常值检测

This module analyzes:
1. Intraclass Correlation Coefficient (ICC) for reliability
2. Individual-to-group correlations
3. Outlier detection
4. Individual variation visualization data
"""

import os
import numpy as np
from scipy import stats
from scipy.stats import zscore
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

NETWORK_ORDER = ['Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention', 
                 'Limbic', 'Frontoparietal', 'Default']


def compute_icc(data_matrix, icc_type='ICC(2,1)'):
    """
    Compute Intraclass Correlation Coefficient.
    
    Args:
        data_matrix: (n_subjects, n_items) array
        icc_type: Type of ICC to compute
    
    Returns:
        icc: ICC value
        ci_low, ci_high: 95% confidence interval
    
    ICC(2,1) - Two-way random effects, single measures
    Measures absolute agreement among raters (subjects)
    """
    n_subjects, n_items = data_matrix.shape
    
    if n_subjects < 2 or n_items < 2:
        return 0.0, 0.0, 1.0
    
    # Grand mean
    grand_mean = np.mean(data_matrix)
    
    # Row means (subject means)
    row_means = np.mean(data_matrix, axis=1)
    
    # Column means (item means)
    col_means = np.mean(data_matrix, axis=0)
    
    # Sum of squares
    ss_total = np.sum((data_matrix - grand_mean) ** 2)
    ss_rows = n_items * np.sum((row_means - grand_mean) ** 2)  # Between subjects
    ss_cols = n_subjects * np.sum((col_means - grand_mean) ** 2)  # Between items
    ss_error = ss_total - ss_rows - ss_cols
    
    # Mean squares
    df_rows = n_subjects - 1
    df_cols = n_items - 1
    df_error = df_rows * df_cols
    
    ms_rows = ss_rows / df_rows if df_rows > 0 else 0
    ms_cols = ss_cols / df_cols if df_cols > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0
    
    # ICC(2,1) - Two-way random, single measures, absolute agreement
    numerator = ms_rows - ms_error
    denominator = ms_rows + (n_items - 1) * ms_error + n_items * (ms_cols - ms_error) / n_subjects
    
    if denominator < 1e-10:
        return 0.0, 0.0, 1.0
    
    icc = numerator / denominator
    
    # Confidence interval (approximate)
    f_value = ms_rows / ms_error if ms_error > 0 else 1
    
    # F distribution bounds
    try:
        f_low = f_value / stats.f.ppf(0.975, df_rows, df_error)
        f_high = f_value / stats.f.ppf(0.025, df_rows, df_error)
        
        ci_low = (f_low - 1) / (f_low + n_items - 1)
        ci_high = (f_high - 1) / (f_high + n_items - 1)
    except:
        ci_low, ci_high = 0.0, 1.0
    
    return max(0, min(1, icc)), max(0, ci_low), min(1, ci_high)


def icc_interpretation(icc):
    """Interpret ICC value according to Koo & Li (2016)."""
    if icc < 0.5:
        return 'poor'
    elif icc < 0.75:
        return 'moderate'
    elif icc < 0.9:
        return 'good'
    else:
        return 'excellent'


def individual_to_group_correlation(subject_pattern, group_mean_pattern):
    """
    Compute correlation between individual pattern and group mean.
    
    Higher correlation indicates the subject follows the group pattern.
    """
    r, p = stats.pearsonr(subject_pattern, group_mean_pattern)
    return r, p


def detect_outliers(values, threshold=2.5):
    """
    Detect outliers using z-score method.
    
    Args:
        values: Array of values
        threshold: Z-score threshold for outlier detection
    
    Returns:
        outlier_mask: Boolean array
        z_scores: Z-scores for each value
    """
    values = np.array(values)
    z_scores = zscore(values)
    outlier_mask = np.abs(z_scores) > threshold
    
    return outlier_mask, z_scores


def analyze_individual_differences(run_dir):
    """
    Analyze individual differences across subjects.
    
    Args:
        run_dir: Path to run directory containing analysis results
    """
    print("=" * 70)
    print("Individual Difference Analysis")
    print("=" * 70)
    
    results = {
        'icc_analysis': {},
        'individual_correlations': {},
        'outlier_detection': {},
        'variation_summary': {},
        'metadata': {
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Load TDS results
    tds_path = os.path.join(run_dir, 'temporal_spectral', 'temporal_spectral_results.json')
    if os.path.exists(tds_path):
        print("\n  Analyzing TDS individual differences...")
        with open(tds_path, 'r') as f:
            tds_data = json.load(f)
        
        if 'cross_subject_variability' in tds_data:
            # Build subject x network matrix
            subjects = []
            for net in NETWORK_ORDER:
                if net in tds_data['cross_subject_variability']:
                    values = tds_data['cross_subject_variability'][net]['values']
                    if not subjects:
                        subjects = list(range(len(values)))
            
            if subjects:
                # Create data matrix (subjects x networks)
                tds_matrix = np.zeros((len(subjects), len(NETWORK_ORDER)))
                for j, net in enumerate(NETWORK_ORDER):
                    if net in tds_data['cross_subject_variability']:
                        values = tds_data['cross_subject_variability'][net]['values']
                        tds_matrix[:len(values), j] = values
                
                # Compute ICC
                icc, ci_low, ci_high = compute_icc(tds_matrix)
                results['icc_analysis']['tds'] = {
                    'icc': float(icc),
                    'ci_95_low': float(ci_low),
                    'ci_95_high': float(ci_high),
                    'interpretation': icc_interpretation(icc),
                    'n_subjects': len(subjects),
                    'n_networks': len(NETWORK_ORDER)
                }
                print(f"    TDS ICC: {icc:.3f} ({icc_interpretation(icc)})")
                
                # Individual-to-group correlations
                group_mean = np.mean(tds_matrix, axis=0)
                ind_corrs = []
                for i in range(len(subjects)):
                    r, p = individual_to_group_correlation(tds_matrix[i], group_mean)
                    ind_corrs.append({'subject': i, 'r': float(r), 'p': float(p)})
                
                results['individual_correlations']['tds'] = ind_corrs
                
                # Outlier detection per network
                outliers = {}
                for j, net in enumerate(NETWORK_ORDER):
                    values = tds_matrix[:, j]
                    mask, z_scores = detect_outliers(values)
                    outliers[net] = {
                        'outlier_subjects': [int(i) for i in np.where(mask)[0]],
                        'z_scores': [float(z) for z in z_scores]
                    }
                
                results['outlier_detection']['tds'] = outliers
    
    # Load DCS results
    dcs_path = os.path.join(run_dir, 'dynamic_fc', 'dynamic_fc_results.json')
    if os.path.exists(dcs_path):
        print("\n  Analyzing DCS individual differences...")
        with open(dcs_path, 'r') as f:
            dcs_data = json.load(f)
        
        if 'cross_subject_variability' in dcs_data:
            subjects = []
            for net in NETWORK_ORDER:
                if net in dcs_data['cross_subject_variability']:
                    values = dcs_data['cross_subject_variability'][net]['values']
                    if not subjects:
                        subjects = list(range(len(values)))
            
            if subjects:
                dcs_matrix = np.zeros((len(subjects), len(NETWORK_ORDER)))
                for j, net in enumerate(NETWORK_ORDER):
                    if net in dcs_data['cross_subject_variability']:
                        values = dcs_data['cross_subject_variability'][net]['values']
                        dcs_matrix[:len(values), j] = values
                
                # Compute ICC
                icc, ci_low, ci_high = compute_icc(dcs_matrix)
                results['icc_analysis']['dcs'] = {
                    'icc': float(icc),
                    'ci_95_low': float(ci_low),
                    'ci_95_high': float(ci_high),
                    'interpretation': icc_interpretation(icc),
                    'n_subjects': len(subjects),
                    'n_networks': len(NETWORK_ORDER)
                }
                print(f"    DCS ICC: {icc:.3f} ({icc_interpretation(icc)})")
                
                # Individual-to-group correlations
                group_mean = np.mean(dcs_matrix, axis=0)
                ind_corrs = []
                for i in range(len(subjects)):
                    r, p = individual_to_group_correlation(dcs_matrix[i], group_mean)
                    ind_corrs.append({'subject': i, 'r': float(r), 'p': float(p)})
                
                results['individual_correlations']['dcs'] = ind_corrs
    
    # Variation summary
    print("\n  Computing variation summary...")
    
    for metric in ['tds', 'dcs']:
        if metric in results['icc_analysis']:
            icc_data = results['icc_analysis'][metric]
            
            # Compute coefficient of variation across subjects
            results['variation_summary'][metric] = {
                'icc': icc_data['icc'],
                'reliability': icc_data['interpretation'],
                'recommendation': (
                    'Results are reliable across subjects' if icc_data['icc'] >= 0.75
                    else 'Results show moderate reliability; interpret with caution'
                    if icc_data['icc'] >= 0.5
                    else 'Results show poor reliability; individual differences are large'
                )
            }
    
    # Save results
    output_dir = os.path.join(run_dir, 'individual_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'individual_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Individual Difference Summary")
    print("=" * 70)
    
    for metric, summary in results['variation_summary'].items():
        print(f"\n  {metric.upper()}:")
        print(f"    ICC: {summary['icc']:.3f} ({summary['reliability']})")
        print(f"    {summary['recommendation']}")
    
    if results['individual_correlations']:
        print("\n  Individual-to-Group Correlations:")
        for metric, corrs in results['individual_correlations'].items():
            mean_r = np.mean([c['r'] for c in corrs])
            print(f"    {metric.upper()}: Mean r = {mean_r:.3f}")
    
    return results


def main():
    """Run individual difference analysis on latest results."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(project_dir, 'runs')
    
    # Find latest run
    existing_runs = [d for d in os.listdir(runs_dir) if d[0].isdigit()]
    if not existing_runs:
        print("No analysis results found!")
        return None
    
    latest_run = sorted(existing_runs)[-1]
    run_dir = os.path.join(runs_dir, latest_run)
    
    print(f"Analyzing individual differences in: {run_dir}")
    
    return analyze_individual_differences(run_dir)


if __name__ == '__main__':
    main()

