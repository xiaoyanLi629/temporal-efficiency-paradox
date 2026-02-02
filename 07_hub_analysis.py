"""
Step 7: Hub Centrality Analysis
网络中心性分析 - 揭示DMN为何需要慢速处理的结构原因

This script analyzes network topology to explain WHY DMN needs slow processing:
1. Betweenness Centrality - DMN regions as information integration hubs
2. Degree Centrality - Number of connections
3. Participation Coefficient - Cross-network integration
4. Module-based analysis

Key Hypothesis: DMN regions have high betweenness centrality,
meaning they integrate information from multiple sources,
which structurally requires more time.
"""

import os
import sys
import numpy as np
import h5py
from scipy.stats import zscore, pearsonr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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


class HubCentralityAnalyzer:
    """Analyze network hub structure and centrality metrics."""
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
        self.num_regions = 1000
        
        # Correlation threshold for binarizing connectivity
        self.threshold_percentile = 90
        
        # Create network labels for each region
        self.region_network_labels = np.zeros(self.num_regions, dtype=int)
        for i, network in enumerate(NETWORK_ORDER):
            for region in SCHAEFER_NETWORK_INDICES[network]:
                self.region_network_labels[region] = i
    
    def load_fmri_data(self, subject):
        """Load fMRI time series data."""
        fmri_path = os.path.join(
            self.data_dir, 'fmri', subject, 'func',
            f'{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
        )
        
        if not os.path.exists(fmri_path):
            return None
            
        with h5py.File(fmri_path, 'r') as f:
            keys = list(f.keys())[:4]  # Use first 4 episodes
            data_list = [f[key][:] for key in sorted(keys)]
            fmri_data = np.concatenate(data_list, axis=0)
            
        return fmri_data
    
    def compute_connectivity_matrix(self, fmri_data):
        """Compute functional connectivity matrix."""
        # Normalize
        fmri_norm = zscore(fmri_data, axis=0)
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(fmri_norm.T)
        
        # Remove diagonal
        np.fill_diagonal(corr_matrix, 0)
        
        return corr_matrix
    
    def binarize_matrix(self, corr_matrix):
        """Binarize connectivity matrix using percentile threshold."""
        threshold = np.percentile(np.abs(corr_matrix), self.threshold_percentile)
        binary_matrix = (np.abs(corr_matrix) > threshold).astype(int)
        return binary_matrix
    
    def compute_degree_centrality(self, binary_matrix):
        """Compute degree centrality for each node."""
        degrees = np.sum(binary_matrix, axis=1)
        # Normalize by maximum possible degree
        degree_centrality = degrees / (self.num_regions - 1)
        return degree_centrality
    
    def compute_betweenness_centrality(self, binary_matrix):
        """
        Compute betweenness centrality - key metric for hub detection.
        High betweenness = node lies on many shortest paths = information hub.
        """
        n = self.num_regions
        
        # Convert to sparse matrix for efficiency
        # Use inverted weights (1 where connected, inf where not)
        dist_matrix = np.where(binary_matrix > 0, 1, np.inf)
        np.fill_diagonal(dist_matrix, 0)
        
        # Compute all-pairs shortest paths
        # Note: This is computationally intensive, so we'll use an approximation
        # by sampling nodes
        
        sample_size = min(200, n)
        sample_nodes = np.random.choice(n, sample_size, replace=False)
        
        betweenness = np.zeros(n)
        
        # Approximate betweenness using sampled source nodes
        for source in sample_nodes:
            # BFS from source
            dist = np.full(n, np.inf)
            dist[source] = 0
            sigma = np.zeros(n)  # Number of shortest paths
            sigma[source] = 1
            predecessors = [[] for _ in range(n)]
            
            # BFS queue
            queue = [source]
            order = []
            
            while queue:
                v = queue.pop(0)
                order.append(v)
                
                for w in np.where(binary_matrix[v] > 0)[0]:
                    # First time seeing w
                    if dist[w] == np.inf:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    # Shortest path to w via v
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)
            
            # Backward pass - accumulate dependencies
            delta = np.zeros(n)
            while order:
                w = order.pop()
                for v in predecessors[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != source:
                    betweenness[w] += delta[w]
        
        # Normalize
        betweenness = betweenness / (sample_size * (n - 1))
        
        return betweenness
    
    def compute_participation_coefficient(self, binary_matrix):
        """
        Compute participation coefficient - measures cross-network integration.
        High PC = node connects to many different networks = integration hub.
        """
        n = self.num_regions
        participation = np.zeros(n)
        
        for i in range(n):
            # Total degree
            k_i = np.sum(binary_matrix[i])
            if k_i == 0:
                continue
            
            # Compute connections to each network
            network_connections = np.zeros(len(NETWORK_ORDER))
            for j, network in enumerate(NETWORK_ORDER):
                network_regions = SCHAEFER_NETWORK_INDICES[network]
                network_connections[j] = np.sum(binary_matrix[i, network_regions])
            
            # Participation coefficient
            participation[i] = 1 - np.sum((network_connections / k_i) ** 2)
        
        return participation
    
    def compute_within_module_degree(self, binary_matrix):
        """Compute within-module degree z-score."""
        n = self.num_regions
        within_module_z = np.zeros(n)
        
        for net_idx, network in enumerate(NETWORK_ORDER):
            network_regions = SCHAEFER_NETWORK_INDICES[network]
            
            # Within-network connectivity submatrix
            submatrix = binary_matrix[np.ix_(network_regions, network_regions)]
            within_degrees = np.sum(submatrix, axis=1)
            
            # Z-score within network
            mean_deg = np.mean(within_degrees)
            std_deg = np.std(within_degrees) + 1e-10
            
            for i, region in enumerate(network_regions):
                within_module_z[region] = (within_degrees[i] - mean_deg) / std_deg
        
        return within_module_z
    
    def identify_hub_types(self, participation, within_module_z):
        """
        Identify hub types based on Guimera & Amaral classification.
        - Provincial hubs: High within-module degree, low participation
        - Connector hubs: High within-module degree, high participation
        """
        pc_threshold = 0.3  # Participation coefficient threshold
        wmd_threshold = 1.0  # Within-module degree z-score threshold
        
        hub_types = np.zeros(self.num_regions, dtype=int)
        # 0 = non-hub, 1 = provincial hub, 2 = connector hub
        
        for i in range(self.num_regions):
            if within_module_z[i] > wmd_threshold:
                if participation[i] > pc_threshold:
                    hub_types[i] = 2  # Connector hub
                else:
                    hub_types[i] = 1  # Provincial hub
        
        return hub_types
    
    def analyze_subject(self, subject):
        """Analyze network topology for a single subject."""
        print(f"\nAnalyzing {subject}...")
        
        # Load data
        fmri = self.load_fmri_data(subject)
        if fmri is None:
            print(f"  ⚠ Could not load data for {subject}")
            return None
        
        print(f"  ✓ Loaded fMRI: {fmri.shape}")
        
        # Compute connectivity
        print("  Computing connectivity matrix...")
        corr_matrix = self.compute_connectivity_matrix(fmri)
        binary_matrix = self.binarize_matrix(corr_matrix)
        
        # Compute centrality metrics
        print("  Computing degree centrality...")
        degree_centrality = self.compute_degree_centrality(binary_matrix)
        
        print("  Computing betweenness centrality (this may take a while)...")
        betweenness_centrality = self.compute_betweenness_centrality(binary_matrix)
        
        print("  Computing participation coefficient...")
        participation = self.compute_participation_coefficient(binary_matrix)
        
        print("  Computing within-module degree...")
        within_module_z = self.compute_within_module_degree(binary_matrix)
        
        print("  Identifying hub types...")
        hub_types = self.identify_hub_types(participation, within_module_z)
        
        return {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'participation_coefficient': participation,
            'within_module_z': within_module_z,
            'hub_types': hub_types,
            'corr_matrix': corr_matrix
        }
    
    def aggregate_by_network(self, metrics):
        """Aggregate metrics by functional network."""
        network_metrics = {}
        
        for network in NETWORK_ORDER:
            regions = SCHAEFER_NETWORK_INDICES[network]
            
            network_metrics[network] = {
                'degree_centrality': {
                    'mean': float(np.mean(metrics['degree_centrality'][regions])),
                    'std': float(np.std(metrics['degree_centrality'][regions]))
                },
                'betweenness_centrality': {
                    'mean': float(np.mean(metrics['betweenness_centrality'][regions])),
                    'std': float(np.std(metrics['betweenness_centrality'][regions]))
                },
                'participation_coefficient': {
                    'mean': float(np.mean(metrics['participation_coefficient'][regions])),
                    'std': float(np.std(metrics['participation_coefficient'][regions]))
                },
                'within_module_z': {
                    'mean': float(np.mean(metrics['within_module_z'][regions])),
                    'std': float(np.std(metrics['within_module_z'][regions]))
                },
                'n_provincial_hubs': int(np.sum(metrics['hub_types'][regions] == 1)),
                'n_connector_hubs': int(np.sum(metrics['hub_types'][regions] == 2)),
                'hub_ratio': float(np.mean(metrics['hub_types'][regions] > 0))
            }
        
        return network_metrics
    
    def run_analysis(self):
        """Run full hub centrality analysis."""
        print("=" * 60)
        print("Hub Centrality Analysis")
        print("=" * 60)
        
        all_results = {}
        
        for subject in self.subjects:
            metrics = self.analyze_subject(subject)
            if metrics is not None:
                network_metrics = self.aggregate_by_network(metrics)
                all_results[subject] = {
                    'network_metrics': network_metrics,
                    'raw_metrics': {
                        'degree_centrality': metrics['degree_centrality'].tolist(),
                        'betweenness_centrality': metrics['betweenness_centrality'].tolist(),
                        'participation_coefficient': metrics['participation_coefficient'].tolist(),
                        'hub_types': metrics['hub_types'].tolist()
                    }
                }
        
        # Aggregate across subjects
        aggregated = {}
        for network in NETWORK_ORDER:
            metrics_list = [all_results[s]['network_metrics'][network] 
                          for s in all_results if network in all_results[s].get('network_metrics', {})]
            
            if metrics_list:
                aggregated[network] = {
                    'degree_centrality': np.mean([m['degree_centrality']['mean'] for m in metrics_list]),
                    'betweenness_centrality': np.mean([m['betweenness_centrality']['mean'] for m in metrics_list]),
                    'participation_coefficient': np.mean([m['participation_coefficient']['mean'] for m in metrics_list]),
                    'connector_hub_ratio': np.mean([m['n_connector_hubs'] / len(SCHAEFER_NETWORK_INDICES[network]) for m in metrics_list]),
                    'hub_ratio': np.mean([m['hub_ratio'] for m in metrics_list])
                }
        
        # Save results
        results_data = {
            'subject_results': {s: v['network_metrics'] for s, v in all_results.items()},
            'aggregated': aggregated
        }
        
        results_path = os.path.join(self.output_dir, 'hub_centrality_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY: Network Hub Characteristics")
        print("=" * 60)
        print(f"{'Network':<18} {'Betweenness':<12} {'Participation':<14} {'Connector Hubs':<15}")
        print("-" * 60)
        
        for network in NETWORK_ORDER:
            if network in aggregated:
                r = aggregated[network]
                print(f"{network:<18} {r['betweenness_centrality']:.4f}       {r['participation_coefficient']:.4f}         {r['connector_hub_ratio']*100:.1f}%")
        
        print("\n" + "=" * 60)
        print("KEY FINDING:")
        print("-" * 60)
        
        # Identify which network has highest betweenness
        max_between_network = max(aggregated.keys(), key=lambda x: aggregated[x]['betweenness_centrality'])
        max_pc_network = max(aggregated.keys(), key=lambda x: aggregated[x]['participation_coefficient'])
        
        print(f"Highest Betweenness Centrality: {max_between_network}")
        print(f"  → Acts as information integration hub")
        print(f"  → Requires more time to integrate inputs from multiple sources")
        print(f"\nHighest Participation Coefficient: {max_pc_network}")
        print(f"  → Most cross-network connections")
        print(f"  → Structural basis for multimodal integration")
        
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
                args.output_dir = os.path.join(runs_dir, latest_run, 'hub_centrality')
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.output_dir = os.path.join(runs_dir, timestamp, 'hub_centrality')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = os.path.join(project_dir, 'runs', timestamp, 'hub_centrality')
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    analyzer = HubCentralityAnalyzer(args.data_dir, args.output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

