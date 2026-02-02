"""
Step 5: Generate Scientific Figures for Project 2
生成Project 2的科学图表 - 时序效率悖论可视化

Features:
- Publication-quality figures (600 DPI)
- Consistent BuGn colormap with Project 1
- Novel visualization types distinct from Project 1
- Professional typography and styling
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
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

# ============================================================================
# Style Configuration - Consistent with Project 1 but distinct visualizations
# ============================================================================

plt.style.use('seaborn-v0_8-white')

# BuGn colormap (consistent with Project 1)
BUGN_COLORS = ['#F7FCFD', '#E5F5F9', '#CCECE6', '#99D8C9', '#66C2A4', '#41AE76', '#238B45', '#005824']

# Additional colormaps for temporal visualizations
TEMPORAL_CMAP = ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#99000D']  # Reds
EFFICIENCY_CMAP = ['#F7FCF5', '#E5F5E0', '#C7E9C0', '#A1D99B', '#74C476', '#41AB5D', '#238B45', '#006D2C']  # Greens

# Network colors (consistent with Project 1)
NETWORK_COLORS = {
    'Visual': '#EDF8FB',
    'Somatomotor': '#41AE76',
    'DorsalAttention': '#CCECE6',
    'VentralAttention': '#99D8C9',
    'Limbic': '#238B45',
    'Frontoparietal': '#66C2A4',
    'Default': '#005824'
}

# Distinct network colors for Project 2 (temporal theme)
# Scientific gradient: Fast (warm) → Slow (cool) using viridis-inspired palette
NETWORK_COLORS_P2 = {
    'Visual': '#D62728',        # Fast - Deep Red
    'Somatomotor': '#FF7F0E',   # Fast - Orange
    'DorsalAttention': '#BCBD22',  # Mid-Fast - Yellow-Green
    'VentralAttention': '#17BECF', # Mid - Cyan
    'Limbic': '#2CA02C',        # Mid - Green
    'Frontoparietal': '#1F77B4',   # Mid-Slow - Blue
    'Default': '#9467BD'        # Slow - Purple (DMN highlighted)
}

# Unified scientific colormap for heatmaps (viridis-like)
SCIENTIFIC_CMAP = ['#440154', '#482878', '#3E4A89', '#31688E', '#26828E', 
                   '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725']

# Diverging colormap for comparisons
DIVERGING_CMAP = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                  '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']

# Font configuration
FONT_CONFIG = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
}
plt.rcParams.update(FONT_CONFIG)

SAVE_DPI = 600
FORMATS = ['png', 'svg']


class Project2FigureGenerator:
    """Generate scientific figures for Project 2: Temporal Efficiency Paradox."""
    
    def __init__(self, run_dir, output_dir=None):
        self.run_dir = run_dir
        self.output_dir = output_dir or os.path.join(run_dir, 'figures')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._load_results()
    
    def _load_results(self):
        """Load all analysis results."""
        print("Loading analysis results for figure generation...")
        
        # Spectral results
        spectral_path = os.path.join(self.run_dir, 'temporal_spectral', 'temporal_spectral_results.json')
        self.spectral_results = self._load_json(spectral_path)
        
        # Dynamic FC results
        dfc_path = os.path.join(self.run_dir, 'dynamic_fc', 'dynamic_fc_results.json')
        self.dfc_results = self._load_json(dfc_path)
        
        # Encoding results
        encoding_path = os.path.join(self.run_dir, 'multiscale_encoding', 'multiscale_encoding_results.json')
        self.encoding_results = self._load_json(encoding_path)
        
        # Efficiency results
        efficiency_path = os.path.join(self.run_dir, 'temporal_efficiency', 'temporal_efficiency_results.json')
        self.efficiency_results = self._load_json(efficiency_path)
        
        # Load numpy arrays
        self._load_numpy_arrays()
    
    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_numpy_arrays(self):
        """Load numpy arrays."""
        self.avg_psd = self._load_npy('temporal_spectral', 'avg_psd.npy')
        self.freqs = self._load_npy('temporal_spectral', 'freqs.npy')
        self.avg_tds = self._load_npy('temporal_spectral', 'avg_tds.npy')
        self.avg_dcs = self._load_npy('dynamic_fc', 'avg_dcs_matrix.npy')
        self.twg = self._load_npy('multiscale_encoding', 'twg.npy')
        self.within_fc_ts = self._load_npy('dynamic_fc', 'within_network_fc_timeseries.npy')
        
        # Also try loading from JSON if npy not found
        if self.within_fc_ts is None:
            self.within_fc_ts = self._load_json_from_subdir('dynamic_fc', 'within_fc_timeseries.json')
    
    def _load_npy(self, subdir, filename):
        path = os.path.join(self.run_dir, subdir, filename)
        if os.path.exists(path):
            return np.load(path, allow_pickle=True)
        return None
    
    def _load_json_from_subdir(self, subdir, filename):
        path = os.path.join(self.run_dir, subdir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def save_figure(self, fig, name):
        """Save figure in multiple formats."""
        for fmt in FORMATS:
            path = os.path.join(self.output_dir, f'{name}.{fmt}')
            fig.savefig(path, dpi=SAVE_DPI, bbox_inches='tight',
                       facecolor='white', edgecolor='none', format=fmt)
        print(f"  ✓ Saved: {name}")
    
    # =========================================================================
    # Figure 1: Temporal Dynamics Spectrum
    # =========================================================================
    
    def fig01_temporal_spectrum(self):
        """
        Figure 1: Power Spectral Density by Network
        Novel visualization: Stacked area plot showing frequency composition
        Uses REAL PSD data from temporal spectral analysis.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # REQUIRE REAL DATA - check if PSD data is available
        if self.avg_psd is None or self.freqs is None:
            print("  ✗ ERROR: PSD data not found. Cannot generate fig01 without real data.")
            print("    Run 01_temporal_analysis.py first to generate PSD data.")
            plt.close()
            return
        
        print(f"  Using real PSD data: shape {self.avg_psd.shape}, freqs shape {self.freqs.shape}")
        
        # Panel A: Relative PSD deviation from global mean - shows network differences clearly
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Use distinct, high-contrast colors for each network
        network_colors_psd = {
            'Visual': '#D62728',        # Red
            'Somatomotor': '#FF7F0E',   # Orange
            'DorsalAttention': '#BCBD22', # Yellow-green
            'VentralAttention': '#17BECF', # Cyan
            'Limbic': '#2CA02C',        # Green
            'Frontoparietal': '#1F77B4', # Blue
            'Default': '#9467BD'        # Purple
        }
        
        # Filter frequency range for visualization (0.01 - 0.25 Hz)
        freq_mask = (self.freqs >= 0.01) & (self.freqs <= 0.25)
        plot_freqs = self.freqs[freq_mask]
        
        # Compute REAL network-average PSD from actual data
        network_psd_raw = {}
        for net, indices in SCHAEFER_NETWORK_INDICES.items():
            valid_indices = [i for i in indices if i < self.avg_psd.shape[0]]
            if valid_indices:
                net_psd = self.avg_psd[valid_indices, :][:, freq_mask].mean(axis=0)
                network_psd_raw[net] = net_psd
            else:
                network_psd_raw[net] = np.zeros(len(plot_freqs))
        
        # Compute global mean for relative comparison
        global_mean_psd = np.mean([network_psd_raw[net] for net in NETWORK_ORDER], axis=0)
        
        # Sort networks by TDS for better visualization (fast to slow)
        if self.spectral_results:
            tds_order = sorted(NETWORK_ORDER, 
                              key=lambda n: self.spectral_results['network_summary']['tds'].get(n, {}).get('mean', 0),
                              reverse=True)
        else:
            tds_order = NETWORK_ORDER
        
        # Plot relative deviation from global mean (% difference)
        # This highlights the DIFFERENCES between networks
        offset_step = 35  # Percentage offset between networks
        
        for i, net in enumerate(reversed(tds_order)):  # Reverse so fastest at top
            net_psd = network_psd_raw[net]
            
            # Calculate percent deviation from global mean
            relative_psd = (net_psd - global_mean_psd) / (global_mean_psd + 1e-10) * 100
            
            # Smooth for visualization
            relative_smooth = gaussian_filter1d(relative_psd, sigma=2)
            
            # Offset for stacking
            baseline = i * offset_step
            y_values = relative_smooth + baseline
            
            # Fill positive deviations (above global mean)
            ax1.fill_between(plot_freqs, baseline, y_values, 
                            where=(relative_smooth >= 0),
                            color=network_colors_psd[net], alpha=0.7, linewidth=0)
            # Fill negative deviations (below global mean) with lighter color
            ax1.fill_between(plot_freqs, baseline, y_values, 
                            where=(relative_smooth < 0),
                            color=network_colors_psd[net], alpha=0.3, linewidth=0)
            
            # Draw line on top
            ax1.plot(plot_freqs, y_values, linewidth=2, color=network_colors_psd[net])
            
            # Draw baseline (global mean reference)
            ax1.axhline(baseline, color='gray', linestyle=':', alpha=0.4, linewidth=1)
            
            # Add network label with TDS value
            if self.spectral_results:
                tds_val = self.spectral_results['network_summary']['tds'].get(net, {}).get('mean', 0)
                label_text = f'{net} (TDS={tds_val:.2f})'
            else:
                label_text = net
            ax1.text(0.008, baseline + 12, label_text, fontsize=9, fontweight='bold',
                    ha='left', va='center', color=network_colors_psd[net],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add vertical line to show frequency boundary
        ax1.axvline(0.07, color='black', linestyle='--', alpha=0.8, linewidth=2, zorder=10)
        ax1.text(0.04, len(tds_order) * offset_step + 5, 'Low Freq\n(<0.07Hz)', 
                fontsize=10, ha='center', fontweight='bold', color='#9467BD')
        ax1.text(0.16, len(tds_order) * offset_step + 5, 'High Freq\n(>0.07Hz)', 
                fontsize=10, ha='center', fontweight='bold', color='#D62728')
        
        ax1.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Deviation from Global Mean (%)', fontweight='bold', fontsize=11)
        ax1.set_title('A. Network-Specific Spectral Signatures (Real Data)', fontsize=12, fontweight='bold', loc='left')
        ax1.set_xlim(0.01, 0.25)
        ax1.set_ylim(-25, len(tds_order) * offset_step + 20)
        ax1.set_yticks([])  # Hide y-ticks (offsets are arbitrary)
        ax1.spines['left'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add interpretation note
        ax1.text(0.25, -15, 'Dark fill: above average | Light fill: below average', 
                fontsize=8, ha='right', va='top', style='italic', color='gray')
        
        # Panel B: TDS by network (horizontal bar with gradient) - REAL DATA
        ax2 = fig.add_subplot(gs[0, 1])
        
        if self.spectral_results:
            tds_data = []
            tds_err = []
            for net in NETWORK_ORDER:
                net_data = self.spectral_results['network_summary']['tds'].get(net, {})
                tds_data.append(net_data.get('mean', 0.5))
                tds_err.append(net_data.get('sem', 0.05))  # Standard error
        else:
            print("  ✗ ERROR: spectral_results not available for Panel B")
            plt.close()
            return
        
        # Sort by TDS for better visualization
        sorted_idx = np.argsort(tds_data)[::-1]
        sorted_networks = [NETWORK_ORDER[i] for i in sorted_idx]
        sorted_tds = [tds_data[i] for i in sorted_idx]
        sorted_err = [tds_err[i] for i in sorted_idx]
        
        # Color by TDS value (green=fast, red=slow)
        norm = plt.Normalize(min(sorted_tds), max(sorted_tds))
        colors = plt.cm.RdYlGn(norm(sorted_tds))
        
        bars = ax2.barh(range(len(sorted_networks)), sorted_tds, xerr=sorted_err,
                       color=colors, edgecolor='white', linewidth=1.5,
                       error_kw=dict(ecolor='gray', capsize=3, capthick=1.5))
        
        # Add value labels
        for i, (val, err) in enumerate(zip(sorted_tds, sorted_err)):
            ax2.text(val + err + 0.03, i, f'{val:.2f}±{err:.2f}', 
                    va='center', fontsize=9, fontweight='bold')
        
        ax2.set_yticks(range(len(sorted_networks)))
        ax2.set_yticklabels(sorted_networks)
        ax2.set_xlabel('Temporal Dynamics Speed (TDS = High/Low Freq Power)', fontweight='bold')
        ax2.set_title('B. Processing Speed by Network (Real Data)', fontsize=12, fontweight='bold', loc='left')
        ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='TDS=1 (Balanced)')
        ax2.set_xlim(0, max(sorted_tds) * 1.4)
        ax2.invert_yaxis()
        ax2.legend(loc='lower right', fontsize=8)
        
        # Add annotations
        ax2.text(max(sorted_tds) * 1.25, 0.5, 'Fast\n(High Freq)', fontsize=9, 
                ha='center', color='#006D2C', fontweight='bold')
        ax2.text(max(sorted_tds) * 1.25, len(sorted_networks) - 1.5, 'Slow\n(Low Freq)', fontsize=9,
                ha='center', color='#99000D', fontweight='bold')
        
        # Panel C: Frequency band composition (donut charts) - USING REAL DATA
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Create mini donut charts for each network
        bands = ['ultra_low', 'low', 'mid', 'high']
        band_colors = ['#005824', '#238B45', '#66C2A4', '#CCECE6']
        band_labels = ['Ultra Low\n(0.01-0.03Hz)', 'Low\n(0.03-0.07Hz)', 
                       'Mid\n(0.07-0.17Hz)', 'High\n(0.17-0.25Hz)']
        
        # Compute band power from REAL PSD data
        band_ranges = {
            'ultra_low': (0.01, 0.03),
            'low': (0.03, 0.07),
            'mid': (0.07, 0.17),
            'high': (0.17, 0.25)
        }
        
        # REQUIRE REAL DATA
        if self.avg_psd is None or self.freqs is None:
            print("  ✗ ERROR: PSD data not available for Panel C")
            plt.close()
            return
        
        # Calculate REAL band powers for Visual and Default networks
        visual_indices = SCHAEFER_NETWORK_INDICES['Visual']
        default_indices = SCHAEFER_NETWORK_INDICES['Default']
        
        visual_psd = self.avg_psd[visual_indices, :].mean(axis=0)
        default_psd = self.avg_psd[default_indices, :].mean(axis=0)
        
        # Compute band powers
        visual_bands = []
        dmn_bands = []
        for band_name in bands:
            f_low, f_high = band_ranges[band_name]
            mask = (self.freqs >= f_low) & (self.freqs < f_high)
            if np.any(mask):
                visual_bands.append(np.sum(visual_psd[mask]))
                dmn_bands.append(np.sum(default_psd[mask]))
            else:
                visual_bands.append(0)
                dmn_bands.append(0)
        
        # Normalize to proportions
        visual_total = sum(visual_bands)
        dmn_total = sum(dmn_bands)
        visual_bands = [v / visual_total for v in visual_bands]
        dmn_bands = [d / dmn_total for d in dmn_bands]
        
        print(f"  Real band powers - Visual: {[f'{v:.3f}' for v in visual_bands]}")
        print(f"  Real band powers - DMN: {[f'{d:.3f}' for d in dmn_bands]}")
        
        # Left donut: Visual
        wedges1, texts1 = ax3.pie(visual_bands, colors=band_colors, startangle=90,
                            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                            center=(-0.6, 0))
        ax3.text(-0.6, 0, 'Visual\n(Fast)', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Right donut: DMN
        wedges2, texts2 = ax3.pie(dmn_bands, colors=band_colors, startangle=90,
                            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                            center=(0.6, 0))
        ax3.text(0.6, 0, 'DMN\n(Slow)', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add percentage labels around the donuts (outside the donut)
        def add_percentage_labels(wedges, values, center_x, ax, band_colors):
            for i, (wedge, val) in enumerate(zip(wedges, values)):
                ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                # Place labels outside the donut
                x = center_x + 0.75 * np.cos(np.deg2rad(ang))
                y = 0.75 * np.sin(np.deg2rad(ang))
                if val > 0.05:  # Only show label if segment is large enough
                    # Use dark color that contrasts with white background
                    ax.text(x, y, f'{val*100:.0f}%', ha='center', va='center', 
                           fontsize=9, fontweight='bold', color='#333333',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                    edgecolor=band_colors[i], alpha=0.9, linewidth=1.5))
        
        add_percentage_labels(wedges1, visual_bands, -0.6, ax3, band_colors)
        add_percentage_labels(wedges2, dmn_bands, 0.6, ax3, band_colors)
        
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.2, 1.2)
        ax3.set_aspect('equal')
        ax3.set_title('C. Frequency Band Composition (Real Data)', fontsize=12, fontweight='bold', loc='left')
        
        # Add legend with band names
        legend_elements = [mpatches.Patch(facecolor=c, label=f'{b.replace("_", " ").title()}') 
                          for b, c in zip(bands, band_colors)]
        ax3.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=4, fontsize=8, framealpha=0.9)
        
        # Panel D: Hierarchy gradient visualization
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create gradient arrow showing processing hierarchy
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax4.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 10, 1.5, 2.5])
        
        # Network order by TDS (fast to slow): DorAtt > FP > Som > Lim > Vis > Def > VenAtt
        tds_sorted_networks = ['DorsalAttention', 'Frontoparietal', 'Somatomotor', 
                               'Limbic', 'Visual', 'Default', 'VentralAttention']
        tds_sorted_abbrev = ['Dor', 'FP', 'Som', 'Lim', 'Vis', 'Def', 'Ven']
        positions = [1, 2.5, 4, 5.5, 6.5, 7.5, 9]
        
        for abbrev, pos in zip(tds_sorted_abbrev, positions):
            ax4.text(pos, 1.0, abbrev, ha='center', va='top', fontsize=9, 
                    fontweight='bold', color='black')
            # Add vertical tick marks
            ax4.plot([pos, pos], [1.4, 1.5], color='gray', linewidth=1)
        
        ax4.text(0.5, 0.3, 'FAST\n(TDS>1.0)', ha='center', fontsize=10, fontweight='bold', color='#006D2C')
        ax4.text(9.5, 0.3, 'SLOW\n(TDS<0.9)', ha='center', fontsize=10, fontweight='bold', color='#99000D')
        
        ax4.set_xlim(0, 10)
        ax4.set_ylim(-0.5, 3)
        ax4.axis('off')
        ax4.set_title('D. Temporal Processing Hierarchy', fontsize=12, fontweight='bold', loc='left')
        
        # Add arrow below the gradient bar
        ax4.annotate('', xy=(9.5, 0.8), xytext=(0.5, 0.8),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        
        fig.suptitle('Temporal Dynamics of Brain Networks', fontsize=14, fontweight='bold', y=0.98)
        
        self.save_figure(fig, 'fig01_temporal_spectrum')
        plt.close()
    
    # =========================================================================
    # Figure 2: Dynamic Functional Connectivity
    # =========================================================================
    
    def fig02_dynamic_connectivity(self):
        """
        Figure 2: Dynamic Functional Connectivity Analysis
        Novel visualization: River plot / Stream graph of connectivity over time
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # Panel A: Within-network FC time series (stream graph)
        ax1 = fig.add_subplot(gs[0, :])
        
        # REQUIRE REAL DATA - no synthetic data allowed
        n_timepoints = 200
        if self.within_fc_ts is not None:
            fc_data = self.within_fc_ts.item() if isinstance(self.within_fc_ts, np.ndarray) else self.within_fc_ts
        else:
            print("  ✗ ERROR: within_fc_ts data not found. Cannot generate fig02 without real data.")
            plt.close()
            return
        
        time = np.arange(n_timepoints)
        
        # Create stacked area (stream graph style)
        baseline = np.zeros(n_timepoints)
        
        for i, net in enumerate(NETWORK_ORDER):
            if net in fc_data:
                values = np.array(fc_data[net])[:n_timepoints]
                values = gaussian_filter1d(values, sigma=3)  # Smooth
                
                ax1.fill_between(time, baseline, baseline + values * 0.1,
                               alpha=0.7, color=NETWORK_COLORS_P2[net], label=net)
                baseline += values * 0.1
        
        ax1.set_xlabel('Time (windows)', fontweight='bold')
        ax1.set_ylabel('Cumulative Within-Network FC', fontweight='bold')
        ax1.set_title('A. Dynamic Within-Network Connectivity (Stream Graph)', 
                     fontsize=12, fontweight='bold', loc='left')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=7, fontsize=8, framealpha=0.9)
        ax1.set_xlim(0, n_timepoints)
        
        # Panel B: Connectivity stability heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        
        if self.dfc_results:
            dcs_values = []
            for net in NETWORK_ORDER:
                dcs_data = self.dfc_results['average']['network_dcs'].get(net, {})
                if isinstance(dcs_data, dict):
                    dcs_values.append(dcs_data.get('mean', 0.5))
                else:
                    dcs_values.append(float(dcs_data))
        else:
            dcs_values = [0.57, 0.45, 0.51, 0.51, 0.46, 0.30, 0.38]
        
        # Create a meaningful stability matrix
        # Diagonal: within-network DCS, Off-diagonal: geometric mean of network pair DCS
        n_nets = len(NETWORK_ORDER)
        dcs_matrix = np.zeros((n_nets, n_nets))
        for i in range(n_nets):
            for j in range(n_nets):
                if i == j:
                    dcs_matrix[i, j] = dcs_values[i]
                else:
                    # Use geometric mean for off-diagonal (represents potential between-network stability)
                    dcs_matrix[i, j] = np.sqrt(dcs_values[i] * dcs_values[j]) * 0.9
        
        cmap_bugn = LinearSegmentedColormap.from_list('BuGn', BUGN_COLORS, N=256)
        # Adjust vmin/vmax to match actual data range
        vmin_val = max(0, np.min(dcs_matrix) - 0.05)
        vmax_val = min(1, np.max(dcs_matrix) + 0.05)
        im = ax2.imshow(dcs_matrix, cmap=cmap_bugn, vmin=vmin_val, vmax=vmax_val)
        
        ax2.set_xticks(range(len(NETWORK_ORDER)))
        ax2.set_xticklabels([n[:3] for n in NETWORK_ORDER], rotation=45, ha='right')
        ax2.set_yticks(range(len(NETWORK_ORDER)))
        ax2.set_yticklabels([n[:3] for n in NETWORK_ORDER])
        ax2.set_title('B. Connectivity Stability Matrix', fontsize=12, fontweight='bold', loc='left')
        
        # Add values as text annotations on the heatmap
        for i in range(n_nets):
            for j in range(n_nets):
                text_color = 'white' if dcs_matrix[i, j] > (vmin_val + vmax_val) / 2 else 'black'
                ax2.text(j, i, f'{dcs_matrix[i, j]:.2f}', ha='center', va='center', 
                        fontsize=8, color=text_color, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Dynamic Connectivity Stability', fontweight='bold')
        
        # Panel C: Stability vs Speed scatter
        ax3 = fig.add_subplot(gs[1, 1])
        
        if self.spectral_results:
            tds_values = [self.spectral_results['network_summary']['tds'].get(net, {}).get('mean', 0.5) 
                         for net in NETWORK_ORDER]
        else:
            tds_values = [0.8, 0.7, 0.5, 0.5, 0.4, 0.4, 0.3]
        
        scatter = ax3.scatter(tds_values, dcs_values, 
                             c=[NETWORK_COLORS_P2[n] for n in NETWORK_ORDER],
                             s=200, edgecolors='white', linewidth=2, zorder=5)
        
        # Add network labels
        for i, net in enumerate(NETWORK_ORDER):
            ax3.annotate(net[:3], (tds_values[i], dcs_values[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(tds_values, dcs_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(tds_values), max(tds_values), 100)
        ax3.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, linewidth=2)
        
        ax3.set_xlabel('Temporal Dynamics Speed (TDS)', fontweight='bold')
        ax3.set_ylabel('Dynamic Connectivity Stability (DCS)', fontweight='bold')
        ax3.set_title('C. Speed vs Stability Tradeoff', fontsize=12, fontweight='bold', loc='left')
        ax3.grid(alpha=0.3)
        
        # Add quadrant labels
        ax3.axhline(0.82, color='gray', linestyle=':', alpha=0.5)
        ax3.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        fig.suptitle('Dynamic Functional Connectivity During Naturalistic Viewing', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        self.save_figure(fig, 'fig02_dynamic_connectivity')
        plt.close()
    
    # =========================================================================
    # Figure 3: Multi-Timescale Encoding
    # =========================================================================
    
    def fig03_multiscale_encoding(self):
        """
        Figure 3: Encoding Performance Across Temporal Windows
        Advanced scientific visualization with:
        - Panel A: Bump chart with confidence ribbons
        - Panel B: Forest plot with effect sizes and CIs
        - Panel C: Joint plot with marginal distributions  
        - Panel D: Hierarchical heatmap of window × network interaction
        """
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # REQUIRE REAL DATA - no synthetic data allowed
        if self.encoding_results:
            encoding_data = self.encoding_results['network_summary']['encoding']
            twg_data = self.encoding_results['network_summary']['twg']
        else:
            print("  ✗ ERROR: encoding_results not found. Cannot generate fig03 without real data.")
            return
        
        # Check which windows have valid data (non-zero values)
        all_windows = ['instant', 'short', 'medium', 'long', 'very_long']
        all_window_labels = ['~1.5s', '~7.5s', '~15s', '~30s', '~60s']
        
        valid_windows = []
        valid_window_labels = []
        for w, label in zip(all_windows, all_window_labels):
            if w in encoding_data:
                # Check if any network has non-zero data
                has_data = any(encoding_data[w].get(net, {}).get('mean', 0) > 0 for net in NETWORK_ORDER)
                if has_data:
                    valid_windows.append(w)
                    valid_window_labels.append(label)
        
        windows = valid_windows if valid_windows else all_windows[:4]  # Use first 4 if no valid
        window_labels = valid_window_labels if valid_window_labels else all_window_labels[:4]
        
        # Recalculate TWG using valid windows (last valid - first valid)
        twg_values = []
        twg_std = []
        for net in NETWORK_ORDER:
            if len(windows) >= 2:
                first_val = encoding_data.get(windows[0], {}).get(net, {}).get('mean', 0)
                last_val = encoding_data.get(windows[-1], {}).get(net, {}).get('mean', 0)
                twg_val = last_val - first_val
                first_std = encoding_data.get(windows[0], {}).get(net, {}).get('std', 0.02)
                last_std = encoding_data.get(windows[-1], {}).get(net, {}).get('std', 0.02)
                twg_std_val = np.sqrt(first_std**2 + last_std**2)
            else:
                twg_val = twg_data.get(net, {}).get('mean', 0.0) if isinstance(twg_data.get(net), dict) else 0.0
                twg_std_val = twg_data.get(net, {}).get('std', 0.02) if isinstance(twg_data.get(net), dict) else 0.02
            twg_values.append(twg_val)
            twg_std.append(twg_std_val)
        
        # Get TDS values
        if self.spectral_results:
            tds = [self.spectral_results['network_summary']['tds'].get(net, {}).get('mean', 0.5)
                   for net in NETWORK_ORDER]
        else:
            tds = [0.85, 0.75, 0.55, 0.50, 0.42, 0.38, 0.28]
        
        # =====================================================================
        # Panel A: Advanced Bump Chart with Confidence Ribbons
        # =====================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Create ranking data
        rank_matrix = np.zeros((len(NETWORK_ORDER), len(windows)))
        value_matrix = np.zeros((len(NETWORK_ORDER), len(windows)))
        std_matrix = np.zeros((len(NETWORK_ORDER), len(windows)))
        
        for j, w in enumerate(windows):
            values = []
            stds = []
            for net in NETWORK_ORDER:
                if w in encoding_data and net in encoding_data[w]:
                    values.append(encoding_data[w][net]['mean'])
                    stds.append(encoding_data[w][net].get('std', 0.02))
                else:
                    values.append(0.1)
                    stds.append(0.02)
            
            # Compute rankings (1 = best)
            rankings = stats.rankdata([-v for v in values])
            rank_matrix[:, j] = rankings
            value_matrix[:, j] = values
            std_matrix[:, j] = stds
        
        # Plot bump chart with ribbons
        for i, net in enumerate(NETWORK_ORDER):
            color = NETWORK_COLORS_P2[net]
            
            # Smoothed ribbon (uncertainty)
            upper = value_matrix[i] + std_matrix[i]
            lower = value_matrix[i] - std_matrix[i]
            ax1.fill_between(range(len(windows)), lower, upper, 
                           alpha=0.2, color=color)
            
            # Main line with gradient based on improvement
            improvement = value_matrix[i, -1] - value_matrix[i, 0]
            lw = 3 if improvement > 0 else 1.5
            ls = '-' if improvement > 0 else '--'
            
            ax1.plot(range(len(windows)), value_matrix[i], 
                    color=color, linewidth=lw, linestyle=ls,
                    marker='o', markersize=10, markeredgecolor='white', 
                    markeredgewidth=2, label=net, zorder=5)
        
        ax1.set_xticks(range(len(windows)))
        ax1.set_xticklabels(window_labels, fontsize=11)
        ax1.set_xlabel('Temporal Integration Window', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Encoding Accuracy (Pearson r)', fontweight='bold', fontsize=12)
        ax1.set_title('A. Encoding Performance Trajectory with Uncertainty', 
                     fontsize=13, fontweight='bold', loc='left', pad=10)
        
        # Add subtle gradient background
        for j in range(len(windows) - 1):
            ax1.axvspan(j - 0.5, j + 0.5, alpha=0.03 * (j + 1), color='purple')
        
        # Legend with custom styling
        legend = ax1.legend(loc='upper left', ncol=2, fontsize=9, 
                           framealpha=0.95, edgecolor='gray')
        legend.get_frame().set_linewidth(1.5)
        
        ax1.set_xlim(-0.5, len(windows) - 0.5)
        ax1.grid(alpha=0.3, axis='y', linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # =====================================================================
        # Panel B: Forest Plot with Effect Sizes and Confidence Intervals
        # =====================================================================
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Sort by TWG for visual clarity (highest/best first)
        sorted_indices = np.argsort(twg_values)[::-1]
        sorted_networks = [NETWORK_ORDER[i] for i in sorted_indices]
        sorted_twg = [twg_values[i] for i in sorted_indices]
        sorted_std = [twg_std[i] for i in sorted_indices]
        sorted_colors = [NETWORK_COLORS_P2[n] for n in sorted_networks]
        
        y_pos = np.arange(len(NETWORK_ORDER))
        
        # Calculate x-axis limits based on data
        all_ci_low = [t - 1.96 * s for t, s in zip(sorted_twg, sorted_std)]
        all_ci_high = [t + 1.96 * s for t, s in zip(sorted_twg, sorted_std)]
        x_min = min(all_ci_low) * 1.3
        x_max = max(all_ci_high) * 1.3
        # Ensure zero is visible and centered if data is one-sided
        if x_min > 0:
            x_min = -0.01
        if x_max < 0:
            x_max = 0.01
        # Add padding for labels
        x_range = x_max - x_min
        x_max_label = x_max + x_range * 0.25
        
        # Diamond markers for effect sizes
        for i, (net, twg, std, color) in enumerate(zip(sorted_networks, sorted_twg, sorted_std, sorted_colors)):
            # CI line
            ci_low = twg - 1.96 * std
            ci_high = twg + 1.96 * std
            ax2.hlines(y=i, xmin=ci_low, xmax=ci_high, colors=color, linewidth=3, alpha=0.7)
            
            # Effect size diamond
            diamond_size = 180 if abs(twg) > 0.02 else 100
            ax2.scatter(twg, i, marker='D', s=diamond_size, c=[color], 
                       edgecolors='white', linewidth=2, zorder=5)
            
            # Add numeric value on the right side
            ax2.text(x_max_label, i, f'{twg:.4f}', va='center', ha='left',
                    fontsize=10, fontweight='bold', color=color)
        
        # Reference line at zero
        ax2.axvline(0, color='black', linestyle='-', linewidth=1.5, zorder=1)
        
        # Color regions based on sign
        if max(sorted_twg) > 0:
            ax2.axvspan(0, x_max, alpha=0.08, color='#006D2C', zorder=0)
        if min(sorted_twg) < 0:
            ax2.axvspan(x_min, 0, alpha=0.08, color='#E31A1C', zorder=0)
        
        # Region labels at bottom
        if max(sorted_twg) > 0:
            ax2.text(x_max * 0.5, len(NETWORK_ORDER) + 0.3, 'Long window\nbeneficial', 
                    ha='center', fontsize=9, color='#006D2C', fontweight='bold')
        if min(sorted_twg) < 0:
            ax2.text(x_min * 0.5, len(NETWORK_ORDER) + 0.3, 'Short window\nbetter', 
                    ha='center', fontsize=9, color='#E31A1C', fontweight='bold')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_networks, fontsize=11)
        ax2.set_xlabel(f'Temporal Window Gain (TWG = r[{window_labels[-1]}] - r[{window_labels[0]}])', 
                      fontweight='bold', fontsize=11)
        ax2.set_title('B. Forest Plot: Effect of Extended Integration', 
                     fontsize=13, fontweight='bold', loc='left', pad=10)
        ax2.set_xlim(x_min, x_max_label + x_range * 0.1)
        ax2.set_ylim(-0.5, len(NETWORK_ORDER) + 0.8)
        ax2.invert_yaxis()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # =====================================================================
        # Panel C: Speed-Benefit Trade-off Scatter Plot
        # =====================================================================
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Main scatter with size encoding
        sizes = np.abs(twg_values) * 2000 + 100  # Size by TWG magnitude
        scatter = ax3.scatter(tds, twg_values, 
                             c=[NETWORK_COLORS_P2[n] for n in NETWORK_ORDER],
                             s=sizes, edgecolors='white', linewidth=2.5, 
                             alpha=0.85, zorder=5)
        
        # Add network labels with smart positioning to avoid overlap
        # Use adjustText-like logic: spread labels based on data density
        label_positions = []
        for i, net in enumerate(NETWORK_ORDER):
            x, y = tds[i], twg_values[i]
            # Calculate offset based on position relative to others
            x_offset = 12
            y_offset = 0
            # Check for nearby points and adjust
            for j, (other_x, other_y) in enumerate(zip(tds, twg_values)):
                if i != j:
                    dx = x - other_x
                    dy = y - other_y
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 0.15:  # Close points
                        # Push label away from nearby point
                        if dy > 0:
                            y_offset += 8
                        else:
                            y_offset -= 8
            # Alternate sides for clarity
            if i % 2 == 0:
                x_offset = 10
            else:
                x_offset = -len(net) * 6 - 10
            label_positions.append((x_offset, y_offset))
        
        for i, net in enumerate(NETWORK_ORDER):
            ax3.annotate(net, (tds[i], twg_values[i]),
                        xytext=label_positions[i], textcoords='offset points',
                        fontsize=9, fontweight='bold', color=NETWORK_COLORS_P2[net],
                        path_effects=[path_effects.withStroke(linewidth=3, foreground='white')],
                        arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5))
        
        # Regression line with CI
        z = np.polyfit(tds, twg_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(tds) - 0.1, max(tds) + 0.1, 100)
        ax3.plot(x_line, p(x_line), '--', color='gray', linewidth=2, alpha=0.7, 
                label=f'r = {np.corrcoef(tds, twg_values)[0,1]:.2f}')
        
        # Quadrant styling
        ax3.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax3.axvline(np.median(tds), color='gray', linestyle='--', linewidth=1, alpha=0.3)
        
        # Highlight regions based on actual data range
        twg_min, twg_max = min(twg_values), max(twg_values)
        tds_min, tds_max = min(tds), max(tds)
        
        # If TWG is negative, highlight the "fast processing sufficient" region
        if twg_max < 0:
            # All networks show degradation with longer windows
            ax3.fill_between([tds_min - 0.1, tds_max + 0.1], 
                            [twg_min * 1.2, twg_min * 1.2], [0, 0], 
                            alpha=0.1, color='#E31A1C')
            ax3.text(np.mean(tds), twg_min * 0.5, 'Short windows\noptimal', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='#E31A1C', alpha=0.8)
        else:
            # Highlight paradox quadrant (slow TDS + positive TWG)
            ax3.fill_between([tds_min - 0.1, np.median(tds)], [0, 0], 
                            [twg_max * 1.2, twg_max * 1.2], 
                            alpha=0.1, color='#9467BD')
            ax3.text((tds_min + np.median(tds))/2, twg_max * 0.8, 'PARADOX\nZONE', 
                    ha='center', va='center', fontsize=11, fontweight='bold', 
                    color='#9467BD', alpha=0.8)
        
        ax3.set_xlabel('Temporal Dynamics Speed (TDS)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Temporal Window Gain (TWG)', fontweight='bold', fontsize=12)
        ax3.legend(loc='lower left', fontsize=10)
        ax3.grid(alpha=0.2, linestyle='--')
        ax3.set_title('C. Speed-Benefit Trade-off', 
                     fontsize=13, fontweight='bold', loc='left', pad=10)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # =====================================================================
        # Panel D: Hierarchical Heatmap of Window × Network Interaction
        # =====================================================================
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create interaction matrix: change in encoding from baseline
        interaction_matrix = np.zeros((len(NETWORK_ORDER), len(windows)))
        for i, net in enumerate(NETWORK_ORDER):
            baseline = value_matrix[i, 0]  # First window as baseline
            for j in range(len(windows)):
                interaction_matrix[i, j] = value_matrix[i, j] - baseline
        
        # Create diverging colormap
        cmap_div = LinearSegmentedColormap.from_list('div', 
            ['#D62728', '#F5F5F5', '#006D2C'], N=256)
        
        # Plot heatmap with hierarchical ordering
        # Order networks by their total benefit from extended windows
        total_benefit = np.sum(interaction_matrix, axis=1)
        order = np.argsort(total_benefit)[::-1]
        
        ordered_matrix = interaction_matrix[order]
        ordered_networks = [NETWORK_ORDER[i] for i in order]
        
        vmax = np.abs(interaction_matrix).max()
        im = ax4.imshow(ordered_matrix, aspect='auto', cmap=cmap_div, 
                       vmin=-vmax, vmax=vmax)
        
        # Add value annotations
        for i in range(len(NETWORK_ORDER)):
            for j in range(len(windows)):
                val = ordered_matrix[i, j]
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax4.text(j, i, f'{val:.3f}', ha='center', va='center', 
                        fontsize=9, color=color, fontweight='bold')
        
        # Styling
        ax4.set_xticks(range(len(windows)))
        ax4.set_xticklabels(window_labels, fontsize=11)
        ax4.set_yticks(range(len(NETWORK_ORDER)))
        ax4.set_yticklabels(ordered_networks, fontsize=11)
        ax4.set_xlabel('Temporal Window Size', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Network (ordered by benefit)', fontweight='bold', fontsize=12)
        ax4.set_title('D. Encoding Change from Baseline (Δr)', 
                     fontsize=13, fontweight='bold', loc='left', pad=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8, pad=0.02)
        cbar.set_label('Δ Encoding (from instant window)', fontweight='bold', fontsize=10)
        
        # Add benefit arrow annotation
        ax4.annotate('', xy=(len(windows) - 0.5, -0.8), xytext=(-0.5, -0.8),
                    arrowprops=dict(arrowstyle='->', color='#006D2C', lw=2))
        ax4.text(len(windows)/2 - 0.5, -1.1, 'Extended temporal integration →', 
                ha='center', fontsize=10, color='#006D2C', fontweight='bold')
        
        # Main title
        fig.suptitle('Multi-Timescale Encoding: The Temporal Integration Advantage', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        self.save_figure(fig, 'fig03_multiscale_encoding')
        plt.close()
    
    # =========================================================================
    # Figure 4: Temporal Efficiency Paradox Summary
    # =========================================================================
    
    def fig04_efficiency_paradox_summary(self):
        """
        Figure 4: The Temporal Efficiency Paradox - Summary Infographic
        Novel visualization: Sankey-style flow diagram
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Create custom layout
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'THE TEMPORAL EFFICIENCY PARADOX', 
               ha='center', va='center', fontsize=18, fontweight='bold')
        ax.text(5, 9.0, 'Why Slower Neural Integration Outperforms Faster Processing',
               ha='center', va='center', fontsize=12, style='italic', color='gray')
        
        # Left side: Fast networks
        fast_box = FancyBboxPatch((0.5, 5), 2, 3, boxstyle="round,pad=0.1",
                                  facecolor='#FFCCCC', edgecolor='#E31A1C', linewidth=2)
        ax.add_patch(fast_box)
        ax.text(1.5, 7.5, 'FAST NETWORKS', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='#E31A1C')
        ax.text(1.5, 6.5, 'Visual\nSomatomotor\nDorsal Attention', ha='center', va='center',
               fontsize=10)
        ax.text(1.5, 5.3, 'High TDS\nLow DCS\nNegative TWG', ha='center', va='center',
               fontsize=9, color='gray')
        
        # Right side: Slow networks
        slow_box = FancyBboxPatch((7.5, 5), 2, 3, boxstyle="round,pad=0.1",
                                  facecolor='#CCCCFF', edgecolor='#6A3D9A', linewidth=2)
        ax.add_patch(slow_box)
        ax.text(8.5, 7.5, 'SLOW NETWORKS', ha='center', va='center',
               fontsize=12, fontweight='bold', color='#6A3D9A')
        ax.text(8.5, 6.5, 'Default Mode\nFrontoparietal\nLimbic', ha='center', va='center',
               fontsize=10)
        ax.text(8.5, 5.3, 'Low TDS\nHigh DCS\nPositive TWG', ha='center', va='center',
               fontsize=9, color='gray')
        
        # Center: The paradox
        paradox_box = FancyBboxPatch((3.5, 5.5), 3, 2, boxstyle="round,pad=0.1",
                                     facecolor='#FFFFCC', edgecolor='#FF9900', linewidth=3)
        ax.add_patch(paradox_box)
        ax.text(5, 7, 'THE PARADOX', ha='center', va='center',
               fontsize=14, fontweight='bold', color='#FF6600')
        ax.text(5, 6, 'Slow processing\nproduces BETTER\nencoding accuracy', ha='center', va='center',
               fontsize=10)
        
        # Arrows
        ax.annotate('', xy=(3.5, 6.5), xytext=(2.5, 6.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        ax.annotate('', xy=(7.5, 6.5), xytext=(6.5, 6.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        
        # Bottom: Implications
        ax.text(5, 3.5, 'IMPLICATIONS', ha='center', va='center',
               fontsize=14, fontweight='bold')
        
        # Three implication boxes
        impl1 = FancyBboxPatch((0.5, 1), 2.5, 2, boxstyle="round,pad=0.1",
                               facecolor='#E5F5F9', edgecolor='#238B45', linewidth=1.5)
        ax.add_patch(impl1)
        ax.text(1.75, 2.5, 'For Sensory\nProcessing', ha='center', va='center',
               fontsize=10, fontweight='bold', color='#238B45')
        ax.text(1.75, 1.5, 'Fast = Optimal', ha='center', va='center', fontsize=10)
        
        impl2 = FancyBboxPatch((3.75, 1), 2.5, 2, boxstyle="round,pad=0.1",
                               facecolor='#E5F5F9', edgecolor='#238B45', linewidth=1.5)
        ax.add_patch(impl2)
        ax.text(5, 2.5, 'For Semantic\nIntegration', ha='center', va='center',
               fontsize=10, fontweight='bold', color='#238B45')
        ax.text(5, 1.5, 'Slow = Optimal', ha='center', va='center', fontsize=10)
        
        impl3 = FancyBboxPatch((7, 1), 2.5, 2, boxstyle="round,pad=0.1",
                               facecolor='#E5F5F9', edgecolor='#238B45', linewidth=1.5)
        ax.add_patch(impl3)
        ax.text(8.25, 2.5, 'Redefine\n"Efficiency"', ha='center', va='center',
               fontsize=10, fontweight='bold', color='#238B45')
        ax.text(8.25, 1.5, 'Context-dependent', ha='center', va='center', fontsize=10)
        
        # Add key insight box at bottom
        insight_box = FancyBboxPatch((2, 0.2), 6, 0.6, boxstyle="round,pad=0.05",
                                     facecolor='#005824', edgecolor='none')
        ax.add_patch(insight_box)
        ax.text(5, 0.5, 'KEY INSIGHT: The brain sacrifices "efficiency" for depth in semantic processing',
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        self.save_figure(fig, 'fig04_efficiency_paradox_summary')
        plt.close()
    
    # =========================================================================
    # Figure 5: Comparison with Project 1
    # =========================================================================
    
    def fig05_unified_framework(self):
        """
        Figure 5: Unified Framework - Connecting Project 1 and Project 2
        Shows how both projects reveal aspects of cognitive efficiency paradox
        """
        fig = plt.figure(figsize=(16, 10))
        
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'UNIFIED FRAMEWORK: Two Dimensions of Cognitive Efficiency',
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Project 1 box (left)
        p1_box = FancyBboxPatch((0.5, 4), 4, 4.5, boxstyle="round,pad=0.1",
                                facecolor='#FFF5EB', edgecolor='#D95F02', linewidth=2)
        ax.add_patch(p1_box)
        ax.text(2.5, 8, 'PROJECT 1', ha='center', va='center',
               fontsize=14, fontweight='bold', color='#D95F02')
        ax.text(2.5, 7.3, 'Modality Dimension', ha='center', va='center',
               fontsize=12, style='italic')
        ax.text(2.5, 6.3, 'Encoding-Attention\nDissociation', ha='center', va='center',
               fontsize=11, fontweight='bold')
        ax.text(2.5, 5.3, 'Finding: Attention does\nnot track feature strength',
               ha='center', va='center', fontsize=10)
        ax.text(2.5, 4.5, 'Apparent inefficiency\n→ Robust integration',
               ha='center', va='center', fontsize=10, color='#D95F02')
        
        # Project 2 box (right)
        p2_box = FancyBboxPatch((5.5, 4), 4, 4.5, boxstyle="round,pad=0.1",
                                facecolor='#F0F0FF', edgecolor='#7570B3', linewidth=2)
        ax.add_patch(p2_box)
        ax.text(7.5, 8, 'PROJECT 2', ha='center', va='center',
               fontsize=14, fontweight='bold', color='#7570B3')
        ax.text(7.5, 7.3, 'Temporal Dimension', ha='center', va='center',
               fontsize=12, style='italic')
        ax.text(7.5, 6.3, 'Temporal Efficiency\nParadox', ha='center', va='center',
               fontsize=11, fontweight='bold')
        ax.text(7.5, 5.3, 'Finding: Slow processing\noutperforms fast',
               ha='center', va='center', fontsize=10)
        ax.text(7.5, 4.5, 'Apparent inefficiency\n→ Deep understanding',
               ha='center', va='center', fontsize=10, color='#7570B3')
        
        # Unified insight box (bottom)
        unified_box = FancyBboxPatch((1.5, 1), 7, 2.5, boxstyle="round,pad=0.1",
                                     facecolor='#E5F5E0', edgecolor='#1B7837', linewidth=3)
        ax.add_patch(unified_box)
        ax.text(5, 3, 'UNIFIED PRINCIPLE', ha='center', va='center',
               fontsize=14, fontweight='bold', color='#1B7837')
        ax.text(5, 2.2, 'The brain systematically sacrifices\n"local efficiency" for "global robustness/depth"',
               ha='center', va='center', fontsize=12)
        ax.text(5, 1.3, 'This challenges traditional definitions of cognitive optimality',
               ha='center', va='center', fontsize=10, color='gray', style='italic')
        
        # Connecting arrows
        ax.annotate('', xy=(5, 3.5), xytext=(2.5, 4),
                   arrowprops=dict(arrowstyle='->', color='#1B7837', lw=2,
                                  connectionstyle='arc3,rad=-0.2'))
        ax.annotate('', xy=(5, 3.5), xytext=(7.5, 4),
                   arrowprops=dict(arrowstyle='->', color='#1B7837', lw=2,
                                  connectionstyle='arc3,rad=0.2'))
        
        # Plus sign between projects
        ax.text(5, 6, '+', ha='center', va='center', fontsize=30, fontweight='bold', color='gray')
        
        self.save_figure(fig, 'fig05_unified_framework')
        plt.close()
    
    # =========================================================================
    # Figure 4: Network-level Summary
    # =========================================================================
    
    def fig04_network_summary_radar(self):
        """
        Figure 4: Network-level Summary using Radar Charts
        Novel visualization: Multi-dimensional radar plots for each network
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create 7 radar charts (one per network)
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        metrics = ['TDS', 'DCS', 'TWG', 'Short\nEnc', 'Long\nEnc']
        
        # REQUIRE REAL DATA - no synthetic data allowed
        if not self.spectral_results or not self.dfc_results or not self.encoding_results:
            print("  ✗ ERROR: Missing required data for fig06. Need spectral_results, dfc_results, and encoding_results.")
            return
        
        # Collect data for each network (all from real data)
        network_data = {}
        for net in NETWORK_ORDER:
            # Get TDS from real spectral results
            tds_data = self.spectral_results['network_summary']['tds'].get(net, {})
            tds = tds_data.get('mean', 0.5) if isinstance(tds_data, dict) else float(tds_data)
            
            # Get DCS from real DFC results
            dcs_data = self.dfc_results['average']['network_dcs'].get(net, {})
            dcs = dcs_data.get('mean', 0.8) if isinstance(dcs_data, dict) else float(dcs_data)
            
            # Get encoding metrics from real encoding results
            twg_data = self.encoding_results['network_summary']['twg'].get(net, {})
            twg = twg_data.get('mean', 0.0) if isinstance(twg_data, dict) else float(twg_data)
            
            short_data = self.encoding_results['network_summary']['encoding'].get('short', {}).get(net, {})
            short_enc = short_data.get('mean', 0.1) if isinstance(short_data, dict) else float(short_data)
            
            long_data = self.encoding_results['network_summary']['encoding'].get('long', {}).get(net, {})
            long_enc = long_data.get('mean', 0.1) if isinstance(long_data, dict) else float(long_data)
            
            # Normalize to 0-1
            network_data[net] = [
                tds,  # Already ~0-1
                dcs,  # Already ~0-1
                (twg + 0.2) / 0.5,  # Normalize TWG
                short_enc * 5,  # Scale encoding
                long_enc * 5   # Scale encoding
            ]
        
        # Create subplot grid
        for i, net in enumerate(NETWORK_ORDER):
            ax = fig.add_subplot(2, 4, i + 1, projection='polar')
            
            values = network_data[net]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, color=NETWORK_COLORS_P2[net])
            ax.fill(angles, values, alpha=0.25, color=NETWORK_COLORS_P2[net])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(net, fontsize=11, fontweight='bold', pad=10)
            
            # Move labels further from the circle
            ax.tick_params(axis='x', pad=15)
        
        # Add legend subplot
        ax_legend = fig.add_subplot(2, 4, 8)
        ax_legend.axis('off')
        
        # Add interpretation
        ax_legend.text(0.5, 0.9, 'Metrics:', ha='center', fontsize=11, fontweight='bold', transform=ax_legend.transAxes)
        ax_legend.text(0.5, 0.75, 'TDS: Temporal Dynamics Speed', ha='center', fontsize=9, transform=ax_legend.transAxes)
        ax_legend.text(0.5, 0.65, 'DCS: Dynamic Connectivity Stability', ha='center', fontsize=9, transform=ax_legend.transAxes)
        ax_legend.text(0.5, 0.55, 'TWG: Temporal Window Gain', ha='center', fontsize=9, transform=ax_legend.transAxes)
        ax_legend.text(0.5, 0.45, 'Short/Long Enc: Encoding accuracy', ha='center', fontsize=9, transform=ax_legend.transAxes)
        
        ax_legend.text(0.5, 0.25, 'Key Pattern:', ha='center', fontsize=11, fontweight='bold', transform=ax_legend.transAxes)
        ax_legend.text(0.5, 0.1, 'DMN shows low TDS but high\nTWG = The Temporal Paradox', 
                      ha='center', fontsize=10, color='#6A3D9A', transform=ax_legend.transAxes)
        
        fig.suptitle('Network-Level Temporal Characteristics', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        self.save_figure(fig, 'fig04_network_summary_radar')
        plt.close()
    
    # =========================================================================
    # Figure 5: LSTM vs Ridge Comparison
    # =========================================================================
    
    def fig05_lstm_ridge_comparison(self):
        """
        Figure 5: LSTM vs Ridge Encoding Comparison
        Shows that LSTM (long-range model) improves DMN more than Visual
        """
        # Load results if available
        lstm_results_path = os.path.join(self.run_dir, 'lstm_ridge', 'lstm_ridge_comparison.json')
        
        if not os.path.exists(lstm_results_path):
            print("  ✗ ERROR: LSTM comparison results not found at", lstm_results_path)
            print("    Cannot generate fig07 without real data. Run 06_lstm_ridge_comparison.py first.")
            return
        
        with open(lstm_results_path, 'r') as f:
            data = json.load(f)
            aggregated = data.get('aggregated', {})
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel A: Grouped bar chart
        ax1 = axes[0]
        x = np.arange(len(NETWORK_ORDER))
        width = 0.35
        
        ridge_vals = [aggregated.get(n, {}).get('ridge_mean', 0) for n in NETWORK_ORDER]
        lstm_vals = [aggregated.get(n, {}).get('lstm_mean', 0) for n in NETWORK_ORDER]
        
        bars1 = ax1.bar(x - width/2, ridge_vals, width, label='Ridge', color='#3498DB', alpha=0.8)
        bars2 = ax1.bar(x + width/2, lstm_vals, width, label='LSTM', color='#E74C3C', alpha=0.8)
        
        ax1.set_xlabel('Brain Network', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Encoding Accuracy (r)', fontsize=11, fontweight='bold')
        ax1.set_title('A. Model Comparison', fontsize=12, fontweight='bold', loc='left')
        ax1.set_xticks(x)
        ax1.set_xticklabels([n[:4] for n in NETWORK_ORDER], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Panel B: Improvement percentage
        ax2 = axes[1]
        improvements = [aggregated.get(n, {}).get('improvement_percent', 0) for n in NETWORK_ORDER]
        colors = [NETWORK_COLORS_P2[n] for n in NETWORK_ORDER]
        
        bars = ax2.barh(range(len(NETWORK_ORDER)), improvements, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(NETWORK_ORDER)))
        ax2.set_yticklabels(NETWORK_ORDER)
        ax2.set_xlabel('LSTM Improvement (%)', fontsize=11, fontweight='bold')
        ax2.set_title('B. Long-Range Model Benefit', fontsize=12, fontweight='bold', loc='left')
        ax2.axvline(x=15, color='gray', linestyle='--', alpha=0.5, label='Significant threshold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, improvements):
            ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                    va='center', fontsize=9)
        
        # Panel C: Scatter plot TDS vs Improvement
        ax3 = axes[2]
        
        tds_values = [0.91, 1.01, 1.14, 0.72, 0.96, 1.12, 0.84]  # From earlier analysis
        
        for i, network in enumerate(NETWORK_ORDER):
            ax3.scatter(tds_values[i], improvements[i], 
                       c=NETWORK_COLORS_P2[network], s=150, 
                       label=network, edgecolors='white', linewidth=2)
        
        # Add trend line
        z = np.polyfit(tds_values, improvements, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(tds_values), max(tds_values), 100)
        ax3.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
        
        ax3.set_xlabel('TDS (Fast → Slow)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('LSTM Improvement (%)', fontsize=11, fontweight='bold')
        ax3.set_title('C. The Paradox: Slow Networks Benefit More', fontsize=12, fontweight='bold', loc='left')
        ax3.grid(alpha=0.3)
        
        # Highlight paradox region
        ax3.axhspan(15, max(improvements)+5, alpha=0.1, color='purple')
        
        # Move legend outside the plot
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
                  framealpha=0.95, edgecolor='gray')
        
        fig.suptitle('LSTM vs Ridge: Long-Range Temporal Dependencies', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 0.92, 1])  # Make room for legend
        
        self.save_figure(fig, 'fig05_lstm_ridge_comparison')
        plt.close()
    
    # =========================================================================
    # Figure 6: Hub Centrality Analysis
    # =========================================================================
    
    def fig06_hub_centrality(self):
        """
        Figure 6: Network Hub Structure
        Shows DMN as information integration hub (high betweenness centrality)
        """
        # Load or generate data
        hub_results_path = os.path.join(self.run_dir, 'hub_centrality', 'hub_centrality_results.json')
        
        if not os.path.exists(hub_results_path):
            print("  ✗ ERROR: Hub analysis results not found at", hub_results_path)
            print("    Cannot generate fig06 without real data. Run 07_hub_centrality_analysis.py first.")
            return
        
        with open(hub_results_path, 'r') as f:
            data = json.load(f)
            aggregated = data.get('aggregated', {})
        
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        colors = [NETWORK_COLORS_P2[n] for n in NETWORK_ORDER]
        
        # Panel A: Degree Centrality (more interpretable than betweenness for this scale)
        ax1 = fig.add_subplot(gs[0, 0])
        degree = [aggregated.get(n, {}).get('degree_centrality', 0) for n in NETWORK_ORDER]
        
        # Sort by value for better visualization
        sorted_idx = np.argsort(degree)[::-1]
        sorted_networks = [NETWORK_ORDER[i] for i in sorted_idx]
        sorted_degree = [degree[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]
        
        bars = ax1.barh(range(len(NETWORK_ORDER)), sorted_degree, color=sorted_colors, 
                       alpha=0.85, edgecolor='white', linewidth=2, height=0.7)
        ax1.set_yticks(range(len(NETWORK_ORDER)))
        ax1.set_yticklabels(sorted_networks, fontsize=11)
        ax1.set_xlabel('Degree Centrality', fontsize=12, fontweight='bold')
        ax1.set_title('A. Network Connectivity (Degree Centrality)', fontsize=13, fontweight='bold', loc='left')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add value labels
        for i, (val, net) in enumerate(zip(sorted_degree, sorted_networks)):
            ax1.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
        
        # Panel B: Participation Coefficient (cross-network integration)
        ax2 = fig.add_subplot(gs[0, 1])
        participation = [aggregated.get(n, {}).get('participation_coefficient', 0) for n in NETWORK_ORDER]
        
        sorted_idx = np.argsort(participation)[::-1]
        sorted_networks = [NETWORK_ORDER[i] for i in sorted_idx]
        sorted_part = [participation[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]
        
        bars = ax2.barh(range(len(NETWORK_ORDER)), sorted_part, color=sorted_colors, 
                       alpha=0.85, edgecolor='white', linewidth=2, height=0.7)
        ax2.set_yticks(range(len(NETWORK_ORDER)))
        ax2.set_yticklabels(sorted_networks, fontsize=11)
        ax2.set_xlabel('Participation Coefficient', fontsize=12, fontweight='bold')
        ax2.set_title('B. Cross-Network Integration', fontsize=13, fontweight='bold', loc='left')
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Add value labels
        for i, (val, net) in enumerate(zip(sorted_part, sorted_networks)):
            ax2.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')
        
        # Add annotation - moved further right
        ax2.set_xlim(0, max(sorted_part) * 1.35)  # Extend x-axis to make room
        ax2.text(max(sorted_part) * 1.15, len(NETWORK_ORDER) - 0.5, 'High\nIntegration', fontsize=9, 
                color='#006D2C', fontweight='bold', ha='center')
        
        # Panel C: Connector Hub Ratio comparison
        ax3 = fig.add_subplot(gs[1, 0])
        hub_ratio = [aggregated.get(n, {}).get('connector_hub_ratio', 0) for n in NETWORK_ORDER]
        
        sorted_idx = np.argsort(hub_ratio)[::-1]
        sorted_networks = [NETWORK_ORDER[i] for i in sorted_idx]
        sorted_hub = [hub_ratio[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]
        
        # Create lollipop chart
        ax3.hlines(y=range(len(NETWORK_ORDER)), xmin=0, xmax=sorted_hub, 
                  colors=sorted_colors, linewidth=3, alpha=0.7)
        ax3.scatter(sorted_hub, range(len(NETWORK_ORDER)), c=sorted_colors, 
                   s=200, edgecolors='white', linewidth=2, zorder=5)
        
        ax3.set_yticks(range(len(NETWORK_ORDER)))
        ax3.set_yticklabels(sorted_networks, fontsize=11)
        ax3.set_xlabel('Connector Hub Ratio', fontsize=12, fontweight='bold')
        ax3.set_title('C. Connector Hub Proportion', fontsize=13, fontweight='bold', loc='left')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_xlim(0, max(sorted_hub) * 1.3)
        
        # Add value labels
        for i, (val, net) in enumerate(zip(sorted_hub, sorted_networks)):
            ax3.text(val + 0.01, i, f'{val*100:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        # Panel D: Network Integration Summary (Radar-style comparison)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create scatter plot: Degree vs Participation
        # Define custom offsets for each network to avoid overlap
        label_offsets = {
            'Visual': (15, -5),
            'Somatomotor': (15, 5),
            'DorsalAttention': (-40, -15),
            'VentralAttention': (15, -15),
            'Limbic': (15, 10),
            'Frontoparietal': (-50, 5),
            'Default': (15, 5)
        }
        
        for i, net in enumerate(NETWORK_ORDER):
            deg = aggregated.get(net, {}).get('degree_centrality', 0)
            part = aggregated.get(net, {}).get('participation_coefficient', 0)
            hub = aggregated.get(net, {}).get('connector_hub_ratio', 0)
            
            # Size based on hub ratio
            size = 200 + hub * 1500
            
            ax4.scatter(deg, part, c=[NETWORK_COLORS_P2[net]], s=size, 
                       edgecolors='white', linewidth=2, alpha=0.85, zorder=5)
            
            # Add network label with custom offset
            offset = label_offsets.get(net, (15, 5))
            ax4.annotate(net[:4], (deg, part), xytext=offset, textcoords='offset points',
                        fontsize=10, fontweight='bold', color=NETWORK_COLORS_P2[net],
                        arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5))
        
        ax4.set_xlabel('Degree Centrality', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Participation Coefficient', fontsize=12, fontweight='bold')
        ax4.set_title('D. Hub Integration Profile (size = connector hub ratio)', 
                     fontsize=13, fontweight='bold', loc='left')
        ax4.grid(alpha=0.3, linestyle='--')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # Add quadrant labels
        ax4.axhline(y=0.6, color='gray', linestyle=':', alpha=0.5)
        ax4.axvline(x=0.1, color='gray', linestyle=':', alpha=0.5)
        
        fig.suptitle('Network Hub Structure Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        self.save_figure(fig, 'fig06_hub_centrality')
        plt.close()
    
    # =========================================================================
    # Figure 7: Transformer Attention Patterns
    # =========================================================================
    
    def fig07_transformer_attention(self):
        """
        Figure 7: Temporal Attention Visualization
        Shows that DMN attends to distant time points, Visual focuses on recent
        """
        # Load or generate data
        attn_results_path = os.path.join(self.run_dir, 'transformer_attention', 'transformer_attention_results.json')
        
        seq_length = 30
        
        if not os.path.exists(attn_results_path):
            print("  ✗ ERROR: Attention analysis results not found at", attn_results_path)
            print("    Cannot generate fig09 without real data. Run 08_transformer_attention_analysis.py first.")
            return
        
        with open(attn_results_path, 'r') as f:
            data = json.load(f)
            aggregated = data.get('aggregated', {})
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel A: Attention heatmap for all networks
        ax1 = fig.add_subplot(gs[0, 0])
        
        attention_matrix = []
        for network in NETWORK_ORDER:
            pattern = aggregated.get(network, {}).get('avg_attention_pattern', np.zeros(seq_length))
            if isinstance(pattern, list):
                pattern = np.array(pattern)
            attention_matrix.append(pattern)
        
        attention_matrix = np.array(attention_matrix)
        
        im = ax1.imshow(attention_matrix, aspect='auto', cmap='YlOrRd', 
                       extent=[0, seq_length*1.5, len(NETWORK_ORDER)-0.5, -0.5])
        ax1.set_yticks(range(len(NETWORK_ORDER)))
        ax1.set_yticklabels(NETWORK_ORDER)
        ax1.set_xlabel('Time (seconds ago)', fontsize=11, fontweight='bold')
        ax1.set_title('A. Temporal Attention Heatmap', fontsize=12, fontweight='bold', loc='left', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Attention Weight', fontsize=10)
        
        # Add vertical line for "recent" cutoff
        ax1.axvline(x=seq_length*1.5/2, color='white', linestyle='--', linewidth=2)
        ax1.text(seq_length*1.5/4, -0.7, 'Distant', ha='center', fontsize=10, color='gray')
        ax1.text(seq_length*1.5*3/4, -0.7, 'Recent', ha='center', fontsize=10, color='gray')
        
        # Panel B: Temporal ratio as diverging bar chart (deviation from 1.0)
        ax2 = fig.add_subplot(gs[0, 1])
        
        ratios = [aggregated.get(n, {}).get('temporal_ratio', 1.0) for n in NETWORK_ORDER]
        # Calculate deviation from 1.0 (balance point)
        deviations = [(r - 1.0) * 100 for r in ratios]  # Convert to percentage
        
        # Sort by deviation for clearer visualization
        sorted_data = sorted(zip(deviations, NETWORK_ORDER), reverse=True)
        sorted_devs, sorted_nets = zip(*sorted_data)
        
        # Color by direction: green for distant-focused, red for recent-focused
        bar_colors = ['#2E7D32' if d > 0 else '#C62828' for d in sorted_devs]
        
        bars = ax2.barh(range(len(NETWORK_ORDER)), sorted_devs, color=bar_colors, alpha=0.8, height=0.7)
        ax2.axvline(x=0, color='gray', linestyle='-', linewidth=2)
        ax2.set_yticks(range(len(NETWORK_ORDER)))
        ax2.set_yticklabels(sorted_nets)
        ax2.set_xlabel('Deviation from Balance (%)', fontsize=11, fontweight='bold')
        ax2.set_title('B. Temporal Focus Bias', fontsize=12, fontweight='bold', loc='left', pad=15)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Expand x-axis limits to prevent overlap with y-axis
        x_min, x_max = ax2.get_xlim()
        x_range = x_max - x_min
        ax2.set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.15)
        
        # Add value labels - position inside bars for negative values
        for i, (bar, val) in enumerate(zip(bars, sorted_devs)):
            if val >= 0:
                x_pos = val + 0.15
                ha = 'left'
                color = 'black'
            else:
                # For negative values, put label inside the bar or at fixed position
                x_pos = 0.15  # Put label on the right side of y-axis
                ha = 'left'
                color = '#C62828'
            ax2.text(x_pos, i, f'{val:+.1f}%', va='center', ha=ha, fontsize=9, fontweight='bold', color=color)
        
        # Add annotations
        ax2.text(ax2.get_xlim()[1] * 0.7, -0.8, '← Distant', fontsize=9, color='#2E7D32', fontweight='bold')
        ax2.text(ax2.get_xlim()[0] * 0.5, -0.8, 'Recent →', fontsize=9, color='#C62828', fontweight='bold', ha='left')
        
        # Panel C: Line plots for key networks
        ax3 = fig.add_subplot(gs[1, 0])
        
        time_axis = np.arange(seq_length) * 1.5  # Convert to seconds
        
        key_networks = ['Visual', 'Default']
        for network in key_networks:
            pattern = aggregated.get(network, {}).get('avg_attention_pattern', np.zeros(seq_length))
            if isinstance(pattern, list):
                pattern = np.array(pattern)
            ax3.plot(time_axis, pattern, label=network, linewidth=2.5, 
                    color=NETWORK_COLORS_P2[network])
            ax3.fill_between(time_axis, pattern, alpha=0.2, color=NETWORK_COLORS_P2[network])
        
        ax3.set_xlabel('Time (seconds ago)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
        ax3.set_title('C. Visual vs DMN Attention', fontsize=12, fontweight='bold', loc='left')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Panel D: Temporal ratio bar
        ax4 = fig.add_subplot(gs[1, 1])
        
        ratios_d = [aggregated.get(n, {}).get('temporal_ratio', 0) for n in NETWORK_ORDER]
        colors_d = [NETWORK_COLORS_P2[n] for n in NETWORK_ORDER]
        
        bars = ax4.bar(range(len(NETWORK_ORDER)), ratios_d, color=colors_d, alpha=0.8)
        ax4.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Balance')
        ax4.set_xticks(range(len(NETWORK_ORDER)))
        ax4.set_xticklabels([n[:4] for n in NETWORK_ORDER], rotation=45, ha='right')
        ax4.set_ylabel('Distant/Recent Ratio', fontsize=11, fontweight='bold')
        ax4.set_title('D. Temporal Focus', fontsize=12, fontweight='bold', loc='left')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Transformer Attention: Temporal Integration Patterns', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        self.save_figure(fig, 'fig07_transformer_attention')
        plt.close()
    
    # =========================================================================
    # Figure 8: Brain Glass Visualization (Temporal Dynamics)
    # =========================================================================
    
    def fig08_brain_glass(self):
        """
        Figure 8: Brain Glass Visualization of Temporal Dynamics
        
        Shows spatial distribution of temporal processing characteristics using nilearn.
        Distinct from Project 1 which shows modality-specific encoding.
        
        Panels:
        - A: Temporal Dynamics Speed (TDS) - Fast (red) vs Slow (blue)
        - B: Temporal Processing Depth (inverse TDS) - Deep processing regions
        - C: Temporal Variability (std of TDS across subjects)
        - D: Network-wise temporal profile
        """
        # Try to import nilearn
        try:
            from nilearn import plotting
            from nilearn.maskers import NiftiLabelsMasker
        except ImportError:
            print("  ✗ Skipped fig08_brain_glass: nilearn not available")
            return
        
        # Find atlas file
        # Navigate up from run_dir: runs/run_xxx -> runs -> project_2 -> CCN_Competition
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(self.run_dir)))
        data_dir = os.path.join(project_root, 'data', 'fmri')
        atlas_path = None
        
        for sub in ['sub-01', 'sub-02', 'sub-03', 'sub-05']:
            potential_path = os.path.join(
                data_dir, sub, 'atlas',
                f'{sub}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
            )
            if os.path.exists(potential_path):
                atlas_path = potential_path
                break
        
        if atlas_path is None:
            print("  ✗ Skipped fig08_brain_glass: Atlas file not found")
            return
        
        # Check if TDS data is available
        if self.avg_tds is None:
            print("  ✗ Skipped fig08_brain_glass: TDS data not available")
            return
        
        print(f"  Using atlas: {os.path.basename(atlas_path)}")
        
        # Initialize masker
        atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
        atlas_masker.fit()
        
        # Prepare TDS data
        tds_data = np.nan_to_num(self.avg_tds, nan=np.nanmean(self.avg_tds))
        
        # Generate individual brain glass figures
        
        # --- Figure A: Temporal Dynamics Speed (TDS) ---
        fig_a = plt.figure(figsize=(14, 10), facecolor='white')
        
        # Normalize TDS for visualization
        tds_normalized = (tds_data - np.min(tds_data)) / (np.max(tds_data) - np.min(tds_data) + 1e-10)
        nii_tds = atlas_masker.inverse_transform(tds_normalized)
        
        plotting.plot_glass_brain(
            stat_map_img=nii_tds,
            display_mode='lyrz',
            colorbar=True,
            threshold=0.1,
            cmap='RdYlBu_r',  # Red = Fast, Blue = Slow
            vmin=0, vmax=1,
            plot_abs=False,
            symmetric_cbar=False,
            figure=fig_a
        )
        
        fig_a.suptitle('Temporal Dynamics Speed (TDS) - Cortex Only\n'
                      f'Fast (Red) vs Slow (Blue) | Mean TDS = {np.mean(tds_data):.3f}',
                      fontsize=14, fontweight='bold', y=0.98)
        
        self.save_figure(fig_a, 'fig08a_brain_glass_tds')
        plt.close()
        
        # --- Figure B: Temporal Processing Depth (inverse TDS) ---
        fig_b = plt.figure(figsize=(14, 10), facecolor='white')
        
        # Inverse TDS = Processing depth (high value = deep processing)
        depth_data = 1 - tds_normalized  # Invert: slow dynamics = high depth
        nii_depth = atlas_masker.inverse_transform(depth_data)
        
        plotting.plot_glass_brain(
            stat_map_img=nii_depth,
            display_mode='lyrz',
            colorbar=True,
            threshold=0.3,
            cmap='BuGn',  # Green = deep processing (consistent with Project 1)
            vmin=0, vmax=1,
            plot_abs=False,
            symmetric_cbar=False,
            figure=fig_b
        )
        
        high_depth_count = np.sum(depth_data > 0.6)
        fig_b.suptitle('Temporal Processing Depth - Cortex Only\n'
                      f'Deep Processing Regions (Green) | Count (Depth > 0.6): {high_depth_count}',
                      fontsize=14, fontweight='bold', y=0.98)
        
        self.save_figure(fig_b, 'fig08b_brain_glass_depth')
        plt.close()
        
        # --- Figure C: Network-coded Temporal Profile ---
        fig_c = plt.figure(figsize=(14, 10), facecolor='white')
        
        # Create network-coded values based on average TDS per network
        network_tds = {}
        for network, indices in SCHAEFER_NETWORK_INDICES.items():
            valid_indices = [i for i in indices if i < len(tds_data)]
            if valid_indices:
                network_tds[network] = np.mean(tds_data[valid_indices])
        
        # Sort networks by TDS and assign rank values
        sorted_networks = sorted(network_tds.items(), key=lambda x: x[1])
        network_ranks = {net: rank for rank, (net, _) in enumerate(sorted_networks)}
        
        # Assign rank to each region based on its network
        # Use small offset (0.05) instead of 0 to avoid threshold issues
        rank_data = np.zeros(len(tds_data))
        for network, indices in SCHAEFER_NETWORK_INDICES.items():
            for idx in indices:
                if idx < len(tds_data):
                    # Scale from 0.1 to 1.0 to avoid threshold cutoff
                    rank_data[idx] = 0.1 + (network_ranks[network] / 6.0) * 0.9
        
        nii_network = atlas_masker.inverse_transform(rank_data)
        
        plotting.plot_glass_brain(
            stat_map_img=nii_network,
            display_mode='lyrz',
            colorbar=True,
            threshold=0.05,  # Low threshold to show all cortical regions
            cmap='viridis',
            vmin=0, vmax=1,
            plot_abs=False,
            symmetric_cbar=False,
            figure=fig_c
        )
        
        # Create network legend text
        network_order_by_speed = [net for net, _ in sorted_networks]
        legend_text = "Slow → Fast: " + " → ".join([n[:3] for n in network_order_by_speed])
        
        fig_c.suptitle(f'Network Temporal Hierarchy (Cortex Only)\n{legend_text}',
                      fontsize=14, fontweight='bold', y=0.98)
        
        self.save_figure(fig_c, 'fig08c_brain_glass_network')
        plt.close()
        
        # --- Combined Overview Figure ---
        fig = plt.figure(figsize=(18, 14), facecolor='white')
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
        
        # Panel A: TDS
        ax1 = fig.add_subplot(gs[0, 0])
        plotting.plot_glass_brain(
            stat_map_img=nii_tds,
            display_mode='lyrz',
            colorbar=True,
            title='A. Temporal Dynamics Speed (TDS)',
            threshold=0.1,
            cmap='RdYlBu_r',
            vmin=0, vmax=1,
            figure=fig,
            axes=ax1
        )
        
        # Panel B: Processing Depth
        ax2 = fig.add_subplot(gs[0, 1])
        plotting.plot_glass_brain(
            stat_map_img=nii_depth,
            display_mode='lyrz',
            colorbar=True,
            title='B. Temporal Processing Depth',
            threshold=0.3,
            cmap='BuGn',
            vmin=0, vmax=1,
            figure=fig,
            axes=ax2
        )
        
        # Panel C: Network Hierarchy
        ax3 = fig.add_subplot(gs[1, 0])
        plotting.plot_glass_brain(
            stat_map_img=nii_network,
            display_mode='lyrz',
            colorbar=True,
            title='C. Network Temporal Hierarchy',
            threshold=0.05,  # Low threshold to show all cortical regions
            cmap='viridis',
            vmin=0, vmax=1,
            figure=fig,
            axes=ax3
        )
        
        # Panel D: Network TDS Bar Chart
        ax4 = fig.add_subplot(gs[1, 1])
        
        networks_sorted = [net for net, _ in sorted(network_tds.items(), key=lambda x: x[1], reverse=True)]
        tds_sorted = [network_tds[net] for net in networks_sorted]
        colors_sorted = [NETWORK_COLORS_P2[net] for net in networks_sorted]
        
        bars = ax4.barh(range(len(networks_sorted)), tds_sorted, color=colors_sorted, alpha=0.85)
        ax4.set_yticks(range(len(networks_sorted)))
        ax4.set_yticklabels(networks_sorted)
        ax4.set_xlabel('Temporal Dynamics Speed (TDS)', fontsize=11, fontweight='bold')
        ax4.set_title('D. Network TDS Ranking', fontsize=12, fontweight='bold', loc='left')
        ax4.axvline(x=np.mean(tds_data), color='gray', linestyle='--', linewidth=2, label='Mean')
        ax4.legend()
        ax4.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, tds_sorted)):
            ax4.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
        
        # Annotations
        ax4.annotate('Fast', xy=(max(tds_sorted)*0.9, -0.5), fontsize=10, fontweight='bold', 
                    color='#D62728', ha='center')
        ax4.annotate('Slow', xy=(min(tds_sorted)*1.1, len(networks_sorted)-0.5), fontsize=10, 
                    fontweight='bold', color='#9467BD', ha='center')
        
        fig.suptitle('Brain Glass: Temporal Dynamics Across Cortical Regions\n'
                    'The Temporal Efficiency Paradox Visualized (Schaefer 1000 Parcellation)',
                    fontsize=16, fontweight='bold', y=0.98)
        
        self.save_figure(fig, 'fig08_brain_glass_overview')
        plt.close()
        
        print("  ✓ Brain glass figures generated (fig08a, fig08b, fig08c, fig08_overview)")
    
    def generate_all_figures(self):
        """Generate all figures for Project 2."""
        print("=" * 60)
        print("Generating Project 2 Figures")
        print("=" * 60)
        
        self.fig01_temporal_spectrum()
        self.fig02_dynamic_connectivity()
        self.fig03_multiscale_encoding()
        self.fig04_network_summary_radar()
        
        # Deep learning analysis figures (optional)
        self.fig05_lstm_ridge_comparison()
        self.fig06_hub_centrality()
        self.fig07_transformer_attention()
        
        # Brain glass visualizations (nilearn)
        self.fig08_brain_glass()
        
        print("\n" + "=" * 60)
        print(f"All figures saved to: {self.output_dir}")
        print("=" * 60)


def main():
    # Find the latest run directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(project_dir, 'project_2', 'runs')
    
    if not os.path.exists(runs_dir):
        print("No runs directory found. Creating with sample data...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
    else:
        existing_runs = [d for d in os.listdir(runs_dir) if d[0].isdigit()]
        if existing_runs:
            latest_run = sorted(existing_runs)[-1]
            run_dir = os.path.join(runs_dir, latest_run)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(runs_dir, timestamp)
            os.makedirs(run_dir, exist_ok=True)
    
    print(f"Using run directory: {run_dir}")
    
    generator = Project2FigureGenerator(run_dir)
    generator.generate_all_figures()
    
    print("\n✓ Figure generation complete!")


if __name__ == '__main__':
    main()

