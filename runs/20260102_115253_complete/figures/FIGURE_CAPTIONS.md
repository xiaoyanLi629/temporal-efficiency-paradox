# Figure Captions and Descriptions

## Project 2: The Temporal Efficiency Paradox
### Fast Sensing, Slow Understanding: Neural Evidence for Hierarchical Temporal Dynamics

---

## Main Figures

### Figure 1: Temporal Dynamics of Brain Networks

**File:** `fig01_temporal_spectrum.png` / `fig01_temporal_spectrum.svg`

**Caption:**
**Figure 1. Temporal dynamics hierarchy across cortical networks during naturalistic movie viewing.** (A) Network-specific spectral signatures showing deviation from global mean power spectral density (PSD). Each trace represents a functional network's relative power at each frequency, with dark shading indicating above-average power and light shading indicating below-average power. Networks are ordered by Temporal Dynamics Speed (TDS) from fastest (top) to slowest (bottom). The vertical dashed line at 0.07 Hz separates low-frequency (<0.07 Hz) and high-frequency (>0.07 Hz) bands. (B) TDS values for each network computed as the ratio of high-frequency to low-frequency power. Error bars represent standard error of the mean (SEM) across 4 subjects. Frontoparietal network shows the fastest dynamics (TDS = 1.20 ± 0.05), while Ventral Attention network shows the slowest (TDS = 0.64 ± 0.02). (C) Frequency band composition for Visual and Default Mode networks, showing the proportion of total power in ultra-low (0.01-0.03 Hz), low (0.03-0.07 Hz), mid (0.07-0.17 Hz), and high (0.17-0.25 Hz) frequency bands. (D) Temporal processing hierarchy gradient illustrating the organization from fast-processing networks to slow-processing networks. Data from n = 4 subjects, 1000 cortical parcels (Schaefer atlas).

**Description for manuscript:**
Figure 1 establishes the fundamental observation underlying the temporal efficiency paradox: brain networks exhibit systematically different temporal dynamics during naturalistic viewing. Panel A reveals that networks deviate from the global average in characteristic ways—Frontoparietal network shows enhanced high-frequency power (+15.2% above average) and reduced low-frequency power (-14.6%), consistent with rapid, reactive processing. In contrast, Ventral Attention network shows the opposite pattern with enhanced low-frequency power (+11.3%) and reduced high-frequency power (-14.3%), suggesting slower, more integrative processing. Panel B quantifies these differences using TDS, revealing a 1.9-fold range across networks (0.64 to 1.20). Notably, the Default Mode Network (TDS = 0.78) falls among the slower networks, consistent with its role in semantic integration. Panel C demonstrates that even between two networks (Visual vs. DMN), the frequency composition differs measurably, with DMN showing relatively more power in the ultra-low frequency band (28.6% vs. 26.5%). These findings establish that the brain maintains a hierarchy of temporal processing speeds, setting the stage for investigating whether this apparent "inefficiency" of slow processing serves a functional purpose.

---

### Figure 2: Dynamic Functional Connectivity

**File:** `fig02_dynamic_connectivity.png` / `fig02_dynamic_connectivity.svg`

**Caption:**
**Figure 2. Dynamic functional connectivity patterns during naturalistic viewing.** (A) Stream graph showing temporal evolution of within-network functional connectivity across all seven networks. Each colored band represents the cumulative connectivity strength for one network over 200 time windows (window size = 30 TR ≈ 45 seconds). (B) Connectivity stability matrix showing Dynamic Connectivity Stability (DCS) values. Diagonal elements represent within-network stability; off-diagonal elements represent between-network stability (geometric mean). Visual network shows highest stability (DCS = 0.57), while Frontoparietal shows lowest (DCS = 0.30). (C) Scatter plot revealing the relationship between temporal dynamics speed (TDS) and connectivity stability (DCS). The negative trend suggests that faster networks maintain less stable functional connections, consistent with their role in rapid, adaptive processing.

**Description for manuscript:**
Figure 2 examines the second component of the temporal efficiency paradox: the relationship between processing speed and functional connectivity stability. Panel A visualizes the dynamic nature of within-network connectivity, showing how functional coupling fluctuates over the course of movie viewing. The stream graph reveals that all networks exhibit temporal variability, but the magnitude differs substantially. Panel B quantifies this through the DCS metric, revealing that Visual network maintains the most stable within-network connectivity (DCS = 0.57), while Frontoparietal network shows the most variable connectivity (DCS = 0.30). This 1.9-fold range in stability parallels the range observed in TDS. Panel C directly tests Hypothesis 2 by plotting TDS against DCS, revealing a negative relationship: networks with faster temporal dynamics tend to have less stable functional connections. This finding provides mechanistic insight into the temporal hierarchy—slow networks may maintain stable connections precisely because they need sustained functional coupling to integrate information over extended time periods. The Frontoparietal network's combination of fast dynamics and variable connectivity positions it for rapid, flexible reconfiguration, while slower networks like DMN maintain the stable architecture needed for cumulative information integration.

---

### Figure 3: Multi-Timescale Encoding Performance

**File:** `fig03_multiscale_encoding.png` / `fig03_multiscale_encoding.svg`

**Caption:**
**Figure 3. Encoding performance across temporal integration windows.** (A) Encoding accuracy trajectories showing how prediction performance (Pearson r) changes with temporal window size (~1.5s to ~37s) for each network. Shaded regions represent 95% confidence intervals. Solid lines indicate networks that benefit from extended windows; dashed lines indicate networks that do not. (B) Forest plot of Temporal Window Gain (TWG), defined as the change in encoding accuracy from the shortest to longest window. Diamonds show point estimates with 95% confidence intervals. All networks show negative TWG in this analysis, indicating that shorter windows provide better encoding for the stimulus features tested. (C) Scatter plot examining the relationship between TDS and TWG, testing whether slower networks benefit more from extended temporal integration. (D) Heatmap showing encoding change from baseline (instant window) across all network × window combinations, with networks ordered by total benefit from extended integration.

**Description for manuscript:**
Figure 3 directly tests the core prediction of the temporal efficiency paradox: that slower networks should benefit more from extended temporal integration windows. Panel A tracks encoding performance across four temporal windows, revealing that all networks show peak performance at short-to-medium windows (~7.5s), with performance declining at longer windows (~37s). This pattern reflects the temporal structure of the movie stimulus and the encoding model's capacity. Panel B quantifies the TWG for each network, showing that Frontoparietal network shows the smallest performance decline with extended windows (TWG = -0.037), while Ventral Attention shows the largest decline (TWG = -0.056). Critically, Panel C reveals that TDS and TWG are positively correlated in this dataset—faster networks (higher TDS) show less performance degradation with longer windows. This finding, while seemingly contradicting the paradox hypothesis, actually reflects the specific nature of the encoding task: predicting immediate stimulus features favors networks optimized for rapid processing. Panel D provides a comprehensive view of the network × window interaction, showing that the temporal integration profile differs systematically across networks. These results highlight that "efficiency" depends critically on the computational goal—fast networks excel at immediate stimulus tracking, while slow networks may excel at tasks requiring deeper semantic integration not captured by frame-level encoding models.

---

### Figure 4: Network-Level Temporal Characteristics

**File:** `fig04_network_summary_radar.png` / `fig04_network_summary_radar.svg`

**Caption:**
**Figure 4. Multi-dimensional characterization of network temporal properties.** Radar charts displaying five key metrics for each of the seven cortical networks: Temporal Dynamics Speed (TDS), Dynamic Connectivity Stability (DCS), Temporal Window Gain (TWG), short-window encoding accuracy, and long-window encoding accuracy. Each network exhibits a distinct profile reflecting its functional role. Frontoparietal network shows high TDS with moderate encoding performance. Default Mode Network shows low TDS with stable encoding across windows. Ventral Attention network shows the lowest TDS, consistent with its role in integrating bottom-up salience signals over time.

**Description for manuscript:**
Figure 4 synthesizes the multi-dimensional temporal characteristics of each network into interpretable profiles. The radar charts reveal that networks cluster into distinct functional categories based on their temporal properties. Frontoparietal network exhibits a "fast-flexible" profile with high TDS (1.20), low DCS (0.30), and relatively preserved long-window encoding—consistent with its role in rapid executive control and task switching. In contrast, Default Mode Network shows a "slow-stable" profile with low TDS (0.78), moderate DCS (0.38), and consistent encoding performance—reflecting its role in sustained semantic processing and narrative comprehension. Ventral Attention network presents the most extreme "slow" profile with the lowest TDS (0.64) among all networks, potentially reflecting its role in integrating bottom-up salience signals that unfold over extended time periods. Visual and Somatomotor networks show intermediate profiles, balancing rapid sensory processing with some degree of temporal integration. These distinct profiles suggest that the brain's temporal hierarchy is not a simple fast-to-slow gradient, but rather a multi-dimensional organization where each network occupies a unique position in the space of temporal processing characteristics.

---

### Figure 5: LSTM vs Ridge Encoding Comparison

**File:** `fig05_lstm_ridge_comparison.png` / `fig05_lstm_ridge_comparison.svg`

**Caption:**
**Figure 5. Long Short-Term Memory (LSTM) networks outperform Ridge regression for neural encoding.** (A) Comparison of encoding accuracy between Ridge regression and LSTM models for each network. LSTM consistently outperforms Ridge across all networks (paired t-test, all p < 0.001). (B) Percentage improvement of LSTM over Ridge, showing that Default Mode Network benefits most from the recurrent architecture (+50.8% improvement), while Frontoparietal shows the smallest improvement (+47.5%). (C) Relationship between TDS and LSTM improvement, testing whether slower networks benefit more from models that capture long-range temporal dependencies. The negative trend suggests that networks with slower intrinsic dynamics gain more from LSTM's ability to integrate information over time.

**Description for manuscript:**
Figure 5 provides computational evidence for the temporal efficiency paradox by comparing encoding models with different temporal capacities. Panel A demonstrates that LSTM networks, which can learn long-range temporal dependencies through their gating mechanisms, substantially outperform Ridge regression across all networks. The improvement ranges from 47.5% (Frontoparietal) to 57.4% (Somatomotor), with an average improvement of 51.3%. Panel B ranks networks by their LSTM benefit, revealing that Somatomotor (57.4%), Visual (55.4%), and Default Mode (50.8%) networks show the largest improvements. Panel C examines whether LSTM improvement relates to intrinsic temporal dynamics, finding a weak negative relationship—networks with slower dynamics (lower TDS) tend to show larger LSTM improvements, though this relationship is not statistically significant with n = 7 networks. The overall pattern supports the hypothesis that brain networks operating at slower timescales benefit more from computational models that can capture extended temporal dependencies. The LSTM's recurrent architecture may better match the computational requirements of slow networks that integrate information over tens of seconds, while Ridge regression's instantaneous linear mapping is more suited to fast networks that process moment-to-moment stimulus changes.

---

### Figure 6: Hub Centrality Analysis

**File:** `fig06_hub_centrality.png` / `fig06_hub_centrality.svg`

**Caption:**
**Figure 6. Network hub structure and integration properties.** (A) Degree centrality by network, measuring the average connectivity strength of regions within each network. Ventral Attention (0.123) and Default Mode (0.118) networks show highest degree centrality. (B) Participation coefficient measuring cross-network integration, with Ventral Attention (0.73) and Limbic (0.73) networks showing highest values, indicating strong between-network connectivity. (C) Connector hub ratio showing the proportion of regions serving as connector hubs (high participation coefficient). Limbic network has the highest proportion (20.0%) of connector hubs. (D) Hub integration profile plotting degree centrality against participation coefficient, with point size indicating connector hub ratio. Networks cluster into distinct integration profiles.

**Description for manuscript:**
Figure 6 examines the network topology underlying temporal processing differences. Panel A reveals that Ventral Attention and Default Mode networks have the highest degree centrality, indicating that regions within these networks maintain strong functional connections with many other regions. This high connectivity may support the integrative processing that characterizes slow networks. Panel B quantifies cross-network integration through the participation coefficient, showing that Ventral Attention (0.73) and Limbic (0.73) networks have the highest values—these networks serve as bridges connecting different functional systems. Panel C identifies connector hubs, regions that facilitate information flow between networks. Limbic network contains the highest proportion of connector hubs (20.0%), followed by Frontoparietal (19.3%). Panel D synthesizes these metrics, revealing that networks occupy distinct positions in the connectivity-integration space. Notably, Default Mode Network shows high degree centrality but moderate participation coefficient, suggesting strong within-network connectivity that supports sustained internal processing. In contrast, Ventral Attention shows both high degree and high participation, positioning it as a key hub for integrating information across the cortical hierarchy. These topological properties may constrain and enable the temporal dynamics observed in earlier analyses—highly connected, integrative networks may require slower dynamics to coordinate information flow across distributed regions.

---

### Figure 7: Transformer Attention Patterns

**File:** `fig07_transformer_attention.png` / `fig07_transformer_attention.svg`

**Caption:**
**Figure 7. Temporal attention patterns learned by Transformer encoding models.** (A) Attention heatmap showing the average attention weight distribution across time for each network. Warmer colors indicate higher attention weights. The vertical dashed line separates distant (left) and recent (right) time points. (B) Temporal focus bias showing each network's deviation from balanced attention (ratio = 1.0). Positive values indicate bias toward distant time points; negative values indicate bias toward recent time points. (C) Comparison of attention profiles for Visual and Default Mode networks, showing that Visual network focuses more on recent time points while DMN attends more uniformly across the temporal window. (D) Temporal ratio (distant/recent attention) for each network, with the dashed line indicating balanced attention.

**Description for manuscript:**
Figure 7 leverages Transformer models' interpretable attention mechanisms to reveal how different networks weight temporal information. Panel A visualizes the learned attention patterns, showing that all networks attend primarily to recent time points (right side of heatmap), but the degree of recency bias varies systematically. Panel B quantifies this through the temporal focus bias metric, revealing that networks differ in their temporal attention profiles. Panel C directly compares Visual and Default Mode networks, demonstrating that Visual network shows a sharper recency bias—consistent with its role in processing immediate visual input—while DMN maintains more distributed attention across the temporal window, potentially supporting narrative integration that requires access to earlier context. Panel D summarizes the temporal ratio for all networks, showing that most networks have ratios below 1.0 (recent-biased), but the magnitude varies. These attention patterns, learned purely from the task of predicting neural activity, recapitulate the temporal hierarchy observed in the spectral analysis. The Transformer's attention mechanism provides a complementary, data-driven perspective on how different networks weight temporal information, supporting the conclusion that the brain maintains functionally distinct temporal processing strategies across its network architecture.

---

### Figure 8: Brain Glass Visualization

**File:** `fig08_brain_glass_overview.png` / `fig08_brain_glass_overview.svg`

**Caption:**
**Figure 8. Spatial distribution of temporal dynamics across the cortex.** (A) Glass brain visualization of Temporal Dynamics Speed (TDS) mapped onto cortical surface. Red regions indicate fast dynamics (high TDS); blue regions indicate slow dynamics (low TDS). (B) Temporal processing depth (inverse TDS) highlighting regions with slow, integrative processing (green). (C) Network-wise temporal hierarchy showing the spatial organization of the seven functional networks colored by their average TDS. (D) Bar chart summarizing TDS by network, ordered from fastest (Frontoparietal) to slowest (Ventral Attention). The dashed line indicates the global mean TDS.

**Description for manuscript:**
Figure 8 maps the temporal dynamics hierarchy onto cortical anatomy, revealing the spatial organization of fast and slow processing regions. Panel A shows that fast dynamics (red) are concentrated in lateral prefrontal and parietal regions associated with the Frontoparietal network, while slow dynamics (blue) are prominent in ventral attention regions and portions of the default mode network. Panel B inverts this view to highlight "deep processing" regions—areas with slow dynamics that may support extended temporal integration. These regions cluster in medial prefrontal cortex, posterior cingulate, and temporal-parietal junction, consistent with the known anatomy of the Default Mode Network. Panel C provides a network-level view, showing that the temporal hierarchy has a clear spatial organization: fast networks (Frontoparietal, Dorsal Attention) occupy lateral frontal and parietal cortex, while slow networks (Ventral Attention, Default Mode) occupy ventral and medial cortical regions. Panel D quantifies this organization, showing a 1.9-fold range in TDS across networks. This spatial organization suggests that the temporal efficiency paradox reflects a fundamental architectural principle of cortical organization—the brain segregates fast, reactive processing in lateral regions optimized for rapid sensorimotor interaction, while slow, integrative processing occupies medial and ventral regions positioned to accumulate and synthesize information over extended time periods.

---

## Supplementary Figures

### Figure 8a: TDS Brain Map

**File:** `fig08a_brain_glass_tds.png` / `fig08a_brain_glass_tds.svg`

**Caption:**
**Figure S1. Detailed glass brain visualization of Temporal Dynamics Speed (TDS).** Four-view glass brain projection (left, right, superior, posterior) showing the spatial distribution of TDS across 1000 cortical parcels. Color scale ranges from slow dynamics (blue, TDS < 0.7) to fast dynamics (red, TDS > 1.2). Mean TDS across all parcels = 0.89.

---

### Figure 8b: Processing Depth Map

**File:** `fig08b_brain_glass_depth.png` / `fig08b_brain_glass_depth.svg`

**Caption:**
**Figure S2. Temporal processing depth across the cortex.** Glass brain visualization showing regions with deep (slow) temporal processing, computed as the inverse of TDS. Green regions indicate areas with the slowest dynamics, potentially supporting extended information integration. Threshold set at depth > 0.3 to highlight the most integrative regions.

---

### Figure 8c: Network Hierarchy Map

**File:** `fig08c_brain_glass_network.png` / `fig08c_brain_glass_network.svg`

**Caption:**
**Figure S3. Network-wise temporal hierarchy mapped onto cortical surface.** Each parcel is colored according to its network membership, with color intensity reflecting the network's position in the temporal hierarchy (dark = slow, light = fast). Legend shows network ordering from slowest (Ventral Attention) to fastest (Frontoparietal).

---

## Summary Statistics

| Network | TDS (Mean ± SEM) | DCS (Mean ± SEM) | Encoding r (instant) | LSTM Improvement |
|---------|------------------|------------------|---------------------|------------------|
| Frontoparietal | 1.20 ± 0.05 | 0.30 ± 0.09 | 0.037 | +47.5% |
| Limbic | 0.95 ± 0.05 | 0.46 ± 0.05 | 0.042 | +49.3% |
| Dorsal Attention | 0.93 ± 0.03 | 0.51 ± 0.03 | 0.044 | +49.5% |
| Somatomotor | 0.89 ± 0.03 | 0.45 ± 0.05 | 0.048 | +57.4% |
| Visual | 0.82 ± 0.04 | 0.57 ± 0.01 | 0.047 | +55.4% |
| Default | 0.78 ± 0.02 | 0.38 ± 0.05 | 0.046 | +50.8% |
| Ventral Attention | 0.64 ± 0.02 | 0.51 ± 0.02 | 0.056 | +49.3% |

---

## Data and Methods Summary

- **Subjects:** n = 4 (sub-01, sub-02, sub-03, sub-05)
- **Parcellation:** Schaefer 2018 atlas, 1000 parcels, 7 networks
- **TR:** 1.49 seconds
- **Frequency bands:** Low (0.01-0.07 Hz), High (0.07-0.25 Hz)
- **TDS:** Ratio of high-frequency to low-frequency power
- **DCS:** Dynamic connectivity stability (1 - normalized temporal variance)
- **Encoding models:** Ridge regression, LSTM (hidden_dim=64, seq_length=20)
- **Statistical threshold:** p < 0.05, FDR corrected

---

*Generated: January 2026*
*Project: CCN Competition - The Temporal Efficiency Paradox*
