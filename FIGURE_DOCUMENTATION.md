# Figure Documentation - Project 2

## The Temporal Efficiency Paradox: Fast Brains React, Slow Brains Understand

This document describes all figures in Project 2, their scientific meaning, and how they should be organized into composite figures for publication.

---

## 完整图表清单 (Complete Figure List)

### 主图 (Main Figures)

| 图号 | 文件名 | 标题 | 类型 | 数据来源 |
|------|--------|------|------|----------|
| Fig 1 | fig01_temporal_hierarchy.png | Temporal Dynamics Hierarchy | 组合图 | TDS分析 |
| Fig 2 | fig02_dynamic_connectivity.png | Dynamic Functional Connectivity | 组合图 | DCS分析 |
| Fig 3 | fig03_encoding_performance.png | Multi-Timescale Encoding | 组合图 | 编码分析 |
| Fig 4 | fig04_temporal_paradox.png | The Temporal Efficiency Paradox | 组合图 | 综合分析 |
| Fig 5 | fig05_lstm_comparison.png | LSTM vs Ridge Encoding | 组合图 | AI模型分析 |
| Fig 6 | fig06_transformer_attention.png | Transformer Attention Patterns | 组合图 | AI模型分析 |
| Fig 7 | fig07_brain_visualization.png | Brain Surface Visualization | 组合图 | 空间分析 |
| Fig 8 | fig08_network_profiles.png | Network Characterization | 组合图 | 综合分析 |

### 补充图 (Supplementary Figures)

| 图号 | 文件名 | 标题 | 类型 | 数据来源 |
|------|--------|------|------|----------|
| S1 | figS01_data_quality.png | Data Quality & Coverage | 组合图 | 数据检查 |
| S2 | figS02_individual_variation.png | Individual Subject Variation | 组合图 | 个体分析 |
| S3 | figS03_sensitivity_analysis.png | Sensitivity Analysis | 组合图 | 敏感性分析 |
| S4 | figS04_statistical_tests.png | Statistical Test Results | 组合图 | 统计检验 |
| S5 | figS05_model_diagnostics.png | Model Diagnostics | 组合图 | 模型诊断 |
| S6 | figS06_feature_analysis.png | Stimulus Feature Analysis | 组合图 | 特征分析 |
| S7 | figS07_hub_centrality.png | Hub Centrality Analysis | 组合图 | 网络分析 |

---

## 主图详细设计 (Main Figure Designs)

### Figure 1: Temporal Dynamics Hierarchy
**文件**: `fig01_temporal_hierarchy.png`
**布局**: 2×2 网格 (4 panels)

```
┌─────────────────────────────────────┐
│  A. Power Spectral Density          │  B. TDS by Network
│     (Ridgeline Plot)                │     (Bar Chart + CI)
│                                     │
├─────────────────────────────────────┤
│  C. Frequency Band Composition      │  D. Temporal Hierarchy
│     (Stacked Bar Chart)             │     Gradient Arrow
│                                     │
└─────────────────────────────────────┘
```

**Panel A**: Ridgeline Plot
- X轴: 频率 (0.01-0.25 Hz)
- Y轴: 7个网络堆叠
- 垂直虚线: 0.07 Hz分界
- 颜色: 网络特定颜色

**Panel B**: TDS Bar Chart with Error Bars
- 水平条形图
- 95%置信区间
- 按TDS排序
- 颜色渐变: 快(绿)→慢(红)

**Panel C**: Frequency Band Composition
- 堆叠条形图
- 4个频带: ultra-low, low, mid, high
- 每个网络一条

**Panel D**: Hierarchy Gradient
- 箭头: Fast → Slow
- 网络标签沿梯度分布

---

### Figure 2: Dynamic Functional Connectivity
**文件**: `fig02_dynamic_connectivity.png`
**布局**: 2×2 网格

```
┌─────────────────────────────────────┐
│  A. FC Time Series                  │  B. DCS Matrix
│     (Stream Graph)                  │     (Heatmap)
│                                     │
├─────────────────────────────────────┤
│  C. TDS vs DCS                      │  D. FC Variability
│     (Scatter + Regression)          │     (Violin Plot)
│                                     │
└─────────────────────────────────────┘
```

**Panel A**: Stream Graph
- X轴: 时间 (TRs)
- Y轴: 堆叠FC值
- 每个网络一层

**Panel B**: DCS Matrix Heatmap
- 7×7 网络矩阵
- 对角线: within-network DCS
- 非对角线: between-network DCS

**Panel C**: Scatter Plot
- X轴: TDS
- Y轴: DCS
- 回归线 + 置信带
- 标注相关系数

**Panel D**: Violin Plot
- 每个网络的FC变异分布
- 显示中位数和四分位数

---

### Figure 3: Multi-Timescale Encoding Performance
**文件**: `fig03_encoding_performance.png`
**布局**: 2×3 网格

```
┌─────────────────────────────────────────────────┐
│  A. Encoding Accuracy    │  B. TWG Forest Plot  │
│     by Window Size       │     with 95% CI      │
│     (Line Plot)          │                      │
├─────────────────────────────────────────────────┤
│  C. TDS vs TWG           │  D. Encoding Heatmap │
│     (Joint Plot)         │     (Clustered)      │
├─────────────────────────────────────────────────┤
│  E. Ridge Coefficients   │  F. Prediction       │
│     (Heatmap)            │     Examples         │
└─────────────────────────────────────────────────┘
```

**Panel A**: Line Plot with CI
- X轴: 时间窗口 (instant → very_long)
- Y轴: 编码准确度 (r)
- 每个网络一条线
- 阴影: 95% CI

**Panel B**: Forest Plot
- TWG点估计 + 95% CI
- 垂直线: TWG=0
- 按TWG排序

**Panel C**: Joint Plot
- 散点图 + 边缘分布
- X: TDS, Y: TWG
- 回归线

**Panel D**: Clustered Heatmap
- 行: 网络 (聚类)
- 列: 时间窗口
- 值: 编码准确度
- 树状图

**Panel E**: Ridge Coefficients
- 热图显示特征权重
- 行: 特征维度
- 列: 网络

**Panel F**: Prediction Examples
- 3个代表性区域
- 真实值 vs 预测值

---

### Figure 4: The Temporal Efficiency Paradox
**文件**: `fig04_temporal_paradox.png`
**布局**: 概念图 + 数据支持

```
┌─────────────────────────────────────────────────┐
│  A. Conceptual Diagram                          │
│     (Fast vs Slow Processing)                   │
├─────────────────────────────────────────────────┤
│  B. Paradox Evidence      │  C. Network Radar   │
│     (Multi-panel scatter)  │     Charts          │
├─────────────────────────────────────────────────┤
│  D. Summary Statistics Table                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Panel A**: Conceptual Diagram
- 左侧: Fast Brain (React)
- 右侧: Slow Brain (Understand)
- 中间: 箭头和关键指标

**Panel B**: Multi-scatter
- TDS vs TWG
- DCS vs TWG
- TDS vs DCS
- 颜色: 网络

**Panel C**: Radar Charts
- 每个网络一个雷达图
- 5个维度: TDS, DCS, TWG, Instant_r, VeryLong_r

**Panel D**: Summary Table
- 网络 | TDS | DCS | TWG | Category

---

### Figure 5: LSTM vs Ridge Encoding Comparison
**文件**: `fig05_lstm_comparison.png`
**布局**: 2×2 网格

```
┌─────────────────────────────────────┐
│  A. Model Architecture              │  B. Performance
│     Diagram                         │     Comparison
│                                     │     (Grouped Bar)
├─────────────────────────────────────┤
│  C. LSTM Improvement                │  D. Learning Curves
│     by Network                      │     (Loss over epochs)
│     (Lollipop Chart)                │
└─────────────────────────────────────┘
```

**Panel A**: Architecture Diagram
- LSTM结构示意图
- 输入 → LSTM → FC → 输出

**Panel B**: Grouped Bar Chart
- 每个网络: Ridge vs LSTM
- 误差条

**Panel C**: Lollipop Chart
- LSTM相对Ridge的改进百分比
- 按改进幅度排序

**Panel D**: Learning Curves
- X轴: Epoch
- Y轴: Loss
- 训练集 vs 验证集

---

### Figure 6: Transformer Attention Patterns
**文件**: `fig06_transformer_attention.png`
**布局**: 2×2 网格

```
┌─────────────────────────────────────┐
│  A. Attention Heatmap               │  B. Attention Span
│     (Time × Time)                   │     by Network
│                                     │     (Bar Chart)
├─────────────────────────────────────┤
│  C. Temporal Attention              │  D. Attention vs TDS
│     Profile                         │     (Scatter)
│     (Line Plot)                     │
└─────────────────────────────────────┘
```

**Panel A**: Attention Heatmap
- X轴: Query时间点
- Y轴: Key时间点
- 颜色: 注意力权重

**Panel B**: Attention Span Bar Chart
- 每个网络的有效注意力范围
- 定义: 累积注意力达到50%的时间距离

**Panel C**: Temporal Attention Profile
- X轴: 时间延迟 (TRs)
- Y轴: 平均注意力权重
- 每个网络一条线

**Panel D**: Scatter Plot
- X轴: TDS
- Y轴: Attention Span
- 预期: 负相关

---

### Figure 7: Brain Surface Visualization
**文件**: `fig07_brain_visualization.png`
**布局**: 3×2 网格

```
┌─────────────────────────────────────────────────┐
│  A. TDS Map                │  B. DCS Map        │
│     (Glass Brain)          │     (Glass Brain)  │
├─────────────────────────────────────────────────┤
│  C. TWG Map                │  D. Network Labels │
│     (Glass Brain)          │     (Glass Brain)  │
├─────────────────────────────────────────────────┤
│  E. Cortical Surface       │  F. Network        │
│     (Lateral View)         │     Boundaries     │
└─────────────────────────────────────────────────┘
```

**Panel A-D**: Glass Brain Views
- 使用nilearn
- 不同指标的空间分布

**Panel E**: Surface Plot
- 侧面视图
- TDS颜色映射

**Panel F**: Network Boundaries
- 7个网络的边界
- 颜色编码

---

### Figure 8: Network Characterization
**文件**: `fig08_network_profiles.png`
**布局**: 网络特定子图

```
┌─────────────────────────────────────────────────┐
│  A. Network Summary        │  B. Correlation    │
│     Table                  │     Matrix         │
├─────────────────────────────────────────────────┤
│  C. PCA of Network         │  D. t-SNE of       │
│     Features               │     Network        │
│                            │     Features       │
├─────────────────────────────────────────────────┤
│  E. Network Hierarchy      │  F. Effect Size    │
│     Dendrogram             │     Comparison     │
└─────────────────────────────────────────────────┘
```

**Panel A**: Summary Table
- 所有指标的网络级汇总

**Panel B**: Correlation Matrix
- 指标间相关性
- TDS, DCS, TWG, encoding等

**Panel C**: PCA Plot
- 网络在PC1-PC2空间的分布
- 显示方差解释比例

**Panel D**: t-SNE Plot
- 网络的非线性降维
- 聚类可视化

**Panel E**: Dendrogram
- 基于多指标的网络聚类

**Panel F**: Cohen's d Effect Sizes
- 网络间差异的效应量

---

## 补充图详细设计 (Supplementary Figure Designs)

### Figure S1: Data Quality & Coverage
```
┌─────────────────────────────────────────────────┐
│  A. Episode Coverage       │  B. Missing Data   │
│     per Subject            │     Pattern        │
├─────────────────────────────────────────────────┤
│  C. Temporal SNR           │  D. Motion         │
│     Distribution           │     Parameters     │
└─────────────────────────────────────────────────┘
```

### Figure S2: Individual Subject Variation
```
┌─────────────────────────────────────────────────┐
│  A. TDS by Subject         │  B. DCS by Subject │
├─────────────────────────────────────────────────┤
│  C. ICC Analysis           │  D. Subject        │
│                            │     Clustering     │
└─────────────────────────────────────────────────┘
```

### Figure S3: Sensitivity Analysis
```
┌─────────────────────────────────────────────────┐
│  A. Frequency Cutoff       │  B. Window Size    │
│     Sensitivity            │     Sensitivity    │
├─────────────────────────────────────────────────┤
│  C. Ridge Alpha            │  D. Bootstrap      │
│     Sensitivity            │     Stability      │
└─────────────────────────────────────────────────┘
```

### Figure S4: Statistical Test Results
```
┌─────────────────────────────────────────────────┐
│  A. Permutation Test       │  B. FDR Correction │
│     Results                │     Results        │
├─────────────────────────────────────────────────┤
│  C. Effect Size Matrix     │  D. Power Analysis │
└─────────────────────────────────────────────────┘
```

### Figure S5: Model Diagnostics
```
┌─────────────────────────────────────────────────┐
│  A. Residual Distribution  │  B. Q-Q Plot       │
├─────────────────────────────────────────────────┤
│  C. Prediction vs Actual   │  D. Cross-         │
│     Scatter                │     Validation     │
│                            │     Folds          │
└─────────────────────────────────────────────────┘
```

### Figure S6: Stimulus Feature Analysis
```
┌─────────────────────────────────────────────────┐
│  A. Feature PCA            │  B. Feature        │
│     (Visual/Audio/Lang)    │     Correlation    │
├─────────────────────────────────────────────────┤
│  C. Feature Time Series    │  D. Modality       │
│     Examples               │     Contribution   │
└─────────────────────────────────────────────────┘
```

### Figure S7: Hub Centrality Analysis
```
┌─────────────────────────────────────────────────┐
│  A. Degree Centrality      │  B. Betweenness    │
│     by Network             │     Centrality     │
├─────────────────────────────────────────────────┤
│  C. Participation          │  D. Hub Regions    │
│     Coefficient            │     Brain Map      │
└─────────────────────────────────────────────────┘
```

---

## 图表组合逻辑 (Figure Composition Logic)

### 按实验流程组织

```
实验流程                    对应图表
─────────────────────────────────────────
1. 数据质量检查        →    Figure S1
2. TDS分析             →    Figure 1 (A-D)
3. DCS分析             →    Figure 2 (A-D)
4. 编码模型分析        →    Figure 3 (A-F)
5. LSTM对比分析        →    Figure 5 (A-D)
6. Transformer分析     →    Figure 6 (A-D)
7. 综合悖论分析        →    Figure 4 (A-D)
8. 空间可视化          →    Figure 7 (A-F)
9. 网络特征总结        →    Figure 8 (A-F)
10. 统计检验           →    Figure S4
11. 敏感性分析         →    Figure S3
12. 个体差异           →    Figure S2
```

### 按科学问题组织

```
科学问题                    支持图表
─────────────────────────────────────────
Q1: 时间动态层级存在吗？   →    Fig 1, Fig 7A
Q2: 慢网络连接更稳定吗？   →    Fig 2, Fig 7B
Q3: 慢网络从长窗口获益吗？ →    Fig 3, Fig 5, Fig 6
Q4: 悖论的证据是什么？     →    Fig 4, Fig 8
```

---

## 技术规格 (Technical Specifications)

### 图像参数
- **分辨率**: 600 DPI (publication quality)
- **格式**: PNG + SVG (矢量)
- **字体**: Arial/Helvetica, 10-12pt
- **线宽**: 1-2pt
- **颜色**: 色盲友好调色板

### 网络颜色方案
```python
NETWORK_COLORS = {
    'Visual': '#E31A1C',           # Red
    'Somatomotor': '#FF7F00',      # Orange  
    'DorsalAttention': '#33A02C',  # Green
    'VentralAttention': '#1F78B4', # Blue
    'Limbic': '#6A3D9A',           # Purple
    'Frontoparietal': '#B15928',   # Brown
    'Default': '#666666'           # Gray
}
```

### 标准尺寸
- **单栏图**: 3.5" × 3.5" (89mm × 89mm)
- **1.5栏图**: 5.5" × 4" (140mm × 100mm)
- **双栏图**: 7" × 5" (180mm × 130mm)
- **全页图**: 7" × 9" (180mm × 230mm)

---

## 图表生成命令 (Figure Generation Commands)

```bash
# 完整分析和图表生成
cd /root/autodl-fs/CCN_Competition/project_2
python run_complete_analysis.py

# 仅生成图表（使用已有结果）
python generate_all_figures.py --run_dir runs/YYYYMMDD_HHMMSS

# 生成单个图表
python generate_figure.py --figure fig01 --run_dir runs/YYYYMMDD_HHMMSS
```

---

## 更新日志

- **2026-01-01**: 添加完整图表清单和组合逻辑
- **2024-12-14**: 初始版本

---

*Document: FIGURE_DOCUMENTATION.md*
*Project: CCN Competition - Temporal Efficiency Paradox*
