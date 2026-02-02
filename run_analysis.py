#!/usr/bin/env python3
"""
Project 2: Temporal Efficiency Paradox - Unified Analysis Pipeline
统一分析流程

功能:
- 时序动态分析 (TDS)
- 动态功能连接分析 (DCS)  
- 多时间尺度编码分析 (TWG)
- LSTM vs Ridge比较
- Transformer注意力分析
- Hub中心性分析
- 统计检验
- 个体差异分析
- 图表生成

运行方式:
    python run_analysis.py                    # 快速模式 (4 episodes)
    python run_analysis.py --mode full        # 完整模式 (所有数据)
    python run_analysis.py --mode quick       # 快速模式
    python run_analysis.py --max_episodes 20  # 指定episodes数量
    python run_analysis.py --skip_ai          # 跳过AI模型分析
    python run_analysis.py --figures_only     # 只生成图表

作者: CCN Competition Team
日期: 2026年1月
"""

import os
import sys
import argparse
import json
import time
import subprocess
from datetime import datetime, timedelta
import traceback

# 确保输出立即刷新
sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# 工具函数
# ============================================================

def print_header(title, char="=", width=70):
    """打印格式化标题"""
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)
    sys.stdout.flush()


def print_status(message, status="info"):
    """打印状态消息"""
    symbols = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✗", "progress": "→"}
    symbol = symbols.get(status, "•")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"  [{timestamp}] {symbol} {message}")
    sys.stdout.flush()


def run_script(script_name, run_dir=None, extra_args=None):
    """运行Python脚本"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_dir, script_name)
    
    if not os.path.exists(script_path):
        print_status(f"Script not found: {script_name}", "error")
        return False
    
    cmd = [sys.executable, '-u', script_path]
    if run_dir:
        cmd.append(run_dir)
    if extra_args:
        cmd.extend(extra_args)
    
    print_status(f"Running: {script_name}", "progress")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print_status(f"Completed: {script_name}", "success")
            return True
        else:
            print_status(f"Failed: {script_name} (exit code {result.returncode})", "error")
            return False
    except Exception as e:
        print_status(f"Error running {script_name}: {e}", "error")
        return False


def create_run_directory(project_dir, suffix=""):
    """创建运行目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}{suffix}" if suffix else timestamp
    run_dir = os.path.join(project_dir, 'runs', run_name)
    
    subdirs = [
        'temporal_spectral',
        'dynamic_fc', 
        'multiscale_encoding',
        'temporal_efficiency',
        'lstm_ridge',           # 统一命名
        'transformer_attention',
        'hub_centrality',       # 统一命名
        'statistical_tests',
        'individual_analysis',
        'figures',
        'logs'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    return run_dir


def save_config(run_dir, config):
    """保存配置"""
    config_path = os.path.join(run_dir, 'config.json')
    config['timestamp'] = datetime.now().isoformat()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


# ============================================================
# 主分析流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Project 2 Unified Analysis Pipeline')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Analysis mode: quick (4 episodes) or full (all data)')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Maximum episodes per subject (overrides mode)')
    parser.add_argument('--skip_ai', action='store_true',
                       help='Skip AI model analyses (LSTM, Transformer)')
    parser.add_argument('--figures_only', action='store_true',
                       help='Only generate figures from existing results')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Use existing run directory')
    args = parser.parse_args()
    
    # 设置路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    # 确定episodes数量
    if args.max_episodes:
        max_episodes = args.max_episodes
    elif args.mode == 'full':
        max_episodes = None  # 使用所有数据
    else:
        max_episodes = 4  # 快速模式
    
    print_header("Project 2: Temporal Efficiency Paradox Analysis")
    print(f"  Mode: {args.mode}")
    print(f"  Max episodes: {max_episodes or 'all'}")
    print(f"  Skip AI models: {args.skip_ai}")
    print(f"  Data directory: {data_dir}")
    
    # 创建或使用运行目录
    if args.figures_only and args.run_dir:
        run_dir = args.run_dir
        if not os.path.exists(run_dir):
            run_dir = os.path.join(project_dir, 'runs', args.run_dir)
        print(f"  Using existing run: {run_dir}")
    elif args.run_dir:
        run_dir = args.run_dir
        if not os.path.exists(run_dir):
            run_dir = os.path.join(project_dir, 'runs', args.run_dir)
    else:
        suffix = "_complete" if args.mode == 'full' else ""
        run_dir = create_run_directory(project_dir, suffix)
    
    print(f"  Output directory: {run_dir}")
    
    # 保存配置
    config = {
        'mode': args.mode,
        'max_episodes': max_episodes,
        'skip_ai': args.skip_ai,
        'data_dir': data_dir,
        'run_dir': run_dir
    }
    save_config(run_dir, config)
    
    start_time = time.time()
    results = {'success': [], 'failed': []}
    
    # 构建max_episodes参数
    episode_args = ['--max_episodes', str(max_episodes)] if max_episodes else []
    
    if not args.figures_only:
        # ============================================================
        # Step 1: 时序动态分析 (TDS)
        # ============================================================
        print_header("Step 1: Temporal Dynamics Analysis (TDS)")
        if run_script('01_temporal_analysis.py', extra_args=episode_args):
            results['success'].append('TDS')
        else:
            results['failed'].append('TDS')
        
        # ============================================================
        # Step 2: 动态功能连接分析 (DCS)
        # ============================================================
        print_header("Step 2: Dynamic Functional Connectivity (DCS)")
        if run_script('02_dynamic_fc.py', extra_args=episode_args):
            results['success'].append('DCS')
        else:
            results['failed'].append('DCS')
        
        # ============================================================
        # Step 3: 多时间尺度编码分析 (TWG)
        # ============================================================
        print_header("Step 3: Multi-Timescale Encoding (TWG)")
        if run_script('03_encoding.py', extra_args=episode_args):
            results['success'].append('TWG')
        else:
            results['failed'].append('TWG')
        
        # ============================================================
        # Step 4: 统计检验
        # ============================================================
        print_header("Step 4: Statistical Testing")
        if run_script('04_statistics.py'):
            results['success'].append('Statistics')
        else:
            results['failed'].append('Statistics')
        
        # ============================================================
        # Step 5: 个体差异分析
        # ============================================================
        print_header("Step 5: Individual Differences Analysis")
        if run_script('05_individual.py'):
            results['success'].append('Individual')
        else:
            results['failed'].append('Individual')
        
        # ============================================================
        # Step 6-8: AI模型分析 (可选)
        # ============================================================
        if not args.skip_ai:
            print_header("Step 6: LSTM vs Ridge Comparison")
            if run_script('06_lstm_comparison.py'):
                results['success'].append('LSTM')
            else:
                results['failed'].append('LSTM')
            
            print_header("Step 7: Hub Centrality Analysis")
            if run_script('07_hub_analysis.py'):
                results['success'].append('Hub')
            else:
                results['failed'].append('Hub')
            
            print_header("Step 8: Transformer Attention Analysis")
            if run_script('08_transformer.py'):
                results['success'].append('Transformer')
            else:
                results['failed'].append('Transformer')
        else:
            print_header("Skipping AI Model Analyses")
            print_status("LSTM, Hub, and Transformer analyses skipped", "warning")
    
    # ============================================================
    # 图表生成
    # ============================================================
    print_header("Generating Figures")
    if run_script('05_generate_figures.py', run_dir):
        results['success'].append('Figures')
    else:
        results['failed'].append('Figures')
    
    # ============================================================
    # 完成总结
    # ============================================================
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    
    print_header("Analysis Complete")
    print(f"  Total time: {elapsed_str}")
    print(f"  Successful: {', '.join(results['success']) or 'None'}")
    print(f"  Failed: {', '.join(results['failed']) or 'None'}")
    print(f"  Results saved to: {run_dir}")
    
    # 列出生成的图表
    figures_dir = os.path.join(run_dir, 'figures')
    if os.path.exists(figures_dir):
        figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        print(f"  Generated {len(figures)} figures")
    
    return 0 if not results['failed'] else 1


if __name__ == '__main__':
    sys.exit(main())

