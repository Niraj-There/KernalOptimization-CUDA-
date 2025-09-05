#!/usr/bin/env python3
"""
Block Configuration Impact Analysis - Performance Heatmap
GFLOPS vs Matrix Size and Block Configuration for Tesla V100 and T4
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

# Base directory for data files
BASE_DIR = Path(r"c:\Users\HP\Desktop\Research Intersnhip\CUDA\Codes")

def load_csv_data(architecture='Tesla V100'):
    """Load and process CSV data for the specified architecture"""
    print(f"Loading {architecture} data...")
    
    arch_dir = BASE_DIR / architecture
    csv_files = [
        'baseline_results.csv',
        'advanced_results.csv',
        'quick_test_results.csv',
        'intelligent_optimization_results.csv'
    ]
    
    combined_data = pd.DataFrame()
    
    for csv_file in csv_files:
        file_path = arch_dir / csv_file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                combined_data = pd.concat([combined_data, df], ignore_index=True)
                print(f"  ✓ Loaded {csv_file}: {len(df)} rows")
            except Exception as e:
                print(f"  ✗ Error loading {csv_file}: {e}")
    
    print(f"  Total rows loaded: {len(combined_data)}")
    return combined_data

def create_performance_heatmap_matrix_vs_block(data, architecture):
    """Create heatmap: GFLOPS vs Matrix Size and Block Configuration"""
    print(f"Creating performance heatmap for {architecture}...")
    
    # Create block configuration string
    data['Block_Config'] = data['Block_Dim_X'].astype(str) + 'x' + data['Block_Dim_Y'].astype(str)
    
    # Get unique matrix sizes and block configurations
    matrix_sizes = sorted(data['Matrix_Size'].unique())
    block_configs = sorted(data['Block_Config'].unique())
    
    # Create pivot table for heatmap data
    heatmap_data = []
    
    for matrix_size in matrix_sizes:
        row_data = []
        size_data = data[data['Matrix_Size'] == matrix_size]
        
        for block_config in block_configs:
            config_data = size_data[size_data['Block_Config'] == block_config]
            
            if len(config_data) > 0:
                # Use maximum GFLOPS for this configuration (best performance)
                max_gflops = config_data['GFLOPS'].max()
                row_data.append(max_gflops)
            else:
                row_data.append(0)  # No data available
        
        heatmap_data.append(row_data)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create the heatmap with larger figure size
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Create heatmap with custom colormap
    im = ax.imshow(heatmap_array, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels with larger font
    ax.set_xticks(range(len(block_configs)))
    ax.set_yticks(range(len(matrix_sizes)))
    ax.set_xticklabels(block_configs, rotation=45, ha='right', fontsize=20, fontweight='bold')
    ax.set_yticklabels([f'{size}×{size}' for size in matrix_sizes], fontsize=20, fontweight='bold')
    
    # Add colorbar with larger font
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance (GFLOPS)', rotation=270, labelpad=25, fontsize=22, fontweight='bold')
    cbar.ax.tick_params(labelsize=20)
    
    # Add value annotations with larger, black text
    for i in range(len(matrix_sizes)):
        for j in range(len(block_configs)):
            value = heatmap_array[i, j]
            if value > 0:
                # Make all text black and larger
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', 
                       color='black', fontweight='bold', fontsize=22)
    
    # Set labels and title with larger fonts
    ax.set_xlabel('Block Configuration (X × Y)', fontweight='bold', fontsize=22)
    ax.set_ylabel('Matrix Size', fontweight='bold', fontsize=22)
    ax.set_title(f'Performance Heatmap: GFLOPS vs Matrix Size and Block Configuration\n{architecture}', 
                fontweight='bold', pad=20, fontsize=24)
    
    # Add grid for better readability with thicker lines
    ax.set_xticks(np.arange(len(block_configs)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(matrix_sizes)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    return fig

def create_energy_efficiency_heatmap_matrix_vs_block(data, architecture):
    """Create heatmap: Energy Efficiency vs Matrix Size and Block Configuration"""
    print(f"Creating energy efficiency heatmap for {architecture}...")
    
    # Create block configuration string
    data['Block_Config'] = data['Block_Dim_X'].astype(str) + 'x' + data['Block_Dim_Y'].astype(str)
    
    # Get unique matrix sizes and block configurations
    matrix_sizes = sorted(data['Matrix_Size'].unique())
    block_configs = sorted(data['Block_Config'].unique())
    
    # Create pivot table for heatmap data
    heatmap_data = []
    
    for matrix_size in matrix_sizes:
        row_data = []
        size_data = data[data['Matrix_Size'] == matrix_size]
        
        for block_config in block_configs:
            config_data = size_data[size_data['Block_Config'] == block_config]
            
            if len(config_data) > 0:
                # Use maximum energy efficiency for this configuration
                max_efficiency = config_data['GFLOPS_per_Watt'].max()
                row_data.append(max_efficiency)
            else:
                row_data.append(0)  # No data available
        
        heatmap_data.append(row_data)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create the heatmap with larger figure size
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Create heatmap with custom colormap
    im = ax.imshow(heatmap_array, cmap='GnBu', aspect='auto')
    
    # Set ticks and labels with larger font
    ax.set_xticks(range(len(block_configs)))
    ax.set_yticks(range(len(matrix_sizes)))
    ax.set_xticklabels(block_configs, rotation=45, ha='right', fontsize=20, fontweight='bold')
    ax.set_yticklabels([f'{size}×{size}' for size in matrix_sizes], fontsize=20, fontweight='bold')
    
    # Add colorbar with larger font
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Energy Efficiency (GFLOPS/Watt)', rotation=270, labelpad=25, fontsize=22, fontweight='bold')
    cbar.ax.tick_params(labelsize=20)
    
    # Add value annotations with larger, black text
    for i in range(len(matrix_sizes)):
        for j in range(len(block_configs)):
            value = heatmap_array[i, j]
            if value > 0:
                # Make all text black and larger
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                       color='black', fontweight='bold', fontsize=22)
    
    # Set labels and title with larger fonts
    ax.set_xlabel('Block Configuration (X × Y)', fontweight='bold', fontsize=22)
    ax.set_ylabel('Matrix Size', fontweight='bold', fontsize=22)
    ax.set_title(f'Energy Efficiency Heatmap: GFLOPS/Watt vs Matrix Size and Block Configuration\n{architecture}', 
                fontweight='bold', pad=20, fontsize=24)
    
    # Add grid for better readability with thicker lines
    ax.set_xticks(np.arange(len(block_configs)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(matrix_sizes)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    return fig

def generate_corrected_heatmaps():
    """Generate corrected heatmaps with Matrix Size vs Block Configuration"""
    print("Generating Block Configuration Impact Analysis Heatmaps")
    print("Matrix Size vs Block Configuration Analysis")
    print("=" * 60)
    
    architectures = ['Tesla V100', 'Tesla T4']
    
    for arch in architectures:
        print(f"\nProcessing {arch}...")
        
        # Load data
        data = load_csv_data(arch)
        
        if len(data) == 0:
            print(f"  ✗ No data available for {arch}")
            continue
        
        # Clean architecture name for filename
        arch_clean = arch.replace(' ', '_').replace('Tesla_', '')
        
        # Generate performance heatmap (GFLOPS vs Matrix Size and Block Config)
        fig1 = create_performance_heatmap_matrix_vs_block(data, arch)
        filename1 = f'Block_Config_Performance_Heatmap_{arch_clean}.png'
        fig1.savefig(filename1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"  ✅ Saved: {filename1}")
        
        # Generate energy efficiency heatmap
        fig2 = create_energy_efficiency_heatmap_matrix_vs_block(data, arch)
        filename2 = f'Block_Config_Energy_Efficiency_Heatmap_{arch_clean}.png'
        fig2.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"  ✅ Saved: {filename2}")
    
    print("\n" + "=" * 60)
    print("✅ CORRECTED BLOCK CONFIGURATION HEATMAPS GENERATED!")
    print("📊 Matrix Size vs Block Configuration analysis")
    print("📈 Performance (GFLOPS) and Energy Efficiency (GFLOPS/Watt)")
    print("📈 Publication-ready quality (300 DPI)")
    print("=" * 60)
    
    print("\nGenerated Files:")
    for arch in ['V100', 'T4']:
        print(f"📊 Block_Config_Performance_Heatmap_{arch}.png")
        print(f"📊 Block_Config_Energy_Efficiency_Heatmap_{arch}.png")
    
    print("\nHeatmap Details:")
    print("• X-axis: Block Configuration (8x8, 16x16, 32x32, etc.)")
    print("• Y-axis: Matrix Size (512x512, 1024x1024, etc.)")
    print("• Color/Values: GFLOPS performance or GFLOPS/Watt efficiency")
    print("• Annotations: Exact values shown in each cell")
    
    print("\nRecommended Paper Placement:")
    print("• Section: V. Results and Analysis")
    print("• Subsection: Block Configuration Impact Analysis")
    print("• After text: 'The impact of block configuration on performance across different matrix sizes...'")

if __name__ == "__main__":
    generate_corrected_heatmaps()
