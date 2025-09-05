#!/usr/bin/env python3
"""
Block Configuration Impact Analysis - Heatmap Generation
Creates heatmaps showing performance impact of different block configurations
for Tesla V100 and Tesla T4 architectures using real CSV data.
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

# Kernel type mapping
KERNEL_NAMES = {
    0: 'Naive Baseline',
    1: 'Memory Coalesced', 
    2: 'Shared Memory Tiled',
    3: 'Register Blocked',
    4: 'Occupancy Optimized',
    5: 'Advanced Optimized'
}

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

def create_block_configuration_heatmap(data, architecture):
    """Create heatmap showing block configuration impact on performance"""
    print(f"Creating block configuration heatmap for {architecture}...")
    
    # Create block configuration string
    data['Block_Config'] = data['Block_Dim_X'].astype(str) + 'x' + data['Block_Dim_Y'].astype(str)
    
    # Get unique configurations and kernel types
    block_configs = sorted(data['Block_Config'].unique())
    kernel_types = sorted(data['Kernel_Type'].unique())
    
    # Create pivot table for heatmap data
    heatmap_data = []
    
    for kernel_type in kernel_types:
        row_data = []
        kernel_data = data[data['Kernel_Type'] == kernel_type]
        
        for block_config in block_configs:
            config_data = kernel_data[kernel_data['Block_Config'] == block_config]
            
            if len(config_data) > 0:
                # Use average GFLOPS for this configuration
                avg_gflops = config_data['GFLOPS'].mean()
                row_data.append(avg_gflops)
            else:
                row_data.append(0)  # No data available
        
        heatmap_data.append(row_data)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap with custom colormap
    im = ax.imshow(heatmap_array, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(block_configs)))
    ax.set_yticks(range(len(kernel_types)))
    ax.set_xticklabels(block_configs, rotation=45, ha='right')
    ax.set_yticklabels([KERNEL_NAMES.get(kt, f'Kernel {kt}') for kt in kernel_types])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance (GFLOPS)', rotation=270, labelpad=20)
    
    # Add value annotations
    for i in range(len(kernel_types)):
        for j in range(len(block_configs)):
            value = heatmap_array[i, j]
            if value > 0:
                text_color = 'white' if value > heatmap_array.max() * 0.5 else 'black'
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', 
                       color=text_color, fontweight='bold', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Block Configuration (X × Y)', fontweight='bold')
    ax.set_ylabel('Kernel Type', fontweight='bold')
    ax.set_title(f'Block Configuration Impact Analysis - {architecture}', 
                fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(block_configs)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(kernel_types)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    return fig

def create_energy_efficiency_heatmap(data, architecture):
    """Create heatmap showing energy efficiency impact of block configurations"""
    print(f"Creating energy efficiency heatmap for {architecture}...")
    
    # Create block configuration string
    data['Block_Config'] = data['Block_Dim_X'].astype(str) + 'x' + data['Block_Dim_Y'].astype(str)
    
    # Get unique configurations and kernel types
    block_configs = sorted(data['Block_Config'].unique())
    kernel_types = sorted(data['Kernel_Type'].unique())
    
    # Create pivot table for heatmap data
    heatmap_data = []
    
    for kernel_type in kernel_types:
        row_data = []
        kernel_data = data[data['Kernel_Type'] == kernel_type]
        
        for block_config in block_configs:
            config_data = kernel_data[kernel_data['Block_Config'] == block_config]
            
            if len(config_data) > 0:
                # Use average GFLOPS per Watt for this configuration
                avg_efficiency = config_data['GFLOPS_per_Watt'].mean()
                row_data.append(avg_efficiency)
            else:
                row_data.append(0)  # No data available
        
        heatmap_data.append(row_data)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap with custom colormap
    im = ax.imshow(heatmap_array, cmap='GnBu', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(block_configs)))
    ax.set_yticks(range(len(kernel_types)))
    ax.set_xticklabels(block_configs, rotation=45, ha='right')
    ax.set_yticklabels([KERNEL_NAMES.get(kt, f'Kernel {kt}') for kt in kernel_types])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Energy Efficiency (GFLOPS/Watt)', rotation=270, labelpad=20)
    
    # Add value annotations
    for i in range(len(kernel_types)):
        for j in range(len(block_configs)):
            value = heatmap_array[i, j]
            if value > 0:
                text_color = 'white' if value > heatmap_array.max() * 0.5 else 'black'
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                       color=text_color, fontweight='bold', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Block Configuration (X × Y)', fontweight='bold')
    ax.set_ylabel('Kernel Type', fontweight='bold')
    ax.set_title(f'Block Configuration Energy Efficiency - {architecture}', 
                fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(block_configs)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(kernel_types)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    return fig

def create_occupancy_heatmap(data, architecture):
    """Create heatmap showing occupancy impact of block configurations"""
    print(f"Creating occupancy heatmap for {architecture}...")
    
    # Calculate theoretical occupancy based on threads per block
    # Assuming 2048 threads per SM for V100, 1024 for T4
    max_threads_per_sm = 2048 if 'V100' in architecture else 1024
    
    data['Block_Config'] = data['Block_Dim_X'].astype(str) + 'x' + data['Block_Dim_Y'].astype(str)
    data['Theoretical_Occupancy'] = np.minimum(
        max_threads_per_sm / data['Threads_per_Block'], 
        32  # Maximum 32 blocks per SM
    ) / 32  # Normalize to percentage
    
    # Get unique configurations and kernel types
    block_configs = sorted(data['Block_Config'].unique())
    kernel_types = sorted(data['Kernel_Type'].unique())
    
    # Create pivot table for heatmap data
    heatmap_data = []
    
    for kernel_type in kernel_types:
        row_data = []
        kernel_data = data[data['Kernel_Type'] == kernel_type]
        
        for block_config in block_configs:
            config_data = kernel_data[kernel_data['Block_Config'] == block_config]
            
            if len(config_data) > 0:
                # Use average theoretical occupancy for this configuration
                avg_occupancy = config_data['Theoretical_Occupancy'].mean()
                row_data.append(avg_occupancy * 100)  # Convert to percentage
            else:
                row_data.append(0)  # No data available
        
        heatmap_data.append(row_data)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap with custom colormap
    im = ax.imshow(heatmap_array, cmap='plasma', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(range(len(block_configs)))
    ax.set_yticks(range(len(kernel_types)))
    ax.set_xticklabels(block_configs, rotation=45, ha='right')
    ax.set_yticklabels([KERNEL_NAMES.get(kt, f'Kernel {kt}') for kt in kernel_types])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Theoretical Occupancy (%)', rotation=270, labelpad=20)
    
    # Add value annotations
    for i in range(len(kernel_types)):
        for j in range(len(block_configs)):
            value = heatmap_array[i, j]
            if value > 0:
                text_color = 'white' if value > 50 else 'black'
                ax.text(j, i, f'{value:.0f}%', ha='center', va='center', 
                       color=text_color, fontweight='bold', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Block Configuration (X × Y)', fontweight='bold')
    ax.set_ylabel('Kernel Type', fontweight='bold')
    ax.set_title(f'Block Configuration Occupancy Analysis - {architecture}', 
                fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(block_configs)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(kernel_types)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    return fig

def generate_all_heatmaps():
    """Generate all block configuration heatmaps for both architectures"""
    print("Generating Block Configuration Impact Analysis Heatmaps")
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
        
        # Generate performance heatmap
        fig1 = create_block_configuration_heatmap(data, arch)
        filename1 = f'Block_Config_Performance_Heatmap_{arch_clean}.png'
        fig1.savefig(filename1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"  ✅ Saved: {filename1}")
        
        # Generate energy efficiency heatmap
        fig2 = create_energy_efficiency_heatmap(data, arch)
        filename2 = f'Block_Config_Energy_Efficiency_Heatmap_{arch_clean}.png'
        fig2.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"  ✅ Saved: {filename2}")
        
        # Generate occupancy heatmap
        fig3 = create_occupancy_heatmap(data, arch)
        filename3 = f'Block_Config_Occupancy_Heatmap_{arch_clean}.png'
        fig3.savefig(filename3, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"  ✅ Saved: {filename3}")
    
    print("\n" + "=" * 60)
    print("✅ ALL BLOCK CONFIGURATION HEATMAPS GENERATED!")
    print("📊 6 heatmaps created (3 for each architecture)")
    print("📈 Publication-ready quality (300 DPI)")
    print("=" * 60)
    
    print("\nGenerated Files:")
    for arch in ['V100', 'T4']:
        print(f"📊 Block_Config_Performance_Heatmap_{arch}.png")
        print(f"📊 Block_Config_Energy_Efficiency_Heatmap_{arch}.png")
        print(f"📊 Block_Config_Occupancy_Heatmap_{arch}.png")
    
    print("\nRecommended Usage:")
    print("• Performance Heatmaps: Results section - show optimal block configurations")
    print("• Energy Efficiency Heatmaps: Energy analysis section")
    print("• Occupancy Heatmaps: Methodology section - explain optimization rationale")

if __name__ == "__main__":
    generate_all_heatmaps()
