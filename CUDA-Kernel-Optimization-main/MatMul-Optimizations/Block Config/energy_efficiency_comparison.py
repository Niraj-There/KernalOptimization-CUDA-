"""
Energy Efficiency Comparison Analysis for CUDA Matrix Multiplication Kernels
Tesla T4 vs Tesla V100 Performance Analysis

This script generates research-quality energy efficiency comparison graphs
for different CUDA optimization techniques across two GPU architectures.

Author: Aniruddha Shete
Research Context: Systematic Performance Engineering Framework for Energy-Aware CUDA Kernel Optimization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Set IEEE-format research styling
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0
})

# IEEE double-column format configuration
FIGURE_SIZE = (7.5, 12)  # IEEE double column width (7.5") with vertical arrangement
DPI = 300
FONT_SIZE = 10
TITLE_SIZE = 11
LABEL_SIZE = 9
AXIS_LABEL_SIZE = 10

# Kernel mapping with specific ordering as requested
KERNEL_MAPPING = {
    0: "Naive Baseline",
    1: "Memory Coalesced", 
    5: "Advanced Optimized",
    2: "Shared Memory Tiled",
    4: "Occupancy Optimized",
    3: "Register Blocked"
}

# Ordered list for consistent presentation
KERNEL_ORDER = [0, 1, 5, 2, 4, 3]
KERNEL_LABELS = [KERNEL_MAPPING[k] for k in KERNEL_ORDER]

# IEEE-compliant color scheme for publication - high contrast
COLORS = {
    'Tesla V100': '#2E86AB',  # Strong blue for V100
    'Tesla T4': '#F24236'     # Strong red for T4
}

def load_and_process_data(csv_files):
    """
    Load CSV files and process energy efficiency data
    
    Args:
        csv_files (dict): Dictionary mapping GPU names to CSV file paths
        
    Returns:
        dict: Processed data for each GPU
    """
    processed_data = {}
    
    for gpu_name, csv_path in csv_files.items():
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path)
            
            # Ensure required columns exist
            required_columns = ['Kernel_Type', 'GFLOPS_per_Watt', 'Matrix_Size', 'Block_Dim_X', 'Block_Dim_Y']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns in {csv_path}: {missing_columns}")
                continue
                
            # Filter for standard matrix size (1024x1024 for consistency)
            df_filtered = df[df['Matrix_Size'] == 1024].copy()
            
            if df_filtered.empty:
                print(f"Warning: No data found for matrix size 1024 in {csv_path}")
                # Try other common sizes
                for size in [2048, 512, 4096]:
                    df_filtered = df[df['Matrix_Size'] == size].copy()
                    if not df_filtered.empty:
                        print(f"Using matrix size {size} for {gpu_name}")
                        break
            
            # Calculate average GFLOPS/Watt for each kernel type
            kernel_efficiency = {}
            
            for kernel_type in KERNEL_ORDER:
                kernel_data = df_filtered[df_filtered['Kernel_Type'] == kernel_type]
                
                if not kernel_data.empty:
                    # Calculate mean efficiency for this kernel type
                    mean_efficiency = kernel_data['GFLOPS_per_Watt'].mean()
                    kernel_efficiency[kernel_type] = mean_efficiency
                    
                    print(f"{gpu_name} - {KERNEL_MAPPING[kernel_type]}: {mean_efficiency:.1f} GFLOPS/W")
                else:
                    print(f"Warning: No data found for kernel type {kernel_type} in {gpu_name}")
                    kernel_efficiency[kernel_type] = 0.0
            
            processed_data[gpu_name] = kernel_efficiency
            
        except FileNotFoundError:
            print(f"Error: CSV file not found: {csv_path}")
        except Exception as e:
            print(f"Error processing {csv_path}: {str(e)}")
    
    return processed_data

def create_energy_efficiency_graph(data, output_path):
    """
    Create IEEE-format energy efficiency comparison graph with vertical arrangement
    
    Args:
        data (dict): Processed efficiency data for each GPU
        output_path (str): Path to save the output graph
    """
    # Create figure with vertical subplots for IEEE double-column format
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZE, facecolor='white')
    fig.suptitle('Energy Efficiency Comparison (GFLOPS/W)', 
                 fontsize=TITLE_SIZE, fontweight='bold', y=0.95)
    
    # Define bar positions
    x_positions = np.arange(len(KERNEL_ORDER))
    bar_width = 0.7
    
    # Tesla V100 subplot (top)
    if 'Tesla V100' in data:
        v100_values = [data['Tesla V100'].get(k, 0) for k in KERNEL_ORDER]
        bars1 = ax1.bar(x_positions, v100_values, bar_width, 
                       color=COLORS['Tesla V100'], alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars with better positioning
        for i, (bar, value) in enumerate(zip(bars1, v100_values)):
            if value > 0:
                # Position label with appropriate offset based on bar height
                label_height = bar.get_height() + max(v100_values) * 0.02
                ax1.text(bar.get_x() + bar.get_width()/2, label_height,
                        f'{value:.1f}', ha='center', va='bottom', 
                        fontsize=LABEL_SIZE, fontweight='bold')
        
        ax1.set_title('(a) Tesla V100 Energy Efficiency', fontsize=TITLE_SIZE, fontweight='bold', pad=10)
        ax1.set_ylim(0, max(v100_values) * 1.15 if max(v100_values) > 0 else 1200)
        ax1.set_ylabel('GFLOPS/W', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        
        # Remove x-axis labels for top subplot
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([])
    
    # Tesla T4 subplot (bottom)
    if 'Tesla T4' in data:
        t4_values = [data['Tesla T4'].get(k, 0) for k in KERNEL_ORDER]
        bars2 = ax2.bar(x_positions, t4_values, bar_width, 
                       color=COLORS['Tesla T4'], alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars with better positioning
        for i, (bar, value) in enumerate(zip(bars2, t4_values)):
            if value > 0:
                # Position label with appropriate offset based on bar height
                label_height = bar.get_height() + max(t4_values) * 0.02
                ax2.text(bar.get_x() + bar.get_width()/2, label_height,
                        f'{value:.1f}', ha='center', va='bottom', 
                        fontsize=LABEL_SIZE, fontweight='bold')
        
        ax2.set_title('(b) Tesla T4 Energy Efficiency', fontsize=TITLE_SIZE, fontweight='bold', pad=10)
        ax2.set_ylim(0, max(t4_values) * 1.15 if max(t4_values) > 0 else 400)
        ax2.set_ylabel('GFLOPS/W', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        
        # Add x-axis labels only for bottom subplot
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(KERNEL_LABELS, rotation=45, ha='right', fontsize=LABEL_SIZE)
        ax2.set_xlabel('Optimization Technique', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
    
    # Configure both subplots with IEEE styling
    for ax in [ax1, ax2]:
        # IEEE-compliant grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='y', labelsize=LABEL_SIZE)
        ax.tick_params(axis='x', labelsize=LABEL_SIZE)
        
        # IEEE-compliant axis styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Set background color
        ax.set_facecolor('white')
    
    # Adjust layout for IEEE double-column format
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15, hspace=0.3)
    
    # Save the figure in high resolution for IEEE publication
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.show()
    
    print(f"Energy efficiency comparison graph saved to: {output_path}")
    print(f"PDF version saved to: {output_path.replace('.png', '.pdf')}")

def generate_summary_statistics(data):
    """
    Generate summary statistics for the energy efficiency analysis
    
    Args:
        data (dict): Processed efficiency data for each GPU
    """
    print("\n" + "="*70)
    print("ENERGY EFFICIENCY ANALYSIS SUMMARY")
    print("="*70)
    
    for gpu_name, kernel_data in data.items():
        print(f"\n{gpu_name} Results:")
        print("-" * 40)
        
        # Sort by efficiency for ranking
        sorted_kernels = sorted(kernel_data.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (kernel_type, efficiency) in enumerate(sorted_kernels, 1):
            kernel_name = KERNEL_MAPPING[kernel_type]
            print(f"{rank}. {kernel_name:20}: {efficiency:8.1f} GFLOPS/W")
        
        # Calculate efficiency improvements
        baseline_efficiency = kernel_data.get(0, 0)  # Naive baseline
        if baseline_efficiency > 0:
            print(f"\nEfficiency Improvements over Naive Baseline:")
            for kernel_type, efficiency in sorted_kernels:
                if kernel_type != 0 and efficiency > 0:
                    improvement = ((efficiency - baseline_efficiency) / baseline_efficiency) * 100
                    kernel_name = KERNEL_MAPPING[kernel_type]
                    print(f"  {kernel_name:20}: {improvement:+6.1f}%")

def create_comparative_analysis(data, output_path):
    """
    Create comparative analysis between GPUs
    
    Args:
        data (dict): Processed efficiency data for each GPU
        output_path (str): Path to save comparative analysis
    """
    if len(data) < 2:
        print("Insufficient data for comparative analysis")
        return
    
    # Create comparison table
    comparison_data = []
    for kernel_type in KERNEL_ORDER:
        row = {'Kernel': KERNEL_MAPPING[kernel_type]}
        for gpu_name, kernel_data in data.items():
            row[gpu_name] = kernel_data.get(kernel_type, 0)
        comparison_data.append(row)
    
    # Convert to DataFrame and save
    df_comparison = pd.DataFrame(comparison_data)
    
    # Add efficiency ratio if both GPUs have data
    if 'Tesla V100' in data and 'Tesla T4' in data:
        df_comparison['V100/T4 Ratio'] = df_comparison['Tesla V100'] / df_comparison['Tesla T4']
    
    # Save comparison table
    comparison_csv = output_path.replace('.png', '_comparison.csv')
    df_comparison.to_csv(comparison_csv, index=False)
    print(f"Comparative analysis saved to: {comparison_csv}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS TABLE")
    print("="*70)
    print(df_comparison.to_string(index=False, float_format='%.1f'))

def verify_data_consistency(data, script_name):
    """
    Verify data consistency and report values for comparison
    
    Args:
        data (dict): Processed efficiency data
        script_name (str): Name of the script for identification
    """
    print(f"\n{script_name} - Data Verification Report")
    print("=" * 50)
    
    for gpu_name, kernel_data in data.items():
        print(f"\n{gpu_name} Results:")
        total_efficiency = 0
        valid_kernels = 0
        
        for kernel_type in KERNEL_ORDER:
            efficiency = kernel_data.get(kernel_type, 0)
            kernel_name = KERNEL_MAPPING[kernel_type]
            print(f"  {kernel_name:22}: {efficiency:7.1f} GFLOPS/W")
            
            if efficiency > 0:
                total_efficiency += efficiency
                valid_kernels += 1
        
        if valid_kernels > 0:
            avg_efficiency = total_efficiency / valid_kernels
            print(f"  {'Average Efficiency':22}: {avg_efficiency:7.1f} GFLOPS/W")
            print(f"  {'Valid Kernels':22}: {valid_kernels:7d}")

def main():
    """
    Main function to orchestrate the energy efficiency analysis
    """
    print("Energy Efficiency Comparison Analysis")
    print("=" * 50)
    
    # Define CSV file paths (adjust these paths as needed)
    csv_files = {
        'Tesla V100': '/content/Tesla V100/baseline_results.csv',
        'Tesla T4': '/content/Tesla T4/baseline_results.csv'
    }
    
    # Alternative paths to try if primary paths don't exist
    alternative_paths = {
        'Tesla V100': [
            '/content/baseline_results_v100.csv',
            '/content/v100_results.csv',
            '/content/baseline_results.csv'
        ],
        'Tesla T4': [
            '/content/baseline_results_t4.csv',
            '/content/t4_results.csv',
            '/content/baseline_results.csv'
        ]
    }
    
    # Try to find existing CSV files
    final_csv_files = {}
    for gpu_name, primary_path in csv_files.items():
        if Path(primary_path).exists():
            final_csv_files[gpu_name] = primary_path
            print(f"Found data for {gpu_name}: {primary_path}")
        else:
            # Try alternative paths
            found = False
            for alt_path in alternative_paths[gpu_name]:
                if Path(alt_path).exists():
                    final_csv_files[gpu_name] = alt_path
                    print(f"Found data for {gpu_name}: {alt_path}")
                    found = True
                    break
            
            if not found:
                print(f"Warning: No data file found for {gpu_name}")
    
    if not final_csv_files:
        print("Error: No CSV files found. Please ensure the data files exist.")
        return
    
    # Load and process data
    print("\nLoading and processing data...")
    data = load_and_process_data(final_csv_files)
    
    if not data:
        print("Error: No data could be processed.")
        return
    
    # Generate energy efficiency graph
    output_path = '/content/energy_efficiency_comparison.png'
    create_energy_efficiency_graph(data, output_path)
    
    # Verify data consistency
    verify_data_consistency(data, "Energy Efficiency Comparison")
    
    # Generate summary statistics
    generate_summary_statistics(data)
    
    # Create comparative analysis
    create_comparative_analysis(data, output_path)
    
    # Verify data consistency
    verify_data_consistency(data, "Main Script")
    
    print(f"\nAnalysis complete. Results saved to: {output_path}")

if __name__ == "__main__":
    main()
