"""
Debug Script for Thermal Analysis Image Saving Issues
This script helps identify and fix image saving problems.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def test_matplotlib_saving():
    """Test basic matplotlib saving functionality"""
    print("Testing Matplotlib Image Saving...")
    print("=" * 50)
    
    # Check matplotlib backend
    print(f"Current matplotlib backend: {matplotlib.get_backend()}")
    
    # Test basic plot creation and saving
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    ax.set_title('Test Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test directory creation
    test_dir = r'c:\Users\HP\Desktop\Research Intersnhip\CUDA\Codes\test_images'
    
    try:
        os.makedirs(test_dir, exist_ok=True)
        print(f"✓ Directory created/exists: {test_dir}")
    except Exception as e:
        print(f"✗ Error creating directory: {e}")
        return False
    
    # Test file saving in different formats
    formats = ['png', 'pdf', 'eps']
    saved_files = []
    
    for fmt in formats:
        test_file = os.path.join(test_dir, f'test_plot.{fmt}')
        try:
            plt.savefig(test_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', format=fmt)
            
            # Check if file was actually created
            if os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                print(f"✓ Successfully saved {fmt.upper()}: {test_file} ({file_size} bytes)")
                saved_files.append(test_file)
            else:
                print(f"✗ File not found after saving: {test_file}")
        except Exception as e:
            print(f"✗ Error saving {fmt.upper()}: {e}")
    
    plt.close(fig)
    
    # Check write permissions
    try:
        test_perm_file = os.path.join(test_dir, 'permission_test.txt')
        with open(test_perm_file, 'w') as f:
            f.write('test')
        os.remove(test_perm_file)
        print(f"✓ Write permissions confirmed for: {test_dir}")
    except Exception as e:
        print(f"✗ Write permission error: {e}")
    
    return len(saved_files) > 0

def check_thermal_analysis_paths():
    """Check if thermal analysis paths exist and are writable"""
    print("\nChecking Thermal Analysis Paths...")
    print("=" * 50)
    
    paths_to_check = [
        r'c:\Users\HP\Desktop\Research Intersnhip\CUDA\Codes',
        r'c:\Users\HP\Desktop\Research Intersnhip\CUDA\Codes\thermal_images',
        r'c:\Users\HP\Desktop\Research Intersnhip\CUDA\Codes\Tesla V100',
        r'c:\Users\HP\Desktop\Research Intersnhip\CUDA\Codes\Tesla T4'
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            if os.path.isdir(path):
                print(f"✓ Directory exists: {path}")
                # Check if it's writable
                try:
                    test_file = os.path.join(path, 'write_test.tmp')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    print(f"  ✓ Writable")
                except Exception as e:
                    print(f"  ✗ Not writable: {e}")
            else:
                print(f"✗ Exists but not a directory: {path}")
        else:
            print(f"✗ Does not exist: {path}")
            # Try to create it
            try:
                os.makedirs(path, exist_ok=True)
                print(f"  ✓ Created directory: {path}")
            except Exception as e:
                print(f"  ✗ Cannot create directory: {e}")

def check_csv_files():
    """Check if CSV files exist for thermal analysis"""
    print("\nChecking CSV Files...")
    print("=" * 50)
    
    csv_files = [
        r'c:\Users\HP\Desktop\Research Intersnhip\CUDA\Codes\Tesla V100\baseline_results.csv',
        r'c:\Users\HP\Desktop\Research Intersnhip\CUDA\Codes\Tesla T4\baseline_results.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                file_size = os.path.getsize(csv_file)
                print(f"✓ CSV file exists: {csv_file} ({file_size} bytes)")
            except Exception as e:
                print(f"✗ Error reading CSV file: {e}")
        else:
            print(f"✗ CSV file not found: {csv_file}")

def main():
    """Main debug function"""
    print("THERMAL ANALYSIS IMAGE SAVING DEBUG")
    print("=" * 60)
    
    # Test 1: Basic matplotlib functionality
    if test_matplotlib_saving():
        print("\n✓ Basic matplotlib saving works correctly")
    else:
        print("\n✗ Basic matplotlib saving failed")
        return
    
    # Test 2: Check thermal analysis paths
    check_thermal_analysis_paths()
    
    # Test 3: Check CSV files
    check_csv_files()
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    
    print("\nTROUBLESHOoting SUGGESTIONS:")
    print("1. If matplotlib backend issues: try 'pip install matplotlib --upgrade'")
    print("2. If permission issues: run as administrator or check folder permissions")
    print("3. If CSV files missing: ensure your CUDA experiments have generated CSV files")
    print("4. If directory issues: check path spelling and existence")
    
    print("\nNow try running your thermal analysis scripts:")
    print("python thermal_profiling_analysis.py")
    print("python kernel_thermal_analysis.py")
    print("python separate_thermal_images.py")

if __name__ == "__main__":
    main()
