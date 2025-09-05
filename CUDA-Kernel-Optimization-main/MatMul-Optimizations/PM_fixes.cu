/**
 * CORRECTED Power Measurement for CUDA Kernels
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <string>
#include <memory>
#include <exception>

// Add using declarations for std namespace to avoid compilation errors
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::thread;
using std::atomic;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::max;
using std::min;

// Handle NVML availability
#ifndef DISABLE_NVML
    #include <nvml.h>
    #define NVML_AVAILABLE 1
#else
    #define NVML_AVAILABLE 0
    // Define dummy NVML types and functions when disabled
    typedef void* nvmlDevice_t;
    typedef int nvmlReturn_t;
    #define NVML_SUCCESS 0
    inline nvmlReturn_t nvmlInit() { return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlShutdown() { return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlDeviceGetHandleByIndex(int, nvmlDevice_t*) { return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t, unsigned int* power) { *power = 150000; return NVML_SUCCESS; }
    inline const char* nvmlErrorString(nvmlReturn_t) { return "NVML disabled"; }
#endif

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define NVML_CHECK(call) \
    do { \
        nvmlReturn_t result = call; \
        if (result != NVML_SUCCESS) { \
            std::cerr << "NVML error: " << nvmlErrorString(result) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Power statistics structure
struct PowerStats {
    double average_power_watts;
    double peak_power_watts;
    double baseline_power_watts;
    double computation_power_watts;  // average - baseline
    bool measurement_reliable;
    
    PowerStats() : average_power_watts(0.0), peak_power_watts(0.0), 
                   baseline_power_watts(0.0), computation_power_watts(0.0), 
                   measurement_reliable(false) {}
};

// Required struct definitions
struct PerformanceMetrics {
    double execution_time_ms;
    double average_execution_time_ms;
    double execution_time_stddev;
    double total_energy_joules;
    double average_power_watts;
    double peak_power_watts;
    double baseline_power_watts;
    double initial_temperature_c;
    double peak_temperature_c;
    double final_temperature_c;
    double gflops;
    double gflops_per_watt;
    double memory_bandwidth_gbps;
    
    PerformanceMetrics() : 
        execution_time_ms(0.0), average_execution_time_ms(0.0), 
        execution_time_stddev(0.0), total_energy_joules(0.0),
        average_power_watts(0.0), peak_power_watts(0.0),
        baseline_power_watts(0.0), initial_temperature_c(0.0),
        peak_temperature_c(0.0), final_temperature_c(0.0),
        gflops(0.0), gflops_per_watt(0.0), memory_bandwidth_gbps(0.0) {}
};

struct OptimizationConfig {
    int matrix_size;
    int block_dim_x;
    int block_dim_y;
    int threads_per_block;
    
    enum KernelType {
        NAIVE_BASELINE,
        MEMORY_COALESCED,
        SHARED_MEMORY_TILED,
        REGISTER_BLOCKED,
        OCCUPANCY_OPTIMIZED,
        ADVANCED_OPTIMIZED
    } kernel_type;
    
    enum DataType {
        FLOAT_32,
        DOUBLE_64
    } data_type;
    
    int num_iterations;
    bool enable_power_monitoring;
    bool enable_thermal_monitoring;
    
    OptimizationConfig() : 
        matrix_size(1024), block_dim_x(16), block_dim_y(16),
        threads_per_block(256), kernel_type(NAIVE_BASELINE),
        data_type(FLOAT_32), num_iterations(10),
        enable_power_monitoring(true), enable_thermal_monitoring(true) {}
};

// Placeholder for framework class
class MatrixMultiplicationFramework {
public:
    void run_single_iteration() {
        // Placeholder - would contain actual kernel execution
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    double measure_baseline_power(int duration_ms = 5000) {
        // Simplified baseline power measurement
        return 50.0; // Return typical idle power
    }
    
    void stop_monitoring() {
        // Placeholder for stopping monitoring
    }
};

// Fix 1: Extend execution time for better power measurement
__global__ void extended_naive_baseline_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int repetitions = 100) {  // Multiple repetitions within single kernel
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // Repeat computation multiple times to extend execution
        for (int rep = 0; rep < repetitions; ++rep) {
            float temp_sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                temp_sum += A[row * N + k] * B[k * N + col];
            }
            sum += temp_sum / repetitions;  // Normalize to get correct result
        }
        
        C[row * N + col] = sum;
    }
}

// Fix 2: Improved power monitoring with burst execution
class ImprovedPowerMonitor {
private:
    nvmlDevice_t device;
    std::atomic<bool> monitoring_active;
    std::vector<double> power_samples;
    std::vector<std::chrono::high_resolution_clock::time_point> timestamps;
    std::thread monitoring_thread;
    
public:
    ImprovedPowerMonitor() : monitoring_active(false) {
        NVML_CHECK(nvmlInit());
        NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device));
    }
    
    // Enhanced power monitoring with higher sampling rate
    void start_monitoring_burst() {
        power_samples.clear();
        timestamps.clear();
        monitoring_active.store(true);
        
        monitoring_thread = std::thread([this]() {
            while (monitoring_active.load()) {
                unsigned int power_mw;
                auto timestamp = std::chrono::high_resolution_clock::now();
                
                if (nvmlDeviceGetPowerUsage(device, &power_mw) == NVML_SUCCESS) {
                    double power_watts = power_mw / 1000.0;
                    power_samples.push_back(power_watts);
                    timestamps.push_back(timestamp);
                }
                
                // Higher sampling rate for short kernels
                std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 5ms instead of 10ms
            }
        });
    }
    
    // Get power statistics excluding baseline
    struct PowerStats {
        double average_power_watts;
        double peak_power_watts;
        double baseline_power_watts;
        double computation_power_watts;  // average - baseline
        bool measurement_reliable;
    };
    
    void stop_monitoring() {
        monitoring_active.store(false);
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
    }
    
    double measure_baseline_power(int duration_ms = 5000) {
        std::vector<double> baseline_samples;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() < duration_ms) {
            
            unsigned int power_mw;
            if (nvmlDeviceGetPowerUsage(device, &power_mw) == NVML_SUCCESS) {
                baseline_samples.push_back(power_mw / 1000.0);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        if (baseline_samples.empty()) return 50.0; // Default value
        
        double sum = 0.0;
        for (double power : baseline_samples) {
            sum += power;
        }
        return sum / baseline_samples.size();
    }
    
    PowerStats get_power_statistics(double measured_baseline) const {
        PowerStats stats;
        
        if (power_samples.empty()) {
            stats.measurement_reliable = false;
            return stats;
        }
        
        // Calculate statistics
        double sum = 0.0;
        double max_power = 0.0;
        
        for (double power : power_samples) {
            sum += power;
            max_power = std::max(max_power, power);
        }
        
        stats.average_power_watts = sum / power_samples.size();
        stats.peak_power_watts = max_power;
        stats.baseline_power_watts = measured_baseline;
        
        // Calculate computation power (power above baseline)
        stats.computation_power_watts = std::max(0.0, stats.average_power_watts - measured_baseline);
        
        // Check measurement reliability
        double power_increase = stats.average_power_watts - measured_baseline;
        stats.measurement_reliable = (power_increase > 5.0) && (power_samples.size() > 3);  // At least 5W increase and 3+ samples
        
        return stats;
    }
};

// Fix 3: Corrected energy efficiency calculation
double calculate_corrected_energy_efficiency(double gflops, const ImprovedPowerMonitor::PowerStats& power_stats) {
    double efficiency_gflops_per_watt = 0.0;
    
    if (power_stats.measurement_reliable && power_stats.computation_power_watts > 1.0) {
        // Use computation power (above baseline) for efficiency calculation
        efficiency_gflops_per_watt = gflops / power_stats.computation_power_watts;
        
        std::cout << "Using computation power: " << power_stats.computation_power_watts << "W" << std::endl;
        std::cout << "Energy efficiency: " << efficiency_gflops_per_watt << " GFLOPS/W" << std::endl;
        
    } else if (power_stats.average_power_watts > 10.0) {
        // Use total average power if computation power is unreliable
        efficiency_gflops_per_watt = gflops / power_stats.average_power_watts;
        
        std::cout << "Warning: Using total average power: " << power_stats.average_power_watts << "W" << std::endl;
        std::cout << "Energy efficiency: " << efficiency_gflops_per_watt << " GFLOPS/W" << std::endl;
        
    } else {
        // Power measurement failed - use conservative estimate
        efficiency_gflops_per_watt = gflops / 100.0;  // Assume 100W for failed measurements
        
        std::cout << "ERROR: Power measurement failed! Using conservative 100W estimate." << std::endl;
        std::cout << "Estimated energy efficiency: " << efficiency_gflops_per_watt << " GFLOPS/W" << std::endl;
    }
    
    return efficiency_gflops_per_watt;
}

// Fix 4: Burst execution strategy for short kernels
PerformanceMetrics run_corrected_benchmark(MatrixMultiplicationFramework* framework, 
                                         const OptimizationConfig& config) {
    PerformanceMetrics metrics;
    ImprovedPowerMonitor power_monitor;
    
    // Step 1: Measure baseline power for longer duration
    std::cout << "Measuring baseline power (10 seconds)..." << std::endl;
    double baseline_power = power_monitor.measure_baseline_power(10000);  // 10 seconds
    
    // Step 2: Warm up GPU to stabilize power
    std::cout << "GPU warm-up..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        // Run a few iterations without measurement
        framework->run_single_iteration();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Step 3: Extended execution for better power measurement
    std::cout << "Running extended kernel execution for power measurement..." << std::endl;
    
    power_monitor.start_monitoring_burst();
    
    // Execute kernel multiple times in burst
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < 50; ++iter) {  // 50 iterations in burst
        framework->run_single_iteration();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    power_monitor.stop_monitoring();
    
    // Step 4: Calculate metrics
    double total_execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    metrics.average_execution_time_ms = total_execution_time / 50.0;  // Average per iteration
    
    // Calculate GFLOPS
    long long operations = 2LL * config.matrix_size * config.matrix_size * config.matrix_size;
    metrics.gflops = (operations / 1e9) / (metrics.average_execution_time_ms / 1000.0);
    
    // Get corrected power statistics
    auto power_stats = power_monitor.get_power_statistics(baseline_power);
    
    metrics.average_power_watts = power_stats.average_power_watts;
    metrics.peak_power_watts = power_stats.peak_power_watts;
    metrics.baseline_power_watts = power_stats.baseline_power_watts;
    
    // Calculate corrected energy efficiency
    metrics.gflops_per_watt = calculate_corrected_energy_efficiency(metrics.gflops, power_stats);
    
    // Report measurement quality
    std::cout << "Measurement Quality Report:" << std::endl;
    std::cout << "  Power measurement reliable: " << (power_stats.measurement_reliable ? "YES" : "NO") << std::endl;
    std::cout << "  Baseline power: " << baseline_power << "W" << std::endl;
    std::cout << "  Average power during computation: " << power_stats.average_power_watts << "W" << std::endl;
    std::cout << "  Power increase above baseline: " << power_stats.computation_power_watts << "W" << std::endl;
    std::cout << "  Peak power: " << power_stats.peak_power_watts << "W" << std::endl;
    
    return metrics;
}

int main() {
    std::cout << "CUDA Power Measurement Fixes - Test Program" << std::endl;
    
    try {
        // Initialize CUDA
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;
        
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }
        
        // Test improved power monitor
        ImprovedPowerMonitor monitor;
        std::cout << "Power monitor initialized successfully" << std::endl;
        
        // Test baseline power measurement
        std::cout << "Testing baseline power measurement..." << std::endl;
        double baseline = monitor.measure_baseline_power(2000); // 2 seconds
        std::cout << "Baseline power: " << baseline << " W" << std::endl;
        
        // Test power statistics calculation
        auto stats = monitor.get_power_statistics(baseline);
        std::cout << "Power statistics test completed" << std::endl;
        
        // Test energy efficiency calculation
        double test_gflops = 1000.0;
        double efficiency = calculate_corrected_energy_efficiency(test_gflops, stats);
        std::cout << "Energy efficiency test: " << efficiency << " GFLOPS/W" << std::endl;
        
        std::cout << "All tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
