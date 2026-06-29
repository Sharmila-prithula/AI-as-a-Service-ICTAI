import pynvml
import psutil
import time
import csv
import datetime
import argparse
import sys

# --- Configuration ---
POLL_INTERVAL_SECONDS = 1
GPU_INDEX = 0

# --- Global state for calculating IO rates ---
last_disk_io = psutil.disk_io_counters()
last_net_io = psutil.net_io_counters()
last_time = time.monotonic()

# --- FIX: Handle missing NVML_TEMP_GPU constant ---
try:
    # Use the official constant if available
    TEMPERATURE_SENSOR_TYPE = pynvml.NVML_TEMP_GPU
except AttributeError:
    # Fallback to the known integer value for GPU Core (typically 0)
    TEMPERATURE_SENSOR_TYPE = 0 
    print("Warning: pynvml.NVML_TEMP_GPU not found. Using integer 0 as fallback for temperature sensor.")


def get_system_metrics():
    """Fetches CPU, Memory, Disk, and Network stats using psutil."""
    global last_disk_io, last_net_io, last_time
    
    current_time = time.monotonic()
    duration = current_time - last_time
    # Set duration to 1 to avoid division by zero or overly large initial rates.
    if duration == 0:
        duration = POLL_INTERVAL_SECONDS 

    # CPU
    cpu_util_percent = psutil.cpu_percent(interval=None) # Get system-wide CPU usage

    # Memory
    mem_info = psutil.virtual_memory()
    memory_used_mb = mem_info.used // (1024**2)
    memory_total_mb = mem_info.total // (1024**2)

    # Disk IO
    current_disk_io = psutil.disk_io_counters()
    # Read/Write in MB/s
    disk_read_mb_s = (current_disk_io.read_bytes - last_disk_io.read_bytes) / (1024**2) / duration
    disk_write_mb_s = (current_disk_io.write_bytes - last_disk_io.write_bytes) / (1024**2) / duration
    last_disk_io = current_disk_io

    # Network IO
    current_net_io = psutil.net_io_counters()
    # RX/TX in MB/s
    network_rx_mb_s = (current_net_io.bytes_recv - last_net_io.bytes_recv) / (1024**2) / duration
    network_tx_mb_s = (current_net_io.bytes_sent - last_net_io.bytes_sent) / (1024**2) / duration
    last_net_io = current_net_io
    last_time = current_time

    return {
        "cpu_util_percent": round(cpu_util_percent, 2),
        "memory_used_mb": memory_used_mb,
        "memory_total_mb": memory_total_mb,
        "network_rx_mb_s": round(network_rx_mb_s, 2),
        "network_tx_mb_s": round(network_tx_mb_s, 2),
        "disk_read_mb_s": round(disk_read_mb_s, 2),
        "disk_write_mb_s": round(disk_write_mb_s, 2),
    }

def main():
    parser = argparse.ArgumentParser(description="GPU and System Statistics Monitor")
    parser.add_argument("--gpu-file", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--run-id", type=str, required=True, help="A unique ID for this benchmark run (e.g., vllm_c1_run1).")
    parser.add_argument("--concurrency", type=int, required=True, help="The concurrency level of the benchmark.")
    args = parser.parse_args()

    output_file = args.gpu_file

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)

        # Removed pcie_tx_mb_s and pcie_rx_mb_s from the header
        HEADER = [
            "timestamp", "run_id", "concurrency", "gpu_util_percent", "gpu_memory_used_mb",
            "gpu_memory_total_mb", "gpu_power_watts", "gpu_temperature", 
            "cpu_util_percent", "memory_used_mb", "memory_total_mb", 
            "network_rx_mb_s", "network_tx_mb_s", "disk_read_mb_s", "disk_write_mb_s"
        ]

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(HEADER)

            print(f"Starting GPU monitoring. Stats will be saved to {output_file}. Press Ctrl+C to stop.")

            while True:
                # NVML GPU stats
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_watts = power / 1000.0  # Convert mW to W
                
                # Use the global constant
                temperature = pynvml.nvmlDeviceGetTemperature(handle, TEMPERATURE_SENSOR_TYPE)

                # System stats
                system_metrics = get_system_metrics()

                row = [
                    datetime.datetime.now().isoformat(),
                    args.run_id,
                    args.concurrency,
                    utilization.gpu,
                    mem_info.used // (1024**2),
                    mem_info.total // (1024**2),
                    round(power_watts, 2),
                    temperature,
                    system_metrics["cpu_util_percent"],
                    system_metrics["memory_used_mb"],
                    system_metrics["memory_total_mb"],
                    system_metrics["network_rx_mb_s"],
                    system_metrics["network_tx_mb_s"],
                    system_metrics["disk_read_mb_s"],
                    system_metrics["disk_write_mb_s"],
                ]
                writer.writerow(row)
                
                time.sleep(POLL_INTERVAL_SECONDS)

    except pynvml.NVMLError as error:
        print(f"Failed to query NVML: {error}")
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped. GPU stats saved to {output_file}")
    except Exception as e:
        # Print a clearer error if it's not a known NVML issue
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        # Check if nvml has been initialized before shutting down
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass 

if __name__ == "__main__":
    main()
