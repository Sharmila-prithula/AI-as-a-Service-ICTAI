# AI-as-a-Service-ICTAI

### Prerequisites: Model Setup

Before running the benchmark, you must download the three evaluated models from Hugging Face and place them in a local directory named `/data/models`. The Docker commands below rely on this specific path for volume mounting. 

Ensure the following model repositories are downloaded and saved in `/data/models`:
*   `Llama-3.2-1B-Instruct`
*   `Llama-3.1-8B-Instruct`
*   `Mistral-7B-Instruct-v0.3`

---

### Reproducibility Guide

To reproduce the benchmark evaluations, you will need to open three separate terminal windows. This ensures the serving engine, hardware monitoring, and workload generation run concurrently without blocking one another.

#### Terminal 1: Start the Inference Engine (vLLM or TGI)
In the first terminal, launch the Docker container for your chosen serving engine (vLLM or TGI) and model. Below are the commands for the different configurations used in the study. Make sure your local model directory is correctly mounted (e.g., `-v /data/models:/data`).

**For vLLM:**
```bash
# Llama-3.2-1B-Instruct
docker run --rm --name vllm-server-docker --gpus all -p 8000:8000 \
  -v /data/models:/data/models vllm/vllm-openai:latest \
  --model /data/models/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8000

# Llama-3.1-8B-Instruct
docker run --rm --name vllm-server-docker --gpus all -p 8000:8000 \
  -v /data/models:/data/models vllm/vllm-openai:latest \
  --model /data/models/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000

# Mistral-7B-Instruct-v0.3
docker run --rm --name vllm-server-docker --gpus all --ipc=host -p 8000:8000 \
  -v /data/models:/models vllm/vllm-openai:latest \
  --model /models/Mistral-7B-Instruct-v0.3 \
  --max-model-len 8192 --dtype auto --gpu-memory-utilization 0.9
```
**For Text Generation Inference (TGI):**
```bash
# Llama-3.2-1B-Instruct
docker run --rm --name tgi-server-docker --gpus all --shm-size 1g -p 8000:80 \
  -v /data/models:/data ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /data/Llama-3.2-1B-Instruct

# Llama-3.1-8B-Instruct
sudo docker run --rm --name tgi-server-docker --gpus all --shm-size 1g -p 8000:80 \
  -v /data/models:/data ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /data/Llama-3.1-8B-Instruct

# Mistral-7B-Instruct-v0.3
docker run --rm --name tgi-server-docker --gpus all --shm-size 4g -p 8000:80 \
  -v /data/models:/data ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /data/Mistral-7B-Instruct-v0.3
```
### Terminal 2: Initialize GPU Monitoring
Once the server is running and ready to accept requests, open a second terminal to log hardware telemetry. Match the output file and concurrency arguments to your specific test run.
```bash
# Example 1: Monitoring Llama-3.1-8B on vLLM at Concurrency 1
python3 monitor_gpu.py \
  --gpu-file results/llama_8B/vllm/c1/gpu_stats_llma_8B_vllm_c1.csv \
  --run-id llma_8B_vllm_c1_run1 \
  --concurrency 1

# Example 2: Monitoring Mistral-7B on vLLM at Concurrency 8
python3 monitor_gpu.py \
  --gpu-file results/mistral/vllm/c8/gpu_stats_mistral_vllm_c8.csv \
  --run-id mistral_vllm_c8 \
  --concurrency 8
```

### Terminal 3: Execute the Benchmark Workload
Finally, in the third terminal, initiate the workload generator. Ensure the prompt file, model path, and concurrency levels match the configuration of your running server and monitor.

```Bash
# Example 1: Benchmarking Mistral-7B on vLLM at Concurrency 1
python3 run_benchmark.py \
  --model "/models/Mistral-7B-Instruct-v0.3" \
  --prompts-file "queries.jsonl" \
  --concurrency 1 \
  --reps 2 \
  --output-csv "results/mistral/vllm/c1/benchmark_results_mistral_vllm_c1.csv"

# Example 2: Benchmarking Llama-3.1-8B on vLLM at Concurrency 1
python3 run_benchmark.py \
  --model "/data/models/Llama-3.1-8B-Instruct" \
  --prompts-file "queries.jsonl" \
  --concurrency 1 \
  --reps 2 \
  --output-csv "results/llama_8B/vllm/c1/benchmark_results_llma_8B_vllm_c1.csv"
```
