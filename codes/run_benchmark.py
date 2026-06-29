import asyncio
import aiohttp
import aiofiles
import time
import json
import argparse
import csv
import random
import datetime
import os
import statistics

def approximate_tokens(text):
    return len(text) // 4

async def send_request(session, semaphore, prompt_details, base_url, model_name, run_id, concurrency, iteration):
    # Extract Data
    prompt_id = prompt_details.get("prompt_id", "unknown")
    category = prompt_details.get("category", "unknown")
    sub_category = prompt_details.get("sub_category", "unknown")
    difficulty = prompt_details.get("difficulty", "unknown")
    prompt_text = prompt_details.get("prompt", "")
    ref_answer = prompt_details.get("reference_answer", "")

    # Normalize URL
    url = base_url.rstrip('/')
    if not url.endswith("/v1/chat/completions") and not url.endswith("/generate"):
        url = f"{url}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 1024,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    start_time = time.monotonic()
    ttft = 0.0
    first_token_time = None
    status = "success"
    error_message = ""
    completion_text = ""
    
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    
    try:
        async with semaphore:
            async with session.post(url, headers=headers, json=payload, timeout=180) as response:
                if response.status != 200:
                    status = "error"
                    error_message = await response.text()
                    print(f"Error {response.status}: {error_message[:100]}...")
                
                if status == "success":
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or line == "data: [DONE]":
                            continue
                        
                        if line.startswith("data: "):
                            json_str = line[6:]
                            try:
                                data = json.loads(json_str)
                                
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', "")
                                    
                                    if content:
                                        if first_token_time is None:
                                            first_token_time = time.monotonic()
                                            ttft = first_token_time - start_time
                                        completion_text += content

                                if 'usage' in data and data['usage']:
                                    prompt_tokens = data['usage'].get('prompt_tokens', 0)
                                    completion_tokens = data['usage'].get('completion_tokens', 0)
                                    total_tokens = data['usage'].get('total_tokens', 0)

                            except json.JSONDecodeError:
                                continue
    except Exception as e:
        status = "error"
        error_message = str(e)
        print(f"Exception: {e}")

    end_time = time.monotonic()
    total_latency = end_time - start_time
    
    # Fallbacks
    if first_token_time is None:
        ttft = total_latency
    if completion_tokens == 0 and len(completion_text) > 0:
        completion_tokens = approximate_tokens(completion_text)
        prompt_tokens = approximate_tokens(prompt_text)
        total_tokens = prompt_tokens + completion_tokens

    req_throughput = 0.0
    if total_latency > 0 and completion_tokens > 0:
        req_throughput = completion_tokens / (total_latency-ttft)

    avg_itl = 0.0
    if completion_tokens > 1:
        generation_time = total_latency - ttft
        avg_itl = max(0.0, generation_time) / (completion_tokens - 1)

    response_log = error_message if status == "error" else completion_text

    # --- RETURN DICTIONARY (CSV Row) ---
    # I re-ordered this so 'prompt' is near the top for visibility
    return {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt_id": prompt_id,
        "prompt": prompt_text,       # <--- MOVED HERE (Column 4)
        "response": response_log,  # <--- MOVED HERE (Column 5)
        "reference_answer": ref_answer,
	"category": category,
        "sub_category": sub_category,
        "difficulty": difficulty,
        "concurrency": concurrency,
        "iteration": iteration,
        "status": status,
        "total_latency_s": round(total_latency, 4),
        "ttft_s": round(ttft, 4),
	"generation_time": round(generation_time, 2),
        "req_throughput_tok_per_s": round(req_throughput, 2),
        "avg_itl_s": round(avg_itl, 5),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "system_rps": 0.0,
        "system_tok_per_s": 0.0
    }

async def main(args):
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"--- Benchmark: {args.model} | Concurrency: {args.concurrency} ---")
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    
    prompts = []
    print(f"Loading prompts from: {args.prompts_file}")
    try:
        async with aiofiles.open(args.prompts_file, mode='r') as f:
            content = await f.read()
            content = content.strip()
            
            try:
                prompts = json.loads(content)
                if not isinstance(prompts, list):
                    prompts = [prompts]
            except json.JSONDecodeError:
                prompts = []
                lines = content.splitlines()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    if line.endswith(","):
                        line = line[:-1]
                    try:
                        prompts.append(json.loads(line))
                    except:
                        pass

    except FileNotFoundError:
        print(f"Error: File '{args.prompts_file}' not found.")
        return

    if not prompts:
        print("Error: No valid prompts found in file.")
        return

    task_list = []
    for i in range(args.reps):
        for p in prompts:
            task_list.append((p, i + 1))
            
    random.shuffle(task_list)
    print(f"Queue size: {len(task_list)} requests")

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, semaphore, p, args.url, args.model, run_id, args.concurrency, i) 
            for p, i in task_list
        ]
        
        start_global = time.monotonic()
        results = await asyncio.gather(*tasks)
        end_global = time.monotonic()
        
    # --- Global Metrics ---
    total_duration = end_global - start_global
    successful_results = [r for r in results if r["status"] == "success"]
    
    total_gen_tokens = sum(r['completion_tokens'] for r in successful_results)
    total_reqs = len(successful_results)
    
    global_rps = 0.0
    if total_duration > 0:
        global_rps = total_reqs / total_duration

    system_tok_per_s = 0.0
    if total_duration > 0:
        system_tok_per_s = total_gen_tokens / total_duration

    print(f"\n{'-'*40}")
    print(f"Benchmark Summary")
    print(f"{'-'*40}")
    print(f"Total Duration    : {total_duration:.2f}s")
    print(f"System RPS        : {global_rps:.2f} req/s")
    print(f"System Throughput : {system_tok_per_s:.2f} tokens/s")

    if successful_results:
        def get_stats(data_list):
            if not data_list: return 0, 0, 0, 0
            return (
                statistics.mean(data_list),
                statistics.median(data_list),
                sorted(data_list)[int(len(data_list) * 0.95)],
                sorted(data_list)[int(len(data_list) * 0.99)]
            )

        ttfts = [r["ttft_s"] for r in successful_results]
        t_mean, t_med, t_95, t_99 = get_stats(ttfts)

        print(f"\nMetric (Per Req) | Mean    | P50     | P95     | P99")
        print(f"-----------------+---------+---------+---------+---------")
        print(f"TTFT (s)         | {t_mean:<7.4f} | {t_med:<7.4f} | {t_95:<7.4f} | {t_99:<7.4f}")

        # Inject global metrics
        for res in results:
            res["system_rps"] = round(global_rps, 2)
            res["system_tok_per_s"] = round(system_tok_per_s, 2)

        keys = results[0].keys()
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved to: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--output-csv", required=True)
    
    args = parser.parse_args()
    asyncio.run(main(args))
