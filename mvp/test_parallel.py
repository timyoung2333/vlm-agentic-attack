from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import time

client = OpenAI(api_key='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')

def call_api(prompt):
    return client.responses.create(
        model="gpt-4",
        input=[{"role": "user", "content": prompt}]
    )

prompts_list = ["Hello"] * 10

# Serial Execution
print("Starting Serial Execution...")
start_time = time.time()
results_serial = []
for prompt in prompts_list:
    results_serial.append(call_api(prompt))
end_time = time.time()
serial_time = end_time - start_time
print(f"Serial execution time: {serial_time:.2f} seconds")

# Parallel Execution
print("Starting Parallel Execution...")
start_time = time.time()
with ThreadPoolExecutor(max_workers=10) as executor:
    results_parallel = list(executor.map(call_api, prompts_list))
end_time = time.time()
parallel_time = end_time - start_time
print(f"Parallel execution time: {parallel_time:.2f} seconds")

print(f"Speedup: {serial_time / parallel_time:.2f}x")

# for result in results_parallel:
#     print(result.output_text)