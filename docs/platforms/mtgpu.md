# Compile sgl-kernel
```bash
pip install --upgrade pip
cd sgl-kernel
python setup_musa.py install
```

# Install sglang python package
```bash
cd ..
pip install -e "python[all_musa]"
```

# Run
```bash
# 1xGPU without graph - OK
python3 -m sglang.launch_server --model-path /home/dist/Qwen3-0.6B/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --disable-cuda-graph

# 8xGPU without graph - OK
python3 -m sglang.launch_server --model-path /home/dist/Qwen3-8B/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --tp 8 --disable-cuda-graph

# 1xGPU with graph - NOK (MUSA error: page fault)
python3 -m sglang.launch_server --model-path /home/dist/Qwen3-0.6B/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --cuda-graph-max-bs 1

# 8xGPU with graph - NOK (MUSA error: page fault)
python3 -m sglang.launch_server --model-path /home/dist/Qwen3-0.6B/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --tp 8 --cuda-graph-max-bs 1

# MoE: 2xGPU without graph - NOK (Garbled)
python3 -m sglang.launch_server --model-path /home/dist/DeepSeek-Coder-V2-Lite-Instruct/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --tp 2 --disable-cuda-graph --disable-radix-cache

# MoE: 2xGPU with graph - NOK (Garbled)
python3 -m sglang.launch_server --model-path /home/dist/DeepSeek-Coder-V2-Lite-Instruct/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --tp 2 --cuda-graph-max-bs 1 --disable-radix-cache

# Dense: 2xGPU without graph - OK
python3 -m sglang.launch_server --model-path /home/dist/DeepSeek-R1-Distill-Qwen-7B/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --tp 2 --disable-cuda-graph

# MoE: 2xGPU without graph - OK
python3 -m sglang.launch_server --model-path /home/dist/Qwen3-30B-A3B-Instruct-2507/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --tp 2 --disable-cuda-graph
```

# Profile

```bash
export SGLANG_TORCH_PROFILER_DIR=/ws/xx
curl http://localhost:43434/start_profile -H "Content-Type: application/json"
# Chat or API call
curl http://localhost:43434/stop_profile -H "Content-Type: application/json"
```