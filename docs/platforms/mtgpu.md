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
python3 -m sglang.launch_server --model-path /home/dist/Qwen3-0.6B/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --disable-cuda-graph
python3 -m sglang.launch_server --model-path /home/dist/Qwen3-8B/ --trust-remote-code --host 0.0.0.0 --nnodes 1 --log-level debug --port 43434 --tp 8 --disable-cuda-graph
```