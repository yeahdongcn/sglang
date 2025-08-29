# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_musa.py install

# Install sglang python package
cd ..
pip install -e "python[all_musa]"
```

python3 -m sglang.launch_server --model-path /home/dist/DeepSeek-Coder-V2-Lite-Instruct/