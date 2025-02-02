# Nano DeepSeek V3

This is a nano version of DeepSeek V3 (the actual model is 671B parameters) (https://github.com/deepseek-ai/deepseek-v3).

## Setup

Run the following commands to setup the environment and run the training script.

```
git clone https://github.com/sushantkumar23/nano-deepseek-v3
cd nano-deepseek-v3
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade
python cached_fineweb10B.py 10
torchrun --standalone --nproc_per_node=1 pretrain.py
```
