# install conda
# wget -O ~/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash /home/ubuntu/Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc

# # clone vllm
# git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

# run docker setup for cpu
sudo DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai --file docker/Dockerfile
# sudo docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model openai/gpt-oss-120b --tensor-parallel-size 8
# OPTIMIZED RUN
# sudo docker run -d --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host --shm-size=10g vllm/vllm-openai:latest --model openai/gpt-oss-120b --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.90 --enforce-eager --disable-log-requests --max-num-batched-tokens 8192 --block-size 16 --swap-space 4
# sudo docker run -d --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host --shm-size=10g vllm/vllm-openai:latest --model openai/gpt-oss-20b --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.90 --enforce-eager --disable-log-requests --max-num-batched-tokens 8192 --block-size 16 --swap-space 4
# sudo docker run -d --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host --shm-size=10g vllm/vllm-openai:latest --model google/gemma-3-27b-it --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.90 --enforce-eager --disable-log-requests --max-num-batched-tokens 8192 --block-size 16 --swap-space 4
# sudo docker run -d --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host --shm-size=10g vllm/vllm-openai:latest --model google/gemma-3-4b-it --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.90 --enforce-eager --disable-log-requests --max-num-batched-tokens 8192 --block-size 16 --swap-space 4
# sudo docker run -d --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host --shm-size=10g vllm/vllm-openai:latest --model Kira-Floris/Qwen3-4B-Stage1CPT-ckpt3500 --tensor-parallel-size 8 --max-model-len 2048 --max-num-seqs 256 --gpu-memory-utilization 0.90 --enforce-eager --disable-log-requests --max-num-batched-tokens 8192 --block-size 16 --swap-space 4
# sudo docker run -d --runtime nvidia --gpus '"device=3"' -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host --shm-size=10g vllm/vllm-openai:latest --model Kira-Floris/Gemma3-1B-Stage1CPT-ckpt4000 --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 256 --gpu-memory-utilization 0.90 --enforce-eager --disable-log-requests --max-num-batched-tokens 8192 --block-size 16 --swap-space 4
sudo docker run -d --runtime nvidia --gpus '"device=0,1,2,3"' -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 4000:4000 --ipc=host --shm-size=10g vllm/vllm-openai:latest --model Kira-Floris/Qwen3-4B --tensor-parallel-size 4 --max-model-len 2048 --max-num-seqs 256 --gpu-memory-utilization 0.90 --enforce-eager --disable-log-requests --max-num-batched-tokens 8192 --block-size 16 --swap-space 4
cd ..

# environment setup script
conda create -n dgp_env python=3.12 -y
conda activate dgp_env
pip install uv==0.9.9
uv pip install fasttext
uv pip install transformers
uv pip install datasets
uv pip install sacrebleu
uv pip install openai
uv pip install unbabel-comet
uv pip install groq>=0.37.0