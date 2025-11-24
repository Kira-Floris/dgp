# install conda
wget -O ~/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /home/ubuntu/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# clone vllm
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

# run docker setup for cpu
sudo DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai --file docker/Dockerfile
sudo docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model openai/gpt-oss-120b --tensor-parallel-size 8
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