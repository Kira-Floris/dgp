# ubuntu setup
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev python3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# environment setup script
conda create -n dgp_env python=3.12 -y
conda activate dgp_env

# install uv
pip install uv==0.9.9

# install vllm
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

uv pip install "cmake>=3.26.1"

uv pip install -r requirements/cpu-build.txt --torch-backend cpu
uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match

uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation

# launch vllm service on cpu
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-30
export VLLM_PLUGINS=ascend,ascend_enhanced_model
vllm serve google/gemma-3-270m --dtype=bfloat16

# launch using docker
sudo docker build -f docker/Dockerfile.cpu --build-arg VLLM_CPU_AVX512BF16=false --build-arg VLLM_CPU_AVX512VNNI=false --build-arg VLLM_CPU_DISABLE_AVX512=false --tag vllm-cpu-env --target vllm-openai .

sudo docker run --rm --security-opt seccomp=unconfined --cap-add SYS_NICE --shm-size=4g -p 8000:8000 -e VLLM_CPU_KVCACHE_SPACE=40 -e VLLM_CPU_OMP_THREADS_BIND=0-8 -e HF_TOKEN=${HF_TOKEN} vllm-cpu-env --model=google/gemma-3-270m --dtype=bfloat16