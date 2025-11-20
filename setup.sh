# clone vllm
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

# run docker setup for cpu
sudo docker build -f docker/Dockerfile.cpu --build-arg VLLM_CPU_AVX512BF16=false --build-arg VLLM_CPU_AVX512VNNI=false --build-arg VLLM_CPU_DISABLE_AVX512=false --tag vllm-cpu-env --target vllm-openai .
sudo docker run --rm --security-opt seccomp=unconfined --cap-add SYS_NICE --shm-size=4g -p 8000:8000 -e VLLM_CPU_KVCACHE_SPACE=40 -e VLLM_CPU_OMP_THREADS_BIND=0-8 -e HF_TOKEN=${HF_TOKEN} vllm-cpu-env --model=google/gemma-3-270m --dtype=bfloat16
cd ..

# environment setup script
conda activate dgp_env
uv pip install unbabel-comet
uv pip install fasttext
uv pip install transformers
