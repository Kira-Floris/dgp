# running nllb-200-distilled-600M
# python -m examples.advanced.nllb_df \
#     data/split_dataset_100.csv results/DigitalUmuganda--finetuned_nllb_1.3B.csv \
#     --max-new-tokens 1280 \
#     --batch-size 64 \
#     --device 0 \
#     --model DigitalUmuganda/finetuned-nllb-1.3B

# python -m examples.advanced.translategemma_df \
#     data/split_dataset_100.csv results/google--translategemma_27b_it.csv \
#     --max-new-tokens 1280 \
#     --batch-size 128 \
#     --device 0 \
#     --model google/translategemma-27b-it

# CUDA_VISIBLE_DEVICES=0,1,2,3 OLLAMA_NUM_PARALLEL=256 OLLAMA_MAX_QUEUE=512 OLLAMA_FLASH_ATTENTION=1 nohup ollama serve > ~/ollama.log 2>&1 &

# ollama run gpt-oss:120b

# python -m examples.advanced.ollama_df data/split_dataset_100.csv results/openai--gpt_oss_120b.csv \
#   --model gpt-oss:120b \
#   --concurrency 256 \
#   --lang-col language \
#   --text-col text
python -m examples.advanced.ollama_df data/sample_10.csv results/sample_100.csv \
  --model gpt-oss:120b \
  --concurrency 256 \
  --lang-col language \
  --text-col text