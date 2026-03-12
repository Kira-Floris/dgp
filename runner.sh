# running nllb-200-distilled-600M
# python -m examples.advanced.nllb_df \
#     data/split_dataset_100.csv results/facebook--nllb_200_3.3B.csv \
#     --max-new-tokens 1280 \
#     --batch-size 32 \
#     --device 0 \
#     --model facebook/nllb-200-3.3B

python -m examples.advanced.translategemma_df \
    data/split_dataset_100.csv results/google__translategemma-4b-it.csv \
    --max-new-tokens 1280 \
    --batch-size 32 \
    --device 0 \
    --model google/translategemma-4b-it