# running nllb-200-distilled-600M
# python -m examples.advanced.nllb_df \
#     data/split_dataset_100.csv results/DigitalUmuganda--finetuned_nllb_1.3B.csv \
#     --max-new-tokens 1280 \
#     --batch-size 64 \
#     --device 0 \
#     --model DigitalUmuganda/finetuned-nllb-1.3B

python -m examples.advanced.translategemma_df \
    data/split_dataset_100.csv results/Kira_Floris--TranslateGemma_4B.csv \
    --max-new-tokens 1280 \
    --batch-size 32 \
    --device 0 \
    --model Kira-Floris/TranslateGemma-4B