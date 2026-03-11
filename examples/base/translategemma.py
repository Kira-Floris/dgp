from dgp.providers import TranslateGemmaProvider
from dgp.config import ModelConfig

model = TranslateGemmaProvider(
    src_lang="en",
    tgt_lang="rw",
)

response = model.invoke(
    "how are you doing?",
    system="",
    config=ModelConfig(
        model_name="google/translategemma-4b-it",
        temperature=0.0,
        max_tokens=512,
    )
)

print(response)