from dgp.providers import NLLBProvider
from dgp.providers import ModelConfig

model = NLLBProvider()

response = model.invoke(
    "how are you doing?",
    system="",
    config=ModelConfig(
        model_name="facebook/nllb-200-distilled-600M",
        temperature=0.0,
        max_tokens=512
    )
)

print(response)