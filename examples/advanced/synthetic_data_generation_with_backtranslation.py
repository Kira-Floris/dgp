from dgp.providers import GroqProvider
from dgp.providers import ModelConfig
from dgp.metrics import BLEUScore, COMETMetric
from dgp.tasks.backtranslation import BackTranslationPipeline

pipeline = BackTranslationPipeline(
    provider=GroqProvider(),
    metrics=[BLEUScore(max_order=4), COMETMetric()],
    model_config=ModelConfig(
        model_name="openai/gpt-oss-120b",
        temperature=0.0
    )
)

source_lang = "kinyarwanda"
intermediate_lang = "english"

sentences = [
    "amakuru yawe?",
    "umeze gute?",
    "tujye kwa muganga"
]

for sentence in sentences:
    result = pipeline.run(
        text=sentence,
        source_lang=source_lang,
        intermediate_lang=intermediate_lang,
        system_template="Translate the following text from {src_lang} to {tgt_lang}. Return the translated text only."
    )
    print(result)

# print(result)