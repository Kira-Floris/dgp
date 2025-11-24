import fasttext
from huggingface_hub import hf_hub_download

metadata = {
    "english": {
        "fasttext_label": "__label__eng_Latn",
    },
    "kinyarwanda": {
        "fasttext_label": "__label__kin_Latn",
    },
}

facebook_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")

def detect_language(text: str, language: str="kinyarwanda"):
    global facebook_model_path
    model = fasttext.load_model(facebook_model_path)
    predictions = model.predict(text)
    predicted_label = predictions[0][0]
    return predicted_label == metadata.get(language, {}).get("fasttext_label", "")

