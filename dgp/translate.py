from openai import OpenAI
import os

PROMPT = """Translate the following text from {source_language} to {target_language}. Return ONLY the translation, nothing else. No explanations, no labels, no additional text.

Text: {sentence}

Translation:"""

def translate_text(
        sentence: str, 
        model: str="google/gemma-3-270m",
        source_language: str = "English",
        target_language: str = "French", 
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        base_url: str="http://localhost:8000/v1",
        api_key: str=None) -> str:
    
    if base_url:
        client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"  # vLLM doesn't require API key
        )
    else:
        client = OpenAI(api_key=api_key if api_key else os.getenv("OPENAI_API", None))
    
    # Craft a strict prompt that instructs model to return ONLY the translation
    global PROMPT
    prompt = PROMPT.format(
        source_language=source_language,
        target_language=target_language,
        sentence=sentence
    )
    
    try:
        # response = client.completions.create(
        #     model=model,
        #     prompt=prompt,
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     top_p=top_p,
        #     stop=["\n\n", "Text:", "Translation:"]  # Stop at these tokens
        # )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["\n\n", "Text:", "Translation:"]  # Stop at these tokens
        )
        
        # Extract and clean the translation
        translation = response.choices[0].message.content.strip()
        
        # Remove any potential labels that might have slipped through
        if translation.startswith(("Translation:", "translation:", "Answer:", "answer:")):
            translation = translation.split(":", 1)[1].strip()
        
        return translation
    
    except Exception as e:
        return f"Error: {str(e)}"