# function to load large jsonl file in a memory efficient way
def load_jsonl_in_chunks(file_path: str, chunk_size: int = 1000):
    """
    Load a large JSONL file in chunks to avoid memory issues.
    
    Args:
        file_path: Path to the JSONL file
        chunk_size: Number of lines to read per chunk       
    Yields:
        A list of JSON objects for each chunk
    """
    import json

    with open(file_path, "r", encoding="utf-8") as f:
        chunk = []
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk