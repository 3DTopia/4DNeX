# core/petrel_fallback.py
import urllib.request
from pathlib import Path

class Client:
    """Minimal fallback for petrel_client.client.Client.
    Supports local paths and http(s) URLs: Client.get(path) -> bytes
    """
    def __init__(self, *args, **kwargs):
        pass

    def get(self, path: str) -> bytes:
        if path.startswith("http://") or path.startswith("https://"):
            with urllib.request.urlopen(path) as r:
                return r.read()
        p = Path(path)
        return p.read_bytes()
