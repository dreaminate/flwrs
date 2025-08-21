from __future__ import annotations
import os, json, hashlib
from typing import Any, Dict

def ensure_dir_for(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def log_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir_for(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def cfg_hash(d: Dict[str, Any], length: int = 16) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:length]

def anon(s: str, length: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:length]
