"""Pinned CPU inference without PyTorch. Downloads are explicit and hash checked."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

LOCK = Path(__file__).with_name("models.lock.json")

def model_directory():
    return Path(os.getenv("RAG_MODEL_DIR", str(Path(__file__).resolve().parents[1]/"data/models")))

def artifact_identity():
    return sha256(LOCK.read_bytes()).hexdigest()

def verify(data, entry):
    if len(data) != entry["size"]:
        return False
    digest = sha256(data).hexdigest() if entry["checksum_type"] == "sha256" else sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
    return digest == entry["checksum"]

def download():
    directory = model_directory()
    for model in json.loads(LOCK.read_text()):
        for entry in model["files"]:
            target = directory/model["repo"]/model["revision"]/entry["path"]
            if target.exists() and verify(target.read_bytes(), entry):
                continue
            url = f'https://huggingface.co/{model["repo"]}/resolve/{model["revision"]}/{entry["path"]}'
            size = entry["size"]
            # Bounded parallel ranges keep large public downloads practical on slow links.
            parts = min(16, max(1, (size + 1048575)//1048576))
            width = (size + parts - 1)//parts
            def fetch(index):
                start, end = index*width, min(size-1, (index+1)*width-1)
                request = Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urlopen(request, timeout=120) as response:
                    if parts > 1 and response.status != 206:
                        raise RuntimeError("Model host did not honor the requested byte range")
                    data = response.read()
                if len(data) != end-start+1:
                    raise RuntimeError("Incomplete model download")
                return data
            with ThreadPoolExecutor(max_workers=parts) as pool:
                data = b"".join(pool.map(fetch, range(parts)))
            if not verify(data, entry):
                raise RuntimeError(f'Checksum mismatch: {model["repo"]}/{entry["path"]}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+".part")
            temporary.write_bytes(data)
            temporary.replace(target)
            print(f'Verified {model["repo"]}/{entry["path"]}', flush=True)

class CPUModel:
    def __init__(self, model):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        directory = model_directory()/model["repo"]/model["revision"]
        for entry in model["files"]:
            path = directory/entry["path"]
            if not path.exists() or not verify(path.read_bytes(), entry):
                raise ValueError("Missing or invalid model files. Run: python -m core.onnx_models --download")
        self.tokenizer = Tokenizer.from_file(str(directory/"tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory/"onnx/model_quint8_avx2.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = {entry.name for entry in self.session.get_inputs()}

    def forward(self, texts):
        import numpy as np
        encoded = self.tokenizer.encode_batch(texts)
        feed = {"input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64)}
        output = self.session.run(None, {key: value for key,value in feed.items() if key in self.inputs})[0]
        return output, feed["attention_mask"]

    def encode(self, texts, normalize_embeddings=True, batch_size=32, **kwargs):
        import numpy as np
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        batches = []
        for start in range(0, len(texts), batch_size):
            output, mask = self.forward(texts[start:start+batch_size])
            if output.ndim == 3:
                output = (output*mask[:,:,None]).sum(axis=1)/np.maximum(mask.sum(axis=1)[:,None], 1)
            if normalize_embeddings:
                output = output/np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
            batches.append(output.astype(np.float32))
        result = np.concatenate(batches, axis=0)
        return result[0] if single else result

    def predict(self, pairs):
        import numpy as np
        scores = []
        for start in range(0, len(pairs), 16):
            output, _ = self.forward(list(pairs[start:start+16]))
            scores.extend(output.reshape(-1).tolist())
        return np.array(scores)

@lru_cache(maxsize=2)
def get_model(kind):
    models = json.loads(LOCK.read_text())
    return CPUModel(models[0 if kind == "embedding" else 1])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", required=True)
    parser.parse_args()
    download()
