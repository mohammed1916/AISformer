import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
import os
import sys

# ensure root package imports work when script is run from this folder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is required for serve_fastapi.py. pip install onnxruntime") from e


class PredictRequest(BaseModel):
    sequence: list


app = FastAPI()

onnx_session = None


@app.on_event("startup")
def load_model():
    global onnx_session
    onnx_path = "model.onnx"
    onnx_session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"])
    print(f"Loaded ONNX model: {onnx_path}")


@app.post("/predict")
def predict(request: PredictRequest):
    seq = np.array(request.sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 4:
        raise HTTPException(
            status_code=400, detail="sequence must be shape [T,4]")

    batch = np.expand_dims(seq, axis=0)
    outputs = onnx_session.run(None, {"x": batch})
    logits = outputs[0]

    # decode predicted argmax for each feature
    pred_indexes = np.argmax(logits, axis=-1)[0]  # shape [T, full_dim]
    return {
        "logits_shape": logits.shape,
        "pred_index": pred_indexes.tolist(),
        "raw_logits_sample": logits[0, -1, :8].tolist(),
    }


if __name__ == "__main__":
    uvicorn.run("serve_fastapi:app", host="0.0.0.0",
                port=8000, log_level="debug")
