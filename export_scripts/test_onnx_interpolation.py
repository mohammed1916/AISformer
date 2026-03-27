import argparse
import numpy as np
import onnxruntime as ort


def make_dummy_inputs(seq_len: int, batch: int = 1):
    # x: float32, shape [batch, seq_len, 4]
    x = np.zeros((batch, seq_len, 4), dtype=np.float32)
    # token_types: int64, shape [batch, seq_len]
    token_types = np.zeros((batch, seq_len), dtype=np.int64)
    return {"x": x, "token_types": token_types}


def run(onnx_path: str, seq_len: int, batch: int = 1):
    sess = ort.InferenceSession(onnx_path)
    inputs = make_dummy_inputs(seq_len, batch)
    out = sess.run(None, inputs)
    print("ONNX outputs shapes:")
    for i, o in enumerate(out):
        print(f" output[{i}] shape: {o.shape} dtype: {o.dtype}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quick test runner for interpolation ONNX model")
    p.add_argument("--onnx", type=str, required=True, help="Path to ONNX file")
    p.add_argument("--seq-len", type=int, default=18, help="Sequence length to test")
    p.add_argument("--batch", type=int, default=1, help="Batch size to test")
    args = p.parse_args()
    run(args.onnx, args.seq_len, args.batch)
