import argparse
import torch
import numpy as np
from models import TrAISformer
from config_trAISformer import Config


def export_onnx(ckpt_path, onnx_path, seq_len, device="cpu"):
    cf = Config()
    cf.max_seqlen = seq_len
    # use same model config that the checkpoint was trained with
    model = TrAISformer(cf, partition_model=None).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    dummy_input = torch.zeros((1, cf.init_seqlen, 4), dtype=torch.float32, device=device)

    print(f"Exporting ONNX model to {onnx_path} (seq_len={cf.init_seqlen})...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=14,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch", 1: "seq_len"}, "logits": {0: "batch", 1: "seq_len"}},
        do_constant_folding=True,
    )
    print("Export completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TrAISformer model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--onnx", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--seq-len", type=int, default=18, help="Sequence length used for export")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    args = parser.parse_args()

    if args.checkpoint is None:
        raise ValueError("Use --checkpoint to specify the model checkpoint path")

    export_onnx(args.checkpoint, args.onnx, args.seq_len, device=args.device)
