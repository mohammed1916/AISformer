import argparse
import os
import sys
import torch
import numpy as np

# ensure repo root imports work when running from export_scripts/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import TrAISformer
from config_trAISformer import Config
from pathlib import Path

# Optional interactive menu with arrows + enter
try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False


def find_checkpoints(root_dir="."):
    print(f"Scanning for checkpoint (.pt) files under '{root_dir}'...")
    p = Path(root_dir)
    ckpts = sorted(p.rglob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"Found {len(ckpts)} model checkpoint(s).")
    return ckpts


def select_checkpoint(root_dir="."):
    ckpts = find_checkpoints(root_dir)
    if not ckpts:
        raise FileNotFoundError("No .pt checkpoints found. Please specify --checkpoint path.")

    if INQUIRER_AVAILABLE:
        root = Path(root_dir).resolve()
        def _display_name(p):
            try:
                return str(p.relative_to(root))
            except Exception:
                return str(p)

        choices = [Choice(value=str(p), name=_display_name(p)) for p in ckpts]
        selected = inquirer.select(message="Select model checkpoint:", choices=choices, pointer="➜").execute()
        return selected

    print("Select a checkpoint:")
    for i, p in enumerate(ckpts):
        print(f" [{i}] {p}")
    while True:
        sel = input("Enter index: ").strip()
        if sel.isdigit() and 0 <= int(sel) < len(ckpts):
            return str(ckpts[int(sel)])
        print("Invalid input, try again.")


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
    parser.add_argument("--search-root", type=str, default=".", help="Root directory to search for checkpoints if --checkpoint is omitted")
    parser.add_argument("--onnx", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--seq-len", type=int, default=18, help="Sequence length used for export")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = select_checkpoint(args.search_root)

    export_onnx(checkpoint_path, args.onnx, args.seq_len, device=args.device)
