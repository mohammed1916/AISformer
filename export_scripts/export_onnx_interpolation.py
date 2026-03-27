import argparse
import os
import sys
import torch

# ensure repo root imports work when running from export_scripts/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# For interpolation mode, import from src_interpolation package
SRC_INTERP = os.path.join(REPO_ROOT, "src_interpolation")
if SRC_INTERP not in sys.path:
    sys.path.insert(0, SRC_INTERP)

from models import TrAISformerInterpolation
from config_trAISformer import Config
from pathlib import Path

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
    trained_max_seqlen = cf.max_seqlen
    if seq_len > trained_max_seqlen:
        raise ValueError(f"seq_len ({seq_len}) cannot exceed trained max_seqlen ({trained_max_seqlen})")

    model = TrAISformerInterpolation(cf).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    dummy_input = torch.zeros((1, seq_len, 4), dtype=torch.float32, device=device)
    # token_types: integer segment ids per position (shape: [batch, seq_len])
    dummy_token_types = torch.zeros((1, seq_len), dtype=torch.long, device=device)

    print(f"Exporting ONNX model to {onnx_path} (seq_len={seq_len})...")
    torch.onnx.export(
        model,
        (dummy_input, dummy_token_types),
        onnx_path,
        opset_version=18,
        input_names=["x", "token_types"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "batch", 1: "seq_len"},
            "token_types": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        do_constant_folding=True,
        verbose=False,
    )
    print("Export completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TrAISformer (interpolation) model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--search-root", type=str, default=".", help="Root directory to search for checkpoints if --checkpoint is omitted")
    parser.add_argument("--onnx", type=str, default="model_interpolation.onnx", help="Output ONNX path")
    parser.add_argument("--seq-len", type=int, default=18, help="Sequence length used for export")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = select_checkpoint(args.search_root)

    export_onnx(checkpoint_path, args.onnx, args.seq_len, device=args.device)
