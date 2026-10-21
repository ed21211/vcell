#!/usr/bin/env python
"""
audit_omiclip_params.py -- what does OmiCLIP "full fine-tuning" actually train?

Methodology doc, section 16, steps 1-6. Loads the pretrained OmiCLIP through
Loki, pushes one batch of real matched H&E patches + gene sentences through the
SAME contrastive loss used in finetune_omiclip_full.py, calls backward(), and
reports, per module group:

  * parameter count
  * how many of those parameters have requires_grad=True
  * how many actually RECEIVE a non-zero gradient

The pilots only checked requires_grad. In open_clip's CoCa, encode_image() and
encode_text() never touch the multimodal text decoder, so those weights get no
gradient and AdamW silently skips them. This script measures that.

It then defines the frozen / single / dual modes (apply_mode), applies each one,
runs another backward pass and ASSERTS that exactly the expected groups receive
gradients. The training script can import apply_mode / classify from here.

Run in the omiclip conda env, from the vcell folder:

    python scripts/audit_omiclip_params.py | tee outputs/audit_omiclip_params.txt

Group names are matched by parameter-name prefix (GROUP_PREFIXES below). Any
parameter that matches nothing is printed as 'unassigned' -- if you see that,
send me the printout and I will fix the prefixes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
PAIRS_CSV = PROJECT_DIR / "datasets" / "omiclip" / "ccrcc_pairs.csv"
PATCH_DIR = PROJECT_DIR / "datasets" / "hest_ccrcc" / "patches"
CHECKPOINT = PROJECT_DIR / "models" / "omiclip" / "checkpoint.pt"
TRAIN_SAMPLES = [f"INT{i}" for i in range(1, 19)]

# open_clip CoCa layout: visual.*, text.*, text_decoder.*, logit_scale (+ logit_bias)
GROUP_PREFIXES = [
    ("image_tower", ("visual.",)),
    ("text_tower", ("text.",)),          # note: does NOT match 'text_decoder.'
    ("text_decoder", ("text_decoder.",)),
    ("scalars", ("logit_scale", "logit_bias")),
]

# groups allowed to update in each mode ('scalars' = logit_scale/bias, reported
# but not asserted: if the temperature is clamped at 100 its gradient is zero)
MODES = {
    "frozen": set(),
    "single": {"image_tower", "scalars"},
    "dual": {"image_tower", "text_tower", "scalars"},
}
ASSERTED_GROUPS = ["image_tower", "text_tower", "text_decoder"]


def classify(name: str) -> str:
    for group, prefixes in GROUP_PREFIXES:
        if name.startswith(prefixes):
            return group
    return "unassigned"


def apply_mode(model, mode: str):
    """Set requires_grad per group. Returns {group: n_trainable_params}."""
    allowed = MODES[mode]
    counts = {}
    for name, p in model.named_parameters():
        g = classify(name)
        p.requires_grad = g in allowed
        if p.requires_grad:
            counts[g] = counts.get(g, 0) + p.numel()
    return counts


# ----------------------------------------------------------------------------
# loss / forward (same maths as finetune_omiclip_full.py)
# ----------------------------------------------------------------------------

def encode_text(model, tokens):
    import torch

    if torch.is_tensor(tokens):
        return model.encode_text(tokens)
    if isinstance(tokens, dict):
        ids = tokens["input_ids"]
        mask = tokens.get("attention_mask")
        try:
            return model.encode_text(ids, text_valid=mask)
        except TypeError:
            return model.encode_text(ids)
    raise TypeError(f"Unsupported tokenizer output: {type(tokens)}")


def contrastive_loss(image_features, text_features, logit_scale):
    import torch
    import torch.nn.functional as F

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    scale = logit_scale.exp().clamp(max=100)
    logits = scale * image_features @ text_features.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def backward_pass(model, images, tokens, amp_dtype=None):
    """One forward/backward of the contrastive loss. Returns the loss value."""
    import torch

    model.train()
    model.zero_grad(set_to_none=True)
    use_amp = amp_dtype is not None and images.device.type == "cuda"
    with torch.autocast(device_type=images.device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
        img = model.encode_image(images)
        txt = encode_text(model, tokens)
        loss = contrastive_loss(img, txt, model.logit_scale)
    loss.backward()
    return float(loss.detach())


def gradient_table(model):
    """Per-parameter record after a backward pass."""
    rows = []
    for name, p in model.named_parameters():
        g = p.grad
        gnorm = float(g.detach().float().norm()) if g is not None else 0.0
        rows.append(
            dict(
                name=name,
                group=classify(name),
                numel=p.numel(),
                requires_grad=bool(p.requires_grad),
                grad_is_none=g is None,
                receives_grad=bool(g is not None and gnorm > 0.0),
                grad_sq=gnorm ** 2,
            )
        )
    return pd.DataFrame(rows)


def summarise(df, by):
    df = df.assign(
        p_rg=df["numel"] * df["requires_grad"],
        p_grad=df["numel"] * df["receives_grad"],
    )
    out = df.groupby(by, sort=False).agg(
        tensors=("name", "size"),
        params=("numel", "sum"),
        requires_grad=("p_rg", "sum"),
        receives_grad=("p_grad", "sum"),
        grad_sq=("grad_sq", "sum"),
    )
    out["grad_l2"] = np.sqrt(out.pop("grad_sq"))
    out["pct_receiving"] = 100.0 * out["receives_grad"] / out["params"].clip(lower=1)
    return out


def sub_name(name, depth):
    return ".".join(name.split(".")[:depth])


def millions(n):
    return f"{n / 1e6:9.2f}M"


def print_table(title, table):
    print(f"\n{title}")
    print(f"  {'group':<34}{'tensors':>8}{'params':>12}{'requires_grad':>15}{'gets gradient':>15}{'%':>7}{'|grad|':>10}")
    for key, r in table.iterrows():
        label = key if isinstance(key, str) else " / ".join(map(str, key))
        print(
            f"  {label:<34}{int(r.tensors):>8d}{millions(r.params):>12}{millions(r.requires_grad):>15}"
            f"{millions(r.receives_grad):>15}{r.pct_receiving:>7.1f}{r.grad_l2:>10.2e}"
        )


# ----------------------------------------------------------------------------
# the audit itself (pure function: testable without Loki/GPU)
# ----------------------------------------------------------------------------

def run_audit(model, images, tokens, amp_dtype=None, depth=2):
    results = {}

    # 1. what the pilots did: everything requires_grad
    for p in model.parameters():
        p.requires_grad = True
    loss = backward_pass(model, images, tokens, amp_dtype)
    full = gradient_table(model)
    results["full_loss"] = loss
    results["full"] = full

    print(f"\nContrastive loss on the audit batch: {loss:.4f}")
    print_table("A. Pilot setting (every parameter requires_grad=True) -- by group:", summarise(full, "group"))
    full = full.assign(sub=full["name"].map(lambda n: f"{classify(n)} / {sub_name(n, depth)}"))
    sub = summarise(full, "sub")
    print_table(f"   ... by sub-module (first {depth} name components):", sub)

    dead = sub[(sub["requires_grad"] > 0) & (sub["receives_grad"] == 0)]
    if len(dead):
        print("\n  Sub-modules marked trainable that received NO gradient (AdamW will never move them):")
        for key, r in dead.iterrows():
            print(f"    {key:<44}{r.params / 1e6:>8.2f}M")
    else:
        print("\n  Every sub-module marked trainable received a gradient.")

    unassigned = full[full["group"] == "unassigned"]
    if len(unassigned):
        print(f"\n  !! {len(unassigned)} tensors matched no group. First names: {unassigned['name'].head(10).tolist()}")

    g = summarise(full, "group")
    if "text_decoder" in g.index and g.loc["text_decoder", "params"] > 0:
        dec = g.loc["text_decoder"]
        if dec.receives_grad == 0:
            print(
                f"\n  FINDING: text_decoder ({dec.params / 1e6:.1f}M params) received NO gradient. AdamW skips "
                "parameters whose .grad is None, so the pilot 'full fine-tuning' runs only ever updated the "
                "image tower, text tower and temperature. The decoder never moved."
            )
        else:
            print(
                f"\n  FINDING: text_decoder ({dec.params / 1e6:.1f}M params) DOES receive gradient "
                f"({dec.pct_receiving:.1f}%). The pilots genuinely trained the decoder too."
            )

    # 2. mode verification (checklist items 4-6)
    print("\nB. Mode verification (one backward pass per mode):")
    print(f"  {'mode':<8}{'trainable params':>18}   groups that receive gradient")
    ok_all = True
    mode_rows = []
    for mode in ["single", "dual"]:
        counts = apply_mode(model, mode)
        backward_pass(model, images, tokens, amp_dtype)
        table = summarise(gradient_table(model), "group")
        receiving = {k for k, r in table.iterrows() if r.receives_grad > 0}
        expected = {gname for gname in MODES[mode] if gname in ASSERTED_GROUPS}
        wrong_extra = (receiving & set(ASSERTED_GROUPS)) - expected
        wrong_missing = expected - receiving
        ok = not wrong_extra and not wrong_missing
        ok_all &= ok
        trainable = sum(counts.values())
        mode_rows.append(dict(mode=mode, trainable_params=trainable, groups=sorted(receiving), ok=ok))
        print(f"  {mode:<8}{millions(trainable):>18}   {sorted(receiving)}   {'PASS' if ok else 'FAIL'}")
        if wrong_extra:
            print(f"    unexpected gradient in: {sorted(wrong_extra)}")
        if wrong_missing:
            print(f"    expected gradient but none in: {sorted(wrong_missing)}")
    n_frozen = sum(p.numel() for p in model.parameters())
    counts = apply_mode(model, "frozen")
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    print(f"  {'frozen':<8}{millions(0):>18}   [] (no parameter requires grad; total params {n_frozen / 1e6:.1f}M)")

    scale = float(model.logit_scale.detach().exp())
    print(f"\n  logit_scale.exp() = {scale:.2f}" + ("  (>=100: the clamp is active, temperature gets no gradient)" if scale >= 99.9 else ""))
    print("\n  ALL MODE CHECKS PASSED" if ok_all else "\n  !! MODE CHECK FAILED -- do not train until this is resolved")

    # restore the pilot setting so the caller gets the model back unchanged
    for p in model.parameters():
        p.requires_grad = True
    results.update(summary=g, modes=pd.DataFrame(mode_rows), ok=ok_all)
    return results


# ----------------------------------------------------------------------------
# real-data entry point
# ----------------------------------------------------------------------------

def prepare_patch(array):
    from PIL import Image

    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating) and array.max() <= 1:
        array = array * 255
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


def load_real_batch(batch_size, seed, preprocess, tokenizer, device):
    import h5py
    import torch

    df = pd.read_csv(PAIRS_CSV)
    df["sample_id"] = df["sample_id"].astype(str).str.upper()
    df = df[df["sample_id"].isin(TRAIN_SAMPLES)]
    picked = df.sample(n=batch_size, random_state=seed)
    images, sentences = [], []
    for row in picked.itertuples():
        with h5py.File(PATCH_DIR / f"{row.sample_id}.h5", "r") as handle:
            images.append(preprocess(prepare_patch(handle["img"][int(row.patch_index)])))
        sentences.append(str(row.gene_sentence))
    tokens = tokenizer(sentences)
    if isinstance(tokens, dict):
        tokens = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tokens.items()}
    else:
        tokens = tokens.to(device)
    return torch.stack(images).to(device), tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=2, help="name components used for the sub-module table")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "outputs" / "audit" / "omiclip_param_audit.csv")
    args = parser.parse_args()

    import torch
    from loki.utils import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    model, preprocess, tokenizer = load_model(str(CHECKPOINT), device)

    if not args.no_grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        print("Gradient checkpointing: on (as in training)")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    print("\nTop-level children:")
    for name, child in model.named_children():
        print(f"  {name:<16}{sum(p.numel() for p in child.parameters()) / 1e6:>10.2f}M  {type(child).__name__}")

    images, tokens = load_real_batch(args.batch_size, args.seed, preprocess, tokenizer, device)
    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    results = run_audit(model, images, tokens, amp_dtype=amp, depth=args.depth)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results["full"].drop(columns=["grad_sq"]).to_csv(args.out, index=False)
    print(f"\nPer-parameter table: {args.out}")


if __name__ == "__main__":
    main()