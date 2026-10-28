#!/usr/bin/env python
"""
finetune_omiclip_modes.py -- OmiCLIP fine-tuning in three modes on the balanced split.

    python scripts/finetune_omiclip_modes.py --mode single --lr 1e-6 --epochs 3 --out-dir models/omiclip_ft/single_lr1e-6
    python scripts/finetune_omiclip_modes.py --mode dual   --lr 1e-6 --epochs 3 --out-dir models/omiclip_ft/dual_lr1e-6
    python scripts/finetune_omiclip_modes.py --mode frozen --out-dir models/omiclip_ft/frozen   # epoch-0 val metrics only
    python scripts/finetune_omiclip_modes.py --mode dual --smoke --out-dir /tmp/ft_smoke       # 4 batches, checks the plumbing

Modes come from audit_omiclip_params.apply_mode (checked on the real checkpoint, 22 Oct):
  frozen  nothing trains
  single  image tower (306.72M) + logit_scale
  dual    image tower + text tower (430.38M trainable in total)
The multimodal text decoder (208M) never receives a gradient from the contrastive loss, so
it is excluded from every mode. The earlier "full" pilots therefore were dual-tower runs.

What differs from finetune_omiclip_full.py:
  * reads the balanced split from datasets/splits/ccrcc_splits_v1.csv (column fixed_split)
  * learning rate, epochs, mode and output folder are command-line arguments
  * evaluates the validation slides BEFORE training (epoch 0 = the pretrained model)
  * validation batches are in a fixed shuffled order, so in-batch negatives come from several
    slides (the old loader was sorted by slide); the order is identical at every epoch
  * only the trainable parameters are saved (about 1.2 to 1.7 GB instead of 2.5 GB)
  * checks after the first backward pass that gradients really reach the trainable tensors

Outputs in --out-dir: history.csv, summary.json, best.pt (lowest val loss among epochs 1..N),
last.pt, and epoch_<n>.pt when --save-every-epoch is given. done.json is written last, so a
driver script can skip finished runs.

Run in the omiclip conda env on the GPU machine, from the vcell folder.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from audit_omiclip_params import MODES, apply_mode, encode_text

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

PROJECT_DIR = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def prepare_patch(array):
    """HEST HDF5 patch -> RGB PIL image (same rules as extract_omiclip_embeddings.py)."""
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


class PairDataset(Dataset):
    def __init__(self, df, patch_dir, preprocess):
        df = df.reset_index(drop=True)
        self.samples = df["sample_id"].astype(str).to_numpy()
        self.patch_index = df["patch_index"].astype(int).to_numpy()
        self.sentences = df["gene_sentence"].astype(str).to_numpy()
        self.patch_dir = Path(patch_dir)
        self.preprocess = preprocess
        self._handles = {}  # opened lazily, one set per DataLoader worker
        for sample in sorted(set(self.samples)):
            if not (self.patch_dir / f"{sample}.h5").exists():
                raise FileNotFoundError(self.patch_dir / f"{sample}.h5")

    def __len__(self):
        return len(self.samples)

    def _handle(self, sample):
        if sample not in self._handles:
            self._handles[sample] = h5py.File(self.patch_dir / f"{sample}.h5", "r")
        return self._handles[sample]

    def __getitem__(self, i):
        array = self._handle(self.samples[i])["img"][int(self.patch_index[i])]
        return self.preprocess(prepare_patch(array)), str(self.sentences[i])


def load_splits(pairs_path, split_csv):
    pairs = pd.read_csv(pairs_path)
    for col in ("sample_id", "patch_index", "gene_sentence"):
        if col not in pairs.columns:
            raise SystemExit(f"{pairs_path} has no '{col}' column. Columns: {pairs.columns.tolist()}")
    pairs = pairs.dropna(subset=["patch_index", "gene_sentence"]).reset_index(drop=True)
    pairs["sample_id"] = pairs["sample_id"].astype(str).str.upper()

    splits = pd.read_csv(split_csv)
    splits["sample_id"] = splits["sample_id"].astype(str).str.upper()
    groups = {name: set(splits.loc[splits["fixed_split"] == name, "sample_id"]) for name in ("train", "val", "test")}
    for a in groups:
        for b in groups:
            if a < b and groups[a] & groups[b]:
                raise SystemExit(f"Slides in both {a} and {b}: {groups[a] & groups[b]}")
    missing = set(splits["sample_id"]) - set(pairs["sample_id"])
    if missing:
        raise SystemExit(f"Slides in the split file but not in the pairs file: {sorted(missing)}")

    train = pairs[pairs["sample_id"].isin(groups["train"])].copy()
    val = pairs[pairs["sample_id"].isin(groups["val"])].copy()
    test_n = int(pairs["sample_id"].isin(groups["test"]).sum())
    if train.empty or val.empty:
        raise SystemExit("Empty train or validation split.")
    # fixed shuffled order for validation: in-batch negatives mix slides, identical at every epoch
    val = val.sample(frac=1.0, random_state=0).reset_index(drop=True)
    print(f"Train: {len(train):,} spots on {len(groups['train'])} slides {sorted(groups['train'])}")
    print(f"Val:   {len(val):,} spots on {len(groups['val'])} slides {sorted(groups['val'])}")
    print(f"Test:  {test_n:,} spots on {len(groups['test'])} slides {sorted(groups['test'])} (never used here)")
    return train, val, groups


# ----------------------------------------------------------------------------
# loss, optimiser, schedule
# ----------------------------------------------------------------------------

def move_tokens(tokens, device):
    if torch.is_tensor(tokens):
        return tokens.to(device, non_blocking=True)
    if isinstance(tokens, dict):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in tokens.items()}
    raise TypeError(f"Unsupported tokenizer output: {type(tokens)}")


def clip_metrics(img, txt, logit_scale):
    img = F.normalize(img.float(), dim=-1)
    txt = F.normalize(txt.float(), dim=-1)
    scale = logit_scale.float().exp().clamp(max=100)
    logits = scale * img @ txt.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
    sim = (img * txt).sum(-1).mean()
    acc = ((logits.argmax(1) == labels).float().mean() + (logits.argmax(0) == labels).float().mean()) / 2.0
    return loss, sim, acc


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name.lower() or "norm" in name.lower() or "logit_scale" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, foreach=False,
    )


def build_scheduler(optimizer, total_steps, warmup_ratio):
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, tokenizer, device, amp_dtype, optimizer=None, scheduler=None, scaler=None,
              max_batches=None, grad_clip=1.0, label="train"):
    training = optimizer is not None
    model.train(training)
    totals = dict(loss=0.0, sim=0.0, acc=0.0)
    rows = 0
    start = time.time()
    n_batches = min(len(loader), max_batches) if max_batches else len(loader)
    for b, (images, sentences) in enumerate(loader, start=1):
        if max_batches and b > max_batches:
            break
        images = images.to(device, non_blocking=True)
        tokens = move_tokens(tokenizer(list(sentences)), device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                img = model.encode_image(images)
                txt = encode_text(model, tokens)
            loss, sim, acc = clip_metrics(img, txt, model.logit_scale)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if b == 1:
                    check_gradients(model)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
        n = images.shape[0]
        totals["loss"] += loss.item() * n
        totals["sim"] += sim.item() * n
        totals["acc"] += acc.item() * n
        rows += n
        if training and b % 100 == 0:
            eta = (n_batches - b) / (b / (time.time() - start)) / 60
            print(f"  {label} batch {b}/{n_batches} | loss {loss.item():.4f} | sim {sim.item():.4f} | "
                  f"retrieval {acc.item():.3f} | lr {optimizer.param_groups[0]['lr']:.2e} | ETA {eta:.1f} min", flush=True)
    return {k: v / rows for k, v in totals.items()}


def check_gradients(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    with_grad = sum(1 for p in trainable if p.grad is not None)
    print(f"  gradient check: {with_grad} of {len(trainable)} trainable tensors received a gradient", flush=True)
    if with_grad == 0:
        raise RuntimeError("No trainable tensor received a gradient (gradient checkpointing or the mode setup is wrong).")
    if with_grad < 0.5 * len(trainable):
        print("  WARNING: fewer than half of the trainable tensors received a gradient.", flush=True)


def save_ckpt(path, model, meta):
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"trainable_state_dict": state, **meta}, path)
    print(f"  saved {path} ({len(state)} tensors)", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=sorted(MODES), required=True)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pairs", type=Path, default=PROJECT_DIR / "datasets" / "omiclip" / "ccrcc_pairs.csv")
    p.add_argument("--patch-dir", type=Path, default=PROJECT_DIR / "datasets" / "hest_ccrcc" / "patches")
    p.add_argument("--split-csv", type=Path, default=PROJECT_DIR / "datasets" / "splits" / "ccrcc_splits_v1.csv")
    p.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "models" / "omiclip" / "checkpoint.pt")
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--smoke", action="store_true", help="4 train batches, 3 val batches, 1 epoch, 2 workers")
    args = p.parse_args()

    if args.smoke:
        args.epochs, args.num_workers = 1, 2
    max_train = 4 if args.smoke else None
    max_val = 3 if args.smoke else None

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = torch.device("cuda")
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mode {args.mode} | lr {args.lr:g} | epochs {args.epochs} | batch {args.batch_size} | out {args.out_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)")

    train_df, val_df, groups = load_splits(args.pairs, args.split_csv)

    from loki.utils import load_model

    model, preprocess, tokenizer = load_model(str(args.checkpoint), device)
    counts = apply_mode(model, args.mode)
    n_trainable = sum(counts.values())
    n_total = sum(q.numel() for q in model.parameters())
    print(f"Trainable by group: { {k: f'{v / 1e6:.2f}M' for k, v in counts.items()} } | "
          f"{n_trainable / 1e6:.2f}M of {n_total / 1e6:.2f}M")
    if args.mode != "frozen" and n_trainable == 0:
        raise SystemExit("Mode left nothing trainable.")
    if args.mode != "frozen" and not args.no_grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        print("Gradient checkpointing on")

    train_loader = DataLoader(
        PairDataset(train_df, args.patch_dir, preprocess), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        PairDataset(val_df, args.patch_dir, preprocess), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False, persistent_workers=args.num_workers > 0,
    )
    print(f"Batches: train {len(train_loader)}, val {len(val_loader)}")

    if torch.cuda.is_bf16_supported():
        amp_dtype, scaler = torch.bfloat16, None
        print("bf16 autocast")
    else:
        amp_dtype, scaler = torch.float16, torch.cuda.amp.GradScaler()
        print("fp16 autocast with GradScaler")

    history = []
    meta_base = dict(mode=args.mode, lr=args.lr, batch_size=args.batch_size, weight_decay=args.weight_decay,
                     seed=args.seed, train_slides=sorted(groups["train"]), val_slides=sorted(groups["val"]))

    print("\nEpoch 0: validation of the starting (pretrained) model")
    v0 = run_epoch(model, val_loader, tokenizer, device, amp_dtype, max_batches=max_val, label="val")
    history.append(dict(epoch=0, train_loss=np.nan, train_similarity=np.nan, train_retrieval=np.nan,
                        val_loss=v0["loss"], val_similarity=v0["sim"], val_retrieval=v0["acc"], seconds=0.0))
    print(f"  val loss {v0['loss']:.4f} | sim {v0['sim']:.4f} | retrieval {v0['acc']:.3f}", flush=True)
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)

    if args.mode == "frozen":
        (args.out_dir / "summary.json").write_text(json.dumps(dict(**meta_base, epoch0_val_loss=v0["loss"]), indent=2))
        (args.out_dir / "done.json").write_text("{}")
        print("Frozen mode: nothing to train. Done.")
        return

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = min(len(train_loader), max_train) if max_train else len(train_loader)
    scheduler = build_scheduler(optimizer, steps_per_epoch * args.epochs, args.warmup_ratio)

    best_loss, best_epoch = float("inf"), None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        t_epoch = time.time()
        tr = run_epoch(model, train_loader, tokenizer, device, amp_dtype, optimizer, scheduler, scaler,
                       max_batches=max_train, grad_clip=args.grad_clip, label="train")
        va = run_epoch(model, val_loader, tokenizer, device, amp_dtype, max_batches=max_val, label="val")
        history.append(dict(epoch=epoch, train_loss=tr["loss"], train_similarity=tr["sim"], train_retrieval=tr["acc"],
                            val_loss=va["loss"], val_similarity=va["sim"], val_retrieval=va["acc"],
                            seconds=time.time() - t_epoch))
        pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
        print(f"  train loss {tr['loss']:.4f} | val loss {va['loss']:.4f} (epoch 0: {v0['loss']:.4f}) | "
              f"val sim {va['sim']:.4f} | val retrieval {va['acc']:.3f}", flush=True)

        meta = dict(meta_base, epoch=epoch, val_loss=va["loss"])
        save_ckpt(args.out_dir / "last.pt", model, meta)
        if args.save_every_epoch:
            save_ckpt(args.out_dir / f"epoch_{epoch}.pt", model, meta)
        if va["loss"] < best_loss:
            best_loss, best_epoch = va["loss"], epoch
            save_ckpt(args.out_dir / "best.pt", model, meta)
            print("  new best epoch", flush=True)

    summary = dict(meta_base, epochs=args.epochs, best_epoch=best_epoch, best_val_loss=best_loss,
                   epoch0_val_loss=v0["loss"], beats_pretrained=bool(best_loss < v0["loss"]),
                   train_hours=(time.time() - t_start) / 3600.0,
                   peak_gpu_gb=torch.cuda.max_memory_allocated() / 1024**3)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "done.json").write_text("{}")
    print("\nDone.", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
