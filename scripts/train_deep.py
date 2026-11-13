"""Train a deep forecaster and save test-set predictions and metrics.

Examples
--------
python scripts/train_deep.py --model lstm
python scripts/train_deep.py --model gwnet --variant full --epochs 40
python scripts/train_deep.py --model gwnet --variant full --eval-only          # re-export from checkpoint
python scripts/train_deep.py --model stgat --spatial attention                 # ST-GAT (road-graph attention)
python scripts/train_deep.py --model stgat --spatial none --tag ablation       # no information across sensors
python scripts/train_deep.py --model stgat --base xgb_oof                      # XGBoost -> GAT hybrid
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from traffic.data import load_metrla, transition_matrices
from traffic.metrics import evaluate
from traffic.models import STGAT, GraphWaveNet, LSTMForecaster
from traffic.trainer import Trainer, input_features, set_seed

GWN_VARIANTS = {
    # name: (use fixed road-graph supports, use adaptive adjacency, use graph conv)
    "full": (True, True, True),
    "no_adaptive": (True, False, True),
    "adaptive_only": (False, True, True),
    "no_graph": (False, False, False),
}
BASES = {"xgb_oof": "xgb_seg_base.npy", "xgb_insample": "xgb_seg_insample.npy"}


def build_model(args, data, in_dim):
    if args.model == "lstm":
        return LSTMForecaster(in_dim=in_dim, hidden=args.hidden, layers=2)
    if args.model == "stgat":
        return STGAT(data.num_nodes, data.adj, in_dim=in_dim, d=args.hidden, heads=args.heads,
                     spatial=args.spatial, dropout=args.dropout, attn_dropout=args.attn_dropout,
                     node_emb=args.node_emb, base_dim=12 if args.base else 0, residual=args.residual)
    fixed, adaptive, graph = GWN_VARIANTS[args.variant]
    supports = [torch.tensor(s) for s in transition_matrices(data.adj)] if fixed else []
    return GraphWaveNet(data.num_nodes, supports, in_dim=in_dim, adaptive=adaptive, use_graph=graph)


def run_name(args) -> str:
    if args.name:
        return args.name
    if args.model == "lstm":
        return f"lstm_{args.tag}"
    if args.model == "gwnet":
        return f"gwnet_{args.variant}_{args.tag}"
    kind = "hybrid" if args.base else "stgat"
    return f"{kind}_{args.spatial}_{args.tag}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "gwnet", "stgat"], required=True)
    p.add_argument("--variant", choices=list(GWN_VARIANTS), default="full", help="Graph WaveNet ablation")
    p.add_argument("--spatial", choices=["attention", "mean", "none", "dense"], default="attention",
                   help="ST-GAT spatial module")
    p.add_argument("--inputs", choices=["raw", "masked"], default=None,
                   help="input pipeline (default: masked for stgat, raw for the Phase 1 models)")
    p.add_argument("--base", choices=list(BASES), default=None, help="hybrid: base forecasts to correct")
    p.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True,
                   help="hybrid: output = base + correction (otherwise base is only an input)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    p.add_argument("--lr-patience", type=int, default=None, help="halve LR after this many flat epochs")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--node-emb", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epoch-fraction", type=float, default=1.0,
                   help="train each epoch on this random fraction of the training windows")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="main")
    p.add_argument("--name", default=None, help="override the run name")
    p.add_argument("--eval-only", action="store_true", help="load the checkpoint and only predict")
    p.add_argument("--out", type=Path, default=ROOT / "results", help="output directory")
    args = p.parse_args()
    stgat = args.model == "stgat"
    args.inputs = args.inputs or ("masked" if stgat else "raw")
    args.optimizer = args.optimizer or ("adamw" if stgat else "adam")
    if args.lr_patience is None and stgat:
        args.lr_patience = 3
    if args.base and not stgat:
        p.error("--base is only supported for --model stgat")

    set_seed(args.seed)
    data = load_metrla()
    in_dim = input_features(data, args.inputs).shape[-1]
    model = build_model(args, data, in_dim)
    run = run_name(args)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"{run}: {n_params:,} parameters, inputs={args.inputs}", flush=True)

    base = np.load(args.out / "predictions" / BASES[args.base]) if args.base else None
    ckpt = args.out.parent / "checkpoints" / f"{run}.pt"
    trainer = Trainer(model, data, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                      ckpt=ckpt, inputs=args.inputs, base=base, optimizer=args.optimizer,
                      lr_patience=args.lr_patience, epoch_fraction=args.epoch_fraction)
    del base
    t0 = time.time()
    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.to(trainer.device)
    else:
        trainer.fit(args.epochs, args.patience, seed=args.seed)
    train_time = time.time() - t0

    test = data.split.test
    t1 = time.time()
    pred = trainer.predict(test)
    infer_time = time.time() - t1
    y = data.speed[test[:, None] + np.arange(1, data.seq_out + 1)]
    metrics = evaluate(pred, y)

    out = args.out
    for d in ("predictions", "metrics", "history"):
        (out / d).mkdir(parents=True, exist_ok=True)
    np.save(out / "predictions" / f"{run}.npy", pred.astype(np.float32))
    if args.model == "gwnet" and model.adaptive:
        np.save(out / "predictions" / f"{run}_adaptive_adj.npy", model.adaptive_adj().detach().cpu().numpy())
    if args.eval_only:                       # keep the training record of the original run
        f = out / "metrics" / f"{run}.json"
        info = json.loads(f.read_text()) if f.exists() else {"model": run}
        info.update({"metrics": metrics, "infer_sec": infer_time})
        f.write_text(json.dumps(info, indent=2))
    else:
        trainer.save_history(out / "history" / f"{run}.json")
        best = min(trainer.history, key=lambda h: h["val_mae"])
        info = {"model": run, "params": n_params, "epochs_run": len(trainer.history),
                "best_epoch": best["epoch"], "best_val_mae": best["val_mae"],
                "train_sec": train_time, "infer_sec": infer_time,
                "args": {k: str(v) for k, v in vars(args).items()}, "metrics": metrics}
        (out / "metrics" / f"{run}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
