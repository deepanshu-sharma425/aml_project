"""Train a deep forecaster and save test-set predictions and metrics.

Examples
--------
python scripts/train_deep.py --model lstm
python scripts/train_deep.py --model gwnet --variant full --epochs 40
python scripts/train_deep.py --model gwnet --variant no_graph --epochs 12 --tag ablation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic.data import load_metrla, transition_matrices  # noqa: E402
from traffic.metrics import evaluate  # noqa: E402
from traffic.models import GraphWaveNet, LSTMForecaster  # noqa: E402
from traffic.trainer import Trainer, set_seed  # noqa: E402

GWN_VARIANTS = {
    # name: (use fixed road-graph supports, use adaptive adjacency, use graph conv)
    "full": (True, True, True),
    "no_adaptive": (True, False, True),
    "adaptive_only": (False, True, True),
    "no_graph": (False, False, False),
}


def build_model(args, data):
    if args.model == "lstm":
        return LSTMForecaster(hidden=args.hidden, layers=2)
    fixed, adaptive, graph = GWN_VARIANTS[args.variant]
    supports = [torch.tensor(s) for s in transition_matrices(data.adj)] if fixed else []
    return GraphWaveNet(data.num_nodes, supports, adaptive=adaptive, use_graph=graph)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "gwnet"], required=True)
    p.add_argument("--variant", choices=list(GWN_VARIANTS), default="full")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="main")
    args = p.parse_args()

    set_seed(args.seed)
    data = load_metrla()
    model = build_model(args, data)
    name = args.model if args.model == "lstm" else f"gwnet_{args.variant}"
    run = f"{name}_{args.tag}"
    print(f"{run}: {sum(q.numel() for q in model.parameters()):,} parameters", flush=True)

    trainer = Trainer(model, data, lr=args.lr, batch_size=args.batch_size,
                      ckpt=ROOT / "checkpoints" / f"{run}.pt")
    t0 = time.time()
    trainer.fit(args.epochs, args.patience, seed=args.seed)
    train_time = time.time() - t0

    test = data.split.test
    t1 = time.time()
    pred = trainer.predict(test)
    infer_time = time.time() - t1
    y = data.speed[test[:, None] + np.arange(1, data.seq_out + 1)]
    metrics = evaluate(pred, y)

    out = ROOT / "results"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "history").mkdir(parents=True, exist_ok=True)
    np.save(out / "predictions" / f"{run}.npy", pred.astype(np.float32))
    trainer.save_history(out / "history" / f"{run}.json")
    info = {"model": run, "params": sum(q.numel() for q in model.parameters()),
            "epochs_run": len(trainer.history), "train_sec": train_time, "infer_sec": infer_time,
            "args": vars(args), "metrics": metrics}
    (out / "metrics" / f"{run}.json").write_text(json.dumps(info, indent=2))
    if args.model == "gwnet" and model.adaptive:
        np.save(out / "predictions" / f"{run}_adaptive_adj.npy", model.adaptive_adj().detach().cpu().numpy())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
