"""exp_045 — entity-embedding MLP, one fit per invocation (1 GPU).

    CUDA_VISIBLE_DEVICES=0 python train_nn.py --fold 2023 --seed 42 --arch A0

Reads cache/fold_{Y}.npz built by prep.py (all train-derived; see prep.py).
Writes:
    res/res_{tag}.json      metrics + full per-epoch valid curve
    pred/pred_{tag}.npy     float64 [n_valid]  -- prediction at the
                            patience-5 early-stopping best epoch
    pred/all_{tag}.npy      float32 [n_epochs, n_valid] -- every epoch, so a
                            deployment epoch derived from ANOTHER fold can be
                            read off later without re-fitting.  Selecting an
                            epoch off fold_2024's own curve is fold-oracle and
                            is reported separately, never as the deployment
                            number (config do_not #2).

The target is `control_success` directly (binary).  No B3 cell decomposition:
config model.why_not_b3 -- the diversity has to come from the architecture.
"""

import argparse
import json
import os
import sys
import time

# must be set before the first cuBLAS call for deterministic GEMM
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# --- architecture registry -------------------------------------------------
# A0 is the config.yaml default.  Everything else is RECON, run on fold_2023 /
# seed 42 ONLY (config do_not #2: fold_2024 selects nothing).
#   emb_mult scales the pitcher_id / batter_id dims only.
BASE = dict(hidden=[512, 256, 128], dropout=0.2, lr=1e-3, wd=1e-4,
            batch_size=8192, max_epochs=40, patience=5, emb_mult=1,
            emb_dropout=0.0)
ARCH = {
    "A0": dict(BASE),                                        # config default
    "A1": dict(BASE, hidden=[1024, 512, 256], dropout=0.3),
    "A2": dict(BASE, hidden=[256, 128], dropout=0.1),
    "A3": dict(BASE, dropout=0.1),
    "A4": dict(BASE, lr=3e-3),
    "A5": dict(BASE, emb_mult=2),                            # id dims 32/32
    # ---- round 2: A0 overfits by epoch 2, so attack capacity / step size ----
    "B0": dict(BASE, lr=3e-4, max_epochs=25),
    "B1": dict(BASE, lr=1e-4, max_epochs=25),
    "B2": dict(BASE, lr=3e-4, dropout=0.4, max_epochs=25),
    "B3": dict(BASE, hidden=[256, 128], dropout=0.3, lr=3e-4, max_epochs=25),
    "B4": dict(BASE, hidden=[128, 64], dropout=0.2, lr=3e-4, max_epochs=25),
    "B5": dict(BASE, lr=3e-4, dropout=0.3, wd=1e-2, max_epochs=25),
    "B6": dict(BASE, hidden=[256, 128], dropout=0.3, lr=3e-4, emb_mult=0.5,
               max_epochs=25),
    "B7": dict(BASE, hidden=[512, 256, 128], dropout=0.3, wd=1e-2,
               max_epochs=25),
    "B8": dict(BASE, hidden=[256, 128], dropout=0.3, lr=3e-4, emb_dropout=0.2,
               max_epochs=25),
    "B9": dict(BASE, hidden=[64, 32], dropout=0.1, lr=3e-4, emb_mult=0.25,
               max_epochs=25),
    # ---- round 3: round-2 winners were B2 (lr 3e-4 + dropout 0.4) and A5
    #      (id embedding dim 32).  Combine and probe around them.
    "C0": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=2, max_epochs=25),
    "C1": dict(BASE, lr=3e-4, dropout=0.5, max_epochs=25),
    "C2": dict(BASE, lr=3e-4, dropout=0.4, wd=1e-3, max_epochs=25),
    "C3": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=2, emb_dropout=0.2,
               max_epochs=25),
    "C4": dict(BASE, hidden=[1024, 512, 256], dropout=0.4, lr=3e-4,
               max_epochs=25),
    "C5": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=4, max_epochs=25),
    "C6": dict(BASE, lr=1e-3, dropout=0.4, max_epochs=25),
    "C7": dict(BASE, lr=3e-4, dropout=0.4, batch_size=4096, max_epochs=25),
    "C8": dict(BASE, lr=3e-4, dropout=0.4, batch_size=16384, max_epochs=25),
    "C9": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=2, wd=1e-3, max_epochs=25),
    "C10": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=0.125, max_epochs=25),
    # ---- round 4: C10 (id embedding dim 2) won by a margin -> the pitcher /
    #      batter embeddings were the dominant overfitting source, which fits
    #      the 16.5% / 21.5% OOV rate of the valid slices.  Probe that axis.
    "D0": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=0.125, emb_dropout=0.2,
               max_epochs=25),
    "D1": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=0.25, max_epochs=25),
    "D2": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=0.5, max_epochs=25),
    "D3": dict(BASE, lr=3e-4, dropout=0.5, emb_mult=0.125, max_epochs=25),
    "D4": dict(BASE, lr=3e-4, dropout=0.3, emb_mult=0.125, max_epochs=25),
    "D5": dict(BASE, hidden=[256, 128], lr=3e-4, dropout=0.4, emb_mult=0.125,
               max_epochs=25),
    "D6": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=0.125, wd=1e-3,
               max_epochs=25),
    "D7": dict(BASE, lr=1e-3, dropout=0.4, emb_mult=0.125, max_epochs=25),
    "D8": dict(BASE, hidden=[1024, 512, 256], lr=3e-4, dropout=0.4,
               emb_mult=0.125, max_epochs=25),
    "D9": dict(BASE, lr=3e-4, dropout=0.4, emb_mult=0.125, batch_size=4096,
               max_epochs=25),
}


class EmbMLP(nn.Module):
    def __init__(self, emb_slots, emb_dims, n_num, hidden, dropout,
                 emb_dropout=0.0):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(s, d) for s, d in zip(emb_slots, emb_dims)])
        self.emb_drop = nn.Dropout(emb_dropout) if emb_dropout > 0 else None
        d_in = int(sum(emb_dims)) + int(n_num)
        layers = []
        prev = d_in
        for h in hidden:
            # BatchNorm1d after each hidden Linear; the raw input is NOT
            # normalised again (config: "BatchNorm1d (first-layer input excluded)")
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(),
                       nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
        self.d_in = d_in

    def forward(self, xn, xe):
        e = torch.cat([emb(xe[:, j]) for j, emb in enumerate(self.embs)], dim=1)
        if self.emb_drop is not None:
            e = self.emb_drop(e)
        return self.net(torch.cat([e, xn], dim=1)).squeeze(1)


def es_simulate(curve, patience):
    """What patience-`patience` early stopping with best-restore would give.

    Stops at the first epoch after which `patience` consecutive epochs failed
    to improve on the running best.  Returns (best_epoch, best_value,
    stopped_after_epoch).  Epochs are 1-indexed.
    """
    best, best_e, bad = float("inf"), 0, 0
    for e, v in enumerate(curve, start=1):
        if v < best:
            best, best_e, bad = v, e, 0
        else:
            bad += 1
            if bad >= patience:
                return best_e, best, e
    return best_e, best, len(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=[2022, 2023, 2024])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", default="A0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--root", default=os.path.expanduser(
        "~/SungjinPark/experiments/exp_045"))
    args = ap.parse_args()

    cfg = ARCH[args.arch]
    tag = args.tag or f"{args.arch}_fold{args.fold}_seed{args.seed}"
    root = args.root
    for d in ("res", "pred", "logs"):
        os.makedirs(os.path.join(root, d), exist_ok=True)

    t_start = time.time()
    # ---- GPU visibility, logged as required by the task brief -------------
    gpu = {"torch_version": torch.__version__,
           "cuda_available": bool(torch.cuda.is_available()),
           "device_count": int(torch.cuda.device_count()),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
           "device_names": [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]}
    print("[gpu] " + json.dumps(gpu), flush=True)
    assert gpu["cuda_available"], "CUDA not available"
    dev = torch.device("cuda:0")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    cache = os.path.join(root, "cache", f"fold_{args.fold}.npz")
    meta = json.load(open(os.path.join(root, "cache", f"fold_{args.fold}.json")))
    z = np.load(cache, allow_pickle=False)
    num_tr, num_va = z["num_tr"], z["num_va"]
    emb_tr, emb_va = z["emb_tr"], z["emb_va"]
    y_tr, y_va = z["y_tr"], z["y_va"]
    is_r = z["is_r_va"]
    assert num_tr.shape[0] == meta["n_train"] == emb_tr.shape[0] == len(y_tr)
    assert num_va.shape[0] == meta["n_valid"] == emb_va.shape[0] == len(y_va)
    if args.fold == 2024:                      # inherited assert (exp_016)
        assert meta["n_valid"] == 253507 and int(is_r.sum()) == 223497

    emb_slots = [meta["vocab"][c]["n_slots_incl_oov"] for c in C.EMB_COLS]
    emb_dims = [max(2, int(round(C.EMB_DIM[c] * cfg["emb_mult"])))
                if c in ("pitcher_id", "batter_id") else C.EMB_DIM[c]
                for c in C.EMB_COLS]

    Xn = torch.from_numpy(num_tr).to(dev)
    Xe = torch.from_numpy(emb_tr).to(dev)
    Y = torch.from_numpy(y_tr).to(dev)
    Vn = torch.from_numpy(num_va).to(dev)
    Ve = torch.from_numpy(emb_va).to(dev)
    yv = y_va.astype(np.float64)

    model = EmbMLP(emb_slots, emb_dims, num_tr.shape[1], cfg["hidden"],
                   cfg["dropout"], cfg.get("emb_dropout", 0.0)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
    lossf = nn.BCEWithLogitsLoss()

    n, bs = num_tr.shape[0], cfg["batch_size"]
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    E = cfg["max_epochs"]
    all_pred = np.empty((E, len(yv)), dtype=np.float32)
    curve = []
    print(f"[fit] {tag} arch={args.arch} cfg={cfg} params={n_par:,} "
          f"d_in={model.d_in} n_train={n:,} steps/epoch={-(-n//bs)}", flush=True)

    for ep in range(1, E + 1):
        te = time.time()
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            out = model(Xn[idx], Xe[idx])
            loss = lossf(out, Y[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * idx.numel()
        sched.step()

        model.eval()
        with torch.no_grad():
            ps = []
            for i in range(0, Vn.shape[0], 65536):
                ps.append(torch.sigmoid(model(Vn[i:i + 65536], Ve[i:i + 65536])))
            p = torch.cat(ps).double().cpu().numpy()
        all_pred[ep - 1] = p.astype(np.float32)
        br = float(np.mean((p - yv) ** 2))
        curve.append(br)
        print(f"[fit] {tag} ep{ep:02d} trainloss={tot/n:.6f} "
              f"validBrier={br:.8f} score={C.score_of(br):.5f} "
              f"lr={sched.get_last_lr()[0]:.2e} {time.time()-te:.1f}s", flush=True)

    best_e, best_b, stop_e = es_simulate(curve, cfg["patience"])
    full_e = int(np.argmin(curve)) + 1
    p_best = all_pred[best_e - 1].astype(np.float64)

    res = {
        "exp_id": "exp_045", "tag": tag, "arch": args.arch, "arch_cfg": cfg,
        "fold": f"fold_{args.fold}", "seed": args.seed,
        "n_train": meta["n_train"], "n_valid": meta["n_valid"],
        "n_params": int(n_par), "d_in": int(model.d_in),
        "emb_slots": emb_slots, "emb_dims": emb_dims,
        "gpu": gpu,
        "valid_brier_curve": curve,
        "valid_score_curve": [round(C.score_of(b), 5) for b in curve],
        "early_stopping": {"patience": cfg["patience"],
                           "best_epoch": best_e, "best_brier": best_b,
                           "best_score": round(C.score_of(best_b), 5),
                           "would_stop_after_epoch": stop_e},
        "full_run_argmin": {"epoch": full_e, "brier": curve[full_e - 1],
                            "score": round(C.score_of(curve[full_e - 1]), 5)},
        "valid_all": C.seg_metrics(yv, p_best),
        "valid_R": C.seg_metrics(yv[is_r], p_best[is_r]),
        "valid_F": C.seg_metrics(yv[~is_r], p_best[~is_r]),
        "cache_sha256": meta["npz_sha256"],
        "src_sha256": {f: C.sha256_file(os.path.join(C.REPO_SRC, f))
                       for f in ("common.py", "prep.py", "train_nn.py")},
        "fit_time_min": round((time.time() - t_start) / 60, 3),
    }
    np.save(os.path.join(root, "pred", f"pred_{tag}.npy"), p_best)
    np.save(os.path.join(root, "pred", f"all_{tag}.npy"), all_pred)
    res["pred_sha256"] = C.sha256_file(os.path.join(root, "pred", f"pred_{tag}.npy"))
    C.jdump(res, os.path.join(root, "res", f"res_{tag}.json"))
    print(f"[fit] {tag} DONE best_ep={best_e} brier={best_b:.8f} "
          f"score={C.score_of(best_b):.5f} in {res['fit_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
