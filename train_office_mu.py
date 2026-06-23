import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.models import resnet18, ResNet18_Weights

from create_dataset import office_dataloader

# Keep model cache inside project (avoids ~/.cache permission issues).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, ".torch"))
DEFAULT_SAVE_ROOT = "/scratch/chuang80/office_outputs"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OfficeMultiTaskNet(nn.Module):
    def __init__(self, tasks: List[str], class_num: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
            try:
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception as e:
                print(f"[WARN] Failed to load pretrained ResNet-18 ({e}). Falling back to random initialization.")
                backbone = resnet18(weights=None)
        else:
            backbone = resnet18(weights=None)
        # Match `train_office.py` encoder structure:
        # backbone feature map -> adaptive avgpool -> flatten -> hidden MLP
        self.resnet_network = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.heads = nn.ModuleDict({task: nn.Linear(512, class_num) for task in tasks})

        nn.init.normal_(self.hidden[0].weight, mean=0.0, std=0.005)
        nn.init.constant_(self.hidden[0].bias, 0.1)

    def forward_task(self, x: torch.Tensor, task: str) -> torch.Tensor:
        z = self.resnet_network(x)
        z = torch.flatten(self.avgpool(z), 1)
        z = self.hidden(z)
        return self.heads[task](z)


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    # Euclidean projection onto simplex {x >= 0, sum x = 1}
    n = v.numel()
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)


def solve_mgda_lambda(grad_matrix: torch.Tensor, iters: int = 60) -> torch.Tensor:
    # grad_matrix: [P, M], columns are per-task gradients
    m = grad_matrix.shape[1]
    gram = grad_matrix.t().matmul(grad_matrix)  # [M, M]
    lam = torch.full((m,), 1.0 / m, device=grad_matrix.device, dtype=grad_matrix.dtype)
    lr = 0.25
    for _ in range(iters):
        grad_lam = 2.0 * gram.matmul(lam)
        lam = project_to_simplex(lam - lr * grad_lam)
    return lam


def adaptive_mu_t(step: int, args: argparse.Namespace) -> float:
    """Threshold mu_t: fixed mu, or mu_scale * t^mu_alpha (Theta(t^alpha))."""
    if getattr(args, "mu_schedule", "fixed") == "fixed":
        return args.mu
    t = max(int(step), 1)
    return args.mu_scale * (t ** args.mu_alpha)


def fallback_bin_index(step: int, total_steps: int, num_bins: int) -> int:
    if num_bins <= 1:
        return 0
    if total_steps <= 1:
        return 0
    log_lo = math.log10(1.0)
    log_hi = math.log10(float(total_steps))
    log_t = math.log10(float(max(step, 1)))
    if log_hi <= log_lo:
        return 0
    idx = int((log_t - log_lo) / (log_hi - log_lo) * num_bins)
    return min(max(idx, 0), num_bins - 1)


def fallback_bin_center(bin_idx: int, total_steps: int, num_bins: int) -> float:
    if num_bins <= 1:
        return float(total_steps)
    log_lo = math.log10(1.0)
    log_hi = math.log10(float(max(total_steps, 1)))
    frac = (bin_idx + 0.5) / num_bins
    return 10.0 ** (log_lo + frac * (log_hi - log_lo))


def smallest_nonzero_eigval(grad_matrix: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Smallest positive eigenvalue of 2 P Q^T Q P, with Q's columns the per-task
    gradients (grad_matrix is [P, M]) and P = I - (1/M) 11^T (centering projector).
    """
    m = grad_matrix.shape[1]
    gram = grad_matrix.t().matmul(grad_matrix)
    ones = torch.ones(m, 1, device=grad_matrix.device, dtype=grad_matrix.dtype)
    p_mat = torch.eye(m, device=grad_matrix.device, dtype=grad_matrix.dtype) - (ones @ ones.t()) / m
    projected = 2.0 * p_mat.matmul(gram).matmul(p_mat)
    eigvals = torch.linalg.eigvalsh(projected)
    positive = eigvals[eigvals > eps]
    if positive.numel() == 0:
        return 0.0
    return float(positive.min().item())


def flatten_grads_for_loss(
    model: nn.Module,
    loss: torch.Tensor,
    params: List[nn.Parameter],
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat_parts = []
    for p, g in zip(params, grads):
        if g is None:
            flat_parts.append(torch.zeros_like(p).reshape(-1))
        else:
            flat_parts.append(g.reshape(-1))
    return torch.cat(flat_parts)


def mgda_stationarity(grad_matrix: torch.Tensor, lam: torch.Tensor) -> float:
    """
    Compute ||G lambda||_2^2 robustly on the current device.
    Uses elementwise multiply + reduction instead of GEMV to avoid occasional
    cublasSgemv INVALID_VALUE issues on older CUDA/PyTorch stacks.
    """
    if grad_matrix.dim() != 2:
        raise ValueError(f"Expected grad_matrix to be 2D [P, M], got shape={tuple(grad_matrix.shape)}")
    if lam.dim() != 1:
        lam = lam.reshape(-1)
    if grad_matrix.shape[1] != lam.numel():
        raise ValueError(
            f"Shape mismatch for stationarity: grad_matrix={tuple(grad_matrix.shape)}, lam={tuple(lam.shape)}"
        )
    direction = (grad_matrix * lam.unsqueeze(0)).sum(dim=1)
    return float(direction.pow(2).sum().item())


def eval_split(
    model: OfficeMultiTaskNet,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    task_acc = {}
    with torch.no_grad():
        for task, loader in loaders.items():
            correct = 0
            total = 0
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model.forward_task(x, task)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
            task_acc[task] = 100.0 * correct / max(total, 1)
    return task_acc


def make_output_dir(base_dir: str, run_name: str | None) -> str:
    os.makedirs(base_dir, exist_ok=True)
    tag = run_name if run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_history_csv(history: List[dict], csv_path: str) -> None:
    if not history:
        return
    keys = sorted(history[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _coerce_scalar(value: str):
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return text
    lower = text.lower()
    if lower in {"inf", "+inf"}:
        return float("inf")
    if lower == "-inf":
        return float("-inf")
    try:
        if any(c in text for c in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_history(history_path: str) -> List[dict]:
    if not os.path.isfile(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    if history_path.endswith(".json"):
        with open(history_path, "r") as f:
            history = json.load(f)
        if not isinstance(history, list):
            raise ValueError(f"Expected history JSON list, got: {type(history)}")
        return history
    if history_path.endswith(".csv"):
        with open(history_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return [{k: _coerce_scalar(v) for k, v in row.items()} for row in reader]
    raise ValueError("Unsupported history format. Use .json or .csv")


def safe_torch_save(obj: dict, path: str) -> None:
    """
    Robust checkpoint save for networked filesystems.
    Writes to a temp file first, then atomically replaces target.
    Uses legacy (non-zip) serialization to avoid occasional
    inline_container.cc position mismatch errors.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    try:
        torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def plot_curves(history: List[dict], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable ({e}). Skipping plot generation.")
        return

    if not history:
        return

    epochs = [r["epoch"] for r in history]
    losses = [r["train_loss"] for r in history]
    best_stationarity = [r["best_stationarity_so_far"] for r in history]
    c = best_stationarity[0] * math.sqrt(max(epochs[0], 1))
    reference_curve = [c / math.sqrt(max(epoch, 1)) for epoch in epochs]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, losses, label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, best_stationarity, label="empirical")
    plt.plot(epochs, reference_curve, "--", label=rf"reference: $c/\sqrt{{T}}$, c={c:.3e}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(
        r"$\min_{t\in[T],\,\lambda\in\Delta_M}\|\nabla F_S(x_t)\lambda\|^2$",
        fontsize=10,
    )
    plt.title("Convergence Curve")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_curve.png"), dpi=180)
    plt.close()


def plot_fallback_vs_step(fallback_bins: List[dict], out_dir: str) -> None:
    if not fallback_bins:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable ({e}). Skipping fallback plot.")
        return

    centers = [b["t_center"] for b in fallback_bins if b["total_steps"] > 0]
    rates = [b["fallback_rate"] for b in fallback_bins if b["total_steps"] > 0]
    if not centers:
        return

    plt.figure(figsize=(7, 4))
    plt.plot(centers, rates, "o-", label="fallback rate")
    plt.xscale("log")
    plt.xlabel("Step t (bin center, log scale)")
    plt.ylabel("Fallback rate in bin")
    plt.title(r"Adaptive fallback rate vs $t$")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fallback_vs_step.png"), dpi=180)
    plt.close()


def run_eval_only(args: argparse.Namespace) -> None:
    if args.dataset != "office-home":
        raise ValueError("This script is designed for --dataset office-home.")
    if not args.eval_checkpoint:
        raise ValueError("Please provide --eval_checkpoint in eval-only mode.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"[INFO] Using device: {device} ({torch.cuda.get_device_name(args.gpu_id)})")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA unavailable. Falling back to CPU for checkpoint evaluation.")

    tasks = ["Art", "Clipart", "Product", "Real_World"]
    class_num = 65
    data_loader, _ = office_dataloader(
        dataset=args.dataset,
        batchsize=args.bs,
        root_path=args.dataset_path,
        balanced=args.balanced,
    )
    test_loaders = {task: data_loader[task]["test"] for task in tasks}

    model = OfficeMultiTaskNet(
        tasks=tasks,
        class_num=class_num,
        pretrained=False,
    ).to(device)
    ckpt = torch.load(args.eval_checkpoint, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"No model_state_dict in checkpoint: {args.eval_checkpoint}")
    model.load_state_dict(ckpt["model_state_dict"])

    test_acc = eval_split(model, test_loaders, device)
    avg_test = sum(test_acc.values()) / len(test_acc)
    print(f"\n[INFO] Evaluated checkpoint: {args.eval_checkpoint}")
    for task, acc in test_acc.items():
        print(f"  {task:10s}: {acc:.2f}%")
    print(f"[INFO] Average test accuracy: {avg_test:.2f}%")


def bootstrap_officehome_from_hf(dataset_path: str, script_dir: str) -> None:
    """
    Populate Office-Home folder structure expected by `data_txt/office-home/*.txt`
    using Hugging Face dataset `flwrlabs/office-home`.
    """
    list_root = os.path.join(script_dir, "data_txt", "office-home")
    if not os.path.isdir(list_root):
        raise FileNotFoundError(f"Split list folder not found: {list_root}")

    required_paths = set()
    for fn in os.listdir(list_root):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(list_root, fn), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    required_paths.add(line.split(" ")[0])

    req_groups = defaultdict(list)
    for rel in sorted(required_paths):
        parts = rel.split("/")
        if len(parts) >= 3:
            req_groups[(parts[0], parts[1])].append(rel)

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Auto-bootstrap needs `datasets` package. "
            "Install with: pip install datasets pillow"
        ) from e

    print("[INFO] Auto-bootstrap: downloading Office-Home images from Hugging Face...")
    ds = load_dataset("flwrlabs/office-home", split="train")

    src_groups = defaultdict(list)
    for i in range(len(ds)):
        item = ds[i]
        domain = item["domain"].replace("Real World", "Real_World")
        cls = ds.features["label"].int2str(item["label"])
        src_groups[(domain, cls)].append(item["image"])

    os.makedirs(dataset_path, exist_ok=True)
    written = 0
    missing_groups = []
    for key, rel_list in req_groups.items():
        imgs = src_groups.get(key, [])
        if not imgs:
            missing_groups.append(key)
            continue
        n = len(imgs)
        for j, rel in enumerate(rel_list):
            out_path = os.path.join(dataset_path, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if not os.path.exists(out_path):
                imgs[j % n].save(out_path, format="JPEG", quality=95)
                written += 1

    print(f"[INFO] Auto-bootstrap complete. Wrote {written} files into {dataset_path}")
    if missing_groups:
        print(f"[WARN] Missing domain/class groups from source: {len(missing_groups)}")


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this training run is configured to require GPU.")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"[INFO] Using device: {device} ({torch.cuda.get_device_name(args.gpu_id)})")

    if args.dataset != "office-home":
        raise ValueError("This script is designed for --dataset office-home.")

    tasks = ["Art", "Clipart", "Product", "Real_World"]
    class_num = 65

    data_loader, _ = office_dataloader(
        dataset=args.dataset,
        batchsize=args.bs,
        root_path=args.dataset_path,
        balanced=args.balanced,
    )
    train_loaders = {task: data_loader[task]["train"] for task in tasks}
    val_loaders = {task: data_loader[task]["val"] for task in tasks}
    test_loaders = {task: data_loader[task]["test"] for task in tasks}

    model = OfficeMultiTaskNet(
        tasks=tasks,
        class_num=class_num,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = [p for p in model.parameters() if p.requires_grad]

    start_epoch = 1
    seed_best_stationarity = float("inf")
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        prev_epoch = int(ckpt.get("epoch", 0))
        start_epoch = prev_epoch + 1
        seed_best_stationarity = float(ckpt.get("best_stationarity_so_far", float("inf")))
        print(f"[INFO] Resumed model weights from: {args.resume_checkpoint}")
        print(f"[INFO] Previous epoch={prev_epoch}, continuing from epoch={start_epoch}")

    lambda0 = torch.full((len(tasks),), 1.0 / len(tasks), device=device)

    train_iters = {task: iter(loader) for task, loader in train_loaders.items()}
    steps_per_epoch = min(len(train_loaders[t]) for t in tasks)
    end_epoch = args.epochs
    if args.resume_checkpoint and args.extra_epochs is not None:
        end_epoch = start_epoch + args.extra_epochs - 1
    total_epochs_this_run = end_epoch - start_epoch + 1
    total_steps = total_epochs_this_run * steps_per_epoch

    # Theorem-aligned setting:
    # alpha_t = alpha with alpha = Theta(T^{-1/2}), |Z_t| = Theta(t), and mu_t >= mu > 0.
    if args.theory_mode:
        # fixed_alpha = args.alpha_scale / max(total_steps, 1) ** 0.5
        # for pg in optimizer.param_groups:
        #     pg["lr"] = fixed_alpha
        print(
            # f"[INFO] theory_mode enabled | alpha={fixed_alpha:.6e} (c/sqrt(T)), "
            f"[INFO] theory_mode enabled "
            f"base_bs={args.bs}, batch_growth_rate={args.batch_growth_rate}, "
            f"mu_schedule={args.mu_schedule}"
        )
    if args.algorithm == "adaptive":
        if args.mu_schedule == "fixed":
            print(f"[INFO] adaptive mu_t = {args.mu} (fixed)")
        else:
            print(
                f"[INFO] adaptive mu_t = {args.mu_scale} * t^{args.mu_alpha} "
                f"(Theta(t^{args.mu_alpha}))"
            )

    out_dir = make_output_dir(args.save_dir, args.run_name)
    history: List[dict] = []
    global_step = 0
    best_stationarity = seed_best_stationarity
    best_val = -float("inf")
    num_fallback_bins = args.fallback_bins if args.algorithm == "adaptive" else 0
    fallback_bin_stats = [
        {"total_steps": 0, "fallback_steps": 0, "eig_sum": 0.0, "mu_sum": 0.0}
        for _ in range(num_fallback_bins)
    ]

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        running_loss = 0.0
        smg_used = 0
        fallback_used = 0
        epoch_stationarity = 0.0
        epoch_eig_sum = 0.0
        epoch_eig_min = float("inf")
        epoch_mu_sum = 0.0

        for _ in range(steps_per_epoch):
            global_step += 1
            if args.theory_mode:
                # Effective batch size grows approximately linearly with t.
                batch_factor = 1 + int(args.batch_growth_rate * global_step)
                batch_factor = min(batch_factor, args.max_batch_factor)
            else:
                batch_factor = 1

            batch_by_task: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for task in tasks:
                try:
                    x, y = next(train_iters[task])
                except StopIteration:
                    train_iters[task] = iter(train_loaders[task])
                    x, y = next(train_iters[task])
                xs = [x.to(device, non_blocking=True)]
                ys = [y.to(device, non_blocking=True)]
                for _k in range(1, batch_factor):
                    try:
                        xk, yk = next(train_iters[task])
                    except StopIteration:
                        train_iters[task] = iter(train_loaders[task])
                        xk, yk = next(train_iters[task])
                    xs.append(xk.to(device, non_blocking=True))
                    ys.append(yk.to(device, non_blocking=True))
                batch_by_task[task] = (torch.cat(xs, dim=0), torch.cat(ys, dim=0))

            optimizer.zero_grad(set_to_none=True)

            task_losses = []
            task_grads = []
            for task in tasks:
                x, y = batch_by_task[task]
                logits = model.forward_task(x, task)
                loss = F.cross_entropy(logits, y)
                task_losses.append(loss)
                task_grads.append(flatten_grads_for_loss(model, loss, params))

            grad_matrix = torch.stack(task_grads, dim=1)  # [P, M]
            eig_min = smallest_nonzero_eigval(grad_matrix)
            epoch_eig_sum += eig_min
            epoch_eig_min = min(epoch_eig_min, eig_min)
            lam_metric = solve_mgda_lambda(grad_matrix, iters=args.lambda_iters)
            stationarity = mgda_stationarity(grad_matrix, lam_metric)
            epoch_stationarity += stationarity
            best_stationarity = min(best_stationarity, stationarity)

            if args.algorithm == "smg":
                lam = solve_mgda_lambda(grad_matrix, iters=args.lambda_iters)
                smg_used += 1
            else:
                mu_t = adaptive_mu_t(global_step, args)
                epoch_mu_sum += mu_t
                used_fallback = eig_min < mu_t
                if used_fallback:
                    lam = lambda0
                    fallback_used += 1
                else:
                    lam = solve_mgda_lambda(grad_matrix, iters=args.lambda_iters)
                    smg_used += 1
                if num_fallback_bins > 0:
                    bidx = fallback_bin_index(global_step, total_steps, num_fallback_bins)
                    bin_rec = fallback_bin_stats[bidx]
                    bin_rec["total_steps"] += 1
                    bin_rec["fallback_steps"] += int(used_fallback)
                    bin_rec["eig_sum"] += eig_min
                    bin_rec["mu_sum"] += mu_t

            combined_loss = sum(w * l for w, l in zip(lam, task_losses))
            combined_loss.backward()
            optimizer.step()
            running_loss += float(combined_loss.detach().item())

        val_acc = eval_split(model, val_loaders, device)
        avg_val = sum(val_acc.values()) / len(val_acc)
        avg_epoch_stationarity = epoch_stationarity / max(steps_per_epoch, 1)
        avg_eig_min = epoch_eig_sum / max(steps_per_epoch, 1)
        msg = (
            f"Epoch {epoch:03d} | loss={running_loss/steps_per_epoch:.4f} | "
            f"val_avg_acc={avg_val:.2f}% | stationarity={avg_epoch_stationarity:.4e} "
            f"| best_stationarity={best_stationarity:.4e} | "
            f"eig_min_avg={avg_eig_min:.4e} | eig_min_min={epoch_eig_min:.4e} | smg_steps={smg_used}"
        )
        if args.algorithm == "adaptive":
            msg += f" | fallback_steps={fallback_used}"
            if args.mu_schedule != "fixed":
                avg_mu_t = epoch_mu_sum / max(steps_per_epoch, 1)
                msg += f" | mu_t_avg={avg_mu_t:.4e}"
        print(msg)

        epoch_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / steps_per_epoch,
            "val_avg_acc": avg_val,
            "epoch_stationarity": avg_epoch_stationarity,
            "best_stationarity_so_far": best_stationarity,
            "smg_steps": smg_used,
            "fallback_steps": fallback_used,
            "effective_batch_factor": batch_factor,
            "effective_batch_size": batch_factor * args.bs,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if args.algorithm == "adaptive":
            epoch_row["fallback_rate"] = fallback_used / max(steps_per_epoch, 1)
            if args.mu_schedule != "fixed":
                epoch_row["mu_t_avg"] = epoch_mu_sum / max(steps_per_epoch, 1)
        history.append(epoch_row)
        if epoch % args.save_every == 0:
            safe_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_avg_acc": avg_val,
                    "best_stationarity_so_far": best_stationarity,
                    "args": vars(args),
                },
                os.path.join(out_dir, f"model_epoch_smg_{epoch:04d}.pt"),
            )
        if avg_val > best_val:
            best_val = avg_val
            safe_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val,
                    "args": vars(args),
                },
                os.path.join(out_dir, "best_model.pt"),
            )

    test_acc = eval_split(model, test_loaders, device)
    avg_test = sum(test_acc.values()) / len(test_acc)
    print("\nFinal test accuracy by domain:")
    for task, acc in test_acc.items():
        print(f"  {task:10s}: {acc:.2f}%")
    print(f"Average test accuracy: {avg_test:.2f}%")
    print(f"Best stationarity proxy: {best_stationarity:.4e}")

    # Save final artifacts
    safe_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": test_acc,
            "avg_test_acc": avg_test,
            "best_stationarity": best_stationarity,
            "args": vars(args),
        },
        os.path.join(out_dir, "last_model.pt"),
    )
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_history_csv(history, os.path.join(out_dir, "history.csv"))
    plot_curves(history, out_dir)
    if args.algorithm == "adaptive" and fallback_bin_stats:
        fallback_bins = []
        for bidx, rec in enumerate(fallback_bin_stats):
            total = rec["total_steps"]
            fb = rec["fallback_steps"]
            fallback_bins.append(
                {
                    "bin": bidx,
                    "t_center": fallback_bin_center(bidx, total_steps, num_fallback_bins),
                    "total_steps": total,
                    "fallback_steps": fb,
                    "fallback_rate": fb / total if total > 0 else float("nan"),
                    "eig_min_avg": rec["eig_sum"] / total if total > 0 else float("nan"),
                    "mu_t_avg": rec["mu_sum"] / total if total > 0 else float("nan"),
                }
            )
        with open(os.path.join(out_dir, "fallback_by_step.json"), "w") as f:
            json.dump(fallback_bins, f, indent=2)
        plot_fallback_vs_step(fallback_bins, out_dir)
        total_fb = sum(r["fallback_steps"] for r in fallback_bin_stats)
        total_all = sum(r["total_steps"] for r in fallback_bin_stats)
        print(
            f"[INFO] Overall fallback rate: {total_fb}/{total_all} "
            f"({100.0 * total_fb / max(total_all, 1):.1f}%) | "
            f"see fallback_by_step.json and fallback_vs_step.png"
        )
    print(f"[INFO] Saved outputs to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Office-Home training with SMG / adaptive SMG")
    parser.add_argument("--dataset", type=str, default="office-home")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to Office-Home root containing Art/Clipart/Product/Real_World. Auto-detected if omitted.",
    )
    parser.add_argument("--algorithm", type=str, choices=["smg", "adaptive"], default="adaptive")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable ImageNet pretrained ResNet-18 initialization.",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument("--lambda_iters", type=int, default=60)
    parser.add_argument(
        "--mu",
        type=float,
        default=0.1,
        help="Fixed adaptive threshold when --mu_schedule fixed.",
    )
    parser.add_argument(
        "--mu_schedule",
        type=str,
        choices=["fixed", "power"],
        default="power",
        help="mu_t schedule: fixed mu, or mu_scale * t^mu_alpha.",
    )
    parser.add_argument(
        "--mu_alpha",
        type=float,
        default=-0.25,
        help="Exponent alpha in mu_t = mu_scale * t^alpha (used when --mu_schedule power).",
    )
    parser.add_argument(
        "--mu_scale",
        type=float,
        default=1,
        help="Scale c in mu_t = c * t^alpha; defaults to --mu when omitted.",
    )
    parser.add_argument(
        "--fallback_bins",
        type=int,
        default=20,
        help="Log-spaced step bins for fallback rate vs t diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--balanced", action="store_true", help="Use balanced text lists if available.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_ROOT,
        help="Directory to save checkpoints and logs (default: /scratch/chuang80/office_outputs).",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--theory_mode",
        action="store_true",
        help="Use theorem-aligned defaults: alpha=c/sqrt(T), increasing effective batch size, fixed mu_t>=mu.",
    )
    parser.set_defaults(theory_mode=False)
    parser.add_argument("--alpha_scale", type=float, default=1.0, help="c in alpha=c/sqrt(T) when --theory_mode.")
    parser.add_argument("--batch_growth_rate", type=float, default=0.005, help="Linear growth rate for effective batch factor.")
    parser.add_argument("--max_batch_factor", type=int, default=4, help="Cap on effective batch factor growth.")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint .pt to resume model weights.")
    parser.add_argument(
        "--eval_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt for evaluation-only mode (no training).",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip training; regenerate figures from a saved history file.",
    )
    parser.add_argument(
        "--history_path",
        type=str,
        default=None,
        help="Path to history.json or history.csv used with --plot_only.",
    )
    parser.add_argument(
        "--extra_epochs",
        type=int,
        default=None,
        help="When resuming, train this many additional epochs from checkpoint epoch.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mu_scale is None:
        args.mu_scale = args.mu

    if args.plot_only:
        if args.history_path is None:
            raise ValueError("With --plot_only, please provide --history_path to a history.json or history.csv file.")
        history = load_history(args.history_path)
        out_dir = os.path.dirname(os.path.abspath(args.history_path))
        plot_curves(history, out_dir)
        print(f"[INFO] Plot-only mode complete. Regenerated figures in: {out_dir}")
        raise SystemExit(0)

    if args.dataset_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        candidates = [
            os.path.join(project_root, "OfficeHomeDataset"),
            os.path.join(project_root, "office-home"),
            os.path.join(script_dir, "OfficeHomeDataset"),
            os.path.join(script_dir, "office-home"),
            project_root,
            script_dir,
            os.getcwd(),
        ]
        required_domains = ["Art", "Clipart", "Product", "Real_World"]
        detected = None
        for candidate in candidates:
            if all(os.path.isdir(os.path.join(candidate, d)) for d in required_domains):
                detected = candidate
                break
        if detected is None:
            detected = os.path.join(project_root, "OfficeHomeDataset")
            print(f"[INFO] No local Office-Home folder detected. Using bootstrap target: {detected}")
            bootstrap_officehome_from_hf(detected, script_dir)
        args.dataset_path = detected
        print(f"[INFO] Auto-detected dataset_path: {args.dataset_path}")

    if args.eval_checkpoint is not None:
        run_eval_only(args)
        raise SystemExit(0)

    run_training(args)
