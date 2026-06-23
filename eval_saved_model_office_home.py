import argparse
import json
import os
from typing import Dict, List

import torch

from create_dataset import office_dataloader
from train_office_stochastic import OfficeMultiTaskNet, eval_split


def auto_detect_dataset_path(script_dir: str) -> str:
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
    for candidate in candidates:
        if all(os.path.isdir(os.path.join(candidate, d)) for d in required_domains):
            return candidate
    raise FileNotFoundError(
        "Could not auto-detect Office-Home dataset root. "
        "Please pass --dataset_path explicitly."
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_path: str,
    batch_size: int,
    gpu_id: int,
) -> Dict[str, float]:
    tasks: List[str] = ["Art", "Clipart", "Product", "Real_World"]
    class_num = 65

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[INFO] Using device: {device} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA unavailable. Falling back to CPU.")

    data_loader, _ = office_dataloader(
        dataset="office-home",
        batchsize=batch_size,
        root_path=dataset_path,
        balanced=False,
    )
    test_loaders = {task: data_loader[task]["test"] for task in tasks}

    model = OfficeMultiTaskNet(tasks=tasks, class_num=class_num, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"No model_state_dict found in checkpoint: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])

    task_acc = eval_split(model, test_loaders, device)
    avg_acc = sum(task_acc.values()) / len(task_acc)
    result = dict(task_acc)
    result["avg_test_acc"] = avg_acc
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate saved Office-Home checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved .pt checkpoint")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to Office-Home root containing Art/Clipart/Product/Real_World",
    )
    parser.add_argument("--bs", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional output JSON path for metrics",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.dataset_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = auto_detect_dataset_path(script_dir)
        print(f"[INFO] Auto-detected dataset_path: {dataset_path}")
    else:
        dataset_path = args.dataset_path

    metrics = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        batch_size=args.bs,
        gpu_id=args.gpu_id,
    )

    print(f"\n[INFO] Evaluated checkpoint: {checkpoint_path}")
    for task in ["Art", "Clipart", "Product", "Real_World"]:
        print(f"  {task:10s}: {metrics[task]:.2f}%")
    print(f"[INFO] Average test accuracy: {metrics['avg_test_acc']:.2f}%")

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved metrics JSON to: {out_path}")


if __name__ == "__main__":
    main()
