#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference_Taiwan_Precip_Metrics_CSVold.py

MAE/RMSE inference for the author-style Taiwan precipitation EDM model.

Compatible local files:
  networkold.py
  DatasetTaiwanERA_IDW_tp_bilinear.py

Compatible checkpoints:
  Pure state_dict checkpoints produced by
  TrainDiffusion_Taiwan_Bilinear_DDP_AuthorStyle.py, for example:
    outputs_taiwan_8km_bilinear_authorstyle_150/Model/149.pt

Outputs:
  1. Per-date CSV containing coarse/model MAE and RMSE.
  2. Summary CSV containing global pixel-weighted MAE and RMSE.

Metric domain:
  Land pixels only, matching the author-style training dataset configuration.

Prediction used for MAE/RMSE:
  Mean of all generated ensemble members for each date.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from tqdm import tqdm

import networkold
from DatasetTaiwanERA_IDW_tp_bilinear import (
    TaiwanERAPrecipDataset,
    load_stats,
)


# -----------------------------------------------------------------------------
# Distributed helpers


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def print0(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_distributed(args) -> torch.device:
    args.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not args.distributed:
        return choose_device(args.device)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    if device.type == "cuda":
        dist.barrier(device_ids=[args.local_rank])
    else:
        dist.barrier()
    return device


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        try:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        finally:
            dist.destroy_process_group()


class DistributedEvalSampler(Sampler):
    """Rank-strided evaluation sampler without padding or duplicated dates."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.rank = get_rank()
        self.world_size = get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


# -----------------------------------------------------------------------------
# General helpers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_module_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }
    return state_dict


def numeric_checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def resolve_checkpoint_path(value: str) -> Path:
    """Resolve a checkpoint file or choose the highest numeric epoch in Model/."""
    path = Path(value).expanduser()

    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Checkpoint path must be a file or directory: {path}")

    model_dir = path if path.name == "Model" else path / "Model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Cannot find Model directory below: {path}")

    candidates = [
        candidate
        for candidate in model_dir.glob("*.pt")
        if numeric_checkpoint_epoch(candidate) >= 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No numeric author-style checkpoint such as 149.pt found in {model_dir}"
        )

    return max(candidates, key=numeric_checkpoint_epoch)


def infer_output_dir_from_checkpoint(checkpoint: Path) -> Optional[Path]:
    if checkpoint.parent.name == "Model":
        return checkpoint.parent.parent
    return None


def resolve_stats_path(
    checkpoint_path: Path,
    stats_path: Optional[str],
    resolution: str,
) -> Path:
    if stats_path:
        path = Path(stats_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Stats file not found: {path}")
        return path

    output_dir = infer_output_dir_from_checkpoint(checkpoint_path)
    if output_dir is not None:
        candidate = output_dir / f"stats_{resolution}.json"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Cannot find stats file. Pass --stats-path, or keep stats_8km.json "
        "in the same training output directory as Model/."
    )


def load_author_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    obj = torch.load(checkpoint_path, map_location=device)

    # The author-style trainer saves a pure state_dict. This fallback also
    # accepts a wrapped dictionary when one is supplied accidentally.
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]

    if not isinstance(obj, dict):
        raise TypeError(
            f"Unsupported checkpoint object type: {type(obj).__name__}"
        )

    return strip_module_prefix(obj)


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Author-style EDM sampling


@torch.no_grad()
def sample_model_eds_batch(
    batch: Dict,
    model: torch.nn.Module,
    dataset: TaiwanERAPrecipDataset,
    device: torch.device,
    num_steps: int = 40,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 40.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
) -> torch.Tensor:
    """Generate one author-style stochastic prediction for one batch."""
    model.eval()

    images_input = batch["inputs"].to(device, non_blocking=True)
    coarse = batch["coarse"]
    condition_params = torch.stack(
        (
            batch["year"].to(device, non_blocking=True),
            batch["doy"].to(device, non_blocking=True),
        ),
        dim=1,
    )

    sigma_min = max(float(sigma_min), float(model.sigma_min))
    sigma_max = min(float(sigma_max), float(model.sigma_max))

    init_noise = torch.randn(
        (
            images_input.shape[0],
            dataset.target_channels,
            images_input.shape[2],
            images_input.shape[3],
        ),
        dtype=torch.float64,
        device=device,
    )

    step_indices = torch.arange(
        num_steps,
        dtype=torch.float64,
        device=device,
    )
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [
            model.round_sigma(t_steps),
            torch.zeros_like(t_steps[:1]),
        ]
    )

    x_next = init_noise * t_steps[0]

    for index, (t_cur, t_next) in enumerate(
        zip(t_steps[:-1], t_steps[1:])
    ):
        x_cur = x_next

        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= float(t_cur) <= S_max
            else 0.0
        )
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = (
            x_cur
            + (t_hat ** 2 - t_cur ** 2).sqrt()
            * S_noise
            * torch.randn_like(x_cur)
        )

        denoised = model(
            x_hat,
            t_hat,
            images_input,
            condition_params,
        ).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if index < num_steps - 1:
            denoised = model(
                x_next,
                t_next,
                images_input,
                condition_params,
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (
                0.5 * d_cur + 0.5 * d_prime
            )

    prediction = dataset.residual_to_fine_image(
        x_next.detach().cpu(),
        coarse,
    )
    return prediction.clamp_min(0)


# -----------------------------------------------------------------------------
# MAE/RMSE


def expand_valid_mask(
    valid_mask: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    valid_mask = valid_mask.float().cpu()
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(0)
    if valid_mask.shape[0] == 1 and batch_size != 1:
        valid_mask = valid_mask.expand(batch_size, -1, -1, -1)
    return valid_mask


def metric_batch(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    ensemble_predictions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Compute per-date and global-sum components.

    ensemble_predictions shape: [M, B, 1, H, W]
    The model forecast used for MAE/RMSE is the ensemble mean.
    """
    coarse = coarse.float().cpu().clamp_min(0)
    fine = fine.float().cpu().clamp_min(0)
    ensemble_predictions = ensemble_predictions.float().cpu().clamp_min(0)
    valid_mask = expand_valid_mask(valid_mask, fine.shape[0])

    prediction_mean = ensemble_predictions.mean(dim=0)

    coarse_error = coarse - fine
    model_error = prediction_mean - fine

    coarse_abs = coarse_error.abs()
    coarse_squared = coarse_error.square()
    model_abs = model_error.abs()
    model_squared = model_error.square()

    pixel_count_by_date = valid_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)

    coarse_mae = (
        (coarse_abs * valid_mask).sum(dim=(1, 2, 3))
        / pixel_count_by_date
    )
    coarse_rmse = torch.sqrt(
        (coarse_squared * valid_mask).sum(dim=(1, 2, 3))
        / pixel_count_by_date
    )
    model_mae = (
        (model_abs * valid_mask).sum(dim=(1, 2, 3))
        / pixel_count_by_date
    )
    model_rmse = torch.sqrt(
        (model_squared * valid_mask).sum(dim=(1, 2, 3))
        / pixel_count_by_date
    )

    rows = []
    for batch_index in range(fine.shape[0]):
        rows.append(
            {
                "n_valid_pixels": int(
                    pixel_count_by_date[batch_index].item()
                ),
                "coarse_mae": float(coarse_mae[batch_index].item()),
                "coarse_rmse": float(coarse_rmse[batch_index].item()),
                "model_mae": float(model_mae[batch_index].item()),
                "model_rmse": float(model_rmse[batch_index].item()),
            }
        )

    totals = {
        "n_pixels": float(valid_mask.sum().item()),
        "coarse_abs_sum": float((coarse_abs * valid_mask).sum().item()),
        "coarse_squared_sum": float(
            (coarse_squared * valid_mask).sum().item()
        ),
        "model_abs_sum": float((model_abs * valid_mask).sum().item()),
        "model_squared_sum": float(
            (model_squared * valid_mask).sum().item()
        ),
    }
    return rows, totals


# -----------------------------------------------------------------------------
# CLI and main


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Author-style Taiwan bilinear diffusion inference: "
            "write land-only MAE/RMSE to CSV."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Numeric .pt checkpoint, Model directory, or training output "
            "directory. A directory selects its highest numeric epoch .pt."
        ),
    )
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--resolution",
        default="8km",
        choices=["1km", "5km", "8km"],
    )
    parser.add_argument("--test-start", default="20180101")
    parser.add_argument("--test-end", default="20181231")
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-members", type=int, default=30)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--S-churn", type=float, default=40.0)
    parser.add_argument("--S-min", type=float, default=0.0)
    parser.add_argument("--S-max", type=float, default=float("inf"))
    parser.add_argument("--S-noise", type=float, default=1.0)

    parser.add_argument(
        "--output-csv",
        default="./precip_metrics_old_per_date.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="./precip_metrics_old_summary.csv",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.ensemble_members < 1:
        raise ValueError("--ensemble-members must be >= 1.")
    if args.num_steps < 2:
        raise ValueError("--num-steps must be >= 2.")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    device = setup_distributed(args)
    set_seed(args.seed + get_rank())

    stats_path = resolve_stats_path(
        checkpoint_path,
        args.stats_path,
        args.resolution,
    )
    stats = load_stats(stats_path)

    if stats.get("resolution") != args.resolution:
        raise ValueError(
            f"Stats resolution={stats.get('resolution')!r}, "
            f"expected {args.resolution!r}."
        )
    if stats.get("target_transform") != "log1p":
        raise ValueError(
            "This author-style inference expects target_transform='log1p'."
        )
    if stats.get("spatial_mask") != "land":
        raise ValueError(
            "This author-style run was trained with a land mask; "
            "stats spatial_mask must be 'land'."
        )
    if stats.get("tp_resize") != "bilinear":
        raise ValueError(
            f"Stats tp_resize={stats.get('tp_resize')!r}; expected 'bilinear'."
        )

    dataset_test = TaiwanERAPrecipDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        start_date=args.test_start,
        end_date=args.test_end,
        condition_vars=["q700", "t2m", "u", "v", "msl", "tp"],
        use_mask=True,
        target_transform="log1p",
        stats=stats,
        max_samples=args.max_test_samples,
    )

    sampler = (
        DistributedEvalSampler(dataset_test)
        if getattr(args, "distributed", False)
        else None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_in_channels = (
        dataset_test.num_input_channels + dataset_test.target_channels
    )
    model_out_channels = dataset_test.target_channels

    model = networkold.EDMPrecond(
        dataset_test.img_resolution,
        model_in_channels,
        model_out_channels,
        label_dim=2,
    ).to(device)

    state_dict = load_author_checkpoint(checkpoint_path, device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not match networkold.EDMPrecond with the "
            f"Taiwan bilinear dataset (in_channels={model_in_channels}, "
            f"out_channels={model_out_channels}). Original error: {exc}"
        ) from exc
    model.eval()

    print0(f"[Device] {device}")
    print0(f"[Checkpoint] {checkpoint_path}")
    print0(f"[Stats] {stats_path}")
    print0(
        f"[Dataset] dates={len(dataset_test)} range="
        f"{args.test_start}-{args.test_end} resolution={args.resolution}"
    )
    print0(
        f"[Dataset] bilinear tp, land-only metrics, "
        f"valid_pixels_per_date={int(dataset_test.valid_mask.sum().item())}"
    )
    print0(
        f"[Inference] distributed={getattr(args, 'distributed', False)} "
        f"world_size={get_world_size()} ensemble_members="
        f"{args.ensemble_members} num_steps={args.num_steps}"
    )

    per_date_rows: List[Dict] = []
    total_pixels = 0.0
    total_coarse_abs = 0.0
    total_coarse_squared = 0.0
    total_model_abs = 0.0
    total_model_squared = 0.0

    progress = tqdm(
        dataloader,
        desc=f"Inference rank {get_rank()}",
        dynamic_ncols=True,
        disable=not is_main_process(),
    )

    for batch in progress:
        fine = batch["fine"].float().cpu().clamp_min(0)
        coarse = batch["coarse"].float().cpu().clamp_min(0)
        valid_mask = batch["valid_mask"].float().cpu()

        prediction_members = []
        for _member in range(args.ensemble_members):
            prediction_members.append(
                sample_model_eds_batch(
                    batch=batch,
                    model=model,
                    dataset=dataset_test,
                    device=device,
                    num_steps=args.num_steps,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    rho=args.rho,
                    S_churn=args.S_churn,
                    S_min=args.S_min,
                    S_max=args.S_max,
                    S_noise=args.S_noise,
                )
            )

        predictions = torch.stack(prediction_members, dim=0)
        rows, totals = metric_batch(
            coarse,
            fine,
            predictions,
            valid_mask,
        )

        dates = batch["date"]
        if isinstance(dates, str):
            dates = [dates]

        for row, date in zip(rows, dates):
            per_date_rows.append(
                {
                    "date": str(date),
                    "n_valid_pixels": row["n_valid_pixels"],
                    "coarse_mae": row["coarse_mae"],
                    "coarse_rmse": row["coarse_rmse"],
                    "model_mae": row["model_mae"],
                    "model_rmse": row["model_rmse"],
                    "ensemble_members": args.ensemble_members,
                    "num_steps": args.num_steps,
                }
            )

        total_pixels += totals["n_pixels"]
        total_coarse_abs += totals["coarse_abs_sum"]
        total_coarse_squared += totals["coarse_squared_sum"]
        total_model_abs += totals["model_abs_sum"]
        total_model_squared += totals["model_squared_sum"]

    output_csv_path = Path(args.output_csv)
    summary_csv_path = Path(args.summary_csv)
    part_dir = output_csv_path.parent / f"_{output_csv_path.stem}_ddp_parts"

    if is_main_process():
        part_dir.mkdir(parents=True, exist_ok=True)
    if is_dist_avail_and_initialized():
        dist.barrier()

    part_csv = part_dir / f"per_date_rank{get_rank()}.csv"
    part_json = part_dir / f"totals_rank{get_rank()}.json"

    per_date_fields = [
        "date",
        "n_valid_pixels",
        "coarse_mae",
        "coarse_rmse",
        "model_mae",
        "model_rmse",
        "ensemble_members",
        "num_steps",
    ]
    write_csv(part_csv, per_date_rows, per_date_fields)
    part_json.write_text(
        json.dumps(
            {
                "n_pixels": total_pixels,
                "coarse_abs_sum": total_coarse_abs,
                "coarse_squared_sum": total_coarse_squared,
                "model_abs_sum": total_model_abs,
                "model_squared_sum": total_model_squared,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if is_dist_avail_and_initialized():
        dist.barrier()

    if is_main_process():
        merged_rows: List[Dict] = []
        totals_all = {
            "n_pixels": 0.0,
            "coarse_abs_sum": 0.0,
            "coarse_squared_sum": 0.0,
            "model_abs_sum": 0.0,
            "model_squared_sum": 0.0,
        }

        for rank in range(get_world_size()):
            rank_csv = part_dir / f"per_date_rank{rank}.csv"
            rank_json = part_dir / f"totals_rank{rank}.json"

            with rank_csv.open("r", newline="", encoding="utf-8") as handle:
                merged_rows.extend(list(csv.DictReader(handle)))

            rank_totals = json.loads(
                rank_json.read_text(encoding="utf-8")
            )
            for key in totals_all:
                totals_all[key] += float(rank_totals[key])

        merged_rows.sort(key=lambda row: row["date"])

        denominator = max(totals_all["n_pixels"], 1e-12)
        coarse_mae = totals_all["coarse_abs_sum"] / denominator
        coarse_rmse = math.sqrt(
            totals_all["coarse_squared_sum"] / denominator
        )
        model_mae = totals_all["model_abs_sum"] / denominator
        model_rmse = math.sqrt(
            totals_all["model_squared_sum"] / denominator
        )

        mae_improvement = coarse_mae - model_mae
        rmse_improvement = coarse_rmse - model_rmse
        mae_improvement_percent = (
            100.0 * mae_improvement / max(coarse_mae, 1e-12)
        )
        rmse_improvement_percent = (
            100.0 * rmse_improvement / max(coarse_rmse, 1e-12)
        )

        common = {
            "n_dates": len(merged_rows),
            "n_valid_pixels_total": int(totals_all["n_pixels"]),
            "ensemble_members": args.ensemble_members,
            "num_steps": args.num_steps,
            "checkpoint": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "stats_path": str(stats_path),
            "mae_improvement": mae_improvement,
            "mae_improvement_percent": mae_improvement_percent,
            "rmse_improvement": rmse_improvement,
            "rmse_improvement_percent": rmse_improvement_percent,
            "beats_coarse_mae": int(model_mae < coarse_mae),
            "beats_coarse_rmse": int(model_rmse < coarse_rmse),
            "beats_coarse_both": int(
                model_mae < coarse_mae and model_rmse < coarse_rmse
            ),
        }

        summary_rows = [
            {
                "comparison": "coarse_vs_groundtruth",
                "mae": coarse_mae,
                "rmse": coarse_rmse,
                **common,
            },
            {
                "comparison": "prediction_vs_groundtruth",
                "mae": model_mae,
                "rmse": model_rmse,
                **common,
            },
        ]

        summary_fields = [
            "comparison",
            "mae",
            "rmse",
            "n_dates",
            "n_valid_pixels_total",
            "ensemble_members",
            "num_steps",
            "checkpoint",
            "checkpoint_path",
            "stats_path",
            "mae_improvement",
            "mae_improvement_percent",
            "rmse_improvement",
            "rmse_improvement_percent",
            "beats_coarse_mae",
            "beats_coarse_rmse",
            "beats_coarse_both",
        ]

        write_csv(output_csv_path, merged_rows, per_date_fields)
        write_csv(summary_csv_path, summary_rows, summary_fields)

        print0(
            f"[Summary] coarse MAE={coarse_mae:.6f} "
            f"RMSE={coarse_rmse:.6f}"
        )
        print0(
            f"[Summary] model  MAE={model_mae:.6f} "
            f"RMSE={model_rmse:.6f}"
        )
        print0(
            f"[Summary] beats_coarse_mae={int(model_mae < coarse_mae)} "
            f"beats_coarse_rmse={int(model_rmse < coarse_rmse)} "
            f"beats_coarse_both="
            f"{int(model_mae < coarse_mae and model_rmse < coarse_rmse)}"
        )
        print0(f"[Saved] per-date metrics: {output_csv_path}")
        print0(f"[Saved] summary metrics:  {summary_csv_path}")

    if is_dist_avail_and_initialized():
        dist.barrier()

    dataset_test.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
