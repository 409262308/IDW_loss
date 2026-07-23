#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference_Taiwan_Precip_Metrics_CSV.py

Precipitation inference + CSV metrics for Taiwan IDW diffusion model.

Outputs two CSV files without plotting:
  1. per-date metrics
  2. summary metrics

Metrics:
  coarse_vs_groundtruth: MAE, CRPS, MASE
  prediction_vs_groundtruth: MAE, CRPS, MASE

Notes:
  - Coarse CRPS is deterministic CRPS, so it equals MAE.
  - Model CRPS uses the empirical ensemble CRPS:
      mean_m |p_m - y| - 0.5 * mean_{m,j} |p_m - p_j|
  - MASE uses coarse MAE as the scaling baseline:
      model_mase = model_mae / coarse_mae
      coarse_mase = 1.0

Expected local files:
  Network.py
  DatasetTaiwanERA_IDW_tp.py

Example:
  python Inference_Taiwan_Precip_Metrics_CSV.py \
    --checkpoint "./outputs_taiwan_8km_idw_precip_loss_ddp/Model/best.pt" \
    --data-dir "/Users/wuyizhen/Desktop/IDW+loss/ERA_dataset" \
    --resolution 8km \
    --test-start 20180101 \
    --test-end 20201231 \
    --batch-size 4 \
    --ensemble-members 30 \
    --num-steps 40 \
    --output-csv "./precip_metrics_per_date.csv" \
    --summary-csv "./precip_metrics_summary.csv"
"""

from __future__ import annotations

import argparse
import csv
import os
import importlib.util
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Sampler

import Network


def _load_dataset_module():
    try:
        from DatasetTaiwanERA_IDW_tp import TaiwanERAPrecipDataset, load_stats
        return TaiwanERAPrecipDataset, load_stats
    except ModuleNotFoundError:
        here = Path(__file__).resolve().parent
        for candidate in [here / "DatasetTaiwanERA_IDW_tp.py", here / "DatasetTaiwanERA_IDW_tp(1).py"]:
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("DatasetTaiwanERA_IDW_tp_dynamic", candidate)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.TaiwanERAPrecipDataset, module.load_stats
        raise


TaiwanERAPrecipDataset, load_stats = _load_dataset_module()


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


def setup_distributed(args) -> torch.device:
    """Initialize torch.distributed when launched by torchrun."""
    args.distributed = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
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


def parse_csv_list(value: Optional[str], cast=str) -> List:
    if value is None or value == "":
        return []
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def parse_condition_vars(value):
    """Return condition variable names from a list/tuple or comma-separated string."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_int_list_config(value, default):
    """Return list[int] from list/tuple, comma-separated string, or default."""
    if value is None:
        value = default
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


class DistributedEvalSampler(Sampler):
    """Rank-strided evaluation sampler with no padded/duplicated examples."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.rank = get_rank()
        self.world_size = get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        n = len(self.dataset) - self.rank
        return max(0, (n + self.world_size - 1) // self.world_size)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return (
            strip_module_prefix(obj["model_state_dict"]),
            obj.get("config", {}),
            obj.get("stats", None),
            obj.get("epoch", None),
            obj.get("val_loss", None),
        )
    return strip_module_prefix(obj), {}, None, None, None


def infer_output_dir_from_checkpoint(checkpoint: Path) -> Optional[Path]:
    if checkpoint.parent.name == "Model":
        return checkpoint.parent.parent
    return None


def load_train_config(checkpoint: Path, train_config: Optional[str]) -> Dict:
    candidates = []
    if train_config:
        candidates.append(Path(train_config))
    out_dir = infer_output_dir_from_checkpoint(checkpoint)
    if out_dir is not None:
        candidates.append(out_dir / "train_config.json")
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def resolve_stats(args, checkpoint_path: Path, checkpoint_stats, train_config: Dict) -> Dict:
    if args.stats_path:
        return load_stats(args.stats_path)
    out_dir = infer_output_dir_from_checkpoint(checkpoint_path)
    if out_dir is not None:
        local_stats = out_dir / f"stats_{args.resolution}.json"
        if local_stats.exists():
            return load_stats(local_stats)
    cfg_stats = train_config.get("stats_path")
    if cfg_stats and Path(cfg_stats).exists():
        return load_stats(cfg_stats)
    if checkpoint_stats is not None:
        return checkpoint_stats
    raise FileNotFoundError(
        "Cannot find stats. Pass --stats-path, or keep stats_8km.json beside the training output directory."
    )


@torch.no_grad()
def sample_model_eds_batch(
    batch: Dict,
    model: torch.nn.Module,
    device: torch.device,
    dataset,
    num_steps: int = 40,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 40.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
) -> torch.Tensor:
    """Generate one stochastic precipitation prediction for a batch. Returns [B,1,H,W] on CPU."""
    model.eval()
    images_input = batch["inputs"].to(device)
    coarse = batch["coarse"]
    condition_params = torch.stack((batch["doy"].to(device), batch["hour"].to(device)), dim=1)

    sigma_min = max(float(sigma_min), float(model.sigma_min))
    sigma_max = min(float(sigma_max), float(model.sigma_max))

    target_channels = getattr(dataset, "target_channels", model.out_channels)
    height = images_input.shape[2]
    width = images_input.shape[3]
    dtype = torch.float32

    init_noise = torch.randn((images_input.shape[0], target_channels, height, width), dtype=dtype, device=device)
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps).to(device=device, dtype=dtype), torch.zeros_like(t_steps[:1])])

    x_next = init_noise * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= float(t_cur) <= S_max else 0.0
        t_hat = model.round_sigma(t_cur + gamma * t_cur).to(device=device, dtype=dtype)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        denoised = model(x_hat, t_hat, images_input, condition_params).to(dtype)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = model(x_next, t_next, images_input, condition_params).to(dtype)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    predicted = dataset.residual_to_fine_image(x_next.detach().cpu(), coarse)
    return predicted.clamp_min(0)


def masked_mean_by_sample(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and values.shape[0] != 1:
        mask = mask.expand(values.shape[0], -1, -1, -1)
    if mask.shape[1] == 1 and values.shape[1] != 1:
        mask = mask.expand(-1, values.shape[1], -1, -1)
    mask = mask.to(dtype=values.dtype, device=values.device)
    return (values * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(eps)


def crps_ensemble_map(preds: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """preds [M,B,C,H,W], truth [B,C,H,W]."""
    m = preds.shape[0]
    first = torch.mean(torch.abs(preds - truth.unsqueeze(0)), dim=0)
    if m == 1:
        return first
    # For sorted x, 0.5 * mean_{i,j}|x_i-x_j| equals
    # sum_i (2*i-m+1)*x_i / m^2.  This avoids M^2 full-image operations.
    sorted_preds = torch.sort(preds, dim=0).values
    coeff = (2 * torch.arange(m, dtype=preds.dtype, device=preds.device) - m + 1)
    coeff = coeff.view(m, *([1] * (preds.ndim - 1)))
    spread = (sorted_preds * coeff).sum(dim=0) / float(m * m)
    return first - spread


def apply_residual_scale(preds: torch.Tensor, coarse: torch.Tensor, scale: float) -> torch.Tensor:
    """Shrink/expand the generated physical-mm correction around the IDW baseline."""
    return (coarse.unsqueeze(0) + float(scale) * (preds - coarse.unsqueeze(0))).clamp_min(0)


@torch.no_grad()
def calibrate_residual_scale(
    model, dataset, dataloader, args, candidates: List[float], device
) -> Tuple[float, List[float], List[float]]:
    """Choose correction scale by MAE and CRPS on validation data.

    Scale zero is the exact coarse forecast, so including it makes the
    calibration result no worse than coarse on the calibration subset.
    """
    abs_sums = torch.zeros(len(candidates), dtype=torch.float64, device=device)
    crps_sums = torch.zeros(len(candidates), dtype=torch.float64, device=device)
    pixel_count = torch.zeros(1, dtype=torch.float64, device=device)

    for batch_index, batch in enumerate(tqdm(
        dataloader,
        desc=f"Calibrate rank {get_rank()}",
        dynamic_ncols=True,
        disable=not is_main_process(),
    )):
        fine = batch["fine"].float().cpu().clamp_min(0)
        coarse = batch["coarse"].float().cpu().clamp_min(0)
        mask = batch["valid_mask"].float().cpu()
        members = []
        for member in range(args.calibration_ensemble_members):
            set_seed(args.seed + 900000000 + get_rank() * 10000000 + batch_index * 100000 + member)
            members.append(sample_model_eds_batch(
                batch, model, device, dataset,
                num_steps=args.num_steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                rho=args.rho, S_churn=args.S_churn, S_min=args.S_min,
                S_max=args.S_max, S_noise=args.S_noise,
            ))
        raw_preds = torch.stack(members, dim=0)
        count = mask.sum().double()
        pixel_count += count.to(device)
        for i, scale in enumerate(candidates):
            calibrated_preds = apply_residual_scale(raw_preds, coarse, scale)
            calibrated_mean = calibrated_preds.mean(dim=0)
            abs_sums[i] += (torch.abs(calibrated_mean - fine) * mask).sum().double().to(device)
            crps = crps_ensemble_map(calibrated_preds, fine)
            crps_sums[i] += (crps * mask).sum().double().to(device)

    if is_dist_avail_and_initialized():
        dist.all_reduce(abs_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(crps_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(pixel_count, op=dist.ReduceOp.SUM)
    maes = (abs_sums / pixel_count.clamp_min(1.0)).cpu().tolist()
    crps_values = (crps_sums / pixel_count.clamp_min(1.0)).cpu().tolist()
    zero_index = candidates.index(0.0)
    tolerance = 1e-10
    eligible = [
        i for i in range(len(candidates))
        if maes[i] <= maes[zero_index] + tolerance and crps_values[i] <= crps_values[zero_index] + tolerance
    ]
    best_index = min(eligible, key=lambda i: (maes[i], crps_values[i], abs(candidates[i])))
    return (
        float(candidates[best_index]),
        [float(v) for v in maes],
        [float(v) for v in crps_values],
    )


def metric_batch(coarse: torch.Tensor, fine: torch.Tensor, preds: torch.Tensor, valid_mask: torch.Tensor):
    coarse = coarse.float().cpu().clamp_min(0)
    fine = fine.float().cpu().clamp_min(0)
    preds = preds.float().cpu().clamp_min(0)
    valid_mask = valid_mask.float().cpu()
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(0)
    if valid_mask.shape[0] == 1 and fine.shape[0] != 1:
        valid_mask = valid_mask.expand(fine.shape[0], -1, -1, -1)

    coarse_abs = torch.abs(coarse - fine)
    coarse_mae = masked_mean_by_sample(coarse_abs, valid_mask)
    coarse_crps = coarse_mae.clone()

    pred_mean = preds.mean(dim=0)
    model_abs = torch.abs(pred_mean - fine)
    model_mae = masked_mean_by_sample(model_abs, valid_mask)

    model_crps_map = crps_ensemble_map(preds, fine)
    model_crps = masked_mean_by_sample(model_crps_map, valid_mask)

    rows = []
    eps = 1e-12
    for b in range(fine.shape[0]):
        scale = float(coarse_mae[b].item())
        if scale > eps:
            coarse_mase = 1.0
            model_mase = float(model_mae[b].item()) / scale
            mae_skill = 1.0 - model_mase
        else:
            coarse_mase = float("nan")
            model_mase = float("nan")
            mae_skill = float("nan")
        rows.append({
            "coarse_mae": float(coarse_mae[b].item()),
            "coarse_crps": float(coarse_crps[b].item()),
            "coarse_mase": coarse_mase,
            "model_mae": float(model_mae[b].item()),
            "model_crps": float(model_crps[b].item()),
            "model_mase": model_mase,
            "mae_improvement_mm": float(coarse_mae[b].item() - model_mae[b].item()),
            "mae_skill": mae_skill,
            "n_valid_pixels": int(valid_mask[b].sum().item()),
        })

    mask = valid_mask
    totals = {
        "n_pixels": float(mask.sum().item()),
        "coarse_abs_sum": float((coarse_abs * mask).sum().item()),
        "coarse_crps_sum": float((coarse_abs * mask).sum().item()),
        "model_abs_sum": float((model_abs * mask).sum().item()),
        "model_crps_sum": float((model_crps_map * mask).sum().item()),
    }
    return rows, totals


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Taiwan precipitation inference: write MAE/CRPS/MASE to CSV.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or best_full.pt.")
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--stats-path", type=str, default=None)

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--resolution", type=str, default="8km", choices=["1km", "5km", "8km"])
    parser.add_argument("--test-start", type=str, default="20180101")
    parser.add_argument("--test-end", type=str, default="20201231")
    parser.add_argument("--condition-vars", type=str, default=None)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--target-transform", type=str, default=None, choices=["log1p", "raw"])
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--tp-idw-k", type=int, default=None)
    parser.add_argument("--tp-idw-power", type=float, default=None)

    parser.add_argument("--model-channels", type=int, default=None)
    parser.add_argument("--channel-mult", type=str, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--attn-resolutions", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--sigma-data", type=float, default=None)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
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
    parser.add_argument("--residual-scale", type=str, default="auto",
                        help="Physical-mm correction scale, or 'auto' to tune it on validation data.")
    parser.add_argument("--residual-scale-candidates", type=str,
                        default="0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--calibration-start", type=str, default=None)
    parser.add_argument("--calibration-end", type=str, default=None)
    parser.add_argument("--calibration-max-samples", type=int, default=256)
    parser.add_argument("--calibration-ensemble-members", type=int, default=4)

    parser.add_argument("--output-csv", type=str, default="./precip_metrics_per_date.csv")
    parser.add_argument("--summary-csv", type=str, default="./precip_metrics_summary.csv")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = setup_distributed(args)
    set_seed(args.seed + get_rank())
    state_dict, ckpt_config, ckpt_stats, epoch, val_loss = load_checkpoint(checkpoint_path, device)
    train_config = load_train_config(checkpoint_path, args.train_config)
    merged_config = dict(ckpt_config)
    merged_config.update(train_config)

    data_dir = args.data_dir or merged_config.get("data_dir")
    if data_dir is None:
        raise ValueError("Please pass --data-dir, or provide train_config.json containing data_dir.")

    if args.condition_vars is not None:
        raw_condition_vars = args.condition_vars
    else:
        raw_condition_vars = merged_config.get(
            "condition_vars",
            merged_config.get("input_channel_names", ["q700", "t2m", "u", "v", "msl", "tp"]),
        )

    condition_vars = parse_condition_vars(raw_condition_vars)
    condition_vars = [v for v in condition_vars if v != "mask"]

    no_mask = bool(args.no_mask)
    if not args.no_mask and "no_mask" in merged_config:
        no_mask = bool(merged_config.get("no_mask"))

    target_transform = args.target_transform or merged_config.get("target_transform", "log1p")
    tp_idw_k = args.tp_idw_k if args.tp_idw_k is not None else int(merged_config.get("tp_idw_k", 8))
    tp_idw_power = args.tp_idw_power if args.tp_idw_power is not None else float(merged_config.get("tp_idw_power", 2.0))
    stats = resolve_stats(args, checkpoint_path, ckpt_stats, merged_config)

    dataset_test = TaiwanERAPrecipDataset(
        data_dir=data_dir,
        resolution=args.resolution,
        start_date=args.test_start,
        end_date=args.test_end,
        condition_vars=condition_vars,
        use_mask=not no_mask,
        target_transform=target_transform,
        stats=stats,
        max_samples=args.max_test_samples,
        tp_idw_k=tp_idw_k,
        tp_idw_power=tp_idw_power,
    )

    sampler = DistributedEvalSampler(dataset_test) if getattr(args, "distributed", False) else None

    dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_channels = args.model_channels if args.model_channels is not None else int(merged_config.get("model_channels", 64))
    channel_mult = parse_int_list_config(
        args.channel_mult if args.channel_mult is not None else merged_config.get("channel_mult", [1, 2, 3, 4]),
        [1, 2, 3, 4],
    )
    num_blocks = args.num_blocks if args.num_blocks is not None else int(merged_config.get("num_blocks", 2))
    attn_resolutions = parse_int_list_config(
        args.attn_resolutions if args.attn_resolutions is not None else merged_config.get("attn_resolutions", [32, 16, 8]),
        [32, 16, 8],
    )
    dropout = args.dropout if args.dropout is not None else float(merged_config.get("dropout", 0.10))
    sigma_data = args.sigma_data if args.sigma_data is not None else float(merged_config.get("sigma_data", 1.0))
    use_fp16 = bool(args.use_fp16 or merged_config.get("use_fp16", False))

    model_in_channels = dataset_test.num_input_channels + dataset_test.target_channels
    model_out_channels = dataset_test.target_channels
    model = Network.EDMPrecond(
        dataset_test.img_resolution,
        model_in_channels,
        model_out_channels,
        label_dim=2,
        use_fp16=use_fp16,
        sigma_data=sigma_data,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=num_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if args.ensemble_members < 1 or args.num_steps < 2:
        raise ValueError("--ensemble-members must be >= 1 and --num-steps must be >= 2.")

    if args.residual_scale.lower() == "auto":
        calibration_start = args.calibration_start or merged_config.get("val_start")
        calibration_end = args.calibration_end or merged_config.get("val_end")
        if not calibration_start or not calibration_end:
            raise ValueError(
                "--residual-scale auto needs --calibration-start/--calibration-end, "
                "or val_start/val_end in train_config.json."
            )
        if args.calibration_ensemble_members < 1:
            raise ValueError("--calibration-ensemble-members must be >= 1.")
        candidates = sorted(set(parse_csv_list(args.residual_scale_candidates, float) + [0.0]))
        if any(scale < 0 for scale in candidates):
            raise ValueError("--residual-scale-candidates cannot contain negative values.")
        if not candidates:
            raise ValueError("--residual-scale-candidates must contain at least one number.")
        calibration_dataset = TaiwanERAPrecipDataset(
            data_dir=data_dir,
            resolution=args.resolution,
            start_date=calibration_start,
            end_date=calibration_end,
            condition_vars=condition_vars,
            use_mask=not no_mask,
            target_transform=target_transform,
            stats=stats,
            max_samples=args.calibration_max_samples,
            tp_idw_k=tp_idw_k,
            tp_idw_power=tp_idw_power,
        )
        calibration_sampler = DistributedEvalSampler(calibration_dataset) if getattr(args, "distributed", False) else None
        calibration_loader = torch.utils.data.DataLoader(
            calibration_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=calibration_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        residual_scale, calibration_maes, calibration_crps = calibrate_residual_scale(
            model, calibration_dataset, calibration_loader, args, candidates, device
        )
        if is_main_process():
            print0("[Calibration] scale -> MAE/CRPS: " + ", ".join(
                f"{scale:g}->{mae:.6f}/{crps:.6f}"
                for scale, mae, crps in zip(candidates, calibration_maes, calibration_crps)
            ))
        calibration_dataset.close()
    else:
        residual_scale = float(args.residual_scale)
        if residual_scale < 0:
            raise ValueError("--residual-scale must be non-negative.")

    print0(f"[Device] {device}")
    print0(f"[Checkpoint] {checkpoint_path}")
    if epoch is not None:
        print0(f"[Checkpoint] epoch={epoch} val_loss={val_loss}")
    print0(f"[Dataset] test={len(dataset_test)} resolution={args.resolution}")
    print0(f"[Dataset] condition_vars={condition_vars}, use_mask={not no_mask}")
    print0(f"[Model] in_channels={model_in_channels}, out_channels={model_out_channels}")
    print0(f"[Inference] distributed={getattr(args, 'distributed', False)} world_size={get_world_size()}")
    print0(f"[Inference] ensemble_members={args.ensemble_members}, num_steps={args.num_steps}")
    print0(f"[Inference] residual_scale={residual_scale:g}")

    per_date_rows = []
    total_pixels = 0.0
    total_coarse_abs = 0.0
    total_coarse_crps = 0.0
    total_model_abs = 0.0
    total_model_crps = 0.0

    for batch_index, batch in enumerate(tqdm(dataloader, desc=f"Inference rank {get_rank()}", dynamic_ncols=True, disable=not is_main_process())):
        fine = batch["fine"].float().cpu().clamp_min(0)
        coarse = batch["coarse"].float().cpu().clamp_min(0)
        valid_mask = batch["valid_mask"].float().cpu()

        preds_members = []
        for member in range(args.ensemble_members):
            set_seed(args.seed + get_rank() * 10000000 + batch_index * 100000 + member)
            pred = sample_model_eds_batch(
                batch=batch,
                model=model,
                device=device,
                dataset=dataset_test,
                num_steps=args.num_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                S_churn=args.S_churn,
                S_min=args.S_min,
                S_max=args.S_max,
                S_noise=args.S_noise,
            )
            preds_members.append(pred)
        preds = torch.stack(preds_members, dim=0)
        preds = apply_residual_scale(preds, coarse, residual_scale)

        rows, totals = metric_batch(coarse, fine, preds, valid_mask)
        dates = batch["date"]
        if isinstance(dates, str):
            dates = [dates]

        for row, date in zip(rows, dates):
            per_date_rows.append({
                "date": str(date),
                "n_valid_pixels": row["n_valid_pixels"],
                "coarse_mae": row["coarse_mae"],
                "coarse_crps": row["coarse_crps"],
                "coarse_mase": row["coarse_mase"],
                "model_mae": row["model_mae"],
                "model_crps": row["model_crps"],
                "model_mase": row["model_mase"],
                "mae_improvement_mm": row["mae_improvement_mm"],
                "mae_skill": row["mae_skill"],
                "residual_scale": residual_scale,
                "ensemble_members": args.ensemble_members,
                "num_steps": args.num_steps,
            })

        total_pixels += totals["n_pixels"]
        total_coarse_abs += totals["coarse_abs_sum"]
        total_coarse_crps += totals["coarse_crps_sum"]
        total_model_abs += totals["model_abs_sum"]
        total_model_crps += totals["model_crps_sum"]

    output_csv_path = Path(args.output_csv)
    summary_csv_path = Path(args.summary_csv)
    part_dir = output_csv_path.parent / "_ddp_inference_parts"

    if is_main_process():
        part_dir.mkdir(parents=True, exist_ok=True)
    if is_dist_avail_and_initialized():
        dist.barrier()

    part_csv = part_dir / f"per_date_rank{get_rank()}.csv"
    part_json = part_dir / f"totals_rank{get_rank()}.json"

    write_csv(part_csv, per_date_rows, [
        "date", "n_valid_pixels", "coarse_mae", "coarse_crps", "coarse_mase",
        "model_mae", "model_crps", "model_mase", "mae_improvement_mm", "mae_skill",
        "residual_scale", "ensemble_members", "num_steps",
    ])
    part_json.write_text(json.dumps({
        "n_pixels": total_pixels,
        "coarse_abs_sum": total_coarse_abs,
        "coarse_crps_sum": total_coarse_crps,
        "model_abs_sum": total_model_abs,
        "model_crps_sum": total_model_crps,
    }, indent=2), encoding="utf-8")

    if is_dist_avail_and_initialized():
        dist.barrier()

    if is_main_process():
        merged_rows = []
        total_pixels_all = 0.0
        total_coarse_abs_all = 0.0
        total_coarse_crps_all = 0.0
        total_model_abs_all = 0.0
        total_model_crps_all = 0.0

        for rank in range(get_world_size()):
            rank_csv = part_dir / f"per_date_rank{rank}.csv"
            rank_json = part_dir / f"totals_rank{rank}.json"

            if rank_csv.exists():
                with rank_csv.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    merged_rows.extend(list(reader))

            totals = json.loads(rank_json.read_text(encoding="utf-8"))
            total_pixels_all += float(totals["n_pixels"])
            total_coarse_abs_all += float(totals["coarse_abs_sum"])
            total_coarse_crps_all += float(totals["coarse_crps_sum"])
            total_model_abs_all += float(totals["model_abs_sum"])
            total_model_crps_all += float(totals["model_crps_sum"])

        merged_rows.sort(key=lambda row: row["date"])

        eps = 1e-12
        coarse_mae_global = total_coarse_abs_all / max(total_pixels_all, eps)
        coarse_crps_global = total_coarse_crps_all / max(total_pixels_all, eps)
        model_mae_global = total_model_abs_all / max(total_pixels_all, eps)
        model_crps_global = total_model_crps_all / max(total_pixels_all, eps)
        coarse_mase_global = 1.0 if coarse_mae_global > eps else float("nan")
        model_mase_global = model_mae_global / coarse_mae_global if coarse_mae_global > eps else float("nan")
        model_mae_skill_global = 1.0 - model_mase_global if coarse_mae_global > eps else float("nan")

        summary_rows = [
            {
                "comparison": "coarse_vs_groundtruth",
                "mae": coarse_mae_global,
                "crps": coarse_crps_global,
                "mase": coarse_mase_global,
                "scale_mae": coarse_mae_global,
                "mae_improvement_mm": 0.0,
                "mae_skill": 0.0,
                "residual_scale": 0.0,
                "n_dates": len(merged_rows),
                "n_valid_pixels_total": int(total_pixels_all),
                "ensemble_members": 1,
                "num_steps": 0,
            },
            {
                "comparison": "prediction_vs_groundtruth",
                "mae": model_mae_global,
                "crps": model_crps_global,
                "mase": model_mase_global,
                "scale_mae": coarse_mae_global,
                "mae_improvement_mm": coarse_mae_global - model_mae_global,
                "mae_skill": model_mae_skill_global,
                "residual_scale": residual_scale,
                "n_dates": len(merged_rows),
                "n_valid_pixels_total": int(total_pixels_all),
                "ensemble_members": args.ensemble_members,
                "num_steps": args.num_steps,
            },
        ]

        write_csv(output_csv_path, merged_rows, [
            "date", "n_valid_pixels", "coarse_mae", "coarse_crps", "coarse_mase",
            "model_mae", "model_crps", "model_mase", "mae_improvement_mm", "mae_skill",
            "residual_scale", "ensemble_members", "num_steps",
        ])
        write_csv(summary_csv_path, summary_rows, [
            "comparison", "mae", "crps", "mase", "scale_mae", "n_dates",
            "mae_improvement_mm", "mae_skill", "residual_scale",
            "n_valid_pixels_total", "ensemble_members", "num_steps",
        ])

        print0(f"[Saved] per-date metrics: {args.output_csv}")
        print0(f"[Saved] summary metrics:  {args.summary_csv}")

    if is_dist_avail_and_initialized():
        dist.barrier()
    cleanup_distributed()



if __name__ == "__main__":
    main()
