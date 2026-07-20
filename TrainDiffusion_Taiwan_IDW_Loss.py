#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainDiffusion_Taiwan_IDW_Loss.py

Taiwan precipitation diffusion training script.

This script keeps Network.py unchanged and adapts TrainDiffusion.py to:

1. Use DatasetTaiwanERA_IDW_tp.TaiwanERAPrecipDataset.
2. Accept all important TaiwanERAPrecipDataset parameters from CLI.
3. Set EDMPrecond channel counts automatically from the dataset:
       model input channels = dataset.target_channels + dataset.num_input_channels
       model output channels = dataset.target_channels
4. Use a precipitation-aware EDM loss:
       EDM denoising loss
       + optional log-intensity weighting
       + optional heavy-rain threshold weighting
       + valid-mask support for padded grids
5. Save checkpoints and sample figures under --output-dir.

Expected files in the same folder:
    Network.py
    DatasetTaiwanERA_IDW_tp.py

Example:
    python3 TrainDiffusion_Taiwan_IDW_Loss.py \
        --data-dir "/Users/wuyizhen/Desktop/IDW+loss/ERA_dataset" \
        --resolution 8km \
        --train-start 19600101 \
        --train-end 20141125 \
        --val-start 20141126 \
        --val-end 20171213 \
        --epochs 100 \
        --batch-size 8 \
        --lr 1e-4 \
        --accum 8 \
        --tp-idw-k 8 \
        --tp-idw-power 2.0 \
        --precip-intensity-alpha 0.5 \
        --heavy-rain-threshold 10.0 \
        --heavy-rain-weight 2.0 \
        --output-dir "./outputs_taiwan_idw_precip"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import Network


def _load_dataset_module():
    """Import DatasetTaiwanERA_IDW_tp, with a fallback for copied filenames."""
    try:
        from DatasetTaiwanERA_IDW_tp import (  # type: ignore
            TaiwanERAPrecipDataset,
            compute_stats,
            save_stats,
            load_stats,
        )
        return TaiwanERAPrecipDataset, compute_stats, save_stats, load_stats
    except ModuleNotFoundError:
        here = Path(__file__).resolve().parent
        candidates = [
            here / "DatasetTaiwanERA_IDW_tp.py",
            here / "DatasetTaiwanERA_IDW_tp(1).py",
        ]
        for candidate in candidates:
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("DatasetTaiwanERA_IDW_tp_dynamic", candidate)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return (
                    module.TaiwanERAPrecipDataset,
                    module.compute_stats,
                    module.save_stats,
                    module.load_stats,
                )
        raise


TaiwanERAPrecipDataset, compute_stats, save_stats, load_stats = _load_dataset_module()


class PrecipitationEDMLoss:
    """EDM denoising loss with precipitation-specific weighting.

    Base EDM target:
        y = clean normalized log-residual target
        x_sigma = y + sigma * epsilon
        D_theta(x_sigma, sigma, condition) ≈ y

    Base EDM loss:
        lambda(sigma) * (D_theta - y)^2

    Additional precipitation weighting:
        fine is raw precipitation in mm or original unit from the dataset.
        precip_weight = 1
        precip_weight += precip_intensity_alpha * log1p(fine)
        precip_weight += heavy_rain_weight * I(fine >= heavy_rain_threshold)

    The precipitation weight is optionally normalized by its valid-grid mean so
    that turning the weighting on does not massively change the overall loss scale.
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 1.0,
        precip_intensity_alpha: float = 0.5,
        heavy_rain_threshold: float = 10.0,
        heavy_rain_weight: float = 2.0,
        normalize_precip_weight: bool = True,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.precip_intensity_alpha = precip_intensity_alpha
        self.heavy_rain_threshold = heavy_rain_threshold
        self.heavy_rain_weight = heavy_rain_weight
        self.normalize_precip_weight = normalize_precip_weight

    def _make_precip_weight(
        self,
        fine: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        loss_like: torch.Tensor,
    ) -> torch.Tensor:
        if fine is None:
            return torch.ones_like(loss_like)

        # fine shape is usually [B, 1, H, W].
        fine = fine.to(dtype=loss_like.dtype, device=loss_like.device)
        fine = torch.clamp(fine, min=0)

        precip_weight = torch.ones_like(fine)

        if self.precip_intensity_alpha > 0:
            precip_weight = precip_weight + self.precip_intensity_alpha * torch.log1p(fine)

        if self.heavy_rain_weight > 0 and self.heavy_rain_threshold > 0:
            precip_weight = precip_weight + self.heavy_rain_weight * (
                fine >= self.heavy_rain_threshold
            ).to(loss_like.dtype)

        if valid_mask is not None:
            mask = valid_mask.to(dtype=loss_like.dtype, device=loss_like.device)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        else:
            mask = torch.ones_like(precip_weight)

        if self.normalize_precip_weight:
            denom = mask.sum().clamp_min(1.0)
            mean_weight = (precip_weight * mask).sum() / denom
            precip_weight = precip_weight / mean_weight.clamp_min(1e-8)

        # Expand from [B,1,H,W] to [B,C,H,W] if target has more than one channel.
        if precip_weight.shape[1] == 1 and loss_like.shape[1] != 1:
            precip_weight = precip_weight.expand(-1, loss_like.shape[1], -1, -1)

        return precip_weight

    def __call__(
        self,
        net: torch.nn.Module,
        images: torch.Tensor,
        conditional_img: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        fine: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        augment_pipe=None,
    ) -> torch.Tensor:
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        edm_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        noise = torch.randn_like(y) * sigma

        denoised = net(
            y + noise,
            sigma,
            conditional_img,
            labels,
            augment_labels=augment_labels,
        )

        loss = edm_weight * ((denoised - y) ** 2)

        precip_weight = self._make_precip_weight(fine, valid_mask, loss)
        loss = loss * precip_weight

        if valid_mask is not None:
            mask = valid_mask.to(dtype=loss.dtype, device=loss.device)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[1] == 1 and loss.shape[1] != 1:
                mask = mask.expand(-1, loss.shape[1], -1, -1)
            loss = loss * mask
            denom = mask.sum().clamp_min(1.0)
            return loss.sum() / denom

        return loss.mean()


def parse_csv_list(value: str, cast=str) -> List:
    if value is None or value == "":
        return []
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    # ZipFile handles inside the dataset are reopened per process through __getstate__.
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def training_step(
    model: torch.nn.Module,
    loss_fn: PrecipitationEDMLoss,
    optimiser: torch.optim.Optimizer,
    data_loader,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    accum: int = 4,
    writer: Optional[SummaryWriter] = None,
    device: torch.device = torch.device("cuda"),
) -> float:
    model.train()
    optimiser.zero_grad(set_to_none=True)

    amp_enabled = device.type == "cuda"
    epoch_losses: List[float] = []
    running_loss = 0.0

    with tqdm(total=len(data_loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch {epoch}")

        for i, batch in enumerate(data_loader):
            tq.update(1)

            image_input = batch["inputs"].to(device, non_blocking=True)
            image_output = batch["targets"].to(device, non_blocking=True)
            fine = batch["fine"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            day = batch["doy"].to(device, non_blocking=True)
            hour = batch["hour"].to(device, non_blocking=True)
            condition_params = torch.stack((day, hour), dim=1)

            with autocast_context(device):
                loss = loss_fn(
                    net=model,
                    images=image_output,
                    conditional_img=image_input,
                    labels=condition_params,
                    fine=fine,
                    valid_mask=valid_mask,
                )
                # Scale loss for gradient accumulation so the effective gradient is
                # approximately the average over accum mini-batches.
                loss_for_backward = loss / accum

            if amp_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            running_loss += loss.item()
            epoch_losses.append(loss.item())

            do_step = ((i + 1) % accum == 0) or ((i + 1) == len(data_loader))
            if do_step:
                if amp_enabled:
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                optimiser.zero_grad(set_to_none=True)

                global_step = epoch * len(data_loader) + i
                if writer is not None:
                    writer.add_scalar("Loss/train_step", running_loss / min(accum, i + 1), global_step)
                running_loss = 0.0

            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

    mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
    return mean_loss


@torch.no_grad()
def validation_step(
    model: torch.nn.Module,
    loss_fn: PrecipitationEDMLoss,
    data_loader,
    epoch: int,
    device: torch.device,
) -> float:
    model.eval()
    losses: List[float] = []

    with tqdm(total=len(data_loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Val   :: Epoch {epoch}")
        for batch in data_loader:
            tq.update(1)

            image_input = batch["inputs"].to(device, non_blocking=True)
            image_output = batch["targets"].to(device, non_blocking=True)
            fine = batch["fine"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            day = batch["doy"].to(device, non_blocking=True)
            hour = batch["hour"].to(device, non_blocking=True)
            condition_params = torch.stack((day, hour), dim=1)

            with autocast_context(device):
                loss = loss_fn(
                    net=model,
                    images=image_output,
                    conditional_img=image_input,
                    labels=condition_params,
                    fine=fine,
                    valid_mask=valid_mask,
                )

            losses.append(loss.item())
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def sample_model(
    model: torch.nn.Module,
    dataloader,
    num_steps: int = 40,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 7,
    S_churn: float = 40,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
    device: torch.device = torch.device("cuda"),
):
    model.eval()
    batch = next(iter(dataloader))

    images_input = batch["inputs"].to(device)
    coarse = batch["coarse"]
    fine = batch["fine"]

    condition_params = torch.stack(
        (batch["doy"].to(device), batch["hour"].to(device)),
        dim=1,
    )

    sigma_min = max(float(sigma_min), float(model.sigma_min))
    sigma_max = min(float(sigma_max), float(model.sigma_max))

    target_channels = getattr(dataloader.dataset, "target_channels", model.out_channels)
    height = images_input.shape[2]
    width = images_input.shape[3]

    dtype = torch.float32
    init_noise = torch.randn(
        (images_input.shape[0], target_channels, height, width),
        dtype=dtype,
        device=device,
    )

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

    predicted = dataloader.dataset.residual_to_fine_image(x_next.detach().cpu(), coarse)
    fig, ax = dataloader.dataset.plot_batch(coarse, fine, predicted)

    valid_mask = batch["valid_mask"]
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)

    base_error = torch.sum(torch.abs(fine - coarse) * valid_mask) / valid_mask.sum().clamp_min(1.0)
    pred_error = torch.sum(torch.abs(fine - predicted) * valid_mask) / valid_mask.sum().clamp_min(1.0)

    return (fig, ax), (base_error.item(), pred_error.item())


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Taiwan precipitation EDM diffusion with IDW tp and precipitation-aware loss.")

    # Dataset parameters: TaiwanERAPrecipDataset.
    parser.add_argument("--data-dir", type=str, default="/Users/wuyizhen/Desktop/IDW+loss/ERA_dataset")
    parser.add_argument("--resolution", type=str, default="8km", choices=["1km", "5km", "8km"])
    parser.add_argument("--train-start", type=str, default="19600101")
    parser.add_argument("--train-end", type=str, default="20141125")
    parser.add_argument("--val-start", type=str, default="20141126")
    parser.add_argument("--val-end", type=str, default="20171213")
    parser.add_argument("--condition-vars", type=str, default="q700,t2m,u,v,msl,tp",
                        help="Comma-separated variables from q700,t2m,u,v,msl,tp.")
    parser.add_argument("--no-mask", action="store_true", help="Disable static land-sea mask input channel.")
    parser.add_argument("--target-transform", type=str, default="log1p", choices=["log1p", "raw"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--tp-idw-k", type=int, default=8)
    parser.add_argument("--tp-idw-power", type=float, default=2.0)

    # Stats.
    parser.add_argument("--stats-path", type=str, default=None,
                        help="Path to JSON stats. Default: <output-dir>/stats_<resolution>.json")
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--stats-samples", type=int, default=2048)

    # Training parameters.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--output-dir", type=str, default="./outputs_taiwan_idw_precip")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=40)

    # Network parameters. Network.py is unchanged; these are only constructor args.
    parser.add_argument("--model-channels", type=int, default=128)
    parser.add_argument("--channel-mult", type=str, default="1,2,3,4")
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--attn-resolutions", type=str, default="32,16,8")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--use-fp16", action="store_true")

    # EDM loss parameters.
    parser.add_argument("--P-mean", type=float, default=-1.2)
    parser.add_argument("--P-std", type=float, default=1.2)
    parser.add_argument("--sigma-data", type=float, default=1.0)
    parser.add_argument("--precip-intensity-alpha", type=float, default=0.5,
                        help="Continuous log1p(fine) precipitation loss weight. Set 0 to disable.")
    parser.add_argument("--heavy-rain-threshold", type=float, default=10.0,
                        help="Raw fine precipitation threshold for extra weighting. Set <=0 to disable.")
    parser.add_argument("--heavy-rain-weight", type=float, default=2.0,
                        help="Additional weight for fine precipitation >= threshold. Set 0 to disable.")
    parser.add_argument("--no-normalize-precip-weight", action="store_true")

    # EDM sampling parameters for monitoring images.
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--S-churn", type=float, default=40.0)
    parser.add_argument("--S-min", type=float, default=0.0)
    parser.add_argument("--S-max", type=float, default=float("inf"))
    parser.add_argument("--S-noise", type=float, default=1.0)

    return parser


def main():
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "Model"
    result_dir = output_dir / "results"
    run_dir = output_dir / "runs"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.stats_path is None:
        args.stats_path = str(output_dir / f"stats_{args.resolution}.json")

    condition_vars = parse_csv_list(args.condition_vars, str)
    channel_mult = parse_csv_list(args.channel_mult, int)
    attn_resolutions = parse_csv_list(args.attn_resolutions, int)

    # Compute or load statistics.
    stats_path = Path(args.stats_path)
    stats = None
    if args.recompute_stats or not stats_path.exists():
        print(f"[Stats] Computing stats and saving to {stats_path}")
        stats_dataset = TaiwanERAPrecipDataset(
            data_dir=args.data_dir,
            resolution=args.resolution,
            start_date=args.train_start,
            end_date=args.train_end,
            condition_vars=condition_vars,
            use_mask=not args.no_mask,
            target_transform=args.target_transform,
            stats=None,
            max_samples=None,
            tp_idw_k=args.tp_idw_k,
            tp_idw_power=args.tp_idw_power,
        )
        stats = compute_stats(stats_dataset, max_samples=args.stats_samples)
        save_stats(stats, stats_path)
        stats_dataset.close()
    else:
        print(f"[Stats] Loading stats from {stats_path}")
        stats = load_stats(stats_path)

    dataset_train = TaiwanERAPrecipDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        start_date=args.train_start,
        end_date=args.train_end,
        condition_vars=condition_vars,
        use_mask=not args.no_mask,
        target_transform=args.target_transform,
        stats=stats,
        max_samples=args.max_train_samples,
        tp_idw_k=args.tp_idw_k,
        tp_idw_power=args.tp_idw_power,
    )

    dataset_val = TaiwanERAPrecipDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        start_date=args.val_start,
        end_date=args.val_end,
        condition_vars=condition_vars,
        use_mask=not args.no_mask,
        target_transform=args.target_transform,
        stats=stats,
        max_samples=args.max_val_samples,
        tp_idw_k=args.tp_idw_k,
        tp_idw_power=args.tp_idw_power,
    )

    dataloader_train = make_dataloader(dataset_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_val = make_dataloader(dataset_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = choose_device(args.device)
    print(f"[Device] {device}")
    print(f"[Dataset] train={len(dataset_train)} val={len(dataset_val)}")
    print(f"[Dataset] input channels={dataset_train.num_input_channels} names={dataset_train.input_channel_names}")
    print(f"[Dataset] target channels={dataset_train.target_channels}")
    print(f"[Dataset] img_resolution={dataset_train.img_resolution}")

    model_in_channels = dataset_train.num_input_channels + dataset_train.target_channels
    model_out_channels = dataset_train.target_channels

    network = Network.EDMPrecond(
        dataset_train.img_resolution,
        model_in_channels,
        model_out_channels,
        label_dim=2,
        use_fp16=args.use_fp16,
        sigma_data=args.sigma_data,
        model_channels=args.model_channels,
        channel_mult=channel_mult,
        num_blocks=args.num_blocks,
        attn_resolutions=attn_resolutions,
        dropout=args.dropout,
    )
    network.to(device)

    optimiser = torch.optim.AdamW(network.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(str(run_dir))

    loss_fn = PrecipitationEDMLoss(
        P_mean=args.P_mean,
        P_std=args.P_std,
        sigma_data=args.sigma_data,
        precip_intensity_alpha=args.precip_intensity_alpha,
        heavy_rain_threshold=args.heavy_rain_threshold,
        heavy_rain_weight=args.heavy_rain_weight,
        normalize_precip_weight=not args.no_normalize_precip_weight,
    )

    config = vars(args).copy()
    config.update({
        "input_channel_names": dataset_train.input_channel_names,
        "num_input_channels": dataset_train.num_input_channels,
        "target_channels": dataset_train.target_channels,
        "model_in_channels": model_in_channels,
        "model_out_channels": model_out_channels,
        "img_resolution": dataset_train.img_resolution,
        "stats_path": str(stats_path),
    })
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val_loss = float("inf")
    losses = []

    for epoch in range(args.epochs):
        train_loss = training_step(
            network,
            loss_fn,
            optimiser,
            dataloader_train,
            scaler,
            epoch,
            accum=args.accum,
            writer=writer,
            device=device,
        )
        val_loss = validation_step(network, loss_fn, dataloader_val, epoch, device=device)

        losses.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)

        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if args.sample_every > 0 and epoch % args.sample_every == 0:
            try:
                (fig, ax), (base_error, pred_error) = sample_model(
                    network,
                    dataloader_val,
                    num_steps=args.sample_steps,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    rho=args.rho,
                    S_churn=args.S_churn,
                    S_min=args.S_min,
                    S_max=args.S_max,
                    S_noise=args.S_noise,
                    device=device,
                )
                fig.savefig(result_dir / f"{epoch:04d}.png", dpi=200)
                plt.close(fig)
                writer.add_scalar("Error/base", base_error, epoch)
                writer.add_scalar("Error/pred", pred_error, epoch)
                print(f"[Sample] base_error={base_error:.6f} pred_error={pred_error:.6f}")
            except Exception as exc:
                print(f"[Warning] sample_model failed at epoch {epoch}: {exc}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(network.state_dict(), model_dir / "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": network.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": config,
                    "stats": stats,
                },
                model_dir / "best_full.pt",
            )
            print(f"[Checkpoint] Saved best.pt with val_loss={best_val_loss:.6f}")

        (output_dir / "losses.json").write_text(json.dumps(losses, indent=2), encoding="utf-8")

    dataset_train.close()
    dataset_val.close()
    writer.close()


if __name__ == "__main__":
    main()
