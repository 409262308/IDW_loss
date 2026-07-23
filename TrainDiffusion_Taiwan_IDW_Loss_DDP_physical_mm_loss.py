#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainDiffusion_Taiwan_IDW_Loss_DDP.py

DDP version for 4-GPU training.

Run example:
    torchrun --standalone --nproc_per_node=4 TrainDiffusion_Taiwan_IDW_Loss_DDP.py \
      --data-dir "/path/to/ERA_dataset" \
      --resolution 8km \
      --train-start 19600101 \
      --train-end 20141125 \
      --val-start 20141126 \
      --val-end 20171213 \
      --epochs 100 \
      --batch-size 16 \
      --lr 1e-4 \
      --accum 8 \
      --num-workers 2 \
      --model-channels 64 \
      --tp-idw-k 8 \
      --tp-idw-power 2.0 \
      --precip-intensity-alpha 0.5 \
      --heavy-rain-threshold 10.0 \
      --heavy-rain-weight 2.0 \
      --sample-every 5 \
      --sample-steps 20 \
      --output-dir "./outputs_taiwan_8km_idw_precip_loss_ddp"

Important:
    --batch-size is PER GPU batch size.
    With --nproc_per_node=4 and --batch-size 16:
        effective raw batch per optimizer step = 16 * 4 = 64
        if --accum 8:
        effective accumulated batch = 16 * 4 * 8 = 512
"""

# + language="bash"
# cd /home/wuyizhen61/IDW_loss
#
# export PYTHONNOUSERSITE=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO
#
# python -m torch.distributed.run --standalone --nproc_per_node=4 TrainDiffusion_Taiwan_IDW_Loss_DDP.py \
#   --data-dir "/home/wuyizhen61/IDW_loss/ERA_dataset" \
#   --resolution 8km \
#   --train-start 19600101 \
#   --train-end 20141125 \
#   --val-start 20141126 \
#   --val-end 20171213 \
#   --epochs 100 \
#   --batch-size 16 \
#   --lr 1e-4 \
#   --accum 8 \
#   --num-workers 2 \
#   --model-channels 64 \
#   --tp-idw-k 8 \
#   --tp-idw-power 2.0 \
# --precip-intensity-alpha 0.5 \
# --heavy-rain-threshold 10.0 \
# --heavy-rain-weight 2.0 \
# --sample-every 5 \
# --sample-steps 20 \
# --output-dir "./outputs_taiwan_8km_idw_precip_loss_ddp"
#
# -

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import Network


def _load_dataset_module():
    try:
        from DatasetTaiwanERA_IDW_tp import (
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
                module = importlib.util.module_from_spec(spec)
                assert spec is not None and spec.loader is not None
                spec.loader.exec_module(module)
                return (
                    module.TaiwanERAPrecipDataset,
                    module.compute_stats,
                    module.save_stats,
                    module.load_stats,
                )
        raise


TaiwanERAPrecipDataset, compute_stats, save_stats, load_stats = _load_dataset_module()


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
        return torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

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


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        try:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        finally:
            dist.destroy_process_group()


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


class PrecipitationEDMLoss:
    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 1.0,
        precip_intensity_alpha: float = 0.5,
        heavy_rain_threshold: float = 10.0,
        heavy_rain_weight: float = 2.0,
        normalize_precip_weight: bool = True,
        target_transform: str = "log1p",
        physical_log_clamp: float = 8.0,
        baseline_excess_weight: float = 2.0,
        baseline_target_ratio: float = 0.95,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.precip_intensity_alpha = precip_intensity_alpha
        self.heavy_rain_threshold = heavy_rain_threshold
        self.heavy_rain_weight = heavy_rain_weight
        self.normalize_precip_weight = normalize_precip_weight
        self.target_transform = target_transform
        self.physical_log_clamp = physical_log_clamp
        self.baseline_excess_weight = baseline_excess_weight
        self.baseline_target_ratio = baseline_target_ratio

    def _make_precip_weight(self, fine, valid_mask, loss_like):
        if fine is None:
            return torch.ones_like(loss_like)

        fine = fine.to(dtype=loss_like.dtype, device=loss_like.device).clamp_min(0)
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

        if precip_weight.shape[1] == 1 and loss_like.shape[1] != 1:
            precip_weight = precip_weight.expand(-1, loss_like.shape[1], -1, -1)

        return precip_weight

    def _stats_tensor(self, stats, key, reference, channels):
        if stats is None or key not in stats:
            raise KeyError(f"stats must contain {key!r} when using physical precipitation loss.")

        value = torch.as_tensor(stats[key], dtype=reference.dtype, device=reference.device)
        if value.ndim == 0:
            return value.view(1, 1, 1, 1)

        value = value.flatten()
        if value.numel() == 1:
            return value.view(1, 1, 1, 1)

        if value.numel() != channels:
            raise ValueError(
                f"stats[{key!r}] has {value.numel()} values, but target has {channels} channel(s)."
            )
        return value.view(1, channels, 1, 1)

    def _mask_loss(self, loss, valid_mask):
        if valid_mask is None:
            return loss.mean()

        mask = valid_mask.to(dtype=loss.dtype, device=loss.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1 and loss.shape[1] != 1:
            mask = mask.expand(-1, loss.shape[1], -1, -1)

        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)

    def __call__(
        self,
        net,
        images,               # [B, 1, H, W] normalized clean residual x_0
        conditional_img=None,  # [B, C, H, W] normalized condition fields
        labels=None,
        fine=None,             # [B, 1, H, W] raw ground-truth precipitation in mm
        coarse=None,           # [B, 1, H, W] raw coarse-upsampled precipitation in mm
        stats=None,            # dict with target_mean and target_std
        valid_mask=None,
        augment_pipe=None,
    ):
        # 1. EDM noise sampling in normalized residual space.
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        edm_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # 2. Noise injection.
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        noise = torch.randn_like(y) * sigma

        # 3. Network still sees normalized inputs and predicts normalized clean residual.
        denoised_x0 = net(
            y + noise,
            sigma,
            conditional_img,
            labels,
            augment_labels=augment_labels,
        )

        # 4. Loss space:
        #    If fine/coarse/stats are available, compute the final error in physical mm.
        #    Otherwise fall back to the original normalized EDM denoising loss.
        if stats is not None and coarse is not None and fine is not None:
            # Keep this branch in float32 even when autocast is enabled. This avoids
            # overflow/underflow in expm1 and keeps the physical mm loss numerically safer.
            denoised_f32 = denoised_x0.float()
            coarse_f32 = coarse.to(dtype=torch.float32, device=denoised_x0.device).clamp_min(0)
            fine_f32 = fine.to(dtype=torch.float32, device=denoised_x0.device).clamp_min(0)
            edm_weight_f32 = edm_weight.to(dtype=torch.float32, device=denoised_x0.device)

            channels = denoised_f32.shape[1]
            target_mean = self._stats_tensor(stats, "target_mean", denoised_f32, channels)
            target_std = self._stats_tensor(stats, "target_std", denoised_f32, channels)

            # Convert normalized residual prediction back to residual units.
            r_hat = denoised_f32 * target_std + target_mean

            if self.target_transform == "log1p":
                z_c = torch.log1p(coarse_f32)
                z_pred = z_c + r_hat
                if self.physical_log_clamp is not None and self.physical_log_clamp > 0:
                    z_pred = z_pred.clamp(max=float(self.physical_log_clamp))
                P_pred = torch.expm1(z_pred).clamp_min(0)
            else:
                P_pred = (coarse_f32 + r_hat).clamp_min(0)

            pred_abs_error = torch.abs(P_pred - fine_f32)
            loss = edm_weight_f32 * pred_abs_error.square()

            # Explicitly train against the IDW coarse baseline.  The ordinary
            # physical MSE has no notion of "beating coarse"; this one-sided
            # term adds pressure only where the prediction has not achieved the
            # requested fraction of the coarse absolute error.  ratio=0.95 asks
            # for a 5% pixel-wise improvement while keeping the EDM objective.
            if self.baseline_excess_weight > 0:
                coarse_abs_error = torch.abs(coarse_f32 - fine_f32)
                allowed_error = self.baseline_target_ratio * coarse_abs_error
                excess = torch.relu(pred_abs_error - allowed_error)
                loss = loss + self.baseline_excess_weight * edm_weight_f32 * excess.square()
        else:
            loss = edm_weight * ((denoised_x0 - y) ** 2)

        # 5. Precipitation intensity and heavy-rain weighting.
        loss = loss * self._make_precip_weight(fine, valid_mask, loss)

        # 6. Valid-area mask.
        return self._mask_loss(loss, valid_mask)

def parse_csv_list(value: str, cast=str) -> List:
    if value is None or value == "":
        return []
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def reduce_mean_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    return float(tensor.item())


def make_dataloader(dataset, batch_size, shuffle, num_workers, distributed):
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=shuffle,
    ) if distributed else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def training_step(model, loss_fn, optimiser, data_loader, scaler, epoch, accum, writer, device, stats, grad_clip):
    model.train()
    optimiser.zero_grad(set_to_none=True)

    amp_enabled = device.type == "cuda"
    epoch_losses = []
    running_loss = 0.0

    iterator = tqdm(total=len(data_loader), dynamic_ncols=True) if is_main_process() else nullcontext()
    with iterator as tq:
        if is_main_process():
            tq.set_description(f"Train :: Epoch {epoch}")

        for i, batch in enumerate(data_loader):
            if is_main_process():
                tq.update(1)

            image_input = batch["inputs"].to(device, non_blocking=True)
            image_output = batch["targets"].to(device, non_blocking=True)
            fine = batch["fine"].to(device, non_blocking=True)
            coarse = batch["coarse"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            condition_params = torch.stack(
                (batch["doy"].to(device, non_blocking=True), batch["hour"].to(device, non_blocking=True)),
                dim=1,
            )

            with autocast_context(device):
                loss = loss_fn(
                    net=model,
                    images=image_output,
                    conditional_img=image_input,
                    labels=condition_params,
                    fine=fine,
                    coarse=coarse,
                    stats=stats,
                    valid_mask=valid_mask,
                )
                loss_for_backward = loss / accum

            if amp_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            running_loss += loss.item()
            epoch_losses.append(loss.item())

            do_step = ((i + 1) % accum == 0) or ((i + 1) == len(data_loader))
            if do_step:
                if grad_clip is not None and grad_clip > 0:
                    if amp_enabled:
                        scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if amp_enabled:
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                optimiser.zero_grad(set_to_none=True)

                if writer is not None and is_main_process():
                    global_step = epoch * len(data_loader) + i
                    writer.add_scalar("Loss/train_step", running_loss / min(accum, i + 1), global_step)
                running_loss = 0.0

            if is_main_process():
                tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

    local_mean = sum(epoch_losses) / max(len(epoch_losses), 1)
    return reduce_mean_scalar(local_mean, device)


@torch.no_grad()
def validation_step(model, loss_fn, data_loader, epoch, device, stats):
    model.eval()
    losses = []

    iterator = tqdm(total=len(data_loader), dynamic_ncols=True) if is_main_process() else nullcontext()
    with iterator as tq:
        if is_main_process():
            tq.set_description(f"Val   :: Epoch {epoch}")

        for batch in data_loader:
            if is_main_process():
                tq.update(1)

            image_input = batch["inputs"].to(device, non_blocking=True)
            image_output = batch["targets"].to(device, non_blocking=True)
            fine = batch["fine"].to(device, non_blocking=True)
            coarse = batch["coarse"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            condition_params = torch.stack(
                (batch["doy"].to(device, non_blocking=True), batch["hour"].to(device, non_blocking=True)),
                dim=1,
            )

            with autocast_context(device):
                loss = loss_fn(
                    net=model,
                    images=image_output,
                    conditional_img=image_input,
                    labels=condition_params,
                    fine=fine,
                    coarse=coarse,
                    stats=stats,
                    valid_mask=valid_mask,
                )

            losses.append(loss.item())
            if is_main_process():
                tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

    local_mean = sum(losses) / max(len(losses), 1)
    return reduce_mean_scalar(local_mean, device)


@torch.no_grad()
def sample_model(model, dataloader, num_steps, sigma_min, sigma_max, rho, S_churn, S_min, S_max, S_noise, device):
    # sample only on rank 0
    model.eval()
    net = model.module if isinstance(model, DDP) else model

    batch = next(iter(dataloader))
    images_input = batch["inputs"].to(device)
    coarse = batch["coarse"]
    fine = batch["fine"]

    condition_params = torch.stack(
        (batch["doy"].to(device), batch["hour"].to(device)),
        dim=1,
    )

    sigma_min = max(float(sigma_min), float(net.sigma_min))
    sigma_max = min(float(sigma_max), float(net.sigma_max))

    target_channels = getattr(dataloader.dataset, "target_channels", net.out_channels)
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
    t_steps = torch.cat([net.round_sigma(t_steps).to(device=device, dtype=dtype), torch.zeros_like(t_steps[:1])])

    x_next = init_noise * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= float(t_cur) <= S_max else 0.0
        t_hat = net.round_sigma(t_cur + gamma * t_cur).to(device=device, dtype=dtype)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Rank 0 samples independently; call the unwrapped module so DDP does
        # not wait for collective buffer synchronization from the other ranks.
        denoised = net(x_hat, t_hat, images_input, condition_params).to(dtype)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next, t_next, images_input, condition_params).to(dtype)
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


def build_argparser():
    parser = argparse.ArgumentParser(description="DDP train Taiwan precipitation EDM diffusion with IDW tp and precipitation-aware loss.")

    parser.add_argument("--data-dir", type=str, default="/Users/wuyizhen/Desktop/IDW+loss/ERA_dataset")
    parser.add_argument("--resolution", type=str, default="8km", choices=["1km", "5km", "8km"])
    parser.add_argument("--train-start", type=str, default="19600101")
    parser.add_argument("--train-end", type=str, default="20141125")
    parser.add_argument("--val-start", type=str, default="20141126")
    parser.add_argument("--val-end", type=str, default="20171213")
    parser.add_argument("--condition-vars", type=str, default="q700,t2m,u,v,msl,tp")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--target-transform", type=str, default="log1p", choices=["log1p", "raw"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--tp-idw-k", type=int, default=8)
    parser.add_argument("--tp-idw-power", type=float, default=2.0)

    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--stats-samples", type=int, default=2048)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Maximum global gradient norm; use <=0 to disable.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="./outputs_taiwan_idw_precip_loss_ddp")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=20)

    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--channel-mult", type=str, default="1,2,3,4")
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--attn-resolutions", type=str, default="32,16,8")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--use-fp16", action="store_true")

    parser.add_argument("--P-mean", type=float, default=-1.2)
    parser.add_argument("--P-std", type=float, default=1.2)
    parser.add_argument("--sigma-data", type=float, default=1.0)
    parser.add_argument("--precip-intensity-alpha", type=float, default=0.5)
    parser.add_argument("--heavy-rain-threshold", type=float, default=10.0)
    parser.add_argument("--heavy-rain-weight", type=float, default=2.0)
    parser.add_argument("--no-normalize-precip-weight", action="store_true")
    parser.add_argument("--physical-log-clamp", type=float, default=8.0,
                        help="Clamp log1p predicted precipitation before expm1 in physical-mm loss. Use <=0 to disable.")
    parser.add_argument("--baseline-excess-weight", type=float, default=2.0,
                        help="Weight for the one-sided penalty when prediction error fails to beat coarse.")
    parser.add_argument("--baseline-target-ratio", type=float, default=0.95,
                        help="Target prediction/coarse absolute-error ratio used by the baseline penalty.")

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
    if args.baseline_excess_weight < 0:
        raise ValueError("--baseline-excess-weight must be non-negative.")
    if not 0.0 <= args.baseline_target_ratio <= 1.0:
        raise ValueError("--baseline-target-ratio must be between 0 and 1.")
    device = setup_distributed(args)
    seed_everything(args.seed, get_rank())

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "Model"
    result_dir = output_dir / "results"
    run_dir = output_dir / "runs"

    if is_main_process():
        model_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

    if is_dist_avail_and_initialized():
        dist.barrier()

    if args.stats_path is None:
        args.stats_path = str(output_dir / f"stats_{args.resolution}.json")

    condition_vars = parse_csv_list(args.condition_vars, str)
    channel_mult = parse_csv_list(args.channel_mult, int)
    attn_resolutions = parse_csv_list(args.attn_resolutions, int)
    stats_path = Path(args.stats_path)

    # Rank 0 computes stats first; other ranks wait and load.
    if is_main_process():
        if args.recompute_stats or not stats_path.exists():
            print0(f"[Stats] Computing stats and saving to {stats_path}")
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
            print0(f"[Stats] Found existing stats at {stats_path}")

    if is_dist_avail_and_initialized():
        dist.barrier()

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

    dataloader_train = make_dataloader(dataset_train, args.batch_size, True, args.num_workers, args.distributed)
    dataloader_val = make_dataloader(dataset_val, args.batch_size, False, args.num_workers, args.distributed)

    # Separate non-distributed val loader for rank-0 sampling.
    sample_loader = None
    if is_main_process():
        sample_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=min(args.batch_size, 4),
            shuffle=False,
            num_workers=0,
        )

    print0(f"[DDP] distributed={args.distributed} world_size={get_world_size()}")
    print0(f"[Device] {device}")
    print0(f"[Dataset] train={len(dataset_train)} val={len(dataset_val)}")
    print0(f"[Dataset] input channels={dataset_train.num_input_channels} names={dataset_train.input_channel_names}")
    print0(f"[Dataset] target channels={dataset_train.target_channels}")
    print0(f"[Batch] per_gpu={args.batch_size} world_size={get_world_size()} accum={args.accum} effective={args.batch_size * get_world_size() * args.accum}")

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
    ).to(device)

    if args.distributed:
        network = DDP(
            network,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            output_device=args.local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimiser = torch.optim.AdamW(network.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(str(run_dir)) if is_main_process() else None

    loss_fn = PrecipitationEDMLoss(
        P_mean=args.P_mean,
        P_std=args.P_std,
        sigma_data=args.sigma_data,
        precip_intensity_alpha=args.precip_intensity_alpha,
        heavy_rain_threshold=args.heavy_rain_threshold,
        heavy_rain_weight=args.heavy_rain_weight,
        normalize_precip_weight=not args.no_normalize_precip_weight,
        target_transform=args.target_transform,
        physical_log_clamp=args.physical_log_clamp,
        baseline_excess_weight=args.baseline_excess_weight,
        baseline_target_ratio=args.baseline_target_ratio,
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
        "world_size": get_world_size(),
        "effective_batch": args.batch_size * get_world_size() * args.accum,
    })

    if is_main_process():
        (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val_loss = float("inf")
    losses = []

    for epoch in range(args.epochs):
        if isinstance(dataloader_train.sampler, DistributedSampler):
            dataloader_train.sampler.set_epoch(epoch)
        if isinstance(dataloader_val.sampler, DistributedSampler):
            dataloader_val.sampler.set_epoch(epoch)

        train_loss = training_step(
            network, loss_fn, optimiser, dataloader_train, scaler, epoch,
            args.accum, writer, device, stats, args.grad_clip,
        )
        # Use identical validation noise at every epoch so best.pt compares
        # model quality instead of a different random sigma/noise draw.
        seed_everything(args.seed + 100000, get_rank())
        val_loss = validation_step(network, loss_fn, dataloader_val, epoch, device, stats)
        seed_everything(args.seed + epoch + 1, get_rank())

        if is_main_process():
            losses.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if writer is not None:
                writer.add_scalar("Loss/train_epoch", train_loss, epoch)
                writer.add_scalar("Loss/val_epoch", val_loss, epoch)

            print0(f"[Epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if args.sample_every > 0 and epoch % args.sample_every == 0:
                try:
                    (fig, ax), (base_error, pred_error) = sample_model(
                        network,
                        sample_loader,
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
                    if writer is not None:
                        writer.add_scalar("Error/base", base_error, epoch)
                        writer.add_scalar("Error/pred", pred_error, epoch)
                    print0(f"[Sample] base_error={base_error:.6f} pred_error={pred_error:.6f}")
                except Exception as exc:
                    print0(f"[Warning] sample_model failed at epoch {epoch}: {exc}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                net_to_save = network.module if isinstance(network, DDP) else network
                torch.save(net_to_save.state_dict(), model_dir / "best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": net_to_save.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "config": config,
                        "stats": stats,
                    },
                    model_dir / "best_full.pt",
                )
                print0(f"[Checkpoint] Saved best.pt with val_loss={best_val_loss:.6f}")

            (output_dir / "losses.json").write_text(json.dumps(losses, indent=2), encoding="utf-8")

    dataset_train.close()
    dataset_val.close()
    if writer is not None:
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
