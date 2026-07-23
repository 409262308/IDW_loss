import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


LOW_RES_SHAPE = (14, 9)

PREDICTOR_SPECS = {
    "q700": ("q700_npy", "q700_{date}.npy"),
    "t2m": ("t2m_daily_14x9", "t2m_{date}.npy"),
    "u": ("u_npy", "u_{date}.npy"),
    "v": ("v_npy", "v_{date}.npy"),
    "msl": ("msl_npy", "msl_{date}.npy"),
    "tp": ("ERA5_tp_14x9", "tp_{date}.npy"),
}

TARGET_SPECS = {
    "5km": {
        "folder": "resized_5km",
        "shape": (70, 45),
        "padded_shape": (72, 48),
    },
    "8km": {
        "folder": "resized_8km",
        "shape": (112, 72),
        "padded_shape": (112, 72),
    },
    "1km": {
        "folder": "resized_1km",
        "shape": (350, 225),
        "padded_shape": (352, 232),
    },
}


def default_data_dir():
    return Path(__file__).resolve().parents[3] / "ERA_dataset - 複製" / "ERA_dataset"


def parse_yyyymmdd(value):
    return datetime.strptime(str(value), "%Y%m%d")


def select_evenly_spaced(items, max_items):
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return list(items)
    indices = np.linspace(0, len(items) - 1, max_items).round().astype(int)
    return [items[i] for i in indices]


def load_npy_from_file(path):
    return np.load(path, allow_pickle=False)


def as_hw_tensor(array, shape):
    return torch.from_numpy(array.reshape(shape)).float()


def resize_2d(tensor, out_shape, mode="bilinear"):
    x = tensor[None, None]
    if mode == "nearest":
        y = F.interpolate(x, size=out_shape, mode=mode)
    else:
        y = F.interpolate(x, size=out_shape, mode=mode, align_corners=False)
    return y[0, 0]


_IDW_CACHE = {}


def _precompute_idw_weights(source_shape, target_shape, k=8, power=2.0, eps=1e-12):
    """Precompute regular-grid IDW indices and weights.

    Coordinates are normalized to [0, 1] in both directions, so this works when
    the low-resolution and high-resolution arrays cover the same Taiwan domain.
    """
    source_shape = tuple(source_shape)
    target_shape = tuple(target_shape)
    key = (source_shape, target_shape, int(k), float(power))
    if key in _IDW_CACHE:
        return _IDW_CACHE[key]

    src_h, src_w = source_shape
    tgt_h, tgt_w = target_shape
    k = min(int(k), src_h * src_w)

    src_r = np.linspace(0.0, 1.0, src_h, dtype=np.float64)
    src_c = np.linspace(0.0, 1.0, src_w, dtype=np.float64)
    tgt_r = np.linspace(0.0, 1.0, tgt_h, dtype=np.float64)
    tgt_c = np.linspace(0.0, 1.0, tgt_w, dtype=np.float64)

    src_R, src_C = np.meshgrid(src_r, src_c, indexing="ij")
    tgt_R, tgt_C = np.meshgrid(tgt_r, tgt_c, indexing="ij")

    src_points = np.stack([src_R.reshape(-1), src_C.reshape(-1)], axis=1)
    tgt_points = np.stack([tgt_R.reshape(-1), tgt_C.reshape(-1)], axis=1)

    dist2 = np.sum((tgt_points[:, None, :] - src_points[None, :, :]) ** 2, axis=2)
    indices = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
    selected_d2 = np.take_along_axis(dist2, indices, axis=1)

    exact = selected_d2 < eps
    weights = np.zeros_like(selected_d2, dtype=np.float64)

    has_exact = np.any(exact, axis=1)
    weights[exact] = 1.0

    non_exact = ~has_exact
    dist = np.sqrt(selected_d2[non_exact]) + eps
    w = 1.0 / (dist ** float(power))
    w = w / np.sum(w, axis=1, keepdims=True)
    weights[non_exact] = w

    indices = indices.astype(np.int64)
    weights = weights.astype(np.float32)
    _IDW_CACHE[key] = (indices, weights)
    return indices, weights


def resize_2d_idw(tensor, out_shape, k=8, power=2.0):
    """Resize a 2D regular-grid tensor using inverse distance weighting.

    This is used only for coarse precipitation tp. Other predictors still use
    resize_2d(..., mode="bilinear").
    """
    if tuple(tensor.shape) == tuple(out_shape):
        return tensor

    indices_np, weights_np = _precompute_idw_weights(
        tuple(tensor.shape),
        tuple(out_shape),
        k=k,
        power=power,
    )

    indices = torch.as_tensor(indices_np, dtype=torch.long, device=tensor.device)
    weights = torch.as_tensor(weights_np, dtype=tensor.dtype, device=tensor.device)

    flat = tensor.reshape(-1)
    values = flat[indices]
    out = (values * weights).sum(dim=1).reshape(out_shape)
    return out


def pad_hw(tensor, padded_shape):
    pad_h = padded_shape[0] - tensor.shape[-2]
    pad_w = padded_shape[1] - tensor.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Cannot pad shape {tuple(tensor.shape)} to {padded_shape}.")
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h))


def transform_precip(tensor, transform):
    if transform == "log1p":
        return torch.log1p(torch.clamp(tensor, min=0))
    if transform == "raw":
        return tensor
    raise ValueError(f"Unsupported precipitation transform: {transform}")


def inverse_transform_precip(tensor, transform):
    if transform == "log1p":
        return torch.expm1(tensor).clamp_min(0)
    if transform == "raw":
        return tensor.clamp_min(0)
    raise ValueError(f"Unsupported precipitation transform: {transform}")


class TaiwanERAPrecipDataset(torch.utils.data.Dataset):
    """Daily Taiwan ERA predictors paired with high-resolution precipitation targets."""

    def __init__(
        self,
        data_dir=None,
        resolution="8km",
        start_date="19600101",
        end_date="20201231",
        condition_vars=None,
        use_mask=True,
        target_transform="log1p",
        stats=None,
        max_samples=None,
        tp_idw_k=8,
        tp_idw_power=2.0,
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.resolution = resolution
        self.condition_vars = list(condition_vars or ["q700", "t2m", "u", "v", "msl", "tp"])
        self.use_mask = use_mask
        self.target_transform = target_transform
        self.stats = stats
        self.tp_idw_k = tp_idw_k
        self.tp_idw_power = tp_idw_power
        self._file_cache = None

        if resolution not in TARGET_SPECS:
            raise ValueError(f"resolution must be one of {sorted(TARGET_SPECS)}.")
        unknown = sorted(set(self.condition_vars) - set(PREDICTOR_SPECS))
        if unknown:
            raise ValueError(f"Unknown condition variables: {unknown}.")

        spec = TARGET_SPECS[resolution]
        self.target_shape = tuple(spec["shape"])
        self.padded_shape = tuple(spec["padded_shape"])
        self.img_resolution = (self.padded_shape[1], self.padded_shape[0])
        self.target_channels = 1

        self.input_channel_names = list(self.condition_vars)
        if use_mask:
            self.input_channel_names.append("mask")
        self.num_input_channels = len(self.input_channel_names)

        self._target_entry_by_date = {}
        self.dates = self._collect_dates(start_date, end_date)
        self.dates = select_evenly_spaced(self.dates, max_samples)
        if not self.dates:
            raise ValueError("No dates found for the requested range.")

        self.static_mask = self._load_static_mask() if use_mask else None
        self.valid_mask = self._build_valid_mask()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_cache"] = None
        return state

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, index):
        sample = self._build_sample(self.dates[index])
        return {
            "inputs": sample["inputs"],
            "targets": sample["targets"],
            "fine": sample["fine"],
            "coarse": sample["coarse"],
            "valid_mask": self.valid_mask,
            "year": sample["year"],
            "doy": sample["doy"],
            "hour": torch.tensor(0.0, dtype=torch.float32),
            "date": self.dates[index],
        }

    def close(self):
        if getattr(self, "_zip_handles", None):
            for handle in self._zip_handles.values():
                handle.close()
        self._file_cache = None


    def _collect_dates(self, start_date, end_date):
        target_folder = self.data_dir / TARGET_SPECS[self.resolution]["folder"]
        start = parse_yyyymmdd(start_date)
        end = parse_yyyymmdd(end_date)
        dates = []

        if not target_folder.exists():
            raise FileNotFoundError(f"Target folder not found: {target_folder}")

        for path in sorted(target_folder.glob("*.npy")):
            match = re.search(r"(\d{8})\.npy$", path.name)
            if match is None:
                continue
            date = match.group(1)
            parsed = parse_yyyymmdd(date)
            if start <= parsed <= end:
                dates.append(date)
                self._target_entry_by_date[date] = str(path)

        return sorted(dates)


    def _required_zip_names(self):
        # Kept for backward compatibility with older training scripts.
        # Extracted .npy files are now used directly, so no zip files are required.
        return set()


    def _ensure_zip_handles(self):
        # Kept for backward compatibility. This dataset no longer opens zip files.
        return {}


    def _load_predictor(self, var, date):
        folder_name, pattern = PREDICTOR_SPECS[var]
        path = self.data_dir / folder_name / pattern.format(date=date)
        if not path.exists():
            raise FileNotFoundError(f"Predictor file not found: {path}")
        array = load_npy_from_file(path)
        return as_hw_tensor(array, LOW_RES_SHAPE)


    def _load_target(self, date):
        path = Path(self._target_entry_by_date[date])
        if not path.exists():
            raise FileNotFoundError(f"Target file not found: {path}")
        array = load_npy_from_file(path)
        return as_hw_tensor(array, self.target_shape)

    def _build_valid_mask(self):
        # For Taiwan precipitation, train and evaluate only where the static
        # land mask is valid.  This prevents the nearly-always-zero ocean area
        # from dominating loss/MAE/CRPS.  --no-mask keeps the legacy full-grid
        # behaviour.
        if self.static_mask is not None:
            return self.static_mask.unsqueeze(0)
        mask = torch.ones(self.target_shape, dtype=torch.float32)
        return pad_hw(mask, self.padded_shape).unsqueeze(0)

    def _load_static_mask(self):
        mask_path = self.data_dir / "mask_sd5km.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Static mask not found: {mask_path}")
        mask = torch.from_numpy(np.load(mask_path, allow_pickle=False)).float()
        if mask.ndim != 2:
            raise ValueError(f"Static mask must be 2D, got shape {tuple(mask.shape)}.")
        if tuple(mask.shape) != self.target_shape:
            mask = resize_2d(mask, self.target_shape, mode="nearest")
        mask = (mask > 0.5).to(torch.float32)
        return pad_hw(mask, self.padded_shape)

    def _normalize_input_channel(self, name, tensor):
        if self.stats is None or name == "mask":
            return tensor
        mean = self.stats["input_mean"][name]
        std = max(self.stats["input_std"][name], 1e-8)
        return (tensor - mean) / std

    def _normalize_target(self, tensor):
        if self.stats is None:
            return tensor
        mean = self.stats["target_mean"]
        std = max(self.stats["target_std"], 1e-8)
        return (tensor - mean) / std

    def inverse_normalize_residual(self, residual_norm):
        if self.stats is None:
            return residual_norm
        mean = torch.as_tensor(self.stats["target_mean"], dtype=residual_norm.dtype, device=residual_norm.device)
        std = torch.as_tensor(self.stats["target_std"], dtype=residual_norm.dtype, device=residual_norm.device)
        return residual_norm * std + mean

    def residual_to_fine_image(self, residual_norm, coarse_image):
        residual = self.inverse_normalize_residual(residual_norm)
        coarse_transformed = transform_precip(coarse_image, self.target_transform)
        return inverse_transform_precip(coarse_transformed + residual, self.target_transform)

    def _build_sample(self, date):
        coarse_tp = self._load_predictor("tp", date)
        coarse_tp_up = resize_2d_idw(
            coarse_tp,
            self.target_shape,
            k=self.tp_idw_k,
            power=self.tp_idw_power,
        )
        coarse_tp_up = pad_hw(coarse_tp_up, self.padded_shape)

        fine = self._load_target(date)
        fine = pad_hw(fine, self.padded_shape)

        inputs = []
        for var in self.condition_vars:
            low = self._load_predictor(var, date)
            if var == "tp":
                high = resize_2d_idw(
                    low,
                    self.target_shape,
                    k=self.tp_idw_k,
                    power=self.tp_idw_power,
                )
                high = pad_hw(high, self.padded_shape)
                high = transform_precip(high, self.target_transform)
            else:
                high = resize_2d(low, self.target_shape, mode="bilinear")
                high = pad_hw(high, self.padded_shape)
            inputs.append(self._normalize_input_channel(var, high))

        if self.use_mask:
            inputs.append(self.static_mask)

        fine_transformed = transform_precip(fine, self.target_transform)
        coarse_transformed = transform_precip(coarse_tp_up, self.target_transform)
        residual = (fine_transformed - coarse_transformed).unsqueeze(0)

        parsed = parse_yyyymmdd(date)
        doy = (parsed.timetuple().tm_yday - 1) / (366.0 if self._is_leap_year(parsed.year) else 365.0)
        year = (parsed.year - 1960) / (2020 - 1960)

        return {
            "inputs": torch.stack(inputs, dim=0),
            "targets": self._normalize_target(residual),
            "fine": fine.unsqueeze(0),
            "coarse": coarse_tp_up.unsqueeze(0),
            "year": torch.tensor(year, dtype=torch.float32),
            "doy": torch.tensor(doy, dtype=torch.float32),
        }

    @staticmethod
    def _is_leap_year(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def plot_batch(self, coarse_image, fine_image, fine_image_pred, valid_mask=None, max_items=3):
        n_items = min(max_items, coarse_image.shape[0])
        fig, axes = plt.subplots(n_items, 3, figsize=(10, 3 * n_items), squeeze=False)
        titles = ["coarse tp upsampled", "prediction", "ground truth"]
        vmax = torch.quantile(fine_image[:n_items].detach().cpu().flatten(), 0.99).item()
        vmax = max(vmax, 1e-6)

        for row in range(n_items):
            arrays = [coarse_image[row, 0], fine_image_pred[row, 0], fine_image[row, 0]]
            for col, array in enumerate(arrays):
                data = array.detach().cpu()
                if valid_mask is not None:
                    data = data * valid_mask[row if valid_mask.shape[0] > 1 else 0, 0].detach().cpu()
                axes[row, col].imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
                axes[row, col].set_title(titles[col])
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        fig.tight_layout()
        return fig, axes


def compute_stats(dataset, max_samples=2048):
    dates = select_evenly_spaced(dataset.dates, max_samples)
    input_sum = {name: 0.0 for name in dataset.input_channel_names}
    input_sumsq = {name: 0.0 for name in dataset.input_channel_names}
    input_count = {name: 0 for name in dataset.input_channel_names}
    target_sum = 0.0
    target_sumsq = 0.0
    target_count = 0

    valid = dataset.valid_mask[0].bool()
    for date in dates:
        sample = dataset._build_sample(date)
        inputs = sample["inputs"]
        target = sample["targets"][0]
        for channel, name in enumerate(dataset.input_channel_names):
            if name == "mask":
                input_sum[name] += 0.0
                input_sumsq[name] += 1.0
                input_count[name] += 1
                continue
            values = inputs[channel][valid].double()
            input_sum[name] += values.sum().item()
            input_sumsq[name] += values.square().sum().item()
            input_count[name] += values.numel()

        target_values = target[valid].double()
        target_sum += target_values.sum().item()
        target_sumsq += target_values.square().sum().item()
        target_count += target_values.numel()

    input_mean = {}
    input_std = {}
    for name in dataset.input_channel_names:
        if name == "mask":
            input_mean[name] = 0.0
            input_std[name] = 1.0
            continue
        mean = input_sum[name] / input_count[name]
        var = max(input_sumsq[name] / input_count[name] - mean * mean, 1e-12)
        input_mean[name] = mean
        input_std[name] = math.sqrt(var)

    target_mean = target_sum / target_count
    target_var = max(target_sumsq / target_count - target_mean * target_mean, 1e-12)
    return {
        "resolution": dataset.resolution,
        "target_transform": dataset.target_transform,
        "spatial_mask": "land" if dataset.use_mask else "full_grid",
        "tp_resize": "idw",
        "tp_idw_k": dataset.tp_idw_k,
        "tp_idw_power": dataset.tp_idw_power,
        "num_stat_samples": len(dates),
        "input_channels": dataset.input_channel_names,
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": math.sqrt(target_var),
    }


def save_stats(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_stats(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
