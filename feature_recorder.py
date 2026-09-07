"""Read-only U-Net feature cache for a single sample/member (pilot use).

This is a recorder, NOT a model loader or a replacement diffusion sampler.
Keep the original sampler, checkpoint, precision, and random-number policy.
Wrap only the chosen denoiser call in ``capture``. Corrector calls outside
that context are still executed, but are not recorded.

Storage: per-layer .npy memory maps, no pickle. Statistics use float32.
Spatial maps default to float16; set spatial_dtype='float32' for the audit.
A .partial directory is promoted only after finish() validates completion.
Dependencies: numpy, torch. No network access required.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional
import gc
import json
import os

import numpy as np
import torch


STAT_NAMES = ["signed_mean", "mean_abs", "rms", "spatial_std_population"]


def native_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detached independent CPU copy; preserve native dtype except bfloat16."""
    value = tensor.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy().copy()


class UNetFeatureRecorder:
    """One recorder per (sample_id, member_id); batch size MUST be one.

    layer_names are exact names from the supplied model.named_modules().
    Call model(...) rather than model.forward(...) so PyTorch hooks execute.
    Each selected module must run exactly once per recorded model call.
    capture_inputs=True also records the first tensor input, before forward.
    For blocks after concatenation, this input includes the concatenated skip.

    Missing/duplicate steps, nonfinite values, float16 overflow, changing
    shapes, and pre-existing output directories raise explicit errors.
    This reference implementation intentionally uses synchronous writes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: Iterable[str],
        output_dir: str | Path,
        num_steps: int = 40,
        spatial_steps: Optional[Iterable[int]] = None,
        spatial_dtype: str = "float16",
        capture_inputs: bool = False,
        metadata: Optional[Mapping] = None,
    ) -> None:
        names = list(layer_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Provide a nonempty list of unique module names.")
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        modules = dict(model.named_modules())
        missing = [name for name in names if name not in modules]
        if missing:
            raise KeyError(f"Unknown modules: {missing}; inspect named_modules().")
        if any(module.training for module in model.modules()):
            raise ValueError("Call model.eval() before constructing the recorder.")
        self.dtype = np.dtype(spatial_dtype)
        if self.dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError("spatial_dtype must be float16 or float32.")
        steps = sorted(set(range(num_steps) if spatial_steps is None else spatial_steps))
        if any(not isinstance(s, int) or not 0 <= s < num_steps for s in steps):
            raise ValueError("spatial_steps must contain valid zero-based integers.")
        self.spatial_indices = {s: row for row, s in enumerate(steps)}
        self.num_steps = num_steps
        self.final_dir = Path(output_dir)
        self.work_dir = self.final_dir.with_name(self.final_dir.name + ".partial")
        if self.final_dir.exists() or self.work_dir.exists():
            raise FileExistsError("Output or partial directory exists; do not overwrite a run.")
        # Test user metadata serializability before creating output files.
        json.dumps(dict(metadata or {}), allow_nan=False)
        self.work_dir.mkdir(parents=True)
        self.maps: dict[str, np.memmap] = {}
        self.catalog: dict[str, dict] = {}
        self.handles = []
        self.active: Optional[int] = None
        self.seen: set[str] = set()
        self.done: set[int] = set()
        self.closed = False
        self.failed = False
        self.context_identity = None
        self.expected: set[str] = set()
        self.meta = {
            "schema": "unet-feature-cache-pilot-v1",
            "complete": False,
            "num_steps": num_steps,
            "batch_size": 1,
            "summary_stat_order": STAT_NAMES,
            "spatial_steps": steps,
            "spatial_dtype": self.dtype.name,
            "capture_inputs": capture_inputs,
            "user_metadata": dict(metadata or {}),
            "records": [],
        }
        for index, name in enumerate(names):
            ports = ["input", "output"] if capture_inputs else ["output"]
            for port in ports:
                key = f"layer_{index:02d}_{port}"
                self.expected.add(key)
                self.catalog[key] = {"module_name": name, "port": port}
                if port == "input":
                    def pre_hook(module, args, key=key):
                        if self.active is not None:
                            if not args:
                                raise TypeError("Selected module has no positional input.")
                            self._snapshot(key, args[0])
                        return None
                    self.handles.append(modules[name].register_forward_pre_hook(pre_hook))
                else:
                    def post_hook(module, args, output, key=key):
                        if self.active is not None:
                            self._snapshot(key, output)
                        return None
                    self.handles.append(modules[name].register_forward_hook(post_hook))
        self._write_meta()

    def _write_meta(self) -> None:
        self.meta["layers"] = self.catalog
        path = self.work_dir / "metadata.json"
        tmp = self.work_dir / "metadata.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)

    def _map(self, name: str, shape: tuple, dtype) -> np.memmap:
        dtype = np.dtype(dtype)
        if name not in self.maps:
            arr = np.lib.format.open_memmap(
                self.work_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
            )
            arr[:] = np.nan
            self.maps[name] = arr
        arr = self.maps[name]
        if arr.shape != shape or arr.dtype != dtype:
            raise ValueError(f"Shape/dtype changed for {name}: {shape}/{dtype}")
        return arr

    @staticmethod
    def _check_tensor(tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise TypeError("Expected a BCHW tensor; choose tensor-valued U-Net blocks.")
        if tensor.shape[0] != 1:
            raise ValueError("Pilot recorder only supports batch_size=1.")
        if not tensor.is_floating_point():
            raise TypeError("Expected a floating-point tensor.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite tensor encountered; refusing to cache.")

    def _snapshot(self, key: str, tensor: torch.Tensor) -> None:
        if key in self.seen:
            raise RuntimeError(f"{key} executed twice in one recorded call.")
        self._check_tensor(tensor)
        step = self.active
        if step is None:
            raise RuntimeError("No active capture context.")
        x = tensor.detach().float()[0]
        shape = tuple(x.shape)
        c, h, w = shape
        stats = torch.stack(
            [x.mean((-2, -1)), x.abs().mean((-2, -1)),
             x.square().mean((-2, -1)).sqrt(),
             x.std((-2, -1), unbiased=False)], dim=-1
        )
        if not bool(torch.isfinite(stats).all()):
            raise ValueError(f"Nonfinite float32 statistics in {key}.")
        self._map(f"{key}_summary", (self.num_steps, c, 4), np.float32)[step] = native_numpy(stats)
        if step in self.spatial_indices:
            raw = native_numpy(x)
            if np.max(np.abs(raw)) > np.finfo(self.dtype).max:
                raise OverflowError(f"{key} exceeds {self.dtype}; use float32 spatial storage.")
            self._map(
                f"{key}_spatial", (len(self.spatial_indices), c, h, w), self.dtype
            )[self.spatial_indices[step]] = raw.astype(self.dtype)
        entry = self.catalog[key]
        if "shape_chw" in entry and entry["shape_chw"] != list(shape):
            raise ValueError(f"Spatial shape changed for {key}.")
        entry.update(shape_chw=list(shape), source_dtype=str(tensor.dtype))
        self.seen.add(key)

    def _state(self, name: str, tensor: torch.Tensor) -> None:
        self._check_tensor(tensor)
        if self.active is None:
            raise RuntimeError("State must be saved inside capture().")
        value = native_numpy(tensor)[0]
        self._map(name, (self.num_steps,) + value.shape, value.dtype)[self.active] = value

    def record_denoised(self, denoised: torch.Tensor) -> None:
        """Save the clean normalized-residual ESTIMATE, not a rain map."""
        if "__denoised__" in self.seen:
            raise RuntimeError("Denoised output was already saved for this step.")
        self._state("denoised_normalized_residual", denoised)
        self.seen.add("__denoised__")

    @contextmanager
    def capture(
        self, *, step: int, sigma_cur: float, sigma_eval: float,
        sigma_next: float, residual_input: torch.Tensor,
        mode: str = "sampling", phase: str = "predictor",
    ) -> Iterator["UNetFeatureRecorder"]:
        if self.closed or self.failed or self.active is not None:
            raise RuntimeError("Recorder is closed, failed, or already recording.")
        if step in self.done or not 0 <= step < self.num_steps:
            raise ValueError("Step is duplicated or outside the expected range.")
        sigmas = [float(sigma_cur), float(sigma_eval), float(sigma_next)]
        if not np.isfinite(sigmas).all() or min(sigmas) < 0 or sigmas[1] <= 0:
            raise ValueError("Evaluation sigma must be finite and positive.")
        identity = (mode, phase)
        if self.context_identity is not None and identity != self.context_identity:
            raise ValueError("Never mix sampling/probe modes or predictor/corrector phases.")
        self.context_identity = identity
        self.active, self.seen = step, set()
        try:
            self._state("input_normalized_residual", residual_input)
            yield self
            missing = (self.expected | {"__denoised__"}) - self.seen
            if missing:
                raise RuntimeError(f"Missing recorded tensors: {sorted(missing)}")
            for arr in self.maps.values():
                arr.flush()
            self.done.add(step)
            self.meta["records"].append({
                "step_index": step, "sigma_cur": sigmas[0],
                "sigma_eval": sigmas[1], "sigma_next": sigmas[2],
                "mode": mode, "phase": phase,
            })
            self._write_meta()
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = None
            self.seen.clear()

    def finish(self, *, prediction: torch.Tensor,
               final_residual: torch.Tensor) -> Path:
        """Call once after the unchanged sampler and physical reconstruction."""
        if self.closed or self.failed or self.active is not None:
            raise RuntimeError("Cannot finish a closed, failed, or active recorder.")
        if self.done != set(range(self.num_steps)):
            raise RuntimeError(f"Incomplete steps: {sorted(set(range(self.num_steps)) - self.done)}")
        self._check_tensor(prediction)
        self._check_tensor(final_residual)
        np.save(self.work_dir / "prediction_physical.npy", native_numpy(prediction)[0], allow_pickle=False)
        np.save(self.work_dir / "final_normalized_residual.npy", native_numpy(final_residual)[0], allow_pickle=False)
        self.meta["records"].sort(key=lambda r: r["step_index"])
        self.meta["complete"] = True
        self._write_meta()
        self.close()
        # Each writer must have its own sample/member directory.
        if self.final_dir.exists():
            raise FileExistsError("Another writer created the target directory.")
        os.rename(self.work_dir, self.final_dir)
        return self.final_dir

    def close(self) -> None:
        """Release hooks and mappings. Incomplete files remain in .partial."""
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for arr in self.maps.values():
            arr.flush()
        self.maps.clear()
        # Mapping references are discarded, without requiring NumPy private APIs.
        gc.collect()
        self.closed = True

    def __enter__(self) -> "UNetFeatureRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
