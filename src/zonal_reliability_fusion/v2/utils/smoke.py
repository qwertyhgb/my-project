"""User-run synthetic resource helper; no images, checkpoints or output files."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from tqdm import trange


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--trainer", default="nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--full-patch", action="store_true", help="Measure plans patch AND plans batch; optional GPU test")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 5:
        parser.error("synthetic smoke only: iterations must be 1..5")
    from zonal_reliability_fusion.nnunet.runtime import check_fixed_nnunet_runtime
    from zonal_reliability_fusion.nnunet.trainers import ACTIVE_V2_TRAINERS, resolve_trainer_class

    check_fixed_nnunet_runtime()
    cls = resolve_trainer_class(args.trainer)
    if cls not in ACTIVE_V2_TRAINERS:
        parser.error("requires unseeded ACTIVE V2 Trainer name")
    configuration = json.loads(Path(args.plans).read_text())["configurations"]["3d_fullres"]
    arch = configuration["architecture"]
    strides = arch["arch_kwargs"]["strides"]
    divisor = [math.prod(s[i] for s in strides) for i in range(3)]
    shape = configuration["patch_size"] if args.full_patch else [max(2, d) for d in divisor]
    # Ensure >1 bottleneck voxel for native InstanceNorm on synthetic CPU shape.
    if not args.full_patch:
        shape[0] = max(shape[0], 2 * divisor[0])
    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = cls.build_network_architecture(
        arch["network_class_name"], arch["arch_kwargs"], arch["_kw_requires_import"],
        6 if cls.anatomy_conditioned else 3, 2, False).to(device)
    batch = configuration["batch_size"] if args.full_patch else 1
    x = torch.randn(batch, 6 if cls.anatomy_conditioned else 3, *shape, device=device)
    if cls.anatomy_conditioned:
        x[:, 3:] = torch.rand_like(x[:, 3:])
    started = time.perf_counter()
    for _ in trange(args.iterations, desc="V2 synthetic forward/backward"):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(x)
            loss = output.float().square().mean()
            if getattr(model, "coarse_logits", None) is not None:
                loss = loss + model.coarse_logits.float().square().mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite synthetic loss")
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(json.dumps({"VERSION": "V2", "trainer": cls.__name__,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "shape": shape, "batch": batch, "precision": "cuda autocast / CPU float32",
                      "ok": args.iterations, "failed": 0, "skipped": 0, "elapsed_seconds": elapsed,
                      "seconds_per_iteration": elapsed / args.iterations,
                      "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated() if device.type == "cuda" else None),
                      "output_path": "stdout only; synthetic, no medical data"}, indent=2))


if __name__ == "__main__":
    main()
