import os
import re
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
import argparse
import math


def cab_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    discretization="edm",
    order=2,
    theta=1.0,
    S_churn=0, S_min=0, S_max=float("inf"), S_noise=1,
):
    assert order in [2, 3], "order must be 2 or 3"

    if sigma_min is None:
        sigma_min = {"edm": 0.002}[discretization]
    if sigma_max is None:
        sigma_max = {"edm": 80}[discretization]

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # internally generate N+1 steps
    internal_steps = num_steps + 1

    step_indices = torch.arange(
        internal_steps,
        dtype=torch.float64,
        device=latents.device
    )

    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (internal_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho

    t_steps = net.round_sigma(t_steps)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    # skip second-last step
    if t_steps.numel() >= 3:
        t_steps = torch.cat([t_steps[:-2], t_steps[-1:]])

    x_next = latents.to(torch.float64) * t_steps[0]

    d_prev = None
    d_prev_prev = None

    h_prev = None
    h_prev_prev = None

    eps = 1e-12

    def safe_div(a, b):
        b_safe = torch.where(
            torch.abs(b) < eps,
            torch.sign(b) * eps + (b == 0).to(b.dtype) * eps,
            b,
        )
        return a / b_safe

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

        x_cur = x_next

        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= t_cur <= S_max else 0
        )

        t_hat = net.round_sigma(t_cur + gamma * t_cur)

        x_hat = x_cur

        if gamma > 0:
            x_hat = x_hat + (
                (t_hat**2 - t_cur**2).sqrt()
                * S_noise
                * randn_like(x_hat)
            )

        h = t_next - t_hat

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat

        # ---------------- Euler ----------------
        if i == 0:

            x_next = x_hat + h * d_cur

        # ---------------- AB2 startup ----------------
        elif i == 1:

            r = safe_div(h, h_prev)

            x_next = x_hat + h * (
                (1.0 + 0.5 * r) * d_cur
                - 0.5 * r * d_prev
            )

        # ---------------- Main update ----------------
        else:

            # ---------------- AB2 ----------------
            if order == 2:

                r = safe_div(h, h_prev)

                x_pred = x_hat + h * (
                    (1.0 + 0.5 * r) * d_cur
                    - 0.5 * r * d_prev
                )

            # ---------------- AB3 ----------------
            else:

                beta0 = safe_div(
                    (h * h) / 3.0
                    + 0.5 * h * (2.0 * h_prev + h_prev_prev)
                    + h_prev * (h_prev + h_prev_prev),

                    h_prev * (h_prev + h_prev_prev),
                )

                beta1 = safe_div(
                    -h * (
                        2.0 * h
                        + 3.0 * h_prev
                        + 3.0 * h_prev_prev
                    ),

                    6.0 * h_prev * h_prev_prev,
                )

                beta2 = safe_div(
                    h * (2.0 * h + 3.0 * h_prev),

                    6.0 * h_prev_prev * (h_prev + h_prev_prev),
                )

                x_pred = x_hat + h * (
                    beta0 * d_cur
                    + beta1 * d_prev
                    + beta2 * d_prev_prev
                )

            # ---------------- CAB corrector ----------------

            r_prev = safe_div(h_prev, h_prev_prev)

            d_ext = (
                (1.0 + r_prev) * d_prev
                - r_prev * d_prev_prev
            )

            defect = d_cur - d_ext

            x_next = x_pred + theta * h * defect

        # ---------------- history update ----------------

        d_prev_prev = (
            d_prev.detach().clone()
            if d_prev is not None else None
        )

        d_prev = d_cur.detach().clone()

        h_prev_prev = (
            h_prev.detach().clone()
            if h_prev is not None else None
        )

        h_prev = h.detach().clone()

    return x_next

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges




def build_parser():
    parser = argparse.ArgumentParser(description="Minimal EDM Ablation Sampler (argparse version)")

    parser.add_argument("--network", required=True, type=str,
                        help="Network .pkl file (path or URL)")
    parser.add_argument("--outdir", required=True, type=str,
                        help="Output directory")

    parser.add_argument("--seeds", type=str, default="0-63",
                        help="Seed list, e.g., 1,2,5-10")
    parser.add_argument("--batch", dest="max_batch_size", type=int, default=32,
                        help="Max batch size")

    parser.add_argument("--class", dest="class_idx", type=int, default=None,
                        help="Class label (default: random)")

    # --- sampler parameters ---
    parser.add_argument("--steps", dest="num_steps", type=int, default=30)
    parser.add_argument("--order", type=int, default=2, choices=[2, 3])
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--S_churn", type=float, default=0.0)
    parser.add_argument("--S_min", type=float, default=0.0)
    parser.add_argument("--S_max", type=float, default=float("inf"))
    parser.add_argument("--S_noise", type=float, default=1.0)
    parser.add_argument("--disc", dest="discretization", default="edm", choices=["edm"])

    return parser


# ------------------------------------------------------------
# MAIN (minimal)
# ------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Parse seeds
    seeds = parse_int_list(args.seeds)

    # Init distributed
    dist.init()
    device = torch.device("cuda")

    num_batches = (
        ((len(seeds) - 1) //
         (args.max_batch_size * dist.get_world_size()) + 1)
        * dist.get_world_size()
    )
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # rank 0 loads model first
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0(f'Loading network from "{args.network}"...')
    with dnnlib.util.open_url(args.network, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)["ema"].to(device)

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    dist.print0(f"Generating {len(seeds)} images into {args.outdir}...")

    # Loop over batches
    for batch_seeds in tqdm.tqdm(rank_batches, disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size,
                             net.img_channels,
                             net.img_resolution,
                             net.img_resolution], device=device)

        # Label logic
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[batch_size], device=device)
            ]
        if args.class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, args.class_idx] = 1

        sampler_kwargs = {
            k: v for k, v in vars(args).items()
            if k in [
                "num_steps",
                "order",
                "theta",
                "S_churn",
                "S_min",
                "S_max",
                "S_noise",
                "discretization",
            ] and v is not None
        }


       
        use_ablation = any(k in sampler_kwargs for k in
                           ["solver", "discretization", "schedule", "scaling"])

        sampler_fn = cab_sampler

        images = sampler_fn(
            net, latents, class_labels,
            randn_like=rnd.randn_like,
            **sampler_kwargs
        )

        img_np = (
            (images * 127.5 + 128).clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        os.makedirs(args.outdir, exist_ok=True)

        for seed, img in zip(batch_seeds, img_np):
            path = os.path.join(args.outdir, f"{seed:06d}.png")
            if img.shape[2] == 1:
                PIL.Image.fromarray(img[:, :, 0], "L").save(path)
            else:
                PIL.Image.fromarray(img, "RGB").save(path)

    torch.distributed.barrier()
    dist.print0("Done.")

if __name__ == "__main__":
    main()