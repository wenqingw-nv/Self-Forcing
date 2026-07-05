"""
Pairs from SYNTHETIC teacher clips (prompts-only regime): h_clean = bidi-Wan 21-latent clips
(wan_gen_synthetic.py), h_gen = causal K=21 rollout seeded from each clip's first NCTX latents
+ same caption. No real videos anywhere. Resumable. Output: wan_cache/pairs_synth.pt
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

DEVICE = "cuda"
K, NCTX = 21, 3
OUT = "wan_cache/pairs_synth.pt"


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None

    d = torch.load("wan_cache/synth_clips.pt", map_location="cpu")
    gt_all, caps = d["gt"], d["captions"]
    N = gt_all.shape[0]

    gens, done = [], 0
    if os.path.exists(OUT):
        prev = torch.load(OUT, map_location="cpu")
        gens = list(prev["gen"].unbind(0)); done = len(gens)
        print(f"resuming from {done}", flush=True)

    for c in range(done, N):
        gt = gt_all[c:c + 1].to(DEVICE).to(torch.bfloat16)
        torch.manual_seed(c)
        noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        _, gen = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                initial_latent=gt[:, :NCTX], return_latents=True)
        gens.append(gen[0, :K].half().cpu())
        if (c + 1) % 20 == 0 or c == N - 1:
            torch.save({"gt": gt_all[:len(gens)], "gen": torch.stack(gens), "captions": caps[:len(gens)]}, OUT)
            print(f"{c + 1}/{N} pairs saved", flush=True)

    print(f"done: {len(gens)} synthetic pairs -> {OUT}")


if __name__ == "__main__":
    main()
