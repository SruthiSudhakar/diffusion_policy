"""Train-set echo test for a diffusion-policy checkpoint.

Loads the same dataset the workspace trained on, picks a few training samples,
runs `policy.predict_action(...)` on the obs, and prints predicted vs.
ground-truth action chunks side-by-side. Optionally saves the input image so
you can compare framing/lighting against what's coming off the live camera.

Usage (from the repo root):
python scripts_pnp_lego/echo_test.py \
-i /proj/vondrick3/sruthi/Appaji/diffusion_policy/data/jgd/2026.05.05/17.36.50_train_diffusion_unet_hybrid_pnp_lego_image/checkpoints/epoch=0800-train_loss=0.0071.ckpt \
--num-samples 5 --num-inference-steps 100

A healthy training run reproduces its own samples to within a small fraction
of a radian per joint. If the policy can't fit its own training set, no
amount of robot-side tuning will help — go fix training (more epochs, more
demos, or check the loss curve).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import click
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option('-i', '--input', 'ckpt_path', required=True, help='Path to .ckpt')
@click.option('--num-samples', default=5, type=int)
@click.option('--num-inference-steps', default=100, type=int,
              help='Use 100 (DDPM-style) for the echo test — clearer signal '
                   'than 16-step DDIM since sampling noise is much lower.')
@click.option('--scheduler', type=click.Choice(['keep', 'ddpm', 'ddim']),
              default='keep',
              help="Inference scheduler. 'keep' (default) reuses whatever "
                   "the checkpoint was trained with. 'ddim' rebuilds a "
                   "DDIMScheduler from the same betas/num_train_timesteps "
                   "and is the right choice for sub-100 step inference; "
                   "'ddpm' explicitly forces DDPM.")
@click.option('--seed', default=0, type=int, help='Index sampling seed')
@click.option('--device', default='auto')
@click.option('--save-images', is_flag=True, default=False,
              help='Dump the first observation frame for each sample to '
                   '<ckpt_dir>/echo_test/.')
@click.option('--use-ema/--no-use-ema', default=True)
def main(ckpt_path, num_samples, num_inference_steps, scheduler, seed,
         device, save_images, use_ema):
    print(f'Loading checkpoint: {ckpt_path}')
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir='/tmp/dp_echo')
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    if use_ema and getattr(workspace, 'ema_model', None) is not None:
        policy = workspace.ema_model
        print('Using EMA model')
    else:
        policy = workspace.model
        print('Using main model')

    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_t = torch.device(device)
    policy.to(device_t).eval()
    if hasattr(policy, 'num_inference_steps'):
        old = policy.num_inference_steps
        policy.num_inference_steps = num_inference_steps
        print(f'num_inference_steps: {old} -> {num_inference_steps}')

    if scheduler != 'keep' and hasattr(policy, 'noise_scheduler'):
        # DDPM and DDIM share the same forward process / training objective
        # (predict epsilon given x_t and t), so a model trained with DDPM
        # can be sampled with DDIM as long as we reuse the same betas and
        # num_train_timesteps. eta=0 -> deterministic ODE sampler.
        old_sched = policy.noise_scheduler
        old_cfg = old_sched.config
        if scheduler == 'ddim':
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            new_sched = DDIMScheduler(
                num_train_timesteps=old_cfg.num_train_timesteps,
                beta_start=old_cfg.beta_start,
                beta_end=old_cfg.beta_end,
                beta_schedule=old_cfg.beta_schedule,
                clip_sample=old_cfg.clip_sample,
                prediction_type=old_cfg.prediction_type,
                set_alpha_to_one=True,
                steps_offset=0,
            )
        else:  # 'ddpm'
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            new_sched = DDPMScheduler(
                num_train_timesteps=old_cfg.num_train_timesteps,
                beta_start=old_cfg.beta_start,
                beta_end=old_cfg.beta_end,
                beta_schedule=old_cfg.beta_schedule,
                clip_sample=old_cfg.clip_sample,
                prediction_type=old_cfg.prediction_type,
                variance_type=getattr(old_cfg, 'variance_type', 'fixed_small'),
            )
        policy.noise_scheduler = new_sched
        print(f'scheduler: {type(old_sched).__name__} -> '
              f'{type(new_sched).__name__}')

    n_obs = policy.n_obs_steps
    n_act = policy.n_action_steps
    print(f'n_obs_steps={n_obs}, n_action_steps={n_act}, action_dim={policy.action_dim}')

    # Build the *training* dataset exactly as the workspace did.
    print(f'Instantiating dataset: {cfg.task.dataset._target_}')
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f'Dataset size: {len(dataset)}')

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dataset), size=num_samples)

    out_dir = None
    if save_images:
        out_dir = pathlib.Path(ckpt_path).resolve().parent.parent / 'echo_test'
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Saving images to {out_dir}')

    np.set_printoptions(precision=3, suppress=True)

    for k, idx in enumerate(indices):
        sample = dataset[int(idx)]  # dict of torch tensors
        # sample['obs']['image']: (T_horizon, 3, H, W);
        # sample['action']: (T_horizon, action_dim).
        # predict_action expects flat dict of tensors with leading batch dim.
        obs = sample['obs']
        gt_action = sample['action'].numpy()  # (T_horizon, action_dim)

        # Take only the first n_obs_steps frames as observation; that's what
        # the policy conditions on (mirrors training; see compute_loss).
        obs_in = {}
        for key, val in obs.items():
            v = val[:n_obs]            # (n_obs, ...)
            v = v.unsqueeze(0).to(device_t)  # (1, n_obs, ...)
            obs_in[key] = v

        with torch.no_grad():
            pred = policy.predict_action(obs_in)
        pred_action = pred['action'][0].cpu().numpy()  # (n_act, action_dim)

        # The dataset returns the FULL horizon. The corresponding ground-truth
        # chunk that predict_action returns is action_pred[start:end] where
        # start = n_obs - 1, end = start + n_act.
        gt_chunk = gt_action[n_obs - 1: n_obs - 1 + n_act]
        if gt_chunk.shape[0] < pred_action.shape[0]:
            # short tail at episode boundary; trim
            pred_action = pred_action[:gt_chunk.shape[0]]

        diff = pred_action - gt_chunk
        per_step_l1 = np.mean(np.abs(diff), axis=1)  # (T,)
        per_joint_rmse = np.sqrt(np.mean(diff ** 2, axis=0))  # (D,)

        print(f'\n=== sample {k} (dataset idx={int(idx)}) ===')
        print(f'gt[0]   = {gt_chunk[0]}')
        print(f'pred[0] = {pred_action[0]}    |Δ|_1={np.mean(np.abs(diff[0])):.3f}')
        print(f'gt[-1]  = {gt_chunk[-1]}')
        print(f'pred[-1]= {pred_action[-1]}   |Δ|_1={np.mean(np.abs(diff[-1])):.3f}')
        print(f'mean per-step L1 across chunk: {per_step_l1.mean():.3f} rad/joint')
        print(f'per-joint RMSE: {per_joint_rmse}')

        if out_dir is not None:
            import cv2
            img_chw = obs['image'][0].numpy()  # (3, H, W) in [0,1]
            img = (img_chw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f'sample_{k:02d}_idx{int(idx)}.jpg'), bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print('\nInterpretation:')
    print('  mean L1 < 0.02 rad/joint -> model fits training data well')
    print('  mean L1 ~ 0.05-0.15      -> partial fit; train more')
    print('  mean L1 > 0.3            -> model is not learning, check dataset/lr')


if __name__ == '__main__':
    main()
