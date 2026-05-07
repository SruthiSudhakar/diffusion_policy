"""
Drive the YAM follower with a trained diffusion policy as a virtual leader.

BLOCKING-LOOP variant.

Loop, top to bottom, with no overlap:
    1. Grab the most recent camera frame and the most recent joint state
       (one observation; tiled to n_obs if the policy expects n_obs > 1).
    2. Run inference -> n_act actions.
    3. Schedule all n_act waypoints back-to-back at dt spacing starting "now".
    4. Sleep until the last waypoint's target time + a small settle margin.
    5. Repeat.

No receding-horizon, no action dropping, no continuous frame pumping. The
robot pauses momentarily at the last action between cycles while inference
runs.

LOCAL-EVERYTHING SETUP (camera + GPU + robot all on one box):
Terminal 1: follower
cd /home/cvlabusers/Appaji/i2rt && source .venv/bin/activate
python examples/minimum_gello/minimum_gello.py --gripper linear_4310 --mode follower --can-channel can0 --bilateral_kp 0.2

Terminal 2: inference (robodiff env)
cd /home/cvlabusers/Appaji/diffusion_policy
conda activate robodiff
python run_diffusion_policy_blocking.py -i /home/cvlabusers/Appaji/diffusion_policy/data/jgd/2026.05.06/23.14.06_train_diffusion_unet_hybrid_pnp_lego_image/checkpoints/epoch=0400-train_loss=0.0049.ckpt
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import datetime
import pathlib
import time

import click
import cv2
import dill
import hydra
import numpy as np
import portal
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def make_local_grab(rs_width, rs_height, rs_fps):
    """Open RealSense locally and return grab() -> (3,360,640) float32 RGB in [0,1]."""
    import pyrealsense2 as rs  # imported here so cv12 doesn't need it
    print(f'Opening RealSense color stream: {rs_width}x{rs_height} @ {rs_fps} BGR8')
    pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, rs_width, rs_height, rs.format.bgr8, rs_fps)
    pipe.start(rs_cfg)

    def grab():
        frames = pipe.wait_for_frames()
        bgr = np.asanyarray(frames.get_color_frame().get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

    return grab, pipe.stop


def make_remote_grab(frame_host, frame_port):
    """Pull JPEG frames from camera_server.py and decode locally."""
    print(f'Connecting to camera server at {frame_host}:{frame_port}')
    cam = portal.Client(f'{frame_host}:{frame_port}')
    f = cam.get_frame().result()
    if not f.get('jpeg'):
        raise RuntimeError('camera server returned empty JPEG; is it ready?')
    print(f'Camera server alive, first JPEG = {len(f["jpeg"])} bytes')

    def grab():
        f = cam.get_frame().result()
        buf = np.frombuffer(f['jpeg'], dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError('cv2.imdecode failed on remote JPEG')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

    def stop():
        pass

    return grab, stop


@click.command()
@click.option('-i', '--input', 'ckpt_path', required=True, help='Path to .ckpt file')
@click.option('--server-host', default='127.0.0.1', help='Follower portal host')
@click.option('--server-port', default=11333, type=int, help='Follower portal port')
@click.option('--frame-source', type=click.Choice(['local', 'remote']), default='local',
              help='local = open RealSense via pyrealsense2; remote = pull JPEGs from camera_server.py')
@click.option('--frame-host', default='127.0.0.1', help='Camera server host (--frame-source remote)')
@click.option('--frame-port', default=11335, type=int, help='Camera server port (--frame-source remote)')
@click.option('--frequency', default=15.0, type=float,
              help='Action rate in Hz. Spacing dt between consecutive scheduled '
                   'waypoints. Must match training (15).')
@click.option('--rs-width', default=1280, type=int)
@click.option('--rs-height', default=720, type=int)
@click.option('--rs-fps', default=30, type=int)
@click.option('--device', default='auto', help="'auto' picks cuda:0 if available else cpu.")
@click.option('--num-inference-steps', default=16, type=int,
              help='Diffusion sampling steps.')
@click.option('--num-samples', default=50, type=int,
              help='Number of action chunks to sample per observation. The same '
                   'obs is tiled along the batch dim and run through the policy '
                   'in a single forward pass; each sample uses an independent '
                   'noise init so the chunks differ. Sample 0 drives the robot; '
                   'all samples are saved when --record is set.')
@click.option('--scheduler', type=click.Choice(['keep', 'ddpm', 'ddim']),
              default='ddim',
              help="Inference scheduler. 'ddim' (default) rebuilds a DDIM "
                   "scheduler from the trained DDPM config. 'keep' leaves "
                   "the trained scheduler untouched.")
@click.option('--max-joint-speed', default=10.0, type=float,
              help='Per-joint L_inf speed cap (rad/s) for the i2rt interpolator.')
@click.option('--start-pose-ramp-sec', default=2.0, type=float,
              help='Wall-clock duration for the smooth ramp to start_pose at boot.')
@click.option('--settle-sec', default=0.23, type=float,
              help='Extra wait after the last waypoint\'s target time before '
                   'capturing the next observation. Lets the interpolator '
                   'finish settling at the final pose.')
@click.option('--dry-run', is_flag=True, default=False,
              help='Run inference but do not send commands to the robot.')
@click.option('--record/--no-record', default=True,
              help='Save each policy-input image (640x360 RGB JPEG) to disk.')
@click.option('--record-jpeg-quality', default=95, type=int)
@click.option('--log-settle-err/--no-log-settle-err', default=True,
              help='After each chunk, log the max per-joint distance between '
                   'the robot\'s actual position and the last scheduled action. '
                   'Use to tune --settle-sec / --max-joint-speed. Skipped under '
                   '--dry-run since no commands are sent.')
def main(ckpt_path, server_host, server_port,
         frame_source, frame_host, frame_port,
         frequency, rs_width, rs_height, rs_fps,
         device, num_inference_steps, scheduler, num_samples,
         max_joint_speed, start_pose_ramp_sec, settle_sec,
         dry_run, record, record_jpeg_quality, log_settle_err):
    if num_samples < 1:
        raise click.BadParameter('--num-samples must be >= 1')
    # 1. Load checkpoint
    print(f'Loading checkpoint: {ckpt_path}')
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir='/tmp/dp_inference')
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_t = torch.device(device)
    if device_t.type == 'cpu':
        print('WARNING: running on CPU. Each inference call will be very slow.')
    policy.to(device_t).eval()
    if hasattr(policy, 'num_inference_steps'):
        old_steps = policy.num_inference_steps
        policy.num_inference_steps = num_inference_steps
        print(f'Set policy.num_inference_steps: {old_steps} -> {num_inference_steps}')
    if scheduler != 'keep' and hasattr(policy, 'noise_scheduler'):
        # DDPM and DDIM share the forward noising process and the epsilon-
        # prediction objective, so a DDPM-trained model samples correctly
        # under DDIM as long as the betas / num_train_timesteps / prediction
        # type match. eta=0 (DDIM default) -> deterministic sampler.
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
        else:
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
        print(f'Swapped scheduler: {type(old_sched).__name__} -> '
              f'{type(new_sched).__name__}')
    n_obs = policy.n_obs_steps
    # assert n_obs==1
    n_act = policy.n_action_steps
    print(f'Policy ready: n_obs_steps={n_obs}, n_action_steps={n_act}, action_dim={policy.action_dim}')

    # Determine which observation keys this checkpoint actually expects.
    # Some checkpoints are image-only; others use image + state. We trust the
    # normalizer's registered keys since the obs encoder and normalizer are
    # both built from the same shape_meta at training time.
    obs_keys = list(policy.normalizer.params_dict.keys())
    if 'image' not in obs_keys:
        raise RuntimeError(
            f'Policy normalizer has no "image" key. Found: {obs_keys}')
    use_state = 'state' in obs_keys
    print(f'Policy obs keys: {obs_keys} (use_state={use_state})')

    dt = 1.0 / frequency

    # 2. Connect to follower
    print(f'Connecting to follower at {server_host}:{server_port}')
    client = portal.Client(f'{server_host}:{server_port}')
    cur = client.get_joint_pos().result()
    print(f'Current follower joint pos: {cur}  (dim={cur.shape[0]})')
    if cur.shape[0] != policy.action_dim:
        raise RuntimeError(
            f'Follower DOF ({cur.shape[0]}) != policy action_dim ({policy.action_dim}). '
            'Check the follower\'s gripper config.')

    # 2b. Move follower to the fixed start pose
    start_pose = np.array([
        -0.02651255, 1.53639277, 1.49328603, -1.66189822,
        0.02994583, 0.05359731, 0.99420248,
    ], dtype=np.float64)
    if start_pose.shape[0] != policy.action_dim:
        raise RuntimeError(
            f'Start pose dim ({start_pose.shape[0]}) != policy action_dim ({policy.action_dim}).')
    print(f'Moving follower to start pose over {start_pose_ramp_sec:.2f}s: {start_pose}')
    client.clear_waypoints()
    client.schedule_waypoint(
        start_pose,
        time.time() + start_pose_ramp_sec,
        max_joint_speed,
    )
    time.sleep(start_pose_ramp_sec + 0.2)
    print(f'Reached start pose: {client.get_joint_pos().result()}')

    # 3. Frame source
    if frame_source == 'local':
        grab, stop_camera = make_local_grab(rs_width, rs_height, rs_fps)
    else:
        grab, stop_camera = make_remote_grab(frame_host, frame_port)

    # 3b. Recording dir
    record_dir = None
    frame_idx = 0
    if record:
        ckpt = pathlib.Path(ckpt_path).resolve()
        run_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        record_dir = ckpt.parent / ckpt.stem / run_stamp
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f'Recording image observations to {record_dir}')

    def save_obs(frame_chw_float: np.ndarray) -> None:
        nonlocal frame_idx
        if record_dir is None:
            return
        rgb = (frame_chw_float.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = record_dir / f'frame_{frame_idx:06d}.jpg'
        cv2.imwrite(str(out), bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), record_jpeg_quality])
        frame_idx += 1

    # 4. Policy warm-up — burn JIT/CUDA-init cost on a dummy obs.
    # Use the same batch size as the real loop so the warm-up covers the
    # CUDA allocator state we will hit during inference.
    print(f'Warming up policy inference (batch={num_samples})')
    f_warm = grab()
    q_warm = client.get_joint_pos().result().astype(np.float32)
    with torch.no_grad():
        if hasattr(policy, 'reset'):
            policy.reset()
        img_warm = np.broadcast_to(
            np.stack([f_warm] * n_obs, axis=0)[None, ...],
            (num_samples, n_obs, *f_warm.shape),
        ).copy()
        warm_obs = {'image': torch.from_numpy(img_warm).to(device_t)}
        if use_state:
            st_warm = np.broadcast_to(
                np.stack([q_warm] * n_obs, axis=0)[None, ...],
                (num_samples, n_obs, q_warm.shape[0]),
            ).copy()
            warm_obs['state'] = torch.from_numpy(st_warm).to(device_t)
        _ = policy.predict_action(warm_obs)

    print(f'Starting blocking policy loop. n_obs={n_obs}, n_act={n_act}, '
          f'num_samples={num_samples}, dt={dt:.4f}s, '
          f'chunk duration={n_act * dt:.2f}s. '
          f'Ctrl+C to stop and hold pose.{" (DRY RUN — no commands sent)" if dry_run else ""}')

    # Ensure no leftover waypoints from the start-pose ramp.
    if not dry_run:
        client.clear_waypoints()

    cycle = 0
    try:
        while True:
            # ---- 5. Capture one fresh observation (most recent image + state) ----
            # If the policy expects n_obs > 1, tile the same most-recent obs to
            # match the expected shape (we do not maintain any history here).
            # We then tile the obs along the batch dim by num_samples so a
            # single forward pass produces num_samples independent action
            # chunks (each draws its own noise init in conditional_sample).
            f = grab()
            save_obs(f)
            obs_img_one = np.stack([f] * n_obs, axis=0)[None, ...]      # (1, n_obs, 3, 360, 640)
            obs_img_np = np.broadcast_to(
                obs_img_one,
                (num_samples, *obs_img_one.shape[1:]),
            ).copy()                                                     # (N, n_obs, 3, 360, 640)
            if use_state:
                q = client.get_joint_pos().result().astype(np.float32)
                obs_state_one = np.stack([q] * n_obs, axis=0)[None, ...]  # (1, n_obs, 7)
                obs_state_np = np.broadcast_to(
                    obs_state_one,
                    (num_samples, *obs_state_one.shape[1:]),
                ).copy()                                                  # (N, n_obs, 7)

            # ---- 6. Inference (batched: one forward pass -> N samples) ----
            t_inf_start = time.time()
            with torch.no_grad():
                obs_t = {
                    'image': torch.from_numpy(obs_img_np).to(device_t),
                }
                if use_state:
                    obs_t['state'] = torch.from_numpy(obs_state_np).to(device_t)
                pred = policy.predict_action(obs_t)
            actions_all = pred['action'].cpu().numpy()  # (N, n_act, 7)
            actions = actions_all[0]                    # send sample 0 to robot
            inference_latency = time.time() - t_inf_start

            # Persist the full (N, n_act, 7) tensor for offline analysis.
            if record_dir is not None:
                np.save(record_dir / f'actions_{cycle:06d}.npy', actions_all)

            # Per-DOF stats across the N samples. mean/min/max collapse over
            # both samples and time (chunk envelope per joint). std is
            # reported per-timestep so we can see how sample disagreement
            # evolves through the chunk (typically small near t=0 where the
            # chunk is conditioned on the obs, larger near t=n_act-1).
            sample_std = actions_all.std(axis=0)                    # (n_act, 7)
            stat_mean = actions_all.mean(axis=(0, 1))               # (7,)
            stat_min = actions_all.min(axis=(0, 1))                 # (7,)
            stat_max = actions_all.max(axis=(0, 1))                 # (7,)
            fmt = lambda v: '[' + ' '.join(f'{x:+.4f}' for x in v) + ']'
            print(f'cycle {cycle}: stats over {num_samples} samples (per-DOF):')
            print(f'  mean = {fmt(stat_mean)}')
            print(f'  min  = {fmt(stat_min)}')
            print(f'  max  = {fmt(stat_max)}')
            print(f'  std across {num_samples} samples, shape '
                  f'(n_act={sample_std.shape[0]}, 7):')
            for t in range(sample_std.shape[0]):
                print(f'    t={t:2d}: {fmt(sample_std[t])}')

            # ---- 7. Schedule the entire chunk back-to-back, starting now ----
            schedule_anchor = time.time()
            action_ts = (np.arange(1, len(actions) + 1, dtype=np.float64) * dt
                         + schedule_anchor)

            if not dry_run:
                for q_a, t_a in zip(actions, action_ts):
                    client.schedule_waypoint(q_a, float(t_a), max_joint_speed)

            chunk_dur = action_ts[-1] - schedule_anchor
            print(f'cycle {cycle}: inference={inference_latency*1000:.0f}ms, '
                  f'submitted {len(actions)} waypoints over {chunk_dur:.2f}s')

            # ---- 8. Wait for the chunk to finish (+ settle) before next obs ----
            sleep_until = action_ts[-1] + settle_sec
            remaining = sleep_until - time.time()
            if remaining > 0:
                time.sleep(remaining)

            # ---- 9. Settle diagnostic ----
            if log_settle_err and not dry_run:
                pos_now = client.get_joint_pos().result()
                err = float(np.max(np.abs(pos_now - actions[-1])))
                print(f'cycle {cycle}: settle err = {err:.4f} rad '
                      f'(max joint dist to last action)')
            cycle += 1
    except KeyboardInterrupt:
        print('\nStopping. Holding current pose.')
        try:
            client.clear_waypoints()
            cur = client.get_joint_pos().result()
            client.command_joint_pos(cur)
        except Exception as e:
            print(f'Failed to hold pose: {e}')
    finally:
        stop_camera()


if __name__ == '__main__':
    main()
