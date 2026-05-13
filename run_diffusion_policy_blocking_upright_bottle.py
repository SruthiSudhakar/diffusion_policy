"""
Drive the YAM follower with a trained diffusion policy as a virtual leader.

UPRIGHT-BOTTLE / TWO-CAMERA variant of run_diffusion_policy_blocking.py.

This script is tailored to checkpoints trained from UprightBottleImageDataset,
which take two image streams ('image1', 'image2') plus 'state' as input. The
two RealSense serials are hardcoded below (CAM0_SERIAL -> image1,
CAM1_SERIAL -> image2). Mapping is fixed by training-time data collection;
see scripts_upright_bottle/convert_upright_bottle_to_zarr.py (line 142-143).

BLOCKING-LOOP semantics (same as the single-camera script):
    1. Grab the most recent frame from each camera + the most recent joint
       state (one obs; tiled to n_obs if the policy expects n_obs > 1).
    2. Run inference -> n_act actions.
    3. Schedule all n_act waypoints back-to-back at dt spacing starting "now".
    4. Sleep until the last waypoint's target time + a small settle margin.
    5. Repeat.

LOCAL-EVERYTHING SETUP (cameras + GPU + robot all on one box):
Terminal 1: follower
cd /home/cvlabusers/Appaji/i2rt && source .venv/bin/activate
python examples/minimum_gello/minimum_gello.py --gripper linear_4310 --mode follower --can-channel can0 --bilateral_kp 0.2

Terminal 2: inference (robodiff env)
cd /home/cvlabusers/Appaji/diffusion_policy
conda activate robodiff
python run_diffusion_policy_blocking_upright_bottle.py -i /home/cvlabusers/Appaji/diffusion_policy/data/jgd/2026.05.12/18.36.37_train_diffusion_unet_hybrid_upright_bottle_image/checkpoints/epoch=0250-train_loss=0.0160.ckpt
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import datetime
import pathlib
import shutil
import subprocess
import threading
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

# RealSense serials. Mapping is fixed by training-time data collection:
# image1 was collected from cam0, image2 from cam1.
# See scripts_upright_bottle/convert_upright_bottle_to_zarr.py (line 142-143).
CAM0_SERIAL = "243522073909"   # -> obs key 'image1'
CAM1_SERIAL = "317222070925"   # -> obs key 'image2'


def make_local_grab(serial, rs_width, rs_height, rs_fps):
    """Open one RealSense by serial and return grab() -> (3,360,640) float32 RGB in [0,1]."""
    import pyrealsense2 as rs  # imported here so cv12 doesn't need it
    print(f'Opening RealSense color stream (serial={serial}): {rs_width}x{rs_height} @ {rs_fps} BGR8')
    pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_device(serial)
    rs_cfg.enable_stream(rs.stream.color, rs_width, rs_height, rs.format.bgr8, rs_fps)
    pipe.start(rs_cfg)

    def grab():
        frames = pipe.wait_for_frames()
        bgr = np.asanyarray(frames.get_color_frame().get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

    return grab, pipe.stop


def make_recording_wrapper(grab_fn, stop_fn, video_path, fps):
    """Pull frames continuously in a background thread; write each to an mp4
    at `fps` and cache the latest one for the main loop to read as an obs.

    grab_fn must return (3, 360, 640) float32 RGB in [0, 1]. The returned
    grab() is non-blocking — it returns the most recent cached frame.
    """
    width, height = 640, 360

    # Pipe raw BGR frames to ffmpeg/libx264 if ffmpeg is on PATH. The OpenCV
    # build in this env doesn't ship libx264, so cv2.VideoWriter falls back
    # to mp4v which produces .mp4 files that some browsers/players refuse
    # to decode. Going through ffmpeg gives a real H.264 + faststart mp4
    # that plays everywhere. cv2 mp4v is kept only as a last-resort fallback.
    ff_path = shutil.which('ffmpeg')
    ff_proc = None
    cv_writer = None
    if ff_path is not None:
        cmd = [
            ff_path, '-y', '-loglevel', 'error',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', f'{float(fps)}',
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            '-movflags', '+faststart',
            str(video_path),
        ]
        ff_proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=10**8,
        )
        print(f'Recording rollout video to {video_path} via ffmpeg/libx264 '
              f'@ {fps}Hz ({width}x{height})')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cv_writer = cv2.VideoWriter(str(video_path), fourcc, float(fps),
                                    (width, height))
        if not cv_writer.isOpened():
            raise RuntimeError(f'cv2.VideoWriter failed to open {video_path}')
        print(f'Recording rollout video to {video_path} via cv2.VideoWriter '
              f'(mp4v fallback — install ffmpeg for H.264) @ {fps}Hz '
              f'({width}x{height})')

    state = {'frame': None, 'frames_written': 0}
    lock = threading.Lock()
    stop_evt = threading.Event()

    def _reader():
        n = 0
        try:
            while not stop_evt.is_set():
                f = grab_fn()
                rgb = (f.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if not bgr.flags['C_CONTIGUOUS']:
                    bgr = np.ascontiguousarray(bgr)
                if ff_proc is not None:
                    ff_proc.stdin.write(bgr.tobytes())
                else:
                    cv_writer.write(bgr)
                n += 1
                if n == 1:
                    print(f'video reader [{video_path.name}]: first frame written, '
                          f'shape={bgr.shape}, dtype={bgr.dtype}, '
                          f'min={bgr.min()}, max={bgr.max()}, mean={bgr.mean():.1f}')
                with lock:
                    state['frame'] = f
                    state['frames_written'] = n
        except BrokenPipeError as e:
            print(f'video reader [{video_path.name}]: ffmpeg pipe closed after {n} frames: {e!r}')
        except Exception as e:
            print(f'video reader [{video_path.name}]: error after {n} frames: {e!r}')

    th = threading.Thread(target=_reader, daemon=True)
    th.start()

    t0 = time.time()
    while True:
        with lock:
            ready = state['frame'] is not None
        if ready:
            break
        if time.time() - t0 > 5.0:
            stop_evt.set()
            raise RuntimeError(f'No frames received within 5s of starting recorder for {video_path}')
        time.sleep(0.01)

    def grab():
        with lock:
            return state['frame'].copy()

    def stop():
        stop_evt.set()
        th.join(timeout=5.0)
        if th.is_alive():
            print(f'video reader [{video_path.name}]: thread did not exit within 5s; '
                  'finalizing writer anyway (last frames may be lost)')
        with lock:
            n = state['frames_written']
        if ff_proc is not None:
            try:
                ff_proc.stdin.close()
            except Exception as e:
                print(f'ffmpeg stdin close failed: {e!r}')
            try:
                rc = ff_proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                ff_proc.kill()
                rc = ff_proc.wait()
                print('ffmpeg did not exit within 30s; killed')
            err = ff_proc.stderr.read().decode('utf-8', errors='replace').strip()
            if rc != 0:
                print(f'ffmpeg exited with code {rc}. stderr:\n{err}')
            elif err:
                print(f'ffmpeg stderr (rc=0):\n{err}')
        else:
            cv_writer.release()
        print(f'Video writer finalized: {n} frames written to {video_path}')
        stop_fn()

    return grab, stop


@click.command()
@click.option('-i', '--input', 'ckpt_path', required=True, help='Path to .ckpt file')
@click.option('--server-host', default='127.0.0.1', help='Follower portal host')
@click.option('--server-port', default=11333, type=int, help='Follower portal port')
@click.option('--frequency', default=15.0, type=float,
              help='Action rate in Hz. Spacing dt between consecutive scheduled '
                   'waypoints. Must match training (15).')
@click.option('--rs-width', default=1280, type=int)
@click.option('--rs-height', default=720, type=int)
@click.option('--rs-fps', default=30, type=int)
@click.option('--device', default='auto', help="'auto' picks cuda:0 if available else cpu.")
@click.option('--num-inference-steps', default=16, type=int,
              help='Diffusion sampling steps.')
@click.option('--n-act-exec', default=0, type=int,
              help='Number of actions to actually execute from each predicted '
                   'chunk of length n_action_steps (e.g. 16). 0 (default) means '
                   'execute the full chunk. Must be in [1, n_action_steps].')
@click.option('--num-samples', default=5, type=int,
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
@click.option('--settle-sec', default=0.5, type=float,
              help='Extra wait after the last waypoint\'s target time before '
                   'capturing the next observation. Lets the interpolator '
                   'finish settling at the final pose.')
@click.option('--dry-run', is_flag=True, default=False,
              help='Run inference but do not send commands to the robot.')
@click.option('--record/--no-record', default=True,
              help='Save observations (640x360 JPEG per cycle, one subdir per '
                   'camera) into an "observations/" folder, plus continuous '
                   'per-camera rollout videos ("rollout_image1.mp4", '
                   '"rollout_image2.mp4") and per-cycle action tensors.')
@click.option('--record-jpeg-quality', default=95, type=int)
@click.option('--video-fps', default=30.0, type=float,
              help='Framerate written into the rollout.mp4 header. The '
                   'background recorder writes one frame per call to the '
                   'underlying camera grab, so this should match the camera '
                   'frame rate (default 30, matching --rs-fps).')
@click.option('--log-settle-err/--no-log-settle-err', default=True,
              help='After each chunk, log the max per-joint distance between '
                   'the robot\'s actual position and the last scheduled action. '
                   'Use to tune --settle-sec / --max-joint-speed. Skipped under '
                   '--dry-run since no commands are sent.')
@click.option('--max-steps', default=300, type=int,
              help='Max total action waypoints scheduled to the robot before '
                   'the loop auto-stops (each cycle schedules n_act_exec). '
                   'Stop is also triggered by Ctrl+C. After either, the '
                   'video is finalized and the user is prompted to label '
                   'the rollout success/failure (used as a prefix on the '
                   'run dir).')
def main(ckpt_path, server_host, server_port,
         frequency, rs_width, rs_height, rs_fps,
         device, num_inference_steps, n_act_exec, scheduler, num_samples,
         max_joint_speed, start_pose_ramp_sec, settle_sec,
         dry_run, record, record_jpeg_quality, video_fps, log_settle_err,
         max_steps):
    if num_samples < 1:
        raise click.BadParameter('--num-samples must be >= 1')
    if n_act_exec < 0:
        raise click.BadParameter('--n-act-exec must be >= 0 (0 = use full chunk)')
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
    n_act = policy.n_action_steps
    print(f'Policy ready: n_obs_steps={n_obs}, n_action_steps={n_act}, action_dim={policy.action_dim}')
    if n_act_exec == 0:
        n_act_exec = n_act
    if n_act_exec > n_act:
        raise click.BadParameter(
            f'--n-act-exec ({n_act_exec}) > policy n_action_steps ({n_act})')
    print(f'Will execute first {n_act_exec}/{n_act} actions per cycle')

    # Verify the policy expects the two image keys we are about to feed it.
    obs_keys = list(policy.normalizer.params_dict.keys())
    for k in ('image1', 'image2'):
        if k not in obs_keys:
            raise RuntimeError(
                f'Policy normalizer is missing required key {k!r}. Found: {obs_keys}')
    use_state = 'state' in obs_keys
    print(f'Policy obs keys: {obs_keys} (use_state={use_state})')
    print(f'Camera mapping: image1 <- serial {CAM0_SERIAL} | '
          f'image2 <- serial {CAM1_SERIAL}')

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

    # 3. Open both RealSense cameras (image1 <- cam0, image2 <- cam1).
    grab1, stop1 = make_local_grab(CAM0_SERIAL, rs_width, rs_height, rs_fps)
    grab2, stop2 = make_local_grab(CAM1_SERIAL, rs_width, rs_height, rs_fps)

    # 3b. Recording dir
    record_dir = None
    obs_dir1 = None
    obs_dir2 = None
    frame_idx = 0
    if record:
        ckpt = pathlib.Path(ckpt_path).resolve()
        run_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        record_dir = ckpt.parent / ckpt.stem / run_stamp
        obs_dir1 = record_dir / 'observations' / 'image1'
        obs_dir2 = record_dir / 'observations' / 'image2'
        obs_dir1.mkdir(parents=True, exist_ok=True)
        obs_dir2.mkdir(parents=True, exist_ok=True)
        print(f'Recording observations to {record_dir / "observations"}')

        # Start background video recorders. From here on grab1()/grab2() are
        # non-blocking and return the most recent frame each reader thread
        # has cached.
        grab1, stop1 = make_recording_wrapper(
            grab1, stop1, record_dir / 'rollout_image1.mp4', video_fps)
        grab2, stop2 = make_recording_wrapper(
            grab2, stop2, record_dir / 'rollout_image2.mp4', video_fps)

    def save_obs(f1: np.ndarray, f2: np.ndarray) -> None:
        nonlocal frame_idx
        if obs_dir1 is None:
            return
        for f, out_dir in ((f1, obs_dir1), (f2, obs_dir2)):
            rgb = (f.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out = out_dir / f'frame_{frame_idx:06d}.jpg'
            cv2.imwrite(str(out), bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), record_jpeg_quality])
        frame_idx += 1

    def stop_cameras():
        for label, stop_fn in (('cam0/image1', stop1), ('cam1/image2', stop2)):
            try:
                stop_fn()
            except Exception as e:
                print(f'stop_cameras: {label} stop failed: {e!r}')

    # 4. Policy warm-up — burn JIT/CUDA-init cost on a dummy obs.
    # Use the same batch size as the real loop so the warm-up covers the
    # CUDA allocator state we will hit during inference.
    print(f'Warming up policy inference (batch={num_samples})')
    f1_warm = grab1()
    f2_warm = grab2()
    q_warm = client.get_joint_pos().result().astype(np.float32)

    def _tile_img(f):
        x = np.broadcast_to(
            np.stack([f] * n_obs, axis=0)[None, ...],
            (num_samples, n_obs, *f.shape),
        ).copy()
        return torch.from_numpy(x).to(device_t)

    with torch.no_grad():
        if hasattr(policy, 'reset'):
            policy.reset()
        warm_obs = {
            'image1': _tile_img(f1_warm),
            'image2': _tile_img(f2_warm),
        }
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
    total_steps = 0
    stop_reason = 'unknown'
    try:
        while True:
            if total_steps >= max_steps:
                print(f'\nReached max-steps={max_steps} (sent {total_steps} '
                      f'action waypoints over {cycle} cycles). Stopping. '
                      f'Holding current pose.')
                if not dry_run:
                    try:
                        client.clear_waypoints()
                        cur = client.get_joint_pos().result()
                        client.command_joint_pos(cur)
                    except Exception as e:
                        print(f'Failed to hold pose: {e}')
                stop_reason = 'max_steps'
                break
            # ---- 5. Capture one fresh observation (both images + state) ----
            # If the policy expects n_obs > 1, tile the same most-recent obs to
            # match the expected shape (we do not maintain any history here).
            # We then tile the obs along the batch dim by num_samples so a
            # single forward pass produces num_samples independent action
            # chunks (each draws its own noise init in conditional_sample).
            f1 = grab1()
            f2 = grab2()
            save_obs(f1, f2)

            def _tile_np(f):
                x = np.stack([f] * n_obs, axis=0)[None, ...]
                return np.broadcast_to(x, (num_samples, *x.shape[1:])).copy()

            obs_img1_np = _tile_np(f1)
            obs_img2_np = _tile_np(f2)
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
                    'image1': torch.from_numpy(obs_img1_np).to(device_t),
                    'image2': torch.from_numpy(obs_img2_np).to(device_t),
                }
                if use_state:
                    obs_t['state'] = torch.from_numpy(obs_state_np).to(device_t)
                pred = policy.predict_action(obs_t)
            actions_all = pred['action'].cpu().numpy()  # (N, n_act, 7)
            actions = actions_all[0, :n_act_exec]       # send first n_act_exec waypoints of sample 0
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
            total_steps += len(actions)
    except KeyboardInterrupt:
        print('\nStopping. Holding current pose.')
        try:
            client.clear_waypoints()
            cur = client.get_joint_pos().result()
            client.command_joint_pos(cur)
        except Exception as e:
            print(f'Failed to hold pose: {e}')
        stop_reason = 'keyboard_interrupt'
    finally:
        stop_cameras()
        print(f'Run finished after {total_steps} action waypoints over '
              f'{cycle} cycles (stop_reason={stop_reason}).')
        final_dir = record_dir
        if record_dir is not None:
            label = None
            while label is None:
                try:
                    ans = input('Label this rollout — [s]uccess or [f]ailure? ').strip().lower()
                except EOFError:
                    print('No input available; leaving run dir unlabeled.')
                    break
                if ans in ('s', 'success'):
                    label = 'success'
                elif ans in ('f', 'failure', 'fail'):
                    label = 'failure'
                else:
                    print("  Please type 's' (success) or 'f' (failure).")
            if label is not None:
                new_dir = record_dir.parent / f'{label}_{record_dir.name}'
                try:
                    record_dir.rename(new_dir)
                    final_dir = new_dir
                    print(f'Renamed run dir -> {new_dir}')
                except Exception as e:
                    print(f'Failed to rename {record_dir} -> {new_dir}: {e}')

        if final_dir is not None:
            viz_script = pathlib.Path(__file__).resolve().parent / 'visualize_run_stats.py'
            print(f'Running {viz_script.name} on {final_dir}')
            try:
                subprocess.run(
                    [sys.executable, str(viz_script), str(final_dir)],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f'visualize_run_stats.py failed (exit {e.returncode})')
            except Exception as e:
                print(f'Failed to launch visualize_run_stats.py: {e!r}')


if __name__ == '__main__':
    main()
