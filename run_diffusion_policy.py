"""
Drive the YAM follower with a trained diffusion policy as a virtual leader.

LOCAL-EVERYTHING SETUP (camera + GPU + robot all on one box):
    Terminal 1: follower
        cd /home/cvlabusers/Appaji/i2rt && source .venv/bin/activate
        python examples/minimum_gello/minimum_gello.py \\
            --gripper linear_4310 --mode follower \\
            --can-channel can0 --bilateral_kp 0.2

    Terminal 2: inference (robodiff env)
        cd /home/cvlabusers/Appaji/diffusion_policy
        ~/miniconda3/envs/robodiff/bin/python run_diffusion_policy.py \\
            -i data/jgd/.../checkpoints/<ckpt>.ckpt

REMOTE-GPU SETUP (camera + robot here, GPU on cv12):
debug witgh echo_test.py

ps -u $USER -o pid,etime,cmd | grep -E 'sshd|ssh '
kill <PID>
ss -ltnp | grep -E '11333|11335'

ssh -N -R 11333:127.0.0.1:11333 -R 11335:127.0.0.1:11335 sruthi@cv12.cs.columbia.edu

python /home/cvlabusers/Appaji/diffusion_policy/scripts_pnp_lego/camera_server.py

python /home/cvlabusers/Appaji/i2rt/examples/minimum_gello/minimum_gello.py --gripper linear_4310 --mode follower --can-channel can0 --bilateral_kp 0.2

python run_diffusion_policy.py \
-i /proj/vondrick3/sruthi/Appaji/diffusion_policy/data/jgd/2026.05.05/17.36.50_train_diffusion_unet_hybrid_pnp_lego_image/checkpoints/epoch=0600-train_loss=0.0072.ckpt \
--frame-source remote \
--frame-host 127.0.0.1 --frame-port 11335 \
--server-host 127.0.0.1 --server-port 11333 \
--num-inference-steps 16 \
--dry-run


"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import datetime
import pathlib
import time
from collections import deque

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
    """Open RealSense locally and return a grab() that yields a (3,360,640)
    float32 RGB tensor in [0, 1], plus a stop() callable."""
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
    # Sanity-check the server is alive.
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
@click.option('--frequency', default=10.0, type=float,
              help='Control rate in Hz. Must match training (10).')
@click.option('--steps-per-inference', default=16, type=int,
              help='Open-loop actions executed before re-planning. '
                   'Defaults to n_action_steps from training.')
@click.option('--rs-width', default=1280, type=int)
@click.option('--rs-height', default=720, type=int)
@click.option('--rs-fps', default=30, type=int)
@click.option('--device', default='auto', help="'auto' picks cuda:0 if available else cpu.")
@click.option('--num-inference-steps', default=16, type=int,
              help='Diffusion sampling steps. Training default (100, DDPM) is too slow '
                   'for real-time 10 Hz; eval_real_robot.py uses 16 (DDIM-style).')
@click.option('--scheduler', type=click.Choice(['keep', 'ddpm', 'ddim']),
              default='ddim',
              help="Inference scheduler. 'ddim' (default) rebuilds a DDIM "
                   "scheduler from the trained DDPM config — required for "
                   "sub-100-step sampling. 'keep' leaves the trained "
                   "scheduler untouched (subsampled DDPM, much worse "
                   "below 100 steps). 'ddpm' explicitly forces DDPM.")
@click.option('--max-joint-speed', default=1.0, type=float,
              help='Per-joint L_inf speed cap (rad/s) passed to the i2rt '
                   'server-side interpolator. Lengthens scheduled waypoint '
                   'durations when the policy commands too aggressively.')
@click.option('--start-pose-ramp-sec', default=2.0, type=float,
              help='Wall-clock duration for the smooth ramp to start_pose at boot.')
@click.option('--dry-run', is_flag=True, default=False,
              help='Run inference but do not send commands to the robot.')
@click.option('--record/--no-record', default=True,
              help='Save each policy-input image (640x360 RGB JPEG) to disk under '
                   '<ckpt_dir>/<ckpt_stem>/<YYYY-mm-dd_HH-MM-SS>/. On by default.')
@click.option('--record-jpeg-quality', default=95, type=int)
def main(ckpt_path, server_host, server_port,
         frame_source, frame_host, frame_port,
         frequency, steps_per_inference, rs_width, rs_height, rs_fps,
         device, num_inference_steps, scheduler,
         max_joint_speed, start_pose_ramp_sec,
         dry_run, record, record_jpeg_quality):
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
        print('WARNING: running on CPU. Each inference call samples 100 DDPM steps '
              'over a 250M-param UNet — expect many seconds per chunk, so real-time '
              '10 Hz control will not be possible.')
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
    if steps_per_inference > n_act:
        raise click.BadParameter(
            f'--steps-per-inference={steps_per_inference} > policy n_action_steps={n_act}')

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
    # Use the server-side trajectory interpolator: schedule one waypoint at
    # start_pose with a generous wall-clock arrival time, and the i2rt
    # MotorChainRobot's 250 Hz tick produces a smooth ramp on the wire.
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

    # 3b. Set up recording directory
    record_dir = None
    frame_idx = 0
    if record:
        ckpt = pathlib.Path(ckpt_path).resolve()
        run_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        record_dir = ckpt.parent / ckpt.stem / run_stamp
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f'Recording image observations to {record_dir}')

    def save_obs(frame_chw_float: np.ndarray) -> None:
        """Persist the policy-input frame (3,360,640) RGB float32 [0,1] as JPEG."""
        nonlocal frame_idx
        if record_dir is None:
            return
        rgb = (frame_chw_float.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = record_dir / f'frame_{frame_idx:06d}.jpg'
        cv2.imwrite(str(out), bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), record_jpeg_quality])
        frame_idx += 1

    dt = 1.0 / frequency

    # 4. Warm up obs buffer
    print(f'Warming up {n_obs}-frame observation buffer...')
    obs_buf = deque(maxlen=n_obs)
    for _ in range(n_obs):
        f = grab()
        obs_buf.append(f)
        save_obs(f)
        time.sleep(dt)

    print(f'Starting policy at {frequency} Hz, {steps_per_inference} actions per inference. '
          f'Ctrl+C to stop and hold pose.{" (DRY RUN — no commands sent)" if dry_run else ""}')

    # Reset the server-side interpolator so the policy episode starts from a
    # clean schedule seeded at the current measured pose.
    if not dry_run:
        client.clear_waypoints()
    eval_t_start = time.time() + 0.5
    iter_idx = 0
    action_exec_latency = 0.01  # seconds; matches eval_real_robot.py:324

    try:
        while True:
            # ---- 5. Inference at `frequency` Hz ----
            obs_np = np.stack(list(obs_buf), axis=0)[None, ...]  # (1, n_obs, 3, 360, 640)
            obs_anchor_time = time.time()
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_np).to(device_t)
                pred = policy.predict_action({'image': obs_t})
            actions = pred['action'][0].cpu().numpy()  # (n_act, 7)
            inference_latency = time.time() - obs_anchor_time

            # Wall-clock target time for each action: spaced by dt, anchored at
            # the obs that produced them. Mirrors eval_real_robot.py:286-347.
            action_ts = np.arange(len(actions), dtype=np.float64) * dt + obs_anchor_time + dt

            # Receding-horizon: drop any whose target_time is already in the past
            # (after a small allowance for one-way RPC latency).
            now = time.time()
            is_new = action_ts > (now + action_exec_latency)
            sched_actions = actions[is_new]
            sched_ts = action_ts[is_new]

            if len(sched_actions) == 0:
                # The whole chunk is stale (inference too slow). Push the last
                # one onto the next available step so we don't stall.
                next_step_idx = int(np.ceil((now - eval_t_start) / dt))
                sched_actions = actions[[-1]]
                sched_ts = np.array([eval_t_start + next_step_idx * dt], dtype=np.float64)
                print(f'Over budget (inference={inference_latency*1000:.0f}ms > {n_act*dt*1000:.0f}ms horizon); '
                      f'fallback at t+{sched_ts[0]-now:.3f}s')
            else:
                print(f'inference={inference_latency*1000:.0f}ms, scheduled {len(sched_actions)}/{len(actions)} waypoints')

            if not dry_run:
                for q, t_target in zip(sched_actions, sched_ts):
                    client.schedule_waypoint(q, float(t_target), max_joint_speed)

            # ---- 6. Refill the obs buffer at ~`frequency` Hz until next inference ----
            iter_idx += steps_per_inference
            next_inference_t = eval_t_start + iter_idx * dt
            while time.time() < next_inference_t - dt * 0.5:
                tick = time.time()
                f = grab()
                obs_buf.append(f)
                save_obs(f)
                rem = dt - (time.time() - tick)
                if rem > 0:
                    time.sleep(rem)
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
