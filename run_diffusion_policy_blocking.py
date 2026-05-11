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
conda activate jgdrobodiff
python run_diffusion_policy_blocking.py \
-i /home/cvlabusers/Appaji/diffusion_policy/data/jgd/2026.05.06/23.17.10_train_diffusion_unet_hybrid_pnp_lego_image/checkpoints/epoch=0200-train_loss=0.0116.ckpt \
--output-prefix 0 --picked-up
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import datetime
import json
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

VIDEOGEN_REMOTE = "sruthi@cv16.cs.columbia.edu"
VIDEOGEN_INBOX = "/proj/vondrick3/HunyuanVideo-1.5-train-sruthi/new_requests"
VIDEOGEN_OUTPUTS = "/proj/vondrick3/HunyuanVideo-1.5-train-sruthi/outputs/generated_videos"
VIDEOGEN_T = 33  # server's --video_length

PICKUP_TRAJ_PATH = '/home/cvlabusers/Appaji/i2rt/pickup.npy'


def make_local_grab(rs_width, rs_height, rs_fps):
    """Open RealSense locally and return (grab, stop, get_raw_bgr).

    grab() -> (3,360,640) float32 RGB in [0,1] (what the policy sees).
    get_raw_bgr() -> latest captured raw BGR uint8 at native rs_width x rs_height,
    for downstream consumers (e.g. video-gen submission). Raises if no frame
    has been captured yet.
    """
    import pyrealsense2 as rs  # imported here so cv12 doesn't need it
    print(f'Opening RealSense color stream: {rs_width}x{rs_height} @ {rs_fps} BGR8')
    pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, rs_width, rs_height, rs.format.bgr8, rs_fps)
    pipe.start(rs_cfg)

    raw_state = {'bgr': None}
    raw_lock = threading.Lock()

    def grab():
        frames = pipe.wait_for_frames()
        bgr = np.asanyarray(frames.get_color_frame().get_data())
        with raw_lock:
            raw_state['bgr'] = bgr.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

    def get_raw_bgr():
        with raw_lock:
            b = raw_state['bgr']
        if b is None:
            raise RuntimeError('get_raw_bgr() called before any frame captured')
        return b.copy()

    return grab, pipe.stop, get_raw_bgr


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

    return grab, stop, None


def _run_net_with_retries(cmd, *, attempts=5, base_delay=2.0, label=None):
    """subprocess.run(check=True) with exponential backoff for transient
    ssh/rsync failures (DNS hiccups, dropped connections). Reraises after
    the final attempt. Backoff: 2, 4, 8, 16s -> ~30s of slack before giving up.
    """
    label = label or cmd[0]
    for i in range(1, attempts + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            if i == attempts:
                raise
            delay = base_delay * (2 ** (i - 1))
            print(f'{label} attempt {i}/{attempts} failed (exit {e.returncode}); '
                  f'retrying in {delay:.1f}s')
            time.sleep(delay)


def submit_videogen(name, image_bgr, actions_NT7, scratch_dir):
    """rsync (image, manifest) to cv16 inbox with atomic .partial -> .npy rename.

    actions_NT7 has shape (N, T, 7). Server (updated) accepts a 3-D
    trajectory and produces N mp4s per request.
    """
    img_path = scratch_dir / f'{name}.jpg'
    cv2.imwrite(str(img_path), image_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    T = actions_NT7.shape[1]
    manifest = {
        'image_paths': np.array([f'{name}.jpg'] * T),
        'trajectory': actions_NT7.astype(np.float64),  # (N, T, 7)
    }
    npy_path = scratch_dir / f'{name}.npy'
    np.save(npy_path, manifest, allow_pickle=True)
    _run_net_with_retries(
        ['rsync', str(img_path),
         f'{VIDEOGEN_REMOTE}:{VIDEOGEN_INBOX}/{name}.jpg'],
        label='rsync img')
    _run_net_with_retries(
        ['rsync', str(npy_path),
         f'{VIDEOGEN_REMOTE}:{VIDEOGEN_INBOX}/{name}.npy.partial'],
        label='rsync npy')
    _run_net_with_retries(
        ['ssh', VIDEOGEN_REMOTE, 'mv',
         f'{VIDEOGEN_INBOX}/{name}.npy.partial',
         f'{VIDEOGEN_INBOX}/{name}.npy'],
        label='ssh mv')
    print(f'videogen submitted: {name} (N={actions_NT7.shape[0]}, T={T}) -> '
          f'{VIDEOGEN_REMOTE}:{VIDEOGEN_OUTPUTS}/{name}/*.mp4')


def fetch_videogen(name, dest_dir):
    """rsync the cv16 output subdir for <name> into dest_dir/<name>/."""
    local = dest_dir / name
    local.mkdir(parents=True, exist_ok=True)
    _run_net_with_retries(
        ['rsync', '-a',
         f'{VIDEOGEN_REMOTE}:{VIDEOGEN_OUTPUTS}/{name}/',
         f'{local}/'],
        label='rsync fetch')
    print(f'videogen fetched: {name} -> {local}')


def wait_for_videogen(name, expected_count, poll_sec, timeout_sec):
    """Block until cv16 has produced >= expected_count mp4s in
    <VIDEOGEN_OUTPUTS>/<name>/ (server writes 0.mp4, 1.mp4, ... per sample).
    Raises on timeout."""
    out_dir = f'{VIDEOGEN_OUTPUTS}/{name}'
    pattern = f'{out_dir}/*.mp4'
    deadline = time.time() + timeout_sec
    t0 = time.time()
    last_count = -1
    while time.time() < deadline:
        result = subprocess.run(
            ['ssh', VIDEOGEN_REMOTE,
             f'ls -1 {pattern} 2>/dev/null | wc -l'],
            capture_output=True, text=True,
        )
        try:
            count = int(result.stdout.strip() or '0')
        except ValueError:
            count = 0
        if count != last_count:
            print(f'videogen waiting: {name} {count}/{expected_count} '
                  f'(elapsed {time.time() - t0:.1f}s)')
            last_count = count
        if count >= expected_count:
            print(f'videogen ready: {name} (waited {time.time() - t0:.1f}s)')
            return
        time.sleep(poll_sec)
    raise RuntimeError(
        f'videogen timeout after {timeout_sec}s waiting for {expected_count} '
        f'mp4s matching {pattern} (last count={last_count})')


def wait_for_ranking(name, poll_sec, timeout_sec):
    """Block until cv16 has produced ranking.json in
    <VIDEOGEN_OUTPUTS>/<name>/. Raises on timeout."""
    out_dir = f'{VIDEOGEN_OUTPUTS}/{name}'
    target = f'{out_dir}/ranking.json'
    deadline = time.time() + timeout_sec
    t0 = time.time()
    last_present = None
    while time.time() < deadline:
        result = subprocess.run(
            ['ssh', VIDEOGEN_REMOTE,
             f'test -f {target} && echo ok || echo missing'],
            capture_output=True, text=True,
        )
        present = result.stdout.strip() == 'ok'
        if present != last_present:
            print(f'ranking waiting: {name} '
                  f'{"present" if present else "missing"} '
                  f'(elapsed {time.time() - t0:.1f}s)')
            last_present = present
        if present:
            print(f'ranking ready: {name} (waited {time.time() - t0:.1f}s)')
            return
        time.sleep(poll_sec)
    raise RuntimeError(
        f'ranking timeout after {timeout_sec}s waiting for {target}')


def _load_pickup_npy(path):
    """np.load(path).item() with a sys.modules shim so files pickled by
    numpy 2.x (which references numpy._core) load under numpy 1.x."""
    import sys
    import numpy.core
    sys.modules.setdefault('numpy._core', numpy.core)
    for sub in ('multiarray', 'numeric', '_multiarray_umath', 'umath',
                'fromnumeric', '_methods', 'arrayprint'):
        full = f'numpy.core.{sub}'
        if full in sys.modules:
            sys.modules.setdefault(f'numpy._core.{sub}', sys.modules[full])
    return np.load(path, allow_pickle=True).item()


def replay_pickup_trajectory(client):
    """Replay PICKUP_TRAJ_PATH on the follower.

    Mirrors the `l` + `p` flow in
    i2rt/examples/record_replay_trajectory/record_replay_trajectory_client.py:
    1.5 s linear ramp from the current pose to trajectory[0] via
    command_joint_pos at 60 Hz, then command_joint_pos for each
    remaining waypoint at the recorded frequency (default 30 Hz).
    """
    data = _load_pickup_npy(PICKUP_TRAJ_PATH)
    traj = np.asarray(data['trajectory'])
    freq = float(data.get('frequency', 30.0))
    dt = 1.0 / freq
    n = traj.shape[0]
    if n == 0:
        raise RuntimeError(f'{PICKUP_TRAJ_PATH} contains an empty trajectory')
    print(f'Replaying pickup trajectory: {PICKUP_TRAJ_PATH} '
          f'({n} samples @ {freq:.1f} Hz, ~{n*dt:.2f}s)')

    client.clear_waypoints()
    start = client.get_joint_pos().result()
    target0 = traj[0]
    ramp_sec = 1.5
    steps = max(2, int(ramp_sec * 60))
    for i in range(1, steps + 1):
        q = start + (target0 - start) * (i / steps)
        client.command_joint_pos(q)
        time.sleep(ramp_sec / steps)

    last = time.monotonic()
    for idx in range(n):
        while time.monotonic() - last < dt:
            time.sleep(0.001)
        client.command_joint_pos(traj[idx])
        last = time.monotonic()
    print(f'Pickup replay finished. Final pose: {client.get_joint_pos().result()}')


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
                    print(f'video reader: first frame written, '
                          f'shape={bgr.shape}, dtype={bgr.dtype}, '
                          f'min={bgr.min()}, max={bgr.max()}, mean={bgr.mean():.1f}')
                with lock:
                    state['frame'] = f
                    state['frames_written'] = n
        except BrokenPipeError as e:
            print(f'video reader: ffmpeg pipe closed after {n} frames: {e!r}')
        except Exception as e:
            print(f'video reader: error after {n} frames: {e!r}')

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
            raise RuntimeError('No frames received within 5s of starting recorder')
        time.sleep(0.01)

    def grab():
        with lock:
            return state['frame'].copy()

    def stop():
        stop_evt.set()
        th.join(timeout=5.0)
        if th.is_alive():
            print('video reader: thread did not exit within 5s; '
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


# Per-joint diversity weights for the 7-DOF i2rt arm. Joints 0-5 are arm DOFs
# and contribute fully; joint 6 is the gripper, which the diffusion policy
# tends to predict noisily across samples. Without downweighting, two
# trajectories differing only in gripper jitter could outrank trajectories
# with genuinely distinct arm motion. Squared values are the per-DOF
# variance weights; sqrt is applied before flattening so the L2 squares it
# back to the intended weight.
JOINT_WEIGHTS_FOR_DIVERSITY = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1], dtype=np.float32
)


def select_diverse_indices(actions_exec, k, terminal_weight=4.0, n_terminal=2,
                           joint_weights=JOINT_WEIGHTS_FOR_DIVERSITY,
                           seed_idx=0):
    """Greedy farthest-point sampling over weighted flattened L2.

    actions_exec: (M, n_act_exec, action_dim) — executed window only. The
        predicted-but-discarded tail past n_act_exec must NOT be included:
        it never runs, so it cannot constitute "different futures."
    k: number of indices to keep (must satisfy 1 <= k <= M).
    terminal_weight, n_terminal: extra weight on the last n_terminal
        timesteps. The last waypoint defines where the robot ends this
        cycle, so it dominates the metric.
    joint_weights: (action_dim,) per-joint weights — gripper downweighted
        in the default to suppress its noisy contribution.
    seed_idx: index of the first selected sample. Keeping seed_idx=0 means
        sample 0 is always returned first, so callers using winner_idx=0
        as the default executor are unaffected.
    """
    M, T, D = actions_exec.shape
    assert 1 <= k <= M, f'k={k} must be in [1, M={M}]'
    assert joint_weights.shape == (D,), f'joint_weights shape {joint_weights.shape} != ({D},)'
    w_t = np.ones(T, dtype=actions_exec.dtype)
    w_t[-n_terminal:] = terminal_weight
    w = (np.sqrt(w_t)[:, None]
         * np.sqrt(joint_weights.astype(actions_exec.dtype))[None, :])  # (T, D)
    flat = (actions_exec * w[None, :, :]).reshape(M, -1)                # (M, T*D)
    selected = [seed_idx]
    min_d = np.linalg.norm(flat - flat[seed_idx], axis=1)
    min_d[seed_idx] = -np.inf
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        d_new = np.linalg.norm(flat - flat[nxt], axis=1)
        min_d = np.minimum(min_d, d_new)
        min_d[nxt] = -np.inf
    return selected


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
@click.option('--n-act-exec', default=0, type=int,
              help='Number of actions to actually execute from each predicted '
                   'chunk of length n_action_steps (e.g. 16). 0 (default) means '
                   'execute the full chunk. Must be in [1, n_action_steps].')
@click.option('--num-samples', default=4, type=int,
              help='Number of action chunks to sample per observation. The same '
                   'obs is tiled along the batch dim and run through the policy '
                   'in a single forward pass; each sample uses an independent '
                   'noise init so the chunks differ. Sample 0 drives the robot; '
                   'all samples are saved when --record is set.')
@click.option('--oversample', default=50, type=int,
              help='Sample this many candidates per cycle, then prune to '
                   '--num-samples via greedy farthest-point sampling on '
                   'terminal-weighted joint-space L2 over the executed window. '
                   'Set equal to --num-samples to disable diversity pruning. '
                   'Sample 0 of the oversample batch is always kept first so '
                   'the existing winner_idx=0 default still drives the robot.')
@click.option('--scheduler', type=click.Choice(['keep', 'ddpm', 'ddim']),
              default='ddim',
              help="Inference scheduler. 'ddim' (default) rebuilds a DDIM "
                   "scheduler from the trained DDPM config. 'keep' leaves "
                   "the trained scheduler untouched.")
@click.option('--max-joint-speed', default=10.0, type=float,
              help='Per-joint L_inf speed cap (rad/s) for the i2rt interpolator.')
@click.option('--start-pose-ramp-sec', default=2.0, type=float,
              help='Wall-clock duration for the smooth ramp to start_pose at boot.')
@click.option('--picked-up', is_flag=True, default=False,
              help='Skip the fixed start-pose ramp and instead replay '
                   '/home/cvlabusers/Appaji/i2rt/pickup.npy on the follower '
                   '(matches pressing l then p in '
                   'examples/record_replay_trajectory/record_replay_trajectory_client.py). '
                   'The diffusion-policy loop then begins from wherever the '
                   'replay ended.')
@click.option('--settle-sec', default=0.5, type=float,
              help='Extra wait after the last waypoint\'s target time before '
                   'capturing the next observation. Lets the interpolator '
                   'finish settling at the final pose.')
@click.option('--dry-run', is_flag=True, default=False,
              help='Run inference but do not send commands to the robot.')
@click.option('--record/--no-record', default=True,
              help='Save observations (640x360 JPEG per cycle) into an '
                   '"observations/" subfolder, plus a continuous rollout '
                   'video ("rollout.mp4") and per-cycle action tensors.')
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
@click.option('--max-steps', default=240, type=int,
              help='Max total action waypoints scheduled to the robot before '
                   'the loop auto-stops (each cycle schedules n_act_exec). '
                   'Stop is also triggered by Ctrl+C. After either, the '
                   'video is finalized and the user is prompted to label '
                   'the rollout success/failure (used as a prefix on the '
                   'run dir).')
@click.option('--videogen/--no-videogen', default=False,
              help='If set, every cycle submits (raw 1280x720 image, all N '
                   'samples\' full 32-step predictions padded to 33) to the '
                   'cv16 HunyuanVideo server and BLOCKS until the resulting '
                   'mp4s and ranking.json appear, then executes '
                   'actions_all[winner_idx] (the VLM-chosen chunk) instead '
                   'of sample 0. Requires --frame-source local.')
@click.option('--videogen-poll-sec', default=5.0, type=float,
              help='How often to poll cv16 for the generated mp4.')
@click.option('--videogen-timeout-sec', default=1800.0, type=float,
              help='Hard ceiling on how long to wait per cycle for cv16. '
                   'Exceeding this raises and the rollout crashes (after '
                   'holding pose).')
@click.option('--videogen-hz', type=click.Choice(['15', '30']), default='30',
              help='Trajectory rate sent to the cv16 video model. 15 (default) '
                   'sends the policy\'s native 15Hz waypoints (first 33 of 32, '
                   'last padded). 30 linearly upsamples to 30Hz (matches the '
                   'video model\'s training rate) and sends the first 33 '
                   'samples (~1.1s of horizon).')
@click.option('--output-prefix', default='', type=str,
              help='Optional string prepended to the run directory name. '
                   'Stays in front even after the success/failure label is '
                   'added (e.g. "exp1" -> "exp1_success_<timestamp>...". '
                   'Default empty (no prefix).')
@click.option('--seed', default=0, type=int,
              help='Base RNG seed. Before each predict_action call we set '
                   'torch.manual_seed(seed + cycle), which makes the N '
                   'diffusion noise inits reproducible across runs: at '
                   'cycle k, sample i is the same noise tensor in every run '
                   'with the same --seed (the N samples within a cycle are '
                   'still diverse, since they are N consecutive draws of one '
                   'randn call). Combined with cudnn.deterministic=True and '
                   'identical obs conditioning, this yields identical '
                   'action chunks across runs. Default 0.')
def main(ckpt_path, server_host, server_port,
         frame_source, frame_host, frame_port,
         frequency, rs_width, rs_height, rs_fps,
         device, num_inference_steps, n_act_exec, scheduler, num_samples, oversample,
         max_joint_speed, start_pose_ramp_sec, picked_up, settle_sec,
         dry_run, record, record_jpeg_quality, video_fps, log_settle_err,
         max_steps,
         videogen, videogen_poll_sec, videogen_timeout_sec, videogen_hz,
         output_prefix, seed):
    if num_samples < 1:
        raise click.BadParameter('--num-samples must be >= 1')
    if oversample < num_samples:
        raise click.BadParameter(
            f'--oversample ({oversample}) must be >= --num-samples ({num_samples})')
    if n_act_exec < 0:
        raise click.BadParameter('--n-act-exec must be >= 0 (0 = use full chunk)')
    if videogen and frame_source != 'local':
        raise click.BadParameter(
            '--videogen requires --frame-source local (raw 1280x720 frames '
            'are only available from the RealSense path).')
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
    # Reproducibility: deterministic cuDNN kernels so that, given identical
    # noise init + identical obs, predict_action produces identical actions
    # across runs (no float-level drift from kernel autotuning).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    if n_act_exec == 0:
        n_act_exec = n_act
    if n_act_exec > n_act:
        raise click.BadParameter(
            f'--n-act-exec ({n_act_exec}) > policy n_action_steps ({n_act})')
    print(f'Will execute first {n_act_exec}/{n_act} actions per cycle')

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

    # 2b. Either replay pickup.npy or move follower to the fixed start pose.
    if picked_up:
        if dry_run:
            print('--dry-run with --picked-up: skipping pickup replay '
                  '(no commands will be sent to the robot).')
        else:
            replay_pickup_trajectory(client)
    else:
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
        grab, stop_camera, get_raw_bgr = make_local_grab(rs_width, rs_height, rs_fps)
    else:
        grab, stop_camera, get_raw_bgr = make_remote_grab(frame_host, frame_port)

    # 3b. Recording dir. run_stamp is always defined (used as <name> prefix
    # for videogen even when --no-record).
    run_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    record_dir = None
    obs_dir = None
    frame_idx = 0
    prefix_str = f'{output_prefix}_' if output_prefix else ''
    if record:
        ckpt = pathlib.Path(ckpt_path).resolve()
        base_name = run_stamp
        if videogen:
            base_name += '_VLM'
        if picked_up:
            base_name += '_pickedup'
        run_dir_name = f'{prefix_str}{base_name}'
        record_dir = ckpt.parent / ckpt.stem / run_dir_name
        obs_dir = record_dir / 'observations'
        obs_dir.mkdir(parents=True, exist_ok=True)
        print(f'Recording observations to {obs_dir}')

        # Start background video recorder. From here on grab() is non-blocking
        # and returns the most recent frame the reader thread has cached.
        grab, stop_camera = make_recording_wrapper(
            grab, stop_camera, record_dir / 'rollout.mp4', video_fps)

    def save_obs(frame_chw_float: np.ndarray) -> None:
        nonlocal frame_idx
        if obs_dir is None:
            return
        rgb = (frame_chw_float.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = obs_dir / f'frame_{frame_idx:06d}.jpg'
        cv2.imwrite(str(out), bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), record_jpeg_quality])
        frame_idx += 1

    # 4. Policy warm-up — burn JIT/CUDA-init cost on a dummy obs.
    # Use the same batch size as the real loop so the warm-up covers the
    # CUDA allocator state we will hit during inference. Real batch is
    # `oversample` (we sample that many candidates, then prune to num_samples).
    print(f'Warming up policy inference (batch={oversample})')
    f_warm = grab()
    q_warm = client.get_joint_pos().result().astype(np.float32)
    # Use a separate seed for warm-up so it doesn't consume the per-cycle
    # RNG state we'll set inside the loop.
    torch.manual_seed(seed - 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed - 1)
    with torch.no_grad():
        if hasattr(policy, 'reset'):
            policy.reset()
        img_warm = np.broadcast_to(
            np.stack([f_warm] * n_obs, axis=0)[None, ...],
            (oversample, n_obs, *f_warm.shape),
        ).copy()
        warm_obs = {'image': torch.from_numpy(img_warm).to(device_t)}
        if use_state:
            st_warm = np.broadcast_to(
                np.stack([q_warm] * n_obs, axis=0)[None, ...],
                (oversample, n_obs, q_warm.shape[0]),
            ).copy()
            warm_obs['state'] = torch.from_numpy(st_warm).to(device_t)
        _ = policy.predict_action(warm_obs)

    print(f'Starting blocking policy loop. n_obs={n_obs}, n_act={n_act}, '
          f'oversample={oversample}, num_samples={num_samples}, dt={dt:.4f}s, '
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
            # ---- 5. Capture one fresh observation (most recent image + state) ----
            # If the policy expects n_obs > 1, tile the same most-recent obs to
            # match the expected shape (we do not maintain any history here).
            # We then tile the obs along the batch dim by `oversample` so a
            # single forward pass produces `oversample` independent action
            # chunks (each draws its own noise init in conditional_sample).
            # The set is pruned to `num_samples` after inference via greedy
            # FPS on terminal-weighted joint-space L2 (see select_diverse_indices).
            f = grab()
            save_obs(f)
            obs_img_one = np.stack([f] * n_obs, axis=0)[None, ...]      # (1, n_obs, 3, 360, 640)
            obs_img_np = np.broadcast_to(
                obs_img_one,
                (oversample, *obs_img_one.shape[1:]),
            ).copy()                                                     # (M, n_obs, 3, 360, 640)
            if use_state:
                q = client.get_joint_pos().result().astype(np.float32)
                obs_state_one = np.stack([q] * n_obs, axis=0)[None, ...]  # (1, n_obs, 7)
                obs_state_np = np.broadcast_to(
                    obs_state_one,
                    (oversample, *obs_state_one.shape[1:]),
                ).copy()                                                  # (M, n_obs, 7)

            # ---- 6. Inference (batched: one forward pass -> M samples) ----
            # Re-seed per cycle so the M noise inits at cycle k are reproducible
            # across runs (run-A's i-th sample == run-B's i-th sample at the
            # same cycle, given identical obs). The M samples within this call
            # are still diverse: they are M consecutive draws from one randn.
            cycle_seed = seed + cycle
            torch.manual_seed(cycle_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cycle_seed)
            t_inf_start = time.time()
            with torch.no_grad():
                obs_t = {
                    'image': torch.from_numpy(obs_img_np).to(device_t),
                }
                if use_state:
                    obs_t['state'] = torch.from_numpy(obs_state_np).to(device_t)
                pred = policy.predict_action(obs_t)
            actions_all = pred['action'].cpu().numpy()  # (M, n_act, 7)
            # Full unsliced horizon prediction from the policy (e.g. 32),
            # before the n_action_steps window is applied. Same shape across
            # samples; first n_act of axis=1 overlap with actions_all.
            actions_full_all = pred['action_pred'].cpu().numpy()  # (M, horizon, 7)
            inference_latency = time.time() - t_inf_start

            # ---- 6a. Prune to num_samples via greedy FPS on the EXECUTED
            # window only. The tail past n_act_exec is predicted-but-discarded
            # and would only add noise to the diversity metric. seed_idx=0
            # ensures sample 0 of the oversample batch is kept first, so the
            # winner_idx=0 default downstream still drives the robot.
            keep = list(range(actions_all.shape[0]))
            if oversample > num_samples:
                actions_exec = actions_all[:, :n_act_exec]   # (M, n_act_exec, 7)
                keep = select_diverse_indices(
                    actions_exec, k=num_samples,
                    terminal_weight=4.0, n_terminal=2,
                    joint_weights=JOINT_WEIGHTS_FOR_DIVERSITY,
                    seed_idx=0,
                )
                # Persist pre-prune diagnostics before reslicing.
                if record_dir is not None:
                    np.save(record_dir / f'actions_full_pre_{cycle:06d}.npy',
                            actions_full_all)
                    np.save(record_dir / f'keep_indices_{cycle:06d}.npy',
                            np.asarray(keep, dtype=np.int32))
                actions_all = actions_all[keep]              # (num_samples, n_act, 7)
                actions_full_all = actions_full_all[keep]    # (num_samples, horizon, 7)
                print(f'cycle {cycle}: oversampled {oversample}, '
                      f'kept indices {keep}')

            # Persist the kept (post-prune) tensors for offline analysis.
            if record_dir is not None:
                np.save(record_dir / f'actions_{cycle:06d}.npy', actions_all)
                np.save(record_dir / f'actions_full_{cycle:06d}.npy', actions_full_all)

            # Per-DOF stats across the N samples. mean/min/max collapse over
            # both samples and time (chunk envelope per joint). std is
            # reported per-timestep so we can see how sample disagreement
            # evolves through the chunk (typically small near t=0 where the
            # chunk is conditioned on the obs, larger near t=n_act-1).
            print(f'cycle {cycle}; {num_samples} samples (per-DOF):')

            # ---- 6b. Submit to cv16 video-gen and BLOCK on completion ----
            # Robot stays paused at the previous chunk's last waypoint until
            # the cv16 mp4s AND ranking.json for THIS cycle exist. The
            # schedule_anchor below is therefore set after the wait, so
            # action timestamps are anchored at "now" post-wait.
            winner_idx = 0
            if videogen:
                raw_bgr = get_raw_bgr()
                if videogen_hz == '15':
                    # actions_full_all: (N, 32, 7) at 15Hz (policy rate). Pad
                    # along time to T=33 by repeating each sample's last
                    # action once (T must be 4n+1 and <= 33 per server
                    # contract; 33 = 4*8+1).
                    full_NT7 = np.concatenate(
                        [actions_full_all, actions_full_all[:, -1:, :]],
                        axis=1)                                     # (N, 33, 7)
                else:  # '30'
                    # The video model was trained at 30Hz, so its expected
                    # per-step delta is ~2x smaller than the policy's 15Hz
                    # output. Linearly upsample to 64 points at 30Hz (even
                    # indices = original; odd indices = midpoints), then send
                    # the first 33 (= 4*8+1, the server's 4n+1 cap). Covers
                    # the first ~1.1s of the horizon, in-distribution for the
                    # video model.
                    N_s, T_orig, D = actions_full_all.shape
                    full_30hz = np.empty((N_s, 2 * T_orig, D),
                                         dtype=actions_full_all.dtype)
                    full_30hz[:, 0::2, :] = actions_full_all
                    full_30hz[:, 1:-1:2, :] = 0.5 * (
                        actions_full_all[:, :-1, :] + actions_full_all[:, 1:, :])
                    full_30hz[:, -1, :] = actions_full_all[:, -1, :]
                    full_NT7 = full_30hz[:, :33, :]                 # (N, 33, 7)
                name = f'{run_stamp}_{cycle:06d}'
                scratch = (record_dir / 'videogen') if record_dir is not None \
                          else pathlib.Path('/tmp')
                scratch.mkdir(parents=True, exist_ok=True)
                submit_videogen(name, raw_bgr, full_NT7, scratch)
                wait_for_videogen(name, num_samples,
                                  videogen_poll_sec, videogen_timeout_sec)
                wait_for_ranking(name, videogen_poll_sec, videogen_timeout_sec)
                fetch_videogen(name, scratch)
                ranking_path = scratch / name / 'ranking.json'
                with open(ranking_path) as rf:
                    ranking = json.load(rf)
                winner_idx = int(ranking['winner_idx'])
                if not (0 <= winner_idx < num_samples):
                    raise RuntimeError(
                        f'ranking.json winner_idx={winner_idx} out of range '
                        f'[0, {num_samples}) at {ranking_path}')
                print(f'cycle {cycle}: VLM winner_idx={winner_idx} '
                      f'(votes={ranking.get("votes")}, '
                      f'tied={ranking.get("tied_indices")})')

            # ---- 7. Schedule the entire chunk back-to-back, starting now ----
            # Under --videogen, winner_idx is the VLM-chosen sample. Otherwise
            # it stays at 0 (sample 0 drives the robot).
            actions = actions_all[winner_idx, :n_act_exec]
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
    except Exception as e:
        import traceback
        print(f'\nLoop crashed: {e!r}. Holding current pose before re-raising.')
        traceback.print_exc()
        try:
            client.clear_waypoints()
            cur = client.get_joint_pos().result()
            client.command_joint_pos(cur)
        except Exception as e2:
            print(f'Failed to hold pose: {e2}')
        stop_reason = f'exception:{type(e).__name__}'
        raise
    finally:
        stop_camera()
        print(f'Run finished after {total_steps} action waypoints over '
              f'{cycle} cycles (stop_reason={stop_reason}).')
        final_dir = record_dir
        if record_dir is not None:
            try:
                label = input('Label this rollout (free text, blank to skip): ').strip()
            except EOFError:
                print('No input available; leaving run dir unlabeled.')
                label = ''
            if label:
                new_dir = record_dir.parent / f'{prefix_str}{label}_{base_name}'
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
