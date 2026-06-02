#!/usr/bin/env python
"""Run the rollout visualization on every ``*VLM`` run dir under a root, in
parallel.

Recursively finds directories whose name matches ``--pattern`` (default
``*VLM``) and that look like a real ``--videogen`` run (they contain at least
one ``videogen/<step>/ranking.json``), then runs the visualization script on
each. Renders run concurrently (``--jobs``) so a big sweep doesn't take forever.

Any unrecognized flags are forwarded verbatim to the visualization script, e.g.

    python scripts_pnp_lego/run_all_visualizations.py \
        /proj/.../checkpoints/epoch=0200-train_loss=0.0116 \
        --jobs 12 --skip_existing -- --denoise_steps 12 --content_width 1280

(everything after ``--`` — or any flag this script doesn't define — is passed
through to each visualization invocation).
"""
import argparse
import concurrent.futures as cf
import fnmatch
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCRIPT = os.path.join(HERE, "visualization_try_2_norankingviz.py")


def find_run_dirs(root, pattern):
    """Recursively find run dirs matching `pattern` that contain a videogen
    step with a ranking.json (so we skip empty/aborted dirs)."""
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if fnmatch.fnmatch(d, pattern):
                full = os.path.join(dirpath, d)
                if glob.glob(os.path.join(full, "videogen", "*", "ranking.json")):
                    out.append(full)
    return sorted(out)


def has_output(run_dir):
    """True if a visualization mp4 already exists for this run dir."""
    return bool(glob.glob(os.path.join(run_dir, "visualization_try_2_norankingviz*.mp4")))


def render_one(run_dir, script, passthrough):
    cmd = [sys.executable, script, "--run_dir", run_dir, "--max_steps", "0",
           *passthrough]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    return run_dir, proc.returncode, dt, proc.stdout, proc.stderr


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory to search under for *VLM run dirs")
    ap.add_argument("--script", default=DEFAULT_SCRIPT,
                    help="visualization script to run (default: "
                         "visualization_try_2_norankingviz.py)")
    ap.add_argument("--pattern", default="*VLM",
                    help="glob for run-dir basenames (default '*VLM')")
    ap.add_argument("--jobs", "-j", type=int, default=8,
                    help="number of renders to run in parallel (default 8)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip run dirs that already have a visualization mp4")
    ap.add_argument("--dry_run", action="store_true",
                    help="just list the run dirs that would be rendered")
    args, passthrough = ap.parse_known_args()
    # allow an explicit '--' separator before pass-through args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if not os.path.isdir(args.root):
        sys.exit(f"root is not a directory: {args.root}")
    if not os.path.exists(args.script):
        sys.exit(f"visualization script not found: {args.script}")

    run_dirs = find_run_dirs(args.root, args.pattern)
    if args.skip_existing:
        run_dirs = [d for d in run_dirs if not has_output(d)]
    n = len(run_dirs)
    print(f"[scan] {n} run dir(s) matching '{args.pattern}' under {args.root}"
          + (" (skipping ones with existing output)" if args.skip_existing else ""))
    for d in run_dirs:
        print(f"   - {d}")
    if not n:
        return
    if args.dry_run:
        return
    if passthrough:
        print(f"[args] forwarding to {os.path.basename(args.script)}: {' '.join(passthrough)}")
    print(f"[run] rendering {n} dir(s), {args.jobs} at a time ...")

    ok, failed = [], []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(render_one, d, args.script, passthrough): d
                for d in run_dirs}
        for fut in cf.as_completed(futs):
            run_dir, rc, dt, out, err = fut.result()
            done += 1
            name = os.path.relpath(run_dir, args.root)
            if rc == 0:
                ok.append(run_dir)
                # echo the script's final "[done] wrote ..." line if present
                last = next((ln for ln in reversed(out.splitlines())
                             if "[done]" in ln), "")
                print(f"[{done}/{n}] OK   ({dt:5.0f}s)  {name}")
                if last:
                    print(f"           {last.strip()}")
            else:
                failed.append(run_dir)
                print(f"[{done}/{n}] FAIL (rc={rc}, {dt:.0f}s)  {name}")
                tail = "\n".join((err or out).splitlines()[-12:])
                print("           " + tail.replace("\n", "\n           "))

    print(f"\n[summary] {len(ok)} ok, {len(failed)} failed (of {n})")
    if failed:
        print("[failed]")
        for d in failed:
            print(f"   - {d}")
        sys.exit(1)


if __name__ == "__main__":
    main()
