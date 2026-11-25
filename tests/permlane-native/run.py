"""Compile the native HLSL with DXC and run the focused permutation control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dxc", default="dxc", help="path to the x64 DirectX Shader Compiler")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--warp", action="store_true", help="use built-in WARP")
    group.add_argument("--warp-dll", type=Path, help="use this x64 d3d10warp.dll beside a temporary copy of the test executable")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    workspace = here.parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        for width in (32, 64):
            for opt in ("O3", "Od"):
                subprocess.run([args.dxc, "-T", "cs_6_6", "-E", "main", "-WX", "-" + opt,
                                "-D", f"WIDTH={width}", "-D", "CROSS=1", "-D", "FI=0",
                                "-D", "BC=0", "-D", "ALIAS=0", "-D", "DIAG=1",
                                "-Fo", work / f"{width}-{opt}.dxil", here / "control.hlsl"], check=True)
        cargo = subprocess.run(["cargo", "test", "--locked", "-p", "victory-shader", "--lib",
                                "--no-run", "--message-format=json-render-diagnostics"],
                               cwd=workspace, stdout=subprocess.PIPE, text=True, check=True)
        artifacts = [json.loads(line) for line in cargo.stdout.splitlines()]
        executable, = [Path(row["executable"]) for row in artifacts
                       if row.get("executable") and row["target"]["name"] == "victory_shader"]
        private = work / executable.name
        shutil.copyfile(executable, private)
        env = dict(os.environ, PERMLANE_CONTROL_DIR=str(work))
        env.pop("PERMLANE_CONTROL_WARP", None)
        if args.warp_dll:
            dll = work / "d3d10warp.dll"
            shutil.copyfile(args.warp_dll.resolve(strict=True), dll)
            print("WARP SHA-256", hashlib.sha256(dll.read_bytes()).hexdigest(), flush=True)
            env["PERMLANE_CONTROL_WARP"] = str(dll)
        elif args.warp:
            env["PERMLANE_CONTROL_WARP"] = str(Path(os.environ["SystemRoot"]) / "System32/d3d10warp.dll")
        status = subprocess.run([private, "emit::permlane_native::compare_indices", "--exact",
                                 "--ignored", "--nocapture", "--test-threads=1"], cwd=work, env=env)
        return status.returncode


if __name__ == "__main__":
    raise SystemExit(main())
