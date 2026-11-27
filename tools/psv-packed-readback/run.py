"""Build the original packed-input fixture from the two public VIctory source trees."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
WRITER = "crates/victory-shader/src/emit/container.rs"
BLOBS = {"dedup": "222deb74daa44336fef752441c6cfbe46a72a410",
         "repeat": "79abfdd84d55e30ab5d9d104001a1c49517bd4d6"}


def chunks(container):
    def u32(offset):
        return struct.unpack_from("<I", container, offset)[0]

    assert container[:4] == b"DXBC" and u32(24) == len(container)
    return {container[o:o+4]: container[o+8:o+8+u32(o+4)]
            for o in (u32(32 + 4*i) for i in range(u32(28)))}


def offsets(psv):
    def u32(offset):
        return struct.unpack_from("<I", psv, offset)[0]

    assert u32(0) == 48 and u32(52) == 0  # PSV2, no resources
    table = 60
    indices = table + u32(table - 4)
    elements = indices + 4 + 4*u32(indices)
    stride = u32(elements)
    assert stride == 16 and sum(psv[32:35]) == 6
    return [u32(elements + 4 + stride*i) for i in range(6)]


def verify_outputs(folder):
    parent = chunks((folder / "ps-dedup.dxbc").read_bytes())
    changed = chunks((folder / "ps-repeat.dxbc").read_bytes())
    assert parent.keys() == changed.keys()
    assert [tag for tag in parent if parent[tag] != changed[tag]] == [b"PSV0"]
    assert offsets(parent[b"PSV0"]) == [1, 1, 1, 1, 0, 0]
    assert offsets(changed[b"PSV0"]) == [1, 10, 19, 28, 0, 0]


def emit(tree, label, work):
    blob = subprocess.check_output(["git", "rev-parse", "HEAD:" + WRITER], cwd=tree, text=True).strip()
    if blob != BLOBS[label]:
        raise ValueError(f"{label} checkout has a different PSV writer")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "crates/victory-shader"], cwd=tree, check=True)
    cargo = subprocess.run(["cargo", "build", "--release", "--locked", "--jobs", "2",
                            "-p", "victory-shader", "--lib", "--message-format=json-render-diagnostics"],
                           cwd=tree, stdout=subprocess.PIPE, text=True, check=True)
    libraries = []
    for line in cargo.stdout.splitlines():
        record = json.loads(line)
        if record.get("reason") == "compiler-artifact" and record["target"]["name"] == "victory_shader":
            libraries.extend(Path(x) for x in record["filenames"] if x.endswith(".rlib"))
    library, = libraries
    driver = work / f"emit-{label}.exe"
    subprocess.run(["rustc", "--edition=2024", "-C", "panic=abort", HERE/"emit_packed.rs",
                    "-L", "dependency=" + str(library.parent), "--extern", "victory_shader=" + str(library),
                    "-o", driver], cwd=tree, check=True)
    subprocess.run([driver, work/"ps-dxc.dxbc", work/f"ps-{label}.dxbc"], cwd=work, check=True)


def validate(executable, work, dll_dir, legacy):
    result = subprocess.run([executable, "validate", dll_dir], cwd=work, capture_output=True,
                            text=True, check=True)
    print(result.stdout, end="")
    for arm in ("dxc", "repeat", "dedup"):
        expected = "0x80aa0013" if legacy and arm == "dedup" else "0x00000000"
        if f"validate ps-{arm}.dxbc {expected}" not in result.stdout.lower():
            raise ValueError(f"Unexpected {arm} validation result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=HERE.parents[1], help="Checkout of 63bcca43 or this fixture branch")
    parser.add_argument("--candidate", required=True, type=Path, help="Checkout of PR #3323 at ba7f9e61")
    parser.add_argument("--dxc", required=True, type=Path, help="Path to modern x64 Windows SDK dxc.exe")
    parser.add_argument("--legacy-bin", type=Path, help="Optional Microsoft DXC 1.7 bin/x64 folder")
    parser.add_argument("--prepare-only", action="store_true", help="Run compile and validator checks without opening adapters")
    args = parser.parse_args()
    if not shutil.which("cl.exe"):
        parser.error("Use an x64 Visual Studio Native Tools prompt with the Windows SDK installed")
    dxc = args.dxc.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="victory-packed-readback-") as tmp:
        work = Path(tmp)
        for profile, entry, name, extra in [
            ("ps_6_6", "ps", "ps-dxc", []), ("vs_6_6", "vs", "vs-flat", []),
            ("vs_6_6", "vs", "vs-perspective", ["-D", "VARIABLE_W=1"])]:
            subprocess.run([dxc, "-E", entry, "-T", profile, "-validator-version", "1.6", "-Qstrip_debug",
                            "-Fo", work/(name + ".dxbc"), *extra, HERE/"packed.hlsl"], cwd=work, check=True)
        for tree, label in [(args.base.resolve(), "dedup"), (args.candidate.resolve(), "repeat")]:
            emit(tree, label, work)
        verify_outputs(work)
        harness = work/"readback.exe"
        subprocess.run(["cl.exe", "/nologo", "/EHsc", "/std:c++17", "/O1", "/W3",
                        "/Fe:"+str(harness), str(HERE/"render.cpp"), "/link", "d3d12.lib", "dxgi.lib"],
                       cwd=work, check=True)
        validate(harness, work, dxc.parent, False)
        if args.legacy_bin:
            validate(harness, work, args.legacy_bin.resolve(), True)
        if not args.prepare_only:
            for adapter in ("hardware", "warp"):
                subprocess.run([harness, adapter], cwd=work, check=True)


if __name__ == "__main__":
    main()
