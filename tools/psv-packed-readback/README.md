# Packed pixel inputs on VIctory and DirectX Shader Compiler

This Windows readback exercises the PSV0 writer change in [VIctory PR #3323](https://github.com/Marvisak/VIctory/pull/3323). It generates all shader instructions, geometry and gradients locally. The readback runs on the first hardware DXGI adapter and Windows' built-in WARP and prints the actual adapter identities.

The guest pixel shader contains four pairs of GFX10 interpolation instructions, a colour export and an end instruction. It reads `TEXCOORD4.y`, `TEXCOORD1` and `TEXCOORD7` from different columns of register 0 using perspective interpolation. Register 1 carries `TEXCOORD9` with no perspective correction. The reference pixel shader reads the same four inputs.

## Run on Windows

Use an x64 Visual Studio Native Tools prompt with Python 3, Rust, Git, the C++ toolchain and a recent Windows SDK. Start in this fixture branch's checkout. The following commands fetch the reviewed PR revision into a second worktree.

```powershell
git fetch https://github.com/Marvisak/VIctory.git refs/pull/3323/head
git worktree add --detach ..\victory-3323-candidate ba7f9e61ebff5162fa61371862274cbbc389909f
python tools\psv-packed-readback\run.py --candidate ..\victory-3323-candidate --dxc "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe"
```

Pass `--prepare-only` for compilation and CPU validation. Pass `--legacy-bin PATH` to also test the x64 DLLs from [Microsoft DXC 1.7.2308](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.7.2308). The harness expects a modern SDK validator to accept both layouts. The 1.7 validator should reject only the shared-name layout with `0x80aa0013`.

The runner builds each checkout's real translator. It checks that only PSV0 differs between the two resulting containers' chunks and that the repeated name offsets change from `1,1,1,1` to `1,10,19,28`. It then draws a reference, a parent and a PR shader at constant and varying clip W on each adapter. Each draw compares 9,794 covered RGBA32F pixels with analytical interpolation (tolerance `2e-5`) and checks all 2,494 pixels outside the scissor against a sentinel. It also checks the two translated images bit for bit against the DXC reference.
