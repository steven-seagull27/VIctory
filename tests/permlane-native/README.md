# Native control for the cross-row shuffle

DXC compiles the HLSL directly to Shader Model 6.6. VIctory's D3D12 test host dispatches one 128-thread workgroup with the same selectors and alternating EXEC bits used in [#1737](https://github.com/Marvisak/VIctory/pull/1737). Each thread records the computed source, the ballot-selected source and the result of reading each. The check compares both reads where the source and destination are enabled, and verifies that the hardware ballot marked every lane active.

On Windows with Python 3, the repository's Rust toolchain and x64 DXC on PATH, run from the checkout root. Supply `--dxc C:\path\to\dxc.exe` if needed.

```powershell
python tests/permlane-native/run.py
python tests/permlane-native/run.py --warp
python tests/permlane-native/run.py --warp-dll C:\path\to\x64\d3d10warp.dll
```

Each invocation prints the adapter, supported wave range and, for WARP, the DLL actually loaded. It tests both Wave32 and Wave64 where supported at `-O3` and `-Od`. An unsupported wave size prints a skip. A difference between the two enabled reads makes the test fail after reporting every supported case.

On the RTX 5080, lane 4 read `0x1234001e` with both source expressions. With x64 Microsoft WARP 1.0.22, lane 4 reported source 30 for both expressions but read `0x1234000e` with the ballot-selected index and `0x1234001e` with the direct index. Built-in WARP on the test PC only supported wave size 4.
