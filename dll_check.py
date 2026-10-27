"""
dll_check.py  ─  找出 c10.dll 缺少的依賴
在 .venv 環境中執行：python dll_check.py
"""
import subprocess, sys, os, ctypes, ctypes.util, platform

print("=" * 55)
print("  c10.dll 依賴診斷")
print("=" * 55)

# 1. 系統基本資訊
print(f"\n[1] Windows 版本 : {platform.version()}")
print(f"    Python       : {sys.version.split()[0]}")

# 2. NVIDIA 驅動版本（最關鍵）
print("\n[2] NVIDIA 驅動版本...")
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name",
                        "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=5)
    if r.returncode == 0:
        driver, gpu = r.stdout.strip().split(", ", 1)
        print(f"    GPU    : {gpu}")
        print(f"    Driver : {driver}")
        d_parts = driver.split(".")
        d_major = int(d_parts[0])
        # CUDA 12.8 需要 Windows driver >= 528.33
        # RTX 5080 Blackwell 建議 >= 572.xx
        if d_major >= 572:
            print(f"    ✓ 驅動版本足夠（>= 572.xx）")
        elif d_major >= 528:
            print(f"    ⚠ 驅動可能過舊，建議更新至 572.xx 以上")
        else:
            print(f"    ✗ 驅動版本不足！請更新：https://www.nvidia.com/drivers")
    else:
        print("    ✗ nvidia-smi 執行失敗")
except Exception as e:
    print(f"    ✗ {e}")

# 3. 關鍵 DLL 是否可被找到
print("\n[3] 關鍵系統 DLL 可用性...")
dlls_to_check = [
    ("vcruntime140.dll",    "Visual C++ 2015-2022 Runtime"),
    ("vcruntime140_1.dll",  "Visual C++ 2015-2022 Runtime (v2)"),
    ("msvcp140.dll",        "Visual C++ C++ Runtime"),
    ("nvcuda.dll",          "NVIDIA CUDA Driver"),
    ("cudart64_12.dll",     "CUDA 12.x Runtime"),
    ("cudart64_128.dll",    "CUDA 12.8 Runtime (system)"),
]
missing = []
for dll, desc in dlls_to_check:
    path = ctypes.util.find_library(dll.replace(".dll",""))
    # 也搜尋 System32
    sys32 = os.path.join(os.environ.get("SystemRoot","C:\\Windows"), "System32", dll)
    found = path or os.path.exists(sys32)
    mark  = "✓" if found else "✗"
    print(f"    {mark} {dll:<28} ({desc})")
    if not found:
        missing.append((dll, desc))

# 4. 嘗試手動載入 torch
print("\n[4] 嘗試載入 torch...")
torch_lib_path = None
for sp in sys.path:
    c = os.path.join(sp, "torch", "lib", "c10.dll")
    if os.path.exists(c):
        torch_lib_path = c
        break

if torch_lib_path:
    print(f"    torch/lib 路徑 : {os.path.dirname(torch_lib_path)}")
    # 列出 torch/lib 中的 DLL（找是否有 cudart）
    lib_dir = os.path.dirname(torch_lib_path)
    bundled = [f for f in os.listdir(lib_dir) if f.endswith(".dll")]
    cuda_bundled = [f for f in bundled if "cudart" in f.lower() or "cublas" in f.lower()]
    print(f"    Bundled CUDA DLLs : {cuda_bundled if cuda_bundled else '(none found)'}")
else:
    print("    torch/lib 路徑未找到")

# 5. 診斷結論與修復建議
print("\n" + "=" * 55)
print("  診斷結論")
print("=" * 55)

if missing:
    vc_missing = [d for d,_ in missing if "vcruntime" in d or "msvcp" in d]
    cuda_missing = [d for d,_ in missing if "cuda" in d.lower()]
    
    if vc_missing:
        print("""
  ✗ 缺少 Visual C++ Runtime！
  → 下載安裝（必須）：
    https://aka.ms/vs/17/release/vc_redist.x64.exe
  → 安裝後重開機，再次執行此腳本
""")
    if cuda_missing:
        print("""
  ✗ 缺少系統層 CUDA Runtime！
  torch 2.11.0+cu128 需要 CUDA 12.8 Toolkit 安裝在系統上。
  → 下載安裝（選 Windows > x86_64 > exe）：
    https://developer.nvidia.com/cuda-12-8-0-download-archive
    （只需安裝 Runtime components，不需要完整 Toolkit）
""")
else:
    print("""
  DLL 都存在，但仍然失敗 → 可能是版本衝突。
  建議改裝更穩定的版本（見下方）
""")

print("""
  ── 已驗證可用的安裝組合 ─────────────────────────────────
  [A] CUDA 12.8 Toolkit 安裝後，重新測試 torch 2.11.0+cu128

  [B] 降版至更穩定的 wheel（不需要額外 Toolkit）：
      pip uninstall torch torchvision torchaudio -y
      pip install torch==2.7.0+cu128 torchvision torchaudio ^
          --index-url https://download.pytorch.org/whl/cu128

  [C] 改用 cu126（若 B 也失敗）：
      pip install torch==2.7.0+cu126 torchvision torchaudio ^
          --index-url https://download.pytorch.org/whl/cu126

  [D] 終極備案 → 用 Colab（RTX 5080 本地問題不影響 Colab）
""")
