import os
import sys
import shutil
import tarfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from best_models import BEST_MODELS

# 输出目录
OUTPUT_DIR = "RL_best_models_package"
ARCHIVE_NAME = "best_models_results.tar.gz"

if os.path.lexists(OUTPUT_DIR):
    print(f"⚠ Removing existing {OUTPUT_DIR}")

    if os.path.islink(OUTPUT_DIR):
        os.unlink(OUTPUT_DIR)
    elif os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    else:
        os.remove(OUTPUT_DIR)

def copy_with_parents(src_path, dst_root):
    """
    Copy file to dst_root while preserving parent directories
    e.g. results/... -> best_models_package/results/...
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"❌ File not found: {src_path}")

    dst_path = os.path.join(dst_root, src_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    print(f"✔ Copied: {src_path}")


def collect_all_paths(cfg):
    paths = []

    def recurse(d):
        if isinstance(d, dict):
            for v in d.values():
                recurse(v)
        elif isinstance(d, list):
            for v in d:
                recurse(v)

    def extract(d):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if k == "model_path" and v is not None:
                paths.append(v)
            if k == "obs_rms_path" and v is not None:
                paths.append(v)
            if isinstance(v, dict):
                extract(v)

    extract(cfg)
    return paths


def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"⚠ Removing existing {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📦 Collecting model files...\n")

    all_paths = collect_all_paths(BEST_MODELS)

    for p in all_paths:
        copy_with_parents(p, OUTPUT_DIR)

    print("\n📦 Creating tar.gz archive...")
    with tarfile.open(ARCHIVE_NAME, "w:gz") as tar:
        tar.add(OUTPUT_DIR, arcname=OUTPUT_DIR)

    print(f"\n✅ Done!")
    print(f"📁 Package directory: {OUTPUT_DIR}")
    print(f"📦 Archive: {ARCHIVE_NAME}")


if __name__ == "__main__":
    main()
