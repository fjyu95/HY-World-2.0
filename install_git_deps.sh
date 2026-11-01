#!/usr/bin/env bash
# Install git-based dependencies from requirements_git.txt one by one.
# - Already-installed packages are skipped (FORCE=1 to reinstall)
# - A failure on one package does not block the rest
#
# Usage:
#   conda activate hyworld2
#   bash install_git_deps.sh          # skip installed
#   FORCE=1 bash install_git_deps.sh  # reinstall everything

set -uo pipefail
cd "$(dirname "$0")"

export MAX_JOBS="$(nproc)"

echo "== MAX_JOBS=$MAX_JOBS"
echo "== pip: $(command -v pip)"
python -c "import torch; print('== torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || true

# --- proxy / connectivity check ---
echo "== proxy env:"
env | grep -i proxy || echo "   (none)"
# verbose probe of the actual install target; shows whether the
# connection goes direct or through CONNECT <proxy>
echo "== curl -v https://github.com:"
curl -v --connect-timeout 5 --max-time 15 -o /dev/null https://github.com 2>&1 | \
    grep -E "Trying|Connected to|CONNECT|HTTP/" | head -8

DONE=()
SKIPPED=()
FAILED=()

while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue

    # derive package name from URL basename (strip .git / @ref)
    name="$(basename "${line%%@*}" .git | tr 'A-Z' 'a-z' | tr '_' '-')"
    if [[ "${FORCE:-0}" != "1" ]] && pip show "$name" >/dev/null 2>&1; then
        SKIPPED+=("$name")
        echo "== SKIP (already installed): $name"
        continue
    fi

    echo
    echo "============================================================"
    echo ">>> $line"
    echo "============================================================"
    if pip install -v --no-build-isolation --progress-bar on "$line"; then
        DONE+=("$name")
    else
        FAILED+=("$line")
    fi
done < requirements_git.txt

echo
echo "================ SUMMARY ================"
printf 'OK:      %d   ' "${#DONE[@]}";  printf '%s\n' "${DONE[@]:-}"
printf 'SKIPPED: %d   ' "${#SKIPPED[@]}"; printf '%s\n' "${SKIPPED[@]:-}"
printf 'FAILED:  %d\n' "${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "  - $f"; done

echo
echo "================ INSTALLED CHECK ================"
while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    name="$(basename "${line%%@*}" .git | tr 'A-Z' 'a-z' | tr '_' '-')"
    # hyphenated names may register differently in pip (e.g. fused-ssim vs fused_ssim);
    # match on the last segment to catch all variants
    pat="$name"; [[ "$name" == *-* ]] && pat="${name##*-}"
    pip list 2>/dev/null | grep -i "$pat" || echo "-- NOT INSTALLED: $name"
done < requirements_git.txt

# ============================================================================
# Fallbacks for hard-to-install git deps (e.g. MoGe — git-lfs smudge filter
# crash, or GitHub clone crawling at ~25KiB/s). Decouple download from build:
# fetch to third_party, pin the commit, then `pip install` the local dir.
#
#   THIRDPARTY=/data/yufangjie/code/HY-World-2.0/hyworld2/worldgen/third_party
#
# --- Option 1: tarball single-file download (resumable, no git history) ---
#   cd $THIRDPARTY
#   curl -L -C - -o moge-0286b49.tar.gz \
#       https://github.com/microsoft/MoGe/archive/0286b495230a074aadf1c76cc5c679e943e5d1c6.tar.gz
#   tar xzf moge-0286b49.tar.gz
#   mv MoGe-0286b495230a074aadf1c76cc5c679e943e5d1c6 MoGe
#   pip install ./MoGe --no-build-isolation      # editable: pip install -e ./MoGe
#   # -C - resumes an interrupted download; note: no git history, cannot
#   # checkout other versions afterwards
#
# --- Option 2: clone through clashctl proxy (http://127.0.0.1:7890) ---
#   # MoGe (and similar repos) use Git LFS — check git-lfs is installed,
#   # otherwise checkout crashes with "git-lfs: not found":
#       which git-lfs && git lfs version
#       # if missing, install system-wide:
#       #   sudo apt-get update && sudo apt-get install -y git-lfs
#       #   git lfs install    # register the filters once
#       # no-sudo alternative: conda install -c conda-forge git-lfs -y
#   export https_proxy=http://127.0.0.1:7890
#   cd $THIRDPARTY
#   git clone https://github.com/microsoft/MoGe.git
#   cd MoGe && git checkout 0286b495230a074aadf1c76cc5c679e943e5d1c6
#   pip install . --no-build-isolation
#   # if git-lfs is truly unavailable, fall back to
#   # GIT_LFS_SKIP_SMUDGE=1 git clone ... — LFS-managed files stay as
#   # pointer stubs (fine unless the runtime actually needs them)
# ============================================================================
