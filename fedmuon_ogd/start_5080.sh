#!/usr/bin/env bash
# One command for the RTX 5080 machine.
# Default: CUDA self-check, then the real CIFAR-100 smoke. It does not start
# the 10-seed preregistered run in configs/h_absent.json.
# This launcher is a tool check. It is not a research result, and it does not
# turn H-absent into a paper claim. This script does not request root and does
# not change the system driver.
#
#   git clone --branch cursor/fedmuon-ogd-bfdd --single-branch https://github.com/DreamSoul-AI/model-merging-for-fl.git && cd model-merging-for-fl && bash fedmuon_ogd/start_5080.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SMOKE_CONFIG="$REPO_ROOT/fedmuon_ogd/configs/h_absent_smoke.json"
VENV="$REPO_ROOT/fedmuon_ogd/.venv"
LOG_DIR="$REPO_ROOT/fedmuon_ogd/hypothesis_logs/start_5080"
SMOKE_OUT="$REPO_ROOT/fedmuon_ogd/hypothesis_logs/smoke"
CIFAR_ROOT="${CIFAR100_ROOT:-$REPO_ROOT/fedmuon_ogd/data/cifar100}"
CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR_TGZ_MD5="eb9058c3a382ffc7106e4002c42a8d85"
CIFAR_TRAIN_MD5="16019d7e3df5f24257cddd939b257f8d"
CIFAR_TEST_MD5="f0ef6b0ae62326f3e7ffdfab6717acfc"
# Official CUDA 12.8 wheels. Pin checked against
# https://pytorch.org/get-started/previous-versions/
TORCH_SPEC="torch==2.9.1"
VISION_SPEC="torchvision==0.24.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

die() {
  echo "启动失败：$*" >&2
  exit 1
}

file_md5() {
  md5sum "$1" | awk '{print $1}'
}

decide_python_action() {
  local venv_ok="$1"
  local current_ok="$2"
  if [[ "$venv_ok" == yes ]]; then
    printf 'use-venv\n'
  elif [[ "$current_ok" == yes ]]; then
    printf 'use-current\n'
  else
    printf 'install\n'
  fi
}

cuda_works() {
  local py="$1"
  "$py" - <<'PY'
import sys
try:
    import torch
    import torchvision
except Exception:
    sys.exit(1)
if not torch.cuda.is_available():
    sys.exit(1)
try:
    value = (torch.ones(4, device="cuda") + torch.ones(4, device="cuda")).sum()
    torch.cuda.synchronize()
except Exception:
    sys.exit(1)
if float(value) != 8.0:
    sys.exit(1)
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.get_device_name(0))
PY
}

cifar_files_ok() {
  local root="$1"
  local train="$root/cifar-100-python/train"
  local test="$root/cifar-100-python/test"
  [[ -f "$train" && -f "$test" ]] || return 1
  [[ "$(file_md5 "$train")" == "$CIFAR_TRAIN_MD5" ]] || return 1
  [[ "$(file_md5 "$test")" == "$CIFAR_TEST_MD5" ]] || return 1
}

checkpoint_decision() {
  local py="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    printf 'fresh\n'
    return 0
  fi
  local phase
  phase=$("$py" -c 'import sys, torch; print(torch.load(sys.argv[1], weights_only=False)["phase"])' "$path")
  if [[ "$phase" == "done" ]]; then
    printf 'done\n'
  else
    printf 'resume\n'
  fi
}

assert_smoke_config() {
  python3 - "$SMOKE_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if path.name != "h_absent_smoke.json":
    raise SystemExit(f"refusing {path.name}; this launcher only runs the smoke config")
config = json.loads(path.read_text())
if config.get("is_new_method", True) or config.get("paper_claim", True) or config.get("status") != "tool_smoke_not_a_result":
    raise SystemExit("smoke config must stay a tool check, not a paper claim")
if config.get("seeds") != [0] or int(config["clients"]) > 4 or int(config["rounds_per_task"]) > 1 or int(config["num_classes"]) > 20:
    raise SystemExit("refusing a config larger than the CIFAR-100 smoke")
print(config["hypothesis_id"])
PY
}

prepare_cifar() {
  local root="$1"
  if cifar_files_ok "$root"; then
    echo "CIFAR-100 已通过校验：$root"
    return 0
  fi
  mkdir -p "$root"
  local archive="$root/cifar-100-python.tar.gz"
  echo "下载 CIFAR-100 到 $archive"
  curl -fL --retry 3 --retry-delay 2 -o "$archive.partial" "$CIFAR_URL"
  mv "$archive.partial" "$archive"
  local found
  found=$(file_md5 "$archive")
  if [[ "$found" != "$CIFAR_TGZ_MD5" ]]; then
    rm -f "$archive"
    die "CIFAR-100 压缩包 md5 是 $found，不是 $CIFAR_TGZ_MD5"
  fi
  tar -xzf "$archive" -C "$root"
  cifar_files_ok "$root" || die "解压后的 CIFAR-100 train/test 校验失败"
  echo "CIFAR-100 校验通过：$root"
}

install_cuda_torch() {
  local venv="$1"
  if [[ ! -x "$venv/bin/python" ]]; then
    python3 -m venv "$venv"
  fi
  echo "当前 Python 不能执行 CUDA。在项目虚拟环境安装 $TORCH_SPEC $VISION_SPEC（cu128）。不请求 root，不改驱动。" >&2
  "$venv/bin/python" -m pip install --upgrade pip >&2
  "$venv/bin/python" -m pip install "$TORCH_SPEC" "$VISION_SPEC" --index-url "$TORCH_INDEX" >&2
}

choose_python() {
  local current="${1:-python3}"
  local venv_py="$VENV/bin/python"
  local venv_ok=no
  local current_ok=no
  if [[ -x "$venv_py" ]] && cuda_works "$venv_py" >/dev/null; then
    venv_ok=yes
  fi
  if cuda_works "$current" >/dev/null; then
    current_ok=yes
  fi
  local action
  action=$(decide_python_action "$venv_ok" "$current_ok")
  echo "Python 选择：$action" >&2
  case "$action" in
    use-venv) printf '%s\n' "$venv_py" ;;
    use-current) printf '%s\n' "$current" ;;
    install)
      install_cuda_torch "$VENV"
      cuda_works "$venv_py" >/dev/null || die "安装后仍不能在 CUDA 上执行张量运算。驱动未改。请在 5080 上检查 nvidia-smi 后重跑本脚本。"
      printf '%s\n' "$venv_py"
      ;;
    *) die "未知的 Python 选择：$action" ;;
  esac
}

build_smoke_cmd() {
  local -n _out=$1
  local py="$2"
  local arm="$3"
  local seed="$4"
  local mode="$5"
  _out=(
    "$py" "$REPO_ROOT/fedmuon_ogd/hypothesis_entry.py"
    --config "$SMOKE_CONFIG"
    --arm "$arm"
    --seed "$seed"
    --device cuda
    --out "$SMOKE_OUT"
  )
  if [[ "$mode" == "resume" ]]; then
    _out+=(--resume)
  elif [[ "$mode" != "fresh" ]]; then
    die "不能用模式 $mode 启动训练"
  fi
}

run_smoke_command() {
  local py="$1"
  local arm="$2"
  local seed="$3"
  local mode="$4"
  local -a cmd=()
  build_smoke_cmd cmd "$py" "$arm" "$seed" "$mode"
  if [[ "$mode" == "resume" ]]; then
    echo "已有检查点，不覆盖，恢复 arm=$arm seed=$seed"
  else
    echo "没有检查点，开始 arm=$arm seed=$seed"
  fi
  printf '命令：'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

main() {
  cd "$REPO_ROOT"
  mkdir -p "$LOG_DIR"
  echo "H-absent 只是工具诊断，不是论文主张。本脚本只跑自检和 CIFAR-100 smoke，不启动正式实验。"
  assert_smoke_config >/dev/null
  local py
  py=$(choose_python python3)
  echo "使用 Python：$py"
  cuda_works "$py"
  prepare_cifar "$CIFAR_ROOT"
  export CIFAR100_ROOT="$CIFAR_ROOT"
  local config_sha
  config_sha=$("$py" -c 'import sys; sys.path.insert(0, "fedmuon_ogd"); import hypothesis_entry as entry; print(entry.config_digest(entry.load_config(sys.argv[1])))' "$SMOKE_CONFIG")
  {
    echo "config=$SMOKE_CONFIG"
    echo "config_sha256=$config_sha"
    echo "cifar_root=$CIFAR100_ROOT"
    echo "python=$py"
    echo "hypothesis_status=unverified"
    echo "is_new_method=false"
    "$py" -c 'import json,platform,torch,torchvision; print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L || true
    else
      echo "nvidia-smi 不在 PATH 中。驱动未被修改。"
    fi
    git -C "$REPO_ROOT" rev-parse HEAD || true
  } | tee "$LOG_DIR/environment.txt"
  echo "自检：--device cuda。检查点比较仍在 CPU 上；CUDA 探针只有实际跑了一步才可能把 executed_on_rtx_5080 标为 true。"
  "$py" "$REPO_ROOT/fedmuon_ogd/hypothesis_entry.py" --self-check --device cuda --out "$LOG_DIR/self_check_out"
  local arm seed=0 prefix="${config_sha:0:12}"
  while IFS= read -r arm; do
    [[ -n "$arm" ]] || continue
    local ckpt="$SMOKE_OUT/$prefix/$arm/seed${seed}/checkpoint.pt"
    local decision
    decision=$(checkpoint_decision "$py" "$ckpt")
    echo "arm=$arm seed=$seed 检查点决定：$decision 路径=$ckpt"
    case "$decision" in
      fresh) run_smoke_command "$py" "$arm" "$seed" fresh ;;
      resume) run_smoke_command "$py" "$arm" "$seed" resume ;;
      done) echo "检查点 phase=done。arm=$arm seed=$seed 已经跑完，不重新开始，也不覆盖。" ;;
      *) die "无法识别检查点状态：$decision" ;;
    esac
  done < <("$py" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["arms"]))' "$SMOKE_CONFIG")
  echo "smoke 结束。这只是工具检查，不是研究成果。日志在 $LOG_DIR 和 $SMOKE_OUT。"
}

launch_main() {
  mkdir -p "$LOG_DIR"
  # The parent disables errexit only so it can read the pipeline status.
  # main itself must run in a subshell that turns errexit back on; otherwise
  # a failure inside main is ignored and the launcher can exit 0.
  set +e
  (set -e; main "$@") 2>&1 | tee -a "$LOG_DIR/launcher.log"
  local status=${PIPESTATUS[0]}
  return "$status"
}

if [[ "${START_5080_SOURCE_ONLY:-0}" != "1" ]]; then
  launch_main "$@"
  exit $?
fi
