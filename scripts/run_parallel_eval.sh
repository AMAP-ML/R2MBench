#!/bin/bash
# ============================================================
# Parallel one-pass NMR evaluation.
#
# Spreads a task's videos across `batch` GPUs x `shard` processes-per-GPU
# (batch*shard total processes), each process running eval_gpu_metrics.py which
# computes ALL requested GPU families (appearance/scene_identity/geometric/object)
# in a SINGLE per-pair pass. The VLM family (persistent_state) is launched
# immediately in the background, fully parallel with the GPU work (API/CPU only).
#
# Tasks are processed serially (each task saturates batch*shard processes).
#
# Usage (run from the full_eval repo root):
#   bash scripts/run_parallel_eval.sh --config run_config.local.conf \
#       --metrics appearance,scene_identity,geometric,object \
#       --gpus 0,1,2,3,4,5,6,7 --batch 8 --shard 10 --tasks 'modelA_*'
# ============================================================
set -euo pipefail

# SCRIPT_DIR = full_eval repo root (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/run_config.local.conf"

# ---- Defaults ----
METRICS_ARG="all"
GPU_LIST="0,1,2,3,4,5,6,7"
TASK_FILTER=""
BATCH=""          # number of GPUs to use (default: all in --gpus)
SHARD=10          # processes per GPU
MAX_EVAL_PAIRS=100
MAX_VLM_JOBS=3    # rolling concurrency for the VLM family

CLI_METRICS=""; CLI_GPUS=""; CLI_TASKS=""; CLI_BATCH=""; CLI_SHARD=""

declare -a CONFIG_TASK_NAMES=() CONFIG_VIDEO_DIRS=() CONFIG_POSE_JSONS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIG_FILE="$2"; shift 2 ;;
        --metrics) CLI_METRICS="$2"; shift 2 ;;
        --gpus)    CLI_GPUS="$2"; shift 2 ;;
        --tasks)   CLI_TASKS="$2"; shift 2 ;;
        --batch)   CLI_BATCH="$2"; shift 2 ;;
        --shard)   CLI_SHARD="$2"; shift 2 ;;
        --max_eval_pairs) MAX_EVAL_PAIRS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash scripts/run_parallel_eval.sh --config <f> [--metrics ...|all] [--gpus 0,1,..] [--batch N] [--shard M] [--tasks pat] [--max_eval_pairs N]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Config parsing (same format as run_full_eval.sh) ----
expand_config_value() {
    local value="$1" previous=""
    value="${value//\$\{SCRIPT_DIR\}/${SCRIPT_DIR}}"
    value="${value//\$\{HOME\}/${HOME}}"
    while [[ "${value}" != "${previous}" ]]; do
        previous="${value}"
        value="${value//\$\{MEM_BENCH_THIRDPARTY_ROOT\}/${MEM_BENCH_THIRDPARTY_ROOT:-}}"
        value="${value//\$\{CHECKPOINT_ROOT\}/${CHECKPOINT_ROOT:-}}"
        value="${value//\$\{TRAJECTORY_DIR\}/${TRAJECTORY_DIR:-}}"
        value="${value//\$\{MODEL_NAME\}/${MODEL_NAME:-}}"
        value="${value//\$\{BOQ_ROOT\}/${BOQ_ROOT:-}}"
        value="${value//\$\{MUTUALVPR_ROOT\}/${MUTUALVPR_ROOT:-}}"
        value="${value//\$\{LIGHTGLUE_ROOT\}/${LIGHTGLUE_ROOT:-}}"
        value="${value//\$\{GROUNDINGDINO_ROOT\}/${GROUNDINGDINO_ROOT:-}}"
        value="${value//\$\{SAM2_ROOT\}/${SAM2_ROOT:-}}"
    done
    printf '%s' "${value}"
}

set_config_var() {
    local key="$1" value="$2"
    case "${key}" in
        METRICS_ARG|GPU_LIST|TASK_FILTER|NUM_VIDEO_SHARDS|WORLD_SCORE_ENV|GEMINI_PYTHON|TORCH_HOME|HF_ENDPOINT|USE_TF|USE_TORCH|TRANSFORMERS_NO_TF|MEM_BENCH_THIRDPARTY_ROOT|BOQ_ROOT|MUTUALVPR_ROOT|LIGHTGLUE_ROOT|GROUNDINGDINO_ROOT|SAM2_ROOT|CHECKPOINT_ROOT|CLIP_CKPT|BOQ_CHECKPOINT|MUTUALVPR_CHECKPOINT|GROUNDINGDINO_CONFIG|GROUNDINGDINO_CKPT|SAM2_CONFIG|SAM2_CKPT|API_KEY|API_URL|NMR_RESULT_BASE|MODEL_NAME|TRAJECTORY_DIR)
            printf -v "${key}" '%s' "${value}"; export "${key}" ;;
        *) echo "ERROR: unknown config key '${key}' in ${CONFIG_FILE}"; exit 1 ;;
    esac
}

load_plain_config() {
    local config_file="$1"
    [[ -f "${config_file}" ]] || { echo "ERROR: config not found: ${config_file}"; exit 1; }
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%$'\r'}"
        line="$(echo "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -z "${line}" || "${line}" == \#* ]] && continue
        if [[ "${line}" == TASK=* ]]; then
            local task_value name video_dir pose_json
            task_value="$(expand_config_value "${line#TASK=}")"
            IFS='|' read -r name video_dir pose_json <<< "${task_value}"
            [[ -n "${name}" && -n "${video_dir}" && -n "${pose_json}" ]] || { echo "ERROR: bad TASK line: ${line}"; exit 1; }
            CONFIG_TASK_NAMES+=("${name}"); CONFIG_VIDEO_DIRS+=("${video_dir}"); CONFIG_POSE_JSONS+=("${pose_json}")
            continue
        fi
        [[ "${line}" == *=* ]] || { echo "ERROR: bad config line: ${line}"; exit 1; }
        local key value
        key="$(echo "${line%%=*}" | xargs)"
        value="$(expand_config_value "${line#*=}")"
        set_config_var "${key}" "${value}"
    done < "${config_file}"
}

load_plain_config "${CONFIG_FILE}"

[[ -n "${CLI_METRICS}" ]] && METRICS_ARG="${CLI_METRICS}"
[[ -n "${CLI_GPUS}" ]] && GPU_LIST="${CLI_GPUS}"
[[ -n "${CLI_TASKS}" ]] && TASK_FILTER="${CLI_TASKS}"
[[ -n "${CLI_BATCH}" ]] && BATCH="${CLI_BATCH}"
[[ -n "${CLI_SHARD}" ]] && SHARD="${CLI_SHARD}"

# ---- Normalize metrics ----
normalize_metric() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]' | xargs)" in
        appearance|appear|pixel) echo "appearance" ;;
        scene_identity|scene|identity) echo "scene_identity" ;;
        geometric|geometry|geo|keypoint) echo "geometric" ;;
        object|objects) echo "object" ;;
        persistent_state|state|gemini|persistent) echo "persistent_state" ;;
        *) echo "" ;;
    esac
}
declare -a METRICS=()
_seen_all=false
IFS=',' read -ra _raw <<< "${METRICS_ARG}"
for _t in "${_raw[@]}"; do
    _t="$(echo "${_t}" | xargs)"; [[ -z "${_t}" ]] && continue
    [[ "$(echo "${_t}" | tr '[:upper:]' '[:lower:]')" == "all" ]] && { _seen_all=true; continue; }
    _c="$(normalize_metric "${_t}")"
    [[ -z "${_c}" ]] && { echo "ERROR: unknown metric '${_t}'"; exit 1; }
    METRICS+=("${_c}")
done
${_seen_all} && METRICS=(appearance scene_identity geometric object persistent_state)
declare -a _uniq=()
for _m in "${METRICS[@]:-}"; do
    _dup=false; for _u in "${_uniq[@]:-}"; do [[ "${_u}" == "${_m}" ]] && _dup=true && break; done
    ${_dup} || _uniq+=("${_m}")
done
METRICS=("${_uniq[@]}")
has_metric() { local t="$1"; for _m in "${METRICS[@]}"; do [[ "${_m}" == "${t}" ]] && return 0; done; return 1; }

# GPU families -> single comma list for eval_gpu_metrics --families
declare -a GPU_FAMS=()
for f in appearance scene_identity geometric object; do has_metric "${f}" && GPU_FAMS+=("${f}"); done
GPU_FAMS_CSV="$(IFS=,; echo "${GPU_FAMS[*]:-}")"

# ---- Environment (robust activation; pin PYTHON to env's interpreter) ----
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh" ]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
fi
conda activate "${WORLD_SCORE_ENV}" 2>/dev/null || source activate "${WORLD_SCORE_ENV}" 2>/dev/null || true
if [ -x "${WORLD_SCORE_ENV}/bin/python" ]; then
    export PATH="${WORLD_SCORE_ENV}/bin:${PATH}"; PYTHON="${WORLD_SCORE_ENV}/bin/python"
else
    PYTHON=python
fi
export PYTHONPATH="${GROUNDINGDINO_ROOT}:${SAM2_ROOT}:${MEM_BENCH_THIRDPARTY_ROOT:-${SCRIPT_DIR}/third_party}:${SCRIPT_DIR}/src:${PYTHONPATH:-}"

PAIRS_DIR="${SCRIPT_DIR}/pairs"; LOG_DIR="${SCRIPT_DIR}/logs"
RESULT_ROOT="${NMR_RESULT_BASE:-${SCRIPT_DIR}/results/nmr}/unified"
mkdir -p "${PAIRS_DIR}" "${LOG_DIR}" "${RESULT_ROOT}"

# ---- Select tasks (config TASK lines; --tasks glob filter) ----
declare -a TASK_NAMES=() VIDEO_DIRS=() POSE_JSONS=()
append_task() {
    local name="$1" vdir="$2" pose="$3"
    if [[ -n "${TASK_FILTER}" ]]; then
        local matched=false pat
        IFS=',' read -ra PATS <<< "${TASK_FILTER}"
        for pat in "${PATS[@]}"; do pat="$(echo "${pat}" | xargs)"; case "${name}" in ${pat}) matched=true; break ;; esac; done
        ${matched} || return 0
    fi
    TASK_NAMES+=("${name}"); VIDEO_DIRS+=("${vdir}"); POSE_JSONS+=("${pose}")
}
[[ ${#CONFIG_TASK_NAMES[@]} -gt 0 ]] || { echo "ERROR: no TASK= lines in ${CONFIG_FILE}"; exit 1; }
for i in "${!CONFIG_TASK_NAMES[@]}"; do
    append_task "${CONFIG_TASK_NAMES[$i]}" "${CONFIG_VIDEO_DIRS[$i]}" "${CONFIG_POSE_JSONS[$i]}"
done
[[ ${#TASK_NAMES[@]} -gt 0 ]] || { echo "ERROR: no tasks matched --tasks '${TASK_FILTER}'"; exit 1; }

# ---- GPUs / batch / shard ----
IFS=',' read -ra GPU_POOL <<< "${GPU_LIST}"
[[ -z "${BATCH}" ]] && BATCH=${#GPU_POOL[@]}
(( BATCH > ${#GPU_POOL[@]} )) && BATCH=${#GPU_POOL[@]}
declare -a GPUS=("${GPU_POOL[@]:0:${BATCH}}")
TOTAL=$(( BATCH * SHARD ))

echo "============================================================"
echo "Parallel NMR Evaluation"
echo "============================================================"
echo "Tasks:        ${#TASK_NAMES[@]}   ${TASK_NAMES[*]}"
echo "Metrics:      ${METRICS[*]}"
echo "GPU families: ${GPU_FAMS_CSV:-<none>}"
echo "GPUs used:    ${GPUS[*]}  (batch=${BATCH})"
echo "Shard/GPU:    ${SHARD}   -> ${TOTAL} procs per task"
echo "Max pairs:    ${MAX_EVAL_PAIRS}"
echo "============================================================"

# ---- Pre-sample shared pairs (once) ----
TASKS_RUNTIME_CONF="${PAIRS_DIR}/selected_tasks.conf"
: > "${TASKS_RUNTIME_CONF}"
for i in "${!TASK_NAMES[@]}"; do
    printf '%s|%s|%s\n' "${TASK_NAMES[$i]}" "${VIDEO_DIRS[$i]}" "${POSE_JSONS[$i]}" >> "${TASKS_RUNTIME_CONF}"
done
echo "[pre-sample] Sampling shared pairs..."
${PYTHON} -u "${SCRIPT_DIR}/src/sample_pairs.py" \
    --tasks_conf "${TASKS_RUNTIME_CONF}" --output_dir "${PAIRS_DIR}" \
    --max_eval_pairs 100 --seed 42 > "${LOG_DIR}/sample_pairs.log" 2>&1
echo "[pre-sample] Done."

get_pairs_json() {
    local pose_json="$1" key
    key=$(${PYTHON} -c "
import hashlib, sys; from pathlib import Path
pj=sys.argv[1]; print(f'{Path(pj).stem}_{hashlib.md5(pj.encode()).hexdigest()[:8]}')" "${pose_json}")
    local f="${PAIRS_DIR}/${key}.json"
    [[ -f "${f}" ]] && echo "${f}" || echo ""
}

# ============================================================
# VLM (persistent_state): launch immediately, parallel with GPU work
# ============================================================
declare -a VLM_PIDS=()
run_vlm_background() {
    local ENGINE=gemini-3.1-pro-preview MAX_PAIRS=10 NUM_WORKERS=3
    local result_base="${NMR_RESULT_BASE}/persistent_state"; mkdir -p "${result_base}"
    echo "[vlm] launching persistent_state (parallel, API) ..."
    for i in "${!TASK_NAMES[@]}"; do
        local task_name="${TASK_NAMES[$i]}" video_dir="${VIDEO_DIRS[$i]}" pose_json="${POSE_JSONS[$i]}"
        [[ "${task_name}" == forcing_* ]] && { echo "  [vlm skip] ${task_name}"; continue; }
        while [[ "$(jobs -rp | wc -l)" -ge "${MAX_VLM_JOBS}" ]]; do sleep 5; done
        local pairs_json extra=""; pairs_json="$(get_pairs_json "${pose_json}")"
        [[ -n "${pairs_json}" ]] && extra="--pairs_json ${pairs_json}"
        echo "  [vlm launch] ${task_name}"
        ${GEMINI_PYTHON} -u "${SCRIPT_DIR}/src/metrics/persistent_state.py" \
            --video_dir "${video_dir}" --pose_json "${pose_json}" \
            --output_json "${result_base}/results_gemini_nmr_${task_name}.json" \
            --engine "${ENGINE}" --max_pairs_per_type "${MAX_PAIRS}" \
            --num_workers "${NUM_WORKERS}" --resume ${extra} \
            > "${LOG_DIR}/vlm_${task_name}.log" 2>&1 &
        VLM_PIDS+=($!)
    done
}

if has_metric persistent_state; then
    run_vlm_background
fi

# ============================================================
# GPU families: one unified pass, batch x shard video-sharded, tasks serial
# ============================================================
run_gpu_task() {
    local task_name="$1" video_dir="$2" pose_json="$3"
    local out_base="${RESULT_ROOT}/${task_name}"
    local pairs_json extra_pairs=""; pairs_json="$(get_pairs_json "${pose_json}")"
    [[ -n "${pairs_json}" ]] && extra_pairs="--pairs_json ${pairs_json}"

    local -a obj_args=()
    if has_metric object; then
        obj_args=(--groundingdino_config "${GROUNDINGDINO_CONFIG}" \
                  --groundingdino_checkpoint "${GROUNDINGDINO_CKPT}" \
                  --sam2_config "${SAM2_CONFIG}" --sam_checkpoint "${SAM2_CKPT}")
    fi
    local lpips_arg=""; has_metric appearance && lpips_arg="--enable_lpips"

    echo ""
    echo "[gpu] ${task_name}: ${TOTAL} procs (${BATCH} GPUs x ${SHARD}) families=${GPU_FAMS_CSV}"
    local -a pids=()
    for (( k = 0; k < TOTAL; k++ )); do
        local gpu="${GPUS[$(( k % BATCH ))]}"
        local out="${out_base}/shard_${k}"; mkdir -p "${out}"
        local log="${LOG_DIR}/gpu_${task_name}_shard${k}.log"
        (
            CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} -u "${SCRIPT_DIR}/src/eval_gpu_metrics.py" \
                --video_dir "${video_dir}" --pose_json "${pose_json}" \
                --output_dir "${out}" --device cuda:0 \
                --families "${GPU_FAMS_CSV}" ${lpips_arg} \
                --clip_checkpoint "${CLIP_CKPT}" \
                --max_eval_pairs "${MAX_EVAL_PAIRS}" --resume ${extra_pairs} \
                --num_video_shards "${TOTAL}" --video_shard_id "${k}" \
                "${obj_args[@]}" > "${log}" 2>&1
        ) &
        pids+=($!)
    done
    echo "[gpu] ${task_name}: launched ${#pids[@]} procs, waiting..."
    local failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || failed=$((failed+1)); done
    echo "[gpu] ${task_name}: done. failed procs: ${failed}/${#pids[@]}"
}

if [[ -n "${GPU_FAMS_CSV}" ]]; then
    for i in "${!TASK_NAMES[@]}"; do
        run_gpu_task "${TASK_NAMES[$i]}" "${VIDEO_DIRS[$i]}" "${POSE_JSONS[$i]}"
    done
fi

# ---- Wait for VLM background jobs ----
if [[ ${#VLM_PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "[vlm] waiting for ${#VLM_PIDS[@]} persistent_state tasks..."
    vfailed=0
    for pid in "${VLM_PIDS[@]}"; do wait "${pid}" || vfailed=$((vfailed+1)); done
    echo "[vlm] done. failed: ${vfailed}/${#VLM_PIDS[@]}"
fi

echo ""
echo "============================================================"
echo "All done."
[[ -n "${GPU_FAMS_CSV}" ]] && echo "GPU results:  ${RESULT_ROOT}/<task>/shard_*/results.json"
has_metric persistent_state && echo "VLM results:  ${NMR_RESULT_BASE}/persistent_state/"
echo "Logs:         ${LOG_DIR}"
echo "============================================================"
