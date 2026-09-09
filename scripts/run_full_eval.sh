#!/bin/bash
# ============================================================
# Unified Full Evaluation Script for 30s Benchmarks
#
# All FIVE metric families compute the SAME quantity -- Normalized Memory
# Retention (NMR) -- over shared pre-sampled revisit/baseline/short pairs.
# They differ only in the metric family and the backend:
#
#   family            metrics                         script / backend
#   ----------------  ------------------------------  --------------------------------
#   appearance        PSNR, SSIM, LPIPS               src/eval_revisit_nmr.py  (GPU, sharded)
#   scene_identity    DINOv2, BoQ, MutualVPR          src/eval_revisit_nmr.py  (GPU, sharded)
#   geometric         SuperPoint + LightGlue          src/eval_revisit_nmr.py  (GPU, sharded)
#   object            GroundingDINO + SAM2 + DINO/CLIP src/metrics/object_identity.py (GPU, sharded)
#   persistent_state  VLM structured rubric           src/metrics/persistent_state.py (VLM API)
#
# Each family is scheduled independently. The first three visual families share
# src/eval_revisit_nmr.py as the implementation entry, but each requested family is
# launched as a separate pass with its own --families selector and result folder.
# The --metrics parameter selects which families to run (any subset).
#
# Usage (run from the full_eval repo root):
#   bash scripts/run_full_eval.sh                                    # run all five families
#   bash scripts/run_full_eval.sh --metrics appearance              # only appearance NMR
#   bash scripts/run_full_eval.sh --metrics scene_identity,object   # scene identity + object
#   bash scripts/run_full_eval.sh --metrics geometric,persistent_state
#   bash scripts/run_full_eval.sh --metrics all --gpus 0,1,2,3
#   bash scripts/run_full_eval.sh --metrics object --tasks "lingbot*,mg3*"
#
# Family tokens (case-insensitive, comma-separated); synonyms accepted:
#   appearance        (aliases: appear, pixel)
#   scene_identity    (aliases: scene, identity)
#   geometric         (aliases: geometry, geo, keypoint)
#   object            (alias:   objects)
#   persistent_state  (aliases: state, gemini, persistent)
#   all               -> all five families
# ============================================================

set -euo pipefail

# SCRIPT_DIR = full_eval repo root (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_CONF="${SCRIPT_DIR}/tasks.conf"
CONFIG_FILE="${SCRIPT_DIR}/run_config.conf"

# ---- Default config ----
METRICS_ARG="all"
GPU_LIST="4,5,6,7"
TASK_FILTER=""
NUM_VIDEO_SHARDS=10

# ---- CLI overrides ----
CLI_METRICS_ARG=""
CLI_GPU_LIST=""
CLI_TASK_FILTER=""
CLI_NUM_VIDEO_SHARDS=""

declare -a CONFIG_TASK_NAMES=()
declare -a CONFIG_VIDEO_DIRS=()
declare -a CONFIG_POSE_JSONS=()

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIG_FILE="$2"; shift 2 ;;
        --metrics) CLI_METRICS_ARG="$2"; shift 2 ;;
        --gpus)    CLI_GPU_LIST="$2"; shift 2 ;;
        --tasks)   CLI_TASK_FILTER="$2"; shift 2 ;;
        --shards)  CLI_NUM_VIDEO_SHARDS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash scripts/run_full_eval.sh [--config run_config.local.conf] [--metrics appearance,scene_identity,geometric,object,persistent_state|all] [--gpus 0,1,2,3] [--tasks pattern] [--shards N]"
            echo "  Config format: KEY=VALUE and TASK=name|video_dir|pose_json"
            echo "  Family tokens: appearance (appear/pixel), scene_identity (scene/identity), geometric (geometry/geo/keypoint), object (objects), persistent_state (state/gemini/persistent), all"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

expand_config_value() {
    local value="$1"
    local previous=""

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
    local key="$1"
    local value="$2"

    case "${key}" in
        METRICS_ARG|GPU_LIST|TASK_FILTER|NUM_VIDEO_SHARDS|WORLD_SCORE_ENV|GEMINI_PYTHON|TORCH_HOME|HF_ENDPOINT|USE_TF|USE_TORCH|TRANSFORMERS_NO_TF|MEM_BENCH_THIRDPARTY_ROOT|BOQ_ROOT|MUTUALVPR_ROOT|LIGHTGLUE_ROOT|GROUNDINGDINO_ROOT|SAM2_ROOT|CHECKPOINT_ROOT|CLIP_CKPT|BOQ_CHECKPOINT|MUTUALVPR_CHECKPOINT|GROUNDINGDINO_CONFIG|GROUNDINGDINO_CKPT|SAM2_CONFIG|SAM2_CKPT|API_KEY|API_URL|NMR_RESULT_BASE|MODEL_NAME|TRAJECTORY_DIR)
            printf -v "${key}" '%s' "${value}"
            export "${key}"
            ;;
        *)
            echo "ERROR: unknown config key '${key}' in ${CONFIG_FILE}"
            exit 1
            ;;
    esac
}

load_plain_config() {
    local config_file="$1"
    [[ -f "${config_file}" ]] || return 0

    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%$'\r'}"
        line="$(echo "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -z "${line}" || "${line}" == \#* ]] && continue

        if [[ "${line}" == TASK=* ]]; then
            local task_value name video_dir pose_json
            task_value="$(expand_config_value "${line#TASK=}")"
            IFS='|' read -r name video_dir pose_json <<< "${task_value}"
            if [[ -z "${name}" || -z "${video_dir}" || -z "${pose_json}" ]]; then
                echo "ERROR: invalid TASK line in ${config_file}: ${line}"
                exit 1
            fi
            CONFIG_TASK_NAMES+=("${name}")
            CONFIG_VIDEO_DIRS+=("${video_dir}")
            CONFIG_POSE_JSONS+=("${pose_json}")
            continue
        fi

        if [[ "${line}" != *=* ]]; then
            echo "ERROR: invalid config line in ${config_file}: ${line}"
            exit 1
        fi

        local key value
        key="$(echo "${line%%=*}" | xargs)"
        value="$(expand_config_value "${line#*=}")"
        set_config_var "${key}" "${value}"
    done < "${config_file}"
}

load_plain_config "${CONFIG_FILE}"

[[ -n "${CLI_METRICS_ARG}" ]] && METRICS_ARG="${CLI_METRICS_ARG}"
[[ -n "${CLI_GPU_LIST}" ]] && GPU_LIST="${CLI_GPU_LIST}"
[[ -n "${CLI_TASK_FILTER}" ]] && TASK_FILTER="${CLI_TASK_FILTER}"
[[ -n "${CLI_NUM_VIDEO_SHARDS}" ]] && NUM_VIDEO_SHARDS="${CLI_NUM_VIDEO_SHARDS}"

# ---- Canonicalize the requested metrics (dedup, preserve order) ----
normalize_metric() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]' | xargs)" in
        appearance|appear|pixel)               echo "appearance" ;;
        scene_identity|scene|identity)         echo "scene_identity" ;;
        geometric|geometry|geo|keypoint)       echo "geometric" ;;
        object|objects)                        echo "object" ;;
        persistent_state|state|gemini|persistent) echo "persistent_state" ;;
        *)                                     echo "" ;;
    esac
}

declare -a METRICS=()
_seen_all=false
IFS=',' read -ra _raw_metrics <<< "${METRICS_ARG}"
for _tok in "${_raw_metrics[@]}"; do
    _tok="$(echo "${_tok}" | xargs)"
    [[ -z "${_tok}" ]] && continue
    if [[ "$(echo "${_tok}" | tr '[:upper:]' '[:lower:]')" == "all" ]]; then
        _seen_all=true
        continue
    fi
    _canon="$(normalize_metric "${_tok}")"
    if [[ -z "${_canon}" ]]; then
        echo "ERROR: unknown metric '${_tok}'. Valid: appearance, scene_identity, geometric, object, persistent_state, all."
        exit 1
    fi
    METRICS+=("${_canon}")
done

if ${_seen_all}; then
    METRICS=(appearance scene_identity geometric object persistent_state)
fi

# Dedup while preserving order
declare -a _uniq_metrics=()
for _m in "${METRICS[@]}"; do
    _dup=false
    for _u in "${_uniq_metrics[@]:-}"; do
        [[ "${_u}" == "${_m}" ]] && _dup=true && break
    done
    ${_dup} || _uniq_metrics+=("${_m}")
done
METRICS=("${_uniq_metrics[@]}")

if [[ ${#METRICS[@]} -eq 0 ]]; then
    echo "ERROR: no metrics selected. Use --metrics appearance,scene_identity,geometric,object,persistent_state (or 'all')."
    exit 1
fi

# Convenience predicate
has_metric() {
    local target="$1"
    for _m in "${METRICS[@]}"; do
        [[ "${_m}" == "${target}" ]] && return 0
    done
    return 1
}

# True if any of the three visual families (appearance/scene_identity/geometric)
# is selected.
has_visual_metric() {
    has_metric appearance || has_metric scene_identity || has_metric geometric
}

# ---- Environment ----
# Activate the eval environment robustly (also works in non-interactive shells,
# where `source activate` alone may not prepend the env to PATH). We additionally
# pin PYTHON to the env's interpreter by absolute path so the correct Python is
# used regardless of how activation resolves.
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh" ]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
fi
conda activate "${WORLD_SCORE_ENV}" 2>/dev/null || source activate "${WORLD_SCORE_ENV}" 2>/dev/null || true
if [ -x "${WORLD_SCORE_ENV}/bin/python" ]; then
    export PATH="${WORLD_SCORE_ENV}/bin:${PATH}"
    PYTHON="${WORLD_SCORE_ENV}/bin/python"
else
    PYTHON=python
fi
export PYTHONPATH="${GROUNDINGDINO_ROOT}:${SAM2_ROOT}:${MEM_BENCH_THIRDPARTY_ROOT:-${SCRIPT_DIR}/third_party}:${SCRIPT_DIR}/src:${PYTHONPATH:-}"

# ---- Working directories (inside full_eval/) ----
PAIRS_DIR="${SCRIPT_DIR}/pairs"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${PAIRS_DIR}" "${LOG_DIR}" "${NMR_RESULT_BASE}"

# ---- Load tasks from config arrays or conf ----
declare -a TASK_NAMES=()
declare -a VIDEO_DIRS=()
declare -a POSE_JSONS=()

append_task_if_selected() {
    local name="$1"
    local video_dir="$2"
    local pose_json="$3"

    # Apply task filter if specified
    if [[ -n "${TASK_FILTER}" ]]; then
        local matched=false
        local pat
        IFS=',' read -ra PATTERNS <<< "${TASK_FILTER}"
        for pat in "${PATTERNS[@]}"; do
            pat="$(echo "${pat}" | xargs)"
            # shellcheck disable=SC2254
            case "${name}" in ${pat}) matched=true; break ;; esac
        done
        ${matched} || return 0
    fi

    TASK_NAMES+=("${name}")
    VIDEO_DIRS+=("${video_dir}")
    POSE_JSONS+=("${pose_json}")
}

if [[ ${#CONFIG_TASK_NAMES[@]} -gt 0 ]]; then
    if [[ ${#CONFIG_TASK_NAMES[@]} -ne ${#CONFIG_VIDEO_DIRS[@]} || ${#CONFIG_TASK_NAMES[@]} -ne ${#CONFIG_POSE_JSONS[@]} ]]; then
        echo "ERROR: TASK entries must provide name, video_dir and pose_json."
        exit 1
    fi

    for i in "${!CONFIG_TASK_NAMES[@]}"; do
        append_task_if_selected \
            "${CONFIG_TASK_NAMES[$i]}" \
            "${CONFIG_VIDEO_DIRS[$i]}" \
            "${CONFIG_POSE_JSONS[$i]}"
    done
else
    while IFS='|' read -r name video_dir pose_json; do
        # Skip comments and empty lines
        [[ -z "${name}" || "${name}" =~ ^[[:space:]]*# ]] && continue
        name="$(echo "${name}" | xargs)"
        video_dir="$(echo "${video_dir}" | xargs)"
        pose_json="$(echo "${pose_json}" | xargs)"
        append_task_if_selected "${name}" "${video_dir}" "${pose_json}"
    done < "${TASKS_CONF}"
fi

TOTAL_TASKS=${#TASK_NAMES[@]}
if [[ ${TOTAL_TASKS} -eq 0 ]]; then
    echo "ERROR: No tasks matched. Check tasks.conf and --tasks filter."
    exit 1
fi

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
NUM_GPUS=${#GPUS[@]}

echo "============================================================"
echo "Full NMR Evaluation Pipeline"
echo "============================================================"
echo "Tasks loaded: ${TOTAL_TASKS}"
echo "Metrics:      ${METRICS[*]}"
echo "GPUs:         ${GPU_LIST}   (used by: appearance/scene_identity/geometric, object)"
echo "Shards/task:  ${NUM_VIDEO_SHARDS} (GPU metrics)"
echo "Task filter:  ${TASK_FILTER:-<all>}"
echo "============================================================"

# ============================================================
# Pre-sample pairs (shared across all metrics)
# ============================================================
run_pre_sample() {
    echo ""
    echo "============================================================"
    echo "[pre-sample] Sampling shared revisit/baseline/short pairs"
    echo "============================================================"
    ${PYTHON} -u "${SCRIPT_DIR}/src/sample_pairs.py" \
        --tasks_conf "${TASKS_CONF}" \
        --output_dir "${PAIRS_DIR}" \
        --max_eval_pairs 100 \
        --seed 42 \
        > "${LOG_DIR}/sample_pairs.log" 2>&1
    echo "[pre-sample] Done. Pairs dir: ${PAIRS_DIR}"
}

# Helper: find pairs JSON for a given pose_json (hash logic mirrors sample_pairs.py)
get_pairs_json() {
    local pose_json="$1"
    local key
    key=$(${PYTHON} -c "
import hashlib, sys
from pathlib import Path
pj = sys.argv[1]
h = hashlib.md5(pj.encode()).hexdigest()[:8]
name = Path(pj).stem
print(f'{name}_{h}')
" "${pose_json}")
    local pairs_file="${PAIRS_DIR}/${key}.json"
    if [[ -f "${pairs_file}" ]]; then
        echo "${pairs_file}"
    else
        echo ""
    fi
}

# ============================================================
# GPU-sharded NMR metric (used by both 'visual' and 'object')
#
# Args: metric_label  python_script  result_base  max_retries  common_args
# Each task is pinned to one GPU (round-robin) and split into NUM_VIDEO_SHARDS
# processes. A shard is retried up to max_retries times (1 = run once).
# ============================================================
run_sharded_gpu_metric() {
    local metric="$1"
    local script="$2"
    local result_base="$3"
    local max_retries="$4"
    local common_args="$5"

    # Filename-safe slug for log names (metric may contain [ ] , ).
    local metric_slug
    metric_slug=$(echo "${metric}" | tr -c 'A-Za-z0-9._-' '_' | sed 's/_*$//')

    echo ""
    echo "============================================================"
    echo "[${metric}] NMR evaluation (GPU, sharded)"
    echo "============================================================"
    echo "Script:  ${script##*/}"
    echo "Shards:  ${NUM_VIDEO_SHARDS} per task, ${max_retries} attempt(s) each"
    echo "Results: ${result_base}"

    local gpu_idx=0
    local pids=()

    for i in "${!TASK_NAMES[@]}"; do
        local task_name="${TASK_NAMES[$i]}"
        local video_dir="${VIDEO_DIRS[$i]}"
        local pose_json="${POSE_JSONS[$i]}"
        local output_dir="${result_base}/${task_name}"
        local gpu_id="${GPUS[$((gpu_idx % NUM_GPUS))]}"
        local pairs_json
        pairs_json=$(get_pairs_json "${pose_json}")

        for ((shard_id = 0; shard_id < NUM_VIDEO_SHARDS; shard_id++)); do
            mkdir -p "${output_dir}/shard_${shard_id}"
            local extra_args=""
            [[ -n "${pairs_json}" ]] && extra_args="--pairs_json ${pairs_json}"
            local log="${LOG_DIR}/${metric_slug}_${task_name}_shard${shard_id}.log"

            (
                : > "${log}"
                local ok=1
                local attempt
                for ((attempt = 1; attempt <= max_retries; attempt++)); do
                    echo "[shard ${shard_id}] attempt ${attempt}/${max_retries}" >> "${log}"
                    if CUDA_VISIBLE_DEVICES="${gpu_id}" ${PYTHON} -u "${script}" \
                        --video_dir "${video_dir}" \
                        --pose_json "${pose_json}" \
                        --output_dir "${output_dir}/shard_${shard_id}" \
                        --device "cuda:0" ${common_args} ${extra_args} \
                        --num_video_shards "${NUM_VIDEO_SHARDS}" \
                        --video_shard_id "${shard_id}" \
                        >> "${log}" 2>&1; then
                        ok=0
                        break
                    fi
                    echo "[shard ${shard_id}] FAILED (attempt ${attempt})" >> "${log}"
                    if [[ ${attempt} -lt ${max_retries} ]]; then
                        sleep 5
                    fi
                done
                exit ${ok}
            ) &
            pids+=($!)
        done

        gpu_idx=$((gpu_idx + 1))
    done

    echo "[${metric}] Launched ${#pids[@]} shard processes. Waiting..."
    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=$((failed + 1))
    done
    echo "[${metric}] Done. Failed shards: ${failed}/${#pids[@]}"
}

run_visual_metric() {
    local family="$1"
    local result_base="${NMR_RESULT_BASE}/${family}"
    local lpips_arg=""

    # LPIPS is part of the appearance family; PSNR/SSIM are gated internally by
    # the --families selector.
    if [[ "${family}" == "appearance" ]]; then
        lpips_arg="--enable_lpips"
    fi

    mkdir -p "${result_base}"

    local common_args="${lpips_arg} --clip_checkpoint ${CLIP_CKPT} --resume --max_eval_pairs 100 --families ${family}"
    run_sharded_gpu_metric "${family}" \
        "${SCRIPT_DIR}/src/eval_revisit_nmr.py" \
        "${result_base}" \
        1 \
        "${common_args}"
}

run_object_metric() {
    local common_args="--resume --max_eval_pairs 100 \
        --groundingdino_config ${GROUNDINGDINO_CONFIG} \
        --groundingdino_checkpoint ${GROUNDINGDINO_CKPT} \
        --sam2_config ${SAM2_CONFIG} \
        --sam_checkpoint ${SAM2_CKPT} \
        --clip_checkpoint ${CLIP_CKPT}"
    run_sharded_gpu_metric "object" \
        "${SCRIPT_DIR}/src/metrics/object_identity.py" \
        "${NMR_RESULT_BASE}/object" \
        3 \
        "${common_args}"
}

# ============================================================
# Persistent-state NMR metric (VLM API, rolling queue)
# ============================================================
run_state_metric() {
    local ENGINE=gemini-3.1-pro-preview
    local MAX_PAIRS=10
    local NUM_WORKERS=3
    local MAX_JOBS=3
    # API_KEY / API_URL / GEMINI_PYTHON come from run_config.conf
    local result_base="${NMR_RESULT_BASE}/persistent_state"
    mkdir -p "${result_base}"

    echo ""
    echo "============================================================"
    echo "[state] NMR evaluation (persistent-state VLM, API)"
    echo "============================================================"
    echo "Engine: ${ENGINE}, max pairs/type: ${MAX_PAIRS}, concurrency: ${MAX_JOBS}"
    echo "Results: ${result_base}"

    local state_pids=()
    for i in "${!TASK_NAMES[@]}"; do
        local task_name="${TASK_NAMES[$i]}"
        local video_dir="${VIDEO_DIRS[$i]}"
        local pose_json="${POSE_JSONS[$i]}"
        local result_json="${result_base}/results_gemini_nmr_${task_name}.json"

        # Forcing tasks are not supported by the state (Gemini) metric
        if [[ "${task_name}" == forcing_* ]]; then
            echo "  [skip] ${task_name} (forcing not supported for state metric)"
            continue
        fi

        # Rolling queue: keep at most MAX_JOBS in flight
        while [[ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]]; do
            sleep 5
        done

        local pairs_json
        pairs_json=$(get_pairs_json "${pose_json}")
        local extra_args=""
        [[ -n "${pairs_json}" ]] && extra_args="--pairs_json ${pairs_json}"

        echo "  [launch] ${task_name}"
        ${GEMINI_PYTHON} "${SCRIPT_DIR}/src/metrics/persistent_state.py" \
            --video_dir "${video_dir}" \
            --pose_json "${pose_json}" \
            --output_json "${result_json}" \
            --engine "${ENGINE}" \
            --max_pairs_per_type "${MAX_PAIRS}" \
            --num_workers "${NUM_WORKERS}" \
            --resume ${extra_args} \
            > "${LOG_DIR}/state_${task_name}.log" 2>&1 &
        state_pids+=($!)
    done

    echo "[state] Launched ${#state_pids[@]} tasks. Waiting..."
    local failed=0
    for pid in "${state_pids[@]}"; do
        wait "${pid}" || failed=$((failed + 1))
    done
    echo "[state] Done. Failed tasks: ${failed}/${#state_pids[@]}"
}

# ============================================================
# Main execution
# ============================================================
echo ""
echo "Starting at $(date '+%Y-%m-%d %H:%M:%S')"

# Materialize the selected config tasks for sample_pairs.py.
TASKS_RUNTIME_CONF="${PAIRS_DIR}/selected_tasks.conf"
: > "${TASKS_RUNTIME_CONF}"
for i in "${!TASK_NAMES[@]}"; do
    printf '%s|%s|%s\n' "${TASK_NAMES[$i]}" "${VIDEO_DIRS[$i]}" "${POSE_JSONS[$i]}" >> "${TASKS_RUNTIME_CONF}"
done
TASKS_CONF="${TASKS_RUNTIME_CONF}"

# Always pre-sample pairs first (needed by every metric)
run_pre_sample

# Schedule each requested family independently.
has_metric appearance       && run_visual_metric appearance
has_metric scene_identity   && run_visual_metric scene_identity
has_metric geometric        && run_visual_metric geometric
has_metric object           && run_object_metric
has_metric persistent_state && run_state_metric

echo ""
echo "============================================================"
echo "All selected NMR evaluations complete at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
has_metric appearance       && echo "appearance results:        ${NMR_RESULT_BASE}/appearance"
has_metric scene_identity   && echo "scene_identity results:    ${NMR_RESULT_BASE}/scene_identity"
has_metric geometric        && echo "geometric results:         ${NMR_RESULT_BASE}/geometric"
has_metric object           && echo "object results:            ${NMR_RESULT_BASE}/object"
has_metric persistent_state && echo "persistent_state results:  ${NMR_RESULT_BASE}/persistent_state"
echo "Logs:                      ${LOG_DIR}"
echo "============================================================"
