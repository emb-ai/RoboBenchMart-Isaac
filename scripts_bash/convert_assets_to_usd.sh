#!/usr/bin/env bash
# Converts all 3D models in assets/preprocessed and assets/fake_shelves to USD format
# using IsaacLab's convert_mesh.py script inside the isaacsim conda environment.

set -euo pipefail

ISAACLAB_DIR="/home/kvsoshin/Work/IsaacLab"
CONVERT_SCRIPT="${ISAACLAB_DIR}/scripts/tools/convert_mesh.py"

# REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="."
INPUT_PREPROCESSED="${REPO_DIR}/assets/preprocessed"
INPUT_FAKE_SHELVES="${REPO_DIR}/assets/fake_shelves"

OUTPUT_BASE="${REPO_DIR}/assets_usd"
OUTPUT_PREPROCESSED="${OUTPUT_BASE}/preprocessed"
OUTPUT_FAKE_SHELVES="${OUTPUT_BASE}/fake_shelves"

CONDA_ENV="isaacsim"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARNING: $*" >&2; }

# activate_conda() {
#     # Works whether conda is initialised via bashrc or not
#     local conda_base
#     conda_base="$(conda info --base 2>/dev/null)" || { warn "conda not found on PATH"; exit 1; }
#     # shellcheck source=/dev/null
#     source "${conda_base}/etc/profile.d/conda.sh"
#     conda activate "${CONDA_ENV}"
# }

convert_dir() {
    local input_dir="$1"
    local output_dir="$2"

    mkdir -p "${output_dir}"

    local total converted skipped failed
    total=0; converted=0; skipped=0; failed=0

    while IFS= read -r -d '' input_file; do
        total=$(( total + 1 ))

        local base_name
        base_name="$(basename "${input_file%.*}")"
        local output_file="${output_dir}/${base_name}/${base_name}.usd"

        if [[ -f "${output_file}" ]]; then
            log "  SKIP  ${base_name}  (USD already exists)"
            skipped=$(( skipped + 1 ))
            continue
        fi

        mkdir -p "$(dirname "${output_file}")"

        log "  CONVERT  ${base_name}"
        if python "${CONVERT_SCRIPT}" \
                "${input_file}" \
                "${output_file}" \
                --make-instanceable \
                --collision-approximation convexDecomposition \
                --mass 1.0 \
                --headless 2>&1; then
            converted=$(( converted + 1 ))
        else
            warn "Failed to convert ${input_file}"
            failed=$(( failed + 1 ))
        fi
    done < <(find "${input_dir}" -maxdepth 1 -type f \( -iname "*.glb" -o -iname "*.obj" -o -iname "*.fbx" -o -iname "*.stl" \) -print0 | sort -z)

    log "  Done: ${converted} converted, ${skipped} skipped, ${failed} failed  (total: ${total})"
    return "${failed}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# log "Activating conda environment '${CONDA_ENV}' ..."
# activate_conda

log "=== Converting preprocessed assets ==="
log "  Input : ${INPUT_PREPROCESSED}"
log "  Output: ${OUTPUT_PREPROCESSED}"
convert_dir "${INPUT_PREPROCESSED}" "${OUTPUT_PREPROCESSED}"
preprocessed_rc=$?

log "=== Converting fake_shelves assets ==="
log "  Input : ${INPUT_FAKE_SHELVES}"
log "  Output: ${OUTPUT_FAKE_SHELVES}"
convert_dir "${INPUT_FAKE_SHELVES}" "${OUTPUT_FAKE_SHELVES}"
fake_shelves_rc=$?

if (( preprocessed_rc + fake_shelves_rc == 0 )); then
    log "All conversions completed successfully."
else
    warn "Some conversions failed. Check the output above for details."
    exit 1
fi
