#!/usr/bin/env bash
# Raw teleop HDF5 -> camera replay HDF5 -> preview MP4s -> LeRobot dataset.
#
#   scripts_bash/convert_hdf5_hdf5cam_lerobot.sh datasets/dex_20260827_113651.hdf5
#
# Every downstream path is derived from the input stem, so the example above writes
#   datasets/dex_20260827_113651cam.hdf5       replay with camera obs
#   videos/dex_20260827_113651cam/demo_*.mp4   preview videos
#   datasets/dex_20260827_113651cam_lerobot/   LeRobot dataset
#
# Stages whose output already exists are skipped (so a failed conversion can be
# re-run without redoing the expensive replay); --force redoes everything.
# Commands run from the repo root regardless of where you invoke the script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

# Replay knobs that matter often enough to be flags; anything else goes through
# --replay-extra / --video-extra / --lerobot-extra.
TASK="Isaac-Demo-Sberchel-XR-v0"
FRAME_STRIDE=8
REPLAY_SOURCE="processed_joint_targets"
VIDEO_ROOT="videos"
OUT_DIR=""
SKIP_REPLAY=0
SKIP_VIDEO=0
SKIP_LEROBOT=0
FORCE=0
DRY_RUN=0
REPLAY_EXTRA=()
VIDEO_EXTRA=()
LEROBOT_EXTRA=()

usage() {
    sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<'EOF'

Options:
  -o, --output-dir DIR    Directory for the cam HDF5 and LeRobot dataset
                          (default: the input file's directory).
      --video-root DIR    Root for preview videos (default: videos, under the repo root).
      --task NAME         Replay task (default: Isaac-Demo-Sberchel-XR-v0).
      --frame-stride N    Camera capture stride (default: 8).
      --replay-source S   policy_loco | processed_joint_targets | actions
                          (default: processed_joint_targets).
      --skip-replay       Reuse an existing <stem>cam.hdf5 as-is.
      --skip-video        Do not encode preview MP4s.
      --skip-lerobot      Stop after the videos.
      --replay-extra "..."   Extra args appended to the replay command.
      --video-extra "..."    Extra args appended to the video command.
      --lerobot-extra "..."  Extra args appended to the LeRobot command.
                          (These are word-split; they carry no quoted arguments.)
  -f, --force             Redo stages whose output already exists.
  -n, --dry-run           Print the commands without running them.
  -h, --help              This help.

Environment:
  PYTHON   Interpreter to use (default: python — activate the Isaac env first).
EOF
}

die() { echo "error: $*" >&2; exit 1; }
log() { echo; echo "=== [$(date '+%H:%M:%S')] $*"; }

# Echo a command in copy-pasteable form, then run it (unless --dry-run).
run() {
    local shown=() arg
    for arg in "$@"; do shown+=("$(printf '%q' "$arg")"); done
    echo "+ ${shown[*]}"
    if (( DRY_RUN == 0 )); then "$@"; fi
}

INPUT=""
while (( $# )); do
    case "$1" in
        -o|--output-dir)   OUT_DIR="$2"; shift 2 ;;
        --video-root)      VIDEO_ROOT="$2"; shift 2 ;;
        --task)            TASK="$2"; shift 2 ;;
        --frame-stride)    FRAME_STRIDE="$2"; shift 2 ;;
        --replay-source)   REPLAY_SOURCE="$2"; shift 2 ;;
        --skip-replay)     SKIP_REPLAY=1; shift ;;
        --skip-video)      SKIP_VIDEO=1; shift ;;
        --skip-lerobot)    SKIP_LEROBOT=1; shift ;;
        --replay-extra)    read -ra REPLAY_EXTRA <<< "$2"; shift 2 ;;
        --video-extra)     read -ra VIDEO_EXTRA <<< "$2"; shift 2 ;;
        --lerobot-extra)   read -ra LEROBOT_EXTRA <<< "$2"; shift 2 ;;
        -f|--force)        FORCE=1; shift ;;
        -n|--dry-run)      DRY_RUN=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        -*)                die "unknown option: $1 (see --help)" ;;
        *)
            [[ -n "$INPUT" ]] && die "only one input HDF5 accepted (got '$INPUT' and '$1')"
            INPUT="$1"; shift ;;
    esac
done

[[ -n "$INPUT" ]] || { usage >&2; die "no input HDF5 given"; }
[[ -f "$INPUT" ]] || die "input not found: $INPUT"

# Resolve everything to absolute paths before cd'ing, so relative arguments are
# interpreted against the caller's directory.
RAW_FILE="$(realpath "$INPUT")"
BASE="${RAW_FILE##*/}"
case "$BASE" in
    *.hdf5) STEM="${BASE%.hdf5}" ;;
    *.h5)   STEM="${BASE%.h5}" ;;
    *)      die "input does not look like an HDF5 file: $BASE" ;;
esac
[[ "$STEM" == *cam ]] && echo "warning: '$BASE' already looks like a replay output; expected the raw teleop file" >&2

if [[ -n "$OUT_DIR" ]]; then
    mkdir -p "$OUT_DIR"
    OUT_DIR="$(realpath "$OUT_DIR")"
else
    OUT_DIR="$(dirname "$RAW_FILE")"
fi

CAM_FILE="${OUT_DIR}/${STEM}cam.hdf5"
LEROBOT_DIR="${OUT_DIR}/${STEM}cam_lerobot"

cd "$REPO_ROOT"
VIDEO_DIR="$(realpath -m "$VIDEO_ROOT")/${STEM}cam"

echo "repo root : $REPO_ROOT"
echo "python    : $PYTHON"
echo "raw       : $RAW_FILE"
echo "cam hdf5  : $CAM_FILE"
echo "videos    : $VIDEO_DIR"
echo "lerobot   : $LEROBOT_DIR"

stage_start=0
begin() { log "$1"; stage_start=$SECONDS; }
finish() { echo "--- done in $(( SECONDS - stage_start ))s"; }

# ---------------------------------------------------------------------------
# 1. Replay with cameras
# ---------------------------------------------------------------------------
if (( SKIP_REPLAY )); then
    log "replay: skipped (--skip-replay)"
    [[ -f "$CAM_FILE" ]] || (( DRY_RUN )) || die "no existing $CAM_FILE to continue from"
elif [[ -f "$CAM_FILE" ]] && (( FORCE == 0 )); then
    log "replay: skipped, $CAM_FILE already exists (use --force to redo)"
else
    begin "replay -> $CAM_FILE"
    run "$PYTHON" scripts/sberchel_replay_demos.py \
        --task "$TASK" \
        --dataset_file "$RAW_FILE" \
        --output_dataset_file "$CAM_FILE" \
        --replay_source "$REPLAY_SOURCE" \
        --enable_cameras \
        --enable_pinocchio \
        --headless \
        --fast_export \
        --third_person \
        --frame_stride "$FRAME_STRIDE" \
        "${REPLAY_EXTRA[@]}"
    finish
fi

# ---------------------------------------------------------------------------
# 2. Preview videos
# ---------------------------------------------------------------------------
if (( SKIP_VIDEO )); then
    log "videos: skipped (--skip-video)"
elif [[ -d "$VIDEO_DIR" ]] && compgen -G "$VIDEO_DIR/*.mp4" > /dev/null && (( FORCE == 0 )); then
    log "videos: skipped, $VIDEO_DIR already has MP4s (use --force to redo)"
else
    begin "videos -> $VIDEO_DIR"
    run "$PYTHON" scripts/sberchel_hdf5_to_video.py \
        --input_file "$CAM_FILE" \
        --output_root "$VIDEO_ROOT" \
        "${VIDEO_EXTRA[@]}"
    finish
fi

# ---------------------------------------------------------------------------
# 3. LeRobot dataset
# ---------------------------------------------------------------------------
if (( SKIP_LEROBOT )); then
    log "lerobot: skipped (--skip-lerobot)"
elif [[ -e "$LEROBOT_DIR" ]] && (( FORCE == 0 )); then
    log "lerobot: skipped, $LEROBOT_DIR already exists (use --force to redo)"
else
    begin "lerobot -> $LEROBOT_DIR"
    overwrite=()
    (( FORCE )) && overwrite=(--overwrite)
    run "$PYTHON" scripts/sberchel_hdf5_to_lerobot.py \
        "$CAM_FILE" \
        --output "$LEROBOT_DIR" \
        "${overwrite[@]}" \
        "${LEROBOT_EXTRA[@]}"
    finish
fi

log "all done"
echo "  cam hdf5 : $CAM_FILE"
echo "  videos   : $VIDEO_DIR"
echo "  lerobot  : $LEROBOT_DIR"
