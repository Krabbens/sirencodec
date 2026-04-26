#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  download-train-clean-360 [--data-root PATH] [--keep-archive]

Downloads LibriSpeech train-clean-360 from OpenSLR and extracts it to:
  PATH/train-clean-360

Defaults:
  PATH is $SIRENCODEC_DATA_ROOT or $SIRENCODEC_WORKDIR/data or /workspace/data.

Environment:
  LIBRISPEECH_360_URL   override download URL
  SIRENCODEC_DATA_ROOT  default data root
EOF
}

repo_dir="${SIRENCODEC_WORKDIR:-/workspace}"
data_root="${SIRENCODEC_DATA_ROOT:-$repo_dir/data}"
keep_archive=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      data_root="${2:?--data-root requires a path}"
      shift 2
      ;;
    --keep-archive)
      keep_archive=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

url="${LIBRISPEECH_360_URL:-https://www.openslr.org/resources/12/train-clean-360.tar.gz}"
dataset="train-clean-360"
target="$data_root/$dataset"
archive_dir="$data_root/downloads"
archive="$archive_dir/$dataset.tar.gz"

if [[ -d "$target" ]] && find "$target" -type f -name '*.flac' -print -quit | grep -q .; then
  echo "Dataset already present: $target"
  exit 0
fi

mkdir -p "$archive_dir"

echo "Downloading $url"
echo "Archive: $archive"
curl -L --fail --retry 5 --retry-delay 5 --continue-at - -o "$archive" "$url"

tmp_dir="$(mktemp -d "$data_root/.extract-${dataset}.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

echo "Extracting to temporary directory: $tmp_dir"
tar -xzf "$archive" -C "$tmp_dir"

extracted="$tmp_dir/LibriSpeech/$dataset"
if [[ ! -d "$extracted" ]]; then
  echo "ERROR: expected directory missing after extraction: $extracted" >&2
  exit 3
fi

mkdir -p "$data_root"
if [[ -e "$target" ]]; then
  echo "ERROR: target exists but does not look complete: $target" >&2
  echo "Move it aside or remove it, then rerun this script." >&2
  exit 4
fi

mv "$extracted" "$target"

if [[ "$keep_archive" != "1" ]]; then
  rm -f "$archive"
fi

echo "Ready: $target"
