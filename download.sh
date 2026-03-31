#!/usr/bin/env bash

set -euo pipefail

usage() {
  printf 'Usage: %s <uri_list.txt> [output_dir] [jobs]\n' "$0" >&2
  exit 1
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
fi

list_file="$1"
output_dir="${2:-downloads}"
jobs="${3:-8}"

if [[ ! -f "$list_file" ]]; then
  printf 'List file not found: %s\n' "$list_file" >&2
  exit 1
fi

mkdir -p "$output_dir"

tmpdir="$(mktemp -d)"
done_file="$tmpdir/done.log"
trap 'rm -rf "$tmpdir"' EXIT

total="$(grep -cv '^[[:space:]]*$' "$list_file")"

progress_bar() {
  local done width filled empty pct bar
  width=40
  while true; do
    done="$(wc -l < "$done_file" 2>/dev/null || printf '0')"
    pct=0
    if [[ "$total" -gt 0 ]]; then
      pct=$((done * 100 / total))
    fi
    filled=$((pct * width / 100))
    empty=$((width - filled))
    bar="$(printf '%0.s#' $(seq 1 "$filled"))$(printf '%0.s-' $(seq 1 "$empty"))"
    printf '\r[%s] %3d%% (%d/%d)' "$bar" "$pct" "$done" "$total" >&2
    if [[ "$done" -ge "$total" ]]; then
      printf '\n' >&2
      break
    fi
    sleep 1
  done
}

download_one() {
  local uri="$1"
  local rel_path local_path

  rel_path="${uri#s3://}"
  rel_path="${rel_path#*/}"
  local_path="$output_dir/$rel_path"

  mkdir -p "$(dirname "$local_path")"
  aws s3 cp "$uri" "$local_path" --only-show-errors
  printf '.\n' >> "$done_file"
}

export -f download_one
export output_dir done_file

progress_bar &
progress_pid=$!

tr -d '\r' < "$list_file" | grep -v '^[[:space:]]*$' | \
  xargs -n 1 -P "$jobs" bash -c 'download_one "$1"' _

wait "$progress_pid"
