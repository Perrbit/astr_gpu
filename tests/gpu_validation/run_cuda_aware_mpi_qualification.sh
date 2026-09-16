#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE_EXE="${PROBE_EXE:-$ROOT_DIR/build_gpu_probe/bin/halo_cuda_aware_probe}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/cuda_aware_mpi_qualification}"
PYTHON_EXE="${PYTHON_EXE:-python3}"
SANITIZER_EXE="${SANITIZER_EXE:-compute-sanitizer}"
SUPPRESSIONS="$ROOT_DIR/tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml"
RESULTS="$OUT_DIR/results.tsv"
SUMMARY="$OUT_DIR/qualification.json"
PAYLOAD_SHAPES=("513 257" "513 513")
CONFIGS=(ucx_ipc ucx_no_ipc)
QUALIFICATION_CUDA_VISIBLE_DEVICES="${QUALIFICATION_CUDA_VISIBLE_DEVICES:-0,1}"
PAYLOAD_REPEATS="${PAYLOAD_REPEATS:-5}"

MPI_STACK_INIT="none"
MPIEXEC_EXE=""
if command -v mpifort >/dev/null 2>&1; then
  if mpi_libdirs="$(mpifort --showme:libdirs 2>/dev/null)"; then
    mpi_libdir="${mpi_libdirs%% *}"
    mpi_prefix="$(cd "$mpi_libdir/.." && pwd)"
    if [[ -x "$mpi_prefix/bin/.bin/mpirun" ]]; then
      MPIEXEC_EXE="$mpi_prefix/bin/.bin/mpirun"
    fi
    hpcx_init="$(dirname "$mpi_prefix")/hpcx-init-ompi.sh"
    if [[ -f "$hpcx_init" ]]; then
      # Use one MPI/UCX installation instead of mixing HPC-X wrappers with system UCX.
      set +u
      source "$hpcx_init"
      hpcx_load
      set -u
      MPI_STACK_INIT="$hpcx_init"
    fi
  fi
fi
if [[ -z "$MPIEXEC_EXE" ]]; then
  MPIEXEC_EXE="$(command -v mpirun)"
fi

if [[ ! -x "$PROBE_EXE" ]]; then
  printf 'CUDA-aware probe is not executable: %s\n' "$PROBE_EXE" >&2
  exit 2
fi
if [[ ! -f "$SUPPRESSIONS" ]]; then
  printf 'Compute Sanitizer suppression file is unavailable: %s\n' "$SUPPRESSIONS" >&2
  exit 2
fi
if [[ ! "$PAYLOAD_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PAYLOAD_REPEATS must be a positive integer: %s\n' "$PAYLOAD_REPEATS" >&2
  exit 2
fi
for command in mpifort mpirun ompi_info ucx_info ldd nvidia-smi sha256sum \
  "$PYTHON_EXE" "$SANITIZER_EXE"; do
  command -v "$command" >/dev/null 2>&1 || {
    printf 'required command is unavailable: %s\n' "$command" >&2
    exit 2
  }
done
if [[ -e "$OUT_DIR" ]]; then
  printf 'refusing to overwrite qualification output: %s\n' "$OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count+0}')"
if [[ "$gpu_count" -lt 2 ]]; then
  printf 'device-aware MPI qualification requires two physical GPUs\n' >&2
  exit 2
fi

{
  printf 'date: %s\n' "$(date -Iseconds)"
  printf 'hostname: %s\n' "$(hostname)"
  printf 'mpi_stack_init: %s\n' "$MPI_STACK_INIT"
  printf 'mpifort_path: %s\n' "$(command -v mpifort)"
  printf 'mpirun_path: %s\n' "$(command -v mpirun)"
  printf 'mpiexec_binary: %s\n' "$MPIEXEC_EXE"
  printf 'ompi_info_path: %s\n' "$(command -v ompi_info)"
  printf 'ucx_info_path: %s\n' "$(command -v ucx_info)"
  printf 'qualification_cuda_visible_devices: %s\n' \
    "$QUALIFICATION_CUDA_VISIBLE_DEVICES"
  printf 'payload_repeats: %s\n' "$PAYLOAD_REPEATS"
  printf 'probe_sha256: '
  sha256sum "$PROBE_EXE"
  printf 'suppressions_sha256: '
  sha256sum "$SUPPRESSIONS"
  printf '\nprobe_dynamic_libraries:\n'
  ldd "$PROBE_EXE"
  printf '\nmpirun:\n'
  mpirun --version
  printf '\nnvidia_smi:\n'
  nvidia-smi
  printf '\nnvidia_topology:\n'
  nvidia-smi topo -m
  printf '\nompi_accelerator_metadata:\n'
  ompi_info --parsable --all
  printf '\nucx_version:\n'
  ucx_info -v
  printf '\nucx_devices:\n'
  ucx_info -d
} > "$OUT_DIR/environment.txt" 2>&1

printf 'config\tshape\tkind\tstatus\texit_code\tlog\n' > "$RESULTS"

run_with_config() {
  local config="$1"
  shift
  local tls
  case "$config" in
    ucx_ipc) tls="self,sm,cuda_copy,cuda_ipc" ;;
    ucx_no_ipc) tls="self,sm,cuda_copy" ;;
    *)
      printf 'unknown qualification configuration: %s\n' "$config" >&2
      return 2
      ;;
  esac
  env CUDA_VISIBLE_DEVICES="$QUALIFICATION_CUDA_VISIBLE_DEVICES" \
    OMPI_MCA_pml=ucx OMPI_MCA_coll='^hcoll,ucc,cuda' \
    OMPI_MCA_coll_hcoll_enable=0 UCX_MEMTYPE_CACHE=n \
    UCX_CUDA_COPY_ENABLE_FABRIC=no UCX_CUDA_COPY_DMABUF=no \
    UCX_CUDA_IPC_ENABLE_MNNVL=no UCX_TLS="$tls" "$@"
}

record_result() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$RESULTS"
}

kind_config_passed() {
  local config="$1"
  local kind="$2"
  local expected_count="$3"
  awk -F '\t' -v config="$config" -v kind="$kind" -v expected="$expected_count" '
    NR > 1 && $1 == config && $3 == kind {count++; if ($4 != "PASS") failed++}
    END {exit !(count == expected && failed == 0)}
  ' "$RESULTS"
}

for config in "${CONFIGS[@]}"; do
  for shape_entry in "${PAYLOAD_SHAPES[@]}"; do
    read -r ny nz <<< "$shape_entry"
    shape="${ny}x${nz}"
    log="$OUT_DIR/${config}_payload_${shape}.log"
    payload_status=0
    : > "$log"
    for ((repeat=1; repeat<=PAYLOAD_REPEATS; repeat++)); do
      printf '\n=== payload repeat %d/%d ===\n' "$repeat" "$PAYLOAD_REPEATS" >> "$log"
      if run_with_config "$config" "$MPIEXEC_EXE" -np 2 "$PROBE_EXE" "$ny" "$nz" \
        "early-bind" >> "$log" 2>&1; then
        :
      else
        payload_status=$?
        break
      fi
    done
    if [[ "$payload_status" -eq 0 ]]; then
      record_result "$config" "$shape" payload PASS 0 "$log"
    else
      record_result "$config" "$shape" payload FAIL "$payload_status" "$log"
    fi
  done

  protocol_log="$OUT_DIR/${config}_protocol_513x513.log"
  if run_with_config "$config" env UCX_PROTO_INFO=y \
    "$MPIEXEC_EXE" -np 2 "$PROBE_EXE" 513 513 "early-bind" \
    > "$protocol_log" 2>&1; then
    protocol_status=0
    if [[ "$config" == "ucx_ipc" ]]; then
      grep -q 'cuda_ipc' "$protocol_log" || protocol_status=97
    elif grep -q 'cuda_ipc' "$protocol_log"; then
      protocol_status=97
    fi
  else
    protocol_status=$?
  fi
  if [[ "$protocol_status" -eq 0 ]]; then
    record_result "$config" 513x513 protocol PASS 0 "$protocol_log"
  else
    record_result "$config" 513x513 protocol FAIL "$protocol_status" "$protocol_log"
  fi

  sanitizer_log="$OUT_DIR/${config}_sanitizer_513x513.log"
  if kind_config_passed "$config" payload 2 && \
     kind_config_passed "$config" protocol 1; then
    if run_with_config "$config" "$SANITIZER_EXE" --tool memcheck \
      --error-exitcode 99 --target-processes all --suppressions "$SUPPRESSIONS" \
      "$MPIEXEC_EXE" -np 2 "$PROBE_EXE" 513 513 "early-bind" \
      > "$sanitizer_log" 2>&1; then
      clean_count="$(grep -c 'ERROR SUMMARY: 0 errors' "$sanitizer_log")"
      if [[ "$clean_count" -ge 1 ]]; then
        record_result "$config" 513x513 sanitizer PASS 0 "$sanitizer_log"
      else
        record_result "$config" 513x513 sanitizer FAIL 98 "$sanitizer_log"
      fi
    else
      status=$?
      record_result "$config" 513x513 sanitizer FAIL "$status" "$sanitizer_log"
    fi
  else
    printf 'sanitizer skipped because payload or protocol qualification failed\n' > "$sanitizer_log"
    record_result "$config" 513x513 sanitizer SKIP 3 "$sanitizer_log"
  fi
done

if "$PYTHON_EXE" "$ROOT_DIR/tests/gpu_validation/summarize_cuda_aware_mpi_qualification.py" \
  --results "$RESULTS" --output "$SUMMARY" \
  --mpi-stack "$("$MPIEXEC_EXE" --version | sed -n '1p')" \
  > "$OUT_DIR/summary.stdout" 2>&1; then
  cat "$OUT_DIR/summary.stdout"
  exit 0
else
  status=$?
  cat "$OUT_DIR/summary.stdout"
  exit "$status"
fi
