#!/usr/bin/env bash

p2_validate_case() {
  case "$1" in
    shuosher|sbli) ;;
    *)
      printf 'CASE must be shuosher or sbli\n' >&2
      return 2
      ;;
  esac
}

p2_default_grid() {
  case "$1" in
    shuosher) printf '256,64,32\n' ;;
    sbli) printf '256,192,32\n' ;;
    *) p2_validate_case "$1" ;;
  esac
}

p2_prepare_case() {
  local case_name="$1"
  local case_dir="$2"
  local grid="$3"
  local maxstep="$4"
  local feqchkpt="$5"
  local feqlist="$6"
  local im jm km

  p2_validate_case "$case_name"
  IFS=, read -r im jm km <<< "$grid"
  if [[ -z "$im" || -z "$jm" || -z "$km" ]]; then
    printf 'GRID must have the form im,jm,km\n' >&2
    return 2
  fi

  P2_RUNTIME_ENV=()
  if [[ "$case_name" == "shuosher" ]]; then
    python3 "$P2_ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
      --src-case "$P2_ROOT_DIR/examples/Shuosher" \
      --dst-case "$case_dir" \
      --input-name input.shuosher \
      --flowtype shuosher \
      --homogeneous t,t,t \
      --bctype 1,1,1,1,1,1 \
      --use-gpu t \
      --maxstep "$maxstep" \
      --feqchkpt "$feqchkpt" \
      --feqlist "$feqlist" \
      --lfilter f \
      --diffterm f \
      --scheme 643e \
      --conschm 543e \
      --difschm 643e \
      --recon-schem 3 \
      --lchardecomp t \
      --grid "$grid" \
      --deltat "${DELTAT:-1.d-4}"
    P2_INPUT=datin/input.shuosher
    return
  fi

  local mach="${MACH:-2.0}"
  local reynolds="${REYNOLDS:-296000.0}"
  local reference_temperature="${REFERENCE_TEMPERATURE:-148.9}"
  local wall_temperature="${WALL_TEMPERATURE:-0.8}"
  local shock_angle="${SHOCK_ANGLE_DEG:-32.6}"
  local shock_x="${SHOCK_X0:-1.0}"
  local shock_y="${SHOCK_Y0:-1.0}"

  python3 "$P2_ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$case_dir" \
    --use-gpu t \
    --im "$im" --jm "$jm" --km "$km" \
    --diffterm t --lchardecomp t \
    --shock-threshold "${SHOCK_THRESHOLD:-0.001}" \
    --conschm 543e --reynolds "$reynolds" --mach "$mach" \
    --reference-temperature "$reference_temperature" \
    --wall-temperature "$wall_temperature" \
    --upper-bctype 52 --x-min-bctype 11 --ninit 3 \
    --x-min -1.0 --x-max 10.0 --y-stretch 5.0 --z-length 0.25 \
    --maxstep "$maxstep" --feqchkpt "$feqchkpt" --feqlist "$feqlist" \
    --sponge-im "${SPONGE_IM:-16}" --deltat "${DELTAT:-2.5e-4}"

  python3 "$P2_ROOT_DIR/tests/gpu_validation/generate_compressible_blasius_profile.py" \
    --grid "$case_dir/datin/grid.flatplate.h5" \
    --output "$case_dir/datin/inlet.prof" \
    --mach "$mach" --reynolds "$reynolds" \
    --reference-temperature "$reference_temperature" \
    --wall-temperature "$wall_temperature" --station-x 1.0 \
    --density-mode provided --pressure-mode reconstruct \
    --field-output "$case_dir/datin/flowini3d.h5" \
    --virtual-leading-edge -2.0 \
    --field-oblique-shock \
    --shock-angle-deg "$shock_angle" \
    --shock-x0 "$shock_x" --shock-y0 "$shock_y" --shock-y-min 0.0

  P2_RUNTIME_ENV=(
    ASTR_NSCBC_FARFIELD_MODE=sbli_shock
    ASTR_NSCBC_FARFIELD_RHO=1.0
    ASTR_NSCBC_FARFIELD_U=1.0
    ASTR_NSCBC_FARFIELD_V=0.0
    ASTR_NSCBC_FARFIELD_W=0.0
    ASTR_NSCBC_FARFIELD_T=1.0
    "ASTR_NSCBC_FARFIELD_SHOCK_X=$shock_x"
    "ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG=$shock_angle"
  )
  P2_INPUT=datin/input.flatplate
}
