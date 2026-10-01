"""Bounded exact continuation with real controller reloads and time archives."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from run_complete_step_clock import CLOCK
from run_output_archive_validation import check_accounting, check_frame, check_series, check_series_reader, frames, groups
from run_output_restart_validation import archive_schedule_payload, compare_fields, controlled_checkpoint_buffers, run_case
from test_checkpoint_bundle import fault_library

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("ASTR_OUTPUT_RUNTIME_EXE", ROOT / "build_gpu_probe/bin/astr")).resolve()
MPIEXEC = Path(os.environ.get("ASTR_OUTPUT_MPIEXEC",
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")).resolve()
DT_NEXT = [0.002] * 5 + [0.0005] * 7
DT_USED = [0.001] + DT_NEXT[:-1]
TIMES = [0.0]
for _dt in DT_USED:
    TIMES.append(TIMES[-1] + _dt)
FINAL = "outdat/new/checkpoints/step000000000012"


@pytest.mark.parametrize("api", range(8))
def test_controller_replay_is_read_only_and_path_scoped(tmp_path, fault_library, api):
    source = tmp_path / "probe.c"
    source.write_text(r'''
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc != 4) return 1;
    int api = atoi(argv[1]);
    char value[32];
    FILE *other = fopen(argv[3], "r");
    if (!other || fscanf(other, "%31s", value) != 1 || fclose(other)) return 2;
    int unchanged = open(argv[2], O_WRONLY);
    if (unchanged < 0 || close(unchanged)) return 3;
    for (int n = 0; n < 4; n++) {
        FILE *file;
        if (api < 2 || api == 4 || api == 5) {
            int flags = api >= 4 ? O_RDWR : O_RDONLY;
            int fd = api % 2 ? open64(argv[2], flags) : open(argv[2], flags);
            file = fd < 0 ? NULL : fdopen(fd, "r");
        } else {
            const char *mode = api >= 6 ? "r+" : "r";
            file = api % 2 ? fopen64(argv[2], mode) : fopen(argv[2], mode);
        }
        if (!file || fscanf(file, "%31s", value) != 1 || fclose(file)) return 4;
        puts(value);
    }
    return 0;
}
''')
    executable = tmp_path / "probe"
    subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", str(source),
                    "-o", str(executable)], check=True, timeout=30)
    controller = tmp_path / "controller"
    controller.write_text("unchanged\n")
    other = tmp_path / "other/controller"
    other.parent.mkdir()
    other.write_text("unrelated\n")
    replay = tmp_path / "replay"
    replay.mkdir()
    for step, text in zip((0, 6, 7, 8), ("initial", "next6", "next7", "next8")):
        (replay / f"controller{step:012d}").write_text(text + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASTR_")}
    env.update(LD_PRELOAD=str(fault_library), ASTR_CONTROLLER_TEST_FILE=str(controller),
               ASTR_CONTROLLER_TEST_REPLAY=str(replay), ASTR_CONTROLLER_TEST_START_STEP="5")
    result = subprocess.run([str(executable), str(api), str(controller), str(other)],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["initial", "next6", "next7", "next8"]
    assert re.findall(r"completed_step=(\d+)", result.stderr) == ["0", "6", "7", "8"]
    assert controller.read_text() == "unchanged\n" and other.read_text() == "unrelated\n"


def test_controller_replay_fortran_runtime(tmp_path, fault_library):
    compiler = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran")
    if not compiler.is_file():
        pytest.skip("NVHPC input-runtime regression requires nvfortran")
    source = tmp_path / "reader.F90"
    source.write_text("""program reader
  implicit none
  integer :: n,u
  character(1024) :: path
  character(32) :: value
  call get_command_argument(1,path)
  do n=1,4
    open(newunit=u,file=trim(path),action='read')
    read(u,*) value
    close(u)
    print '(A)',trim(value)
  enddo
end program
""")
    executable = tmp_path / "reader"
    subprocess.run([str(compiler), str(source), "-o", str(executable)], check=True, timeout=30)
    controller = tmp_path / "controller"
    controller.write_text("unchanged\n")
    replay = tmp_path / "replay"
    replay.mkdir()
    for step, text in zip((0, 6, 7, 8), ("initial", "next6", "next7", "next8")):
        (replay / f"controller{step:012d}").write_text(text + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASTR_")}
    env.update(LD_PRELOAD=str(fault_library), ASTR_CONTROLLER_TEST_FILE=str(controller),
               ASTR_CONTROLLER_TEST_REPLAY=str(replay), ASTR_CONTROLLER_TEST_START_STEP="5")
    result = subprocess.run([str(executable), str(controller)], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["initial", "next6", "next7", "next8"], result.stderr
    assert re.findall(r"completed_step=(\d+)", result.stderr) == ["0", "6", "7", "8"]


def check_used_clock(case, start, end):
    log = (case / "run.log").read_text()
    records = [(int(step), float(time), float(dt)) for step, time, dt in CLOCK.findall(log)]
    assert len(records) == end - start
    for completed, (label, time, dt) in zip(range(start + 1, end + 1), records):
        assert label == completed - 1
        assert (time, dt) == (TIMES[completed], DT_USED[completed - 1])
    reads = [int(step) for step in re.findall(r"ASTR_CONTROLLER_TEST_READ completed_step=(\d+)", log)]
    assert reads == [0, *range(start + 1, end + 1)]
    return records


def check_checkpoint_clock(path, step):
    with h5py.File(path / "state.h5") as state:
        assert state["identity"][8] == step
        assert tuple(state["identity"][9:12].view(np.float64)) == (
            TIMES[step], DT_USED[step - 1], DT_NEXT[step - 1])
    with h5py.File(path / "statistics.h5") as state:
        region = state["metadata"][4:].view(np.float64)
        duration = min(TIMES[step], 0.0115) - 0.0005
        assert region[2] == TIMES[step] and region[33] == 1
        assert abs(region[7] - duration) <= 32 * math.ulp(duration)
        # GPU owns 16^3 unique periodic nodes; duplicate endpoints are padding.
        nodes = (slice(0, 16),) * 3
        assert np.all(state["q0003"][nodes] == TIMES[step])
        assert np.all(np.abs(state["q0008"][nodes] - duration) <= 32 * math.ulp(duration))
        assert np.all(state["q0034"][nodes] == 1)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("ranks,axis", [(1, "x"), (2, "x"), (2, "y"), (2, "z")])
def test_variable_clock_exact_continuation(backend, ranks, axis, tmp_path, fault_library, record_property):
    if not EXE.is_file() or not MPIEXEC.is_file():
        pytest.skip("build the root solver and provide MPI first")
    args = SimpleNamespace(output=tmp_path, executable=EXE, mpiexec=MPIEXEC,
        case="tgv", mode="steps", restart_step=5, initial_dimension=0,
        legacy_statistics=False, statistics=True, initial_restart=False,
        filter_workspace="scalar", force="feedback", axis=axis, no_samples=True)
    archive = groups("time")

    def launch(name, steps, start=0, restore=None, enabled=True):
        selected = archive if enabled else groups("time", volume=False, slices=False)
        return run_case(args, ROOT, backend, ranks, name, steps, restore=restore,
            buffer_bytes=4096, checkpoint_interval=99, archive_groups=selected,
            controller_replay=(fault_library, start, DT_NEXT[:steps]))[0]

    continuous = launch("continuous", 12)
    seed = launch("seed", 5)
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed = launch("resumed", 12, start=5, restore=source)
    baseline = launch("archives_off", 12, enabled=False)
    for case, start, end in ((continuous, 0, 12), (seed, 0, 5), (resumed, 5, 12), (baseline, 0, 12)):
        check_used_clock(case, start, end)
        checkpoint = case / f"outdat/new/checkpoints/step{end:012d}"
        check_checkpoint_clock(checkpoint, end)
        assert max(controlled_checkpoint_buffers(checkpoint / "state.h5", 0, backend)) < 64 * 1024**2
        with h5py.File(checkpoint / "statistics.h5") as state:
            for row in state["partitions"][:].reshape(-1, 8):
                assert 16 * int(np.prod(row[3:6] + 1)) * 34 + 4096 < 64 * 1024**2
    for filename in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / filename, resumed / FINAL / filename)
        compare_fields(continuous / FINAL / filename, baseline / FINAL / filename)
    for filename in ("control.bin", "insitu_control.bin"):
        assert (continuous / FINAL / filename).read_bytes() == (resumed / FINAL / filename).read_bytes()
    assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(
        resumed / FINAL / "archives.bin")
    expected = {"fields": [0, 2, 3, 4, 5, 6, 8, 12], "slices": [0, 2, 4, 5, 8, 12]}
    reader_frames = {}
    for product in ("fields", "slices"):
        for case, selected in ((continuous, expected[product]), (seed, [s for s in expected[product] if s <= 5]),
                               (resumed, [s for s in expected[product] if s > 5])):
            assert sorted(frames(case, product)) == selected
            assert check_series(case, product) == [TIMES[s] for s in selected]
            check_frame(frames(case, product)[selected[-1]], product,
                case / f"outdat/new/checkpoints/step{selected[-1]:012d}",
                case / "outdat/new/resources/geometry.h5", backend)
        # Compare every resumed physical field, not only the final frame.
        for step, path in frames(resumed, product).items():
            compare_fields(path / "data.h5", frames(continuous, product)[step] / "data.h5")
        reader_frames[product] = check_series_reader(resumed, product)["frames"]
        assert not frames(baseline, product)
    accounting = {name: check_accounting(case, backend, ranks)
                  for name, case in (("continuous", continuous), ("resumed", resumed))}
    # Separate slice-only restart also preserves state and downloads selected planes.
    slice_only, _ = run_case(args, ROOT, backend, ranks, "slice_only", 12, restore=source,
        override=True, buffer_bytes=4096, checkpoint_interval=99,
        archive_groups=groups("time", volume=False), controller_replay=(fault_library, 5, DT_NEXT))
    check_used_clock(slice_only, 5, 12)
    for filename in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / filename, slice_only / FINAL / filename)
    assert not frames(slice_only, "fields")
    accounting["slice_only"] = check_accounting(slice_only, backend, ranks)
    slice_frames = frames(slice_only, "slices")
    assert sorted(slice_frames) == [8, 12]
    for step, path in slice_frames.items():
        compare_fields(path / "data.h5", frames(continuous, "slices")[step] / "data.h5")
    if backend == "gpu":
        assert accounting["slice_only"]["downloaded_field_bytes"] == {"slices": 2 * 3 * 17**2 * 6 * 8}
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    size = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert size < 64 * 1024**2
    with EXE.open("rb") as stream:
        executable_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(backend=backend, np=ranks, axis=axis, dt_used=DT_USED, dt_next=DT_NEXT,
        final_time=TIMES[-1], duration=0.011, frames=expected, reader_frames=reader_frames,
        slice_only_steps=list(slice_frames), accounting=accounting, test_root_bytes=size,
        executable=str(EXE), executable_sha256=executable_sha256,
        exact_state=True, exact_statistics=True, exact_restart_schedule=True, output_switch_unchanged=True)
    (tmp_path / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    record_property("result", report)
