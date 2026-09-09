#!/usr/bin/env python3
"""Integration test for segmented TGV production orchestration."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/gpu_validation/run_tgv_512_production.sh"


class Tgv512ProductionDriverTests(unittest.TestCase):
    def test_two_segments_restart_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            bin_dir = work / "bin"
            bin_dir.mkdir()
            mpirun = bin_dir / "mpirun"
            solver = work / "fake_astr"
            dt_one = work / "cfl_dt_one.txt"
            reference = work / "reference.gdiag"
            output = work / "production"

            mpirun.write_text(
                "#!/usr/bin/env bash\nshift 2\nexec \"$@\"\n",
                encoding="ascii",
            )
            solver.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
controller=datin/controller
input=datin/input.tgv
maxstep=$(awk '/^# maxstep,feqchkpt/{getline; gsub(/[[:space:]]/,\"\"); split($0,a,\",\"); print a[1]}' "$controller")
dt=$(awk '/^# deltat/{getline; gsub(/[dD]/,\"e\"); print $1}' "$controller")
restart=$(awk '/lrestar/{getline; gsub(/[[:space:]]/,\"\"); print; exit}' "$input")
start=0
mkdir -p outdat bakup
if [[ "$restart" == t ]]; then
  start=$(awk -F= '/nstep=/{gsub(/[[:space:]]/,\"\",$2); print $2; exit}' outdat/auxiliary.txt)
  cp outdat/auxiliary.txt bakup/auxiliary.txt
  cp outdat/flowfield.h5 bakup/flowfield.h5
else
  printf 'nstep time kenergy enstophy dissipation\\n' > flowstate.dat
fi
python3 - "$start" "$maxstep" "$dt" >> flowstate.dat <<'PY'
import sys
start, end = map(int, sys.argv[1:3])
dt = float(sys.argv[3])
for step in range(start + (start > 0), end + 1):
    t = step * dt
    print(step, t, 0.125 - 10.0*t, 0.3 + 100.0*t, 0.001 + 2.0*t)
PY
printf '&restart\\nnstep=%s\\nfilenumb=0\\nfnumslic=0\\nninflowslice=0\\nnsamples=0\\n/End\\n' "$maxstep" > outdat/auxiliary.txt
printf 'checkpoint step %s\\n' "$maxstep" > outdat/flowfield.h5
printf '           current CFL:     0.5000000\\n'
printf '   time step for CFL=1:   0.10000E-02\\n'
printf 'The job is done!\\n'
""",
                encoding="ascii",
            )
            mpirun.chmod(0o755)
            solver.chmod(0o755)
            dt_one.write_text("0.001\n", encoding="ascii")
            reference.write_text(
                "# Time Energy Dissipation Enstrophy\n"
                "0.000 0.125 0.001 0.3\n"
                "0.001 0.115 0.003 0.4\n"
                "0.002 0.105 0.005 0.5\n",
                encoding="ascii",
            )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "ROOT_DIR": str(ROOT),
                    "GPU_EXE": str(solver),
                    "OUT_DIR": str(output),
                    "CFL_DT_ONE_FILE": str(dt_one),
                    "DLR_REFERENCE": str(reference),
                    "PYTHON_EXE": sys.executable,
                    "TARGET_TIME": "0.002",
                    "TARGET_CFL": "0.50",
                    "SEGMENT_STEPS": "2",
                    "MAX_RETRIES": "1",
                }
            )

            subprocess.run([str(DRIVER)], env=env, check=True, capture_output=True, text=True)

            statuses = (output / "segment_status.tsv").read_text(encoding="ascii")
            self.assertIn("0\t2\t0\tPASS", statuses)
            self.assertIn("2\t4\t0\tPASS", statuses)
            self.assertTrue((output / "dlr_comparison/metrics.json").is_file())
            self.assertTrue((output / "dlr_comparison/tgv_dlr_comparison.eps").is_file())
            self.assertTrue((output / "dlr_comparison/tgv_dlr_comparison.jpeg").is_file())


if __name__ == "__main__":
    unittest.main()
