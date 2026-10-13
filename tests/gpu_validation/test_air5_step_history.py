"""Actual CPU/GPU ROS-2 optional-history transaction and continuation probe."""
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_optional_history_transaction_and_continuation(tmp_path):
    probe = Path(os.environ.get('ASTR_CHEMISTRY_COMPENSATION_GPU_PROBE_EXE',
                                ROOT / 'build_gpu_probe/bin/chemistry_ros2_compensation_gpu_probe'))
    result = subprocess.run([str(probe), 'history-contract'], capture_output=True,
                            text=True, check=True, timeout=180)
    (tmp_path / 'history-contract.log').write_text(result.stdout + result.stderr)
    for name in ('FIRST_CALL_BASELINE_EXACT', 'INTERVAL_CLIP', 'FINAL_REMAINDER_NOT_REUSED',
                 'REJECTION_AND_PREFIX_ROLLBACK', 'INVALID_BITWISE_ROLLBACK',
                 'NONDEFAULT_CPU_GPU_CONTINUATION_EXACT'):
        assert f'HISTORY_{name}_PASS' in result.stdout
