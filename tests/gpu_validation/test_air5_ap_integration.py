"""AP engineering changes against the existing qualified G2 flow reference."""
import json
import os
from pathlib import Path

import pytest

from air5_radau_reference import Air5RadauReference
from check_air5_flow_actual_error import compare_checkpoints, require_flow_error
from run_air5_flow_control_gate import load_references, same_initial
from run_output_air5_restart_validation import ROOT, launch, state_sanity
from run_output_restart_validation import compare_fields
from test_output_air5_conservation import arguments


@pytest.fixture(scope="module")
def references():
    path = os.environ.get("ASTR_AIR5_AP_REFERENCE_SUMMARY")
    if not path:
        pytest.skip("ASTR_AIR5_AP_REFERENCE_SUMMARY is not set")
    return load_references(Path(path), "conservative-flux")


def run_g2_case(tmp_path, ranks, launch_tuning="fixed", chemistry_dispatch="general", flux_halo="separate",low_activity="off"):
    args = arguments(tmp_path, case="sbli", initial=True)
    args.grid = (64, 64, 64)
    args.mean_statistics = False
    args.air5_filter_mode = "conservative-flux"
    args.source_evaluation = "source-only"
    args.chemistry_block_size = 32
    args.primitive_reuse = "chemistry"
    args.chemistry_reductions = "packed"
    args.report_interval = 4
    args.runtime_timeout_seconds = 1800
    args.launch_tuning = launch_tuning
    args.chemistry_dispatch = chemistry_dispatch
    args.flux_halo = flux_halo
    args.low_activity = low_activity
    case, _ = launch(args, "gpu", ranks, "g2", 24, interval=24,
                     chemistry_diagnostics="report-steps",
                     host_budget_bytes=256 * 1024**2,
                     device_budget_bytes=256 * 1024**2,
                     directory_budget_bytes=1024**3)
    state = case / "outdat/new/checkpoints/step000000000024/state.h5"
    return case, state


@pytest.mark.parametrize("ranks", [1, 2])
def test_existing_g2_reference_margin(tmp_path, references, ranks):
    case, state = run_g2_case(tmp_path, ranks)
    same_initial(case / "outdat/new/checkpoints/step000000000000/state.h5", references["initial"])
    model = Air5RadauReference(ROOT / "chemMech/air5_kimjo12.json")
    error = compare_checkpoints(model, references["initial"],
                                references["references"]["12"]["state"],
                                references["references"]["11"]["state"],
                                state, references["freestream"])
    require_flow_error(error)
    result = dict(scope="existing G2 outer-discretization reference; not physical validation",
                  ranks=ranks, sanity=state_sanity(state), error=error)
    (case / "actual_error.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize("ranks", [1, 2])
def test_auto_keeps_saved_g2_fields(tmp_path, ranks):
    value=os.environ.get("ASTR_AIR5_AP_BASELINE_STATES")
    if not value:
        pytest.skip("ASTR_AIR5_AP_BASELINE_STATES is not set")
    reference=Path(json.loads(value)[str(ranks)])
    case,state=run_g2_case(tmp_path,ranks,"auto")
    equal=compare_fields(state,reference)
    (case/"auto_field_identity.json").write_text(json.dumps(dict(
        scope="saved-state bitwise identity; numerical qualification is recorded separately",
        reference=str(reference),equal_datasets=equal,sanity=state_sanity(state)),
        indent=2,allow_nan=False)+"\n")


@pytest.mark.parametrize("ranks", [1, 2])
def test_specialized_auto_keeps_saved_g2_fields(tmp_path, ranks):
    value=os.environ.get("ASTR_AIR5_AP_BASELINE_STATES")
    if not value:
        pytest.skip("ASTR_AIR5_AP_BASELINE_STATES is not set")
    reference=Path(json.loads(value)[str(ranks)])
    case,state=run_g2_case(tmp_path,ranks,"auto","specialized")
    equal=compare_fields(state,reference)
    (case/"specialized_field_identity.json").write_text(json.dumps(dict(
        scope="saved-state bitwise identity; numerical qualification is recorded separately",
        reference=str(reference),equal_datasets=equal,sanity=state_sanity(state)),
        indent=2,allow_nan=False)+"\n")


@pytest.mark.parametrize("ranks", [1, 2])
def test_packed_overlap_keeps_saved_g2_fields(tmp_path, ranks):
    value=os.environ.get("ASTR_AIR5_AP_BASELINE_STATES")
    if not value:
        pytest.skip("ASTR_AIR5_AP_BASELINE_STATES is not set")
    reference=Path(json.loads(value)[str(ranks)])
    case,state=run_g2_case(tmp_path,ranks,"auto","specialized","packed-overlap")
    equal=compare_fields(state,reference)
    (case/"packed_field_identity.json").write_text(json.dumps(dict(
        scope="saved-state bitwise identity; numerical qualification is recorded separately",
        reference=str(reference),equal_datasets=equal,sanity=state_sanity(state)),
        indent=2,allow_nan=False)+"\n")


@pytest.mark.parametrize('ranks',[1,2])
def test_activity_independent_g2_holdout(tmp_path,references,ranks):
    threshold=os.environ.get('ASTR_AIR5_AP_ACTIVITY_THRESHOLD')
    if not threshold:
        pytest.skip('run only after the activity threshold has been frozen')
    case,state=run_g2_case(tmp_path,ranks,'auto','specialized','packed-overlap',threshold)
    same_initial(case/'outdat/new/checkpoints/step000000000000/state.h5',references['initial'])
    baseline=os.environ.get('ASTR_AIR5_AP_BASELINE_STATES')
    identity=None
    if baseline:
        saved=Path(json.loads(baseline)[str(ranks)])
        try:
            identity=compare_fields(state,saved)
        except AssertionError as exc:
            if not str(exc).startswith('exact state mismatch:'):
                raise
            # Approximate candidates need actual error evaluation, not old-code bit identity.
            identity=None
    result=dict(threshold=float(threshold),ranks=ranks,sanity=state_sanity(state))
    if identity is not None:
        result.update(error_basis='all saved-state datasets bitwise equal to the separately G2-qualified field',
                      reference=str(saved),equal_datasets=identity)
    else:
        model=Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
        error=compare_checkpoints(model,references['initial'],references['references']['12']['state'],
                                 references['references']['11']['state'],state,references['freestream'])
        require_flow_error(error)
        result.update(error_basis='direct comparison with immutable independent G2 references',error=error)
    (case/'activity_g2_holdout.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
