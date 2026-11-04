"""Explicit final conversion of positional test fixtures, never a solver fallback."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/input'))
from convert_legacy_input import MODEL_ENVIRONMENT,convert_case,convert_controller,namelist


def prepare_named_run(case,primary,environment):
    """Call after all fixture edits; preserve legacy fixed dt and update count."""
    case=Path(case)
    main=namelist(convert_case((case/primary).read_text(),model_environment=environment))
    run=namelist(convert_controller((case/'datin/controller').read_text()))
    (case/'datin/input.case').write_text(main)
    (case/'datin/input.run').write_text(run)
    for name in MODEL_ENVIRONMENT:
        environment.pop(name,None)
    return 'datin/input.case'
