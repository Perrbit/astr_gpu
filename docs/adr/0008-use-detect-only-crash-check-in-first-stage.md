# Use detect-only crash check in first-stage validation

First-stage GPU validation uses crash checking only to detect and report invalid states, then stops. Automatic crash fixing is deferred because repairing the field would hide the numerical or memory logic error that the GPU port must expose and correct.
