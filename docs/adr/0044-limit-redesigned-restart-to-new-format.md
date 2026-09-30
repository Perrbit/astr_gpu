# Limit redesigned restart to the new format

Accepted for the output redesign on 2026-09-30. The new restart entry and offline checkpoint exporter accept only the new format and explicitly reject legacy checkpoints. Existing simulations using legacy saves remain tied to their compatible solver version; an old-to-new migration tool is outside the first delivery.

This avoids carrying legacy stage-phase and auxiliary-file recovery contracts into the redesigned complete-step interface. Relabeling a legacy save cannot recover missing exact state or change its numerical phase. The decision does not authorize deleting existing source or checkpoint files, or changing running jobs during design.
