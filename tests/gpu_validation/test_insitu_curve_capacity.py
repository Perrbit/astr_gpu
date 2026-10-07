"""Bounded 256^3 index/message audit, distinct from actual memory admission."""
import json
import os
from pathlib import Path
import re

import pytest


@pytest.mark.parametrize('cells', [64, 128, 256])
def test_flying_edges_slab_capacity(cells, record_property):
    root = Path(__file__).resolve().parents[2]
    source = Path(os.environ.get('ASTR_INSITU_VISKORES_SOURCE',
        root.parent/'astr_dependencies/ParaView-v6.1.1-astr-device/VTK/ThirdParty/viskores/vtkviskores/viskores'))
    table = source/'viskores/filter/contour/worklet/contour/FlyingEdgesTables.h'
    if not table.is_file():
        pytest.skip('Select the actual private Viskores source for the table audit')
    body = re.search(r'numTris\[256\]\s*=\s*\{([^}]+)\}', table.read_text())
    assert body, 'The actual FlyingEdges primitive table changed'
    counts = [int(n) for n in re.findall(r'\d+', body[1])]
    assert len(counts) == 256 and max(counts) == 5
    extent = (cells//2, cells, cells)
    owned_nodes = (extent[0]+1)*(extent[1]+1)*(extent[2]+1)
    halo_nodes = (extent[0]+7)*(extent[1]+7)*(extent[2]+7)
    volume_cells = extent[0]*extent[1]*extent[2]
    triangles = volume_cells*max(counts)
    indices = 3*triangles
    # FlyingEdges metadata scans Int32 triangle counts; the GL draw is signed Int32.
    assert triangles <= 2**31-1 and indices <= 2**31-1
    # Even a non-deduplicated three-points-per-triangle bound fits UInt32 display IDs.
    assert indices <= 2**32-1 and 14*halo_nodes <= 2**31-1
    maximum_vector_face_values = 3*(cells+7)**2*3
    assert maximum_vector_face_values <= 2**31-1
    record_property('scale_capacity', json.dumps(dict(cells=cells, topology=[2, 1, 1],
        owned_nodes=owned_nodes, halo_nodes=halo_nodes, volume_cells=volume_cells,
        maximum_triangles=triangles, maximum_draw_indices=indices,
        maximum_vector_face_values=maximum_vector_face_values,
        note='Index limits only; buffers still require actual allocation preflight and measured budgets')))
