"""Small real device images with display-only coordinate annotations."""
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case
from test_runtime_catalog_images import arguments, dynamic_image, image_segments
from test_insitu_device_products import resident_configuration
from test_index_slice_runtime import ROOT


EXE = os.environ.get("ASTR_COORDINATE_RENDER_EXE")
pytestmark = pytest.mark.skipif(not EXE, reason="Select the current root Catalyst/device executable")


@pytest.mark.parametrize("ranks,pipeline", ((1, "standard-device"), (2, "standard-device"), (2, "direct-device"),
                                          (4, "standard-device")))
def test_real_device_coordinates_are_display_only(tmp_path, ranks, pipeline):
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    config = resident_configuration(pipeline, statistics=False, interval=1, profile="velocity_slice")
    records = [dynamic_image("slice", "native")]
    normalized = deepcopy(records[0]); normalized["id"] = "normalized"
    normalized["display"]["coordinates"] = dict(reference_length=2., origin=[1., 0., 0.])
    exaggerated = deepcopy(records[0]); exaggerated["id"] = "stretched"
    exaggerated["display"]["coordinates"] = dict(display_scale=[1., 2., 1.])
    exaggerated["display"]["parallel_scale"] = 8.
    hidden = deepcopy(records[0]); hidden["id"] = "hidden"
    hidden["display"]["coordinates"] = dict(visible=False)
    records.extend([normalized, exaggerated, hidden, dynamic_image("q_surface", "surface"),
                    dynamic_image("streamlines", "lines")])
    case, _ = run_case(args, ROOT, "gpu", ranks, "coordinates", 2, grid="32,32,32",
                       checkpoint_enabled=False, insitu_config=config, product_catalog_records=records,
                       topology=(2, 2, 1) if ranks == 4 else (ranks, 1, 1))
    status = json.loads((case / "outdat/new/products.status.json").read_text())
    assert all(row["state"] == "active" for row in status["products"]), status
    products = {path.parent.parent.name: path for path in image_segments(case)}
    assert set(products) == {item["id"] for item in records}
    for name, segment in products.items():
        assert len(list(segment.glob("*.jpeg"))) == 2
        for frame in segment.glob("*.jpeg"):
            with Image.open(frame) as image:
                assert image.size == (800, 600)
                assert np.any(np.asarray(image.convert("RGB")) < 230)
        receipts = [json.loads(path.read_text()) for path in segment.glob("frame.*.rank*.json")]
        assert len(receipts) == 2*ranks
        for receipt in receipts:
            assert receipt["field_d2h_bytes"] == receipt["geometry_d2h_bytes"] == receipt["geometry_host_bytes"] == 0
            model = receipt["coordinates"]
            assert model["native_coordinate_unit"] == "1" and not model["gridlines"]
            assert model["visible"] == (name != "hidden")
            if name in ("native", "normalized", "stretched", "hidden"):
                assert model["axis_visible"] == [True, True, False]
            else:
                assert model["axis_visible"] == [True, True, True]
            if name == "normalized":
                assert model["normalized"] and model["reference_length"] == 2.
                assert model["display_bounds"] == model["physical_bounds"]
            if name == "stretched":
                assert model["display_note"] == "y display x2"
                assert model["axis_ranges"][1] == model["physical_bounds"][2:4]
                assert model["display_bounds"][3] == 2*model["physical_bounds"][3]
    with Image.open(next(products["native"].glob("*.jpeg"))) as a, \
            Image.open(next(products["normalized"].glob("*.jpeg"))) as b:
        assert np.any(np.asarray(a) != np.asarray(b))


@pytest.mark.parametrize("ranks", (1, 2))
def test_axis_aligned_physical_plane_uses_two_physical_axes(tmp_path, ranks):
    args = arguments(tmp_path); args.executable = Path(EXE).resolve()
    config = resident_configuration("standard-device", statistics=False, interval=1, profile="velocity_slice")
    item = dynamic_image("physical_plane", "plane")
    item["display"]["coordinates"] = dict(reference_length=2., origin=[1., 0., 0.])
    case, _ = run_case(args, ROOT, "gpu", ranks, "physical-plane", 2, grid="32,32,32",
                       checkpoint_enabled=False, insitu_config=config, product_catalog_records=[item])
    segment = image_segments(case)[0]
    receipts = [json.loads(path.read_text()) for path in segment.glob("frame.*.rank*.json")]
    assert len(receipts) == 2*ranks
    for receipt in receipts:
        model = receipt["coordinates"]
        assert model["axis_visible"] == [True, True, False]
        assert model["normalized"] and model["reference_length"] == 2.
        assert model["physical_bounds"] == model["display_bounds"]
        assert receipt["geometry_d2h_bytes"] == receipt["field_d2h_bytes"] == 0


def test_unrepresentable_coordinate_range_disables_only_that_product(tmp_path):
    args = arguments(tmp_path); args.executable = Path(EXE).resolve()
    config = resident_configuration("standard-device", statistics=False, interval=1, profile="velocity_slice")
    bad = dynamic_image("slice", "unrepresentable")
    bad["display"]["coordinates"] = dict(display_scale=[1e308, 1., 1.])
    healthy = dynamic_image("slice", "healthy")
    case, _ = run_case(args, ROOT, "gpu", 2, "isolation", 2, grid="32,32,32", checkpoint_enabled=False,
                       insitu_config=config, product_catalog_records=[bad, healthy])
    status = {row["id"]: row for row in json.loads((case / "outdat/new/products.status.json").read_text())["products"]}
    assert status["unrepresentable"]["state"] == "resource-disabled"
    assert "not finite" in status["unrepresentable"]["reason"]
    assert status["healthy"]["state"] == "active" and status["healthy"]["last_output_step"] == 2
    assert len(list((case / "outdat/new/render_products/healthy").rglob("*.jpeg"))) == 2
    assert not list((case / "outdat/new/render_products/unrepresentable").rglob("*.jpeg"))
