from prepare_opensbli_astr_case import prepare_case


def test_static_profile_periodic_spanwise_restart_case(tmp_path):
    case = tmp_path / "case"
    prepare_case(case, "t", 33, 33, 9, 2, 2, 1e-6, .72)
    text = (case / "datin/input.opensbli").read_text()
    boundary = text.split("# bctype\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert boundary[0] == "11,prof"
    assert boundary[1] == "21"
    assert boundary[3:] == ["52", "1", "1"]
    for name in ("grid.flatplate.h5", "flowini3d.h5", "inlet.prof",
                 "conservative_boundary.nml", "controller"):
        assert (case / "datin" / name).is_file()
