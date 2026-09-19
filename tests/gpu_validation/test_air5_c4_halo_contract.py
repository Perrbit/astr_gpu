from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compact(text: str) -> str:
    return "".join(text.lower().split())


def test_production_halo_contract_covers_full_air5_state() -> None:
    source = compact(
        (ROOT / "tests/gpu_validation/halo_exchange_contract_test.cuf").read_text(
            encoding="utf-8"
        )
    )

    assert "numq=11" in source
    assert "components(4)=[1,3,5,6]" in source
    assert "callreference(11,.true.)" in source
    assert "callcheck_values(11,'qswaphm+1andshared-planeaveraging')" in source


def test_air5_conservation_uses_unique_cells_and_small_device_record() -> None:
    source = compact(
        (ROOT / "src_gpu/chemistry_solver_gpu.cuf").read_text(encoding="utf-8")
    )

    assert "astr_air5_c4_conservation" in source
    assert "ntot=im*jm*km" in source
    assert "air5_conservation_totals_d(air5_num_conservative)" in source
    assert "callmpi_allreduce(local_totals,global_totals" in source
    assert "callrecord_air5_conservation_gpu(nstep+1)" in source
