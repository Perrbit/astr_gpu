from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compact(text: str) -> str:
    return "".join(text.lower().split())


def test_reacting_flowtype_policy_is_shared_by_cpu_and_gpu() -> None:
    runtime = compact(
        (ROOT / "src/chemistry_flow_runtime.F90").read_text(encoding="utf-8")
    )
    cpu = compact((ROOT / "src/mainloop.F90").read_text(encoding="utf-8"))
    gpu = compact((ROOT / "src_gpu/mainloop_gpu.cuf").read_text(encoding="utf-8"))

    assert "purelogicalfunctionair5_reacting_flowtype(flowtype)" in runtime
    assert "case('air5reactor','air5postshock','air5tgv','air5hbl')" in runtime
    assert "air5_reacting_case=lcomb.and.air5_reacting_flowtype(flowtype)" in cpu
    assert "if(air5_reacting_flowtype(flowtype))then" in gpu


def test_source_mode_is_runtime_selected_and_shared_by_cpu_and_gpu() -> None:
    runtime = compact(
        (ROOT / "src/chemistry_flow_runtime.F90").read_text(encoding="utf-8")
    )
    cpu = compact(
        (ROOT / "src/chemistry_flow_solver.F90").read_text(encoding="utf-8")
    )
    gpu = compact(
        (ROOT / "src_gpu/chemistry_coupling_gpu.cuf").read_text(encoding="utf-8")
    )

    assert "astr_air5_source_mode" in runtime
    for mode in ("coupled", "chemical", "vt"):
        assert f"case('{mode}')" in runtime
    assert "source_mode=air5_active_source_mode()" in cpu
    assert "air5_ros2_advance(" in cpu
    assert "status,source_mode)" in cpu
    assert "source_mode=air5_active_source_mode()" in gpu
    assert "air5_chemistry_half_step_kernel<<<grid,block>>>(duration,source_mode" in gpu
    assert "status,source_mode)" in gpu


def test_cpu_strang_sequence_wraps_transport_once() -> None:
    source = compact((ROOT / "src/mainloop.F90").read_text(encoding="utf-8"))

    first = source.index("callair5_chemistry_half_step(0.5_real64*deltat,1)")
    transport = source.index("dorkstep=1,n_rk_steps", first)
    second = source.index("callair5_chemistry_half_step(0.5_real64*deltat,2)", transport)
    assert first < transport < second
    assert "callqswap(timerept=ltimrpt)" in source[first:transport]
    assert "callqswap(timerept=ltimrpt)" not in source[second : second + 300]


def test_cpu_chemistry_integrates_only_active_nodes() -> None:
    source = compact(
        (ROOT / "src/chemistry_flow_solver.F90").read_text(encoding="utf-8")
    )

    assert "subroutineair5_chemistry_half_step(duration,half_index)" in source
    assert "dok=ks,ke" in source
    assert "doj=js,je" in source
    assert "doi=is,ie" in source
    assert "callair5_ros2_advance(" in source


def test_gpu_strang_sequence_reprepares_transport_state() -> None:
    source = compact(
        (ROOT / "src_gpu/mainloop_gpu.cuf").read_text(encoding="utf-8")
    )

    first = source.index("callair5_chemistry_half_step_gpu(0.5_real64*deltat,1)")
    transport = source.index("callair5_transport_rk3_step_gpu(.true.,.false.)", first)
    second = source.index("callair5_chemistry_half_step_gpu(0.5_real64*deltat,2)", transport)
    assert first < transport < second


def test_gpu_chemistry_kernel_uses_active_nodes_and_explicit_sync() -> None:
    source = compact(
        (ROOT / "src_gpu/chemistry_coupling_gpu.cuf").read_text(encoding="utf-8")
    )

    assert "attributes(global)subroutineair5_chemistry_half_step_kernel" in source
    assert "callair5_ros2_advance_gpu(" in source
    assert "air5_chemistry_half_step_kernel<<<grid,block>>>" in source
    assert "call sync_after_kernel('air5_chemistry_half_step_kernel',.true.)".replace(
        " ", ""
    ) in source


def test_reactor_case_uses_fixed_air5_initializer() -> None:
    initializer = compact(
        (ROOT / "src/initialisation.F90").read_text(encoding="utf-8")
    )
    grid = compact((ROOT / "src/gridgeneration.F90").read_text(encoding="utf-8"))
    preparer = (
        ROOT / "tests/gpu_validation/prepare_air5_c4_case.py"
    ).read_text(encoding="utf-8")

    assert "case('air5reactor')" in initializer
    assert "callair5reactorini" in initializer
    assert "subroutineair5reactorini" in initializer
    assert "trim(flowtype)=='air5reactor'" in grid
    for choice in ("tgv", "species-wave", "reactor"):
        assert f'"{choice}"' in preparer


def test_high_temperature_tgv_uses_dedicated_air5_initializer() -> None:
    initializer = compact(
        (ROOT / "src/initialisation.F90").read_text(encoding="utf-8")
    )
    grid = compact((ROOT / "src/gridgeneration.F90").read_text(encoding="utf-8"))
    preparer = compact(
        (ROOT / "tests/gpu_validation/prepare_air5_c4_case.py").read_text(
            encoding="utf-8"
        )
    )

    assert "case('air5tgv')" in initializer
    assert "callair5tgvini" in initializer
    assert "subroutineair5tgvini" in initializer
    assert "trim(flowtype)=='air5tgv'" in grid
    assert '"high-temperature-tgv"' in preparer
    assert 'replace_after_marker(lines,"flowtype","air5tgv")' in preparer


def test_high_temperature_tgv_initial_state_is_explicitly_bounded() -> None:
    initializer = compact(
        (ROOT / "src/initialisation.F90").read_text(encoding="utf-8")
    )
    start = initializer.index("subroutineair5tgvini")
    end = initializer.index("endsubroutineair5tgvini", start)
    source = initializer[start:end]

    assert "target_density=5.0e-2_real64" in source
    assert "target_temperature=6.0e3_real64" in source
    assert "target_tv=1.0e3_real64" in source
    assert "velocity_amplitude=1.0e2_real64" in source
    assert "target_mass_fraction(air5_num_species)=" in source
    assert "tmp(i,j,k)=prs(i,j,k)/(rho(i,j,k)*mixture_gas_constant)" in source
