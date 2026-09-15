"""Independent JSON-driven zero-dimensional reference for the fixed air5 model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class Air5RadauReference:
    RU = 8.3144626181532403
    AVOGADRO = 6.02214076e23
    ATMOSPHERE_PA = 101325.0

    def __init__(self, mechanism_path: Path):
        self.data = json.loads(Path(mechanism_path).read_text(encoding="ascii"))
        if self.data.get("mechanism_id") != "air5_kimjo12":
            raise ValueError("unsupported mechanism identifier")
        self.names = tuple(self.data["species_order"])
        if self.names != ("N2", "O2", "N", "O", "NO"):
            raise ValueError("unexpected species order")
        species = self.data["species"]
        self.molar_mass = np.array(
            [item["molecular_weight_kg_per_mol"] for item in species], dtype=float
        )
        self.formation_energy = np.array(
            [item["formation_energy_j_per_kg"] for item in species], dtype=float
        )
        self.cv_factor = np.array(
            [item["cv_tr_over_species_gas_constant"] for item in species], dtype=float
        )
        self.theta_v = np.array(
            [item["vibrational_temperature_k"] for item in species], dtype=float
        )
        self.atom_counts = np.array(
            [[item["atoms"]["N"], item["atoms"]["O"]] for item in species],
            dtype=float,
        )
        self.gas_constant = self.RU / self.molar_mass
        self.cv_tr = self.cv_factor * self.gas_constant
        self.temperature_bounds = tuple(self.data["validity"]["temperature_k"])
        self.pressure_bounds = tuple(self.data["validity"]["pressure_pa"])
        relaxation = self.data["vibrational_relaxation"]
        self.mw_a = np.asarray(relaxation["millikan_white_A"], dtype=float)
        self.mw_b = np.asarray(relaxation["millikan_white_B"], dtype=float)
        self.sigma0 = np.asarray(relaxation["collision_sigma0_m2"], dtype=float)
        self.sigma_power = np.asarray(
            relaxation["collision_sigma_temperature_power"], dtype=float
        )
        self.reactions = tuple(self.data["reactions"])
        preferential = self.data["preferential_dissociation"]
        self.psi_coefficients = {
            self.names.index(name): np.asarray(entry["coefficients"], dtype=float)
            for name, entry in preferential.items()
        }
        self.psi_bounds = {
            self.names.index(name): tuple(entry["bounds"])
            for name, entry in preferential.items()
        }

    def _validate_species(self, rho_species: np.ndarray) -> np.ndarray:
        values = np.asarray(rho_species, dtype=float)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError("species densities must be five finite values")
        if np.any(values < 0.0) or float(np.sum(values)) <= np.finfo(float).tiny:
            raise ValueError("species densities must be nonnegative with positive sum")
        return values

    def _validate_temperature(self, temperature: float) -> float:
        value = float(temperature)
        lower, upper = self.temperature_bounds
        if not math.isfinite(value) or value < lower or value > upper:
            raise ValueError("temperature is outside the fixed mechanism domain")
        return value

    def species_vibrational_energy(self, species: int, temperature: float) -> float:
        temperature = self._validate_temperature(temperature)
        return self._species_vibrational_energy_raw(species, temperature)

    def _species_vibrational_energy_raw(self, species: int, temperature: float) -> float:
        theta = self.theta_v[species]
        if theta == 0.0:
            return 0.0
        return self.gas_constant[species] * theta / math.expm1(theta / temperature)

    def ev_from_tv(self, rho_species: np.ndarray, tv: float) -> float:
        rho_species = self._validate_species(rho_species)
        tv = self._validate_temperature(tv)
        if float(np.sum(rho_species[self.theta_v > 0.0])) <= np.finfo(float).tiny:
            raise ValueError("species state has no vibrational capacity")
        return self._ev_from_tv_raw(rho_species, tv)

    def _ev_from_tv_raw(self, rho_species: np.ndarray, tv: float) -> float:
        return float(sum(
            rho_species[index] * self._species_vibrational_energy_raw(index, tv)
            for index in range(5)
        ))

    def tv_from_ev(self, rho_species: np.ndarray, ev: float) -> float:
        rho_species = self._validate_species(rho_species)
        ev = float(ev)
        if not math.isfinite(ev) or ev < 0.0:
            raise ValueError("vibrational energy is invalid")
        return self._tv_from_ev_raw(rho_species, ev)

    def _tv_from_ev_raw(self, rho_species: np.ndarray, ev: float) -> float:
        lower, upper = self.temperature_bounds
        if float(np.sum(rho_species[self.theta_v > 0.0])) <= np.finfo(float).tiny:
            raise ValueError("species state has no vibrational capacity")
        lower_ev = self._ev_from_tv_raw(rho_species, lower)
        upper_ev = self._ev_from_tv_raw(rho_species, upper)
        tolerance = 8.0 * np.finfo(float).eps * max(abs(lower_ev), abs(ev), 1.0)
        if ev < lower_ev - tolerance or ev > upper_ev + tolerance:
            raise ValueError("vibrational energy is outside the fixed temperature domain")
        if abs(ev - lower_ev) <= tolerance:
            return lower
        if abs(ev - upper_ev) <= tolerance:
            return upper
        return float(brentq(
            lambda trial: self._ev_from_tv_raw(rho_species, trial) - ev,
            lower,
            upper,
            xtol=1.0e-12,
            rtol=4.0 * np.finfo(float).eps,
        ))

    def q5_from_state(
        self,
        rho: float,
        momentum: np.ndarray,
        rho_species: np.ndarray,
        ev: float,
        temperature: float,
    ) -> float:
        rho_species = self._validate_species(rho_species)
        temperature = self._validate_temperature(temperature)
        rho = float(rho)
        momentum = np.asarray(momentum, dtype=float)
        if not math.isfinite(rho) or rho <= 0.0 or momentum.shape != (3,):
            raise ValueError("density or momentum is invalid")
        if not np.all(np.isfinite(momentum)) or not math.isfinite(ev) or ev < 0.0:
            raise ValueError("momentum or vibrational energy is invalid")
        kinetic = float(np.dot(momentum, momentum)) / (2.0 * rho)
        return float(
            kinetic
            + np.dot(rho_species, self.cv_tr) * temperature
            + ev
            + np.dot(rho_species, self.formation_energy)
        )

    def temperature_from_q5(
        self,
        rho: float,
        momentum: np.ndarray,
        rho_species: np.ndarray,
        ev: float,
        q5: float,
    ) -> float:
        rho_species = self._validate_species(rho_species)
        rho = float(rho)
        momentum = np.asarray(momentum, dtype=float)
        if not math.isfinite(rho) or rho <= 0.0 or momentum.shape != (3,):
            raise ValueError("density or momentum is invalid")
        if not np.all(np.isfinite(momentum)) or not math.isfinite(ev) or ev < 0.0:
            raise ValueError("momentum or vibrational energy is invalid")
        if not math.isfinite(q5):
            raise ValueError("complete total energy is invalid")
        temperature = self._temperature_from_q5_raw(
            rho, momentum, rho_species, ev, q5
        )
        return self._validate_temperature(temperature)

    def _temperature_from_q5_raw(
        self,
        rho: float,
        momentum: np.ndarray,
        rho_species: np.ndarray,
        ev: float,
        q5: float,
    ) -> float:
        cv_density = float(np.dot(rho_species, self.cv_tr))
        if cv_density <= 0.0:
            raise ValueError("species state has no translational heat capacity")
        kinetic = float(np.dot(momentum, momentum)) / (2.0 * rho)
        return float(
            (q5 - kinetic - ev - np.dot(rho_species, self.formation_energy))
            / cv_density
        )

    def pressure(self, rho_species: np.ndarray, temperature: float) -> float:
        rho_species = self._validate_species(rho_species)
        temperature = self._validate_temperature(temperature)
        pressure = self._pressure_raw(rho_species, temperature)
        if not self._pressure_in_domain(pressure):
            raise ValueError("pressure is outside the fixed mechanism domain")
        return pressure

    def _pressure_in_domain(self, pressure: float) -> bool:
        lower, upper = self.pressure_bounds
        tolerance = 64.0 * np.finfo(float).eps * max(abs(upper), 1.0)
        return math.isfinite(pressure) and (
            lower - tolerance <= pressure <= upper + tolerance
        )

    def _pressure_raw(self, rho_species: np.ndarray, temperature: float) -> float:
        return float(np.dot(rho_species, self.gas_constant) * temperature)

    def _mass_action(self, concentration: np.ndarray, entries: dict[str, int]) -> float:
        result = 1.0
        for name, exponent in entries.items():
            result *= concentration[self.names.index(name)] ** exponent
        return float(result)

    def _preferential_factor(self, species: int, temperature: float) -> float:
        coefficients = self.psi_coefficients[species]
        exponent = (
            coefficients[0] / temperature
            + coefficients[1]
            + coefficients[2] * math.log(temperature)
            + coefficients[3] * temperature
            + coefficients[4] * temperature**2
        )
        lower, upper = self.psi_bounds[species]
        return min(max(math.exp(exponent), lower), upper)

    def _vt_source(
        self, rho_species: np.ndarray, temperature: float, tv: float, pressure: float
    ) -> float:
        concentration = rho_species / self.molar_mass
        mole_fraction = concentration / np.sum(concentration)
        result = 0.0
        for molecule in range(5):
            if self.theta_v[molecule] <= 0.0 or rho_species[molecule] <= 0.0:
                continue
            number_density = concentration[molecule] * self.AVOGADRO
            pair_time = np.empty(5)
            for partner in range(5):
                mw_time = self.ATMOSPHERE_PA / pressure * math.exp(
                    self.mw_a[molecule, partner]
                    * (temperature ** (-1.0 / 3.0) - self.mw_b[molecule, partner])
                    - 18.42
                )
                park_time = 0.0
                if self.sigma0[molecule, partner] > 0.0:
                    sigma = self.sigma0[molecule, partner] * temperature ** (
                        self.sigma_power[molecule, partner]
                    )
                    reduced_mass = (
                        self.molar_mass[molecule]
                        * self.molar_mass[partner]
                        / (self.molar_mass[molecule] + self.molar_mass[partner])
                    )
                    mean_speed = math.sqrt(
                        8.0 * self.RU * temperature / (math.pi * reduced_mass)
                    )
                    park_time = 1.0 / (number_density * sigma * mean_speed)
                pair_time[partner] = mw_time + park_time
            inverse_time = float(np.sum(mole_fraction / pair_time))
            equilibrium = self._species_vibrational_energy_raw(molecule, temperature)
            current = self._species_vibrational_energy_raw(molecule, tv)
            result += rho_species[molecule] * (equilibrium - current) * inverse_time
        return result

    def invariant_drifts(
        self, initial_state: np.ndarray, final_state: np.ndarray
    ) -> tuple[float, float, float]:
        initial_state = np.asarray(initial_state, dtype=float)
        final_state = np.asarray(final_state, dtype=float)
        if initial_state.shape != (6,) or final_state.shape != (6,):
            raise ValueError("invariant states must contain six values")
        initial_species = self._validate_species(initial_state[:5])
        final_species = self._validate_species(final_state[:5])
        mass_drift = abs(np.sum(final_species) - np.sum(initial_species)) / np.sum(
            initial_species
        )
        initial_elements = self.atom_counts.T @ (initial_species / self.molar_mass)
        final_elements = self.atom_counts.T @ (final_species / self.molar_mass)
        element_drift = np.abs(final_elements - initial_elements) / np.maximum(
            np.abs(initial_elements), np.finfo(float).tiny
        )
        return float(mass_drift), float(element_drift[0]), float(element_drift[1])

    def temperatures(
        self,
        rho: float,
        momentum: np.ndarray,
        q5: float,
        states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[0] != 6:
            raise ValueError("trajectory states must have shape (6, n)")
        temperature = np.empty(states.shape[1])
        vibrational_temperature = np.empty(states.shape[1])
        for index in range(states.shape[1]):
            temperature[index] = self.temperature_from_q5(
                rho, momentum, states[:5, index], states[5, index], q5
            )
            vibrational_temperature[index] = self.tv_from_ev(
                states[:5, index], states[5, index]
            )
        return temperature, vibrational_temperature

    def integrate_radau(
        self,
        rho: float,
        momentum: np.ndarray,
        q5: float,
        initial_state: np.ndarray,
        duration: float,
        *,
        rtol: float,
        atol: np.ndarray,
        source_mode: str = "coupled",
    ):
        initial_state = np.asarray(initial_state, dtype=float)
        atol = np.asarray(atol, dtype=float)
        if initial_state.shape != (6,) or not np.all(np.isfinite(initial_state)):
            raise ValueError("initial state must contain six finite values")
        initial_species = self._validate_species(initial_state[:5])
        if abs(float(np.sum(initial_species)) - rho) > 1.0e-12 * rho:
            raise ValueError("initial species densities do not sum to density")
        if initial_state[5] < 0.0:
            raise ValueError("initial vibrational energy is invalid")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("integration duration must be finite and positive")
        if not math.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("relative tolerance must be finite and positive")
        if atol.shape != (6,) or not np.all(np.isfinite(atol)) or np.any(atol <= 0.0):
            raise ValueError("absolute tolerances must be six finite positive values")
        self.temperature_from_q5(rho, momentum, initial_species, initial_state[5], q5)
        self.tv_from_ev(initial_species, initial_state[5])

        result = solve_ivp(
            lambda time, state: self._rhs(
                time,
                state,
                rho,
                momentum,
                q5,
                enforce_nonnegative=False,
                source_mode=source_mode,
            ),
            (0.0, duration),
            initial_state,
            method="Radau",
            rtol=rtol,
            atol=atol,
        )
        if not result.success:
            raise RuntimeError(f"Radau integration failed: {result.message}")
        if not np.all(np.isfinite(result.y)):
            raise RuntimeError("Radau trajectory contains nonfinite values")
        if np.any(result.y[:5, :] < 0.0) or np.any(result.y[5, :] < 0.0):
            raise RuntimeError("Radau trajectory left the admissible state domain")
        self.temperatures(rho, momentum, q5, result.y)
        return result

    def rhs(
        self,
        _time: float,
        state: np.ndarray,
        rho: float,
        momentum: np.ndarray,
        q5: float,
        *,
        source_mode: str = "coupled",
    ) -> np.ndarray:
        return self._rhs(
            _time,
            state,
            rho,
            momentum,
            q5,
            enforce_nonnegative=True,
            source_mode=source_mode,
        )

    def rhs_with_progress(
        self,
        _time: float,
        state: np.ndarray,
        rho: float,
        momentum: np.ndarray,
        q5: float,
        *,
        source_mode: str = "coupled",
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._rhs(
            _time,
            state,
            rho,
            momentum,
            q5,
            enforce_nonnegative=True,
            source_mode=source_mode,
            return_progress=True,
        )

    def _rhs(
        self,
        _time: float,
        state: np.ndarray,
        rho: float,
        momentum: np.ndarray,
        q5: float,
        *,
        enforce_nonnegative: bool,
        source_mode: str,
        return_progress: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if source_mode not in {"coupled", "chemical", "vt"}:
            raise ValueError("unsupported source mode")
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("state must contain six finite values")
        if enforce_nonnegative:
            rho_species = self._validate_species(state[:5])
        else:
            rho_species = state[:5]
            if float(np.sum(rho_species)) <= np.finfo(float).tiny:
                raise ValueError("internal species trial has nonpositive total density")
        if state[5] < 0.0:
            raise ValueError("vibrational energy is invalid")
        temperature = self._validate_temperature(
            self._temperature_from_q5_raw(rho, momentum, rho_species, state[5], q5)
        )
        tv = self._tv_from_ev_raw(rho_species, state[5])
        pressure = self._pressure_raw(rho_species, temperature)
        if not self._pressure_in_domain(pressure):
            raise ValueError("pressure is outside the fixed mechanism domain")
        concentration = rho_species / self.molar_mass
        source = np.zeros(6)
        reaction_progress = np.zeros(len(self.reactions))

        if source_mode in {"coupled", "chemical"}:
            for reaction_index, reaction in enumerate(self.reactions):
                efficiencies = np.asarray(
                    reaction["third_body_efficiencies"], dtype=float
                )
                third_body = float(np.dot(efficiencies, concentration))
                if not np.any(efficiencies):
                    third_body = 1.0
                forward_product = self._mass_action(
                    concentration, reaction["reactants"]
                )
                reverse_product = self._mass_action(
                    concentration, reaction["products"]
                )
                rate_temperature = (
                    math.sqrt(temperature * tv)
                    if reaction["rate_temperature"] == "sqrt_T_Tv"
                    else temperature
                )
                arrhenius = reaction["arrhenius"]
                forward_coefficient = (
                    arrhenius["A"]
                    * rate_temperature ** arrhenius["temperature_exponent"]
                    * math.exp(
                        -arrhenius["activation_temperature_k"] / rate_temperature
                    )
                )
                equilibrium = reaction["equilibrium"]
                coefficient = equilibrium["ln_kc_coefficients"]
                z10000 = 10000.0 / temperature
                ln_kc = (
                    coefficient[0] / z10000
                    + coefficient[1]
                    + coefficient[2] * math.log(z10000)
                    + coefficient[3] * z10000
                    + coefficient[4] * z10000**2
                )
                reverse_coefficient = (
                    arrhenius["A"]
                    * temperature ** arrhenius["temperature_exponent"]
                    * math.exp(-arrhenius["activation_temperature_k"] / temperature)
                    / math.exp(ln_kc)
                )
                progress = third_body * (
                    forward_coefficient * forward_product
                    - reverse_coefficient * reverse_product
                )
                reaction_progress[reaction_index] = progress
                for species, name in enumerate(self.names):
                    net = reaction["products"].get(name, 0) - reaction[
                        "reactants"
                    ].get(name, 0)
                    if net == 0:
                        continue
                    mass_factor = self.molar_mass[species] * net
                    source[species] += mass_factor * progress
                    if species not in self.psi_bounds:
                        continue
                    if reaction["rate_temperature"] == "sqrt_T_Tv":
                        energy_per_mass = (
                            self._preferential_factor(species, temperature)
                            * self.gas_constant[species]
                            * arrhenius["activation_temperature_k"]
                        )
                    else:
                        energy_per_mass = self._species_vibrational_energy_raw(
                            species, tv
                        )
                    source[5] += mass_factor * progress * energy_per_mass

        if source_mode in {"coupled", "vt"}:
            source[5] += self._vt_source(rho_species, temperature, tv, pressure)
        if not np.all(np.isfinite(source)):
            raise ValueError("source evaluation produced a nonfinite value")
        if not np.all(np.isfinite(reaction_progress)):
            raise ValueError("reaction progress evaluation produced a nonfinite value")
        if return_progress:
            return source, reaction_progress
        return source
