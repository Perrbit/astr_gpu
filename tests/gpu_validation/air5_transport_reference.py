"""Independent JSON-driven transport oracle for the fixed air5 model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


RU = 8.3144626181532403


def derivative_sixth_order(values: np.ndarray, spacing: float) -> np.ndarray:
    """Return the periodic explicit sixth-order centered derivative."""
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[0] < 7:
        raise ValueError("sixth-order derivative requires at least seven points")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("derivative spacing must be positive and finite")
    return (
        0.75 * (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0))
        - 0.15 * (np.roll(values, -2, axis=0) - np.roll(values, 2, axis=0))
        + (np.roll(values, -3, axis=0) - np.roll(values, 3, axis=0)) / 60.0
    ) / spacing


@dataclass(frozen=True)
class Air5DiffusiveFlux:
    momentum_flux: np.ndarray
    species_flux: np.ndarray
    energy_flux: np.ndarray
    ev_flux: np.ndarray
    correction_velocity: np.ndarray


class Air5TransportReference:
    """Transport formulas implemented independently from the Fortran backend."""

    def __init__(self, mechanism_path: Path):
        data = json.loads(Path(mechanism_path).read_text(encoding="ascii"))
        if data.get("mechanism_id") != "air5_kimjo12":
            raise ValueError("unsupported mechanism identifier")
        if tuple(data["species_order"]) != ("N2", "O2", "N", "O", "NO"):
            raise ValueError("unexpected species order")
        species = data["species"]
        self.molar_mass = np.asarray(
            [item["molecular_weight_kg_per_mol"] for item in species], dtype=float
        )
        self.atom_count = np.asarray(
            [sum(item["atoms"].values()) for item in species], dtype=int
        )
        self.formation_energy = np.asarray(
            [item["formation_energy_j_per_kg"] for item in species], dtype=float
        )
        self.cv_factor = np.asarray(
            [item["cv_tr_over_species_gas_constant"] for item in species], dtype=float
        )
        self.theta_v = np.asarray(
            [item["vibrational_temperature_k"] for item in species], dtype=float
        )
        self.blottner = np.asarray(
            [item["blottner_coefficients"] for item in species], dtype=float
        )
        self.binary_coefficients = np.asarray(
            data["binary_diffusion_coefficients"], dtype=float
        )
        self.gas_constant = RU / self.molar_mass
        self.cv_tr = self.cv_factor * self.gas_constant

    def species_vibrational_energy(self, tv: float) -> np.ndarray:
        tv = float(tv)
        if not np.isfinite(tv) or tv <= 0.0:
            raise ValueError("vibrational temperature must be positive and finite")
        result = np.zeros(5)
        active = self.theta_v > 0.0
        result[active] = (
            self.gas_constant[active]
            * self.theta_v[active]
            / np.expm1(self.theta_v[active] / tv)
        )
        return result

    def species_vibrational_cv(self, tv: float) -> np.ndarray:
        tv = float(tv)
        if not np.isfinite(tv) or tv <= 0.0:
            raise ValueError("vibrational temperature must be positive and finite")
        result = np.zeros(5)
        active = self.theta_v > 0.0
        ratio = self.theta_v[active] / tv
        result[active] = (
            self.gas_constant[active]
            * ratio**2
            * np.exp(ratio)
            / np.expm1(ratio) ** 2
        )
        return result

    def transport_properties(
        self,
        temperature: float,
        tv: float,
        pressure: float,
        mass_fraction: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray]:
        mass_fraction = np.asarray(mass_fraction, dtype=float)
        if mass_fraction.shape != (5,) or np.any(mass_fraction < 0.0):
            raise ValueError("mass fractions must contain five nonnegative values")
        if not np.isclose(np.sum(mass_fraction), 1.0, atol=2.0e-14, rtol=0.0):
            raise ValueError("mass fractions must sum to one")
        if min(temperature, tv, pressure) <= 0.0:
            raise ValueError("transport state must be positive")

        mole_fraction = mass_fraction / self.molar_mass
        mole_fraction /= np.sum(mole_fraction)
        log_temperature = np.log(temperature)
        viscosity_species = 0.1 * np.exp(
            self.blottner[:, 0] * log_temperature**2
            + self.blottner[:, 1] * log_temperature
            + self.blottner[:, 2]
        )
        conductivity_species = np.where(
            self.atom_count == 1, 3.75, 4.75
        ) * viscosity_species * self.gas_constant
        conductivity_v_species = (
            viscosity_species * self.species_vibrational_cv(tv)
        )

        phi = np.empty((5, 5))
        for species in range(5):
            for partner in range(5):
                phi[species, partner] = (
                    1.0
                    + np.sqrt(
                        viscosity_species[species] / viscosity_species[partner]
                    )
                    * (self.molar_mass[partner] / self.molar_mass[species]) ** 0.25
                ) ** 2 / np.sqrt(
                    8.0
                    * (1.0 + self.molar_mass[species] / self.molar_mass[partner])
                )

        denominators = phi @ mole_fraction
        viscosity = float(np.sum(mole_fraction * viscosity_species / denominators))
        conductivity_tr = float(
            np.sum(mole_fraction * conductivity_species / denominators)
        )
        conductivity_v = float(
            np.sum(mole_fraction * conductivity_v_species / denominators)
        )

        coefficient = self.binary_coefficients
        binary_diffusion = 10.1325 / pressure * np.exp(coefficient[:, :, 3]) * (
            temperature
            ** (
                coefficient[:, :, 0] * log_temperature**2
                + coefficient[:, :, 1] * log_temperature
                + coefficient[:, :, 2]
            )
        )
        mixture_diffusion = np.zeros(5)
        for species in range(5):
            partners = np.arange(5) != species
            numerator = float(np.sum(mole_fraction[partners]))
            denominator = float(
                np.sum(
                    mole_fraction[partners]
                    / binary_diffusion[species, partners]
                )
            )
            if numerator > 0.0:
                if denominator <= 0.0:
                    raise ValueError("invalid mixture diffusion denominator")
                mixture_diffusion[species] = numerator / denominator
        return viscosity, conductivity_tr, conductivity_v, mixture_diffusion

    def diffusive_flux(
        self,
        *,
        rho: float,
        velocity: np.ndarray,
        temperature: float,
        tv: float,
        pressure: float,
        grad_velocity: np.ndarray,
        grad_temperature: np.ndarray,
        grad_tv: np.ndarray,
        mass_fraction: np.ndarray,
        grad_mass_fraction: np.ndarray,
    ) -> Air5DiffusiveFlux:
        velocity = np.asarray(velocity, dtype=float)
        grad_velocity = np.asarray(grad_velocity, dtype=float)
        grad_temperature = np.asarray(grad_temperature, dtype=float)
        grad_tv = np.asarray(grad_tv, dtype=float)
        mass_fraction = np.asarray(mass_fraction, dtype=float)
        grad_mass_fraction = np.asarray(grad_mass_fraction, dtype=float)
        if velocity.shape != (3,) or grad_velocity.shape != (3, 3):
            raise ValueError("velocity inputs have invalid shapes")
        if grad_temperature.shape != (3,) or grad_tv.shape != (3,):
            raise ValueError("temperature-gradient inputs have invalid shapes")
        if grad_mass_fraction.shape != (5, 3):
            raise ValueError("mass-fraction gradient has invalid shape")
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError("density must be positive and finite")

        viscosity, conductivity_tr, conductivity_v, mixture_diffusion = (
            self.transport_properties(temperature, tv, pressure, mass_fraction)
        )
        divergence = float(np.trace(grad_velocity))
        momentum_flux = viscosity * (grad_velocity + grad_velocity.T)
        momentum_flux -= np.eye(3) * (2.0 / 3.0) * viscosity * divergence

        raw_flux = -rho * mixture_diffusion[:, None] * grad_mass_fraction
        raw_sum = np.sum(raw_flux, axis=0)
        species_flux = raw_flux - mass_fraction[:, None] * raw_sum[None, :]
        correction_velocity = -raw_sum / rho
        vibrational_energy = self.species_vibrational_energy(tv)
        enthalpy = (
            self.cv_tr * temperature
            + vibrational_energy
            + self.formation_energy
            + self.gas_constant * temperature
        )
        energy_flux = (
            momentum_flux.T @ velocity
            + conductivity_tr * grad_temperature
            + conductivity_v * grad_tv
            - enthalpy @ species_flux
        )
        ev_flux = (
            conductivity_v * grad_tv - vibrational_energy @ species_flux
        )
        return Air5DiffusiveFlux(
            momentum_flux=momentum_flux,
            species_flux=species_flux,
            energy_flux=energy_flux,
            ev_flux=ev_flux,
            correction_velocity=correction_velocity,
        )
