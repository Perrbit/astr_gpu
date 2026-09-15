#!/usr/bin/env python3
"""Validate the fixed air5 mechanism and generate byte-stable Fortran data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


SPECIES_ORDER = ["N2", "O2", "N", "O", "NO"]
SI_UNITS = {
    "amount": "mol",
    "energy": "J",
    "length": "m",
    "mass": "kg",
    "temperature": "K",
    "time": "s",
}
ARRHENIUS_UNITS = "m^3 mol^-1 s^-1 K^-temperature_exponent"
LEGACY_ARCHIVE_SHA256 = "353edc87d665026a6650f3aa06862e1574df74248a2b2864d284fe0ea3692da1"


class MechanismError(ValueError):
    pass


def _exact_keys(mapping: Any, expected: set[str], context: str) -> None:
    if not isinstance(mapping, dict):
        raise MechanismError(f"{context}: expected an object")
    unknown = set(mapping)-expected
    if unknown:
        raise MechanismError(f"{context}: unknown field {sorted(unknown)}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MechanismError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise MechanismError(f"{context}: missing required field {key}")
    return mapping[key]


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanismError(f"{context}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MechanismError(f"{context}: value must be finite")
    return result


def _positive(value: Any, context: str) -> float:
    result = _finite(value, context)
    if result <= 0.0:
        raise MechanismError(f"{context}: value must be positive")
    return result


def _bounded_pair(value: Any, context: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise MechanismError(f"{context}: expected [minimum, maximum]")
    lower = _positive(value[0], f"{context}[0]")
    upper = _positive(value[1], f"{context}[1]")
    if lower >= upper:
        raise MechanismError(f"{context}: minimum must be less than maximum")
    return lower, upper


def _numeric_vector(value: Any, length: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise MechanismError(f"{context}: expected {length} values")
    return [_finite(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _stoichiometry(value: Any, species: set[str], context: str) -> list[int]:
    if not isinstance(value, dict) or not value:
        raise MechanismError(f"{context}: expected a nonempty object")
    unknown = set(value) - species
    if unknown:
        raise MechanismError(f"{context}: unknown species {sorted(unknown)}")
    result = []
    for name in SPECIES_ORDER:
        coefficient = value.get(name, 0)
        if isinstance(coefficient, bool) or not isinstance(coefficient, int) or coefficient < 0:
            raise MechanismError(f"{context}.{name}: expected a nonnegative integer")
        result.append(coefficient)
    if sum(result) == 0:
        raise MechanismError(f"{context}: stoichiometry cannot be empty")
    return result


def _species_matrix(value: Any, context: str, *, nonnegative: bool = False) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != len(SPECIES_ORDER):
        raise MechanismError(f"{context}: expected one row per species")
    matrix = []
    for index, row in enumerate(value):
        values = _numeric_vector(row, len(SPECIES_ORDER), f"{context}[{index}]")
        if nonnegative and any(item < 0.0 for item in values):
            raise MechanismError(f"{context}[{index}]: values must be nonnegative")
        matrix.append(values)
    return matrix


def _species_tensor(value: Any, depth: int, context: str) -> list[list[list[float]]]:
    if not isinstance(value, list) or len(value) != len(SPECIES_ORDER):
        raise MechanismError(f"{context}: expected one row per species")
    tensor = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != len(SPECIES_ORDER):
            raise MechanismError(f"{context}[{row_index}]: expected one entry per species")
        tensor.append([
            _numeric_vector(entry, depth, f"{context}[{row_index}][{column_index}]")
            for column_index, entry in enumerate(row)
        ])
    return tensor


def validate(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise MechanismError("mechanism root must be an object")
    _exact_keys(data, {
        "schema_version", "mechanism_id", "description", "species_order", "units",
        "validity", "provenance", "species", "preferential_dissociation",
        "vibrational_relaxation", "binary_diffusion_coefficients", "reactions",
    }, "mechanism")
    if _require(data, "schema_version", "mechanism") != 1:
        raise MechanismError("schema_version must be 1")
    if _require(data, "mechanism_id", "mechanism") != "air5_kimjo12":
        raise MechanismError("mechanism_id must be air5_kimjo12")
    description = _require(data, "description", "mechanism")
    if not isinstance(description, str) or not description.strip():
        raise MechanismError("description must be nonempty")
    if _require(data, "species_order", "mechanism") != SPECIES_ORDER:
        raise MechanismError(f"species_order must be {SPECIES_ORDER}")
    if _require(data, "units", "mechanism") != SI_UNITS:
        raise MechanismError("units must use the declared strict SI basis")

    validity = _require(data, "validity", "mechanism")
    if not isinstance(validity, dict):
        raise MechanismError("validity must be an object")
    _exact_keys(validity, {"temperature_k", "pressure_pa"}, "validity")
    temperature_bounds = _bounded_pair(
        _require(validity, "temperature_k", "validity"), "temperature_k"
    )
    pressure_bounds = _bounded_pair(
        _require(validity, "pressure_pa", "validity"), "pressure_pa"
    )
    if temperature_bounds != (300.0, 8000.0):
        raise MechanismError("temperature_k must equal the frozen [300, 8000] K domain")
    if pressure_bounds != (1.0e3, 1.0e6):
        raise MechanismError("pressure_pa must equal the frozen [1e3, 1e6] Pa domain")

    provenance = _require(data, "provenance", "mechanism")
    if not isinstance(provenance, dict):
        raise MechanismError("provenance must be an object")
    _exact_keys(provenance, {"kinetics", "thermodynamics", "unit_conversion",
                             "legacy_archive_sha256"}, "provenance")
    for key in ("kinetics", "thermodynamics", "unit_conversion"):
        value = _require(provenance, key, "provenance")
        if not isinstance(value, str) or not value.strip():
            raise MechanismError(f"provenance.{key} must be nonempty")
    archive_sha256 = _require(provenance, "legacy_archive_sha256", "provenance")
    if archive_sha256 != LEGACY_ARCHIVE_SHA256:
        raise MechanismError("provenance.legacy_archive_sha256 does not match the audited archive")

    species_data = _require(data, "species", "mechanism")
    if not isinstance(species_data, list) or len(species_data) != len(SPECIES_ORDER):
        raise MechanismError("species must contain exactly five entries")
    names = [_require(item, "name", f"species[{index}]")
             for index, item in enumerate(species_data)]
    if names != SPECIES_ORDER:
        raise MechanismError("species entries must follow species_order")
    if len(set(names)) != len(names):
        raise MechanismError("duplicate species name")

    atom_counts = []
    for index, item in enumerate(species_data):
        context = f"species[{index}]"
        if not isinstance(item, dict):
            raise MechanismError(f"{context}: expected an object")
        _exact_keys(item, {"name", "molecular_weight_kg_per_mol", "atoms",
                           "formation_energy_j_per_kg", "cv_tr_over_species_gas_constant",
                           "vibrational_temperature_k", "blottner_coefficients"}, context)
        _positive(_require(item, "molecular_weight_kg_per_mol", context),
                  f"{context}.molecular_weight_kg_per_mol")
        _finite(_require(item, "formation_energy_j_per_kg", context),
                f"{context}.formation_energy_j_per_kg")
        cv_factor = _positive(_require(item, "cv_tr_over_species_gas_constant", context),
                              f"{context}.cv_tr_over_species_gas_constant")
        if cv_factor not in (1.5, 2.5):
            raise MechanismError(f"{context}.cv_tr_over_species_gas_constant is not frozen")
        theta = _finite(_require(item, "vibrational_temperature_k", context),
                        f"{context}.vibrational_temperature_k")
        if theta < 0.0:
            raise MechanismError(f"{context}.vibrational_temperature_k must be nonnegative")
        atoms = _require(item, "atoms", context)
        if not isinstance(atoms, dict) or set(atoms) != {"N", "O"}:
            raise MechanismError(f"{context}.atoms must contain N and O")
        pair = []
        for element in ("N", "O"):
            count = atoms[element]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise MechanismError(f"{context}.atoms.{element} must be nonnegative integer")
            pair.append(count)
        if sum(pair) == 0:
            raise MechanismError(f"{context}.atoms cannot be empty")
        if (theta == 0.0) != (sum(pair) == 1):
            raise MechanismError(f"{context}: vibrational model and atom count disagree")
        atom_counts.append(pair)
        blottner = _numeric_vector(
            _require(item, "blottner_coefficients", context), 3,
            f"{context}.blottner_coefficients",
        )
        if not any(blottner):
            raise MechanismError(f"{context}.blottner_coefficients cannot all be zero")

    relaxation = _require(data, "vibrational_relaxation", "mechanism")
    if not isinstance(relaxation, dict):
        raise MechanismError("vibrational_relaxation must be an object")
    _exact_keys(relaxation, {"model", "mixture_rule", "pressure_basis",
                             "collision_partner_number_density", "collision_speed_mass",
                             "millikan_white_A", "millikan_white_B",
                             "collision_sigma0_m2", "collision_sigma_temperature_power"},
                "vibrational_relaxation")
    if _require(relaxation, "model", "vibrational_relaxation") != \
            "millikan_white_with_park_collision_limit":
        raise MechanismError("vibrational_relaxation.model is invalid")
    for key, nonnegative in (("millikan_white_A", False), ("millikan_white_B", False),
                             ("collision_sigma0_m2", True),
                             ("collision_sigma_temperature_power", True)):
        _species_matrix(_require(relaxation, key, "vibrational_relaxation"),
                        f"vibrational_relaxation.{key}", nonnegative=nonnegative)
    correction = _require(relaxation, "collision_partner_number_density", "vibrational_relaxation")
    if correction != "relaxing_species":
        raise MechanismError("collision_partner_number_density must be relaxing_species")
    if _require(relaxation, "mixture_rule", "vibrational_relaxation") != \
            "mole_fraction_weighted_inverse_pair_time":
        raise MechanismError("vibrational_relaxation.mixture_rule is invalid")
    if _require(relaxation, "pressure_basis", "vibrational_relaxation") != "Pa":
        raise MechanismError("vibrational_relaxation.pressure_basis must be Pa")
    if _require(relaxation, "collision_speed_mass", "vibrational_relaxation") != \
            "reduced_pair_molar_mass":
        raise MechanismError("vibrational_relaxation.collision_speed_mass is invalid")

    _species_tensor(_require(data, "binary_diffusion_coefficients", "mechanism"), 4,
                    "binary_diffusion_coefficients")

    psi = _require(data, "preferential_dissociation", "mechanism")
    if not isinstance(psi, dict):
        raise MechanismError("preferential_dissociation must be an object")
    _exact_keys(psi, {"N2", "O2", "NO"}, "preferential_dissociation")
    for molecular in ("N2", "O2", "NO"):
        entry = _require(psi, molecular, "preferential_dissociation")
        _exact_keys(entry, {"coefficients", "bounds"},
                    f"preferential_dissociation.{molecular}")
        _numeric_vector(_require(entry, "coefficients", molecular), 5,
                        f"preferential_dissociation.{molecular}.coefficients")
        bounds = _numeric_vector(_require(entry, "bounds", molecular), 2,
                                 f"preferential_dissociation.{molecular}.bounds")
        if not (0.0 <= bounds[0] <= bounds[1] <= 1.0):
            raise MechanismError(f"preferential_dissociation.{molecular}.bounds invalid")

    reactions = _require(data, "reactions", "mechanism")
    if not isinstance(reactions, list) or len(reactions) != 12:
        raise MechanismError("reactions must contain exactly 12 entries")
    species_set = set(SPECIES_ORDER)
    ids: set[str] = set()
    signatures: set[tuple[Any, ...]] = set()
    for index, reaction in enumerate(reactions):
        context = f"reactions[{index}]"
        if not isinstance(reaction, dict):
            raise MechanismError(f"{context}: expected an object")
        _exact_keys(reaction, {"id", "equation", "reactants", "products",
                               "third_body_efficiencies", "rate_temperature",
                               "arrhenius", "equilibrium"}, context)
        reaction_id = _require(reaction, "id", context)
        if not isinstance(reaction_id, str) or not reaction_id.strip():
            raise MechanismError(f"{context}: reaction id must be nonempty")
        if reaction_id in ids:
            raise MechanismError(f"duplicate reaction id {reaction_id}")
        ids.add(reaction_id)
        equation = _require(reaction, "equation", context)
        if not isinstance(equation, str) or not equation.strip():
            raise MechanismError(f"{context}.equation must be nonempty")
        reactants = _stoichiometry(_require(reaction, "reactants", context), species_set,
                                   f"{context}.reactants")
        products = _stoichiometry(_require(reaction, "products", context), species_set,
                                  f"{context}.products")
        net = [product - reactant for reactant, product in zip(reactants, products)]
        if all(value == 0 for value in net):
            raise MechanismError(f"{context}: reaction has no net stoichiometry")
        for element_index, element in enumerate(("N", "O")):
            imbalance = sum(net[i] * atom_counts[i][element_index]
                            for i in range(len(SPECIES_ORDER)))
            if imbalance:
                raise MechanismError(f"{context}: {element} atoms are not conserved")

        third_body = _numeric_vector(
            _require(reaction, "third_body_efficiencies", context), len(SPECIES_ORDER),
            f"{context}.third_body_efficiencies",
        )
        if any(value < 0.0 for value in third_body):
            raise MechanismError(f"{context}.third_body_efficiencies must be nonnegative")
        has_third_body = any(value > 0.0 for value in third_body)
        effective_order = sum(reactants) + int(has_third_body)
        if effective_order != 2:
            raise MechanismError(f"{context}: forward effective order must be two")

        temperature_model = _require(reaction, "rate_temperature", context)
        if temperature_model not in ("sqrt_T_Tv", "T"):
            raise MechanismError(f"{context}.rate_temperature is invalid")
        if has_third_body != (temperature_model == "sqrt_T_Tv"):
            raise MechanismError(f"{context}: third-body and temperature model disagree")

        arrhenius = _require(reaction, "arrhenius", context)
        if not isinstance(arrhenius, dict):
            raise MechanismError(f"{context}.arrhenius must be an object")
        _exact_keys(arrhenius, {"A", "temperature_exponent", "activation_temperature_k",
                                "A_units"}, f"{context}.arrhenius")
        _positive(_require(arrhenius, "A", f"{context}.arrhenius"), "Arrhenius A")
        _finite(_require(arrhenius, "temperature_exponent", f"{context}.arrhenius"),
                f"{context}.arrhenius.temperature_exponent")
        _positive(_require(arrhenius, "activation_temperature_k", f"{context}.arrhenius"),
                  f"{context}.arrhenius.activation_temperature_k")
        if _require(arrhenius, "A_units", f"{context}.arrhenius") != ARRHENIUS_UNITS:
            raise MechanismError(f"{context}: Arrhenius A units are not strict SI")

        equilibrium = _require(reaction, "equilibrium", context)
        _exact_keys(equilibrium, {"model", "concentration_basis", "delta_nu",
                                  "ln_kc_coefficients"}, f"{context}.equilibrium")
        if _require(equilibrium, "model", f"{context}.equilibrium") != "park_ln_kc_z10000":
            raise MechanismError(f"{context}.equilibrium.model is invalid")
        if _require(equilibrium, "concentration_basis", f"{context}.equilibrium") != "mol m^-3":
            raise MechanismError(f"{context}.equilibrium concentration is not strict SI")
        coefficients = _numeric_vector(
            _require(equilibrium, "ln_kc_coefficients", f"{context}.equilibrium"), 5,
            f"{context}.equilibrium.ln_kc_coefficients",
        )
        delta_nu = sum(products) - sum(reactants)
        if _require(equilibrium, "delta_nu", f"{context}.equilibrium") != delta_nu:
            raise MechanismError(f"{context}.equilibrium.delta_nu disagrees with stoichiometry")

        signature = (tuple(reactants), tuple(products), tuple(third_body), temperature_model)
        if signature in signatures:
            raise MechanismError(f"duplicate reaction at {context}")
        signatures.add(signature)

    if ids != {f"R{index}" for index in range(1, 13)}:
        raise MechanismError("reaction ids must be exactly R1 through R12")
    return data


def _real(value: float) -> str:
    if value == 0.0:
        return "0.00000000000000000d+00"
    return f"{value:.17e}".replace("e", "d")


def _wrap(items: list[str], indent: str = "    ", width: int = 104) -> list[str]:
    lines: list[str] = []
    current = indent
    for index, item in enumerate(items):
        suffix = "," if index + 1 < len(items) else ""
        token = item + suffix
        if len(current) + len(token) + 1 > width and current.strip():
            lines.append(current.rstrip() + " &")
            current = indent + token + " "
        else:
            current += token + " "
    lines.append(current.rstrip())
    return lines


def _array_declaration(name: str, shape: str, values: list[str], type_name: str,
                       reshape_shape: str | None = None) -> list[str]:
    result = [f"  {type_name}, parameter :: {name}{shape} = " +
              ("reshape([ " if reshape_shape else "[ ") + "&"]
    result.extend(_wrap(values, indent="    "))
    result[-1] += " &"
    closing = f"  ], {reshape_shape})" if reshape_shape else "  ]"
    result.append(closing)
    return result


def render(data: dict[str, Any]) -> bytes:
    validate(data)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    species = data["species"]
    reactions = data["reactions"]
    relaxation = data["vibrational_relaxation"]

    lines = [
        "! Generated by scripts/generate_air5_mechanism.py. Do not edit.",
        "module chemistry_air5_data",
        "  use iso_fortran_env, only: real64",
        "  implicit none",
        "  private",
        "",
        "  integer, parameter, public :: air5_real = real64",
        "  integer, parameter, public :: air5_num_species = 5",
        "  integer, parameter, public :: air5_num_reactions = 12",
        "  integer, parameter, public :: air5_idx_n2 = 1, air5_idx_o2 = 2",
        "  integer, parameter, public :: air5_idx_n = 3, air5_idx_o = 4, air5_idx_no = 5",
        "  real(real64), parameter, public :: air5_ru = 8.31446261815324030d+00",
        "  real(real64), parameter, public :: air5_temperature_min_k = 3.00000000000000000d+02",
        "  real(real64), parameter, public :: air5_temperature_max_k = 8.00000000000000000d+03",
        "  real(real64), parameter, public :: air5_pressure_min_pa = 1.00000000000000000d+03",
        "  real(real64), parameter, public :: air5_pressure_max_pa = 1.00000000000000000d+06",
        "  character(len=*), parameter, public :: air5_mechanism_id = 'air5_kimjo12'",
        f"  character(len=*), parameter, public :: air5_canonical_sha256 = '{digest}'",
        "",
    ]
    lines.extend(_array_declaration(
        "air5_species_name", "(air5_num_species)",
        [f"'{name:<2}'" for name in SPECIES_ORDER],
        "character(len=2), public",
    ))
    lines.extend(_array_declaration(
        "air5_molar_mass", "(air5_num_species)",
        [_real(item["molecular_weight_kg_per_mol"]) for item in species],
        "real(real64), public",
    ))
    lines.extend(_array_declaration(
        "air5_formation_energy", "(air5_num_species)",
        [_real(item["formation_energy_j_per_kg"]) for item in species],
        "real(real64), public",
    ))
    lines.extend(_array_declaration(
        "air5_cv_tr_factor", "(air5_num_species)",
        [_real(item["cv_tr_over_species_gas_constant"]) for item in species],
        "real(real64), public",
    ))
    lines.extend(_array_declaration(
        "air5_theta_v", "(air5_num_species)",
        [_real(item["vibrational_temperature_k"]) for item in species],
        "real(real64), public",
    ))
    atom_values = [str(species[i]["atoms"][element])
                   for i in range(5) for element in ("N", "O")]
    # atom_counts(element, species): species is the outer constructor dimension.
    lines.extend(_array_declaration(
        "air5_atom_count", "(2, air5_num_species)", atom_values,
        "integer, public", "[2, air5_num_species]",
    ))
    blottner = [item["blottner_coefficients"][coefficient]
                for item in species for coefficient in range(3)]
    lines.extend(_array_declaration(
        "air5_blottner", "(3, air5_num_species)", [_real(value) for value in blottner],
        "real(real64), public", "[3, air5_num_species]",
    ))

    reactant = [reaction["reactants"].get(name, 0)
                for reaction in reactions for name in SPECIES_ORDER]
    product = [reaction["products"].get(name, 0)
               for reaction in reactions for name in SPECIES_ORDER]
    third_body = [value for reaction in reactions
                  for value in reaction["third_body_efficiencies"]]
    lines.extend(_array_declaration(
        "air5_nu_reactant", "(air5_num_species, air5_num_reactions)",
        [str(value) for value in reactant], "integer, public",
        "[air5_num_species, air5_num_reactions]",
    ))
    lines.extend(_array_declaration(
        "air5_nu_product", "(air5_num_species, air5_num_reactions)",
        [str(value) for value in product], "integer, public",
        "[air5_num_species, air5_num_reactions]",
    ))
    lines.extend(_array_declaration(
        "air5_third_body_efficiency", "(air5_num_species, air5_num_reactions)",
        [_real(value) for value in third_body], "real(real64), public",
        "[air5_num_species, air5_num_reactions]",
    ))
    lines.extend(_array_declaration(
        "air5_has_third_body", "(air5_num_reactions)",
        [".true." if any(value > 0.0 for value in reaction["third_body_efficiencies"])
         else ".false." for reaction in reactions], "logical, public",
    ))
    lines.extend(_array_declaration(
        "air5_rate_uses_tv", "(air5_num_reactions)",
        [".true." if reaction["rate_temperature"] == "sqrt_T_Tv" else ".false."
         for reaction in reactions], "logical, public",
    ))
    for name, key in (("air5_arrhenius_a", "A"),
                      ("air5_arrhenius_b", "temperature_exponent"),
                      ("air5_activation_temperature", "activation_temperature_k")):
        lines.extend(_array_declaration(
            name, "(air5_num_reactions)",
            [_real(reaction["arrhenius"][key]) for reaction in reactions],
            "real(real64), public",
        ))
    lines.extend(_array_declaration(
        "air5_delta_nu", "(air5_num_reactions)",
        [str(reaction["equilibrium"]["delta_nu"]) for reaction in reactions],
        "integer, public",
    ))
    equilibrium = [coefficient for reaction in reactions
                   for coefficient in reaction["equilibrium"]["ln_kc_coefficients"]]
    lines.extend(_array_declaration(
        "air5_ln_kc_coefficient", "(5, air5_num_reactions)",
        [_real(value) for value in equilibrium], "real(real64), public",
        "[5, air5_num_reactions]",
    ))

    psi_coefficients = []
    psi_bounds = []
    for name in SPECIES_ORDER:
        entry = data["preferential_dissociation"].get(name)
        psi_coefficients.extend(entry["coefficients"] if entry else [0.0] * 5)
        psi_bounds.extend(entry["bounds"] if entry else [0.0, 0.0])
    lines.extend(_array_declaration(
        "air5_psi_coefficient", "(5, air5_num_species)",
        [_real(value) for value in psi_coefficients], "real(real64), public",
        "[5, air5_num_species]",
    ))
    lines.extend(_array_declaration(
        "air5_psi_bounds", "(2, air5_num_species)",
        [_real(value) for value in psi_bounds], "real(real64), public",
        "[2, air5_num_species]",
    ))

    def flatten_matrix(matrix: list[list[float]]) -> list[float]:
        return [matrix[row][column] for column in range(5) for row in range(5)]

    for output_name, key in (
        ("air5_mw_a", "millikan_white_A"),
        ("air5_mw_b", "millikan_white_B"),
        ("air5_collision_sigma0", "collision_sigma0_m2"),
        ("air5_collision_sigma_power", "collision_sigma_temperature_power"),
    ):
        lines.extend(_array_declaration(
            output_name, "(air5_num_species, air5_num_species)",
            [_real(value) for value in flatten_matrix(relaxation[key])],
            "real(real64), public", "[air5_num_species, air5_num_species]",
        ))
    lines.append(
        "  logical, parameter, public :: air5_relaxation_uses_relaxing_species_density = " +
        (".true." if relaxation["collision_partner_number_density"] ==
         "relaxing_species" else ".false.")
    )
    diffusion = data["binary_diffusion_coefficients"]
    diffusion_values = [diffusion[row][column][coefficient]
                        for column in range(5) for row in range(5)
                        for coefficient in range(4)]
    lines.extend(_array_declaration(
        "air5_binary_diffusion", "(4, air5_num_species, air5_num_species)",
        [_real(value) for value in diffusion_values], "real(real64), public",
        "[4, air5_num_species, air5_num_species]",
    ))
    lines.extend(["", "end module chemistry_air5_data", ""])
    return "\n".join(lines).encode("ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        data = json.loads(
            arguments.input.read_text(encoding="ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                MechanismError(f"non-finite JSON value {value}")),
        )
        content = render(data)
        if arguments.check:
            if not arguments.output.exists() or arguments.output.read_bytes() != content:
                raise MechanismError(f"generated file is stale: {arguments.output}")
        else:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_bytes(content)
    except (OSError, json.JSONDecodeError, MechanismError) as error:
        print(f"air5 mechanism error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
