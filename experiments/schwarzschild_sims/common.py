"""Shared helpers for the schwarzschild -> nbody pipeline.

Everything needed to rebuild the physical setup (units + potentials) is stored in the
HDF5 file under `/schwarzschild/{potential,inputs}`, so later stages never read ini files.
"""
import configparser
import subprocess
import agama
from astropy import units as u
from astropy import constants as cnst

SCHEMA_VERSION = 1
UNIT_NOTE = "Multiply by the numbers provided here to get back to the (kpc, km/s, Msun) unit-basis"
G_UNIT = (u.km / u.s) ** 2 * (u.kpc / u.Msun)


def load_potential_parameters(*ini_paths):
    parameters = {}
    for ini_path in ini_paths:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(ini_path)
        for section in config.sections():
            if not section.startswith("Potential "):
                continue
            name = section.removeprefix("Potential ")
            parameters[name] = {
                key: value if key == "type" else float(value)
                for key, value in config[section].items()
            }
    return parameters


def rescale_ini_dict(parameters, unitM, unitL, unitM_OG=u.Msun, unitL_OG=u.kpc):
    """Expects the output of load_potential_parameters (Msun, kpc)."""
    rescaled = {}
    for name, params in parameters.items():
        rescaled[name] = {}
        for key, value in params.items():
            if key.startswith("mass"):
                rescaled[name][key] = (value * unitM_OG).to(unitM).value
            elif key.startswith("scale"):
                rescaled[name][key] = (value * unitL_OG).to(unitL).value
            else:
                rescaled[name][key] = value
    return rescaled


def make_units(bar_mass_msun, bar_scale_kpc):
    """Unit system defined by the bar: M = bar mass, L = bar scale radius, G = 1."""
    G = cnst.G.to(G_UNIT)
    unitM = bar_mass_msun * u.Msun
    unitL = bar_scale_kpc * u.kpc
    unitV = (G * unitM / unitL) ** (1 / 2)
    unitT = unitL / unitV
    return dict(G=G, unitM=unitM, unitL=unitL, unitV=unitV, unitT=unitT, unitRot=unitV / unitL)


def set_agama_units(units):
    agama.setUnits(length=units["unitL"].value, mass=units["unitM"].value, velocity=units["unitV"].value)


def build_potentials(parameters_physical, units):
    """Returns (bar, potential_external) in code units from physical (Msun, kpc) parameters."""
    rescaled = rescale_ini_dict(parameters_physical, units["unitM"], units["unitL"])
    external = [agama.Potential(**p) for name, p in rescaled.items() if name != "bar"]
    potential_external = external[0]
    for pot in external[1:]:
        potential_external = potential_external + pot
    return agama.Potential(**rescaled["bar"]), potential_external, rescaled


# ---------------------------------------------------------------- HDF5 I/O

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_units(group, units):
    group.attrs["unitM"] = units["unitM"].to(u.Msun).value
    group.attrs["unitL"] = units["unitL"].to(u.kpc).value
    group.attrs["unitV"] = units["unitV"].to(u.km / u.s).value
    group.attrs["unitT"] = units["unitT"].to(u.kpc / (u.km / u.s)).value
    group.attrs["unitG"] = units["G"].value
    group.attrs["unitNote"] = UNIT_NOTE


def read_units(group):
    return make_units(group.attrs["unitM"], group.attrs["unitL"])


def write_potential(group, parameters_physical, omega_kms_kpc, ini_texts):
    """Store physical-unit potential parameters, the pattern speed and the raw ini text."""
    pot = group.create_group("potential")
    for name, params in parameters_physical.items():
        sub = pot.create_group(name)
        for key, value in params.items():
            sub.attrs[key] = value
    pot.attrs["omega_kms_kpc"] = omega_kms_kpc
    inputs = group.create_group("inputs")
    for key, text in ini_texts.items():
        inputs[key] = text


def read_potential(group):
    """Returns (units, parameters_physical, omega_kms_kpc)."""
    parameters = {}
    for name, sub in group["potential"].items():
        parameters[name] = {
            k: (v.decode() if isinstance(v, bytes) else v) for k, v in sub.attrs.items()
        }
    return read_units(group), parameters, float(group["potential"].attrs["omega_kms_kpc"])


def write_attrs(group, attrs):
    for key, value in attrs.items():
        group.attrs[key] = value
