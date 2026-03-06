#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""AIMNet2 energy and gradient calculator using the binary protocol."""

import json
import sys

import numpy as np
from aimnet.calculators import AIMNet2Calculator

from .energy import EnergyServer

# Conversion factors
_HARTREE_TO_KJ_MOL = 2625.4996394799


def run(model_name: str = "aimnet2"):
    # Avogadro sends one compact JSON line on stdin, then switches to binary.
    # Read from sys.stdin.buffer (not sys.stdin) so the text wrapper's
    # internal read-ahead doesn't consume bytes needed by the binary protocol.
    bootstrap = json.loads(sys.stdin.buffer.readline())
    mol_cjson = bootstrap["cjson"]

    atoms = np.array(mol_cjson["atoms"]["elements"]["number"])
    props = mol_cjson.get("properties", {})
    charge = float(props.get("totalCharge", 0.0))
    mult = float(props.get("totalSpinMultiplicity", 1.0))
    num_atoms = len(atoms)

    calc = AIMNet2Calculator(model_name)
    use_mult = "nse" in model_name

    with EnergyServer(sys.stdin.buffer, sys.stdout.buffer, num_atoms) as server:
        for request in server.requests():
            coords = request.coords  # (N, 3) or (batch, N, 3) in Angstrom

            if request.is_batch:
                batch_size = request.batch_size
                # coords shape: (batch, N, 3)
                data = {
                    "coord": coords.astype(np.float32),
                    "numbers": np.tile(atoms, (batch_size, 1)),
                    "charge": np.full(batch_size, charge, dtype=np.float32),
                }
                if use_mult:
                    data["mult"] = np.full(batch_size, mult, dtype=np.float32)
                results = calc(data, forces=request.wants_gradient)
                energies_np = results["energy"] * _HARTREE_TO_KJ_MOL
                if request.wants_gradient:
                    # forces = -gradient; convert and negate
                    grads_np = -results["forces"] * _HARTREE_TO_KJ_MOL
                    request.send_gradients(grads_np)
                else:
                    request.send_energies(energies_np)
            else:
                data = {
                    "coord": np.array([coords], dtype=np.float32),   # (1, N, 3)
                    "numbers": atoms[np.newaxis],                     # (1, N)
                    "charge": np.array([charge], dtype=np.float32),  # (1,)
                }
                if use_mult:
                    data["mult"] = np.array([mult], dtype=np.float32)
                results = calc(data, forces=request.wants_gradient)
                energy = float(results["energy"][0]) * _HARTREE_TO_KJ_MOL
                if request.wants_gradient:
                    # forces = -gradient; convert and negate
                    grad_np = -results["forces"][0] * _HARTREE_TO_KJ_MOL
                    request.send_gradient(grad_np)
                else:
                    request.send_energy(energy)
