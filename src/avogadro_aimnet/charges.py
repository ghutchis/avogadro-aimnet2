#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""AIMNet2-NSE partial charge calculator."""

import json
import sys

import numpy as np
from aimnet.calculators import AIMNet2Calculator


def run():
    bootstrap = json.load(sys.stdin)
    mol_cjson = bootstrap["cjson"]

    atoms = np.array(mol_cjson["atoms"]["elements"]["number"])
    props = mol_cjson.get("properties", {})
    charge = float(props.get("totalCharge", 0.0))
    mult = float(props.get("totalSpinMultiplicity", 1.0))

    data = {
        "coord": np.array(mol_cjson["atoms"]["coords"]["3d"], dtype=np.float32).reshape(1, -1, 3),  # (1, N, 3)
        "numbers": atoms[np.newaxis],                                                                # (1, N)
        "charge": np.array([charge], dtype=np.float32),                                             # (1,)
        "mult": np.array([mult], dtype=np.float32),                                                 # (1,)
    }

    calc = AIMNet2Calculator("aimnet2nse")
    results = calc(data)

    charges = results["charges"][0].tolist()  # (N,) for the single conformer
    print(json.dumps({"charges": charges}))
