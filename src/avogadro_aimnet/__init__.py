#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""Entry point for the avogadro-aimnet-energy plugin.

Avogadro calls this as:
    avogadro-aimnet-energy <identifier> [--lang <locale>] [--debug]

with the molecule bootstrap JSON on stdin (one compact JSON line).
"""

import argparse


def setup():
    """Download AIMNet2 model parameters by running a test calculation."""
    import numpy as np
    import torch
    from aimnet.calculators import AIMNet2Calculator

    device = torch.device('cpu')
    # Water molecule: O, H, H
    species = torch.tensor([[8, 1, 1]], device=device)
    coordinates = torch.tensor([[
        [0.000,  0.000,  0.119],
        [0.000,  0.757, -0.477],
        [0.000, -0.757, -0.477],
    ]], dtype=torch.float32, device=device)

    data = {
        "coord": coordinates,  # Nx3 array
        "numbers": species,  # N array
        "charge": 0.0,
    }

    print("Downloading AIMNet2 model parameters (this may take a moment)...")
    calc = AIMNet2Calculator("aimnet2")
    calc(data)

    print("AIMNet2 setup complete.")

    print("Downloading AIMNet2-NSE model parameters (this may take a moment)...")
    calc = AIMNet2Calculator("aimnet2nse")
    data["mult"] = 1.0
    calc(data)

    print("AIMNet2-NSE setup complete.")

def main():
    parser = argparse.ArgumentParser("avogadro-aimnet-energy")
    parser.add_argument("feature")
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--protocol", nargs="?", default="binary-v1")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--charges", action="store_true")
    parser.add_argument("--potential", action="store_true")
    args = parser.parse_args()

    match args.feature:
        case "AIMNet2":
            from .aimnet2 import run
            run("aimnet2")
        case "AIMNet2-NSE":
            from .aimnet2 import run
            run("aimnet2nse")
        case "nse_charges":
            from .charges import run
            run()
