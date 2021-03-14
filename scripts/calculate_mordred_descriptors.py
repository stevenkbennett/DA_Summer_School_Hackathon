"""Use Mordred to calculate both 2D and 3D descriptors for molecules from the CSD."""
import collections
from mordred import descriptors, Calculator
from mordred import error
from multiprocessing import Process, Manager
from mordred._base.pandas_module import MordredDataFrame
from rdkit.Chem.MolStandardize import standardize_smiles
import os
from rdkit import Chem
import multiprocessing as mp
import itertools as it
from time import sleep
from warnings import warn
import pandas as pd
from collections import defaultdict


def mol2_to_smiles(file=None, sanitize=True):
    smiles = []
    with open(file, "r") as f:
        line = f.readline()
        while not f.tell() == os.fstat(f.fileno()).st_size:
            if line.startswith("@<TRIPOS>MOLECULE"):
                mol = []
                mol.append(line)
                line = f.readline()
                while not line.startswith("@<TRIPOS>MOLECULE"):
                    mol.append(line)
                    line = f.readline()
                    if f.tell() == os.fstat(f.fileno()).st_size:
                        mol.append(line)
                        break
                mol[-1] = mol[-1].rstrip()  # removes blank line at file end
                block = ",".join(mol).replace(",", "")
                m = Chem.MolFromMol2Block(
                    block, sanitize=sanitize, removeHs=False
                )
                try:
                    smiles.append(Chem.MolToSmiles(m))
                except:
                    smiles.append(None)
    return smiles


def quick_conf_search(smiles, cache):
    try:
        print(f"Optimising {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        Chem.Kekulize(mol)
        inchikey = Chem.MolToInchiKey(mol)
        if inchikey in cache:
            return cache[inchikey]
        # Perform conformer search using ETKDGv3
        params = Chem.rdDistGeom.ETKDGv3()
        random_seed = 0
        params.random_seed = random_seed
        num_confs = 500
        embed_res = Chem.rdDistGeom.EmbedMultipleConfs(mol, num_confs, params)
        if embed_res == -1:
            warn(
                f"Embedding failed with random seed {random_seed}. "
                "Returning None."
            )
            return None
        # Optimise all conformers of the molecule.
        res = Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            mol, numThreads=0
        )
        while True:
            if len(res) == 0:
                warn(
                    "All force field optimisations could not converge. "
                    "Returning None."
                )
                return None
            e_min_ind = res.index(min(res, key=lambda x: x[1]))
            converged = res[e_min_ind][0]
            if converged == 0:
                for i in range(num_confs):
                    if i == e_min_ind:
                        continue
                    mol.RemoveConformer(i)
                # Update ID of conformer
                mol.GetConformer(-1).SetId(0)
                # Return the optimised molecule
                cache[inchikey] = mol
                return mol
            res.pop(e_min_ind)
    except:
        return None


def calculate_descriptors(mol, cache):
    try:
        inchikey = Chem.MolToInchiKey(mol)
        if inchikey in cache:
            return cache[inchikey]
        if list(mol.GetConformers()) == 0:
            raise RuntimeError(f"Molecule has no 3D coordinates.")
        # Setup descriptor calculation
        calc = Calculator(descriptors, ignore_3D=False)
        try:
            res = calc(mol).asdict()
            cache[inchikey] = res
            return res
        except Exception as err:
            print(err)
            return None
    except Exception as err:
        print(err)
        return None


def parallel_conf_search(smiles, processes=-1):
    if processes == -1:
        processes = mp.cpu_count()
    m = Manager()
    d = m.dict()
    # Create processes
    print(f"Running conformer searches with {processes} processes.")
    pool = mp.Pool(processes)
    f = pool.starmap_async(
        quick_conf_search,
        it.product(smiles, [d]),
        error_callback=lambda x: print(
            "An error occured during conformer calculation."
        ),
    )
    return list(f.get())


def parallel_descriptor_calc(mols, processes=-1):
    if processes == -1:
        processes = mp.cpu_count()
    m = Manager()
    d = m.dict()
    # Create processes
    print(f"Running descriptor calculations with {processes} processes.")
    pool = mp.Pool(processes)
    f = pool.starmap_async(
        calculate_descriptors,
        it.product(mols, [d]),
        error_callback=lambda x: print(
            "An error occured during conformer calculation."
        ),
    )
    return list(f.get())


def parse_results(smiles, res):
    d = defaultdict(list)
    for i, r in enumerate(res):
        if r is None:
            [d[k].append(None) for k in d.keys()]
            continue
        d["SMILES"].append(smiles[i])
        for name, val in r.items():
            if type(val) == error.Missing:
                val = None
            d[name].append(val)
    return pd.DataFrame(d)


def main():
    print("Calculating Mordred descriptors.")
    # Molecules where an error occured at any part of the calculation
    # are stored as None.
    # # This is so the index is maintained throughout the descriptor calculation.
    # smiles = mol2_to_smiles(
    #     "/Users/stevenbennett/PhD/FONS_Datathon/small_molecule_search.mol2"
    # )
    # Smiles for testing
    smiles = ["C" * i for i in range(1, 200)][:8]
    smiles += [None]
    smiles += ["C" * i for i in range(1, 200)][:8]
    print("Performing conformer search")
    rdmols = parallel_conf_search(smiles)
    res = parallel_descriptor_calc(rdmols)
    df = parse_results(smiles, res)
    df = df.dropna(how="all")
    df["Original_Index"] = df.index
    df = df.reset_index(drop=True)
    print(df)
    df.to_csv("Mordred_Descriptors.csv", index=False, header=True)


if __name__ == "__main__":
    main()