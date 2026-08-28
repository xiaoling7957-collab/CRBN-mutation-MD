# CRBN-mutation-MD

# CK1α–Lenalidomide–CRBN Ternary Complex MD Analysis

Molecular dynamics analysis of the CK1α–lenalidomide–CRBN ternary complex, comparing wild type CRBN against the V388I mutant across three independent simulation replicates. The V388I substitution is a species-selectivity residue implicated in whether lenalidomide-dependent CK1α recruitment and degradation occurs.

## Scientific background

Lenalidomide acts as a molecular glue: it binds CRBN's thalidomide-binding domain and creates a composite surface that recruits CK1α for ubiquitination. Val388 on CRBN lies adjacent to CK1α's Gly40-containing β-hairpin loop. Replacing Val with the bulkier Ile is predicted to sterically crowd that loop and weaken ternary complex formation.

## Repository structure

- `analysis_functions.py` — all reusable analysis and plotting functions
- `compare_WT_vs_V388I.ipynb` — main notebook (edit Cell 2 only per run)
- `analysis_*/` — per-run output folders (PDB, stripped DCD, figures)

## Analysis pipeline

Each replicate's wild type is compared to its own mutant independently. Computed for all 3 runs:

- **RMSD** — backbone fold stability (self-fit) and drift from frame 0
- **RMSF** — per-residue flexibility overlaid WT vs mutant
- **Interface H-bonds** — CK1α Ile-Thr-Asn triplet → CRBN, length vs time
- **Binding-site RMSD** — CK1α, CRBN and combined interface (10 Å cutoff, defined on WT frame 0)
- **Ligand pocket RMSD** — CRBN residues within 10 Å of lenalidomide carbons
- **Literature contacts** — minimum heavy-atom distances for Pro352↔ligand, Gly40↔ligand, and the key Val388/Ile388↔Gly40 steric-clash test

## Dependencies

Python >= 3.10, with MDAnalysis >= 2.0, numpy, pandas, matplotlib.

Install with:

    pip install MDAnalysis numpy pandas matplotlib

## Usage

1. Open `compare_WT_vs
