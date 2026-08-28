"""
analysis_functions.py
Reusable routines for CK1a-lenalidomide-CRBN MD analysis (WT vs mutant).
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.lib.distances import calc_bonds, distance_array

try:
    from MDAnalysis.lib.distances import minimize_vectors
    _HAVE_MINVEC = True
except ImportError:
    from MDAnalysis.lib.mdamath import triclinic_vectors
    _HAVE_MINVEC = False

SOLVENT_IONS = {"WAT", "HOH", "TIP3", "SOL", "SPC", "NA", "NA+", "CL", "CL-",
                "K", "K+", "MG", "CA", "ZN", "ZN2"}
WATER = {"WAT", "HOH", "TIP3", "TIP3P", "TIP4P", "TIP5P", "SOL",
         "SPC", "SPCE", "T3P", "T4P", "OPC", "OPC3", "H2O"}


# ======================================================================
# 1. SETUP
# ======================================================================

def apply_style():
    plt.rcParams["figure.figsize"] = (9, 4)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.grid"] = True


def load_universe(top, traj, dt_ps=None):
    """Return (universe, dt_ps, n_frames)."""
    u = mda.Universe(top, traj)
    dt = float(u.trajectory.dt) if dt_ps is None else float(dt_ps)
    return u, dt, u.trajectory.n_frames


def dt_from_total_time(n_frames, total_time_ns):
    """ps/frame from known total time. Pass RAW frame count (before striding)."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    return total_time_ns * 1000.0 / n_frames


def assign_chains(universe, ligand_resname=None, chain_a_frag_index=0):
    """Chain A = largest protein fragment (CK1a); chain B = the rest (CRBN).
    Returns selection strings {"A","B","LIG","ZN","COMPLEX"}."""
    frags = sorted(universe.select_atoms("protein").fragments,
                   key=lambda f: f.n_residues, reverse=True)
    if ligand_resname is None:
        prot_names = set(universe.select_atoms("protein").residues.resnames)
        others = [r for r in sorted(set(universe.residues.resnames))
                  if r not in prot_names]
        cands = [r for r in others if r not in SOLVENT_IONS
                 and universe.select_atoms("resname " + r).n_atoms
                     / max(universe.select_atoms("resname " + r).n_residues, 1) > 8]
        if not cands:
            raise RuntimeError("No ligand candidate found; set ligand_resname.")
        ligand_resname = max(cands, key=lambda r:
            universe.select_atoms("resname " + r).n_atoms
            / max(universe.select_atoms("resname " + r).n_residues, 1))
    a_ag = frags[chain_a_frag_index].select_atoms("protein")
    lo, hi = int(a_ag.residues.resids.min()), int(a_ag.residues.resids.max())
    return {"A": f"protein and resid {lo}:{hi}",
            "B": f"protein and not resid {lo}:{hi}",
            "LIG": "resname " + ligand_resname,
            "ZN": "resname ZN ZN1 ZN2",
            "COMPLEX": f"protein or (resname {ligand_resname})"}


def _min_image_shift(vec, box):
    if _HAVE_MINVEC:
        v = np.asarray(vec, dtype=np.float32).reshape(1, 3)
        return v[0] - minimize_vectors(v, box=box)[0]
    H = triclinic_vectors(box)
    v = np.asarray(vec, float)
    best_n, shift = v @ v, np.zeros(3)
    for i in (-2, -1, 0, 1, 2):
        for j in (-2, -1, 0, 1, 2):
            for k in (-2, -1, 0, 1, 2):
                s = i * H[0] + j * H[1] + k * H[2]
                t = v - s; n = t @ t
                if n < best_n:
                    best_n, shift = n, s
    return shift


def build_clean_universe(top, traj, sels, stride, dt_ps, outdir, system_name="system"):
    """Make molecules whole, reassemble around the largest fragment (any box
    shape), remove ONLY water, write a water-free copy. Returns
    (u, times_ns, pdb_path, dcd_path)."""
    os.makedirs(outdir, exist_ok=True)
    u_full = mda.Universe(top, traj)
    keep = u_full.select_atoms("not resname " + " ".join(sorted(WATER)))
    frag_list = list(keep.fragments)
    anchor = max(frag_list, key=lambda f: f.n_atoms)

    def reassemble(ts):
        ref = anchor.center_of_mass()
        for frag in frag_list:
            shift = _min_image_shift(frag.center_of_mass() - ref, ts.dimensions)
            if np.abs(shift).sum() > 1e-6:
                frag.atoms.positions = frag.atoms.positions - shift
        return ts

    u_full.trajectory.add_transformations(trans.unwrap(keep), reassemble)
    pdb_out = os.path.join(outdir, system_name + "_complex.pdb")
    dcd_out = os.path.join(outdir, system_name + "_complex_nowat.dcd")
    u_full.trajectory[0]
    keep.write(pdb_out)
    with mda.Writer(dcd_out, keep.n_atoms) as W:
        for ts in u_full.trajectory[::stride]:
            W.write(keep)
    u = mda.Merge(keep)
    u.load_new(dcd_out)
    times_ns = np.arange(u.trajectory.n_frames) * dt_ps * stride / 1000.0
    return u, times_ns, pdb_out, dcd_out


# ======================================================================
# 2. ANALYSIS
# ======================================================================

def rmsd_self_fit(u, sels):
    """Backbone RMSD vs frame 0, each group fit on ITSELF (fold stability)."""
    rc = rms.RMSD(u, u, select="protein and backbone", ref_frame=0).run().results.rmsd[:, 2]
    ra = rms.RMSD(u, u, select=sels["A"] + " and backbone", ref_frame=0).run().results.rmsd[:, 2]
    rb = rms.RMSD(u, u, select=sels["B"] + " and backbone", ref_frame=0).run().results.rmsd[:, 2]
    return pd.DataFrame({"complex": rc, "chain_A": ra, "chain_B": rb})


def rmsd_vs_frame0(u, sels):
    """Backbone RMSD vs frame 0, one common whole-protein alignment (drift)."""
    Rg = rms.RMSD(u, u, select="protein and backbone",
                  groupselections=[sels["A"] + " and backbone",
                                   sels["B"] + " and backbone",
                                   sels["LIG"] + " and not name H*"],
                  ref_frame=0).run()
    return pd.DataFrame({"complex": Rg.results.rmsd[:, 2],
                         "chain_A": Rg.results.rmsd[:, 3],
                         "chain_B": Rg.results.rmsd[:, 4],
                         "ligand":  Rg.results.rmsd[:, 5]})


def _chain_rmsf(u, sel):
    """Per-residue Cα RMSF for one chain (aligns u onto that chain's average)."""
    avg = align.AverageStructure(u, u, select=sel + " and name CA", ref_frame=0).run()
    align.AlignTraj(u, avg.results.universe, select=sel + " and name CA", in_memory=True).run()
    ca = u.select_atoms(sel + " and name CA")
    vals = rms.RMSF(ca).run().results.rmsf
    return pd.DataFrame({"resid": ca.residues.resids,
                         "resname": ca.residues.resnames, "rmsf": vals})


def rmsf_both_chains(u, sels):
    """RMSF for both chains, tagged. (Aligns u in place - harmless downstream.)"""
    a = _chain_rmsf(u, sels["A"]); a["chain"] = "chain_A"
    b = _chain_rmsf(u, sels["B"]); b["chain"] = "chain_B"
    return pd.concat([a, b], ignore_index=True)


def find_motif_triplet(u, sel_chain, sel_other, motif=("ILE", "THR", "ASN")):
    """Consecutive residues matching `motif`; nearest to sel_other if several."""
    res = u.select_atoms(sel_chain).residues
    nm, ids = list(res.resnames), list(res.resids)
    L = len(motif)
    hits = [tuple(ids[i:i + L]) for i in range(len(nm) - L + 1)
            if tuple(nm[i:i + L]) == tuple(motif)]
    if not hits:
        raise RuntimeError(f"No consecutive {motif} run found.")
    other = u.select_atoms(sel_other)
    def mind(t):
        ag = u.select_atoms(sel_chain + " and resid " + " ".join(map(str, t)))
        return float(distance_array(ag.positions, other.positions).min())
    return list(min(hits, key=mind))


def hbonds_for_residues(u, sel_chain, sel_other, resids):
    """H-bonds between given residues of sel_chain and sel_other, best per
    residue. Returns (summary_df, series[n_frames,n_bonds], labels)."""
    nfr = u.trajectory.n_frames
    sel_t = "(" + sel_chain + ") and resid " + " ".join(map(str, resids))
    hb = HBA(universe=u, between=[sel_t, sel_other], d_a_cutoff=3.5, d_h_a_angle_cutoff=150)
    hb.hydrogens_sel = hb.guess_hydrogens("protein")
    hb.acceptors_sel = hb.guess_acceptors("protein")
    if not hb.acceptors_sel:
        hb.acceptors_sel = "protein and (name O* or name N* or name S*)"
    hb.run()
    chain_ids = set(u.select_atoms(sel_chain).residues.resids)
    pair_frames = defaultdict(set)
    for row in hb.results.hbonds:
        pair_frames[(int(row[1]), int(row[3]))].add(int(row[0]))
    best = {}
    for (d, a), fr in pair_frames.items():
        da, aa = u.atoms[d], u.atoms[a]
        key = da.resid if da.resid in chain_ids else aa.resid
        occ = 100 * len(fr) / nfr
        if key not in best or occ > best[key]["occ"]:
            best[key] = dict(d=d, a=a, occ=occ,
                             text=f"{da.resname}{da.resid}:{da.name} -> "
                                  f"{aa.resname}{aa.resid}:{aa.name}")
    chosen = [best[r] for r in resids if r in best]
    if not chosen:
        raise RuntimeError("No H-bonds found for the requested residues.")
    d_at = u.atoms[[c["d"] for c in chosen]]
    a_at = u.atoms[[c["a"] for c in chosen]]
    series = np.array([calc_bonds(d_at.positions, a_at.positions) for _ in u.trajectory])
    summary = pd.DataFrame({"bond": [c["text"] for c in chosen],
                            "occupancy": [c["occ"] for c in chosen],
                            "mean_len": series.mean(axis=0),
                            "sd_len": series.std(axis=0),
                            "min_len": series.min(axis=0),
                            "max_len": series.max(axis=0)})
    labels = [f"{c['text']} ({c['occ']:.0f}%)" for c in chosen]
    return summary, series, labels

def define_binding_site(u, sels, cutoff=10.0, frame=0):
    """Cα of each chain within `cutoff` of the OTHER chain's nearest Cα.
    Returns {"CK1a":[resids],"CRBN":[resids]}."""
    u.trajectory[frame]
    a_ca = u.select_atoms(sels["A"] + " and name CA")
    b_ca = u.select_atoms(sels["B"] + " and name CA")
    d = distance_array(a_ca.positions, b_ca.positions)
    ck1a = np.array(a_ca.resids)[d.min(axis=1) <= cutoff]
    crbn = np.array(b_ca.resids)[d.min(axis=0) <= cutoff]
    return {"CK1a": sorted(int(x) for x in ck1a),
            "CRBN": sorted(int(x) for x in crbn)}


def define_ligand_pocket(u, sels, cutoff=10.0, frame=0):
    """CRBN Cα within `cutoff` of the nearest ligand carbon at `frame`.
    Returns (pocket_resids, contact_table)."""
    u.trajectory[frame]
    crbn_ca = u.select_atoms(sels["B"] + " and name CA")
    lig_c = u.select_atoms(sels["LIG"] + " and name C*")
    if lig_c.n_atoms == 0:
        raise ValueError("No ligand carbons (name C*) - check sels['LIG'].")
    d = distance_array(crbn_ca.positions, lig_c.positions)
    nearest_d, nearest_col = d.min(axis=1), d.argmin(axis=1)
    mask = nearest_d <= cutoff
    rows = []
    for i in np.where(mask)[0]:
        r, la = crbn_ca[i], lig_c[int(nearest_col[i])]
        rows.append({"CRBN_resid": int(r.resid), "CRBN_res": r.resname,
                     "nearest_lig_C": la.name, "dist": round(float(nearest_d[i]), 2)})
    table = pd.DataFrame(rows).sort_values("dist").reset_index(drop=True)
    return sorted(int(x) for x in np.array(crbn_ca.resids)[mask]), table


def rmsd_of_resids(u, resids):
    """Cα RMSD vs frame 0, fit on those same Cα (region's own shape change)."""
    if not resids:
        raise ValueError("No residues supplied.")
    sel = "name CA and resid " + " ".join(str(r) for r in resids)
    n = u.select_atoms(sel).n_atoms
    if n != len(resids):
        raise ValueError(f"Expected {len(resids)} Cα, found {n} - resid missing.")
    return rms.RMSD(u, u, select=sel, ref_frame=0).run().results.rmsd[:, 2]


def resid_from_motif(u, chain_sel, motif, target_index):
    """File resid of position `target_index` in one-letter `motif` within chain_sel."""
    t2o = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
           'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
           'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
           'HID':'H','HIE':'H','HIP':'H','ASH':'D','GLH':'E','LYN':'K',
           'CYX':'C','CYM':'C','CY1':'C','CY2':'C'}
    res = u.select_atoms(chain_sel).residues
    order = np.argsort(res.resids)
    resids, resnames = np.array(res.resids)[order], np.array(res.resnames)[order]
    seq = "".join(t2o.get(rn, "X") for rn in resnames)
    i = seq.find(motif)
    if i == -1:
        raise RuntimeError(f"motif {motif} not found in chain")
    return int(resids[i + target_index])


def pair_min_distance(u, sel1, sel2):
    """Minimum heavy-atom distance between two selections, per frame."""
    g1 = u.select_atoms(sel1 + " and not name H*")
    g2 = u.select_atoms(sel2 + " and not name H*")
    if g1.n_atoms == 0 or g2.n_atoms == 0:
        raise ValueError("empty selection - check resid/ligand name")
    return np.array([distance_array(g1.positions, g2.positions).min()
                     for _ in u.trajectory])


def literature_contact_distances(u, sels):
    """Locate Val388, Pro352, CK1a Gly40 and the 4 nearest CRBN pocket residues.
    Returns {label: (sel1, sel2)} defined on this universe (use on WT, apply to both)."""
    u.trajectory[0]
    ck1a_lo = int(u.select_atoms(sels["A"]).residues.resids.min())
    crbn_seq = f"(protein or resname CY1) and resid 1:{ck1a_lo-1}"
    val388 = resid_from_motif(u, crbn_seq, "FPGYAWTVAQ", "FPGYAWTVAQ".index("V"))
    pro352 = 352 + (val388 - 388)
    lig = u.select_atoms(sels["LIG"] + " and not name H*")
    vsite = u.select_atoms(f"resid {val388} and not name H*")
    gly40, best = None, 1e9
    for r in u.select_atoms(sels["A"] + " and resname GLY").residues:
        g = u.select_atoms(f"resid {r.resid} and not name H*")
        s = (distance_array(g.positions, lig.positions).min()
             + distance_array(g.positions, vsite.positions).min())
        if s < best:
            gly40, best = int(r.resid), s
    pairs = {"Pro352 ↔ lig": (f"resid {pro352}", sels["LIG"]),
             "Gly40 ↔ lig":  (f"resid {gly40}",  sels["LIG"]),
             "388 ↔ lig":    (f"resid {val388}", sels["LIG"]),
             "388 ↔ Gly40 (clash)": (f"resid {val388}", f"resid {gly40}")}
    crbn_ca = u.select_atoms(sels["B"] + " and name CA")
    dmin = distance_array(crbn_ca.positions, lig.positions).min(axis=1)
    for rid in [int(x) for x in crbn_ca.resids[np.argsort(dmin)][:4]]:
        rn = u.select_atoms(f"resid {rid}").residues.resnames[0]
        pairs[f"{rn}{rid} ↔ lig (pocket)"] = (f"resid {rid}", sels["LIG"])
    return pairs


# ======================================================================
# 3. PIPELINE
# ======================================================================

def analyse_system(top, traj, stride, total_time_ns, outdir, system_name,
                   ligand_resname=None, chain_a_frag_index=0):
    """Load, fix PBC / strip water, compute both RMSDs and RMSF. Returns dict."""
    u_full, _, n_frames = load_universe(top, traj)
    sels = assign_chains(u_full, ligand_resname=ligand_resname,
                         chain_a_frag_index=chain_a_frag_index)
    dt_ps = dt_from_total_time(n_frames, total_time_ns)
    u, times_ns, pdb_path, dcd_path = build_clean_universe(
        top, traj, sels, stride, dt_ps, outdir, system_name)
    rmsd_self = rmsd_self_fit(u, sels)
    rmsd_frame0 = rmsd_vs_frame0(u, sels)
    rmsf = rmsf_both_chains(u, sels)
    print(f"[{system_name}] {n_frames} raw frames, {dt_ps:.3f} ps/frame, "
          f"{u.trajectory.n_frames} frames after stride {stride}")
    return {"u": u, "times_ns": times_ns, "sels": sels, "dt_ps": dt_ps,
            "rmsd_self": rmsd_self, "rmsd_frame0": rmsd_frame0, "rmsf": rmsf}


# ======================================================================
# 4. PLOTTING
# ======================================================================

def plot_hbonds(series, times_ns, labels, title="Hydrogen bond lengths"):
    for i, lab in enumerate(labels):
        plt.plot(times_ns, series[:, i], lw=1, label=lab)
    plt.axhline(3.5, ls="--", c="grey", lw=0.8, label="3.5 Å cutoff")
    plt.xlabel("time (ns)"); plt.ylabel("donor-acceptor distance (Å)")
    plt.title(title); plt.legend(fontsize=7); plt.show()


def plot_rmsd_compare(df_wt, times_wt, df_mut, times_mut, columns, col_labels,
                      name_wt="Wild type", name_mut="Mutant", suptitle=""):
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        name = col_labels.get(col, col)
        ax.plot(times_wt, df_wt[col], lw=1, label=name_wt)
        ax.plot(times_mut, df_mut[col], lw=1, label=name_mut)
        ax.set_title(name); ax.set_ylabel(f"{name} RMSD (Å)"); ax.legend(fontsize=8)
    axes[-1].set_xlabel("time (ns)")
    if suptitle:
        fig.suptitle(suptitle, y=1.01)
    plt.tight_layout(); plt.show()


def plot_rmsf_compare(rmsf_wt, rmsf_mut, chain_labels,
                      name_wt="Wild type", name_mut="Mutant", suptitle=""):
    chains = list(rmsf_wt["chain"].unique())
    for ch in chains:
        nw = int((rmsf_wt["chain"] == ch).sum())
        nm = int((rmsf_mut["chain"] == ch).sum())
        if nw != nm:
            raise ValueError(f"{chain_labels.get(ch, ch)}: WT {nw} vs mutant {nm} "
                             "residues; RMSF overlay would misalign.")
    fig, axes = plt.subplots(len(chains), 1, figsize=(10, 3 * len(chains)), sharey=True)
    if len(chains) == 1:
        axes = [axes]
    for ax, ch in zip(axes, chains):
        name = chain_labels.get(ch, ch)
        w = rmsf_wt[rmsf_wt["chain"] == ch]; m = rmsf_mut[rmsf_mut["chain"] == ch]
        ax.plot(w["resid"], w["rmsf"], lw=1, label=name_wt)
        ax.plot(m["resid"], m["rmsf"], lw=1, label=name_mut)
        ax.set_title(name); ax.set_ylabel(f"{name} RMSF (Å)"); ax.legend(fontsize=8)
    axes[-1].set_xlabel("residue (file numbering)")
    if suptitle:
        fig.suptitle(suptitle, y=1.01)
    plt.tight_layout(); plt.show()


def compare_means(df_wt, df_mut, columns, col_labels,
                  name_wt="Wild type", name_mut="Mutant"):
    rows = []
    for col in columns:
        mw, mm = df_wt[col].mean(), df_mut[col].mean()
        rows.append({"quantity": col_labels.get(col, col),
                     name_wt: round(mw, 2), name_mut: round(mm, 2),
                     "Δ (mut−wt)": round(mm - mw, 2)})
    return pd.DataFrame(rows)


def plot_site_rmsd(r_wt, times_wt, r_mut, times_mut, title,
                   name_wt="Wild type", name_mut="V388I mutant"):
    plt.plot(times_wt, r_wt, lw=1, label=name_wt)
    plt.plot(times_mut, r_mut, lw=1, label=name_mut)
    plt.xlabel("time (ns)"); plt.ylabel("Cα RMSD (Å)")
    plt.title(title); plt.legend(); plt.show()


def plot_contacts_combined(pairs, u_wt, times_wt, u_mut, times_mut,
                           name_wt="Wild type", name_mut="V388I mutant"):
    """One combined graph per system (all contacts overlaid) + comparison table.
    `pairs` = {label:(sel1,sel2)} defined on WT and applied to both."""
    dist_wt = {lab: pair_min_distance(u_wt, s1, s2) for lab, (s1, s2) in pairs.items()}
    dist_mut = {lab: pair_min_distance(u_mut, s1, s2) for lab, (s1, s2) in pairs.items()}
    for dist, times, title in [(dist_wt, times_wt, name_wt), (dist_mut, times_mut, name_mut)]:
        for lab in pairs:
            plt.plot(times, dist[lab], lw=1, label=lab)
        plt.xlabel("time (ns)"); plt.ylabel("min heavy-atom distance (Å)")
        plt.title(f"Key contacts: {title}"); plt.legend(fontsize=7, ncol=2); plt.show()
    summary = pd.DataFrame({"contact": list(pairs.keys()),
                            f"{name_wt}_mean":  [dist_wt[l].mean() for l in pairs],
                            f"{name_mut}_mean": [dist_mut[l].mean() for l in pairs]}).round(2)
    summary["Δ (mut−wt)"] = (summary[f"{name_mut}_mean"] - summary[f"{name_wt}_mean"]).round(2)
    return summary


def write_view_pdb(u, sels, outpath, frame=0):
    """Optional: write one frame with chain IDs (A=CK1a, B=CRBN, L=ligand,
    Z=zinc) for PyMOL. Not part of the main flow."""
    if not hasattr(u.atoms, "chainIDs"):
        u.add_TopologyAttr("chainID")
    u.atoms.chainIDs = "X"
    u.select_atoms(sels["A"]).atoms.chainIDs = "A"
    u.select_atoms(sels["B"]).atoms.chainIDs = "B"
    u.select_atoms(sels["LIG"]).atoms.chainIDs = "L"
    zn = u.select_atoms(sels["ZN"])
    if len(zn):
        zn.atoms.chainIDs = "Z"
    u.trajectory[frame]
    u.atoms.write(outpath)
    return outpath

