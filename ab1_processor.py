from Bio.Align import PairwiseAligner
from Bio import SeqIO
import pandas as pd
import numpy as np
import regex
import sys
import os

# directory setup
outdir_int = 'integrated'
outdir_rep = 'replicates'
indir = 'raw'

# target sequences (temporary solution)
r255x = (
    'ATGCCTTTTCAAACTTCGCCAGGGGGCAAGGCTGAGGGGGGTGGGGCCACCACATCCACCCAGGTCAT'
    'GGTGATCAAACGCCCCGGCAGGAAGTGAAAAGCTGAGGCCGACCCTCAGGCCATTCCCAAGAAACGGG'
    'GCCGAAAGCCGGGGAGTGTGGTGGCAGCCGCTGCCGCCGAGGCCAAAAAGAAAGCCGTGAAGGAGTCT'
    'TCTATCCGATCTGTGCAGGAGACCGTACTCCCCATCAAGAA'
    )
r255x_trgt = (
    'AGTGAAAAGC'
    )

# alignment algorithm
aligner = PairwiseAligner()

# print verbose
def printv(msg: str, v: bool):
    if v: print(msg)

# return a csv with relevant information from the Sanger rawdata.
def process_ab1(infile: str):
    record = SeqIO.read(infile, "abi")
    abi = record.annotations["abif_raw"]

    bases = abi["PBAS2"].decode("ascii")
    positions = abi["PLOC2"]
    dye_order = abi["FWO_1"].decode()
    channels = [abi[f"DATA{i}"] for i in range(9, 13)]
    traces = dict(zip(dye_order, channels))

    rows = []
    for base, pos in zip(bases, positions):
        pos = int(pos)  # ensure Python int, not numpy scalar
        row = {
            "base": base,
            "position": pos,
            "A_height": int(traces["A"][pos]),
            "C_height": int(traces["C"][pos]),
            "G_height": int(traces["G"][pos]),
            "T_height": int(traces["T"][pos]),
        }
        rows.append(row)

    return pd.DataFrame(rows)

# compile target into master df.
def compile_master(trgt: str):
    chars = []
    for char in trgt:
        chars.append(char)
    df_master = pd.DataFrame({
        'base': chars
        })
    return(df_master)

# align a sanger csv to the master df.
def align_sanger_to_master(
        alignment,
        df_master,
        df_sanger,
        sanger_name='sanger'
        ):
    trgt_coords = alignment.aligned[0]
    seq_coords = alignment.aligned[1]

    rows = []
    last_t = 0
    last_s = 0
    for (left_s, right_s), (left_t, right_t) in zip(seq_coords, trgt_coords):
        if left_t > last_t:
            gap_len = left_t - last_t
            for _ in range(gap_len):
                gap_row = {
                    **{col: np.nan for col in df_sanger.columns},
                    'target': df_master.iloc[last_t]['base']
                    }
                rows.append(gap_row)
                last_t += 1

        s_slice = df_sanger.iloc[left_s:right_s]
        t_slice = df_master.iloc[left_t:right_t]
        for (_, s_row), (_, t_row) in zip(
            s_slice.iterrows(), t_slice.iterrows()
            ):
            merged = s_row.to_dict()
            merged['target'] = t_row['base']
            rows.append(merged)

        last_t = right_t
        last_s = right_s

    df = pd.DataFrame(rows)
    df = df.rename(columns={'base': sanger_name})
    col_order = ['target', f'{sanger_name}']
    col_order += [c for c in df.columns if c not in col_order]
    df = df[col_order]
    return df

# regularize a Sanger df. wrapper for the align_sanger_to_master(...) fxn.
def regularize_target(
        df_master: pd.DataFrame,
        df_sanger: pd.DataFrame,
        sanger_name: str
    ):
    seq = ''.join(df_sanger['base'].astype(str))
    trgt = ''.join(df_master['base'].astype(str))

    alignment = aligner.align(trgt, seq)[0]
    df_master = align_sanger_to_master(
        alignment, df_master, df_sanger, sanger_name
        )
    return df_master

def find_target_region(df_master, sanger_name, trgt: str):
    pattern = f'{trgt}'
    pattern += '{s<=5}'
    pattern = regex.compile(pattern)
    seq = ''.join(df_master[sanger_name].fillna('N').astype(str))
    match = pattern.search(seq)
    if not match: return None, None
    return match.span()

def append_GA_statistics(df_regularized):
    df = df_regularized

    cols = ['A_height', 'C_height', 'G_height', 'T_height']
    df['total_height'] = df[cols].sum(axis=1, min_count=1).replace(0, np.nan)
    df['A_fraction'] = df['A_height'] / df['total_height']
    df['G_fraction'] = df['G_height'] / df['total_height']
    df['C_fraction'] = df['C_height'] / df['total_height']
    df['T_fraction'] = df['T_height'] / df['total_height']

    df['R_i'] = (
        df['G_fraction'] / (df['A_fraction'] + df['G_fraction'])
        )
    df['w_i'] = (
        (df['A_height'] + df['G_height']) / df['total_height']
        )

    return df

# NaN's bad reads based on total peak height and a noise cutoff determined from
# an eCDF of a bunch of clean reads. noise_cutoff is the 95th percentile of
# background noise. see quantify_noise.py
def quality_control(df, noise_cutoff=0.2378):
    bases = ['A_fraction', 'G_fraction', 'C_fraction', 'T_fraction']

    height_cutoff = df['total_height'].median() * 0.10
    height_mask = df['total_height'] >= height_cutoff

    df['minor_frac'] = [
        (df.loc[i, bases].sum() - df.at[i, f"{b}_fraction"])
        if pd.notna(b) else np.nan
        for i, b in zip(df.index, df['read'])
        ]
    # for A->G event permissibility
    df['minor_frac'] = [
        df.loc[i, bases].sum() - df.at[i, "G_fraction"] - df.at[i, "A_fraction"]
        if (t == 'A') else df.at[i, 'minor_frac']
        for i, b, t in zip(df.index, df['read'], df['target'])
        ]
    noise_mask = df['minor_frac'] <= noise_cutoff

    nan_mask = df['read'].isna()

    df['qc_ok'] = 1
    df.loc[~height_mask, 'qc_ok'] = 0
    df.loc[~noise_mask, 'qc_ok'] = 0
    df.loc[nan_mask, 'qc_ok'] = np.nan

    return df.drop(columns=['minor_frac'])

# process one Sanger csv to be aligned with the target and to carry relevant
# statistics. save to replicate outdir.
def process_sanger_csv(
        df_sanger: pd.DataFrame,
        infile: str,
        outdir: str,
        outfile: str,
        trgt: str,
        verbose: bool
    ):
    v = verbose
    df_master = compile_master(trgt=trgt)

    df = regularize_target(df_master, df_sanger, 'read')
    df = append_GA_statistics(df)
    df = quality_control(df)

    # print entire ADAR-relevant Sanger readout
    printv(df.drop(columns=['C_height', 'T_height']).to_string(), v)

    # print all adenosines in Sanger readout
    df_A = (
        df.query("target == 'A'")
          .sort_values('R_i', ascending=False)
        )
    printv(df_A.drop(columns=['C_height','T_height']).to_string(), v)

    # print target region
    left, right = find_target_region(df, 'read', trgt=r255x_trgt)
    if not left:
        printv(f"{infile} does not have an identifiable target (s<=5)", v)
    else:
        printv(
            f'Guide canonically pairs with ({left}, {right}); '
            f'targeted A is at {left + 4}',
            v
            )

    df.to_csv(f"{outdir}/{outfile}")

def main():
    files = sys.argv[1:]
    infiles = [os.path.basename(f) for f in files]
    for infile in infiles:
        without_dir = infile.split('/')[1]
        outfile = without_dir.replace('ab1', 'csv')

        df = process_ab1(infile=infile)
        df = process_sanger_csv(
            df_sanger=df,
            infile=infile,
            outdir=outdir_rep,
            outfile=outfile,
            trgt=r255x,
            verbose=False
            )

if __name__ == "__main__":
    main()

