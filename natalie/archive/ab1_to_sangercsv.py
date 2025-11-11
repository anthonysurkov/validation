from Bio import SeqIO
import pandas as pd
import regex
import sys

outdir = "as_pandas"

# currently unnecessary, full Sanger data may actually open up more adenosines
trgt = regex.compile(
    r"([CN][CN][CN][GN][GN][CN][AG][GN][GN][AG][AG][GN]T[GN][AG][AG][AG][AG]"
    r"[GN][CN]T[GN][AG][GN][GN][CN][CN][GN][AG][CN]){e<=1}"
)

"""
def find_guide_region(df, infile):
    seq = ''.join(df['base'].astype(str))
    found = trgt.search(seq)
    if not found:
        print(df.to_string())
        print(seq)
        raise ValueError(f"{infile} did not yield an R255X target")
    return found.span()
"""

def process_ab1(infile: str, outfile: str):
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

    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/{outfile}")

    """
    left, right = find_guide_region(df, infile)
    idx = range(left, right)
    df_extract = df.iloc[idx]
    print(df_extract.to_string())
    print(f'Saving as {outfile}...')
    df_extract.to_csv(f"{outdir}/{outfile}")
    """

for infile in sys.argv[1:]:
    print(f'Processing {infile}...')
    outfile = infile.replace(".ab1", ".csv")
    process_ab1(infile, outfile)
