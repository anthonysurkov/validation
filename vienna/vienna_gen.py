import pandas as pd
import sys, os
import subprocess, shutil, glob, tempfile

def rnaup_out_to_pd(infile):
    fluff_count = 7 # number of useless lines in .out output
    with open(infile) as f:
        for i, line in enumerate(f):
            if i == fluff_count:
                break

        trgt_rows = []
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0] == '#':
                break
            trgt_rows.append({
                "pos": parts[0],
                "u4S": parts[1],
                "dG": parts[2],
            })

        guide_rows = []
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            guide_rows.append({
                "pos": parts[0],
                "u4S": parts[1]
            })

    df_trgt = pd.DataFrame(trgt_rows)
    df_guide = pd.DataFrame(guide_rows)
    return df_trgt, df_guide


def run_rnaup(target, guide, outfile, w=200, ulen='4'):
    name = "rnaup_tmp"
    input_text = f">{name}\n{target}\n>g\n{guide}\n"

    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["RNAup", "-b", "-w", str(w), "-u", str(ulen)]
        res = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            cwd=tmp,
            capture_output=True,
        )
        pattern = os.path.join(tmp, f"*.out")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(
                f"No RNAup .out produced, got: {os.listdir(tmp)}"
                )

        #print(matches[0])
        shutil.move(matches[0], outfile)

    return rnaup_out_to_pd(infile=outfile)

guides = pd.read_json('../guides.json')
targets = pd.read_json('../targets.json')

guides = guides.merge(
    targets[['id', 'seq']],
    left_on='target',
    right_on='id',
    how='inner'
)

guides = guides.rename(
    columns={
        'id_x': 'id',
        'seq_x': 'guide',
        'seq_y': 'target_seq'
    })

flanks = pd.read_json('../flanks.json')
flank_wide = flanks.pivot(
    index='target', columns='id', values='seq'
    ).reset_index()
flank_wide.columns.name = None

guides = guides[['id', 'guide', 'target', 'target_seq']]
guides = guides.merge(flank_wide, on='target', how='left')
guides['guide'] = guides['left'] + guides['guide'] + guides['right']

for _, row in guides.iterrows():
    trgt_id = row['target']
    trgt_seq = row['target_seq']
    guide_seq = row['left'] + row['guide'] + row['right']
    guide_id = row['id']
    df_trgt, df_guide = run_rnaup(
        trgt_seq, guide_seq, outfile=f"{guide_id}.out"
        )
    df_trgt.to_csv(f"rnaup_target_{guide_id}.csv")
    df_guide.to_csv(f"rnaup_guide_{guide_id}.csv")

