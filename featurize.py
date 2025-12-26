import pandas as pd
import numpy as np
import sys
import os
import glob
from functools import lru_cache

guides_repo = pd.read_json('guides.json')

@lru_cache(maxsize=1)
def load_vienna_features():
    target_feature_sets = {}
    for infile in glob.glob('vienna/*_target_*'):
        filename = os.path.basename(infile)

        guide_id = filename.split('_')[0].lower()
        df_target = pd.read_csv(infile, index_col=0)

        target_feature_sets[guide_id] = df_target
    return target_feature_sets

def get_vienna_features(guide_id: str):
    target_feature_sets = load_vienna_features()
    return target_feature_sets.get(guide_id)

def featurize_integrated_csv(df: pd.DataFrame, guide_id: str, target_id: str):
    target_features = get_vienna_features(guide_id=guide_id)

    df = df[df['valid'] == 1]
    df = df[['pos', 'target', 'R_i', 'w_i', 'edit_bool']]
    for i in range(-3,4):
        df[f'{i}_pos'] = df['pos'] + i
        df = (
            df.merge(
                target_features[['pos', 'u4S', 'dG']],
                left_on=f'{i}_pos',
                right_on='pos',
                how='left'
                )
            .rename(columns={
                'pos_x': 'pos',
                'u4S': f'{i}_u4S',
                'dG': f'{i}_dG'
                })
            .drop(columns=['pos_y', f'{i}_pos'])
            )
    return df

outdir = 'natalie/feature_sets'
infiles = sys.argv[1:]
filenames = [os.path.basename(f) for f in infiles]
dfs = []
for infile, filename in zip(infiles, filenames):
    df = pd.read_csv(infile, index_col=0)
    df = df.reset_index(names='idx')

    df['pos'] = df['idx'] + 1 # idx --> pos
    df = df.drop(columns=['idx'])

    guide_id = filename.split('_')[0].lower()
    print(guide_id)
    guide_id = guides_repo.set_index('legacy').at[guide_id, 'id']
    target_id = guides_repo.set_index('id').at[guide_id, 'target']

    df_f = featurize_integrated_csv(
        df=df,
        guide_id=guide_id,
        target_id=target_id
        )
    outfile = f'{outdir}/{guide_id}_feature_set.csv'
    df_f.to_csv(outfile)
