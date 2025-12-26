import pandas as pd
from collections import defaultdict
from replicates import SangerHandler
from triplicates import Triplicate

DATA_DIR = 'data/raw'

def load_sanger_files(metadata_infile: str):
    targets_json = 'data/targets.json'
    df_metadata = pd.read_json(metadata_infile)

    sanger = defaultdict(lambda: {"target": None, "handlers": []})
    for (file, target, guide) in zip(
        df_metadata['file'],
        df_metadata['target'],
        df_metadata['guide']
    ):
        # temporary:
        if target == 'ctd1':
            continue

        infile = f'{DATA_DIR}/{file}'
        handler = SangerHandler(target, infile, targets_json)

        if sanger[guide]['target'] is None:
            sanger[guide]['target'] = target
        elif sanger[guide]['target'] != target:
            raise ValueError(
                f'Guide {guide!r} has multiple targets: '
                f'{sanger[guide]["target"]!r} vs {target!r}")'
            )
        sanger[guide]['handlers'].append(handler)

    return sanger

sanger_by_guide = load_sanger_files(f'{DATA_DIR}/metadata.json')

triplicates = []
for guide, payload in sanger_by_guide.items():
    triplicates.append(
        Triplicate(
            target_id = payload['target'],
            guide_id = guide,
            handlers = payload['handlers']
        )
    )

for t in triplicates:
    t.to_csv()
