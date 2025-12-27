import pandas as pd
import RNA
from collections import defaultdict

class Bppm_Features:
    def __init__(self, target_id: str, zero_idx: bool = True):
        self.target_id = target_id
        self.zero_idx = zero_idx

        target_info = pd.read_json('data/targets.json')
        self.template = target_info.loc[
            target_info['target_id'] == self.target_id,
            'emerge_seq'
        ].item()
        self.center = target_info.loc[
            target_info['target_id'] == self.target_id,
            'emerge_At_idx'
        ].item()

        # replace with something more robust:
        self.df_emerge = pd.read_csv(
            f'data/emerge/{target_id}_top.csv',
            index_col=0
        )
        self.df_emerge['5to3'] = self.df_emerge['5to3'].str.replace('T', 'U')
        self.load_hairpinized_emerge()

        # get self.var_idx_At from a json
        # temporary:
        self.var_idx = list(range(65, 75))
        self.var_idx_At = [i - self.center for i in self.var_idx]

        features = []
        for idx, row in self.df_emerge.iterrows():
            print(idx) # temporary progress bar
            seq = row['hairpin']
            bppm = self.get_bppm(seq)
            labeled_bps = row['labeled_bps']
            feats = self.collect_hairpin_features(bppm, labeled_bps)
            features.append(feats)

        def squash(features):
            out = {}
            for f in features:
                out.update(f)
            return out
        self.X = pd.DataFrame.from_records([squash(row) for row in features])

        if self.X.isna().any().any():
            raise ValueError('X matrix contains NaN. Check basis')

        self.prune_features() # for 0

    def collect_hairpin_features(self, bppm, labeled_bps):
        n = len(bppm)
        features = []
        for i in range(n):
            for j in range(i+1, n):
                base_i = labeled_bps[i]
                base_j = labeled_bps[j]
                if i in self.var_idx or j in self.var_idx:
                    feats = self.compose_variable_features(
                        base_i,
                        base_j,
                        bppm[i][j]
                    )
                else:
                    feats = {base_i + base_j: bppm[i][j]}
                features.append(feats)
        return features

    @staticmethod
    def get_bppm(hairpin_seq: str):
        fc = RNA.fold_compound(hairpin_seq)
        fc.pf()
        bpp1 = fc.bpp() # (N+1) x (N+1), dummy at 0. thanks Vienna
        bpp0 = [row[1:] for row in bpp1[1:]] # adjustment to 0-idx
        return bpp0

    def compose_variable_features(
        self,
        base_L: str,
        base_R: str,
        bp_prob: float,
        base_alphabet: set[str] = {'A', 'G', 'C', 'U'}
    ):
        nt_L = base_L[0]
        nt_R = base_R[0]

        idx_L = base_L[1:]
        idx_R = base_R[1:]

        var_L: bool = int(idx_L) in self.var_idx_At
        var_R: bool = int(idx_R) in self.var_idx_At

        bases_L = base_alphabet if var_L else {nt_L}
        bases_R = base_alphabet if var_R else {nt_R}

        constructed = defaultdict(float)
        for bL in bases_L:
            for bR in bases_R:
                constructed[bL + idx_L + bR + idx_R]

        constructed[base_L + base_R] = bp_prob
        return constructed

    def prune_features(self):
        self.X = self.X.loc[:, (self.X != 0).any(axis=0)]

    def load_hairpin(self, 5to3: str):
        return self.template.replace('NNNNNNNNNN', 5to3)

    def load_hairpinized_emerge(self):
        hairpins = []
        indices = []
        labeled_bps = []
        for _, row in self.df_emerge.iterrows():
            5to3 = row['5to3']

            hairpin = self.load_hairpin(5to3)
            hairpins.append(hairpin)

            row_indices = self.index_string(hairpin)
            hairpinized_indices = self.hairpinize_indices(row_indices)

            labeled_bp = [
                hairpin[i] + str(hairpinized_indices[i])
                for i in range(len(hairpinized_indices))
            ]
            labeled_bps.append(labeled_bp)

        self.df_emerge['hairpin'] = hairpins
        self.df_emerge['labeled_bps'] = labeled_bps

    def hairpinize_indices(self, indices: list[int]):
        center = self.center if self.zero_idx else self.center + 1
        return self.offset_indices(indices, center)

    @staticmethod
    def offset_indices(indices: list[int], new_center: int):
        return [index - new_center for index in indices]

    def index_string(self, string: str):
        indices = []
        count = 0 if self.zero_idx else 1
        for char in string:
            indices.append(count)
            count += 1
        return indices

r255x = Bppm_Features('ctd1')
print(r255x.X)
print(r255x.center)
print(r255x.template)
print(r255x.var_idx)
print(r255x.var_idx_At)
print([r255x.template[i] for i in r255x.var_idx])
print([r255x.template[i] for i in r255x.var_idx_At])

