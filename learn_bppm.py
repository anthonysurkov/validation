# model
# per structure: gamma(s) = sigmoid(-deltaF(s) / RT), deltaF(s) = Beta^T x_i(s)
# per guide: theta_i = (sum over s) p_i(s) gamma(s)
# observed: k_i ~ binomial(n_i, theta_i)
# objective: min_beta { (sum over i) k_i log(theta_i) + (n_i - k_i)log(1 - theta_i) } + lambda || beta ||_2^2

from bppm import Bppm_Features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -------------------------
# 0) Inputs
# -------------------------
r255x = Bppm_Features('r255x')
Xdf = r255x.X.copy().fillna(0.0)          # (n_guides, n_features)
k = r255x.df_emerge["k"].to_numpy(float)  # successes
n = r255x.df_emerge["n"].to_numpy(float)  # trials

# -------------------------
# 1) Physics sign/scale: eta = -(X beta)/RT
# -------------------------
R = 1.98720425864083e-3   # kcal/(mol*K)
T = 310.15                # 37C in Kelvin; change if you want
RT = R * T

X = Xdf.to_numpy()
Xrt = (-1.0 / RT) * X     # bake in -1/RT

# -------------------------
# 2) Train/test split (by guide)
# -------------------------
idx = np.arange(len(n))
tr, te = train_test_split(idx, test_size=0.2, random_state=0)

Xtr, Xte = Xrt[tr], Xrt[te]
ktr, ntr = k[tr], n[tr]
kte, nte = k[te], n[te]

# -------------------------
# 3) Weighted Bernoulli construction: 2 rows per guide
# -------------------------
X2_tr = np.vstack([Xtr, Xtr])
y2_tr = np.concatenate([np.ones_like(ktr), np.zeros_like(ktr)])
w2_tr = np.concatenate([ktr, (ntr - ktr)])

# -------------------------
# 4) Fit L2-logistic (this is your beta)
# -------------------------
clf = LogisticRegression(
    penalty="l2",
    C=1.0,                # sweep this later; larger C = weaker reg
    solver="lbfgs",
    fit_intercept=False,  # set True if you want a baseline term
    max_iter=4000
)
clf.fit(X2_tr, y2_tr, sample_weight=w2_tr)

beta_hat = clf.coef_.ravel()

# -------------------------
# 5) Predict theta on test guides
# -------------------------
theta_hat = clf.predict_proba(Xte)[:, 1]  # (n_test,)

# -------------------------
# 6) Validations
# -------------------------
def binom_nll(theta, k, n, eps=1e-12):
    theta = np.clip(theta, eps, 1 - eps)
    return -np.sum(k*np.log(theta) + (n-k)*np.log(1-theta))

# (A) Test NLL vs baseline constant rate
p0 = ktr.sum() / ntr.sum()
nll_model = binom_nll(theta_hat, kte, nte)
nll_base  = binom_nll(np.full_like(theta_hat, p0), kte, nte)
print("Test NLL:", nll_model)
print("Baseline NLL:", nll_base)
print("NLL improvement (base - model):", nll_base - nll_model)

# (B) Rate tracking (k/n) + weighted MAE
rate = kte / nte
wmae = np.sum(nte * np.abs(theta_hat - rate)) / np.sum(nte)
corr = np.corrcoef(theta_hat, rate)[0, 1]
print("Weighted MAE on rates:", wmae)
print("Pearson corr(theta, k/n):", corr)

# (C) Calibration table (deciles by predicted theta)
cal = pd.DataFrame({"theta": theta_hat, "k": kte, "n": nte})
cal["bin"] = pd.qcut(cal["theta"], q=10, duplicates="drop")
g = cal.groupby("bin", observed=True)
cal_table = pd.DataFrame({
    "theta_mean": g["theta"].mean(),
    "obs_rate": g["k"].sum() / g["n"].sum(),
    "n_sum": g["n"].sum(),
    "k_sum": g["k"].sum(),
})
print("\nCalibration (by theta decile):")
print(cal_table)

# (D) Quick coefficient sanity: top |beta|
top = np.argsort(np.abs(beta_hat))[::-1][:20]
top_features = Xdf.columns[top]
print("\nTop |beta| features:")
for f, b in zip(top_features, beta_hat[top]):
    print(f, b)
