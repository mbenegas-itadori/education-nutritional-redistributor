"""
05_elasticidades_nutricionais.py
=================================
Calcula elasticidades-renda nutricionais eta_n e eta_n^k:

    eta_n   = sum_j Omega_nj * mu_j          (S1, baseline)
    eta_n^k = sum_j Omega_nj * mu_j^k        (S4, por estrato k)

Erros-padrão por delta method:
    Var(eta_n) = Omega_n' * V_mu * Omega_n
    onde V_mu é construída a partir da covariância dos beta_i em V_SUR.

Inputs:
    POF/output/quaids_s1_resultados.json
    POF/output/quaids_s4_resultados.json
    POF/output/quaids_cov_mats.npz
    POF/output/matriz_omega.parquet

Outputs:
    POF/output/eta_nutricional.json
    (impressão em texto para colar no chat)

Uso:
    exec(open('/content/drive/MyDrive/POF/script/05_elasticidades_nutricionais.py').read())
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path('/content/drive/MyDrive/POF/output')

# ── Carrega inputs ───────────────────────────────────────────────
s1    = json.load(open(OUT / 'quaids_s1_resultados.json'))
s4    = json.load(open(OUT / 'quaids_s4_resultados.json'))
cov   = np.load(OUT  / 'quaids_cov_mats.npz')
omega = pd.read_parquet(OUT / 'matriz_omega.parquet')

GRUPOS     = s1['grupos']
N_G        = len(GRUPOS)
IDX_DROP   = s1['IDX_DROP']        # 7 = Carnes
EQS        = [i for i in range(N_G) if i != IDX_DROP]
nutrientes = omega.index.tolist()

w_bar   = np.array(s1['w_bar'])
mu_S1   = np.array(s1['mu'])
V_S1    = cov['V_SUR_S1']
V_S4    = cov['V_SUR_S4']

Ks_S1   = s1['Ks']
offs_S1 = s1['offs']
Ks_S4   = s4['Ks']
offs_S4 = s4['offs']

POS_BETA = 14

# Omega: (N_nut × N_G), colunas na ordem de GRUPOS
Omega = omega.reindex(columns=GRUPOS).fillna(0).values

# ── Sanidade ─────────────────────────────────────────────────────
print("Verificação Omega (soma por nutriente):")
for i, n in enumerate(nutrientes):
    s = Omega[i].sum()
    ok = "✓" if abs(s - 1) < 1e-3 else "✗"
    print(f"  {n:20s}: {s:.6f} {ok}")

# ── V_mu S1 (covariância dos mu_i por delta method) ─────────────
pos_beta_S1 = [offs_S1[j] + POS_BETA for j in range(len(EQS))]
V_mu_S1 = np.full((N_G, N_G), np.nan)
for j1, i1 in enumerate(EQS):
    for j2, i2 in enumerate(EQS):
        V_mu_S1[i1, i2] = (V_S1[pos_beta_S1[j1], pos_beta_S1[j2]]
                            / (w_bar[i1] * w_bar[i2]))

# ── eta_n S1 e SE ─────────────────────────────────────────────────
eta_S1     = Omega @ mu_S1
var_eta_S1 = np.zeros(len(nutrientes))
for n in range(len(nutrientes)):
    om     = Omega[n]
    idx_ok = [i for i in range(N_G) if not np.isnan(V_mu_S1[i, i])]
    var_eta_S1[n] = om[idx_ok] @ V_mu_S1[np.ix_(idx_ok, idx_ok)] @ om[idx_ok]
se_eta_S1 = np.sqrt(np.maximum(var_eta_S1, 0))

# ── mu por estrato S4 ────────────────────────────────────────────
theta_S4 = np.array(s4['theta_hat'])
beta_S4  = np.array(s4['beta_b'])
estratos = ['fund', 'medio', 'sup']

pos_delta = {}
for j, i in enumerate(EQS):
    base = offs_S4[j]; k_eq = Ks_S4[j]
    if k_eq == 20:
        pos_delta[i] = (base+17, base+18, base+19)
    elif k_eq == 19:
        pos_delta[i] = (base+16, base+17, base+18)
    else:
        raise ValueError(f"Ks_S4 inesperado: {k_eq} (eq {j}, grupo {GRUPOS[i]})")

mu_base = 1 + beta_S4 / w_bar
mu_strat = {'base': mu_base.copy()}
for ek, estrato in enumerate(estratos):
    mu_k = mu_base.copy()
    for i in EQS:
        mu_k[i] = 1 + (beta_S4[i] + theta_S4[pos_delta[i][ek]]) / w_bar[i]
    soma_outros = sum(beta_S4[i] + theta_S4[pos_delta[i][ek]] for i in EQS)
    mu_k[IDX_DROP] = 1 + (-soma_outros) / w_bar[IDX_DROP]
    mu_strat[estrato] = mu_k

eta_strat = {k: Omega @ mu_k for k, mu_k in mu_strat.items()}

# SE por estrato (delta method com V_S4)
pos_beta_S4 = [offs_S4[j] + POS_BETA for j in range(len(EQS))]
se_eta_strat = {}
for ek, estrato in enumerate(estratos):
    var_eta_k = np.zeros(len(nutrientes))
    for n in range(len(nutrientes)):
        om = Omega[n]
        V_mu_k = np.full((N_G, N_G), np.nan)
        for j1, i1 in enumerate(EQS):
            for j2, i2 in enumerate(EQS):
                pb1 = pos_beta_S4[j1]; pb2 = pos_beta_S4[j2]
                pd1 = pos_delta[i1][ek]; pd2 = pos_delta[i2][ek]
                V_mu_k[i1, i2] = (
                    V_S4[pb1,pb2] + V_S4[pd1,pd2] +
                    V_S4[pb1,pd2] + V_S4[pd1,pb2]
                ) / (w_bar[i1] * w_bar[i2])
        idx_ok = [i for i in range(N_G) if not np.isnan(V_mu_k[i, i])]
        var_eta_k[n] = om[idx_ok] @ V_mu_k[np.ix_(idx_ok,idx_ok)] @ om[idx_ok]
    se_eta_strat[estrato] = np.sqrt(np.maximum(var_eta_k, 0))

# ── Impressão ────────────────────────────────────────────────────
def sig(t):
    a = abs(t)
    return '***' if a>2.576 else ('**' if a>1.96 else ('*' if a>1.645 else 'ns'))

print("\n" + "="*70)
print("BLOCO 1 — eta_n S1 (baseline) com erros-padrão")
print("="*70)
print(f"{'nutriente':22s} {'eta_n':>8s} {'se':>8s} {'t':>7s} {'sig':>5s}")
print("-"*55)
for n, nut in enumerate(nutrientes):
    e = eta_S1[n]; se = se_eta_S1[n]
    t = e/se if se > 0 else float('nan')
    print(f"{nut:22s} {e:8.4f} {se:8.4f} {t:7.2f} {sig(t):>5s}")

print("\n" + "="*70)
print("BLOCO 2 — eta_n por estrato S4 (base/fund/medio/sup)")
print("="*70)
print(f"{'nutriente':22s} {'base':>7s} {'fund':>7s} {'medio':>7s} {'sup':>7s}")
print("-"*55)
for n, nut in enumerate(nutrientes):
    vals = [eta_strat[k][n] for k in ['base','fund','medio','sup']]
    print(f"{nut:22s} " + " ".join(f"{v:7.4f}" for v in vals))

print("\n" + "="*70)
print("BLOCO 3 — Delta eta_n = eta_sup - eta_base (com SE e sig)")
print("="*70)
print(f"{'nutriente':22s} {'Deta':>8s} {'se':>8s} {'t':>7s} {'sig':>5s}")
print("-"*55)
for n, nut in enumerate(nutrientes):
    deta = eta_strat['sup'][n] - eta_strat['base'][n]
    se_d = se_eta_strat['sup'][n]
    t    = deta/se_d if se_d > 0 else float('nan')
    print(f"{nut:22s} {deta:8.4f} {se_d:8.4f} {t:7.2f} {sig(t):>5s}")

print("\n" + "="*70)
print("FIM. Copie os blocos 1-3 e cole no chat.")
print("="*70)

# ── Salva JSON ───────────────────────────────────────────────────
resultado = {
    'nutrientes': nutrientes,
    'grupos': GRUPOS,
    'Omega': Omega.tolist(),
    'S1': {'eta': eta_S1.tolist(), 'se_eta': se_eta_S1.tolist()},
    'S4': {
        'estratos': ['base','fund','medio','sup'],
        'eta_por_estrato': {k: eta_strat[k].tolist()
                            for k in ['base','fund','medio','sup']},
        'se_eta_sup': se_eta_strat['sup'].tolist(),
    }
}
with open(OUT / 'eta_nutricional.json', 'w') as f:
    json.dump(resultado, f, indent=2)
print(f"\n✓ Salvo: eta_nutricional.json")
