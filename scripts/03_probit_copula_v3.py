"""
03_probit_copula_v3.py
======================
Pipeline unificado: Probit (estágio 1) + CMLE (seleção de cópula).
Versão 3 — adiciona salvamento das matrizes de covariância dos Probits
para cálculo posterior de Murphy-Topel no estágio 2.

Mudanças em relação à v2:
  + Salva cov_params() de cada Probit em probit_cov_mats.npz
  + Salva xb (índice latente) para cada grupo em probit_xb.parquet
    (necessário para calcular ∂η/∂π no Murphy-Topel)
  + Documenta a estrutura esperada pelo 04_quaids_estimacao.py

Outputs adicionais:
  POF/output/probit_cov_mats.npz
      arrays: cov_{grupo} para cada grupo com cópula
      ex: cov_01.Cereais shape=(13, 13) — (K_probit × K_probit)
  POF/output/probit_xb.parquet
      colunas: xb_{grupo} — índice latente x'π̂ para cada UC

Uso no Murphy-Topel (04_quaids_estimacao.py):
  ∂η_ih/∂xb = -(η_ih² + xb_ih)   [derivada do resíduo generalizado]
  C_i = ρ̂_i · Σ_h (∂²Q₂/∂θ₂∂η_ih) · (∂η_ih/∂π_i') · x_ih'
  V_MT = V₂ + V₂ · (C · V₁ · C') · V₂
"""

import logging
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from pathlib import Path

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURAÇÃO
# ============================================================
OUT_DIR = Path('/content/drive/MyDrive/POF/output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRUPOS_QUAIDS = [
    '01.Cereais', '02.Farinhas_Massas', '03.Tuberculos',
    '04.Acucares_Industrializados', '05.Leguminosas_Oleaginosas',
    '06.Frutas', '07.Legumes_Verduras', '08.Carnes', '09.Laticinios',
    '10.Oleos_Gorduras', '11.Bebidas_NA', '12.Alimentacao_Fora',
    '13.Alcool', '14.Tabaco'
]
N_GRUPOS = len(GRUPOS_QUAIDS)

PROBIT_VARS = [
    'LOG_WELFARE', 'N_MORADORES', 'D_URBANO',
    'D_EDUC_FUND', 'D_EDUC_MEDIO', 'D_EDUC_SUPERIOR',
    'SEXO_REF', 'D_NRESP_RENDA',
    'D_REG_2', 'D_REG_3', 'D_REG_4', 'D_REG_5'
]

LIMIAR_ZEROS = 0.25
EPS          = 1e-8
BOUND_TOL    = 0.01
FRANK_LIMIAR = 1e-4

BOUNDS = {
    'Gaussiana': (-0.99,  0.99),
    'Gumbel':    (1.001, 20.0),
    'Frank':     (-30.0, 30.0),
    'Clayton':   (0.001, 20.0),
}

# ============================================================
# VALIDAÇÃO
# ============================================================
def valida_colunas(base):
    necessarias = (
        PROBIT_VARS
        + [f"x_{g}" for g in GRUPOS_QUAIDS]
        + [f"w_{g}" for g in GRUPOS_QUAIDS]
        + ['REGIAO', 'LN_M_C', 'PESO_FINAL']
    )
    ausentes = [c for c in necessarias if c not in base.columns]
    if ausentes:
        raise KeyError(f"Colunas ausentes: {ausentes}")
    log.info("Validação de colunas: OK (%d verificadas)", len(necessarias))

def verifica_bound(theta, familia, grupo):
    lo, hi = BOUNDS[familia]
    intervalo = hi - lo
    if (abs(theta - lo) < BOUND_TOL * intervalo or
            abs(theta - hi) < BOUND_TOL * intervalo):
        log.warning("CMLE %s | %s: θ=%.4f próximo do bound [%.3f, %.3f]",
                    grupo, familia, theta, lo, hi)
        return True
    return False

# ============================================================
# LOG-VEROSSIMILHANÇAS
# ============================================================
def ll_gaussiana(rho, u, v):
    u, v = np.clip(u, EPS, 1-EPS), np.clip(v, EPS, 1-EPS)
    x, y = norm.ppf(u), norm.ppf(v)
    det = 1.0 - rho**2
    if det <= 1e-10: return -1e10
    ll = -0.5*np.log(det) - (rho**2*(x**2+y**2) - 2*rho*x*y) / (2*det)
    return float(np.sum(np.where(np.isfinite(ll), ll, -1e10)))

def ll_gumbel(theta, u, v):
    if theta < 1 + 1e-6: return -1e10
    u, v = np.clip(u, EPS, 1-EPS), np.clip(v, EPS, 1-EPS)
    t1, t2 = (-np.log(u))**theta, (-np.log(v))**theta
    s = t1 + t2
    s1 = s**(1.0/theta)
    log_c = (-s1 + (1.0/theta - 2.0)*np.log(s)
             + np.log(theta - 1.0 + s1)
             + (theta-1.0)*(np.log(-np.log(u)) + np.log(-np.log(v)))
             - np.log(u) - np.log(v))
    return float(np.sum(np.where(np.isfinite(log_c), log_c, -1e10)))

def ll_frank(theta, u, v):
    if abs(theta) < FRANK_LIMIAR: return -1e10
    u, v = np.clip(u, EPS, 1-EPS), np.clip(v, EPS, 1-EPS)
    et = np.expm1(-theta)
    num = -theta * et * np.exp(-theta*(u+v))
    den = np.clip((et + np.expm1(-theta*u)*np.expm1(-theta*v))**2, EPS, None)
    ll = np.log(np.clip(num/den, EPS, None))
    return float(np.sum(np.where(np.isfinite(ll), ll, -1e10)))

def ll_clayton(theta, u, v):
    if theta < 1e-6: return -1e10
    u, v = np.clip(u, EPS, 1-EPS), np.clip(v, EPS, 1-EPS)
    ll = (np.log(1.0+theta) + (-1.0-theta)*(np.log(u)+np.log(v))
          + (-2.0-1.0/theta)*np.log(u**(-theta)+v**(-theta)-1.0))
    return float(np.sum(np.where(np.isfinite(ll), ll, -1e10)))

LL_FUNCS = {'Gaussiana': ll_gaussiana, 'Gumbel': ll_gumbel,
            'Frank': ll_frank, 'Clayton': ll_clayton}

def u_from_eta(eta): return norm.cdf(eta)
def v_from_w(w):
    n = len(w)
    return (w.argsort().argsort() + 1) / (n + 1)

# ============================================================
# CARREGA BASE
# ============================================================
log.info("Carregando base_quaids_v2.parquet...")
base = pd.read_parquet(OUT_DIR / 'base_quaids_v2.parquet')
log.info("  %d UCs × %d variáveis", *base.shape)

for r in [2, 3, 4, 5]:
    col = f'D_REG_{r}'
    if col not in base.columns:
        base[col] = (base['REGIAO'] == r).astype(int)

valida_colunas(base)

mask_ok = base[PROBIT_VARS].notna().all(axis=1)
log.info("  UCs com regressores completos: %d", mask_ok.sum())
N_OK = mask_ok.sum()

X_prob = sm.add_constant(
    base.loc[mask_ok, PROBIT_VARS].values.astype(float))

# ============================================================
# ETAPA 1 — PROBITS + MATRIZES DE COVARIÂNCIA
# ============================================================
log.info("\n[1/2] Estimando Probits e salvando matrizes de covariância...")
print(f"\n{'Grupo':35s} {'Zeros%':>7s} {'Pseudo-R²':>10s} {'Cópula?':>8s}")
print("-" * 65)

resultados_probit = {}
eta_mat  = np.zeros((len(base), N_GRUPOS))
xb_mat   = np.full((N_OK, N_GRUPOS), np.nan)    # índice latente
cov_mats = {}                                     # covariâncias dos Probits

for i, grupo in enumerate(GRUPOS_QUAIDS):
    d_i      = (base[f"x_{grupo}"] > 0).astype(int).values
    pct_zero = (d_i == 0).mean()

    if pct_zero < LIMIAR_ZEROS:
        print(f"{grupo:35s} {100*pct_zero:7.1f}% {'—':>10s} {'Não':>8s}")
        resultados_probit[grupo] = {'copula': False, 'pct_zero': float(pct_zero)}
        continue

    y_i = d_i[mask_ok]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = sm.Probit(y_i, X_prob).fit(disp=False, maxiter=200)

        # Índice latente e resíduo generalizado
        xb  = res.predict(X_prob, which='linear')
        phi = norm.pdf(xb)
        Phi = norm.cdf(xb)
        eta_i = np.where(
            y_i == 1,
             phi / np.clip(Phi,   EPS, 1.0),
            -phi / np.clip(1-Phi, EPS, 1.0)
        )
        eta_mat[mask_ok, i] = eta_i
        xb_mat[:, i]        = xb   # salva índice latente

        # *** SALVA MATRIZ DE COVARIÂNCIA DO PROBIT ***
        # cov_params() retorna (K_probit × K_probit)
        # K_probit = 1 (const) + 12 (PROBIT_VARS) = 13
        cov_mats[grupo] = res.cov_params()

        resultados_probit[grupo] = {
            'copula':    True,
            'pct_zero':  float(pct_zero),
            'pseudo_r2': float(res.prsquared),
            'params':    res.params.tolist(),
            'K_probit':  int(X_prob.shape[1]),
        }
        print(f"{grupo:35s} {100*pct_zero:7.1f}% "
              f"{res.prsquared:10.4f} {'Sim':>8s}")

    except Exception as exc:
        log.warning("Probit FALHOU para %s: %s", grupo, exc)
        resultados_probit[grupo] = {
            'copula': False, 'pct_zero': float(pct_zero), 'erro': str(exc)
        }
        print(f"{grupo:35s} {100*pct_zero:7.1f}% {'ERRO':>10s}  {str(exc)[:35]}")

grupos_copula     = [g for g, r in resultados_probit.items() if r.get('copula')]
grupos_sem_copula = [g for g, r in resultados_probit.items() if not r.get('copula')]
log.info("Grupos com cópula (%d): %s", len(grupos_copula), grupos_copula)
log.info("Grupos sem cópula (%d): %s", len(grupos_sem_copula), grupos_sem_copula)

# ============================================================
# ETAPA 2 — CMLE
# ============================================================
log.info("\n[2/2] CMLE — seleção de família de cópula...")
print(f"\n{'Grupo':35s} {'N_part':>7s} {'Gauss':>8s} {'Gumbel':>8s} "
      f"{'Frank':>8s} {'Clayton':>8s} {'Melhor':>10s} {'ΔAIC':>8s}")
print("-" * 95)

copula_selecionada = {}
cmle_resultados    = {}

for i, grupo in enumerate(GRUPOS_QUAIDS):
    if not resultados_probit.get(grupo, {}).get('copula'):
        copula_selecionada[grupo] = None
        continue

    eta_i  = eta_mat[mask_ok, i]
    w_i    = base.loc[mask_ok, f"w_{grupo}"].values
    mask_p = w_i > 0
    n      = mask_p.sum()
    u      = u_from_eta(eta_i[mask_p])
    v      = v_from_w(w_i[mask_p])

    aics = {}
    for familia, ll_func in LL_FUNCS.items():
        lo, hi = BOUNDS[familia]
        res = minimize_scalar(lambda t, f=ll_func: -f(t, u, v),
                              bounds=(lo, hi), method='bounded')
        verifica_bound(res.x, familia, grupo)
        aics[familia] = {'aic': float(2*res.fun + 2), 'theta': float(res.x)}

    melhor = min(aics, key=lambda k: aics[k]['aic'])
    delta  = aics[melhor]['aic'] - aics['Gaussiana']['aic']

    if abs(delta) < 2.0 and melhor != 'Gaussiana':
        log.warning("CMLE %s: ΔAIC=%.1f < 2 — diferença marginal, "
                    "Gaussiana igualmente válida", grupo, delta)

    copula_selecionada[grupo] = melhor
    cmle_resultados[grupo]    = {
        'melhor': melhor, 'n_part': int(n),
        'theta':  float(aics[melhor]['theta']),
        'aics':   {k: float(v['aic'])   for k, v in aics.items()},
        'thetas': {k: float(v['theta']) for k, v in aics.items()},
        'delta_aic_vs_gaussiana': float(delta),
    }

    print(f"{grupo:35s} {n:7,d} "
          f"{aics['Gaussiana']['aic']:8.1f} {aics['Gumbel']['aic']:8.1f} "
          f"{aics['Frank']['aic']:8.1f} {aics['Clayton']['aic']:8.1f} "
          f"{melhor:>10s} {delta:8.1f}")

# Resumo
print(f"\n=== RESUMO FINAL ===")
for grupo, copula in copula_selecionada.items():
    if copula is None:
        print(f"  {grupo:35s}: sem correção (zeros < {LIMIAR_ZEROS*100:.0f}%)")
    else:
        th = cmle_resultados[grupo]['theta']
        da = cmle_resultados[grupo]['delta_aic_vs_gaussiana']
        print(f"  {grupo:35s}: {copula:10s} θ={th:.4f}  ΔAIC={da:.1f}")

# ============================================================
# SALVA RESULTADOS
# ============================================================
log.info("\nSalvando resultados...")

# 1. probit_copula_resultados.parquet — eta_mat + metadados
metadados = {
    'copula_selecionada': copula_selecionada,
    'cmle_resultados':    cmle_resultados,
    'resultados_probit':  {g: {k: v for k, v in r.items() if k != 'params'}
                           for g, r in resultados_probit.items()},
    'grupos_copula':      grupos_copula,
    'grupos_sem_copula':  grupos_sem_copula,
    'n_total':            int(len(base)),
    'n_ok':               int(mask_ok.sum()),
    'probit_vars':        PROBIT_VARS,
}
eta_df = pd.DataFrame(
    eta_mat[mask_ok],
    columns=[f"eta_{g}" for g in GRUPOS_QUAIDS],
    index=base[mask_ok].index
)
eta_df['_metadados_json'] = json.dumps(metadados, default=str)
eta_df.to_parquet(OUT_DIR / 'probit_copula_resultados.parquet')
log.info("Salvo: probit_copula_resultados.parquet (%d UCs × %d cols)",
         len(eta_df), eta_df.shape[1])

# 2. probit_cov_mats.npz — matrizes de covariância dos Probits
# Chaves: nomes dos grupos com cópula (ponto substituído por underline)
cov_dict = {
    g.replace('.', '_'): mat
    for g, mat in cov_mats.items()
}
np.savez(OUT_DIR / 'probit_cov_mats.npz', **cov_dict)
log.info("Salvo: probit_cov_mats.npz (%d matrizes)", len(cov_dict))
for g, mat in cov_mats.items():
    log.info("  %s: shape=%s  (K_probit=%d)", g, mat.shape, mat.shape[0])

# 3. probit_xb.parquet — índice latente x'π̂ por UC e grupo
xb_df = pd.DataFrame(
    xb_mat,
    columns=[f"xb_{g}" for g in GRUPOS_QUAIDS],
    index=base[mask_ok].index
)
xb_df.to_parquet(OUT_DIR / 'probit_xb.parquet')
log.info("Salvo: probit_xb.parquet (%d UCs × %d cols)",
         len(xb_df), xb_df.shape[1])

# 4. copula_metadados.json — legibilidade humana
with open(OUT_DIR / 'copula_metadados.json', 'w') as f:
    json.dump(metadados, f, indent=2, default=str)
log.info("Salvo: copula_metadados.json")

log.info("\nResumo das matrizes de covariância salvas:")
log.info("  Arquivo: probit_cov_mats.npz")
log.info("  Leitura: data = np.load('probit_cov_mats.npz')")
log.info("           V1_cereais = data['01_Cereais']  # shape (13,13)")
log.info("  Uso no Murphy-Topel (04_quaids_estimacao.py):")
log.info("    ∂η/∂xb = -η·(xb + η)")
log.info("    C_i = ρ̂_i · ΣX · diag(∂η/∂xb) · X_probit'")
log.info("    V_MT = V2 + V2 · (C · V1 · C') · V2")
log.info("\nPróximo passo: 04_quaids_estimacao.py")
