"""
04_quaids_estimacao.py
======================
Estimação do sistema QUAIDS via FGLS iterado (Zellner SUR).
Especificações S1 (base) e S4 (interação escolaridade × renda).
Inclui correção de Murphy-Topel para regressores gerados (η̂).

Inputs:
    POF/output/base_quaids_v2.parquet
    POF/output/probit_copula_resultados.parquet
    POF/output/probit_cov_mats.npz
    POF/output/probit_xb.parquet

Outputs:
    POF/output/quaids_s1_resultados.parquet   — parâmetros S1
    POF/output/quaids_s4_resultados.parquet   — parâmetros S4
    POF/output/quaids_elasticidades.parquet   — elasticidades S1 e S4
    POF/output/quaids_metadados.json

Execução:
    !python 04_quaids_estimacao.py

Referências:
    Banks, Blundell & Lewbel (1997) — QUAIDS
    Murphy & Topel (1985) — correção de regressores gerados
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURAÇÃO
# ============================================================
OUT_DIR = Path('/content/drive/MyDrive/POF/output')

GRUPOS_QUAIDS = [
    '01.Cereais', '02.Farinhas_Massas', '03.Tuberculos',
    '04.Acucares_Industrializados', '05.Leguminosas_Oleaginosas',
    '06.Frutas', '07.Legumes_Verduras', '08.Carnes', '09.Laticinios',
    '10.Oleos_Gorduras', '11.Bebidas_NA', '12.Alimentacao_Fora',
    '13.Alcool', '14.Tabaco'
]
N_GRUPOS  = len(GRUPOS_QUAIDS)           # 14
IDX_DROP  = GRUPOS_QUAIDS.index('08.Carnes')  # equação dropada (Carnes)
N_EQ      = N_GRUPOS - 1                 # 13 equações estimadas
EQS       = [i for i in range(N_GRUPOS) if i != IDX_DROP]
IDX_EXCL  = N_GRUPOS - 1                # Tabaco: preço excluído (homogeneidade)

MAX_ITER  = 15
TOL       = 1e-7

# ============================================================
# CARREGA DADOS
# ============================================================
log.info("Carregando bases...")
base    = pd.read_parquet(OUT_DIR / 'base_quaids_v2.parquet')
probit  = pd.read_parquet(OUT_DIR / 'probit_copula_resultados.parquet')
cov_mats= np.load(OUT_DIR / 'probit_cov_mats.npz')
xb_df   = pd.read_parquet(OUT_DIR / 'probit_xb.parquet')

# Metadados de cópula
meta = json.loads(probit['_metadados_json'].iloc[0])
copula_selecionada = meta['copula_selecionada']
grupos_copula      = meta['grupos_copula']

log.info("  base_quaids_v2:  %d UCs × %d vars", *base.shape)
log.info("  probit_resultados: %d UCs × %d cols", *probit.shape)
log.info("  Grupos com cópula: %d", len(grupos_copula))

# Alinha índices
assert len(base) == len(probit), "Tamanhos incompatíveis"

# ============================================================
# VARIÁVEIS PRINCIPAIS
# ============================================================
N   = len(base)
w_cols = [f"w_{g}" for g in GRUPOS_QUAIDS]
p_cols = [f"p_{g}" for g in GRUPOS_QUAIDS]
x_cols = [f"x_{g}" for g in GRUPOS_QUAIDS]
eta_cols = [f"eta_{g}" for g in GRUPOS_QUAIDS]

W      = base[w_cols].values.astype(float)      # (N, 14) shares
ln_p   = np.log(base[p_cols].values.astype(float))  # (N, 14) log-preços
lnm_c  = base['LN_M_C'].values.astype(float)    # (N,) log-despesa centralizado
pesos  = base['PESO_FINAL'].values.astype(float)
pesos  = pesos / pesos.mean()                    # normaliza para média 1
w_bar  = W.mean(axis=0)                          # shares médios (pesos Stone)

# Resíduos generalizados do Probit (alinhados com base)
ETA = probit[eta_cols].values.astype(float)      # (N, 14)

# ============================================================
# b(p) = exp(Σ β_i ln p_i)
# ============================================================
def calc_bp(beta_b, ln_p):
    return np.exp(ln_p @ beta_b)

# ============================================================
# MONTA REGRESSORES X_i PARA EQUAÇÃO i
# Colunas: const + 13 ln_p livres (excluindo Tabaco=idx 13)
#          + lnm_c + lnm_c²/b(p)
#          + eta_i   (se grupo tem cópula)
# ============================================================
def monta_X(i_glob, beta_b, has_copula_i):
    bp = calc_bp(beta_b, ln_p)
    # 13 preços livres (exclui Tabaco)
    ln_p_livre = np.delete(ln_p, IDX_EXCL, axis=1)   # (N, 13)
    X = np.column_stack([
        np.ones(N),        # constante
        ln_p_livre,        # 13 preços
        lnm_c,             # ln(m/P*)
        lnm_c**2 / bp      # ln(m/P*)²/b(p)
    ])                     # (N, 16)
    if has_copula_i:
        X = np.column_stack([X, ETA[:, i_glob]])  # (N, 17)
    return X

# ============================================================
# FGLS ITERADO
# ============================================================
def estima_sur(beta_b, especificacao='S1', dummies_educ=None):
    """
    Uma rodada de FGLS para o sistema QUAIDS.

    especificacao: 'S1' ou 'S4'
    dummies_educ: para S4, array (N, 3) com D_EDUC_FUND, MEDIO, SUPERIOR
    """
    Xs  = []
    Ks  = []
    for j, i_glob in enumerate(EQS):
        grupo = GRUPOS_QUAIDS[i_glob]
        has_cop = copula_selecionada.get(grupo) is not None
        X_i = monta_X(i_glob, beta_b, has_cop)

        # S4: adiciona interações educ × lnm_c
        if especificacao == 'S4' and dummies_educ is not None:
            for k in range(3):   # fund, medio, superior
                X_i = np.column_stack([X_i, dummies_educ[:, k] * lnm_c])

        Xs.append(X_i)
        Ks.append(X_i.shape[1])

    K_tot = sum(Ks)
    offs  = np.cumsum([0] + Ks)

    # OLS equação por equação para estimar Σ
    Y     = W[:, EQS]
    resid = np.zeros((N, N_EQ))
    for j, X in enumerate(Xs):
        Xw = X * pesos[:, None]**0.5
        yw = Y[:, j] * pesos**0.5
        b  = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        resid[:, j] = Y[:, j] - X @ b

    Sigma     = (resid.T @ (resid * pesos[:, None])) / N
    Sigma_inv = np.linalg.inv(Sigma)

    # GLS sistema completo
    XtSiX = np.zeros((K_tot, K_tot))
    XtSiy = np.zeros(K_tot)
    for j in range(N_EQ):
        Xjw = Xs[j] * pesos[:, None]**0.5
        yjw = Y[:, j] * pesos**0.5
        for k in range(N_EQ):
            Xkw = Xs[k] * pesos[:, None]**0.5
            XtSiX[offs[j]:offs[j+1], offs[k]:offs[k+1]] += (
                Sigma_inv[j, k] * (Xjw.T @ Xkw))
        for k in range(N_EQ):
            Xkw = Xs[k] * pesos[:, None]**0.5
            XtSiy[offs[k]:offs[k+1]] += Sigma_inv[j, k] * (Xkw.T @ yjw)

    cond = np.linalg.cond(XtSiX)
    theta_hat = np.linalg.lstsq(XtSiX, XtSiy, rcond=None)[0]
    V_SUR = np.linalg.inv(XtSiX)

    return theta_hat, Sigma, Sigma_inv, XtSiX, V_SUR, Xs, Ks, offs, cond

# ============================================================
# POSIÇÃO DE β_i NOS REGRESSORES
# pos_beta = 1 (const) + 13 (preços) + 0 (index de lnm_c) = 14
# ============================================================
POS_BETA = 14   # posição de lnm_c em X_i (0-based)

def extrai_beta_b(theta_hat, Ks, offs, especificacao='S1'):
    """Extrai β_i de theta_hat e reconstrói beta_b para b(p)."""
    beta_b_new = np.zeros(N_GRUPOS)
    for j, i_glob in enumerate(EQS):
        pos = offs[j] + POS_BETA
        beta_b_new[i_glob] = theta_hat[pos]
    # Carnes por aditividade: Σ β_i = 0
    beta_b_new[IDX_DROP] = -sum(beta_b_new[i]
                                 for i in range(N_GRUPOS)
                                 if i != IDX_DROP)
    return beta_b_new

# ============================================================
# S1 — ESTIMAÇÃO ITERADA
# ============================================================
log.info("\n[1/2] Estimando S1 (QUAIDS base)...")
beta_b = w_bar.copy()   # inicialização

for it in range(MAX_ITER):
    theta_hat, Sigma, Sigma_inv, XtSiX, V_SUR, Xs, Ks, offs, cond = \
        estima_sur(beta_b, 'S1')
    beta_b_new = extrai_beta_b(theta_hat, Ks, offs)
    delta = np.max(np.abs(beta_b_new - beta_b))
    log.info("  iter %d: Δβ=%.2e  cond(XtSiX)=%.2e", it+1, delta, cond)
    beta_b = beta_b_new.copy()
    if delta < TOL:
        log.info("  Convergiu em %d iterações.", it+1)
        break

beta_b_S1 = beta_b.copy()
theta_S1  = theta_hat.copy()
V_SUR_S1  = V_SUR.copy()
Xs_S1     = Xs
Ks_S1     = Ks
offs_S1   = offs

log.info("  β_i estimados (S1):")
for i, g in enumerate(GRUPOS_QUAIDS):
    log.info("    %s: β=%.5f", g, beta_b_S1[i])

# ============================================================
# MURPHY-TOPEL — S1
# ============================================================
log.info("\nCalculando correção Murphy-Topel (S1)...")

# Para cada grupo com cópula, calcula C_i = ρ̂_i · A_i · B_i
# onde A_i = Σ_h (∂²Q₂/∂θ₂∂η_ih) e B_i = ∂η_ih/∂π_i' = d_eta_dxb * x_ih'
# ∂η/∂xb = -(η² + xb)

xb_mat = xb_df.values.astype(float)   # (N, 14)

V_MT_S1 = V_SUR_S1.copy()
mt_ratios = {}

for j, i_glob in enumerate(EQS):
    grupo = GRUPOS_QUAIDS[i_glob]
    if copula_selecionada.get(grupo) is None:
        continue

    # Posição de ρ_i em theta_hat (último regressor da equação j)
    pos_rho = offs_S1[j] + Ks_S1[j] - 1
    rho_i   = theta_S1[pos_rho]

    # V1 = covariância do Probit para o grupo i
    chave = grupo.replace('.', '_')
    if chave not in cov_mats.files:
        log.warning("  Sem V1 para %s — pulando MT", grupo)
        continue
    V1_i = cov_mats[chave]   # (13, 13)
    K1   = V1_i.shape[0]

    # X_probit = X_prob (mesmo para todos os grupos)
    # eta_i e xb_i
    eta_i = ETA[:, i_glob]
    xb_i  = xb_mat[:, i_glob]

    # ∂η/∂xb (escalar por observação)
    d_eta_dxb = -(eta_i**2 + eta_i * xb_i)   # d(eta)/d(xb) = -eta*(eta + xb)

    # Gradiente ∂Q₂/∂θ₂ · ∂η/∂π: contribuição de cada obs à derivada cruzada
    # C_i = ρ_i · (1/N) · Σ_h [X_i_h * d_eta_dxb_h * X_probit_h']
    # Dimensão: (K_i, K1)
    X_i   = Xs_S1[j]                      # (N, K_i)
    X_prob_mat = np.column_stack([         # (N, K1) — regressores do Probit
        np.ones(N),
        base[['LOG_WELFARE','N_MORADORES','D_URBANO',
               'D_EDUC_FUND','D_EDUC_MEDIO','D_EDUC_SUPERIOR',
               'SEXO_REF','D_NRESP_RENDA',
               'D_REG_2','D_REG_3','D_REG_4','D_REG_5']].values
    ])

    # C_i = ρ_i · X_i' · diag(d_eta_dxb) · X_probit / N
    C_i = (rho_i / N) * (X_i.T @ (d_eta_dxb[:, None] * X_prob_mat))
    # C_i shape: (K_i, K1)

    # Contribuição ao termo de correção: V2 · C_i · V1_i · C_i' · V2
    # Monta matriz K_tot × K_tot com zeros fora do bloco j
    K_tot = V_SUR_S1.shape[0]
    C_full = np.zeros((K_tot, K1))
    C_full[offs_S1[j]:offs_S1[j+1], :] = C_i

    corr_j = V_SUR_S1 @ C_full @ V1_i @ C_full.T @ V_SUR_S1
    V_MT_S1 = V_MT_S1 + corr_j

    # Razão se_MT / se_SUR para ρ_i
    se_sur = np.sqrt(V_SUR_S1[pos_rho, pos_rho])
    se_mt  = np.sqrt(V_MT_S1[pos_rho, pos_rho])
    ratio  = se_mt / se_sur if se_sur > 0 else np.nan
    mt_ratios[grupo] = float(ratio)
    log.info("  %s: ρ=%.4f  se_SUR=%.5f  se_MT=%.5f  razão=%.4f",
             grupo, rho_i, se_sur, se_mt, ratio)

max_ratio = max(mt_ratios.values()) if mt_ratios else 1.0
log.info("  Razão máxima se_MT/se_SUR: %.4f", max_ratio)
if max_ratio < 1.01:
    log.info("  → Correção MT negligenciável (< 1%%). V_SUR é confiável.")

# ============================================================
# S4 — INTERAÇÃO ESCOLARIDADE × RENDA
# ============================================================
log.info("\n[2/2] Estimando S4 (interação escolaridade × renda)...")

dummies_educ = base[['D_EDUC_FUND','D_EDUC_MEDIO','D_EDUC_SUPERIOR']].values.astype(float)
beta_b_S4 = beta_b_S1.copy()   # inicializa com S1

for it in range(MAX_ITER):
    theta_hat_S4, Sigma_S4, Sigma_inv_S4, XtSiX_S4, V_SUR_S4, Xs_S4, Ks_S4, offs_S4, cond_S4 = \
        estima_sur(beta_b_S4, 'S4', dummies_educ)
    beta_b_new = extrai_beta_b(theta_hat_S4, Ks_S4, offs_S4, 'S4')
    delta = np.max(np.abs(beta_b_new - beta_b_S4))
    log.info("  iter %d: Δβ=%.2e  cond=%.2e", it+1, delta, cond_S4)
    beta_b_S4 = beta_b_new.copy()
    if delta < TOL:
        log.info("  Convergiu em %d iterações.", it+1)
        break

# ============================================================
# ELASTICIDADES S1
# ============================================================
log.info("\nCalculando elasticidades S1...")

# Shares e preços no ponto médio
w_star = np.zeros(N_GRUPOS)
for j, i_glob in enumerate(EQS):
    pos_a = offs_S1[j]
    w_star[i_glob] = theta_S1[pos_a]   # aproximação: intercepto ≈ w̄

# Usa shares médios como w_star
w_star = w_bar.copy()

# β_i do S1
beta_all_S1 = beta_b_S1.copy()

# Elasticidade-renda: μ_i = 1 + β_i / w_i*
mu_S1 = 1 + beta_all_S1 / w_star

# Matriz γ (14×14) — extrai do theta_S1
# γ_ij está nas posições 1:14 (13 preços livres) de cada equação
# Os 13 preços livres excluem Tabaco (idx=13)
# γ_i,Tabaco recuperado por homogeneidade: Σ_j γ_ij = 0

G = np.zeros((N_GRUPOS, N_GRUPOS))   # matriz γ completa

for j, i_glob in enumerate(EQS):
    # Posições 1:14 = coeficientes dos 13 preços livres
    gamma_livre = theta_S1[offs_S1[j]+1 : offs_S1[j]+14]   # 13 valores
    # Mapeia de volta para índices completos (exceto Tabaco)
    idx_livres = [k for k in range(N_GRUPOS) if k != IDX_EXCL]
    for jj, k in enumerate(idx_livres):
        G[i_glob, k] = gamma_livre[jj]
    # Homogeneidade: γ_i,Tabaco = -Σ_{j≠Tabaco} γ_ij
    G[i_glob, IDX_EXCL] = -np.sum(G[i_glob, :IDX_EXCL])

# Carnes por aditividade: γ_Carnes_j = -Σ_{i≠Carnes} γ_ij
for k in range(N_GRUPOS):
    G[IDX_DROP, k] = -sum(G[i, k] for i in range(N_GRUPOS) if i != IDX_DROP)

# Elasticidades Marshallianas e Hicksianas
# e_ij^M = (γ_ij - w̄_j β_i) / w_i* - δ_ij
# e_ij^H = e_ij^M + μ_i w̄_j / w_i*

E_M = np.zeros((N_GRUPOS, N_GRUPOS))
E_H = np.zeros((N_GRUPOS, N_GRUPOS))

for i in range(N_GRUPOS):
    for k in range(N_GRUPOS):
        d_ik = 1.0 if i == k else 0.0
        E_M[i, k] = (G[i, k] - w_bar[k]*beta_all_S1[i]) / w_star[i] - d_ik
        E_H[i, k] = E_M[i, k] + mu_S1[i] * w_bar[k]

# Verificação de homogeneidade
hom_resid = max(abs(E_M[i, :].sum() + mu_S1[i]) for i in range(N_GRUPOS))
log.info("  Homogeneidade — resíduo máx: %.2e", hom_resid)

# ============================================================
# TABELAS DE RESULTADOS
# ============================================================
log.info("\n=== ELASTICIDADES-RENDA S1 ===")
print(f"\n{'Grupo':35s} {'μ_i':>8s}  Tipo")
print("-" * 50)
for i, g in enumerate(GRUPOS_QUAIDS):
    tipo = 'Luxo' if mu_S1[i] > 1 else 'Necessidade'
    print(f"{g:35s} {mu_S1[i]:8.4f}  {tipo}")

log.info("\n=== ELASTICIDADES-PREÇO PRÓPRIAS S1 ===")
print(f"\n{'Grupo':35s} {'e_ii^M':>9s} {'e_ii^H':>9s}  Tipo")
print("-" * 60)
for i, g in enumerate(GRUPOS_QUAIDS):
    tipo = 'Elástico' if abs(E_M[i, i]) > 1 else 'Inelástico'
    print(f"{g:35s} {E_M[i,i]:9.4f} {E_H[i,i]:9.4f}  {tipo}")

# ============================================================
# SALVA RESULTADOS
# ============================================================
log.info("\nSalvando resultados...")

# S1 — parâmetros
s1_dict = {
    'beta_b':      beta_b_S1.tolist(),
    'theta_hat':   theta_S1.tolist(),
    'Ks':          Ks_S1,
    'offs':        offs_S1.tolist(),
    'mu':          mu_S1.tolist(),
    'gamma':       G.tolist(),
    'E_M':         E_M.tolist(),
    'E_H':         E_H.tolist(),
    'mt_ratios':   mt_ratios,
    'hom_resid':   float(hom_resid),
    'grupos':      GRUPOS_QUAIDS,
    'IDX_DROP':    IDX_DROP,
    'IDX_EXCL':    IDX_EXCL,
    'w_bar':       w_bar.tolist(),
    'w_star':      w_star.tolist(),
}

# S4 — parâmetros
s4_dict = {
    'beta_b':    beta_b_S4.tolist(),
    'theta_hat': theta_hat_S4.tolist(),
    'Ks':        Ks_S4,
    'offs':      offs_S4.tolist(),
}

# Salva parâmetros como JSON (matrizes numpy não vão em parquet facilmente)
with open(OUT_DIR / 'quaids_s1_resultados.json', 'w') as f:
    json.dump(s1_dict, f, indent=2)
with open(OUT_DIR / 'quaids_s4_resultados.json', 'w') as f:
    json.dump(s4_dict, f, indent=2)

# Salva matrizes de covariância
np.savez(OUT_DIR / 'quaids_cov_mats.npz',
         V_SUR_S1=V_SUR_S1,
         V_MT_S1=V_MT_S1,
         Sigma_S1=Sigma,
         V_SUR_S4=V_SUR_S4,
         Sigma_S4=Sigma_S4)

# Salva elasticidades em parquet para uso posterior
elast_df = pd.DataFrame({
    'Grupo':  GRUPOS_QUAIDS,
    'mu_S1':  mu_S1,
    'e_M_ii': [E_M[i,i] for i in range(N_GRUPOS)],
    'e_H_ii': [E_H[i,i] for i in range(N_GRUPOS)],
})
elast_df.to_parquet(OUT_DIR / 'quaids_elasticidades.parquet', index=False)

log.info("Salvo: quaids_s1_resultados.json")
log.info("Salvo: quaids_s4_resultados.json")
log.info("Salvo: quaids_cov_mats.npz")
log.info("Salvo: quaids_elasticidades.parquet")
log.info("\nPróximo passo: 05_elasticidades_nutricionais.py")
