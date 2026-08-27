"""
06_engel_nutricional_FINAL.py
==============================
Versão final para publicação.
Usa bin_loess_boot (igual ao 08) para IC estáveis e tempo razoável.
Principais mudanças vs. v3:
  - loess_pond substituída por bin_loess_boot (bins + bootstrap por bin)
  - x_lo=8, x_hi=92 (caudas mais estáveis)
  - LN_M_C no lugar de LOG_WELFARE (consistência com 08)
  - n_boot=200 (mesma qualidade do 08)
  - FIG A: 6 nutrientes, base vs terciário
  - FIG B: sódio e cálcio, 4 estratos
  - FIG C: 10 nutrientes agregados (apêndice)

Uso:
  exec(open('/content/drive/MyDrive/POF/script/06_engel_nutricional_FINAL.py').read())
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings, time
warnings.filterwarnings('ignore')

OUT  = Path('/content/drive/MyDrive/POF/output')
FIGS = OUT / 'figuras'
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 9,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.20,
    'grid.linestyle'   : '--',
    'figure.dpi'       : 150,
    'pdf.fonttype'     : 42,   # TrueType embedado no PDF
    'ps.fonttype'      : 42,
    'axes.titlesize'   : 10,
    'axes.labelsize'   : 8.5,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
})

COR = {
    'base': '#1f77b4',
    'fund': '#ff7f0e',
    'med' : '#2ca02c',
    'tert': '#d62728',
}
LAB = {
    'base': 'No schooling / Primary',
    'fund': 'Lower secondary',
    'med' : 'Upper secondary',
    'tert': 'Tertiary',
}

FRAC = {
    'energia_kcal'  : 0.40,
    'proteina_g'    : 0.40,
    'carboidrato_g' : 0.40,
    'lipideos_g'    : 0.40,
    'fibra_g'       : 0.40,
    'sodio_mg'      : 0.40,
    'ferro_mg'      : 0.40,
    'calcio_mg'     : 0.50,
    'zinco_mg'      : 0.40,
    'vitaminaC_mg'  : 0.50,
}

NUT_LABEL = {
    'energia_kcal'  : 'Energy (kcal)',
    'proteina_g'    : 'Protein (g)',
    'carboidrato_g' : 'Carbohydrate (g)',
    'lipideos_g'    : 'Lipids (g)',
    'fibra_g'       : 'Fibre (g)',
    'sodio_mg'      : 'Sodium (mg)',
    'ferro_mg'      : 'Iron (mg)',
    'calcio_mg'     : 'Calcium (mg)',
    'zinco_mg'      : 'Zinc (mg)',
    'vitaminaC_mg'  : 'Vitamin C (mg)',
}

# ── Carrega dados ─────────────────────────────────────────────────
print("[1] Carregando dados...")
base  = pd.read_parquet(OUT / 'base_quaids_v2.parquet')
omega = pd.read_parquet(OUT / 'matriz_omega.parquet')

GRUPOS = [
    '01.Cereais','02.Farinhas_Massas','03.Tuberculos',
    '04.Acucares_Industrializados','05.Leguminosas_Oleaginosas',
    '06.Frutas','07.Legumes_Verduras','08.Carnes','09.Laticinios',
    '10.Oleos_Gorduras','11.Bebidas_NA','12.Alimentacao_Fora',
    '13.Alcool','14.Tabaco'
]
NUTRIENTES  = omega.index.tolist()
Omega_mat   = omega.reindex(columns=GRUPOS).fillna(0).values  # (10 × 14)

pesos   = base['PESO_FINAL'].values
pesos_n = pesos / pesos.mean()
lnm     = base['LN_M_C'].values          # welfare centrado — consistente com 08
def weighted_median(values, weights):
    """
    Mediana ponderada: ordena os valores, acumula os pesos normalizados,
    e retorna o valor onde a soma acumulada cruza 50% do peso total.
    Substitui np.average() (media), que estava sendo usado indevidamente
    para a linha vertical rotulada "median" nos graficos.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum_w = np.cumsum(w_sorted)
    cutoff = w_sorted.sum() / 2.0
    idx = np.searchsorted(cum_w, cutoff)
    return v_sorted[min(idx, len(v_sorted) - 1)]

lnm_med = weighted_median(lnm, pesos_n)  # CORRIGIDO: era np.average (media, nao mediana)
N       = len(base)

W        = base[[f'w_{g}' for g in GRUPOS]].values   # (N × 14)
NUT_share = W @ Omega_mat.T                            # (N × 10)

masks = {
    'base': base['EDUC_REF'].values <= 1,
    'fund': base['EDUC_REF'].values == 2,
    'med' : base['EDUC_REF'].values == 3,
    'tert': base['EDUC_REF'].values == 4,
}
print(f"    N={N:,} | " + " | ".join(f"{k}={v.sum():,}" for k, v in masks.items()))

# ── Bin-then-LOESS (igual ao 08) ──────────────────────────────────
def bin_loess_boot(x, y, w=None, n_bins=80, frac=0.40,
                   n_boot=200, seed=0, plo=5, phi=95,
                   x_lo=8, x_hi=92):
    mask = np.isfinite(x) & np.isfinite(y)
    if w is not None: mask &= (w > 0)
    x, y = x[mask], y[mask]
    w    = (w[mask] if w is not None else np.ones(len(x)))
    w    = w / w.sum()

    xlo, xhi = np.percentile(x, x_lo), np.percentile(x, x_hi)
    edges    = np.linspace(xlo, xhi, n_bins + 1)
    centers  = 0.5 * (edges[:-1] + edges[1:])
    bin_idx  = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    bins_x, bins_y, bins_w = [], [], []
    for k in range(n_bins):
        m = bin_idx == k
        bins_x.append(x[m]); bins_y.append(y[m]); bins_w.append(w[m])

    def bin_means(bx, by, bw):
        cx = np.array([bx[k].mean() if len(bx[k]) > 0 else centers[k]
                       for k in range(n_bins)])
        cy = np.array([np.average(by[k], weights=bw[k])
                       if len(by[k]) > 0 else np.nan
                       for k in range(n_bins)])
        ok = np.isfinite(cy)
        return cx[ok], cy[ok]

    cx, cy = bin_means(bins_x, bins_y, bins_w)
    res    = lowess(cy, cx, frac=frac, it=3, return_sorted=True)
    x_grid = centers.copy()
    y_hat  = np.interp(x_grid, res[:, 0], res[:, 1])

    rng   = np.random.default_rng(seed)
    boots = np.full((n_boot, n_bins), np.nan)
    for b in range(n_boot):
        bx_b, by_b, bw_b = [], [], []
        for k in range(n_bins):
            n_k = len(bins_x[k])
            if n_k == 0:
                bx_b.append(np.array([])); by_b.append(np.array([]))
                bw_b.append(np.array([])); continue
            idx_k = rng.integers(0, n_k, n_k)
            bx_b.append(bins_x[k][idx_k]); by_b.append(bins_y[k][idx_k])
            w_k = bins_w[k][idx_k]
            bw_b.append(w_k / w_k.sum() if w_k.sum() > 0 else w_k)
        cxb, cyb = bin_means(bx_b, by_b, bw_b)
        if len(cxb) < 5: continue
        rb = lowess(cyb, cxb, frac=frac, it=1, return_sorted=True)
        boots[b] = np.interp(x_grid, rb[:, 0], rb[:, 1])

    lo = np.nanpercentile(boots, plo, axis=0)
    hi = np.nanpercentile(boots, phi, axis=0)
    return x_grid, y_hat, lo, hi

def add_median(ax):
    ylo, yhi = ax.get_ylim()
    ax.axvline(lnm_med, color='#555', lw=0.8, ls=':', alpha=0.7)
    ax.text(lnm_med + 0.05, ylo + (yhi - ylo) * 0.97,
            'median', fontsize=6.5, color='#555', va='top')

# ── FIG A — 6 nutrientes, base vs terciário ───────────────────────
print("\n[2] FIG A — base vs terciário...")
NUTS_A = ['sodio_mg', 'calcio_mg', 'vitaminaC_mg',
          'fibra_g',  'ferro_mg',  'zinco_mg']
t0 = time.time()
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle(
    f'Nutritional portfolio by income: no schooling vs. tertiary\n'
    r'$\hat\omega_n(\ln m)=\sum_j\Omega_{nj}\,\hat w_j(\ln m)$'
    '  —  LOESS on binned means, 90% bootstrap CI',
    fontsize=10, y=1.01)

for pos, nut in enumerate(NUTS_A):
    ax  = axes.flatten()[pos]
    idx = NUTRIENTES.index(nut)
    y   = NUT_share[:, idx]
    fr  = FRAC[nut]
    for strato in ['base', 'tert']:
        m = masks[strato]
        n_bins = max(50, min(80, m.sum() // 80))
        xg, yh, yl, yhi = bin_loess_boot(
            lnm[m], y[m], w=pesos_n[m],
            frac=fr, n_boot=200,
            x_lo=8, x_hi=92, n_bins=n_bins)
        ax.fill_between(xg, yl, yhi, alpha=0.18, color=COR[strato])
        ax.plot(xg, yh, color=COR[strato], lw=2,
                label=LAB[strato])
    add_median(ax)
    ax.set_title(NUT_LABEL[nut], fontsize=10, fontweight='bold')
    ax.set_xlabel(r'$\ln(m/P^*)$ — centred', fontsize=8.5)
    ax.set_ylabel('Nutritional share', fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.tick_params(labelsize=8)
    if pos == 0:
        ax.legend(fontsize=8, framealpha=0.85, loc='upper right')

plt.tight_layout()
plt.savefig(FIGS / 'figA_nutricional_FINAL.pdf', bbox_inches='tight')
plt.close()
print(f"    FIG A salva ({time.time()-t0:.0f}s)")

# ── FIG B — sódio e cálcio, 4 estratos ───────────────────────────
print("\n[3] FIG B — 4 estratos (Sódio e Cálcio)...")
NUTS_B = ['sodio_mg', 'calcio_mg']
t0 = time.time()
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(
    'Nutritional portfolio by income and schooling stratum\n'
    r'$\hat\omega_n(\ln m)$  —  LOESS on binned means, 90% bootstrap CI',
    fontsize=10, y=1.01)

for pos, nut in enumerate(NUTS_B):
    ax  = axes[pos]
    idx = NUTRIENTES.index(nut)
    y   = NUT_share[:, idx]
    fr  = FRAC[nut] + 0.05
    for strato in ['base', 'fund', 'med', 'tert']:
        m      = masks[strato]
        n_bins = max(40, min(65, m.sum() // 80))
        xg, yh, yl, yhi = bin_loess_boot(
            lnm[m], y[m], w=pesos_n[m],
            frac=fr, n_boot=200,
            x_lo=8, x_hi=92, n_bins=n_bins)
        ax.fill_between(xg, yl, yhi, alpha=0.15, color=COR[strato])
        ax.plot(xg, yh, color=COR[strato], lw=2,
                label=LAB[strato])
    add_median(ax)
    ax.set_title(NUT_LABEL[nut], fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$\ln(m/P^*)$ — centred', fontsize=9)
    ax.set_ylabel('Nutritional share', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8, framealpha=0.85,
              loc='upper right' if nut == 'sodio_mg' else 'lower right')

plt.tight_layout()
plt.savefig(FIGS / 'figB_nutricional_FINAL.pdf', bbox_inches='tight')
plt.close()
print(f"    FIG B salva ({time.time()-t0:.0f}s)")

# ── FIG C — 10 nutrientes agregados (apêndice) ────────────────────
print("\n[4] FIG C — apêndice (10 nutrientes)...")
t0 = time.time()
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle(
    'Online Appendix — Nutritional portfolio share by income level\n'
    r'$\hat\omega_n(\ln m)$  —  LOESS on binned means, 90% CI',
    fontsize=10, y=1.01)

for i, nut in enumerate(NUTRIENTES):
    ax = axes.flatten()[i]
    y  = NUT_share[:, i]
    xg, yh, yl, yhi = bin_loess_boot(
        lnm, y, w=pesos_n, frac=FRAC[nut],
        n_boot=200, x_lo=8, x_hi=92)
    ax.fill_between(xg, yl, yhi, alpha=0.22, color='#2c6fad')
    ax.plot(xg, yh, color='#2c6fad', lw=2)
    ax.axhline(np.average(y, weights=pesos_n), color='#888', lw=0.8, ls=':')
    add_median(ax)
    ax.set_title(NUT_LABEL[nut], fontsize=9)
    ax.set_xlabel(r'$\ln(m/P^*)$', fontsize=8)
    ax.set_ylabel('Nutrit. share', fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.tick_params(labelsize=7.5)

plt.tight_layout()
plt.savefig(FIGS / 'figC_nutricional_apendice_FINAL.pdf', bbox_inches='tight')
plt.close()
print(f"    FIG C salva ({time.time()-t0:.0f}s)")

print(f"\n[OK] 06_engel_nutricional_FINAL.py concluido.")
print(f"  figA_nutricional_FINAL.pdf  — texto principal (Sod/Cal/VitC/Fibra/Fe/Zn)")
print(f"  figB_nutricional_FINAL.pdf  — texto principal (Sod e Cal, 4 estratos)")
print(f"  figC_nutricional_apendice_FINAL.pdf — apendice online")