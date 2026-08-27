"""
08_engel_grupos_FINAL.py
========================
Versão final para publicação.
Ajustes sobre v5:
  - FIG A: x_lo=8, x_hi=92 (corta mais caudas instáveis)
           N corrigido para 54,207
           Anotação de grupos com trecho crescente (rejeita AIDS linear)
  - FIG B: x_lo=8, x_hi=92
           alpha IC aumentado para 0.15 (mais visível)
           Legenda em posição otimizada por painel
           n_boot=200 mantido

Uso:
  exec(open('/content/drive/MyDrive/POF/script/08_engel_grupos_FINAL.py').read())
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
    'figure.dpi'       : 180,
    'pdf.fonttype'     : 42,   # TrueType embedado no PDF
    'ps.fonttype'      : 42,
    'axes.titlesize'   : 9,
    'axes.labelsize'   : 8,
    'xtick.labelsize'  : 7.5,
    'ytick.labelsize'  : 7.5,
})

COR_AGG  = '#1a1a1a'
COR_FILL = '#cccccc'
COR_EDUC = {'base':'#1f77b4','fund':'#ff7f0e','med':'#2ca02c','sup':'#d62728'}
LAB_EDUC = {'base':'None/Primary','fund':'Lower sec.','med':'Upper sec.','sup':'Tertiary'}

GRUPOS_12 = [
    '01.Cereais','02.Farinhas_Massas','03.Tuberculos',
    '04.Acucares_Industrializados','05.Leguminosas_Oleaginosas',
    '06.Frutas','07.Legumes_Verduras','08.Carnes','09.Laticinios',
    '10.Oleos_Gorduras','11.Bebidas_NA','12.Alimentacao_Fora',
]
LABEL_SHORT = {
    '01.Cereais'                  : 'Cereals',
    '02.Farinhas_Massas'          : 'Flours & Pasta',
    '03.Tuberculos'               : 'Roots & Tubers',
    '04.Acucares_Industrializados': 'Sugars & Processed',
    '05.Leguminosas_Oleaginosas'  : 'Legumes & Oilseeds',
    '06.Frutas'                   : 'Fruits',
    '07.Legumes_Verduras'         : 'Vegetables',
    '08.Carnes'                   : 'Meat & Fish',
    '09.Laticinios'               : 'Dairy',
    '10.Oleos_Gorduras'           : 'Oils & Fats',
    '11.Bebidas_NA'               : 'Non-alc. Beverages',
    '12.Alimentacao_Fora'         : 'Food Away from Home',
}

# Grupos com trecho crescente — rejeita linearidade do AIDS
# Anotados com † no título
GRUPOS_CRESCENTES = {
    '01.Cereais', '04.Acucares_Industrializados',
    '05.Leguminosas_Oleaginosas', '08.Carnes', '12.Alimentacao_Fora',
}

GRUPOS_B = ['02.Farinhas_Massas', '08.Carnes', '12.Alimentacao_Fora', '09.Laticinios']

# Posição da legenda por painel FIG B
LEG_LOC_B = {
    '02.Farinhas_Massas'  : 'upper right',
    '08.Carnes'           : 'upper left',
    '12.Alimentacao_Fora' : 'upper left',
    '09.Laticinios'       : 'upper right',
}

# ── Carrega dados ─────────────────────────────────────────────────
print("[1] Carregando base_quaids_v2.parquet...")
base    = pd.read_parquet(OUT / 'base_quaids_v2.parquet')
pesos   = base['PESO_FINAL'].values
pesos_n = pesos / pesos.mean()
lnm     = base['LN_M_C'].values
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

masks_educ = {
    'base': base['EDUC_REF'].values <= 1,
    'fund': base['EDUC_REF'].values == 2,
    'med' : base['EDUC_REF'].values == 3,
    'sup' : base['EDUC_REF'].values == 4,
}
print(f"    N={N:,} | "
      + " | ".join(f"{k}={v.sum():,}" for k, v in masks_educ.items()))

# ── Bin-then-LOESS com bootstrap por bin ──────────────────────────
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
    ax.axvline(lnm_med, color='#666', lw=0.8, ls=':', alpha=0.8)

# ── FIG A — 4×3 painel agregado ───────────────────────────────────
print("\n[2] Gerando FIG A (4x3)...")
t0  = time.time()
fig, axes = plt.subplots(4, 3, figsize=(13, 14))
fig.suptitle(
    f'Engel curves for food groups — budget share $w_i$ vs. $\\ln(m/P^*)$\n'
    f'LOESS on binned means, 90\\% bootstrap CI, POF 2017–18 ($N={N:,}$)\n'
    r'$\dagger$ Non-monotone shape rejects linear AIDS specification',
    fontsize=9.5, y=1.008)

for pos, g in enumerate(GRUPOS_12):
    t1  = time.time()
    ax  = axes.flatten()[pos]
    wg  = base[f'w_{g}'].values
    xg, yh, yl, yhi = bin_loess_boot(
        lnm, wg, w=pesos_n, frac=0.40,
        n_boot=200, x_lo=8, x_hi=92)
    ax.fill_between(xg, yl * 100, yhi * 100, color=COR_FILL, alpha=0.55, lw=0)
    ax.plot(xg, yh * 100, color=COR_AGG, lw=1.6)
    add_median(ax)
    titulo = LABEL_SHORT[g]
    if g in GRUPOS_CRESCENTES:
        titulo += r' $\dagger$'
    ax.set_title(titulo, pad=3)
    ax.set_xlabel('$\\ln(m/P^*)$ — centred', labelpad=2)
    ax.set_ylabel('Budget share (%)', labelpad=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    print(f"    [{pos+1:2d}/12] {LABEL_SHORT[g]:<24} {time.time()-t1:.1f}s")

fig.tight_layout(rect=[0, 0, 1, 1.008])
path_A = FIGS / 'engel_grupos_figA.pdf'
fig.savefig(path_A, bbox_inches='tight')
plt.close(fig)
print(f"    FIG A salva ({time.time()-t0:.0f}s total)")

# ── FIG B — 2×2 estratificado ─────────────────────────────────────
print("\n[3] Gerando FIG B (2x2 estratificado)...")
t0  = time.time()
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle(
    'Engel curves by education level — selected food groups\n'
    'LOESS on binned means, 90% bootstrap CI, POF 2017–18',
    fontsize=10, y=1.005)

for pos, g in enumerate(GRUPOS_B):
    ax = axes.flatten()[pos]
    for estrato, mask in masks_educ.items():
        n_est  = mask.sum()
        n_bins = max(40, min(70, n_est // 80))
        t1     = time.time()
        xg, yh, yl, yhi = bin_loess_boot(
            lnm[mask], base[f'w_{g}'].values[mask],
            w=pesos_n[mask], n_bins=n_bins,
            frac=0.60, n_boot=200,
            x_lo=8, x_hi=92)
        ax.fill_between(xg, yl * 100, yhi * 100,
                        color=COR_EDUC[estrato], alpha=0.15, lw=0)
        ax.plot(xg, yh * 100, color=COR_EDUC[estrato], lw=1.8,
                label=LAB_EDUC[estrato])
        print(f"    {LABEL_SHORT[g]:<24} {estrato:<5} "
              f"N={n_est:,} bins={n_bins} {time.time()-t1:.1f}s")
    add_median(ax)
    ax.set_title(LABEL_SHORT[g], pad=3)
    ax.set_xlabel('$\\ln(m/P^*)$ — centred', labelpad=2)
    ax.set_ylabel('Budget share (%)', labelpad=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.legend(fontsize=7.5, loc=LEG_LOC_B[g], framealpha=0.75)

fig.tight_layout(rect=[0, 0, 1, 1.005])
path_B = FIGS / 'engel_grupos_figB.pdf'
fig.savefig(path_B, bbox_inches='tight')
plt.close(fig)
print(f"    FIG B salva ({time.time()-t0:.0f}s total)")

print("\n[OK] 08_engel_grupos_FINAL.py concluido.")