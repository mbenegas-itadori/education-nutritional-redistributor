"""
01b_base_analitica_v2.py
========================
Constrói base_analitica_v2.parquet e base_quaids_v2.parquet a partir de:
  - consumo_quaids_v2.parquet   (gastos anualizados por grupo)
  - socioeconomico.parquet       (variáveis socioeconômicas)
  - CADERNETA_COLETIVA.txt       (preços implícitos)
  - DESPESA_INDIVIDUAL.txt       (preços de tabaco)

Etapas:
  1. Merge consumo + socioeconomico
  2. Variáveis derivadas (UF, REGIAO, LOG_WELFARE, dummies)
  3. Tratamento de renda zero (não-resposta ao módulo)
  4. Imputação de preços (Deaton + hierárquica)
  5. Índice de Stone e centralização de log-despesa
  6. Salva base_analitica_v2 (58.039 UCs) e base_quaids_v2 (54.208 UCs)

Decisões metodológicas:
  - Renda zero (10,2%): não-resposta ao módulo de rendimentos
    LOG_WELFARE = LOG_RENDA_PC se renda>0, LOG_X_PC se renda=0
    D_NRESP_RENDA = 1 para não-respondentes
  - Preços: valor bruto V8000/QTD_FINAL na unidade de venda
    Correção de qualidade Deaton para Carnes, Laticínios, Frutas
    Imputação hierárquica: PSU → UF×Situação → Região
  - Tabaco: preço imputado da DESPESA_INDIVIDUAL (sem quantidade)
  - Amostra QUAIDS: exclui UCs com X_TOTAL=0 (3.831 UCs)

Inputs:
    POF/output/consumo_quaids_v2.parquet
    POF/output/socioeconomico.parquet
    POF/output/classificacao_grupos_v2.parquet
    POF/dados/CADERNETA_COLETIVA.txt
    POF/dados/DESPESA_INDIVIDUAL.txt
    POF/documentacao/Cadastro de Produtos.xls
    POF/documentacao/Cadastro de Unidades de Medida.xls

Outputs:
    POF/output/base_analitica_v2.parquet   (58.039 × 80+ vars)
    POF/output/base_quaids_v2.parquet      (54.208 × 83 vars)

Execução:
    !python 01b_base_analitica_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('/content/drive/MyDrive/POF/dados')
DOC_DIR  = Path('/content/drive/MyDrive/POF/documentacao')
OUT_DIR  = Path('/content/drive/MyDrive/POF/output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRUPOS_QUAIDS = [
    '01.Cereais','02.Farinhas_Massas','03.Tuberculos',
    '04.Acucares_Industrializados','05.Leguminosas_Oleaginosas',
    '06.Frutas','07.Legumes_Verduras','08.Carnes','09.Laticinios',
    '10.Oleos_Gorduras','11.Bebidas_NA','12.Alimentacao_Fora',
    '13.Alcool','14.Tabaco'
]
GRUPOS_Q = {'08.Carnes','09.Laticinios','06.Frutas'}   # Deaton

def widths_to_colspecs(widths):
    specs, start = [], 0
    for w in widths:
        specs.append((start, start+w))
        start += w
    return specs

def monta_id_uc(df):
    return (df['COD_UPA'].astype('Int64').astype(str) + '_' +
            df['NUM_DOM'].astype('Int64').astype(str) + '_' +
            df['NUM_UC'].astype('Int64').astype(str))

def uf_para_regiao(uf):
    if uf in [11,12,13,14,15,16,17]: return 1
    if uf in [21,22,23,24,25,26,27,28,29]: return 2
    if uf in [31,32,33,35]: return 3
    if uf in [41,42,43]: return 4
    if uf in [50,51,52,53]: return 5
    return np.nan

# ============================================================
# 1. MERGE CONSUMO + SOCIOECONOMICO
# ============================================================
print("[1/6] Merge consumo + socioeconomico...")
consumo = pd.read_parquet(OUT_DIR/'consumo_quaids_v2.parquet')
socio   = pd.read_parquet(OUT_DIR/'socioeconomico.parquet')
base    = consumo.merge(
    socio.rename(columns={'ID_UC':'ID_FAMILIA'}),
    on='ID_FAMILIA', how='left')
print(f"   {base.shape[0]:,} UCs × {base.shape[1]} variáveis")

# ============================================================
# 2. VARIÁVEIS DERIVADAS
# ============================================================
print("[2/6] Criando variáveis derivadas...")
base['UF']     = base['ESTRATO_POF'].astype(str).str.zfill(4).str[:2].astype(int)
base['REGIAO'] = base['UF'].apply(uf_para_regiao)
base['D_URBANO'] = (base['TIPO_SITUACAO_REG'] == 1).astype(int)

for r in [2,3,4,5]:
    base[f'D_REG_{r}'] = (base['REGIAO']==r).astype(int)

base['D_EDUC_MISSING']  = base['EDUC_REF'].isna().astype(int)
base['EDUC_REF']        = base['EDUC_REF'].fillna(0).astype(int)
base['D_EDUC_FUND']     = (base['EDUC_REF']==2).astype(int)
base['D_EDUC_MEDIO']    = (base['EDUC_REF']==3).astype(int)
base['D_EDUC_SUPERIOR'] = (base['EDUC_REF']==4).astype(int)

base['LOG_X_PC'] = np.log(
    (base['X_TOTAL']/base['N_MORADORES']).replace(0, np.nan))

# ============================================================
# 3. TRATAMENTO DE RENDA ZERO
# ============================================================
print("[3/6] Tratamento de renda zero (não-resposta)...")
base['D_NRESP_RENDA'] = (base['RENDA_TOTAL_ANUAL']==0).astype(int)
base['LOG_WELFARE']   = base['LOG_RENDA_PC'].copy()
mask_nr = base['RENDA_TOTAL_ANUAL'] == 0
base.loc[mask_nr,'LOG_WELFARE'] = base.loc[mask_nr,'LOG_X_PC']
print(f"   Renda positiva: {(~mask_nr).sum():,} | "
      f"Não-respondentes: {mask_nr.sum():,}")

# ============================================================
# 4. IMPUTAÇÃO DE PREÇOS
# ============================================================
print("[4/6] Imputando preços...")

# --- Lê CADERNETA para preços ---
classif   = pd.read_parquet(OUT_DIR/'classificacao_grupos_v2.parquet')
grupo_map = dict(zip(classif['V9001'], classif['GRUPO_FINAL']))
unid      = pd.read_excel(DOC_DIR/'Cadastro de Unidades de Medida.xls')
unid.columns = ['CODIGO','UNIDADE']

w_cad = [2,4,1,9,2,1,2,3,7,2,10,12,10,1,2,14,14,10,9,4,5,9,5]
c_cad = ['UF','ESTRATO_POF','TIPO_SITUACAO_REG','COD_UPA','NUM_DOM',
         'NUM_UC','QUADRO','SEQ','V9001','V9002','V8000','DEFLATOR',
         'V8000_DEFLA','COD_IMPUT_VALOR','FATOR_ANUALIZACAO','PESO',
         'PESO_FINAL','RENDA_TOTAL','V9005','V9007','V9009',
         'QTD_FINAL','V9004']

cad = pd.read_fwf(DATA_DIR/'CADERNETA_COLETIVA.txt',
                  colspecs=widths_to_colspecs(w_cad),
                  names=c_cad, encoding='latin-1',
                  header=None, dtype=str)
for col in ['V9001','V8000','V8000_DEFLA','V9007','QTD_FINAL',
            'COD_UPA','NUM_DOM','NUM_UC','ESTRATO_POF',
            'TIPO_SITUACAO_REG']:
    cad[col] = pd.to_numeric(cad[col], errors='coerce')

cad['ID_UC']    = monta_id_uc(cad)
cad['GRUPO']    = cad['V9001'].map(grupo_map)
cad['UF']       = cad['ESTRATO_POF'].astype(str).str.zfill(4).str[:2].astype(int)
cad['D_URBANO'] = (cad['TIPO_SITUACAO_REG']==1).astype(int)
cad['REGIAO']   = cad['UF'].apply(uf_para_regiao)
cad['COD_UPA_int'] = cad['COD_UPA'].astype('Int64')

# Preço bruto com trimming IQR
mask_val = (cad['V8000'].gt(0) & cad['V8000'].lt(9999999) &
            cad['QTD_FINAL'].gt(0) & cad['GRUPO'].notna())
cad_v = cad[mask_val].copy()
cad_v['PRECO_BRUTO'] = cad_v['V8000'] / cad_v['QTD_FINAL']

masks_trim = []
for g in cad_v['GRUPO'].unique():
    idx = cad_v['GRUPO']==g
    vals = cad_v.loc[idx,'PRECO_BRUTO']
    q1,q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3-q1
    masks_trim.append(idx & cad_v['PRECO_BRUTO'].between(
        q1-1.5*iqr, q3+1.5*iqr))
mask_trim = pd.concat(masks_trim).groupby(level=0).any()
cad_trim = cad_v[mask_trim].copy()
cad_trim['LOG_PRECO'] = np.log(cad_trim['PRECO_BRUTO'])
print(f"   CADERNETA válida: {len(cad_trim):,} registros após trimming")

# --- Correção de Deaton para Carnes, Laticínios, Frutas ---
regressores = ['LOG_WELFARE','LOG_X_TOTAL','N_MORADORES','D_URBANO',
               'D_EDUC_FUND','D_EDUC_MEDIO','D_EDUC_SUPERIOR',
               'SEXO_REF','D_NRESP_RENDA',
               'D_REG_2','D_REG_3','D_REG_4','D_REG_5']
base['LOG_X_TOTAL'] = np.log(base['X_TOTAL'].replace(0,np.nan))

uc_preco_bruto = (cad_trim
    .groupby(['ID_UC','GRUPO'])['LOG_PRECO']
    .median().reset_index()
    .rename(columns={'LOG_PRECO':'LOG_P_BRUTO'}))
uc_preco_corr = uc_preco_bruto.copy()
uc_preco_corr['LOG_P_CORR'] = uc_preco_corr['LOG_P_BRUTO']

import statsmodels.api as sm
for grupo in GRUPOS_Q:
    uc_g = (uc_preco_bruto[uc_preco_bruto['GRUPO']==grupo]
            .merge(base[['ID_FAMILIA']+regressores],
                   left_on='ID_UC',right_on='ID_FAMILIA',how='inner'))
    mask = uc_g[regressores+['LOG_P_BRUTO']].notna().all(axis=1)
    uc_g = uc_g[mask].copy()
    y = uc_g['LOG_P_BRUTO'].values
    X = np.column_stack([np.ones(len(uc_g)),
                         uc_g[regressores].values])
    coef,_,_,_ = np.linalg.lstsq(X,y,rcond=None)
    residuos = y - X@coef
    ln_p_corr = coef[0] + residuos
    r2 = 1 - np.var(residuos)/np.var(y)
    corr_map = dict(zip(uc_g['ID_UC'], ln_p_corr))
    idx = ((uc_preco_corr['GRUPO']==grupo) &
            uc_preco_corr['ID_UC'].isin(corr_map))
    uc_preco_corr.loc[idx,'LOG_P_CORR'] = (
        uc_preco_corr.loc[idx,'ID_UC'].map(corr_map))
    print(f"   Deaton {grupo}: N={len(uc_g):,}  R²={r2:.4f}")

# --- Imputação hierárquica ---
cad_trim2 = cad_trim.merge(
    uc_preco_corr[['ID_UC','GRUPO','LOG_P_CORR']],
    on=['ID_UC','GRUPO'], how='left')
mask_sem = cad_trim2['LOG_P_CORR'].isna()
cad_trim2.loc[mask_sem,'LOG_P_CORR'] = cad_trim2.loc[mask_sem,'LOG_PRECO']

psu_p  = (cad_trim2.groupby(['COD_UPA_int','GRUPO'])['LOG_P_CORR']
          .median().reset_index().rename(columns={'LOG_P_CORR':'LOG_P_PSU'}))
ufst_p = (cad_trim2.groupby(['UF','D_URBANO','GRUPO'])['LOG_P_CORR']
          .median().reset_index().rename(columns={'LOG_P_CORR':'LOG_P_UFST'}))
reg_p  = (cad_trim2.groupby(['REGIAO','GRUPO'])['LOG_P_CORR']
          .median().reset_index().rename(columns={'LOG_P_CORR':'LOG_P_REG'}))

base['COD_UPA_int'] = base['ID_FAMILIA'].str.split('_').str[0].astype('Int64')

for grupo in GRUPOS_QUAIDS:
    col = f"p_{grupo}"
    base[col] = np.nan
    uc_g = (uc_preco_corr[uc_preco_corr['GRUPO']==grupo]
            .set_index('ID_UC')['LOG_P_CORR'].astype(float))
    base[col] = base['ID_FAMILIA'].map(uc_g)
    m = base[col].isna()
    base.loc[m,col] = base.loc[m,'COD_UPA_int'].map(
        psu_p[psu_p['GRUPO']==grupo].set_index('COD_UPA_int')['LOG_P_PSU'])
    ufst_d = (ufst_p[ufst_p['GRUPO']==grupo]
              .set_index(['UF','D_URBANO'])['LOG_P_UFST'].to_dict())
    m = base[col].isna()
    base.loc[m,col] = base.loc[m,['UF','D_URBANO']].apply(
        lambda r: ufst_d.get((int(r['UF']),int(r['D_URBANO'])),np.nan),
        axis=1).astype(float).values
    m = base[col].isna()
    base.loc[m,col] = base.loc[m,'REGIAO'].map(
        reg_p[reg_p['GRUPO']==grupo].set_index('REGIAO')['LOG_P_REG'])
    base[col] = np.exp(base[col].astype(float))

# --- Tabaco: preço da DESPESA_INDIVIDUAL ---
w_desp = [2,4,1,9,2,1,2,2,2,7,2,10,2,2,1,1,1,12,10,1,2,14,14,10,5]
c_desp = ['UF','ESTRATO_POF','TIPO_SITUACAO_REG','COD_UPA','NUM_DOM',
          'NUM_UC','COD_INFORMANTE','QUADRO','SEQ','V9001','V9002',
          'V8000','V9010','V9011','V9012','V4104','V4105','DEFLATOR',
          'V8000_DEFLA','COD_IMPUT_VALOR','FATOR_ANUALIZACAO','PESO',
          'PESO_FINAL','RENDA_TOTAL','V9004']

DOC_DIR2 = Path('/content/drive/MyDrive/POF/documentacao')
cad_prod = pd.read_excel(DOC_DIR2/'Cadastro de Produtos.xls', header=0)
cad_prod.columns = ['QUADRO','COD_PRODUTO','DESCRICAO']
cad_prod['QUADRO'] = pd.to_numeric(cad_prod['QUADRO'],
                                    errors='coerce').astype('Int64')
cad_prod = cad_prod.dropna(subset=['QUADRO'])
tab_v7 = {int(str(r['COD_PRODUTO']).strip())
          for _, r in cad_prod[cad_prod['QUADRO']==21].iterrows()}

desp = pd.read_fwf(DATA_DIR/'DESPESA_INDIVIDUAL.txt',
                   colspecs=widths_to_colspecs(w_desp),
                   names=c_desp, encoding='latin-1',
                   header=None, dtype=str)
for col in ['COD_UPA','NUM_DOM','NUM_UC','V9001','V8000',
            'ESTRATO_POF','TIPO_SITUACAO_REG']:
    desp[col] = pd.to_numeric(desp[col], errors='coerce')

desp['ID_UC']    = monta_id_uc(desp)
desp['UF']       = desp['ESTRATO_POF'].astype(str).str.zfill(4).str[:2].astype(int)
desp['D_URBANO'] = (desp['TIPO_SITUACAO_REG']==1).astype(int)
desp['REGIAO']   = desp['UF'].apply(uf_para_regiao)
desp['COD_UPA_int'] = desp['COD_UPA'].astype('Int64')

desp_tab = desp[desp['V9001'].isin(tab_v7) &
                desp['V8000'].between(0.01,9999)].copy()

for nivel, key_col, key_ref in [
    ('UC', 'ID_UC', 'ID_FAMILIA'),
    ('PSU', 'COD_UPA_int', 'COD_UPA_int'),
]:
    med = (desp_tab.groupby(key_col)['V8000'].median()
           .apply(np.log).to_dict())
    m = base['p_14.Tabaco'].isna()
    base.loc[m,'p_14.Tabaco'] = base.loc[m,key_ref].map(med)

ufst_tab = (desp_tab.groupby(['UF','D_URBANO'])['V8000'].median()
            .apply(np.log).to_dict())
m = base['p_14.Tabaco'].isna()
base.loc[m,'p_14.Tabaco'] = base.loc[m,['UF','D_URBANO']].apply(
    lambda r: ufst_tab.get((int(r['UF']),int(r['D_URBANO'])),np.nan),
    axis=1).astype(float).values
reg_tab = desp_tab.groupby('REGIAO')['V8000'].median().apply(np.log).to_dict()
m = base['p_14.Tabaco'].isna()
base.loc[m,'p_14.Tabaco'] = base.loc[m,'REGIAO'].map(reg_tab)
base['p_14.Tabaco'] = np.exp(base['p_14.Tabaco'].astype(float))

p_miss = sum(base[f"p_{g}"].isna().sum() for g in GRUPOS_QUAIDS)
print(f"   Missing em preços após imputação: {p_miss:,}")

# ============================================================
# 5. ÍNDICE DE STONE E CENTRALIZAÇÃO
# ============================================================
print("[5/6] Calculando índice de Stone e centralizando...")
base_q = base[base['X_TOTAL']>0].copy()
w_cols = [f"w_{g}" for g in GRUPOS_QUAIDS]
w_bar  = base_q[w_cols].mean().values
p_cols = [f"p_{g}" for g in GRUPOS_QUAIDS]
ln_p   = np.log(base_q[p_cols].values)
base_q['LN_P_STONE'] = ln_p @ w_bar
base_q['LOG_X_TOTAL'] = np.log(base_q['X_TOTAL'])
lnm_bruto = base_q['LOG_X_TOTAL'] - base_q['LN_P_STONE']
base_q['LN_M_C'] = lnm_bruto - lnm_bruto.mean()
print(f"   Média LN_M_C (deve ser ≈0): {base_q['LN_M_C'].mean():.6f}")
print(f"   Std LN_M_C: {base_q['LN_M_C'].std():.4f}")

# ============================================================
# 6. SALVA
# ============================================================
print("[6/6] Salvando...")
base.to_parquet(OUT_DIR/'base_analitica_v2.parquet', index=False)
base_q.to_parquet(OUT_DIR/'base_quaids_v2.parquet', index=False)
print(f"\n[OK] base_analitica_v2.parquet: {base.shape[0]:,} × {base.shape[1]}")
print(f"[OK] base_quaids_v2.parquet:    {base_q.shape[0]:,} × {base_q.shape[1]}")
print(f"\nPróximo passo: 03_probit_copula_v3.py")
