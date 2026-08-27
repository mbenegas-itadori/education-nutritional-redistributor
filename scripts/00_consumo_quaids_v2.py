"""
00_consumo_quaids_v2.py
=======================
Constrói consumo_quaids_v2.parquet a partir de:
  - CADERNETA_COLETIVA.txt   (compras domiciliares coletivas)
  - DESPESA_INDIVIDUAL.txt   (gastos individuais: tabaco, álcool, alim. fora)

Layout oficial: widths do script R do IBGE (POF 2017-18).
Valor anualizado = V8000 × FATOR_ANUALIZACAO.
Missing monetário: V8000 = 9999999.99 → usa V8000_DEFLA.

Quadros alimentares da DESPESA_INDIVIDUAL:
  21 → Tabaco
  24 → Alimentação Fora + Álcool (separados por palavras-chave)
  41 → Alimentação Fora (viagens)

Nota metodológica:
  Tabaco e álcool: preços imputados apenas da CADERNETA (sem quantidade
  na DESPESA_INDIVIDUAL). Limitação documentada para análise de P2.

Inputs:
    POF/dados/CADERNETA_COLETIVA.txt
    POF/dados/DESPESA_INDIVIDUAL.txt
    POF/dados/Cadastro de Produtos.xls
    POF/output/classificacao_grupos_v2.parquet
    POF/output/socioeconomico.parquet

Outputs:
    POF/output/consumo_quaids_v2.parquet   (58.039 UCs × 31 colunas)

Execução:
    !python 00_consumo_quaids_v2.py
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
    '13.Alcool','14.Tabaco','Numerario_Nao_Alimento'
]

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

# ============================================================
# CLASSIFICAÇÃO V2
# ============================================================
print("[1/4] Carregando classificação v2...")
classif   = pd.read_parquet(OUT_DIR/'classificacao_grupos_v2.parquet')
grupo_map = dict(zip(classif['V9001'], classif['GRUPO_FINAL']))
socio     = pd.read_parquet(OUT_DIR/'socioeconomico.parquet')
print(f"   {len(classif):,} produtos | {len(socio):,} UCs na amostra")

# ============================================================
# CADERNETA_COLETIVA
# ============================================================
print("[2/4] Lendo CADERNETA_COLETIVA...")
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
for col in ['V9001','V8000','V8000_DEFLA','FATOR_ANUALIZACAO',
            'COD_UPA','NUM_DOM','NUM_UC']:
    cad[col] = pd.to_numeric(cad[col], errors='coerce')

cad['ID_UC'] = monta_id_uc(cad)
cad['GRUPO'] = cad['V9001'].map(grupo_map)

# Trata missing e anualiza
mask_miss = cad['V8000'] >= 9999999
cad.loc[mask_miss, 'V8000'] = cad.loc[mask_miss, 'V8000_DEFLA']
cad = cad[cad['V8000'].between(0.01, 9999) & cad['GRUPO'].notna() &
          (cad['GRUPO'] != 'Numerario_Nao_Alimento')].copy()
cad['VALOR_ANUAL'] = cad['V8000'] * cad['FATOR_ANUALIZACAO']

print(f"   {len(cad):,} registros | {cad['ID_UC'].nunique():,} UCs")

# ============================================================
# DESPESA_INDIVIDUAL
# ============================================================
print("[3/4] Lendo DESPESA_INDIVIDUAL...")
w_desp = [2,4,1,9,2,1,2,2,2,7,2,10,2,2,1,1,1,12,10,1,2,14,14,10,5]
c_desp = ['UF','ESTRATO_POF','TIPO_SITUACAO_REG','COD_UPA','NUM_DOM',
          'NUM_UC','COD_INFORMANTE','QUADRO','SEQ','V9001','V9002',
          'V8000','V9010','V9011','V9012','V4104','V4105','DEFLATOR',
          'V8000_DEFLA','COD_IMPUT_VALOR','FATOR_ANUALIZACAO','PESO',
          'PESO_FINAL','RENDA_TOTAL','V9004']

desp = pd.read_fwf(DATA_DIR/'DESPESA_INDIVIDUAL.txt',
                   colspecs=widths_to_colspecs(w_desp),
                   names=c_desp, encoding='latin-1',
                   header=None, dtype=str)
for col in ['V9001','V8000','V8000_DEFLA','FATOR_ANUALIZACAO',
            'COD_UPA','NUM_DOM','NUM_UC']:
    desp[col] = pd.to_numeric(desp[col], errors='coerce')

desp['ID_UC'] = monta_id_uc(desp)

# Mapa de quadros alimentares
ALCOOL_PALAVRAS = {
    'CERVEJA','VINHO','CACHACA','CACHAÇA','CHOPE','CAIPIRINHA',
    'VODKA','WHISKY','DOSE','DRINQUE','DRINK','APERITIVO',
    'AGUARDENTE','PINGA','GIN','RUM','LICOR','CONHAQUE',
    'ESPUMANTE','CHAMPAGNE','SIDRA','SAKE','ALCOOLICA','ALCOOLICO'
}

cad_prod = pd.read_excel(DOC_DIR/'Cadastro de Produtos.xls', header=0)
cad_prod.columns = ['QUADRO','COD_PRODUTO','DESCRICAO']
cad_prod['QUADRO'] = pd.to_numeric(cad_prod['QUADRO'],
                                    errors='coerce').astype('Int64')
cad_prod = cad_prod.dropna(subset=['QUADRO'])

grupo_map_desp = {}
for _, row in cad_prod[cad_prod['QUADRO'].isin([21,24,41])].iterrows():
    q    = int(row['QUADRO'])
    v7   = int(str(row['COD_PRODUTO']).strip())
    desc = str(row['DESCRICAO']).upper()
    if q == 21:
        g = '14.Tabaco'
    elif q == 24:
        g = '13.Alcool' if any(w in desc for w in ALCOOL_PALAVRAS) \
            else '12.Alimentacao_Fora'
    elif q == 41:
        g = '12.Alimentacao_Fora'
    grupo_map_desp[v7] = g

desp['GRUPO'] = desp['V9001'].map(grupo_map_desp)
mask_miss_d = desp['V8000'] >= 9999999
desp.loc[mask_miss_d, 'V8000'] = desp.loc[mask_miss_d, 'V8000_DEFLA']
desp_alim = desp[desp['GRUPO'].notna() &
                 desp['V8000'].between(0.01, 99999)].copy()
desp_alim['VALOR_ANUAL'] = desp_alim['V8000'] * desp_alim['FATOR_ANUALIZACAO']

print(f"   {len(desp_alim):,} registros alimentares | "
      f"{desp_alim['ID_UC'].nunique():,} UCs")

# ============================================================
# COMBINA E AGREGA
# ============================================================
print("[4/4] Agregando e construindo base...")
todas = pd.concat([
    cad[['ID_UC','GRUPO','VALOR_ANUAL']],
    desp_alim[['ID_UC','GRUPO','VALOR_ANUAL']]
], ignore_index=True)

pivot = (todas
    .groupby(['ID_UC','GRUPO'])['VALOR_ANUAL']
    .sum()
    .unstack(fill_value=0)
    .reset_index())

for g in GRUPOS_QUAIDS:
    if g not in pivot.columns:
        pivot[g] = 0.0

# Merge com âncora de UCs da amostra
base = (socio[['ID_UC']]
    .rename(columns={'ID_UC':'ID_FAMILIA'})
    .merge(pivot.rename(columns={'ID_UC':'ID_FAMILIA'}),
           on='ID_FAMILIA', how='left'))

for g in GRUPOS_QUAIDS:
    base[g] = base[g].fillna(0)
    base.rename(columns={g: f"x_{g}"}, inplace=True)

# X_TOTAL e shares (exclui Numerário)
x_cols = [f"x_{g}" for g in GRUPOS_QUAIDS
          if g != 'Numerario_Nao_Alimento']
base['X_TOTAL'] = base[x_cols].sum(axis=1)

for g in GRUPOS_QUAIDS:
    if g == 'Numerario_Nao_Alimento': continue
    base[f"w_{g}"] = (base[f"x_{g}"] /
                      base['X_TOTAL'].replace(0, np.nan)).fillna(0)

# Salva
base.to_parquet(OUT_DIR/'consumo_quaids_v2.parquet', index=False)
print(f"\n[OK] Salvo: consumo_quaids_v2.parquet")
print(f"     {len(base):,} UCs × {base.shape[1]} colunas")
print(f"     X_TOTAL médio: R${base['X_TOTAL'].mean():.2f}/ano")
print(f"     UCs com X_TOTAL=0: {(base['X_TOTAL']==0).sum():,}")
print(f"\n{'Grupo':35s} {'Média R$/ano':>12s} {'Zeros%':>8s}")
print("-"*58)
for g in GRUPOS_QUAIDS:
    xc = f"x_{g}"
    if xc not in base.columns: continue
    print(f"{g:35s} {base[xc].mean():12.2f} "
          f"{100*(base[xc]==0).mean():8.1f}%")
