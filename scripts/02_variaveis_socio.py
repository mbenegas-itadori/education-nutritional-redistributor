"""
02_variaveis_socio.py
=====================
Consolida os parquets do script 01 e calcula renda total por UC.

Entrada (de OUT_DIR):
    - domicilio.parquet
    - morador.parquet
    - rendimento_bf.parquet
    + RENDIMENTO_DO_TRABALHO.TXT  (bruto, de DATA_DIR)
    + OUTROS_RENDIMENTOS.TXT      (bruto, de DATA_DIR)

Saída em OUT_DIR:
    - socioeconomico.parquet

Execução no Colab:
    !python 02_variaveis_socio.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR, OUT_DIR

LAYOUT_REND_TRAB = [
    ('UF',                0,   2,  0),
    ('COD_UPA',           7,   9,  0),
    ('NUM_DOM',          16,   2,  0),
    ('NUM_UC',           18,   1,  0),
    ('V8500',            28,  10,  2),
    ('FATOR_ANUALIZACAO',38,  10,  6),
]

LAYOUT_OUTROS_REND = [
    ('UF',                0,   2,  0),
    ('COD_UPA',           7,   9,  0),
    ('NUM_DOM',          16,   2,  0),
    ('NUM_UC',           18,   1,  0),
    ('V9001',            21,   7,  0),
    ('V8500',            28,  10,  2),
    ('FATOR_ANUALIZACAO',38,  10,  6),
]

def le_arquivo_fixo(caminho, layout, encoding='latin-1', chunksize=100_000):
    colspecs = [(c[1], c[1]+c[2]) for c in layout]
    names    = [c[0] for c in layout]
    decimais = {c[0]: c[3] for c in layout}
    chunks = []
    for chunk in pd.read_fwf(caminho, colspecs=colspecs, names=names,
                              encoding=encoding, dtype=str,
                              chunksize=chunksize, header=None):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    for col in names:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if decimais[col] > 0:
            df[col] = df[col] / (10 ** decimais[col])
    print(f"  {caminho.name}: {len(df):,} linhas")
    return df

def cria_id_uc(df):
    df['ID_UC'] = (df['COD_UPA'].astype(int).astype(str) + '_' +
                   df['NUM_DOM'].astype(int).astype(str) + '_' +
                   df['NUM_UC'].astype(int).astype(str))
    return df

def resolve_arquivo(data_dir, *nomes):
    for nome in nomes:
        p = data_dir / nome
        if p.exists(): return p
    raise FileNotFoundError(f"Não encontrado: {nomes} em {data_dir}")

# ============================================================
print("\n[1/4] Carregando parquets do script 01 ...")
dom = pd.read_parquet(OUT_DIR / 'domicilio.parquet')
mor = pd.read_parquet(OUT_DIR / 'morador.parquet')
bf  = pd.read_parquet(OUT_DIR / 'rendimento_bf.parquet')
print(f"  domicilio: {len(dom):,} | morador: {len(mor):,} | bf: {len(bf):,}")

# ============================================================
print("\n[2/4] Renda do trabalho ...")
arq_trab = resolve_arquivo(DATA_DIR,
    'RENDIMENTO_DO_TRABALHO.TXT', 'Rendimento_do_Trabalho.txt')
rt = le_arquivo_fixo(arq_trab, LAYOUT_REND_TRAB)
rt = cria_id_uc(rt)
rt['VALOR_ANUAL'] = rt['V8500'] * rt['FATOR_ANUALIZACAO']
renda_trab = (rt.groupby('ID_UC')['VALOR_ANUAL'].sum()
                .reset_index().rename(columns={'VALOR_ANUAL':'RENDA_TRAB_ANUAL'}))
print(f"  UCs com renda do trabalho: {len(renda_trab):,}")

# ============================================================
print("\n[3/4] Outros rendimentos ...")
arq_outr = resolve_arquivo(DATA_DIR,
    'OUTROS_RENDIMENTOS.TXT', 'Outros_Rendimentos.txt')
ro = le_arquivo_fixo(arq_outr, LAYOUT_OUTROS_REND)
ro = cria_id_uc(ro)
ro['VALOR_ANUAL'] = ro['V8500'] * ro['FATOR_ANUALIZACAO']
renda_outr = (ro.groupby('ID_UC')['VALOR_ANUAL'].sum()
                .reset_index().rename(columns={'VALOR_ANUAL':'RENDA_OUTR_ANUAL'}))
print(f"  UCs com outros rendimentos: {len(renda_outr):,}")

# ============================================================
print("\n[4/4] Consolidando tabela socioeconômica ...")

# Base de UCs vem do morador (cobre todas as UCs incluindo UC 2 e 3)
socio = mor.copy()

# Peso: domicílio tem peso único → cria chave sem NUM_UC para merge
dom['ID_DOM'] = dom['ID_UC'].str.rsplit('_', n=1).str[0]
socio['ID_DOM'] = socio['ID_UC'].str.rsplit('_', n=1).str[0]

socio = socio.merge(
    dom[['ID_DOM','PESO_FINAL','TIPO_SITUACAO_REG','ESTRATO_POF','V6199']],
    on='ID_DOM', how='left')
socio = socio.merge(renda_trab, on='ID_UC', how='left')
socio = socio.merge(renda_outr, on='ID_UC', how='left')
socio = socio.merge(
    bf[['ID_UC','VALOR_BF_ANUAL','VALOR_BF_MENSAL','D_BF','D_TITULAR_MULHER']],
    on='ID_UC', how='left')

# Preenche zeros
for col in ['RENDA_TRAB_ANUAL','RENDA_OUTR_ANUAL','VALOR_BF_ANUAL','VALOR_BF_MENSAL']:
    socio[col] = socio[col].fillna(0)
socio['D_BF'] = socio['D_BF'].fillna(0).astype(int)

# Renda total e per capita
socio['RENDA_TOTAL_ANUAL']  = socio['RENDA_TRAB_ANUAL'] + socio['RENDA_OUTR_ANUAL']
socio['RENDA_TOTAL_MENSAL'] = socio['RENDA_TOTAL_ANUAL'] / 12
socio['RENDA_PC_MENSAL']    = socio['RENDA_TOTAL_MENSAL'] / socio['N_MORADORES']
socio['LOG_RENDA_PC']       = np.log(socio['RENDA_PC_MENSAL'].replace(0, np.nan))
socio['LOG_RENDA_PC_SQ']    = socio['LOG_RENDA_PC'] ** 2
socio['D_URBANO']           = (socio['TIPO_SITUACAO_REG'] == 1).astype(int)
socio['D_INSEG_ALI']        = (socio['V6199'] > 1).astype(int)

socio = socio.drop(columns='ID_DOM', errors='ignore')

print(f"  UCs na base: {len(socio):,}")
print(f"  Peso presente: {socio['PESO_FINAL'].notna().sum():,}")
print(f"  % BF: {100*socio['D_BF'].mean():.1f}%")
print(f"  Renda PC média: R$ {socio['RENDA_PC_MENSAL'].mean():.2f}")

socio.to_parquet(OUT_DIR / 'socioeconomico.parquet', index=False)
print(f"\n  Salvo: {OUT_DIR / 'socioeconomico.parquet'}")
print("\n[OK] 02_variaveis_socio.py concluído.")
