"""
01_leitura_pof.py
=================
Lê os arquivos brutos da POF 2017-18 (largura fixa, .txt)
e salva os registros relevantes em parquet.

Arquivos lidos:
    - DOMICILIO.TXT         → peso amostral, estrato, situação
    - MORADOR.TXT           → pessoa de referência: sexo, escolaridade
    - OUTROS_RENDIMENTOS.TXT → Bolsa Família (valor + titular)

Saídas em OUT_DIR (definido em config.py):
    - domicilio.parquet
    - morador.parquet
    - rendimento_bf.parquet

Execução no Colab:
    !python 01_leitura_pof.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR, OUT_DIR, COD_BOLSA_FAMILIA

# ============================================================
# LAYOUTS DE LARGURA FIXA (posições 0-based, do dicionário IBGE)
# Cada tupla: (nome, pos_inicial, tamanho, decimais_implícitos)
# ============================================================

LAYOUT_DOMICILIO = [
    ('UF',                0,   2,  0),
    ('ESTRATO_POF',       2,   4,  0),
    ('TIPO_SITUACAO_REG', 6,   1,  0),
    ('COD_UPA',           7,   9,  0),
    ('NUM_DOM',          16,   2,  0),
    ('PESO',             50,  14,  8),
    ('PESO_FINAL',       64,  14,  8),
    ('V6199',            78,   1,  0),   # segurança alimentar
]

LAYOUT_MORADOR = [
    ('UF',                0,   2,  0),
    ('ESTRATO_POF',       2,   4,  0),
    ('TIPO_SITUACAO_REG', 6,   1,  0),
    ('COD_UPA',           7,   9,  0),
    ('NUM_DOM',          16,   2,  0),
    ('NUM_UC',           18,   1,  0),
    ('COD_INFORMANTE',   19,   2,  0),
    ('V0306',            21,   2,  0),   # condição na UC (01=referência)
    ('V0403',            32,   3,  0),   # idade
    ('V0404',            35,   1,  0),   # sexo (1=H, 2=M)
    ('V0414',            47,   1,  0),   # sabe ler e escrever
    ('V0425',            67,   2,  0),   # último curso frequentado
]

LAYOUT_OUTROS_REND = [
    ('UF',                0,   2,  0),
    ('ESTRATO_POF',       2,   4,  0),
    ('TIPO_SITUACAO_REG', 6,   1,  0),
    ('COD_UPA',           7,   9,  0),
    ('NUM_DOM',          16,   2,  0),
    ('NUM_UC',           18,   1,  0),
    ('COD_INFORMANTE',   19,   2,  0),
    ('V9001',            21,   7,  0),   # código da fonte
    ('V8500',            28,  10,  2),   # valor declarado
    ('FATOR_ANUALIZACAO',38,  10,  6),
]

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def le_arquivo_fixo(caminho: Path, layout: list,
                    encoding='latin-1', chunksize=100_000) -> pd.DataFrame:
    colspecs = [(c[1], c[1] + c[2]) for c in layout]
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


def cria_id_uc(df: pd.DataFrame) -> pd.DataFrame:
    df['ID_UC'] = (df['COD_UPA'].astype(int).astype(str) + '_' +
                   df['NUM_DOM'].astype(int).astype(str) + '_' +
                   df['NUM_UC'].astype(int).astype(str))
    return df


def resolve_arquivo(data_dir: Path, *nomes) -> Path:
    for nome in nomes:
        p = data_dir / nome
        if p.exists():
            return p
    raise FileNotFoundError(f"Arquivo não encontrado. Tentativas: {nomes}\nPasta: {data_dir}")


# ============================================================
# 1. DOMICILIO
# ============================================================
print("\n[1/3] Lendo DOMICILIO.TXT ...")
arq_dom = resolve_arquivo(DATA_DIR, 'DOMICILIO.TXT', 'Domicilio.txt', 'domicilio.txt')
dom = le_arquivo_fixo(arq_dom, LAYOUT_DOMICILIO)
dom['NUM_UC'] = 1
dom = cria_id_uc(dom)
dom = dom.drop(columns='NUM_UC')

print(f"  Peso_Final médio: {dom['PESO_FINAL'].mean():,.2f}")
print(f"  Domicílios únicos: {dom['ID_UC'].nunique():,}")

dom.to_parquet(OUT_DIR / 'domicilio.parquet', index=False)
print(f"  Salvo: {OUT_DIR / 'domicilio.parquet'}")


# ============================================================
# 2. MORADOR
# ============================================================
print("\n[2/3] Lendo MORADOR.TXT ...")
arq_mor = resolve_arquivo(DATA_DIR, 'MORADOR.TXT', 'Morador.txt', 'morador.txt')
mor = le_arquivo_fixo(arq_mor, LAYOUT_MORADOR)
mor = cria_id_uc(mor)

print(f"  Total de moradores: {len(mor):,}")
print(f"  UCs únicas: {mor['ID_UC'].nunique():,}")

# Pessoa de referência (V0306 == 1)
ref = mor[mor['V0306'] == 1].copy()
dups = ref['ID_UC'].duplicated().sum()
if dups > 0:
    print(f"  AVISO: {dups} UCs com >1 pessoa de referência — mantendo primeira")
    ref = ref.drop_duplicates(subset='ID_UC', keep='first')
print(f"  Pessoas de referência: {len(ref):,}")

def categoriza_educ(v):
    if pd.isna(v): return np.nan
    v = int(v)
    if v <= 4:  return 1   # sem instrução / básico incompleto
    elif v <= 8: return 2  # fundamental
    elif v <= 11: return 3 # médio
    else: return 4         # superior ou mais

ref['EDUC_REF']      = ref['V0425'].apply(categoriza_educ)
ref['SEXO_REF']      = ref['V0404']
ref['IDADE_REF']     = ref['V0403']
ref['D_REF_MULHER']  = (ref['V0404'] == 2).astype(int)

tam_uc = mor.groupby('ID_UC').size().reset_index(name='N_MORADORES')

morador_final = ref[['ID_UC', 'SEXO_REF', 'IDADE_REF',
                      'EDUC_REF', 'D_REF_MULHER']].merge(tam_uc, on='ID_UC', how='left')

# Salva versão completa do morador para usar na titularidade do BF
mor[['ID_UC', 'COD_INFORMANTE', 'V0404']].to_parquet(
    OUT_DIR / 'morador_completo.parquet', index=False)

morador_final.to_parquet(OUT_DIR / 'morador.parquet', index=False)
print(f"  Salvo: {OUT_DIR / 'morador.parquet'}")
print(f"  Distribuição escolaridade:\n{morador_final['EDUC_REF'].value_counts().sort_index()}")


# ============================================================
# 3. OUTROS RENDIMENTOS — isola Bolsa Família
# ============================================================
print("\n[3/3] Lendo OUTROS_RENDIMENTOS.TXT ...")
arq_rend = resolve_arquivo(DATA_DIR,
                            'OUTROS_RENDIMENTOS.TXT',
                            'Outros_Rendimentos.txt',
                            'outros_rendimentos.txt')
rend = le_arquivo_fixo(arq_rend, LAYOUT_OUTROS_REND)
rend = cria_id_uc(rend)
rend['VALOR_ANUAL'] = rend['V8500'] * rend['FATOR_ANUALIZACAO']

bf = rend[rend['V9001'] == COD_BOLSA_FAMILIA].copy()
print(f"  Registros de Bolsa Família: {len(bf):,}")
print(f"  UCs beneficiárias: {bf['ID_UC'].nunique():,}")

bf_valor = bf.groupby('ID_UC')['VALOR_ANUAL'].sum().reset_index()
bf_valor.columns = ['ID_UC', 'VALOR_BF_ANUAL']
bf_valor['VALOR_BF_MENSAL'] = bf_valor['VALOR_BF_ANUAL'] / 12
bf_valor['D_BF'] = 1

# Titularidade: cruza COD_INFORMANTE do BF com sexo no MORADOR
mor_completo = pd.read_parquet(OUT_DIR / 'morador_completo.parquet')
bf_titular = bf.merge(
    mor_completo.rename(columns={'V0404': 'SEXO_TITULAR'}),
    on=['ID_UC', 'COD_INFORMANTE'], how='left'
)
titular_sexo = (bf_titular.groupby('ID_UC')['SEXO_TITULAR']
                           .first().reset_index())
titular_sexo['D_TITULAR_MULHER'] = (titular_sexo['SEXO_TITULAR'] == 2).astype(int)

bf_final = bf_valor.merge(
    titular_sexo[['ID_UC', 'SEXO_TITULAR', 'D_TITULAR_MULHER']],
    on='ID_UC', how='left')

print(f"  Titular mulher: {bf_final['D_TITULAR_MULHER'].sum():,} ({100*bf_final['D_TITULAR_MULHER'].mean():.1f}%)")
print(f"  Valor BF mensal médio: R$ {bf_final['VALOR_BF_MENSAL'].mean():.2f}")

bf_final.to_parquet(OUT_DIR / 'rendimento_bf.parquet', index=False)
print(f"  Salvo: {OUT_DIR / 'rendimento_bf.parquet'}")

print("\n[OK] 01_leitura_pof.py concluído.")
