"""
00_classificacao_v2.py — VERSÃO ROBUSTA (CORRIGIDA v2.1)
=========================================================

Classifica produtos POF em 14 grupos QUAIDS + Numerário.

Inputs:
  - POF/dados/CADERNETA_COLETIVA.txt (dados brutos)
  - POF/dados/POF_Classificada_CORRIGIDA.xlsx (ground truth com lookup table)

Outputs:
  - POF/output/classificacao_grupos_v2.parquet
  - POF/dados/POF_Classificada_v2.xlsx

Algoritmo (3 camadas):
  1. LOOKUP TABLE EXATO (ground truth corrigida)
  2. KEYWORDS TEMÁTICAS (fallback para variações)
  3. FAIXAS V9001 (fallback final por código)

Correções v2.1 (2026-06):
  - FIX 1: DESCRICAO agora vem da POF_Classificada_CORRIGIDA.xlsx (já carregada)
            em vez de tentar abrir POF_Classificada_Final.xlsx (arquivo inexistente)
  - FIX 2: VALOR_TOTAL usa V8000_DEFLA * PESO_FINAL (valor deflacionado × peso amostral)
            em vez de V8000 bruto sem fator de expansão

Execução:
  !python 00_classificacao_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATA_DIR = Path('/content/drive/MyDrive/POF/dados')
OUT_DIR = Path('/content/drive/MyDrive/POF/output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRUPOS_VALIDOS = [
    '01.Cereais', '02.Farinhas_Massas', '03.Tuberculos',
    '04.Acucares_Industrializados', '05.Leguminosas_Oleaginosas',
    '06.Frutas', '07.Legumes_Verduras', '08.Carnes', '09.Laticinios',
    '10.Oleos_Gorduras', '11.Bebidas_NA', '12.Alimentacao_Fora',
    '13.Alcool', 'Numerario_Nao_Alimento'
]

# ============================================================
# CAMADA 1: LOOKUP TABLE EXATO (GROUND TRUTH)
# ============================================================

def load_ground_truth(xlsx_path):
    try:
        df = pd.read_excel(xlsx_path)
        assert 'V9001' in df.columns
        assert 'DESCRICAO' in df.columns
        assert 'GRUPO_FINAL' in df.columns

        df['DESCRICAO_NORM'] = df['DESCRICAO'].str.upper().str.strip()
        lookup_v9001 = dict(zip(df['V9001'], df['GRUPO_FINAL']))
        lookup_desc  = dict(zip(df['DESCRICAO_NORM'], df['GRUPO_FINAL']))

        # FIX 1: desc_map vem do ground truth (antes usava POF_Classificada_Final.xlsx
        # que não existe → desc_map = {} → todos INDEFINIDO)
        desc_map = dict(zip(df['V9001'], df['DESCRICAO']))

        print(f"✓ Ground truth carregado: {len(df)} produtos")
        print(f"  Grupos únicos: {df['GRUPO_FINAL'].nunique()}")
        print(f"  Descrições mapeadas: {len(desc_map)}")
        return lookup_v9001, lookup_desc, df, desc_map

    except Exception as e:
        print(f"❌ Erro ao carregar ground truth: {e}")
        return {}, {}, pd.DataFrame(), {}

# ============================================================
# CAMADA 2: KEYWORDS TEMÁTICAS (FALLBACK)
# ============================================================

KEYWORDS_GRUPOS = {
    '05.Leguminosas_Oleaginosas': ('FEIJAO','FEIJÃO','LENTILHA','GRAO DE BICO','SOJA','FAVA','ERVILHA','AMENDOIM','CASTANHA','NÓZES','AMÊNDOA'),
    '07.Legumes_Verduras': ('CENOURA','BETERRABA','NABO','RABANETE','COUVE','ALFACE','REPOLHO','BRÓCOLIS','ABÓBORA','ABOBRINHA','TOMATE','BATATA DOCE','BATATA','VAGEM'),
    '06.Frutas': ('MAÇÃ','BANANA','LARANJA','MORANGO','UVA','MELANCIA','MELÃO','ABACAXI','PÊSSEGO','PERA','GOIABA','MANGA','FRUTA'),
    '08.Carnes': ('CARNE','FRANGO','PEIXE','PRESUNTO','PEITO','DRUMETE','BEEF','BIFE','COSTELA','LINGUIÇA','SALSICHA','BACON','EMPANADO','PEÇA'),
    '09.Laticinios': ('LEITE','QUEIJO','IOGURTE','NATA','MANTEIGA','CREME'),
    '11.Bebidas_NA': ('SUCO','ÁGUA','CHÁ','CAFÉ','REFRIGERANTE','BEBIDA','MATE','SUMO'),
    '13.Alcool': ('CERVEJA','VINHO','BEBIDA ALCOÓLICA','DESTILADO','CACHAÇA','CHAMPAGNE','ESPUMANTE'),
}

def match_keywords(descricao, grupo_faixa):
    desc_upper = str(descricao).upper().strip()
    for grupo, keywords in KEYWORDS_GRUPOS.items():
        if any(kw in desc_upper for kw in keywords):
            return grupo
    return grupo_faixa

# ============================================================
# CAMADA 3: FAIXAS V9001 (FALLBACK FINAL)
# ============================================================

FAIXAS_FINAL = [
    (6300101,6399999,'01.Cereais'),(6400101,6499999,'03.Tuberculos'),
    (6500101,6599999,'02.Farinhas_Massas'),(6600101,6699999,'06.Frutas'),
    (6700101,6799999,'07.Legumes_Verduras'),(6800101,6899999,'06.Frutas'),
    (6900101,6999999,'04.Acucares_Industrializados'),
    (7000101,7099999,'Numerario_Nao_Alimento'),(7100101,7699999,'08.Carnes'),
    (7700101,7700599,'07.Legumes_Verduras'),(7700601,7701999,'Numerario_Nao_Alimento'),
    (7702101,7705999,'08.Carnes'),(7706101,7799999,'04.Acucares_Industrializados'),
    (7800101,7899999,'08.Carnes'),(7900101,7999999,'09.Laticinios'),
    (8000101,8099999,'02.Farinhas_Massas'),(8100101,8199999,'08.Carnes'),
    (8200101,8299999,'11.Bebidas_NA'),(8300101,8399999,'13.Alcool'),
    (8400101,8499999,'10.Oleos_Gorduras'),(8500101,8599999,'12.Alimentacao_Fora'),
    (8600101,8999999,'Numerario_Nao_Alimento'),(9000101,9099999,'Numerario_Nao_Alimento'),
]

CASTANHAS_66 = {6600501,6600502,6600503,6600601,6600602,6600603,6600604,6600605,
    6600701,6600702,6600703,6600704,6601401,6601402,6601403,6601501,6601502,
    6601503,6602001,6602002,6602003,6602101,6602102,6602401,6602402,6602501,
    6602502,6602503}

def faixa_v9001(v9001):
    v = int(v9001)
    if v in CASTANHAS_66:
        return '05.Leguminosas_Oleaginosas'
    for inicio, fim, grupo in FAIXAS_FINAL:
        if inicio <= v <= fim:
            return grupo
    return 'INDEFINIDO'

# ============================================================
# FUNÇÃO PRINCIPAL: CLASSIFICA COM 3 CAMADAS
# ============================================================

def classifica_produto(row, lookup_v9001, lookup_desc):
    v9001 = int(row['V9001'])
    descricao = str(row['DESCRICAO']).upper().strip()

    if v9001 in lookup_v9001:
        return lookup_v9001[v9001], 'LOOKUP_V9001'

    if descricao in lookup_desc:
        return lookup_desc[descricao], 'LOOKUP_DESC'

    grupo_faixa = faixa_v9001(v9001)
    if grupo_faixa != 'INDEFINIDO':
        return match_keywords(descricao, grupo_faixa), 'KEYWORDS_FAIXA'

    return 'INDEFINIDO', 'NENHUMA'

# ============================================================
# [1/4] EXTRAI CADERNETA
# ============================================================

print("\n[1/4] Extraindo CADERNETA_COLETIVA...")

def widths_to_colspecs(widths):
    specs, start = [], 0
    for w in widths:
        specs.append((start, start+w))
        start += w
    return specs

w_cad = [2,4,1,9,2,1,2,3,7,2,10,12,10,1,2,14,14,10,9,4,5,9,5]
c_cad = ['UF','ESTRATO_POF','TIPO_SITUACAO_REG','COD_UPA','NUM_DOM',
         'NUM_UC','QUADRO','SEQ','V9001','V9002','V8000','DEFLATOR',
         'V8000_DEFLA','COD_IMPUT_VALOR','FATOR_ANUALIZACAO','PESO',
         'PESO_FINAL','RENDA_TOTAL','V9005','V9007','V9009','QTD_FINAL','V9004']

try:
    cad = pd.read_fwf(DATA_DIR/'CADERNETA_COLETIVA.txt',
                      colspecs=widths_to_colspecs(w_cad),
                      names=c_cad, encoding='latin-1',
                      header=None, dtype=str)
    cad['V9001']      = pd.to_numeric(cad['V9001'],      errors='coerce')
    cad = cad.dropna(subset=['V9001'])

    # FIX 2: valor expandido = V8000_DEFLA × PESO_FINAL
    # (antes: soma de V8000 bruto → 16× maior que o esperado)
    cad['V8000_DEFLA'] = pd.to_numeric(cad['V8000_DEFLA'], errors='coerce')
    cad['PESO_FINAL']  = pd.to_numeric(cad['PESO_FINAL'],  errors='coerce')
    cad['VALOR_EXPANDIDO'] = cad['V8000_DEFLA'] * cad['PESO_FINAL']

    prod_valor = cad.groupby('V9001')['VALOR_EXPANDIDO'].sum().reset_index()
    prod_valor.columns = ['V9001', 'VALOR_TOTAL']
    print(f"  ✓ Produtos únicos na caderneta: {len(prod_valor):,}")
    print(f"  ✓ Gasto total expandido: R$ {prod_valor['VALOR_TOTAL'].sum():,.2f}")

except Exception as e:
    print(f"  ❌ Erro ao ler CADERNETA: {e}")
    sys.exit(1)

# ============================================================
# [2/4] CARREGA GROUND TRUTH
# ============================================================

print("\n[2/4] Carregando ground truth...")
lookup_v9001, lookup_desc, df_ground, desc_map = load_ground_truth(
    DATA_DIR / 'POF_Classificada_CORRIGIDA.xlsx'
)

# ============================================================
# [3/4] DESCRIÇÕES + CLASSIFICAÇÃO
# ============================================================

print("\n[3/4] Carregando descrições e aplicando classificação...")

produtos = prod_valor.copy()
produtos['V9001'] = produtos['V9001'].astype(int)
produtos['DESCRICAO'] = produtos['V9001'].map(desc_map).fillna('INDEFINIDO')

n_sem_desc = (produtos['DESCRICAO'] == 'INDEFINIDO').sum()
if n_sem_desc > 0:
    print(f"  ⚠️  {n_sem_desc} V9001 sem descrição no ground truth")
else:
    print(f"  ✓ Todas as descrições preenchidas")

print("  Aplicando classificação em 3 camadas...")
result = produtos.apply(
    lambda row: pd.Series(classifica_produto(row, lookup_v9001, lookup_desc)),
    axis=1
)
produtos['GRUPO_FINAL'] = result[0]
produtos['METODO']      = result[1]

# ============================================================
# [4/4] VALIDAÇÕES E OUTPUTS
# ============================================================

print("\n[4/4] Validações e resumo...")

n_indef   = (produtos['GRUPO_FINAL'] == 'INDEFINIDO').sum()
cobertura = 100 * (1 - n_indef / len(produtos))
print(f"\n  Total: {len(produtos):,} | Indefinidos: {n_indef} | Cobertura: {cobertura:.1f}%")

print(f"\n  Métodos:")
for metodo, count in produtos['METODO'].value_counts().items():
    print(f"    {metodo:<30} {count:>6,} ({100*count/len(produtos):>5.1f}%)")

print(f"\n  Distribuição por grupo:")
for grupo in GRUPOS_VALIDOS:
    count = (produtos['GRUPO_FINAL'] == grupo).sum()
    valor = produtos[produtos['GRUPO_FINAL'] == grupo]['VALOR_TOTAL'].sum()
    if count > 0:
        print(f"    {grupo:<35} {count:>5,} prod | R$ {valor:>15,.2f}")

total = produtos['VALOR_TOTAL'].sum()
print(f"\n  Checksum total: R$ {total:,.2f}")

# Salva
classif = produtos[['V9001','DESCRICAO','GRUPO_FINAL','VALOR_TOTAL','METODO']].copy()
classif = classif.sort_values('VALOR_TOTAL', ascending=False).reset_index(drop=True)

classif.to_parquet(OUT_DIR / 'classificacao_grupos_v2.parquet', index=False)
classif.to_excel(DATA_DIR / 'POF_Classificada_v2.xlsx', index=False)
print(f"\n  ✓ Salvo: classificacao_grupos_v2.parquet")
print(f"  ✓ Salvo: POF_Classificada_v2.xlsx")

print("\n" + "="*80)
if n_indef == 0:
    print("✅ CLASSIFICAÇÃO CONCLUÍDA — 100% de cobertura")
else:
    print(f"⚠️  CLASSIFICAÇÃO CONCLUÍDA — {n_indef} indefinidos")
print("="*80)
