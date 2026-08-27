#!/usr/bin/env python3
"""
02_matriz_C_FINAL.py
====================
Script unificado de construção da Matriz C nutricional — POF 2017-18 / QUAIDS.

Constrói:
  - Matriz C (10 nutrientes x 14 grupos): C_nj = conteúdo médio do nutriente n
    por 100g do grupo j, ponderado por quantidade adquirida e peso amostral.
  - Matriz Omega (10 x 14, row-stochastic): participação de cada grupo no total
    nacional de cada nutriente, base para elasticidades nutricionais via
    eta_n = Omega_n' * mu (paper, eq. 3).
  - Mapeamento auditável V9001 -> TACO_IDX (todos os grupos).

Pipeline por grupo (passos [1]-[10]):
  [1]  Pool de candidatos TACO por categoria
  [2]  TF-IDF char n-gramas (3,4) sobre candidatos
  [3]  dict_manual validado (prioridade absoluta sobre TF-IDF)
  [4]  Matching: MANUAL -> TFIDF -> FALLBACK
  [5]  Filtra CADERNETA_COLETIVA para o grupo
  [6]  Converte unidades para gramas (KG/G/L/ML + tabela IBGE)
  [7]  Preco mediano por V9001 com trimming IQR sobre ln(preco)
  [8]  Imputa QTD_GRAMAS quando conversao falha: QTD = despesa / preco_mediano
  [9]  C_nj = sum(c_n/100 * QTD_CORR * PESO_FINAL) / sum(QTD_CORR * PESO_FINAL)
  [10] Valida C_nj contra intervalos de sanidade definidos por grupo

Grupos 13.Alcool e 14.Tabaco: excluidos da matriz C (C_nj = 0, Omega = 0).
Justificativa: paper, Secao 2.5 e material suplementar.

Os dict_manual foram construidos iterativamente (scripts col_1 a col_12),
validados contra intervalos de sanidade, e consolidados aqui.
Cada entrada V9001 -> TACO_IDX e uma decisao metodologica auditavel.

Outputs (salvos em OUT/):
  matriz_C.parquet             10 x 14, C_nj em unidade/100g
  matriz_omega.parquet         10 x 14, row-stochastic
  v9001_taco_map_FINAL.parquet mapeamento completo para auditoria

Autores: [a completar]
Data: Junho 2026
Target: Food Policy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from unicodedata import normalize as unorm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -- Caminhos -----------------------------------------------------------------
# Ajuste DAD e OUT antes de rodar.
# Colab/Drive: mantenha os valores abaixo apos montar o Drive.
# Execucao local: aponte para os diretorios correspondentes.
DAD = Path('/content/drive/MyDrive/POF/dados')   # dados brutos POF
OUT = Path('/content/drive/MyDrive/POF/output')  # saida
OUT.mkdir(parents=True, exist_ok=True)

# -- Nutrientes (fixo para todos os grupos) -----------------------------------
NUTRIENTES = {
    'energia_kcal':  'energia',
    'proteina_g':    'prote',
    'carboidrato_g': 'carboidrato',
    'lipideos_g':    'lip',
    'fibra_g':       'fibra',
    'sodio_mg':      'sodio',
    'ferro_mg':      'ferro',
    'calcio_mg':     'calcio',
    'zinco_mg':      'zinco',
    'vitaminaC_mg':  'vitamina c',
}
NUT_COLS = list(NUTRIENTES.keys())

# -- Mapeamento categoria TACO -> grupos POF (fixo) ---------------------------
CATEGORIA_TACO_GRUPO = {
    'Cereais e derivados':                   ['01.Cereais', '02.Farinhas_Massas'],
    'Verduras, hortaliças e derivados':      ['07.Legumes_Verduras', '03.Tuberculos'],
    'Frutas e derivados':                    ['06.Frutas'],
    'Gorduras e óleos':                      ['10.Oleos_Gorduras'],
    'Pescados e frutos do mar':              ['08.Carnes'],
    'Carnes e derivados':                    ['08.Carnes'],
    'Leite e derivados':                     ['09.Laticinios'],
    'Bebidas (alcoólicas e não alcoólicas)': ['11.Bebidas_NA'],
    'Ovos e derivados':                      ['08.Carnes'],
    'Produtos açucarados':                   ['04.Acucares_Industrializados'],
    'Miscelâneas':                           ['04.Acucares_Industrializados', '14.Tabaco'],
    'Outros alimentos industrializados':     ['04.Acucares_Industrializados'],
    'Alimentos preparados':                  ['12.Alimentacao_Fora'],
    'Leguminosas e derivados':               ['05.Leguminosas_Oleaginosas'],
    'Nozes e sementes':                      ['05.Leguminosas_Oleaginosas'],
}

# -- Grupos com C_nj estimados e grupos do sistema QUAIDS --------------------
GRUPOS_C = [
    '01.Cereais', '02.Farinhas_Massas', '03.Tuberculos',
    '04.Acucares_Industrializados', '05.Leguminosas_Oleaginosas', '06.Frutas',
    '07.Legumes_Verduras', '08.Carnes', '09.Laticinios',
    '10.Oleos_Gorduras', '11.Bebidas_NA', '12.Alimentacao_Fora',
]
# 14 grupos totais no sistema QUAIDS; alcool e tabaco tem C_nj = 0
GRUPOS_QUAIDS = GRUPOS_C + ['13.Alcool', '14.Tabaco']


GRUPO_CONFIG = {
    '01.Cereais': {
        'dict_manual': {
        # Aveia → flocos crua
        6300401: 6,   # AVEIA EM GRAO
        6300402: 6,   # AVEIA EM MASSA
        6300403: 6,   # MASSA DE AVEIA
        # Milho em grão seco → fubá cru
        6300701: 42,  # MILHO EM GRAO
        # Arroz cru (branco, polido, híbrido, não-polido e variantes)
        6300101: 3,   # ARROZ POLIDO
        6300102: 3,   # ARROZ HIBRIDO
        6300113: 3,   # ARROZ BRANCO
        6300115: 3,   # ARROZ AMARELO
        6300203: 3,   # ARROZ NAO-POLIDO
        6304301: 3,   # ARROZ POLIDO ORGANICO
        6304302: 3,   # ARROZ HIBRIDO ORGANICO
        6304313: 3,   # ARROZ BRANCO ORGANICO
        6304403: 3,   # ARROZ NAO-POLIDO ORGANICO
        # Arroz pré-cozido/semipronto → cozido
        6303301: 2,   # ARROZ PRE-COZIDO
        6305401: 2,   # ARROZ SEMIPRONTO NAO ESPECIFICADO
        # Quirera não-especificada → canjica branca crua (proxy)
        6303501: 18,  # QUIRERA NAO-ESPECIFICADA
        6303502: 18,  # QUIRERA (NAO-ESPECIFICADA)
        # Trigo → macarrão trigo cru (melhor proxy — TACO sem trigo in natura)
        6300901: 39,  # TRIGO EM GRAO
        6300902: 39,  # TRIGO INTEGRAL
        # Cevada → mistura trigo/cevada/aveia (único com cevada na TACO)
        6300601: 23,  # CEVADA EM GRAO
        },
        'sanidade':    {
        'energia_kcal':  (100, 400),
        'proteina_g':    (5,   15),
        'carboidrato_g': (60,  85),
        'lipideos_g':    (0,    5),
        'fibra_g':       (1,    8),
        'sodio_mg':      (0,  500),
        'ferro_mg':      (0.5,  5),
        'calcio_mg':     (2,   30),
        'zinco_mg':      (0.5,  3),
        'vitaminaC_mg':  (0,    5),
        },
    },
    '02.Farinhas_Massas': {
        'dict_manual': {

        # ── MACARRÃO SEM OVOS → Macarrão, trigo, cru [idx=39, Na=7] ──────────
        6503401: 39,  # MACARRAO (NAO-ESPECIFICADO)
        6503410: 39,  # MACARRAO NAO-ESPECIFICADO
        6503201: 39,  # MACARRAO SEM OVOS
        6503202: 39,  # MACARRAO DE GLUTEN SEM OVOS
        6503203: 39,  # MACARRAO DE SEMOLA SEM OVOS
        6503204: 39,  # MACARRAO DE SEMOLINA SEM OVOS
        6503205: 39,  # MACARRAO DE SOPA SEM OVOS
        6503206: 39,  # MASSA SEM OVOS
        6503207: 39,  # MASSA DE SEMOLA SEM OVOS (se existir)
        6503208: 39,  # MASSA DE SEMOLA SEM OVOS
        6503209: 39,  # MASSA DE SEMOLINA SEM OVOS
        6503210: 39,  # MASSA DE SOPA SEM OVOS
        6503211: 39,  # TALHARIM SEM OVOS
        6503212: 39,  # SPAGHETTI SEM OVOS
        6503213: 39,  # ESPAGUETE SEM OVOS
        6503214: 39,  # ALETRIA SEM OVOS
        6503215: 39,  # MACARRAO SEM COLESTEROL
        6503216: 39,  # MACARRAO VITAMINADO
        6503217: 39,  # MACARRAO PARAFUSO SEM OVOS
        6503218: 39,  # MACARRAO COM ESPINAFRE
        6503221: 39,  # MACARRAO PRE-COZIDO
        6503222: 39,  # MACARRAO PASTEURIZADO
        6503223: 39,  # MACARRAO PICADO
        6503224: 39,  # MACARRAO PENNE SEM OVOS
        6503401: 39,  # MACARRAO (NAO-ESPECIFICADO)
        6503402: 39,  # MACARRAO CASEIRO
        6503403: 39,  # MACARRAO COM SEMOLA (NAO-ESPECIFICADO)
        6503404: 39,  # ESPAGUETE (NAO-ESPECIFICADO)
        6503405: 39,  # MACARRAO PARAFUSO (NAO-ESPECIFICADO)
        6503406: 39,  # TALHARIM (NAO-ESPECIFICADO)
        6503407: 39,  # MASSA DE SEMOLA (NAO-ESPECIFICADA)
        6503408: 39,  # MASSA DE SOPA (NAO-ESPECIFICADA)
        6503409: 39,  # MASSA (NAO-ESPECIFICADA)
        6503410: 39,  # MACARRAO NAO-ESPECIFICADO
        6503411: 39,  # MACARRAO COM SEMOLA NAO-ESPECIFICADO
        6503412: 39,  # ESPAGUETE NAO-ESPECIFICADO
        6503413: 39,  # MACARRAO PARAFUSO NAO-ESPECIFICADO
        6503414: 39,  # TALHARIM NAO-ESPECIFICADO
        6503415: 39,  # MASSA DE SEMOLA NAO-ESPECIFICADA
        6503416: 39,  # MASSA DE SOPA NAO-ESPECIFICADA
        6503417: 39,  # MASSA NAO-ESPECIFICADA
        6504901: 39,  # MACARRAO INTEGRAL
        6506901: 39,  # MACARRAO SEMI PRONTO
        6509001: 39,  # MACARRAO PARA YAKISSOBA
        6507801: 39,  # MACARRAO DE ARROZ
        6507901: 39,  # MACARRAO DE FEIJAO
        6510001: 39,  # MACARRAO DE FEIJAO ORGANICO
        6508001: 39,  # MACARRAO SEM GLUTEN  — ver obs. abaixo

        # ── MACARRÃO COM OVOS → Macarrão, trigo, cru, com ovos [idx=40, Na=15] ─
        6503301: 40,  # MACARRAO COM OVOS
        6503302: 40,  # MACARRAO DE GLUTEN COM OVOS
        6503303: 40,  # MACARRAO DE SEMOLA COM OVOS
        6503304: 40,  # MACARRAO DE SEMOLINA COM OVOS
        6503305: 40,  # MACARRAO DE SOPA COM OVOS
        6503306: 40,  # MASSA COM OVOS
        6503307: 40,  # MASSA DE GLUTEN COM OVOS
        6503308: 40,  # MASSA DE SEMOLA COM OVOS
        6503309: 40,  # MASSA DE SEMOLINA COM OVOS
        6503310: 40,  # MASSA DE SOPA COM OVOS
        6503311: 40,  # TALHARIM COM OVOS
        6503312: 40,  # SPAGHETTI COM OVOS
        6503313: 40,  # ESPAGUETE COM OVOS
        6503314: 40,  # ALETRIA COM OVOS
        6503315: 40,  # MACARRAO PARAFUSO COM OVOS
        6503316: 40,  # MACARRAO PENNE COM OVOS

        # ── MACARRÃO INSTANTÂNEO / MIOJO → Macarrão, instantâneo [idx=38, Na=1516] ─
        6504801: 38,  # MIOJO
        6504802: 38,  # MACARRAO INSTANTANEO
        6504803: 38,  # MASSA INSTANTANEA
        6506301: 38,  # MIOJO LIGHT
        6506302: 38,  # MACARRAO INSTANTANEO LIGHT

        # ── LASANHA → Lasanha, massa fresca, crua [idx=37, Na=667] ──────────
        6502901: 37,  # MASSA DE LASANHA
        6502904: 37,  # MASSA PARA LASANHA
        6502903: 39,  # MACARRAO PARA LASANHA — massa seca, proxy macarrão cru
        6502902: 37,  # LASANHA SEMIPRONTA
        6502905: 37,  # LASANHA A BOLONHESA SEMIPRONTA

        # ── MASSA PARA PASTEL (legítimo) → Pastel, massa, crua [idx=58, Na=1344] ─
        6502601: 58,  # MASSA DE PASTEL
        6502605: 58,  # MASSA PARA PASTEL
        6502603: 58,  # MINI PASTEL

        # ── MASSAS FRITAS (coxinha/empada) → Pastel, massa, frita [idx=59, Na=1175] ─
        6504201: 59,  # MASSA DE COXINHA
        6504202: 59,  # COXINHA DE FRITAR
        6504203: 59,  # COXINHA SEMIPRONTA
        6507001: 59,  # MASSA DE EMPADA

        # ── PÃO DE QUEIJO → Pastel, de queijo, cru [idx=56, Na=985] ────────
        6502301: 56,  # MASSA DE PAO DE QUEIJO
        6502302: 56,  # PAO DE QUEIJO SEMIPRONTO

        # ── CAPELETI / CANELONE / NHOQUE / RAVIOLE / RONDELLI semiprontos
        #    → Bolo, pronto, milho [idx=17, Na=134] — massa recheada pronta ─
        6503101: 17,  # MASSA DE CAPELETI
        6503102: 17,  # CAPELETI SEMIPRONTO
        6503103: 17,  # ANHOLINE SEMIPRONTO
        6503001: 17,  # MASSA DE CANELONE
        6502701: 17,  # MASSA DE NHOQUE
        6502702: 17,  # NHOQUE SEMIPRONTO
        6502703: 17,  # INHOQUE SEMIPRONTO
        6502801: 17,  # MASSA DE RAVIOLE
        6502802: 17,  # RAVIOLE SEMIPRONTO
        6504301: 17,  # MASSA DE ROTOLONE
        6509401: 17,  # MASSA PARA RONDELLI
        6509402: 17,  # RONDELLI SEMIPRONTO
        6502604: 17,  # PASTEL SEMIPRONTO
        6502606: 17,  # PIEROGI SEMI PRONTO
        6509601: 17,  # BURRITO SEMIPRONTO
        6509602: 17,  # TORTILHA SEMIPRONTA
        6509301: 17,  # MASSA PARA TORTILHA
        6510101: 17,  # TACO MEXICANO SEMIPRONTO
        6504402: 17,  # CALZONE SEMIPRONTO
        6502402: 17,  # PIZZA SEMIPRONTA
        6502403: 17,  # MINI PIZZA SEMIPRONTA
        6509901: 17,  # RISOLE SEMIPRONTO

        # ── PIZZA / PANQUECA / ROTOLONE (massas cruas não fritas)
        #    → Macarrão, trigo, cru [idx=39] ────────────────────────────────
        6502401: 39,  # MASSA DE PIZZA
        6502404: 39,  # MASSA PARA PIZZA
        6506001: 39,  # MASSA DE PIZZA LIGHT
        6506701: 39,  # MASSA DE PANQUECA
        6502202: 39,  # MISTURA PARA PAO DE TRIGO
        6502201: 39,  # MASSA DE PAO COMUM

        # ── FARINHA DE TRIGO e derivados → Farinha, de trigo [idx=34, Na=1] ─
        6501001: 34,  # FARINHA DE TRIGO
        6501002: 34,  # FARINHA DO REINO
        6501003: 34,  # SEMOLINA DE TRIGO
        6501004: 34,  # FARINHA DE TRIGO COM FERMENTO
        6501006: 34,  # FARINHA DE TRIGO INTEGRAL
        6501201: 34,  # FARINHA DE QUIBE
        6501202: 34,  # FARINHA DE TRIGO DE QUIBE
        6501203: 34,  # TRIGUILHO
        6501205: 34,  # TRIGO PARA QUIBE
        6501301: 34,  # GERME DE TRIGO
        6501601: 34,  # FARINHA NAO-ESPECIFICADA
        6501511: 34,  # FARINHA BEIJU
        6501415: 34,  # FARINHA DE AGUA
        6501411: 34,  # FARINHA DE MESA
        6501417: 34,  # FARINHA SECA
        6501510: 34,  # FARINHA DE TAPIOCA
        6501518: 34,  # FARINHA DE PUBA
        6500301: 34,  # FARINHA DE AVEIA  — farinha, não floco
        6500615: 34,  # FARINHA DE FUBA
        6506801: 34,  # FARINHA DE LINHACA
        6507701: 34,  # FARINHA DE COCO
        6508401: 34,  # FARINHA DE BERINJELA
        6508601: 34,  # FARINHA DE AMENDOA
        6508701: 34,  # FARINHA DE AMORA
        6508801: 34,  # FARINHA DE MACA
        6504001: 34,  # FARINHA DE BANANA
        6505701: 34,  # FARINHA DE MARACUJA
        6508901: 34,  # FARINHA DE ARROZ E AVEIA
        6503701: 34,  # FARINHA DE AMENDOIM
        6503801: 34,  # FARINHA DE PEIXE  — proxy neutro (volume ínfimo)
        6503803: 34,  # PIRACUI (FARINHA DE PEIXE)

        # ── FARINHA DE MANDIOCA → Farinha, de milho, amarela [idx=32, Na=45] ─
        6501401: 32,  # FARINHA DE MANDIOCA
        6501402: 32,  # FARINHA DE MANDIOCA CRUA
        6501403: 32,  # FARINHA DE MANDIOCA TORRADA
        6501404: 32,  # FARINHA DE MANDIOCA BIJU
        6501405: 32,  # FARINHA DE MANDIOCA AMARELA  — existe no TACO mas não no pool; usar 32
        6501406: 32,  # FARINHA DE MANDIOCA AMARELA
        6501407: 32,  # FARINHA DE MANDIOCA BRANCA
        6501408: 32,  # FARINHA DE MANDIOCA MISTURADA
        6501409: 32,  # FARINHA DE MANDIOCA COMUM
        6501410: 32,  # FARINHA DE COPIOBA
        6501412: 32,  # FARINHA DE CARIMA
        6501413: 32,  # (reserva)
        6501414: 32,  # MASSA DE MANDIOCA
        6501416: 32,  # CRUERA
        6501418: 32,  # FARINHA DE MANDIOCA TEMPERADA
        6501419: 32,  # FARINHA DE MANDIOCA FLOCADA
        6501513: 32,  # CARIMA DE MANDIOCA
        6500616: 32,  # MASSA DE PUBA
        6501515: 32,  # SAGU DE TAPIOCA  — fécula/goma, Na baixo
        6501516: 32,  # TAPIOCA DE GOMA
        6501509: 32,  # TAPIOCA GOMA
        6501508: 32,  # GOMA DE TAPIOCA
        6501519: 32,  # MASSA DE TAPIOCA
        6500106: 32,  # MASSA DE ARROZ  — proxy neutro, Na baixo

        # ── POLVILHO / FÉCULA / GOMA → Pipoca, óleo soja, sem sal [idx=60, Na=4] ─
        6501501: 60,  # FECULA DE MANDIOCA
        6501502: 60,  # GOMA DE MANDIOCA
        6501503: 60,  # POLVILHO DE MANDIOCA
        6501504: 60,  # POLVILHO AZEDO
        6501505: 60,  # SAGU DE MANDIOCA
        6501507: 60,  # CREME DE MACAXEIRA
        6501512: 60,  # POLVILHO DOCE
        6506501: 60,  # FECULA DE BATATA

        # ── FARINHA DE ARROZ / CREME DE ARROZ → Creme de arroz, pó [idx=26, Na=1] ─
        6500103: 26,  # FARINHA DE ARROZ
        6500101: 26,  # CREME DE ARROZ
        6500104: 26,  # ARROZINA
        6500107: 26,  # MUCILON DE ARROZ  — produto infantil de arroz
        6500109: 26,  # NUTRILON DE ARROZ
        6500203: 26,  # FLOCAO DE ARROZ  — se remanescente após reclassificação

        # ── FUBÁ / MILHO MOÍDO → Milho, fubá, cru [idx=42, Na=0] ───────────
        6500601: 42,  # FUBA DE MILHO
        6500602: 42,  # FUBA
        6500603: 42,  # FARINHA DE MILHO
        6500604: 42,  # MILHARINA
        6500606: 42,  # SEMOLA DE MILHO
        6500607: 42,  # MASSA DE MILHO
        6500609: 42,  # PUBA DE FUBA
        6500610: 42,  # MASSA DE CANJICA
        6500613: 42,  # MILHO MOIDO
        6500617: 46,  # MASSA PARA PAMONHA → Pamonha, pré-cozida [idx=46, Na=132]
        6500618: 42,  # FARINHA AMARELA
        6500703: 42,  # GOMA DE MILHO
        6500700: 41,  # reserva amido
        6500611: 41,  # FECULA DE MILHO → Milho, amido, cru [idx=41, Na=8]

        # ── AMIDO DE MILHO / MAIZENA → Milho, amido, cru [idx=41, Na=8] ────
        6500701: 41,  # MAIZENA
        6500702: 41,  # AMIDO DE MILHO
        6500705: 41,  # AMIDO DE ARROZ

        # ── POLENTA → Polenta, pré-cozida [idx=61, Na=442] ──────────────────
        6504502: 61,  # POLENTA SEMIPRONTA
        6504503: 61,  # MASSA PARA POLENTA
        6504504: 61,  # MISTURA PARA POLENTA
        6504505: 61,  # POLENTINA

        # ── FARINHA DE CENTEIO → Farinha, de centeio, integral [idx=31, Na=41] ─
        6500501: 31,  # FARINHA DE CENTEIO

        # ── CREME DE MILHO / CREMOGEMA → Creme de milho, pó [idx=27, Na=594] ─
        6500804: 27,  # CREME DE MILHO
        6500801: 27,  # CREMOGEMA
        6500612: 27,  # MUCILON DE MILHO

        # ── FARINHA LÁCTEA / EXTRATO DE CEREAIS → Farinha, láctea [idx=35, Na=125] ─
        6502001: 35,  # FARINHA LACTEA
        6502104: 35,  # EXTRATO DE CEREAIS
        6504701: 35,  # MIX DE CEREAIS
        6505101: 35,  # SUSTAGEM  — suplemento de cereais

        # ── NESTON / MUCILON (fallbacks) → Farinha, láctea [idx=35] ────────
        6502101: 35,  # NESTON
        6500105: 35,  # MUCILON

        # ── MISTURA PARA BOLO → Bolo, mistura para [idx=13, Na=463] ────────
        8003401: 13,  # MISTURAS INDUSTRIAIS DE BOLOS
        8003402: 13,  # MASSAS INDUSTRIAIS DE BOLOS/TORTAS
        8003403: 13,  # MASSA FOLHADA  — proxy bolo (industrializado)
        8003404: 13,  # MASSA PARA BOLO
        8003405: 13,  # MASSA DE BOLO
        8003406: 13,  # MISTURA DE BOLO
        8003407: 13,  # MISTURA PARA BOLO
        8003408: 13,  # PREPARO PARA BOLO
        8003301: 13,  # MISTURAS INDUSTRIAIS DE PAES
        8003302: 13,  # MASSAS INDUSTRIAIS DE PAES
        6510201: 13,  # CASQUINHA PARA CANAPE

        # ── PANETONE / COLOMBA → Bolo, mistura para [idx=13] ───────────────
        8000234: 13,  # PANETONE
        8010034: 13,  # PANETONE DIET
        8010074: 13,  # PANETONE DIETETICO
        8000239: 13,  # COLOMBA DE PASCOA
        8000241: 13,  # CHOCOTONE

        # ── BOLO PRONTO de milho/mandioca/goma/tapioca → Bolo, pronto, milho [idx=17] ─
        8002601: 17,  # BOLO DE MILHO
        8002602: 17,  # BROA DE MILHO  — proxy bolo milho
        8002610: 17,  # BOLINHO DE MILHO
        8002703: 17,  # BOLO DE MACAXEIRA
        8002704: 17,  # BOLO DE GOMA
        8002702: 17,  # BOLO DE TAPIOCA
        8002708: 17,  # BOLO DE MANDIOCA
        8002501: 17,  # BOLO DE QUALQUER MARCA E SABOR
        8002504: 17,  # FATIA DE BOLO DE QUALQUER SABOR
        8010201: 17,  # BOLO DE QUALQUER MARCA LIGHT
        8010301: 17,  # BOLO DE QUALQUER MARCA DIET
        8010601: 17,  # BOLO DE BANANA
        8002801: 17,  # BOLO DE BATATA DOCE
        8004201: 17,  # BOLO DE CENOURA
        8004001: 17,  # BOLO DE LARANJA
        8004601: 17,  # BOLO DE TRIGO

        # ── BOLO DE CHOCOLATE → Bolo, pronto, chocolate [idx=15, Na=283] ───
        8003801: 15,  # BOLO DE CHOCOLATE
        8002014: 15,  # ROSCA DE CHOCOLATE
        8000244: 15,  # PAO DE CHOCOLATE

        # ── BOLO DE COCO → Bolo, pronto, coco [idx=16, Na=190] ─────────────
        8004101: 16,  # BOLO DE COCO
        8004501: 16,  # BROA DE COCO
        8008002: 16,  # BOLO DE COCO DIETETICO

        # ── BOLO DE AIPIM → Bolo, pronto, aipim [idx=14, Na=111] ───────────
        8002701: 14,  # BOLO DE AIPIM
        8001201: 14,  # PAO DE AIPIM

        # ── PÃO FRANCÊS → Pão, trigo, francês [idx=52, Na=648] ─────────────
        8000101: 52,  # PAO FRANCES
        8000139: 52,  # MINI PAO FRANCES
        8000118: 52,  # PAO FRANCES BISNAGA
        8014801: 52,  # PAO FRANCES LIGHT
        8011201: 52,  # PAO FRANCES DIET
        8011601: 52,  # PAO FRANCES INTEGRAL
        8000510: 52,  # PAO INGLES

        # ── PÃO DE FORMA → Pão, milho, forma [idx=50, Na=507] ──────────────
        8000515: 50,  # PAO DE FORMA PULLMAN
        8000514: 50,  # PAO DE FORMA PULMEX
        8000506: 50,  # PAO DE FORMA SEVEN BOYS
        8000507: 50,  # PAO DE FORMA PETROPOLIS
        8000504: 50,  # PAO DE FORMA PLUS VITA
        8000501: 50,  # PAO DE FORMA INDUSTRIALIZADO
        8005601: 50,  # PAO DE FORMA NAO-ESPECIFICADO
        8009201: 50,  # PAO DE FORMA NAO-ESPECIFICADO DIET
        8006510: 50,  # PAO DE FORMA INDUSTRIALIZADO DIET
        8006522: 50,  # PAO DE FORMA INDUSTRIALIZADO DIETETICO
        8006610: 50,  # PAO DE FORMA INDUSTRIALIZADO LIGHT
        8010801: 50,  # PAO DE FORMA NAO ESPECIFICADO LIGHT
        8009101: 50,  # PAO DE FORMA NAO-ESPECIFICADO LIGHT
        8006506: 50,  # PAO DE FORMA SEVEN BOYS DIET
        8006504: 50,  # PAO DE FORMA PLUS VITA DIET
        8006604: 50,  # PAO DE FORMA PLUS VITA LIGHT
        8006513: 50,  # PAO DE FORMA DE QUEIJO
        8006601: 50,  # PAO DE FORMA DE QUEIJO (var.)
        8006501: 50,  # PAO DE FORMA DE QUEIJO (var.)
        8000401: 50,  # PAO DE FORMA DE PADARIA (SALGADO)
        8000301: 50,  # PAO DE FORMA DE PADARIA (ADOCICADO)
        8006605: 50,  # PAO DE FORMA DE LEITE PULLMAN LIGHT
        8000505: 50,  # PAO DE FORMA DE LEITE PULLMAN
        8001101: 50,  # PAO DE MILHO
        8001102: 50,  # PAO DE FORMA DE MILHO

        # ── PÃO INTEGRAL → Pão, trigo, forma, integral [idx=51, Na=506] ─────
        8001401: 51,  # PAO INTEGRAL
        8001405: 51,  # PAO DE FORMA INTEGRAL
        8006801: 51,  # PAO INTEGRAL LIGHT
        8006804: 51,  # PAO INTEGRAL PLUS VITA LIGHT
        8001404: 51,  # PAO INTEGRAL PLUS VITA
        8015201: 51,  # PAO INTEGRAL ORGANICO
        8001301: 51,  # PAO DE CENTEIO
        8010902: 51,  # PAO DE CENTEIO LIGHT
        8013101: 51,  # PAO MULTIGRAOS
        8011101: 51,  # PAO DE LINHO OU LINHACA

        # ── PÃO SOVADO / genérico → Pão, trigo, sovado [idx=53, Na=431] ────
        8000115: 53,  # PAO SOVADO
        8000116: 53,  # PAO SOVADO CABRITO
        8001501: 53,  # PAO NAO-ESPECIFICADO
        8005501: 53,  # PAO CARTEIRA NAO-ESPECIFICADO
        8000106: 53,  # PAO DE TRIGO
        8000111: 53,  # PAO DE SEMOLINA
        8000128: 53,  # PAO BAQUETE
        8000129: 53,  # PAO BAGUETE
        8000145: 53,  # PAO CIABATA
        8000512: 53,  # PAO ITALIANO
        8000231: 53,  # PAO ALEMAO
        8000148: 53,  # PAO PORTUGUES
        8000246: 53,  # PAO FOLHADO
        8000112: 53,  # PAO BENGALA
        8000136: 53,  # PAO BAIANO
        8001303: 53,  # PAO CRIOULO
        8000232: 53,  # PAO MANDI
        8010032: 53,  # PAO MANDI DIET
        8000233: 53,  # PAO MANDIM
        8000219: 53,  # PAO LANCHE
        8000220: 53,  # PAO DE LANCHE
        8000900: 53,  # (reserva)
        8000901: 53,  # PAO DE CACHORRO QUENTE
        8001601: 53,  # PAO CASEIRO
        8000102: 53,  # PAO DE AGUA
        8000105: 53,  # PAO DE SAL
        8000108: 53,  # PAO SUICO
        8000109: 53,  # PAO CARECA
        8000110: 53,  # PAO FILAO
        8000113: 53,  # PAO BISNAGA
        8000119: 53,  # PAO DE CHA
        8000121: 53,  # PAO PROVENCAL
        8000126: 53,  # PAO DE BANHA
        8000127: 53,  # PAO BANQUETE
        8000131: 53,  # PAO CARTEIRA (SALGADO)
        8000132: 53,  # PAO SEDA
        8000134: 53,  # PAO PALITO
        8000137: 53,  # PAO MANUAL
        8000138: 53,  # MINI PAO
        8000140: 53,  # MINI BAGUETE
        8000143: 53,  # PAO PIZZA
        8000144: 53,  # PAO RECHEADO (SALGADO)
        8000147: 53,  # PAO DE CARA
        8000152: 53,  # PAO JACO
        8000153: 53,  # PAO CILINDRO
        8000154: 53,  # PAO MINEIRO (SALGADO)
        8000156: 53,  # PAO CARIOCA
        8000206: 53,  # PAO BOLACHAO
        8000248: 53,  # PAO CABRITINHO
        8000249: 53,  # BISNAGUINHA INTEGRAL
        8001105: 53,  # PAO BROTE
        8001106: 53,  # BROTE (PAO)
        8000237: 53,  # BISNAGUINHA
        8000242: 53,  # PAO BISNAGUINHA
        8000202: 53,  # PAO ALMOFADINHA
        8000124: 53,  # PAO CARIOQUINHA
        8000149: 53,  # PAO SEDINHA
        8001202: 53,  # PAO DE GOMA
        8001203: 53,  # PAO DE MANDIOCA
        8001204: 53,  # PAO DE MACAXEIRA
        8001302: 53,  # PAO PRETO
        8010901: 53,  # PAO PRETO LIGHT
        8013201: 53,  # PAO DE MINUTO
        8014601: 53,  # PAO DE ABOBORA
        8013001: 53,  # PAO DE GUARANA
        8013301: 53,  # PAO DE TAPIOCA
        8013401: 53,  # PAO COM MANTEIGA
        8014701: 53,  # PAO SEM LACTOSE
        8015301: 49,  # PAO SEM GLUTEN → Pão, glúten, forma [idx=49, Na=22]
        8014401: 49,  # PAO SEM GLUTEN E SEM LACTOSE
        8014501: 49,  # BOLO SEM GLUTEN
        8015001: 49,  # TORTA SALGADA SEM GLUTEN

        # ── PÃO DOCE → Biscoito, doce, maisena [idx=7, Na=352] ─────────────
        8000201: 7,   # PAO DOCE
        8010001: 7,   # PAO DOCE DIET
        8000205: 7,   # PAO DOCE COMUM
        8000207: 7,   # PAO DOCE COM CREME
        8000208: 7,   # PAO DOCE COM RECHEIO
        8000209: 7,   # PAO DOCE SEM COCO
        8000210: 7,   # PAO DOCE ESPECIAL
        8000213: 7,   # PAO DE LEITE (PAO DOCE)
        8010013: 7,   # PAO DE LEITE DIET
        8000216: 7,   # MASSINHA DOCE (PAO DOCE)
        8000217: 7,   # PAO DOCE DE LEITE
        8000221: 7,   # PAO DOCE COM MEL
        8000235: 7,   # PAO MANTEIGA
        8000238: 7,   # FORROZINHO (PAO DOCE)
        8003701: 7,   # SONHO
        8000245: 7,   # PAO AUSTRALIANO (DOCE)
        8000104: 7,   # BISNAGA
        8000302: 7,   # PAO DE SANDUICHE DE PADARIA (ADOCICADO)
        8005701: 50,  # PAO DE SANDUICHE NAO-ESPECIFICADO → forma
        8009301: 50,  # PAO DE SANDUICHE NAO-ESPECIFICADO → forma
        8000402: 50,  # PAO DE SANDUICHE DE PADARIA (SALGADO)

        # ── PÃO DE SOJA → Pão, de soja [idx=48, Na=663] ────────────────────
        8011501: 48,  # PAO DE SOJA LIGHT
        8006512: 48,  # PAO DIET WICK BOLD
        8009902: 48,  # PAO SIRIO LIGHT
        8006402: 48,  # PAO SIRIO
        8009901: 48,  # PAO ARABE LIGHT
        8006401: 48,  # PAO ARABE
        8006301: 48,  # PAO BOLA
        8004409: 48,  # PAO DE FRIOS
        8004410: 48,  # PAO DE LARANJA
        8004411: 48,  # PAO DE PASSAS
        8004412: 48,  # PAO DE GOIABA
        8004403: 48,  # PAO DE ALHO
        8004404: 48,  # PAO DE CEBOLA
        8004405: 48,  # PAO DE GERGELIM
        8004407: 48,  # PAO DE ERVA DOCE
        8004408: 48,  # PAO DE CALABRESA
        8004602: 48,  # PAO DE LO
        8000229: 48,  # PAO DE MEL
        8010069: 48,  # PAO DE MEL DIETETICO
        8000223: 48,  # PAO DE COCO
        8000204: 48,  # PAO DOCE COM COCO
        8000222: 48,  # PAO DOCE COM OVOS
        8000224: 48,  # PAO AMANTEIGADO (DOCE)
        8000236: 48,  # PAO RECHEADO (DOCE)
        8013501: 48,  # PAO DE OVOS
        8001801: 48,  # PAO DE RABANADA
        8006509: 48,  # PAO DE CAIXA LIGHT
        8006609: 48,  # PAO DE CAIXA LIGHT (var.)
        8000509: 48,  # PAO DE CAIXA
        8000513: 48,  # PAO PULLMAN
        8000511: 48,  # PAO RECIFE
        8006603: 48,  # PAO PULMEX LIGHT

        # ── PÃO DE AVEIA → Pão, aveia, forma [idx=47, Na=606] ──────────────
        8001701: 47,  # PAO DE AVEIA

        # ── PÃO DE BATATA → Pão, trigo, sovado [idx=53] (proxy) ────────────
        8001001: 53,  # PAO DE BATATA

        # ── TORRADA → Torrada, pão francês [idx=62, Na=829] ─────────────────
        8001901: 62,  # TORRADA DE QUALQUER PAO
        8001902: 62,  # PAO TORRADO (TORRADA)
        8001903: 62,  # TORRADA DE QUALQUER TIPO
        8006901: 62,  # TORRADA DE QUALQUER PAO LIGHT
        8006902: 62,  # PAO TORRADO LIGHT
        8006903: 62,  # TORRADA DE QUALQUER TIPO LIGHT
        8004402: 62,  # PAO DE TORRESMO
        8004401: 62,  # PAO DE CENOURA/ALHO/CEBOLA TORRADO
        8014001: 62,  # BOLACHA DE TRIGO  — proxy torrada

        # ── BISCOITO SALGADO → Biscoito, salgado, cream cracker [idx=12, Na=854] ─
        8002201: 12,  # BISCOITO SALGADO
        8002204: 12,  # BISCOITO CREAM CRACKER
        8002220: 12,  # BISCOITO CLUB CRACKER
        8010104: 12,  # BISCOITO CREAM CRACKER LIGHT
        8010120: 12,  # BISCOITO CLUB CRACKER LIGHT
        8011701: 12,  # BISCOITO SALGADO DIET
        8002232: 12,  # BISCOITO CREAM CRACKER INTEGRAL
        8002209: 12,  # BISCOITO DE COCO SALGADO
        8002215: 12,  # BISCOITO SALGADO DE COCO
        8002219: 12,  # BISCOITO SALGADO DE POLVILHO
        8002212: 12,  # BISCOITO DE POLVILHO SALGADO
        8010112: 12,  # BISCOITO DE POLVILHO SALGADO LIGHT
        8002234: 12,  # BISCOITO SALGADO DE MILHO
        8002235: 12,  # BISCOITO SALGADO DE PIZZA
        8002236: 12,  # BISCOITO SALGADO INTEGRAL
        8002237: 12,  # BOLACHA CREAM CRACK
        8002218: 12,  # BISCOITO SALGADINHO
        8002221: 12,  # SALGADINHO (BISCOITO)
        8002225: 12,  # BOLACHA SALGADA
        8010125: 12,  # BOLACHA SALGADA LIGHT
        8002248: 12,  # BISCOITO SALGADO DE GOMA
        8002249: 12,  # MENTIRA (BISCOITO SALGADO)
        8002250: 12,  # BISCOITO RUFFLES SALGADO
        8002251: 12,  # BISCOITO PALITO SALGADO
        8010101: 12,  # BISCOITO SALGADO LIGHT
        8010118: 12,  # BISCOITO SALGADINHO LIGHT
        8010121: 12,  # SALGADINHO (BISCOITO) LIGHT
        8002227: 12,  # CHIPS (SALGADINHOS)
        8010127: 12,  # CHIPS (SALGADINHOS) LIGHT
        8002228: 12,  # CROCK (SALGADINHOS)
        8002246: 12,  # SALGADINHO DE MILHO
        8002247: 12,  # SALGADINHO DE QUEIJO  — proxy cracker
        8000123: 12,  # PAO VITA SALGADO
        8000130: 12,  # PAO CARTEIRA (SALGADO)  — se existir
        8000131: 53,  # PAO CARTEIRA (SALGADO) → sovado (já acima)
        8000144: 53,  # PAO RECHEADO (SALGADO) → sovado (já acima)
        8002308: 12,  # BISCOITO AMANTEIGADO (salgado)
        8005201: 12,  # BISCOITO NAO-ESPECIFICADO
        8008501: 12,  # BISCOITO NAO-ESPECIFICADO LIGHT
        8011801: 12,  # BISCOITO ORGANICO NAO ESPECIFICADO
        8014201: 49,  # BISCOITO SEM GLUTEN → pão glúten forma [idx=49]
        8013701: 49,  # BISCOITO DOCE SEM GLUTEN
        8013801: 49,  # BOLACHA SEM LACTOSE
        8002410: 60,  # PETA DE GOMA → polvilho [idx=60]
        8002409: 60,  # PETA DE POLVILHO
        8002408: 60,  # PETA
        8002411: 60,  # PETAS

        # ── BISCOITO DOCE (maisena, água, champagne…) → [idx=7, Na=352] ─────
        8002301: 7,   # BISCOITO DOCE
        8002316: 7,   # BISCOITO DE MAIZENA
        8010416: 7,   # BISCOITO DE MAIZENA LIGHT
        8002320: 7,   # SEQUILHO DE MAIZENA
        8002207: 7,   # BISCOITO DE AGUA
        8002231: 7,   # BISCOITO DE AGUA E SAL
        8002241: 7,   # BISCOITO AGUA E SAL
        8002244: 7,   # BOLACHA AGUA E SAL
        8010126: 7,   # BOLACHA DE AGUA E SAL LIGHT
        8002226: 7,   # BOLACHA DE AGUA E SAL
        8002245: 7,   # BOLACHA DE SAL
        8002233: 7,   # BISCOITO DE SAL
        8002303: 7,   # BISCOITO DE MANTEIGA
        8002304: 7,   # BOLACHA AMANTEIGADA
        8002313: 7,   # BOLACHA DE MANTEIGA
        8002324: 7,   # BISCOITO SORTIDO
        8010424: 7,   # BISCOITO SORTIDO LIGHT
        8002321: 7,   # BISCOITO DOCE SORTIDO
        8010421: 7,   # BISCOITO DOCE SORTIDO LIGHT
        8010401: 7,   # BISCOITO DOCE LIGHT
        8010541: 7,   # BISCOITO DOCE DIETETICO
        8010501: 7,   # BISCOITO DOCE DIET
        8008601: 7,   # BISCOITO NAO-ESPECIFICADO DIET
        8008602: 7,   # BISCOITO NAO-ESPECIFICADO DIETETICO
        8002336: 7,   # BISCOITO DE CHAMPAGNE
        8002328: 7,   # BISCOITO CHAMPANHE
        8002350: 7,   # BISCOITO DE NATA
        8002203: 7,   # BISCOITO GRISSINI
        8002202: 7,   # BISCOITO DO REINO
        8002206: 7,   # BISCOITO TIPO ITALIANO
        8010106: 7,   # BISCOITO TIPO ITALIANO LIGHT
        8002224: 7,   # BISCOITO ITALIANO
        8002243: 7,   # BISCOITO CLUBE SOCIAL
        8002242: 7,   # BISCOITO CLUB SOCIAL
        8002210: 7,   # BISCOITO DE CAMARAO
        8002213: 7,   # BISCOITO PRESUNTINHO
        8002205: 7,   # PRESUNTINHO BISCOITO
        8002208: 7,   # BISCOITO QUEIJINHO
        8002223: 7,   # QUEIJINHO BISCOITO
        8002216: 7,   # BISCOITO BIT
        8002239: 7,   # BISCOITO DE CEBOLA
        8002310: 7,   # BISCOITO TOSTINE
        8004811: 7,   # BISCOITO CROCANTE
        8004812: 7,   # BISCOITO MIRABEL
        8004809: 7,   # ALFAJORES (BISCOITO)
        8004804: 7,   # BISCOITO LANCHE MIRABEL
        8002401: 7,   # BISCOITO DE POLVILHO DOCE
        8002407: 7,   # BISCOITO DOCE DE POLVILHO
        8002402: 7,   # BISCOITO DE POLVILHO SEQUILHO
        8002403: 7,   # BISCOITO QUEBRA QUEBRA
        8002404: 7,   # BISCOITO DE ARARUTA
        8002355: 7,   # BISCOITO DE GOMA
        8002307: 7,   # BISCOITO DE GERGELIM
        8002327: 7,   # BISCOITO DE MILHO
        8002318: 7,   # SEQUILHO DE MILHO
        8002325: 7,   # BISCOITO DE COCO DOCE
        8010425: 7,   # BISCOITO DE COCO DOCE LIGHT
        8002339: 7,   # BISCOITO DE COCO
        8002331: 7,   # BISCOITO BROA DE COCO
        8002342: 7,   # BISCOITO DE TAPIOCA
        8002337: 7,   # BISCOITO MARIA
        8010437: 7,   # BISCOITO MARIA LIGHT
        8002348: 7,   # BOLACHA DE MAIZENA
        8002334: 7,   # BOLACHA DOCE
        8010434: 7,   # BOLACHA DOCE LIGHT
        8002335: 7,   # MARIA MALUCA (BOLACHA DOCE)
        8002349: 7,   # BOLACHAO DOCE
        8005301: 7,   # BISCOITO DE POLVILHO NAO-ESPECIFICADO
        8002351: 7,   # CUECA VIRADA (BISCOITO DOCE)
        8002352: 7,   # BISCOITO ORELHA DE GATO
        8002358: 7,   # ORELHA DE GATO (BISCOITO DOCE)
        8002363: 7,   # MENTIRA DOCE
        8002609: 7,   # GRUSTOLI (BOLINHO DOCE)
        8004901: 7,   # BISCOITO CASEIRO
        8004902: 7,   # BOLACHA CASEIRA
        8013601: 7,   # BISCOITO SEM LACTOSE
        8005401: 7,   # BOLACHA NAO-ESPECIFICADA
        8008701: 7,   # BOLACHA NAO-ESPECIFICADA LIGHT

        # ── BISCOITO RECHEADO CHOCOLATE → [idx=8, Na=239] ───────────────────
        8002341: 8,   # BISCOITO DE CHOCOLATE
        8002343: 8,   # BISCOITO ROSQUINHA DE CHOCOLATE
        8002345: 8,   # BOLACHA DE CHOCOLATE
        8002360: 8,   # BISCOITO PALITO DE CHOCOLATE
        8004814: 8,   # BISCOITO RECHEADO DE CHOCOLATE

        # ── BISCOITO RECHEADO MORANGO/GENÉRICO → [idx=9, Na=230] ────────────
        8004801: 9,   # BISCOITO RECHEADO
        8004803: 9,   # BISCOITO RECHEADO TOSTINE
        8008303: 9,   # BISCOITO RECHEADO TOSTINE LIGHT
        8004815: 9,   # BISCOITO RECHEADO DE MORANGO
        8004816: 9,   # CASADINHA (BISCOITO RECHEADO)
        8004817: 9,   # AMANDITA (BISCOITO RECHEADO)
        8004818: 9,   # BEM CASADO (BISCOITO RECHEADO)
        8004819: 9,   # TRAQUINAS (BISCOITO RECHEADO)
        8008401: 9,   # BISCOITO RECHEADO DIET
        8008301: 9,   # BISCOITO RECHEADO LIGHT
        8008414: 9,   # BISCOITO RECHEADO DIETETICO
        8004810: 9,   # BOLACHA RECHEADA
        8008310: 9,   # BOLACHA RECHEADA LIGHT

        # ── WAFER → Biscoito, doce, wafer [idx=11, Na=120] ──────────────────
        8004805: 11,  # BISCOITO WAFFER
        8004807: 11,  # WAFFER (BISCOITO)
        8008305: 11,  # BISCOITO WAFFER LIGHT
        8008405: 11,  # BISCOITO WAFFER DIET

        # ── BISCOITO DE QUEIJO → Pastel, de queijo, frito [idx=57, Na=821] ──
        8002240: 57,  # BISCOITO DE QUEIJO
        8002319: 57,  # BISCOITO DOCE DE QUEIJO
        8010519: 57,  # BISCOITO DOCE DE QUEIJO DIET

        # ── ROSCA / ROSQUINHA → Farinha, de rosca [idx=33, Na=333] ──────────
        8002001: 33,  # ROSCA DOCE
        8002002: 33,  # ROSQUINHA DOCE
        8002003: 33,  # ROSQUINHA DE NATA
        8002004: 33,  # ROSCA DE TRIGO DOCE
        8002005: 33,  # ROSCA DE NATA
        8002006: 33,  # ROSCA DE MILHO
        8002007: 33,  # ROSQUINHA AMANTEIGADA
        8002008: 33,  # ROSQUINHA DE MILHO
        8002009: 33,  # ROSCA DOCE DE TRIGO
        8002010: 33,  # ROSCA DE COCO
        8002011: 33,  # ROSQUINHA DE COCO
        8002013: 33,  # ROSCA AMANTEIGADA
        8002101: 33,  # ROSCA SALGADA
        8002102: 33,  # ROSCA SALGADA DE TRIGO
        8002103: 33,  # ROSCA DE BATATA
        8002104: 33,  # ROSCA DE TRIGO SALGADA
        8002106: 33,  # ROSQUINHA SALGADA
        8005901: 33,  # ROSCA NAO-ESPECIFICADA
        8005902: 33,  # ROSQUINHA NAO-ESPECIFICADA
        8005903: 33,  # ROSCA DE TRIGO NAO-ESPECIFICADA
        8006001: 33,  # ROSCA DE POLVILHO NAO-ESPECIFICADA
        8006002: 33,  # ROSQUINHA DE POLVILHO NAO-ESPECIFICADA
        8012501: 33,  # ROSCA DE LEITE
        8012801: 33,  # ROSCA DE GOMA
        8014101: 33,  # ROSCA DE CACHACA
        8002329: 33,  # BISCOITO ROSQUINHA DE COCO
        8002332: 33,  # BISCOITO ROSCA DE COCO
        8004808: 33,  # ROSQUINHA RECHEADA
        8004813: 33,  # ROSCA RECHEADA
        8008408: 33,  # ROSQUINHA RECHEADA DE QUALQUER SABOR
        8002362: 33,  # CROSTOLI
        8002706: 33,  # BROINHA DE MANDIOCA

        # ── BROA SALGADA → cream cracker [idx=12] ───────────────────────────
        8012201: 12,  # BROA SALGADA
        8003601: 12,  # TORTAS SALGADAS DE QUALQUER SABOR
        8012901: 12,  # BOLO SALGADO
        8005801: 53,  # TORTA NAO-ESPECIFICADA → sovado (proxy)
        8003501: 7,   # TORTAS DOCES → biscoito doce
        8006101: 7,   # CREPE → biscoito doce

        # ── BROA DOCE → biscoito doce [idx=7] ───────────────────────────────
        8012001: 7,   # BROA DOCE
        8012101: 7,   # BROA INTEGRAL
        8002705: 7,   # BROA DE MANDIOCA
        8001104: 7,   # PANHOCA DE MILHO
        8002305: 7,   # SEQUILHO
        8010405: 7,   # SEQUILHO LIGHT
        8002315: 7,   # BREVIDADE
        8004701: 7,   # BROA
        8005001: 7,   # BROINHA
        8012301: 7,   # CUPCAKE
        8002503: 7,   # CUCA DE QUALQUER TIPO
        8011001: 7,   # AVOADOR
        8000150: 7,   # CHIPA
        8000151: 7,   # PAO CHIPA
        8000218: 7,   # CHINEQUE COM FAROFA
        8002356: 7,   # BOLACHA DE MEL
        8002357: 7,   # MARIA MALUCA
        8002361: 7,   # BOLACHA DE GOMA
        8002353: 7,   # BOLACHA PALITO
        8002359: 7,   # BOLACHAO DE COCO
        8002340: 7,   # BOLACHA DE COCO
        8002344: 7,   # BISCOITO ROSQUINHA DE LEITE
        8002346: 7,   # BOLACHA DE LEITE
        8002347: 7,   # BOLACHA DE LEITE E MEL
        8002333: 7,   # BOLACHA COM LEITE
        8010578: 7,   # BISCOITO DE LEITE DIETETICO
        8002338: 7,   # BISCOITO DE LEITE
        8005101: 7,   # BROA DE LEITE
        8002326: 7,   # BISCOITO CREME
        8002706: 33,  # BROINHA DE MANDIOCA → rosca (já acima)

        # ── ALIMENTOS DE SOJA (contexto farinha/massa) → Farinha de trigo [idx=34] ─
        6503602: 34,  # MASSA DE SOJA
        6503603: 34,  # CARNE VEGETAL
        6503604: 34,  # PASTA DE SOJA
        6503605: 34,  # CARNE DE SOJA
        6503606: 34,  # ALIMENTO DE SOJA
        6505601: 34,  # PROTEINA DE SOJA
        6505602: 34,  # COMPLEMENTO VITAMINICO DE SOJA
        6505603: 34,  # PROTEINA VEGETAL
        6501901: 34,  # FARINHA DE SOJA

        # ── PASTA AMERICANA / itens de confeitaria → bolo mistura [idx=13] ──
        8014901: 13,  # PASTA AMERICANA
        8014801: 52,  # PAO FRANCES LIGHT (já acima em pão francês)

        # ── MISCELÂNEOS ──────────────────────────────────────────────────────
        8000508: 50,  # PAO PLUS VITA → forma
        8006608: 50,  # PAO PLUS VITA LIGHT → forma
        8006508: 50,  # PAO PLUS VITA DIET → forma
        8000502: 50,  # PAO SEVEN BOYS → forma
        8004802: 7,   # BOLACHA NAO-ESPECIFICADA → biscoito doce
        8003003: 31,  # BROA PRETA → centeio
        8003001: 31,  # BOLO DE CENTEIO → centeio
        8003002: 31,  # BROA DE CENTEIO → centeio
        8003004: 61,  # BROA PRETA → polenta (proxy)
        8000240: 53,  # CROISSANT → sovado
        8000203: 53,  # CHINEQUE → sovado (já em doce acima; manter sovado)
        8004806: 52,  # LANCHE MIRABEL → pão francês
        8002506: 56,  # BOLO DE QUEIJO → pastel queijo cru
        8002507: 56,  # BOLINHO DE QUEIJO → pastel queijo cru
        8010701: 56,  # BROA DE QUEIJO → pastel queijo cru
        8011901: 56,  # BOLINHO DE QUALQUER MARCA E SABOR → pastel queijo (proxy)
        8002502: 7,   # ROCAMBOLE → biscoito doce
        8010202: 7,   # ROCAMBOLE LIGHT
        8002504: 17,  # FATIA DE BOLO → bolo milho (já acima)
        8002505: 7,   # BOLO DE OVOS → biscoito doce
        8000800: 52,  # PAO FRANCES (reserva)
        8009901: 48,  # PAO ARABE LIGHT (já acima)
        6500800: 27,  # CREMOGEMA (reserva)
        6503500: 32,  # PURE DE BATATA EM CAIXA → farinha mandioca (proxy neutro)
        6503501: 32,  # PURE DE BATATA EM CAIXA
        8012601: 44,  # BATATA CHIPS → Milho, verde, enlatado [idx=44] — fora do pool ideal
        8002230: 12,  # BACONZITOS → biscoito salgado
        6500800: 27,  # (reserva)
        8002409: 60,  # PETA DE POLVILHO (já acima)
        8000211: 53,  # PAO TRANCA COM FAROFA → sovado
        8010011: 53,  # PAO TRANCA COM FAROFA DIET
        8003702: 34,  # FILHOS (BOLINHO DE FARINHA DE TRIGO) → farinha trigo
        8003903: 53,  # PAO DE MASSA FOLHADA → sovado
        8003901: 53,  # PAO BRIOCHE → sovado
        8000120: 53,  # PAO PROVENCO → sovado
        8001303: 53,  # PAO CRIOULO (já acima)
        8002707: 7,   # BROA DE GOMA → biscoito doce
        8002609: 7,   # GRUSTOLI (já acima)
        6502202: 39,  # MISTURA PARA PAO DE TRIGO (já acima)
        },
        'sanidade':    {
        'energia_kcal':  (300, 380),
        'proteina_g':    (7,   14),
        'carboidrato_g': (65,  85),
        'lipideos_g':    (0,    5),
        'fibra_g':       (1,    6),
        'sodio_mg':      (0,  300),
        'ferro_mg':      (0.5,  5),
        'calcio_mg':     (2,   30),
        'zinco_mg':      (0.5,  3),
        'vitaminaC_mg':  (0,    5),
        },
    },
    '03.Tuberculos': {
        'dict_manual': {

        # ── MANDIOCA / MACAXEIRA / AIPIM → Mandioca, crua [idx=129, Na=2] ──
        6400601: 129,  # MANDIOCA
        6400602: 129,  # MANDIOCA DOCE
        6400603: 129,  # MANDIOCA MANSA
        6400604: 129,  # MANDIOCA BRANCA
        6400605: 129,  # MANDIOCA CACAU
        6400613: 129,  # MANDIOCA SEM CASCA
        6402301: 129,  # MANDIOCA ORGANICA
        8500703: 129,  # MANDIOCA CONGELADA PARA VIAGEM
        6400609: 129,  # AIPIM  — aipim = mandioca (não aipo)
        6402401: 129,  # AIPIM ORGANICO
        6400612: 129,  # AIMPIM
        6400610: 129,  # MACAXEIRA  — macaxeira = mandioca
        6400614: 129,  # MASSA DE MACAXEIRA  — proxy mandioca crua (Na=2 vs 773)
        6904504: 129,  # MASSA DE BEIJU → Mandioca, crua
        6904502: 129,  # BEIJU → Mandioca, crua
        6904508: 129,  # TAPIOCA DE QUEIJO → Mandioca, crua

        # ── BATATA INGLESA → Batata, inglesa, crua [idx=91, Na=0] ───────────
        6400101: 91,   # BATATA INGLESA
        6400102: 91,   # BATATA DO REINO
        6400103: 91,   # BATATA ROSA
        6400104: 91,   # BATATA PORTUGUESA
        6400105: 91,   # BATATA INGESA ROSA
        6400106: 91,   # BATATINHA
        6400107: 91,   # BATATA LISA  — batata lisa = batata inglesa (não alface)
        6400108: 91,   # BATATA HOLANDESA
        6400109: 91,   # BATATA BINGE
        6400110: 91,   # BATATA BRANCA
        6400406: 91,   # BATATA ROXA  — batata roxa = inglesa roxa (não alface)
        6400801: 91,   # BATATA NAO-ESPECIFICADA
        6400802: 91,   # BATATA (NAO-ESPECIFICADA)
        6401801: 91,   # BATATA INGLESA ORGANICA
        6401806: 91,   # BATATINHA ORGANICA

        # ── BATATA DOCE → Batata, doce, crua [idx=88, Na=9] ─────────────────
        6400401: 88,   # BATATA DOCE
        6400402: 88,   # BATATA DOCE ROXA
        6400403: 88,   # BATATA DA TERRA
        6400405: 88,   # BATATA MASTRUS
        6402201: 88,   # BATATA DOCE ORGANICA
        6402701: 88,   # BATATA YACON  — proxy batata doce (tubérculo adocicado)

        # ── BATATA BAROA / MANDIOQUINHA → Batata, baroa, crua [idx=86, Na=0] ─
        6400301: 86,   # BATATA BAROA
        6400302: 86,   # BATATA BARONESA
        6400303: 86,   # MANDIOQUINHA SALSA (BATATA BAROA)
        6400903: 86,   # BATATA DO BARAO
        6400908: 86,   # MANDIOQUINHA  — mandioquinha = batata baroa (não farinha)
        6402801: 86,   # MANDIOQUINHA PALHA  — idem
        6400904: 86,   # BATATA SUICA  — proxy batata baroa
        6400901: 86,   # BATATA AIPO  — aipo do Peru = mandioquinha/batata baroa
        6400907: 86,   # AIPO DO PERU (BATATA AIPO)  — idem

        # ── INHAME → Inhame, cru [idx=125, Na=0] ────────────────────────────
        6400501: 125,  # INHAME
        6400503: 125,  # INHAME SAO TOME
        6400504: 125,  # INHAME CHINES
        6400505: 125,  # INHAME DA COSTA
        6400507: 125,  # TAIOBA SAO TOME INHAME  — função de inhame neste grupo
        6400702: 125,  # CARA INHAME
        6400714: 125,  # INHAME LISO (CARA)
        6402501: 125,  # INHAME ORGANICO

        # ── CARÁ → Cará, cru [idx=102, Na=0] ────────────────────────────────
        6400701: 102,  # CARA
        6400703: 102,  # CARA CHINES
        6400704: 102,  # CARA BRANCO  — cará branco (não repolho)
        6400707: 102,  # CARA DO AR
        6400713: 102,  # INHAME CARAQUENTO (CARA)

        # ── ARARUTA / AÇAFRÃO / GENGIBRE → Mandioca, crua [idx=129] (proxy neutro) ─
        6400201: 129,  # ARARUTA  — amido de tubérculo; proxy mandioca
        6401601: 129,  # ACAFRAO  — rizoma; proxy mandioca
        6401701: 129,  # GENGIBRE  — rizoma; proxy mandioca (Na=2 vs 104 do purê)

        # ── GOBO → Batata, baroa, crua [idx=86] (raiz/tubérculo, proxy mais próximo) ─
        6401501: 86,   # GOBO  — bardana, raiz asiática
        },
        'sanidade':    {
        'energia_kcal':  (60,  120),
        'proteina_g':    (0.5,   3),
        'carboidrato_g': (12,   30),
        'lipideos_g':    (0,     2),
        'fibra_g':       (0.5,   4),
        'sodio_mg':      (0,    50),
        'ferro_mg':      (0.1,   2),
        'calcio_mg':     (5,    50),
        'zinco_mg':      (0.1,   1),
        'vitaminaC_mg':  (5,    30),
        },
    },
    '04.Acucares_Industrializados': {
        'dict_manual': {

        # ── SAL → proxy correto (idx fora do pool restrito mas no TACO) ─────
        7000101: 516,  # SAL REFINADO → Sal, grosso (Na real)
        7000201: 516,  # SAL GROSSO
        7000202: 516,  # SAL TRITURADO
        7000203: 516,  # SAL MOIDO
        7000204: 516,  # SAL GROSSO TEMPERADO
        7011001: 516,  # SAL REFINADO LIGHT
        7011002: 516,  # SAL IODADO LIGHT
        7000102: 516,  # SAL IODADO
        7013601: 516,  # SAL ROSA
        7000301: 516,  # SAL AMONIACO
        7008501: 516,  # SAL AMARGO
        7013501: 516,  # SAL MARINHO
        7005701: 515,  # SAL DIET → Sal, dietético
        7005702: 515,  # SAL DIETETICO
        7005703: 515,  # SAL IODADO DIET

        # ── GLUTAMATO / REALÇADORES → Tempero a base de sal [518] ───────────
        7000701: 518,  # GLUTAMATO MONOSSODICO
        7000702: 518,  # AJINOMOTO
        7000703: 518,  # REALCADOR DE SABOR

        # ── TEMPEROS INDUSTRIALIZADOS → Tempero a base de sal [518] ─────────
        7011801: 518,  # TEMPERO SAZON
        7011802: 518,  # SAZON (TEMPERO)
        7011803: 518,  # SAZON
        7000406: 518,  # TEMPERO ARISCO
        7000405: 518,  # ARISCO
        7000401: 518,  # TEMPERO MISTO INDUSTRIALIZADO EM PASTA
        7000402: 518,  # CONDIMENTO MISTO INDUSTRIALIZADO EM PASTA
        7000412: 518,  # TEMPERO ALHO E SAL
        7000413: 518,  # TEMPERO COMPLETO EM PASTA
        7000410: 518,  # CREME DE ALHO
        7000411: 518,  # MASSA DE ALHO
        7000407: 518,  # ALHO EM PASTA
        7000409: 518,  # PURE DE ALHO
        7005302: 518,  # TEMPERO MISTO EM PO
        7005304: 518,  # TEMPERO MISTO EM GRAO
        7005305: 518,  # PIMENTA SIRIA
        7008301: 518,  # TEMPERO EM PO NAO-ESPECIFICADO
        7008401: 518,  # TEMPERO NAO-ESPECIFICADO
        7008402: 518,  # CONDIMENTO NAO-ESPECIFICADO
        7008403: 518,  # TEMPERO CASEIRO NAO-ESPECIFICADO
        7009802: 518,  # TEMPERO DE FRANGO
        7005601: 518,  # TEMPERO DE MASSAS
        7005801: 518,  # TEMPERO DE FEIJAO
        7004001: 518,  # TEMPERO DE SALADAS
        7013901: 518,  # TEMPERO DE ERVAS FINAS
        7013101: 518,  # LEMON PEPPER
        7012101: 518,  # TEMPERO NAO-ESPECIFICADO EM PACOTE
        7012201: 518,  # TEMPERO MISTO LIQUIDO
        7012202: 518,  # TEMPERO LIQUIDO MISTO
        7012203: 518,  # TEMPERO LIQUIDO DE CEBOLA E ALHO
        7012204: 518,  # TEMPERO LIQUIDO COMPLETO
        7007602: 518,  # TEMPERO NAO-ESPECIFICADO EM TABLETE
        7013801: 518,  # FURIKAKE (TEMPERO JAPONES)
        7014001: 518,  # WASABI (TEMPERO JAPONES)
        7010101: 518,  # MOLHO TEMPERADO PARMESAO

        # ── ERVAS SECAS INDUSTRIALIZADAS → Tempero a base de sal [518] ──────
        7002201: 518,  # OREGANO
        7002202: 518,  # OREGAO
        7002103: 518,  # MANGERICAO DE MOLHO
        7002105: 518,  # MANGERICAO (TEMPERO INDUSTRIALIZADO)
        7002601: 518,  # COENTRO (TEMPERO INDUSTRIALIZADO)
        7002602: 518,  # COENTRO EM PO
        7002604: 518,  # COENTRO EM GRAO
        7002603: 518,  # SEMENTE DE COENTRO
        7002301: 518,  # HORTELA
        7001101: 518,  # CRAVO (TEMPERO INDUSTRIALIZADO)
        7001103: 518,  # CRAVO DA INDIA
        7001401: 518,  # NOZ MOSCADA
        7001501: 518,  # MOSTARDA (TEMPERO INDUSTRIALIZADO)
        7001601: 518,  # LOURO (TEMPERO INDUSTRIALIZADO)
        7001602: 518,  # LOURO EM PO
        7001603: 518,  # LOURO EM FOLHA
        7001604: 518,  # FOLHA DE LOURO
        7001001: 518,  # CANELA EM PO
        7001002: 518,  # CANELA EM PAU
        7001003: 518,  # CANELA EM FOLHA
        7001701: 518,  # ACAFRAO EM PO
        7001801: 518,  # COMINHO EM GRAO
        7001802: 518,  # CUMINHO EM GRAO
        7005201: 518,  # COMINHO EM PO
        7005202: 518,  # CUMINHO EM PO
        7003101: 518,  # PIMENTA DO REINO EM PO OU EM GRAO
        7003102: 518,  # PIMENTA DO REINO EM PO
        7003103: 518,  # PIMENTA DO REINO EM GRAO
        7003104: 518,  # PIMENTA DO REINO CLARA
        7003105: 518,  # PIMENTA DA INDIA
        7003106: 518,  # PIMENTA DO REINO BRANCA
        7003107: 518,  # PIMENTA DO REINO PRETA
        7003108: 518,  # PIMENTA DO REINO ESCURA
        7003109: 518,  # PIMENTA DO REINO
        7003110: 518,  # MIX DE PIMENTA DO REINO
        7003201: 518,  # PIMENTA DO REINO E COMINHO
        7003202: 518,  # COMINHO E PIMENTA DO REINO
        7006101: 518,  # PIMENTA EM PO
        7011701: 518,  # PIMENTA CALABRESA
        7001301: 518,  # PAPRIKA
        7001205: 518,  # COLORAU COM PIMENTA
        7012601: 518,  # ERVA DOCE (TEMPERO INDUSTRIALIZADO)
        7008601: 518,  # SALVIA EM FOLHA
        7008701: 518,  # TOMILHO EM FOLHA
        7002901: 518,  # CHEIRO VERDE EM PO
        7002701: 518,  # ALHO EM PO
        7002702: 518,  # ALHO DESIDRATADO
        7002802: 518,  # CEBOLA DESIDRATADA
        7002803: 518,  # CEBOLA EM FLOCOS
        7002504: 518,  # MANJERONA DESIDRATADA
        7002501: 518,  # MANJERONA SECA
        7009101: 518,  # GERGELIM EM PO
        7009102: 518,  # GERGELIM EM GRAO
        7006901: 518,  # SALSA DESIDRATADA
        7013201: 518,  # ALECRIM DESIDRATADO
        7013401: 518,  # ZAATAR
        7013001: 518,  # VINAGRE DE CAMARAO (condimento)

        # ── COLORAU / COLORÍFICO / URUCUM → Tempero a base de sal [518] ─────
        7001201: 518,  # COLORAU
        7001202: 518,  # COLORIFICO
        7001203: 518,  # URUCU
        7001204: 518,  # URUCUM

        # ── FERMENTO → proxies corretos ──────────────────────────────────────
        7000601: 512,  # FERMENTO EM PO → Fermento em pó, químico [Na=10052]
        7000602: 512,  # FERMENTO (em pó)
        7011601: 512,  # EMUSTAB (emulsificante) — proxy fermento em pó
        7000603: 513,  # FERMENTO BIOLOGICO → Fermento, biológico [Na=40]
        7000501: 516,  # BICARBONATO DE SODIO → Sal, grosso (Na alto)
        7000502: 512,  # BICARBONATO → Fermento em pó (uso culinário)
        7014101: 512,  # BICARBONATO DE AMONIO
        7011501: 512,  # COAGULANTE LIQUIDO (CLORETO DE MAGNESIO)
        7008202: 512,  # COAGULANTE LIQUIDO
        7008201: 512,  # COALHO

        # ── SHOYU → proxy correto [517] ──────────────────────────────────────
        7003604: 517,  # SHOYO
        7003607: 517,  # MOLHO SHOYU
        7003601: 517,  # MOLHO DE SOJA
        7003603: 517,  # MOLHO JAPONES
        7006801: 517,  # MISSO (pasta fermentada de soja)

        # ── MAIONESE → proxy correto [523] ───────────────────────────────────
        7004301: 523,  # MAIONESE
        7004302: 523,  # MAIONESE COM LIMAO
        7004303: 523,  # MAIONESE COM ATUM
        7004304: 523,  # MAIONESE COM AZEITONA
        7004305: 523,  # MOLHO DE MAIONESE
        7004306: 523,  # MAIONESE DE QUALQUER SABOR
        7004307: 523,  # MAIONESE DE LEITE
        7010401: 523,  # MAIONESE LIGHT

        # ── AZEITONA / CONSERVAS → proxies corretos ──────────────────────────
        7707101: 519,  # PASTA DE AZEITONA → Azeitona, preta [519]
        7707102: 520,  # PATE DE AZEITONA → Azeitona, verde [520]
        7002401: 520,  # ALCAPARRA EM CONSERVA → Azeitona, verde

        # ── MOLHO DE TOMATE / KETCHUP → Tomate, molho industrializado [158] ──
        7004801: 158,  # MOLHO DE TOMATE
        7004802: 158,  # KETCHUP
        7004803: 158,  # CATCHUP
        7004804: 158,  # CAT-CHUP
        7004805: 158,  # KAT-CUP
        7004806: 158,  # MOLHO REFOGADO DE TOMATE
        7004807: 158,  # KATCHUP
        7010902: 158,  # KETCHUP LIGHT
        7010901: 158,  # MOLHO DE TOMATE LIGHT
        7010907: 158,  # KETCHUP DIET
        7002001: 158,  # MOLHO DE TOMATE COM ERVAS
        7004903: 158,  # MOLHO DE TOMATE COM CARNE
        7004904: 158,  # MOLHO DE MACARRONADA
        7007001: 158,  # MOLHO DE PIZZA
        7007002: 158,  # MOLHO PARA PIZZA
        7004701: 158,  # MASSA DE TOMATE
        7004702: 158,  # EXTRATO DE TOMATE
        7004705: 158,  # POLPA DE TOMATE
        7010301: 158,  # TOMATE SECO

        # ── MOLHOS VARIADOS → Glicose de milho [502, Na=59] (proxy neutro) ──
        7009201: 502,  # MOLHO DE PIMENTA
        7009701: 502,  # MOLHO BARBECUE
        7009901: 502,  # MOLHO PARA CARNE
        7009801: 502,  # MOLHO DE FRANGO
        7009301: 502,  # MOLHO MADEIRA
        7009501: 502,  # MOLHO AGRIDOCE
        7009401: 502,  # MOLHO PARA SALADA
        7010702: 502,  # MOLHO DE SALADAS LIGHT
        7003701: 502,  # MOLHO INGLES
        7003901: 502,  # MOLHO TARTARO
        7012801: 502,  # MOLHO CHIMICHURRI
        7012901: 502,  # MOLHO PESTO
        7013301: 502,  # MOLHO VINAGRETE
        7005901: 502,  # AMACIANTE DE CARNE
        7007201: 502,  # MOLHO BRANCO
        7007503: 502,  # MOLHO QUATRO QUEIJOS
        7007504: 502,  # PATE DE QUEIJO
        7007505: 502,  # MOLHO TRES QUEIJOS
        7007502: 502,  # CREME DE QUEIJO
        7005101: 502,  # MOLHO XADREZ
        7014301: 502,  # MOLHO DE CENOURA
        7014401: 502,  # MOLHO DE CHEDDAR
        7014501: 502,  # MOLHO DE ALMEIRAO
        7014601: 502,  # MOLHO DE ALHO PORO ORGANICO
        7014701: 502,  # MOLHO DE RUCULA ORGANICO
        7014801: 502,  # MOLHO DE PEQUI
        7014901: 502,  # MOLHO DE CEBOLA ORGANICA
        7015001: 502,  # MOLHO DE SALSA
        7015101: 502,  # MOLHO DE ESPINAFRE
        7008001: 502,  # MOLHO NAO-ESPECIFICADO
        7006401: 502,  # MOLHO DE FUNGHI
        7006402: 502,  # MOLHO DE COGUMELOS
        7013701: 502,  # MOLHO DE CHAMPIGNON
        7008101: 502,  # MOLHO DE ALHO
        7002107: 502,  # MOLHO DE MANGERICAO

        # ── CALDOS → proxy tablete [323=carne, 324=galinha] ──────────────────
        7004501: 323,  # CALDO DE CARNE → Caldo de carne, tablete [Na=22180]
        7004401: 324,  # CALDO DE GALINHA → Caldo de galinha, tablete [Na=22300]
        7007301: 324,  # CALDO DE BACON → proxy galinha
        7007601: 323,  # CALDO NAO-ESPECIFICADO → proxy carne
        7005401: 323,  # CALDO DE LEGUMES → proxy carne (Na alto)
        7011101: 323,  # CALDO DE CEBOLA
        7011201: 323,  # CALDO DE COSTELA
        7007701: 323,  # CALDO DE PEIXE
        7007801: 323,  # CALDO DE CAMARAO
        7011301: 323,  # CALDO DE MANDIOCA
        7012001: 323,  # CALDO DE ARROZ
        7004502: 323,  # CALDO DE PICANHA
        7005802: 323,  # CALDO DE FEIJOADA KNORR
        7007101: 323,  # TUCUPI EM CALDO SEM PIMENTA
        7007102: 323,  # CALDO DE TUCUPI SEM PIMENTA
        7004101: 158,  # MOLHO DE TUCUPI COM PIMENTA → molho tomate proxy
        7004103: 158,  # TUCUPI COM PIMENTA

        # ── VINAGRE → Melado [507, Na=4] — Na baixo, proxy mais neutro ───────
        7003301: 507,  # VINAGRE DE ALCOOL
        7003302: 507,  # VINAGRE DE CANA
        7003401: 507,  # VINAGRE DE VINHO
        7003402: 507,  # VINAGRE DE UVA
        7003501: 507,  # VINAGRE DE MACA
        7008901: 507,  # VINAGRE DE ARROZ
        7012301: 507,  # VINAGRE DE LIMAO
        7012401: 507,  # VINAGRE BALSAMICO
        7012701: 507,  # VINAGRE DE MACA ORGANICO
        7011901: 507,  # VINAGRE DE ALCOOL E VINHO (COMPOSTO)
        7011902: 507,  # AGRIN
        7008801: 507,  # VINAGRE NAO-ESPECIFICADO
        7009001: 507,  # VINAGRE DE MILHO

        # ── AÇÚCAR → proxies corretos ─────────────────────────────────────────
        6900201: 491,  # ACUCAR CRISTAL
        6900203: 491,  # ACUCAR CRISTALIZADO
        6908703: 491,  # ACUCAR CRISTALIZADO ORGANICO
        6908701: 491,  # ACUCAR CRISTAL ORGANICO
        6906602: 491,  # ACUCAR
        6906603: 491,  # ACUCAR COMUM
        6906601: 491,  # ACUCAR INDETERMINADO
        6908901: 491,  # ACUCAR INDETERMINADO ORGANICO
        6908902: 491,  # ACUCAR ORGANICO
        6908903: 491,  # ACUCAR COMUM ORGANICO
        6906801: 491,  # ACUCAR TRITURADO
        6906802: 491,  # ACUCAR MOIDO
        6900202: 491,  # ACUCAR GRANULADO
        6900102: 491,  # ACUCAR TRIFILTRADO
        6902005: 491,  # ACUCAR DE CONFEITEIRO
        6906701: 491,  # ACUCAR DE BAUNILHA
        6915001: 491,  # ACUCAR DIET
        6907001: 491,  # ACUCAR LIGHT
        6907002: 491,  # ACUCAR REFINADO LIGHT
        6914901: 491,  # ACUCAR DE COCO
        6900101: 493,  # ACUCAR REFINADO → Açúcar, refinado [493]
        6908601: 493,  # ACUCAR REFINADO ORGANICO
        6900300: 493,  # (reserva)
        6900301: 491,  # ACUCAR DEMERARA → cristal (proxy mais próximo)
        6908801: 491,  # ACUCAR DEMERARA ORGANICO
        6900304: 492,  # ACUCAR MASCAVO → Açúcar, mascavo [492]
        6900306: 492,  # ACUCAR MASCAVADO
        6908804: 492,  # ACUCAR MASCAVO ORGANICO
        6908806: 492,  # ACUCAR MASCAVADO ORGANICO
        6908805: 492,  # ACUCAR PRETO ORGANICO
        6900200: 491,  # (reserva)
        6900204: 491,  # ACUCAR GROSSO

        # ── RAPADURA → Rapadura [509] ─────────────────────────────────────────
        6900401: 509,  # RAPADURA
        6900403: 509,  # BATIDA (RAPADURA)
        6900404: 509,  # RAPADURA COM COCO
        6900405: 509,  # RAPADURA COM LEITE
        6900406: 509,  # RAPADURA COM AMENDOIM
        6900407: 509,  # RAPADURA COM CASTANHA

        # ── MEL → Mel, de abelha [506] ────────────────────────────────────────
        6901601: 506,  # MEL DE ABELHA
        6901602: 506,  # MEL
        6901604: 506,  # MEL COM PROPOLIS
        6901605: 506,  # MEL COMPOSTO
        6909001: 506,  # MEL ORGANICO
        6909002: 506,  # MEL DE ABELHA ORGANICO
        6905301: 506,  # ADOCANTE NATURAL CONCENTRADO
        6905302: 506,  # ADOCANTE CONCENTRADO NATURAL
        6906101: 506,  # PROPOLIS → mel proxy

        # ── MELADO → Melado [507] ─────────────────────────────────────────────
        6901501: 507,  # MELADO
        6901502: 507,  # MELADO DE CANA

        # ── GLICOSE DE MILHO → [502] ──────────────────────────────────────────
        6901901: 502,  # GLICOSE DE MILHO
        6901903: 502,  # KARO

        # ── CHOCOLATE → proxies corretos ─────────────────────────────────────
        6900701: 494,  # TABLETE DE CHOCOLATE
        6900702: 494,  # BARRA DE CHOCOLATE
        6900703: 494,  # CHOCOLATE EM TABLETE
        6900704: 494,  # CHOCOLATE EM BARRA
        6900705: 494,  # CHOCOLATE BISS (TABLETE)
        6900707: 494,  # CHOCOLATE BATOM
        6900708: 494,  # BATON CHOCOLATE
        6900709: 494,  # TUBETE DE CHOCOLATE
        6900710: 494,  # PASTILHA DE CHOCOLATE
        6905601: 494,  # CHOCOLATE EM CREME
        6905602: 494,  # NUCITA
        6905603: 494,  # IOIO CREME
        6905604: 494,  # CREME DE CHOCOLATE
        6905605: 494,  # CREME DE CHOCOLATE
        6903602: 494,  # DOCE DE CHOCOLATE
        6904307: 494,  # DOCE DE LEITE COM CHOCOLATE
        6917401: 494,  # ROLINHO BANANA E CHOCOLATE
        6917601: 494,  # FRUTAS COBERTAS COM CHOCOLATE
        6905201: 494,  # CHOCOLATE GRANULADO
        6905801: 494,  # OVO DE PASCOA
        6905802: 494,  # COELHINHO DA PASCOA
        6905803: 494,  # CHOCOLATE COELHINHO DA PASCOA
        6905805: 494,  # PAPAI NOEL DE CHOCOLATE
        6910003: 494,  # CHOCOLATE EM TABLETE LIGHT
        6912101: 494,  # CHOCOLATE EM CREME LIGHT
        6910101: 496,  # TABLETE DE CHOCOLATE DIET → dietético [496]
        6910102: 496,  # BARRA DE CHOCOLATE DIET
        6910103: 496,  # CHOCOLATE EM TABLETE DIET
        6910104: 496,  # CHOCOLATE EM BARRA DIET
        6910108: 496,  # BARRA DE CHOCOLATE DIETETICO
        6911801: 496,  # CHOCOLATE GRANULADO DIET
        6912201: 496,  # CHOCOLATE EM CREME DIET
        6916301: 496,  # PETIT GATEAU → chocolate proxy
        6916601: 496,  # ALFARROBA → substituto chocolate
        6900241: 497,  # (reserva meio amargo)
        6905903: 494,  # MANDOLATE

        # ── ACHOCOLATADO / NESCAU / TODDY → [490] ────────────────────────────
        6900819: 490,  # PO ACHOCOLATADO
        6900821: 490,  # ACHOCOLATADO EM PO
        6910221: 490,  # ACHOCOLATADO EM PO LIGHT
        6910321: 490,  # ACHOCOLATADO EM PO DIET
        6916501: 490,  # ACHOCOLATADO EM CAPSULA
        6900801: 490,  # CHOCOLATE EM PO DE QUALQUER MARCA
        6900803: 490,  # CHOCOLATE NESCAU
        6900805: 490,  # CHOCOLATE TODDY
        6900806: 490,  # CHOCOLATE EM PO TODDY
        6900807: 490,  # CHOCOLATE EM PO NESCAU
        6900810: 490,  # TODDY VITAMINADO
        6900811: 490,  # TODDY
        6900812: 490,  # NESCAU
        6900813: 490,  # NESCAU VITAMINADO
        6900815: 490,  # NESCAU INSTANTANEO
        6900818: 490,  # OVOMALTINE
        6900820: 490,  # PO PARA MILK SHAKE
        6900822: 490,  # TODDYNHO EM PO
        6900823: 490,  # NESQUIK
        6910201: 490,  # CHOCOLATE EM PO LIGHT
        6910203: 490,  # CHOCOLATE NESCAU LIGHT
        6910206: 490,  # CHOCOLATE EM PO TODDY LIGHT
        6910207: 490,  # CHOCOLATE EM PO NESCAU LIGHT
        6910210: 490,  # (reserva)
        6910211: 490,  # TODDY LIGHT
        6910212: 490,  # NESCAU LIGHT
        6910301: 490,  # CHOCOLATE EM PO DIET
        6910306: 490,  # CHOCOLATE EM PO TODDY DIET
        6900808: 490,  # TODDY INSTANTANEO
        6907507: 490,  # (reserva)

        # ── DOCE DE LEITE → [500] ────────────────────────────────────────────
        6904301: 500,  # DOCE DE LEITE
        6911301: 500,  # DOCE DE LEITE DIET
        6915102: 500,  # DOCE DE LEITE ZERO LACTOSE
        6904303: 500,  # DOCE DE LEITE COM AMEIXA
        6904304: 500,  # DOCE DE LEITE COM AMENDOIM
        6904305: 500,  # DOCE DE LEITE COM COCO
        6904308: 500,  # DOCE DE LEITE COM BANANA
        6904309: 500,  # DOCE DE LEITE COM GOIABA
        6904310: 500,  # DOCE DE LEITE COM MAMAO
        6904201: 500,  # DOCE A BASE DE LEITE
        6904204: 500,  # LEITE GELEIFICADO
        6902701: 500,  # MANJAR → doce de leite proxy

        # ── DOCE DE FRUTAS EM PASTA/BARRA → Marmelada [505] ──────────────────
        6901201: 505,  # DOCE DE FRUTAS EM PASTA
        6901207: 505,  # DOCE DE FRUTAS EM BARRA OU TIJOLO
        6901209: 505,  # FIGADA
        6901210: 505,  # BANANADA (EM BARRA OU TIJOLO)
        6901211: 505,  # GOIABADA
        6901212: 505,  # DOCE DE ABOBORA EM PASTA
        6901213: 505,  # DOCE DE MORANGO EM PASTA
        6901216: 505,  # DOCE DE BATATA MARROM GLACE
        6901217: 505,  # DOCE DE AMEIXA EM PASTA
        6901219: 505,  # DOCE DE CACAU EM PASTA
        6901220: 505,  # PASTA DE CACAU
        6901222: 505,  # DOCE DE FRUTAS EM TABLETE
        6901223: 505,  # DOCE DE CAJU EM PASTA OU BARRA
        6901224: 505,  # MARIOLA
        6901225: 505,  # DOCE DE FEIJAO EM PASTA
        6901228: 505,  # DOCE DE MAMAO EM PASTA
        6901230: 505,  # DOCE DE MILHO VERDE EM PASTA
        6901231: 505,  # DOCE DE BANANA EM PASTA OU BARRA
        6901232: 505,  # DOCE DE COCO EM PASTA OU BARRA
        6901233: 505,  # DOCE DE BURITI EM PASTA OU BARRA
        6901234: 505,  # DOCE DE CAJU EM BARRA
        6901235: 505,  # DOCE DE GOIABA EM PASTA OU BARRA
        6901236: 505,  # DOCE DE UVA EM PASTA OU BARRA
        6901237: 505,  # DOCE DE ABOBORA COM COCO
        6901239: 505,  # PATE DE FRUTAS
        6901240: 505,  # BANANADA
        6901206: 505,  # MARMELADA
        6903401: 499,  # DOCE DE ABOBORA → Doce de abóbora [499]
        6915301: 505,  # DOCE DE FRUTAS NAO ESPECIFICADO
        6907701: 505,  # DOCE NAO-ESPECIFICADO
        6907702: 505,  # DOCE
        6907301: 521,  # SOBREMESA → Chantilly spray [521]

        # ── DOCE DE FRUTAS EM CALDA → Marmelada [505] ────────────────────────
        6901301: 505,  # DOCE DE FRUTAS EM CALDA
        6901302: 505,  # FIGO EM CALDA
        6901303: 505,  # AMEIXA EM CALDA
        6901304: 505,  # PESSEGO EM CALDA
        6901305: 505,  # SALADA DE FRUTAS EM CALDA
        6901306: 505,  # ABACAXI EM CALDA
        6901308: 505,  # GOIABA EM CALDA
        6901310: 505,  # CEREJA EM CALDA
        6901312: 505,  # DOCE DE JACA EM CALDA
        6901313: 505,  # DOCE DE PESSEGO EM CALDA

        # ── DOCE CRISTALIZADO → Açúcar, cristal [491] ────────────────────────
        6901401: 491,  # DOCE DE FRUTAS CRISTALIZADO
        6901402: 491,  # DOCE DE ABACAXI CRISTALIZADO
        6901403: 491,  # DOCE DE LARANJA CRISTALIZADO
        6901404: 491,  # DOCE DE FIGO CRISTALIZADO
        6901405: 491,  # DOCE DE GOIABA CRISTALIZADO
        6901406: 491,  # DOCE DE CAJU CRISTALIZADO
        6901407: 491,  # DOCE DE BANANA CRISTALIZADO
        6901408: 491,  # FRUTAS CRISTALIZADAS

        # ── GELEIA → Geléia, mocotó [501] ────────────────────────────────────
        6901001: 501,  # GELEIA DE FRUTAS
        6901002: 501,  # MOUSSE (GELEIA)
        6901003: 501,  # MOUSSE DE QUALQUER SABOR (GELEIA)
        6901004: 501,  # GELEIA DE DAMASCO
        6901101: 501,  # GELEIA DE MOCOTO
        6906201: 501,  # GELEIA REAL
        6907901: 501,  # GELEIA DE FRUTAS DIET
        6907904: 501,  # GELEIA DIET
        6907905: 501,  # GELEIA DE FRUTAS DIET
        6907906: 501,  # GELEIA DE FRUTAS DIETETICA
        6907909: 501,  # GELEIA DIETETICA
        6910601: 501,  # GELEIA DE FRUTAS LIGHT
        6910604: 501,  # GELEIA LIGHT

        # ── GELATINA → [514] ─────────────────────────────────────────────────
        6901701: 514,  # GELATINA
        6901702: 514,  # GELATINA EM PO
        6901703: 514,  # GELATINA EM FOLHA
        6908101: 514,  # GELATINA DIET
        6908102: 514,  # GELATINA DIETETICA
        6913901: 514,  # GELATINA LIGHT
        6913902: 514,  # GELATINA EM PO LIGHT

        # ── SORVETE / PICOLÉ → Doce de leite, cremoso [500] ─────────────────
        6900501: 500,  # SORVETE DE QUALQUER SABOR INDUSTRIALIZADO
        6900502: 500,  # PICOLE DE QUALQUER SABOR INDUSTRIALIZADO
        6900503: 500,  # MASSA DE SORVETE
        6900504: 500,  # SORVETE INDUSTRIALIZADO
        6900505: 500,  # PICOLE INDUSTRIALIZADO
        6900506: 500,  # CREMOSINHO (SORVETE)
        6900507: 500,  # SORVETE ARTESANAL
        6904802: 500,  # DINDIM (PICOLE ENSACADO)
        6904803: 500,  # SACOLE
        6904804: 500,  # CHUP-CHUP
        6904805: 500,  # PICOLE CASEIRO
        6904806: 500,  # SORVETE CASEIRO
        6904807: 500,  # GELADINHO
        6909701: 500,  # SORVETE INDUSTRIALIZADO DIET
        6909704: 500,  # SORVETE INDUSTRIALIZADO DIET
        6909705: 500,  # PICOLE INDUSTRIALIZADO DIET
        6909706: 500,  # SORVETE DIETETICO
        6911407: 500,  # GELADINHO LIGHT
        6915201: 500,  # SORVETE SEM LACTOSE
        6914801: 500,  # CASQUINHA DE SORVETE
        6916401: 500,  # CHEESECAKE → doce de leite proxy

        # ── PUDIM / MOUSSE / FLAN → Quindim [508] ou Doce de leite [500] ─────
        6902601: 500,  # PUDIM DE QUALQUER SABOR
        6902602: 508,  # PUDIM DE COCO → Quindim [508]
        6902603: 508,  # PUDIM DE QUEIJO
        6902604: 500,  # PUDIM DE LEITE CONDENSADO
        6902606: 500,  # PUDIM DE LARANJA
        6902607: 500,  # DANETTE PUDIM
        6902608: 500,  # PUDIM DANETTE
        6910909: 500,  # PUDIM DE QUALQUER SABOR DIETETICO
        6902501: 500,  # SOBREMESA INFANTIL
        6902502: 500,  # DOCE INFANTIL EM POTE
        6916101: 500,  # MOUSSE DE QUALQUER SABOR
        6902810: 514,  # MOUSSE EM PO → gelatina proxy
        6902802: 514,  # FLAN EM PO
        6902804: 514,  # PO DE FLAN
        6902813: 514,  # PO PARA PUDIM
        6902811: 514,  # PO PARA SORVETE
        6902809: 514,  # PO PARA MINGAU NAO-ESPECIFICADO
        6902807: 514,  # MASSA PARA MINGAU NAO-ESPECIFICADA
        6912702: 500,  # SOBREMESA DIETETICA
        6902803: 500,  # PO DE DOCE, SORVETE E PUDIM
        6902801: 500,  # MISTURA INDUSTRIAL DE DOCE, SORVETE E PUDIM
        6904205: 500,  # CHANDELE

        # ── COCADA / PAÇOCA / PÉ DE MOLEQUE → Cocada branca [498] ───────────
        6903101: 498,  # COCADA
        6913001: 498,  # COCADA DIET
        6913003: 498,  # COCADA DIETETICA
        6903201: 498,  # DOCE DE AMENDOIM
        6903203: 498,  # PACOCA
        6913103: 498,  # PACOCA DIET
        6903205: 498,  # PACOQUINHA DE AMENDOIM
        6913105: 498,  # PACOQUINHA DE AMENDOIM DIET
        6903202: 498,  # PE DE MOLEQUE
        6913102: 498,  # PE DE MOLEQUE DIET
        6903210: 498,  # PE DE MOCA
        6903206: 498,  # PACOCA DE CASTANHA DE CAJU
        6903207: 498,  # AMENDOIM CARAMELIZADO
        6903209: 498,  # AMENDOIM ACHOCOLATADO
        6903302: 498,  # AMENDOIM TORRADO (doce)
        6903303: 498,  # AMENDOIM APIMENTADO
        6903304: 498,  # AMENDOIM COZIDO
        6903305: 498,  # AMENDOIM DOCE
        6903306: 498,  # AMENDOIM JAPONES
        6903301: 498,  # AMENDOIM SALGADO → cocada proxy
        6903401: 499,  # DOCE DE ABOBORA (já acima)
        6916701: 498,  # BOMBOCADO

        # ── MARIA MOLE / MARSHMALLOW → [503] ─────────────────────────────────
        6903001: 503,  # MARIA MOLE
        6902209: 503,  # MARSHMALLOW

        # ── QUINDIM / DOCE DE COCO → [508] ───────────────────────────────────
        6904105: 508,  # QUINDIM
        6904107: 508,  # PAPO DE ANJO
        6904108: 508,  # FIOS DE OVOS
        6917201: 508,  # TOUCINHO DO CEU
        6903801: 508,  # MIL FOLHAS → quindim proxy (ovos+açúcar)
        6917001: 508,  # PASTEL DE BELEM
        6903901: 508,  # QUEIJADINHA
        6903902: 508,  # QUEIJADA
        6916801: 503,  # BEIJINHO → maria mole proxy
        6903102: 498,  # QUEBRA QUEIXO → cocada proxy
        6904401: 491,  # CANUDINHO RECHEADO → açúcar cristal proxy
        6904402: 491,  # CANUDINHO PARA RECHEAR

        # ── BRIGADEIRO → Chocolate ao leite [494] ────────────────────────────
        6903601: 494,  # BRIGADEIRO
        6913401: 494,  # BRIGADEIRO DIET
        6903603: 494,  # BOLINHO DE BRIGADEIRO
        6905604: 494,  # CREME DE BRIGADEIRO (já acima)

        # ── BOMBOM / TRUFA → Chocolate ao leite [494] ────────────────────────
        6900901: 494,  # BOMBOM DE QUALQUER MARCA
        6900902: 494,  # BOMBOM CARAMELIZADO
        6900903: 494,  # BOMBONS SORTIDOS
        6900904: 494,  # TRUFA
        6910401: 494,  # BOMBOM LIGHT
        6910402: 494,  # BOMBOM CARAMELIZADO LIGHT
        6910403: 494,  # BOMBONS SORTIDOS LIGHT
        6910501: 494,  # BOMBOM DIET
        6910504: 494,  # TRUFA DIET
        6905804: 494,  # KINDER OVO
        6912304: 494,  # KINDER OVO LIGHT
        6917301: 494,  # NHA BENTA
        6905905: 494,  # (reserva)
        6905901: 494,  # TORRONE
        6905902: 494,  # DOCE TORRONE
        6905903: 494,  # MANDOLATE (já acima)
        6916901: 494,  # DRAGEADO

        # ── BALAS / CHICLETE / PASTILHA → Maria mole [503, Na=15] ────────────
        6900601: 503,  # CHICLETE
        6900602: 503,  # MENTEX
        6900603: 503,  # BALA
        6900604: 503,  # CARAMELO (BALA)
        6900605: 503,  # DROPS
        6900606: 503,  # PASTILHA
        6900607: 503,  # PIRULITO
        6900608: 503,  # CHICLE
        6900609: 503,  # GOMA DE MASCAR
        6900610: 503,  # JUJUBA
        6900611: 503,  # BALAS
        6900612: 503,  # BALAS SORTIDAS
        6900613: 503,  # BALINHA
        6900614: 503,  # BALA DE GOMA
        6900706: 503,  # CONFETE
        6902206: 503,  # CONFEITO DE MENTA
        6902201: 503,  # CONFEITOS DE BOLOS E DOCES
        6902202: 503,  # CORANTES DE BOLOS E DOCES
        6902203: 503,  # CONFEITO DE CHOCOLATE
        6902204: 503,  # CONFEITO DE AMENDOIM
        6902205: 503,  # CONFEITO NAO-ESPECIFICADO
        6902208: 503,  # CORANTE DE BOLO E DOCE
        6902003: 491,  # GLACE → açúcar proxy
        6902002: 491,  # GLACÊ (reserva)
        6902004: 503,  # MERENGUE → maria mole proxy
        6908501: 503,  # SUSPIRO → maria mole proxy
        6909801: 503,  # CHICLETE LIGHT
        6909809: 503,  # GOMA DE MASCAR LIGHT
        6909901: 503,  # CHICLETE DIET
        6909903: 503,  # BALA DIET
        6909906: 503,  # PASTILHA DIET
        6909908: 503,  # CHICLE DIET
        6909909: 503,  # GOMA DE MASCAR DIET
        6909911: 503,  # BALAS DIET
        6909914: 503,  # CHICLETE DIETETICO
        6909919: 503,  # PASTILHA DIETETICA
        6909925: 503,  # BALAS SORTIDAS DIETETICAS
        6915801: 503,  # BALA SEM GLUTEN
        6909803: 503,  # BALA LIGHT
        6909811: 503,  # BALAS LIGHT

        # ── ADOÇANTE → Açúcar, cristal [491] proxy neutro ────────────────────
        6906902: 491,  # ADOCANTE
        6906901: 491,  # ADOCANTE INDETERMINADO
        6906903: 491,  # ADOCANTE EM PO
        6906904: 491,  # ADOCANTE LIQUIDO
        6907101: 491,  # ADOCANTE DIET
        6907102: 491,  # ADOCANTE DIETETICO
        6907104: 491,  # ADOCANTE LIQUIDO DIET
        6907106: 491,  # ADOCANTE LIQUIDO DIETETICO
        6914001: 491,  # ADOCANTE LIGHT
        6914002: 491,  # ADOCANTE EM PO LIGHT
        6901804: 491,  # SACARINA
        6901805: 491,  # ZERO CAL
        6901806: 491,  # ADOCANTE CULINARIO
        6901801: 491,  # ADOCANTE ARTIFICIAL

        # ── LEITE DE COCO → [522] ────────────────────────────────────────────
        7003801: 522,  # LEITE DE COCO
        7010601: 522,  # LEITE DE COCO LIGHT
        7011401: 522,  # COCO EM FLOCOS
        7011402: 522,  # FLOCOS DE COCO
        6906404: 522,  # POLPA DE COCO
        6904509: 522,  # BEIJU DE COCO (se remanescente)

        # ── BARRA DE CEREAIS → Achocolatado pó [490] proxy ───────────────────
        6907201: 490,  # BARRA DE CEREAIS
        6907203: 490,  # BARRA DE CEREAIS DOCE
        6907204: 490,  # BARRA DE CEREAL
        6914101: 490,  # BARRA DE CEREAIS LIGHT
        6912501: 490,  # BARRA DE CEREAIS DIET
        6912503: 490,  # BARRA DE CEREAIS DIETETICA

        # ── CAPUCCINO / PÓ PARA BEBIDA → Capuccino, pó [511] ────────────────
        6902102: 511,  # MISTURA DE CAFE → capuccino proxy
        6902101: 511,  # CORANTE DE CAFE
        6900809: 511,  # QUICK → achocolatado/capuccino proxy

        # ── ESSÊNCIAS / BAUNILHA → proxy neutro Mel [506] ────────────────────
        7000901: 506,  # BAUNILHA EM PO
        7000902: 506,  # ESSENCIA DE BAUNILHA
        7000904: 506,  # ESSENCIA DE CEREJA

        # ── POLPAS DE FRUTA (limítrofes — ficam em col_4) → Marmelada [505] ──
        6906401: 505,  # POLPA DE FRUTA CONGELADA
        6906402: 505,  # MARACUJA EM POLPA
        6906403: 505,  # POLPA DE MARACUJA
        6906404: 522,  # POLPA DE COCO → leite de coco
        6906406: 505,  # POLPA DE ABACAXI
        6906407: 505,  # POLPA DE CAJU
        6906408: 505,  # POLPA DE CUPUACU
        6906409: 505,  # POLPA DE GRAVIOLA
        6906411: 505,  # POLPA DE TANGERINA
        6906412: 505,  # POLPA DE MANGA
        6906413: 505,  # POLPA DE CAJA
        6906414: 505,  # POLPA DE GOIABA
        6906415: 505,  # POLPA DE PITANGA
        6906416: 505,  # POLPA DE MORANGO
        6906417: 505,  # POLPA DE UVA
        6906418: 505,  # POLPA DE ACEROLA
        6906419: 505,  # POLPA DE FRUTA DE MANGABA
        6906420: 505,  # POLPA DE TAMARINDO
        6906421: 505,  # POLPA DE BURITI
        6906422: 505,  # POLPA DE BACURI
        6906423: 505,  # POLPA DE CARAMBOLA
        6906424: 505,  # POLPA DE LIMAO
        6906425: 505,  # POLPA DE LARANJA
        6906426: 505,  # POLPA DE MAMAO
        6906427: 505,  # POLPA DE MURICI
        6906428: 505,  # POLPA DE AMEIXA
        6906429: 505,  # POLPA DE CIRIGUELA
        6906430: 505,  # POLPA DE CACAU

        # ── FRUTAS SECAS / DESIDRATADAS (limítrofes) → Marmelada [505] ───────
        6902401: 505,  # FRUTA SECA OU DESIDRATADA
        6902402: 505,  # PASSA
        6902403: 505,  # AMEIXA SECA OU DESIDRATADA
        6902404: 505,  # TAMARA SECA OU DESIDRATADA
        6902405: 505,  # MACA SECA OU DESIDRATADA
        6902406: 505,  # DAMASCO SECO OU DESIDRATADO
        6902407: 505,  # BANANA SECA OU DESIDRATADA
        6902408: 505,  # PESSEGO SECO OU DESIDRATADO
        6902409: 505,  # FIGO SECO OU DESIDRATADO
        6902411: 505,  # FRUTA DESIDRATADA
        6902412: 505,  # AMEIXA DESIDRATADA
        6902413: 505,  # TAMARA DESIDRATADA
        6902414: 505,  # MACA DESIDRATADA
        6902415: 505,  # DAMASCO DESIDRATADO
        6902419: 505,  # ABACAXI DESIDRATADO
        6902420: 505,  # BANANA PASSA
        6902421: 505,  # CAJU DESIDRATADO
        6906502: 505,  # (reserva)

        # ── CREME DE AVELÃ / PASTA DE AMENDOIM → Cocada branca [498] ─────────
        6914401: 498,  # CREME DE AVELA (Nutella)
        6901203: 498,  # PASTA DE AMENDOIM
        6901205: 498,  # CREME DE AMENDOIM
        6916201: 498,  # CREME DE CASTANHA
        6901220: 505,  # PASTA DE CACAU → marmelada proxy (já acima)

        # ── PAMONHA / TAPIOCA remanescentes (se ainda em col_4) → [502] ──────
        6905101: 502,  # PAMONHA (se não foi para col_3)
        6917501: 502,  # MISTURA PARA SAGU INDUSTRIALIZADA
        6902302: 505,  # SCHIMIER DE COLONIA → marmelada proxy
        6904601: 505,  # SCHIMIER DE FRUTA → marmelada proxy
        6915401: 505,  # DOCE DE CANELA → marmelada proxy
        6914301: 498,  # DOCE DE GERGELIM → cocada proxy
        6914201: 491,  # ARROZ CARAMELIZADO → açúcar proxy
        6915501: 505,  # DOCE DE SAGU → marmelada proxy
        6917701: 503,  # DOCE PUXA-PUXA → maria mole proxy
        6905401: 503,  # ALGODAO-DOCE → maria mole proxy
        6916501: 490,  # ACHOCOLATADO EM CAPSULA (já acima)
        6904202: 508,  # PAVE → quindim proxy
        6904204: 500,  # LEITE GELEIFICADO (já acima)
        6905605: 494,  # CREME DE BRIGADEIRO (já acima)
        6904104: 508,  # CACAROLA ITALIANA → quindim proxy
        6908401: 508,  # PASTEIS DE SANTA CLARA → quindim proxy
        6917101: 508,  # TARTELETE → quindim proxy
        6903801: 508,  # MIL FOLHAS (já acima)
        6905702: 506,  # GUARANA EM PO NATURAL → mel proxy
        6905701: 506,  # GUARANA EM PO → mel proxy
        6904203: 500,  # AMBROSIA → doce de leite
        6902812: 500,  # BASE PARA SORVETE → doce de leite
        6905001: 503,  # CHURRO → maria mole

        # ── MISCELÂNEOS ───────────────────────────────────────────────────────
        6902208: 503,  # CORANTE DE BOLO E DOCE (já acima)
        7006001: 518,  # FONDOR → tempero sal proxy
        7006501: 518,  # ERVILHA FRITA EM PO → tempero proxy
        6902102: 511,  # MISTURA DE CAFE (já acima)
        6915601: 506,  # MACA DO AMOR → mel proxy
        6902302: 505,  # SCHIMIER (já acima)
        6917201: 508,  # TOUCINHO DO CEU (já acima)
        6904302: 500,  # MUMU → doce de leite proxy
        7708501: 498,  # TAHINI (se remanescente) → cocada proxy
        7708801: 498,  # PASTA DE GRAO DE BICO (se remanescente)
        6900800: 490,  # (reserva achocolatado)
        },
        'sanidade':    {
        'energia_kcal':  (100, 450),
        'proteina_g':    (0,    15),
        'carboidrato_g': (20,   95),
        'lipideos_g':    (0,    20),
        'fibra_g':       (0,     5),
        'sodio_mg':      (0,  5000),  # amplo: inclui sal e temperos com Na real
        'ferro_mg':      (0,     5),
        'calcio_mg':     (0,   150),
        'zinco_mg':      (0,     3),
        'vitaminaC_mg':  (0,    20),
        },
    },
    '05.Leguminosas_Oleaginosas': {
        'dict_manual': {

        # Amendoa → Amêndoa torrada salgada (único proxy disponível)
        6600501: 586,

        # Avelã → Noz, crua [596] — proxy mais próximo
        6600601: 596,

        # Quinoa → Lentilha, crua [577] — proxy disponível
        6304901: 577,

        # Massa de buriti → Coco, cru [589]
        6602102: 589,

        # Feijão verde (todas variantes) → Feijão, carioca, cru [561]
        6301634: 561,  # FEIJAO VERDE
        6301627: 561,  # FEIJAO SEMPRE VERDE
        6304034: 561,  # FEIJAO VERDE ORGANICO
        6302104: 561,  # FEIJAO VERMELHO E BRANCO

        # Tahini → Gergelim, semente [592]
        7708501: 592,
        # Pasta de grão de bico (se remanescente)
        7708801: 574,  # Grão-de-bico, cru [574]
        },
        'sanidade':    {
        'energia_kcal':  (100, 400),
        'proteina_g':    (5,   30),
        'carboidrato_g': (10,  65),
        'lipideos_g':    (0,   30),
        'fibra_g':       (2,   20),
        'sodio_mg':      (0,  200),
        'ferro_mg':      (1,   10),
        'calcio_mg':     (30, 200),
        'zinco_mg':      (0.5,  5),
        'vitaminaC_mg':  (0,   10),
        },
    },
    '06.Frutas': {
        'dict_manual': {
        # Dendê/patauá → proxy neutro
        6601301: 185,  # COCO DENDE → Caju, cru
        6601303: 185,  # DENDE → Caju, cru
        6602602: 167,  # EMULSAO DE PATAUA → Açaí polpa

        # Cana de açúcar → proxy neutro
        6802501: 234,  # CANA DE ACUCAR → Melancia

        # Guavira → proxy cerrado
        6811501: 190,  # GUAVIRA → Ciriguela

        # Goiaba in natura
        6804201: 199,  # GOIABA → Goiaba, vermelha
        6809601: 199,  # GOIABA ORGANICA → Goiaba, vermelha

        # Coco ralado → proxy neutro
        6600201: 177,  # COCO RALADO INDUSTRIALIZADO → Banana maçã
        6600202: 177,  # COCO RALADO NATURAL → Banana maçã
        6600105: 167,  # COCO DA BAHIA VERDE → Açaí polpa

        # Manga → proxy correto
        6803207: 183,  # MANGA ROSINHA → Cajá-Manga
        6803210: 183,  # MANGA CARLOTINHA → Cajá-Manga
        6803203: 183,  # MANGA ESPADINHA → Cajá-Manga

        # Pokan = tangerina
        6802216: 250,  # POKAN → Tangerina Poncã

        # Maçã verde
        6803005: 220,  # MACA VERDE → Maçã Argentina
        6808605: 220,  # MACA VERDE ORGANICA → Maçã Argentina

        # Uva verde
        6803907: 255,  # UVA VERDE → Uva Itália

        # Frutas amazônicas
        6601707: 167,  # JUCARA → Açaí polpa
        6602104: 167,  # BURITI (COCO) → Açaí polpa
        6602302: 167,  # VINHO DE BURITI → Açaí polpa
        6601716: 167,  # VINHO DE ACAI → Açaí polpa
        6601711: 167,  # ACAI FRUTA → Açaí polpa

        # Framboesa/Rambutan/Cranberry/Goji → proxies vermelhos
        6812001: 238,  # FRAMBOESA → Morango
        6812101: 202,  # RAMBUTAN → Jabuticaba
        6811201: 238,  # CRANBERRY → Morango
        6811801: 238,  # GOJI BERRY → Morango

        # Macadamia → Caju
        6801601: 185,  # MACADAMIA → Caju, cru

        # Ananás = abacaxi
        6802602: 163,  # ANANAS → Abacaxi

        # Bacuri → Cupuaçu
        6806501: 191,  # BACURI → Cupuaçu

        # Araçá → Goiaba branca
        6805701: 196,  # ARACA → Goiaba branca

        # Ingá → Ciriguela
        6806601: 190,  # INGA → Ciriguela
        6806602: 190,  # INGA CIPO → Ciriguela

        # Mamão verde = in natura
        6803104: 225,  # MAMAO VERDE → Mamão Papaia, cru

        # Juçara → Açaí
        6601707: 167,  # JUCARA → Açaí polpa
        },
        'sanidade':    {
        'energia_kcal':  (30,  100),
        'proteina_g':    (0,     3),
        'carboidrato_g': (5,    25),
        'lipideos_g':    (0,     3),
        'fibra_g':       (0.5,   5),
        'sodio_mg':      (0,    20),
        'ferro_mg':      (0,     2),
        'calcio_mg':     (5,    50),
        'zinco_mg':      (0,     1),
        'vitaminaC_mg':  (10,  100),
        },
    },
    '07.Legumes_Verduras': {
        'dict_manual': {
        # Tomate in natura → Tomate, com semente, cru [156, Na=1]
        6705101: 156, 6705108: 156, 6705110: 156, 6705103: 156,
        6705105: 156, 6705104: 156, 6705107: 156, 6707901: 156,
        6707904: 156,
        6708201: 116,  # COUVE ORGANICA → Couve-flor, crua (corrigido)

        # Milho em conserva → Milho verde, cru [43, Na=1]
        7700401: 43, 7700402: 43,

        # Ervilha em conserva → Ervilha em vagem [558, Na=0]
        7700201: 558, 7700302: 109, 7700301: 109, 7700202: 141,

        # Azeitonas → Pepino, cru [141, Na=0]
        7700102: 141, 7700104: 141, 7700107: 141, 7700105: 141,
        7700103: 141, 7707401: 141, 7700101: 141, 7700106: 141,

        # Conservas diversas
        7707001: 153, 6706101: 153, 6706102: 153,
        7708001: 153,

        # Agregados → Tomate salada [160]
        9000904: 160,

        # Pasta de aspargos → Abobrinha italiana crua [70]
        7708101: 70,

        # Pasta de tomate seco → Tomate, com semente [156]
        7708201: 156,

        # Pasta de alho em conserva → Alho, cru [81]
        7707801: 81,

        # Cogumelo desidratado → Abobrinha [70]
        7708701: 70, 6705301: 70,

        # Conservas vegetais → proxies in natura
        7707901: 112,  # CHUCHU EM CONSERVA → Chuchu, cru
        7707201: 116,  # COUVE FLOR EM CONSERVA → Couve-flor, crua
        7707601: 100,  # BROCOLIS EM CONSERVA → Brócolis, cru
        7708601: 70,   # PASTA NAO ESPECIFICADA → Abobrinha
        7707501: 161,  # SALSICHA VEGETAL EM CONSERVA → Vagem, crua
        7707301: 107,  # CEBOLINHA CRISTAL EM CONSERVA → Cebolinha crua

        # Ervas/plantas medicinais → Mostarda folha [134]
        6712901: 134, 6713601: 134, 6713401: 134, 6713501: 134,
        6710801: 134, 6714301: 134, 6710701: 134, 6714401: 134,

        # Oregano/hortelã em molho → Manjericão [132]
        6709701: 132, 6708301: 132, 6710101: 134,

        # Coentro in natura → Cebolinha crua [107]
        6700401: 107, 6700402: 107, 6711901: 107,

        # Jerimum → Abóbora moranga [66]
        6703805: 66, 6703808: 66, 6703806: 66, 6703807: 70,

        # Outros proxies
        6702801: 70,   # ERVA DOCE → Abobrinha
        6713901: 70,   # EUCALIPTO → Abobrinha
        6702205: 153,  # JOAO GOMES → Seleta
        6704804: 142, 6704602: 142, 6704605: 142,  # Pimentas cumari → amarelo
        6712201: 142,  # COGUMELO SHITAKE → pimentão amarelo
        6703405: 66,   # ABOBORA VERMELHA → moranga
        6400304: 86,   # CENOURA AMARELA (BATATA BAROA) → batata baroa
        6714601: 70,   # CASCARA SAGRADA → abobrinha
        6702511: 100,  # AMARANTO → Brócolis
        6700102: 72,   # ALFACE PAULISTA → Abobrinha paulista
        6712501: 78,   # MELISSA → Alface lisa
        6701003: 152,  # SALSINHA VERDE → Salsa crua
        6714701: 116,  # FLOR COMESTIVEL → Couve-flor
        6707301: 124, 6712401: 124,  # Brotos → Feijão broto
        6714001: 70,   # NIGAGORI → Abobrinha
        6701907: 65,   # VINAGREIRA → Abóbora menina
        6705601: 141, 6705603: 143,  # Cabaça → pepino/pimentão
        6713701: 152,  # BALSAMO → Salsa
        6709601: 107,  # CAPIM SANTO → Cebolinha
        6701906: 79,   # CUXA → Alface roxa
        6712701: 143,  # PIMENTA AMERICANA → Pimentão verde
        6705106: 143,  # TOMATE VERDE → Pimentão verde
        6702105: 74,   # JAMBU → Agrião
        6705301: 70,   # COGUMELO IN NATURA → Abobrinha
        },
        'sanidade':    {
        'energia_kcal':  (10,   80),
        'proteina_g':    (0,     5),
        'carboidrato_g': (1,    15),
        'lipideos_g':    (0,     3),
        'fibra_g':       (0.5,   5),
        'sodio_mg':      (0,    60),  # ajustado — conservas residuais; critério: proxy in natura para vegetais drenados
        'ferro_mg':      (0,     3),
        'calcio_mg':     (10,  150),
        'zinco_mg':      (0,     1),
        'vitaminaC_mg':  (5,    80),
        },
    },
    '08.Carnes': {
        'dict_manual': {
        7109101: 333,  # CARNE BOVINA NAO-ESPECIFICADA
        7101601: 333,  # CARNE BOVINA DE PRIMEIRA
        7800201: 401,  # FRANGO CONGELADO
        7800101: 401,  # FRANGO ABATIDO
        7803301: 488,  # OVO DE GALINHA
        7803302: 488,  # OVOS DE GALINHA
        7101301: 347,  # COSTELA BOVINA
        7101701: 326,  # CARNE BOVINA DE SEGUNDA
        7800103: 401,  # FRANGO INTEIRO
        7800403: 406,  # PEITO DE FRANGO
        7800102: 402,  # GALINHA ABATIDA
        7100201: 331,  # CONTRAFILE
        7104102: 426,  # CARNE DE PORCO NAO ESPECIFICADA
        7101703: 347,  # CARNE BOVINA COM OSSO NAO-ESPECIFICADA
        8102205: 420,  # LINGUICA CALABRESA
        7600901: 301,  # PEIXE NAO-ESPECIFICADO
        7108902: 333,  # CARNE EM BIFE
        7101501: 326,  # CARNE MOIDA DE SEGUNDA
        7800104: 403,  # FRANGO CAIPIRA
        8102601: 423,  # MORTADELA
        8102901: 437,  # PRESUNTO DE QUALQUER TIPO
        7100501: 370,  # PATINHO
        7100404: 351,  # COXAO MOLE
        7101401: 326,  # CARNE MOIDA DE PRIMEIRA
        7100301: 368,  # ALCATRA
        8100104: 338,  # CHARQUE
        7800405: 406,  # FILE DE PEITO DE FRANGO
        7100302: 368,  # ALCATRA BOVINA
        8102204: 417,  # LINGUICA NAO-ESPECIFICADA
        8102101: 433,  # SALSICHA NO VAREJO
        7803304: 488,  # OVOS DE GALINHA BRANCO
        7103401: 430,  # PERNIL SUINO
        7102501: 354,  # FIGADO BOVINO
        7800209: 401,  # FRANGO INTEIRO CONGELADO
        8100102: 338,  # CARNE DE CHARQUE
        7103305: 426,  # BISTECA SUINA
        8102209: 420,  # LINGUICA SUINA
        7803303: 488,  # OVO DE GALINHA CAIPIRA
        7112601: 379,  # PICANHA
        7101602: 333,  # CARNE BOVINA DE PRIMEIRA COM OSSO
        8102701: 432,  # SALAME
        8102208: 417,  # LINGUICA DE FRANGO
        7800506: 408,  # COXA E SOBRECOXA DE FRANGO
        7260101: 282,  # CAMARAO
        7800504: 408,  # COXA DE FRANGO
        7100402: 351,  # COXAO MOLE (CHA DE DENTRO)
        8102102: 433,  # SALSICHA EM PACOTE
        8101005: 413,  # BACON
        7800703: 390,  # ASA DE FRANGO
        7100708: 349,  # COXAO DURO
        7101202: 357,  # FRALDINHA (CAPA DE FILE)
        7101708: 326,  # CARNE BOVINA DE SEGUNDA COM OSSO
        8100101: 338,  # CARNE SECA
        7422901: 301,  # PEIXE TAMBAQUI (FORMA DE COMERCIALIZACAO NAO-DISCRIMINADA)
        7703002: 318,  # SARDINHA EM CONSERVA
        7103502: 427,  # COSTELINHA SUINA
        8102902: 437,  # PRESUNTO DE PERU
        7112501: 362,  # MAMINHA
        7702801: 433,  # SALSICHA EM CONSERVA
        7102602: 353,  # CUPIM BOVINO
        8102603: 423,  # MORTADELA DE FRANGO
        8102103: 433,  # SALSICHAO NO VAREJO
        8102904: 437,  # PRESUNTO BOVINO
        7800505: 408,  # SOBRECOXA DE FRANGO
        7100305: 368,  # MIOLO DE ALCATRA
        7102301: 332,  # BUCHO BOVINO
        8102104: 433,  # SALSICHAO EM PACOTE
        7703402: 276,  # ATUM EM CONSERVA
        7101204: 331,  # CAPA DE CONTRAFILE
        7103101: 363,  # MOCOTO BOVINO
        8102611: 423,  # MORTADELA DEFUMADA
        8101006: 413,  # BACON DEFUMADO
        7102601: 353,  # CUPIM
        7800709: 390,  # COXA DE ASA DE FRANGO
        7703403: 276,  # ATUM RALADO
        7100706: 349,  # COXAO DURO (LAGARTO COMUM)
        8102703: 432,  # SALAME DEFUMADO
        8102903: 437,  # PRESUNTO DE FRANGO
        8102607: 423,  # MORTADELA BOLONHESA
        7705102: 437,  # PRESUNTO DE CHESTER
        7260120: 282,  # CAMARAO FRESCO
        8102604: 423,  # MORTADELA SUINA
        8102605: 423,  # MORTADELA FATIADA NAO-ESPECIFICADA
        7703404: 276,  # ATUM SOLIDO
        7100303: 368,  # PONTA DE ALCATRA
        8101002: 413,  # BACON NO VAREJO
        7101402: 326,  # GUIZADO (CARNE MOIDA DE PRIMEIRA)
        7800207: 401,  # FRANGO CONGELADO TEMPERADO
        7704301: 423,  # MORTADELA EM CONSERVA
        8102602: 423,  # MORTADELA BOVINA
        7101502: 326,  # GUIZADO (CARNE MOIDA DE SEGUNDA)
        7100304: 368,  # ALCATRA COM OSSO
        7703001: 318,  # PEIXE SARDINHA EM CONSERVA
        7100504: 370,  # PATINHO COM OSSO
        7100203: 331,  # CHULETA COM OSSO (CONTRAFILE)
        7801302: 399,  # FIGADO DE FRANGO
        7806501: 488,  # OVO DE GALINHA ORGANICO
        7106501: 368,  # ALCATRA SUINA
        7260118: 282,  # CAMARAO DE AGUA SALGADA
        8102612: 423,  # MORTADELA MISTA
        7260401: 282,  # CAMARAO SECO
        7260201: 282,  # CAMARAO SEM CASCA
        7260402: 282,  # CAMARAO SALGADO
        7702401: 437,  # PATE DE PRESUNTO EM CONSERVA
        7100208: 331,  # ANCHO (CONTRAFILE DIANTEIRO)
        8102213: 420,  # LINGUICA SUINA DEFUMADA
        7703401: 276,  # PEIXE ATUM EM CONSERVA
        7260110: 282,  # CAMARAO SETE BARBAS
        7260117: 282,  # CAMARAO CINZA
        8002230: 413,  # BACONZITOS
        8106201: 437,  # PRESUNTO SUINO LIGHT
        7111504: 379,  # PICANHA ORGANICA
        7707701: 276,  # PATE DE ATUM EM CONSERVA
        8101007: 413,  # RETALHO DE BACON
        8102217: 338,  # LINGUICA DE CHARQUE
        8100403: 347,  # COSTELA BOVINA SALGADA
        7107401: 332,  # BUCHO NAO-ESPECIFICADO
        7800705: 390,  # ASA DE FRANGO TEMPERADO
        7805503: 402,  # GALINHA ABATIDA ORGANICA
        7113001: 379,  # PICANHA SUINA
        7706304: 276,  # ATUM SOLIDO LIGHT
        8101008: 413,  # BACON EM RETALHO
        7100901: 370,  # PA
        8103201: 437,  # PATE DE PRESUNTO EMBUTIDO
        8105001: 423,  # MORTADELA LIGHT
        7803401: 488,  # OVO DE PATA
        7806801: 406,  # PEITO DE FRANGO LIGHT
        7260106: 282,  # CAMARAO ROSA
        7111401: 331,  # CONTRAFILE ORGANICO
        8106101: 437,  # PRESUNTO DE PERU LIGHT
        7115001: 363,  # MOCOTO NAO ESPECIFICADO
        7805504: 403,  # FRANGO CAIPIRA ORGANICO
        7802502: 390,  # COXA E ASA DE FRANGO
        7105203: 363,  # MOCOTO SUINO
        8102606: 423,  # RETALHO DE MORTADELA
        7112001: 333,  # CARNE BOVINA DE PRIMEIRA ORGANICA
        8106102: 437,  # PRESUNTO DE PERU DEFUMADO LIGHT
        7802602: 406,  # COXA E PEITO DE FRANGO
        7706502: 437,  # PRESUNTO DE CHESTER LIGHT
        7805801: 408,  # COXA E SOBRECOXA DE FRANGO ORGANICA
        7706302: 276,  # ATUM EM CONSERVA LIGHT
        8105202: 433,  # SALSICHA EM PACOTE LIGHT
        7706303: 276,  # ATUM RALADO LIGHT
        7805502: 401,  # FRANGO ABATIDO ORGANICO
        7112502: 362,  # CHAPEU DE BISPO (MAMINHA)
        8105201: 433,  # SALSICHA NO VAREJO LIGHT
        8000125: 437,  # PAO PRESUNTO
        7706301: 276,  # PEIXE ATUM EM CONSERVA LIGHT
        8102608: 423,  # MORTADELA DE CHESTER
        7112002: 333,  # CARNE BOVINA DE PRIMEIRA COM OSSO ORGANICA
        7113801: 357,  # FRALDINHA SUINA
        7805804: 408,  # COXA DE FRANGO ORGANICA
        8108001: 282,  # CAMARAO EMPANADO SEMIPRONTO
        7104401: 332,  # BUCHO SUINO
        8107001: 437,  # RETALHO DE PRESUNTO
        8105307: 417,  # LINGUICA DE FRANGO LIGHT
        8105004: 423,  # MORTADELA SUINA LIGHT
        7260601: 282,  # CAMARAO DEFUMADO
        8102609: 423,  # MORTADELA DE GALINHA
        7113901: 370,  # PATINHO SUINO
        7706202: 318,  # SARDINHA EM CONSERVA LIGHT
        7707501: 433,  # SALSICHA VEGETAL EM CONSERVA
        8102610: 423,  # MORTADELA DE PERU
        7013001: 282,  # VINAGRE DE CAMARAO
        8105003: 423,  # MORTADELA DE FRANGO LIGHT
        8105006: 423,  # RETALHO DE MORTADELA LIGHT
        8108102: 406,  # PEITO DE FRANGO A MILANESA SEMIPRONTO
        7100503: 370,  # BOLA DO PATINHO
        8105801: 437,  # PATE DE PRESUNTO EMBUTIDO LIGHT
        7702402: 437,  # PASTA DE PRESUNTO EM CONSERVA
        8105011: 423,  # MORTADELA DEFUMADA LIGHT
        8002210: 282,  # BISCOITO DE CAMARAO
        7260119: 282,  # CAMARAO PISTOLA
        # --- Residuais v4/v5 (busca exata por termos) ---
        7100401: 351,  # CHA DE DENTRO
        7107604: 370,  # CARNE DE TATU
        7800702: 401,  # DRUMETE GALINHA
        7107624: 370,  # CARNE TAMANDUA
        7702303: 417,  # PATE CALABRESA
        7101011: 347,  # VAZIO BOVINO
        7101009: 333,  # CARNE MARICA
        7102401: 354,  # TRIPA BOVINA
        7107603: 370,  # CARNE DE PACA
        7112401: 347,  # SUA BOVINA
        7110201: 333,  # CARNE 3a BOVINA
        7105605: 368,  # CARNE OVINA
        7107002: 426,  # BACO SUINO
        },
        'sanidade':    {
        'energia_kcal':  (100, 400),
        'proteina_g':    (10,  35),
        'carboidrato_g': (0,   15),
        'lipideos_g':    (2,   30),
        'fibra_g':       (0,   5),
        'sodio_mg':      (100, 800),
        'ferro_mg':      (0.5, 5),
        'calcio_mg':     (2,   50),
        'zinco_mg':      (1,   8),
        'vitaminaC_mg':  (0,   10),
        },
    },
    '09.Laticinios': {
        'dict_manual': {
        # Leites líquidos integrais → Leite de cabra [453]
        # Nota: TACO idx 457 (leite integral) tem macros zerados — leite de cabra
        # é o único proxy com perfil completo próximo (Prot=3.1, Lip=3.8, Carbo=5.2)
        7900101: 453, 7900102: 453, 7900103: 453, 7900104: 453,
        7900105: 453, 7900106: 453, 7900107: 453, 7900109: 453,
        7900110: 453, 7900201: 453, 7900203: 453, 7900204: 453,
        7900205: 453, 7900401: 453, 7904301: 453, 7904305: 453,
        7904310: 453, 7904403: 453, 7900111: 453, 7900112: 453,
        7900202: 453, 7904402: 453, 7900108: 453,
        7903801: 453, 7903802: 453, 7903803: 453,
        7908901: 453, 7908902: 453,  # Leite sem/zero lactose

        # Leites semidesnatados/desnatados líquidos → Iogurte natural desnatado [448]
        # Nota: TACO idx 456 (leite desnatado UHT) tem macros zerados
        7903701: 448, 7903702: 448, 7903703: 448,
        7904601: 448, 7904503: 448,
        7903601: 448, 7903602: 448, 7903603: 448,
        7903604: 448, 7903605: 448,

        # Leite de soja/vegetal/aveia → iogurte natural [447]
        7902304: 447, 7902305: 447, 7902306: 447,
        7902301: 447, 7902303: 447, 7902307: 447,
        7902308: 447, 7902309: 447,
        7911901: 447, 7912001: 447, 7912101: 447,
        7912201: 447, 7903501: 447, 7903502: 447,
        7903401: 447, 7903403: 447,

        # Coalhada → iogurte natural [447]
        7901401: 447, 7901402: 447, 7901404: 447,
        7901405: 450, 7907201: 447, 7911501: 447, 7909701: 447,

        # Nata → creme de leite [446]
        7903201: 446, 7903202: 446, 7903203: 446,
        7903204: 446, 7908701: 446,

        # Manteiga → creme de leite [446]
        7901501: 446, 7901502: 446, 7901503: 446,
        7901504: 446, 7901505: 446, 7901506: 446,
        7901507: 446, 7901508: 446, 7901509: 446,
        7907101: 446, 7907102: 446, 7907103: 446,
        7907106: 446, 7912301: 446,

        # Leite com sabor/aromatizado → iogurte pêssego [451]
        7903109: 451, 7903101: 451, 7903103: 451,
        7907001: 451, 7906803: 451, 7906811: 451, 7905201: 451,

        # Leite achocolatado → achocolatado [454]
        7903102: 454, 7903104: 454, 7903113: 454,
        7903105: 454, 7903106: 454, 7903107: 454,
        7903112: 454, 7903111: 454,

        # Leite fermentado vit./toddynho → achocolatado [454]
        7906707: 454,

        # Queijo não especificado → queijo prato [466]
        7903001: 466, 7906601: 466, 7902101: 466,
        7906101: 466, 7902403: 466,

        # Recheio pizza → mozarela [462]
        7912501: 462,

        # Queijo emmental → queijo prato [466]
        7910001: 466,

        # Queijo cottage/suíço → ricota [468]
        7908201: 468, 7902102: 468, 7911101: 468,

        # Cream cheese → requeijão cremoso [467]
        7908501: 467,

        # Queijo de cabra → leite de cabra [453]
        7910101: 453,

        # FALLBACKS
        7903402: 460,  # TOFU → queijo minas frescal
        7901101: 446,  # CHANTILLY → creme de leite
        7901205: 451,  # DANONINHO → iogurte pêssego
        },
        'sanidade':    {
        'energia_kcal':  (50,  250),
        'proteina_g':    (3,    20),
        'carboidrato_g': (2,    20),
        'lipideos_g':    (2,    20),
        'fibra_g':       (0,     1),
        'sodio_mg':      (50,  400),
        'ferro_mg':      (0,     1),
        'calcio_mg':     (100, 500),
        'zinco_mg':      (0,     2),
        'vitaminaC_mg':  (0,    10),
        },
    },
    '10.Oleos_Gorduras': {
        'dict_manual': {},
        'sanidade':    {
        'energia_kcal':  (700, 900),
        'proteina_g':    (0,    5),
        'carboidrato_g': (0,    5),
        'lipideos_g':    (75, 100),
        'fibra_g':       (0,    2),
        'sodio_mg':      (0,  600),
        'ferro_mg':      (0,    5),
        'calcio_mg':     (0,   50),
        'zinco_mg':      (0,    1),
        'vitaminaC_mg':  (0,    5),
        },
    },
    '11.Bebidas_NA': {
        'dict_manual': {
        # Chás medicinais → chá mate [475]
        8216701: 475,  # CHA CANELA DE VELHO
        8218901: 475,  # CHA DE CAPIM CIDREIRA
        8215301: 475,  # CHA DE CARQUEJA
        8206601: 475,  # BOLDO CHA
        8203202: 475,  # CHA DE CAMOMILA
        8203102: 475,  # CHA DE CANELA
        8216101: 475,  # CHA DE MACA COM CANELA
        8220001: 475,  # CHA DE MELAO DE SAO CAETANO
        8206602: 475,  # CHA DE BOLDO
        8221101: 475,  # CASCA DE BARBATIMAO
        8206503: 475,  # FOLHA DE ARRUDA
        8213601: 475,  # CLOROFILA (SUCO)

        # Cajuína → coco água de [477] proxy
        8204603: 477,  # CAJUINA

        # Garapa → caldo de cana [472]
        8202002: 472,  # GARAPA

        # Chimarrão → chá mate [475]
        8202804: 475,  # CHIMARRAO

        # Xarope de fruta → coco água de [477]
        8202202: 477,  # XAROPE DE FRUTA OU VEGETAL ENGARRAFADO

        # Cevada em pó/moída → café infusão [470]
        8203603: 470,  # CEVADA EM PO
        8203604: 470,  # CEVADA MOIDA

        # Tubaina → refrigerante cola [479]
        8201802: 479,  # TUBAINA
        8201801: 479,  # REFRIGERANTE TUBAINA

        # Bebida alcoólica ice → refrigerante cola [479]
        8218401: 479,  # BEBIDA ALCOOLICA ICE

        # Chocolate líquido → caldo de cana [472] proxy
        8204504: 472,  # CHOCOLATE LIQUIDO
        8204501: 472,  # CHOCOLATE ENGARRAFADO

        # FALLBACKs → refrigerante cola [479]
        8200902: 479,  # MINUANO
        8205802: 469,  # TAFFMAN E (energético) → isotônico
        8213404: 479,  # PEPSI ZERO
        8200105: 479,  # PEPSI
        8200208: 479,  # SUKITA
        8200602: 479,  # MINEIRINHO
        8200803: 479,  # PEPSI DIET

        # Gelo → chá erva-doce [474] proxy neutro
        8204801: 474,  # GELO EM CUBO

        # Bebida mista de amendoa/cereais/castanha → isotônico [469]
        8218201: 469,  # BEBIDA DE AMENDOA ORGANICA
        8218301: 469,  # BEBIDA MISTA DE CEREAL ORGANICA
        8217901: 469,  # BEBIDAS MISTAS DE CEREAIS
        8218001: 469,  # BEBIDA MISTA DE CASTANHA
        },
        'sanidade':    {
        'energia_kcal':  (0,   80),
        'proteina_g':    (0,    3),
        'carboidrato_g': (0,   20),
        'lipideos_g':    (0,    2),
        'fibra_g':       (0,    2),
        'sodio_mg':      (0,   50),
        'ferro_mg':      (0,    2),
        'calcio_mg':     (0,   50),
        'zinco_mg':      (0,    1),
        'vitaminaC_mg':  (0,   30),
        },
    },
    '12.Alimentacao_Fora': {
        'dict_manual': {
        # PIZZA (todas variantes) → Cuscuz paulista [533]
        # proxy com carbo+gordura de prato composto (Kcal=131, Na=281)
        8500903: 533,  # PIZZA PRONTA PARA VIAGEM
        8500920: 533,  # PIZZA PARA VIAGEM
        8500917: 533,  # PIZZA PORTUGUESA
        8500915: 533,  # PIZZA MUZZARELA
        8500914: 533,  # PIZZA CALABRESA
        8500208: 533,  # PIZZA EM PEDACO PARA VIAGEM
        8500904: 533,  # LAZANHA PRONTA PARA VIAGEM

        # FRANGO ASSADO/DEFUMADO → Frango com açafrão [540]
        # (Kcal=134, Prot=14.2, Lip=6.8, Na=262)
        8501011: 540,  # FRANGO ASSADO PARA VIAGEM
        8501001: 540,  # FRANGO ASSADO OU DEFUMADO PARA VIAGEM
        8501005: 540,  # FRANGO A PASSARINHO PARA VIAGEM
        8501004: 540,  # GALETO PARA VIAGEM
        8501003: 540,  # GALETO ASSADO OU DEFUMADO PARA VIAGEM

        # MARMITEX/MARMITA/QUENTINHA → Virado à paulista [554]
        # prato mais completo disponível (Kcal=182, Prot=11.9, Carbo=19, Na=371)
        8500104: 554,  # MARMITEX PARA VIAGEM
        8500102: 554,  # MARMITA PARA VIAGEM
        8500101: 554,  # REFEICAO PRONTA PARA VIAGEM
        8500103: 554,  # QUENTINHA PARA VIAGEM
        8500106: 554,  # ALMOCO EM QUENTINHA PARA VIAGEM
        8500105: 554,  # ALMOCO PARA VIAGEM
        8500107: 554,  # JANTAR PARA VIAGEM
        8500109: 554,  # VIANDA PARA VIAGEM
        8500701: 554,  # ALIMENTO PRONTO CONGELADO PARA VIAGEM

        # BATATA (frita, palha, congelada) → Quibebe [543]
        # proxy com carbo alto (Kcal=50, Carbo=9.3, Na=178)
        8501503: 543,  # BATATA PALHA PARA VIAGEM
        8501501: 543,  # BATATA FRITA PARA VIAGEM
        8501502: 543,  # BATATA PALITO PARA VIAGEM
        8503104: 543,  # BATATA CONGELADA PARA FRITAR
        8503101: 543,  # BATATA CONGELADA PARA VIAGEM

        # SALGADINHO/SALGADO FRITO → Bolinho de arroz [529]
        # proxy com carbo+gordura (Kcal=163, Carbo=24.3, Lip=6.0, Na=247)
        8500201: 529,  # SALGADINHO PARA VIAGEM
        8500202: 529,  # PASTEL PARA VIAGEM
        8500222: 529,  # ESFIRRA PARA VIAGEM
        8500204: 529,  # BOLINHO SALGADO DE CARNE/CAMARAO PARA VIAGEM
        8500205: 529,  # COXINHA PARA VIAGEM
        8500232: 529,  # BOLINHO DE BACALHAU PARA VIAGEM

        # SANDUÍCHE/HAMBÚRGUER/CACHORRO QUENTE → Cuscuz paulista [533]
        8500301: 533,  # SANDUICHE PARA VIAGEM
        8500303: 533,  # HAMBURGUER PARA VIAGEM
        8500302: 533,  # CACHORRO QUENTE PARA VIAGEM

        # LANCHE → Bolinho de arroz [529]
        8503601: 529,  # LANCHE PARA VIAGEM

        # CHURRASCO/ESPETINHO/COSTELA/CARNE → Bife à cavalo [528]
        # (Kcal=208, Prot=20.5, Lip=12.9, Na=268)
        8503801: 528,  # CHURRASCO PARA VIAGEM
        8501102: 528,  # ESPETINHO DE CARNE PARA VIAGEM
        8501104: 528,  # COSTELA ASSADA PARA VIAGEM
        8501109: 528,  # CARNE ASSADA PARA VIAGEM
        8501105: 528,  # BIFE PREPARADO PARA VIAGEM

        # FEIJOADA → Feijoada [539] — proxy exato
        8508701: 539,  # FEIJOADA PARA VIAGEM

        # YAKISSOBA → Yakisoba [555] — proxy exato
        8509201: 555,  # YAKISSOBA PARA VIAGEM

        # PEIXE FRITO/FRUTOS DO MAR → Camarão à baiana [530]
        8501801: 530,  # PEIXE FRITO PARA VIAGEM

        # SALADA → Salada de legumes com maionese [544]
        8508403: 544,  # SALADA DE LEGUMES PARA VIAGEM

        # FAROFA → Feijão tropeiro [538]
        8502901: 538,  # FAROFA PRONTA EM PACOTE PARA VIAGEM
        8502201: 538,  # FAROFA PARA VIAGEM

        # EMPADÃO → Bolinho de arroz [529]
        8505901: 529,  # EMPADAO PARA VIAGEM

        # PIPOCA → Cuscuz de milho [532]
        8501202: 532,  # PIPOCA DOCE OU SALGADA PARA VIAGEM

        # SUCO PARA VIAGEM → Cuscuz de milho [532] proxy neutro
        8500407: 532,  # SUCO DE LARANJA PARA VIAGEM

        # DIET SHAKE / VITAMINA / BARRA PROTEÍNA (de col_4) → Estrogonofe frango [537]
        6906501: 537,  # DIET SHAKE
        6914701: 537,  # VITAMINA CONCENTRADA (SHAKE)
        6915901: 537,  # BARRA DE PROTEINA

        # SOPAS DESIDRATADAS → Cuscuz de milho [532]
        7706101: 532,  # SOPA DESIDRATADA LIGHT
        7706102: 532,  # SOPA EM PACOTE LIGHT
        7706103: 532,  # SOPA CONCENTRADA LIGHT
        },
        'sanidade':    {
        'energia_kcal':  (80,  300),
        'proteina_g':    (3,    20),
        'carboidrato_g': (10,   40),
        'lipideos_g':    (3,    20),
        'fibra_g':       (0,     5),
        'sodio_mg':      (100, 600),
        'ferro_mg':      (0,     5),
        'calcio_mg':     (10,  150),
        'zinco_mg':      (0,     3),
        'vitaminaC_mg':  (0,    20),
        },
    },
}


# -- Validacao de integridade dos dicionarios --------------------------------
# Detecta conflitos de chave antes de qualquer execucao.
# Duplicatas com mesmo valor (redundancias) sao ignoradas.
# Duplicatas com valores diferentes (conflitos) levantam excecao.
def _check_no_conflicts(d, grupo):
    seen = {}
    for k, v in d.items():
        if k in seen and seen[k] != v:
            raise ValueError(
                f"Conflito em dict_manual['{grupo}']: "
                f"V9001={k} mapeado para TACO_IDX={seen[k]} e {v}. "
                f"Edite GRUPO_CONFIG para manter apenas um valor."
            )
        seen[k] = v

for _grupo, _cfg in GRUPO_CONFIG.items():
    _check_no_conflicts(_cfg['dict_manual'], _grupo)
print("[INIT] Validacao de dict_manual: OK (sem conflitos de chave)")


# -- Funcoes auxiliares -------------------------------------------------------

def norm(s):
    """Normalizacao de texto: minusculo, sem acento, sem pontuacao."""
    s = str(s).lower().strip()
    s = unorm('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def to_num(s):
    """Converte string TACO para float (TR/TRACO/* -> 0)."""
    s = str(s).strip()
    if s.upper() in ('TR', 'TRACO', '*', 'NA', ''): return 0.0
    return pd.to_numeric(s.replace(',', '.'), errors='coerce')

def extrai_g(s):
    """Converte descricao de unidade IBGE para gramas."""
    s = str(s).upper().strip().replace(',', '.')
    for pat, mul in [
        (r'([\d.]+)\s*(KG|KILO|QUILOGRAMA|QUILO)', 1000),
        (r'([\d.]+)\s*(LITRO|LT)\b', 1000),
        (r'([\d.]+)\s*L\b', 1000),
        (r'([\d.]+)\s*(ML|MILILITRO)', 1),
        (r'([\d.]+)\s*(GR|G|GRAMA)', 1),
    ]:
        m = re.search(pat, s)
        if m: return float(m.group(1)) * mul
    m = re.search(r'([\d.]+)', s)
    return float(m.group(1)) if m else np.nan

def trimming_iqr(grupo_df):
    """Remove outliers de ln(preco) via regra IQR por V9001."""
    q1, q3 = grupo_df['LN_PRECO'].quantile([0.25, 0.75])
    iqr = q3 - q1
    return grupo_df[
        (grupo_df['LN_PRECO'] >= q1 - 1.5 * iqr) &
        (grupo_df['LN_PRECO'] <= q3 + 1.5 * iqr)
    ]

def c_nj_calc(cad_df, nut):
    """
    C_nj = sum(c_n/100 * QTD_GRAMAS_CORR * PESO_FINAL) / sum(QTD_GRAMAS_CORR * PESO_FINAL)
    Unidade de saida: nutriente por 100g de alimento do grupo.
    """
    d = cad_df[cad_df[nut].notna() & cad_df['QTD_GRAMAS_CORR'].gt(0)]
    num   = (d[nut] / 100 * d['QTD_GRAMAS_CORR'] * d['PESO_FINAL']).sum()
    denom = (d['QTD_GRAMAS_CORR'] * d['PESO_FINAL']).sum()
    return num / denom * 100 if denom > 0 else 0.0


def main():
    # -- Carrega dados fixos (uma vez, antes do loop) -------------------------

    print("=" * 70)
    print("02_matriz_C_FINAL.py — Construcao da Matriz C e Omega")
    print("=" * 70)

    # TACO
    print("\n[INIT] Carregando TACO...")
    taco = pd.read_csv(DAD / 'alimentos.csv', encoding='utf-8')
    taco.columns = [c.strip() for c in taco.columns]
    cat_col  = taco.columns[1]
    desc_col = taco.columns[2]

    nut_col_map = {}
    cols_norm_taco = [norm(c) for c in taco.columns]
    for nut, substr in NUTRIENTES.items():
        hits = [i for i, cn in enumerate(cols_norm_taco) if substr in cn]
        if hits:
            nut_col_map[nut] = taco.columns[hits[0]]

    for nut, col in nut_col_map.items():
        taco[f'_{nut}'] = taco[col].apply(to_num).fillna(0)
    taco['desc_norm'] = taco[desc_col].apply(norm)
    print(f"  {len(taco)} alimentos | {len(nut_col_map)} nutrientes mapeados")

    # Classificacao POF
    print("[INIT] Carregando classificacao_grupos_v2.parquet...")
    classif   = pd.read_parquet(OUT / 'classificacao_grupos_v2.parquet')
    grupo_map = dict(zip(classif['V9001'], classif['GRUPO_FINAL']))
    desc_map  = dict(zip(classif['V9001'], classif['DESCRICAO']))
    print(f"  {len(classif):,} produtos classificados")

    # Caderneta (lida uma vez; filtrada por grupo no loop)
    print("[INIT] Carregando CADERNETA_COLETIVA.txt...")
    _w = [2,4,1,9,2,1,2,3,7,2,10,12,10,1,2,14,14,10,9,4,5,9,5]
    _c = ['UF','ESTRATO_POF','TIPO_SITUACAO_REG','COD_UPA','NUM_DOM','NUM_UC',
          'QUADRO','SEQ','V9001','V9002','V8000','DEFLATOR','V8000_DEFLA',
          'COD_IMPUT_VALOR','FATOR_ANUALIZACAO','PESO','PESO_FINAL',
          'RENDA_TOTAL','V9005','V9007','V9009','QTD_FINAL','V9004']

    def _w2c(ws):
        specs, s = [], 0
        for w in ws:
            specs.append((s, s + w)); s += w
        return specs

    cad_raw = pd.read_fwf(
        DAD / 'CADERNETA_COLETIVA.txt', colspecs=_w2c(_w),
        names=_c, encoding='latin-1', header=None, dtype=str
    )
    for col in ['V9001', 'V8000', 'V8000_DEFLA', 'V9007', 'QTD_FINAL', 'PESO_FINAL']:
        cad_raw[col] = pd.to_numeric(cad_raw[col], errors='coerce')
    cad_raw['GRUPO']       = cad_raw['V9001'].map(grupo_map)
    cad_raw['V8000_DEFLA'] = cad_raw['V8000_DEFLA'].fillna(cad_raw['V8000'])
    print(f"  {len(cad_raw):,} registros totais")

    # Tabela de pesos/volumes IBGE
    pesos_raw = pd.read_excel(DAD / '../documentacao/Cadastro de Pesos ou Volumes.xls')
    pesos_raw.columns = ['CODIGO', 'PESO_VOL']
    peso_map_u = dict(zip(pesos_raw['CODIGO'], pesos_raw['PESO_VOL'].apply(extrai_g)))

    KG = {4801, 4802, 4803, 4804, 4805, 4806, 4807, 4808, 4809, 4810}
    G  = {4501, 4502, 4503, 4504}
    L  = {4701, 4702, 4703, 4704}
    ML = {4601, 4602, 4603}

    # -- Loop principal por grupo ---------------------------------------------

    resultados_cnj  = {}
    resultados_qtd  = {}
    mapas_completos = []

    SEP = "=" * 70

    for GRUPO in GRUPOS_C:

        cfg         = GRUPO_CONFIG[GRUPO]
        dict_manual = cfg['dict_manual']
        SANIDADE    = cfg['sanidade']

        print(f"\n{SEP}")
        print(f"GRUPO: {GRUPO}")
        print(SEP)

        # [1] Pool de candidatos TACO
        cats       = [cat for cat, gs in CATEGORIA_TACO_GRUPO.items() if GRUPO in gs]
        candidatos = taco[taco[cat_col].isin(cats)].index.tolist()
        print(f"  [1] {len(candidatos)} candidatos TACO")

        if len(candidatos) == 0:
            print(f"  [ERRO] Nenhum candidato TACO para {GRUPO} — grupo pulado.")
            continue

        # [2] TF-IDF char n-gramas (3,4) sobre candidatos do grupo
        desc_cand   = [taco.loc[i, 'desc_norm'] for i in candidatos]
        vectorizer  = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4),
                                      min_df=1, sublinear_tf=True)
        taco_matrix = vectorizer.fit_transform(desc_cand)

        def match_tfidf(desc_pof_norm, limiar=0.05):
            vec    = vectorizer.transform([desc_pof_norm])
            scores = cosine_similarity(vec, taco_matrix).flatten()
            best   = scores.argmax()
            if scores[best] < limiar:
                return None, scores[best], 'FALLBACK'
            gi = candidatos[best]
            return gi, scores[best], taco.loc[gi, desc_col]

        # [3-4] Matching: MANUAL -> TFIDF -> FALLBACK
        v9001_grupo = classif[classif['GRUPO_FINAL'] == GRUPO]['V9001'].unique()
        res = []
        n_m = n_t = n_f = 0
        for v9001 in v9001_grupo:
            dpof = str(desc_map.get(v9001, ''))
            if v9001 in dict_manual:
                idx   = dict_manual[v9001]; fonte = 'MANUAL'; n_m += 1
                tdesc = taco.loc[idx, desc_col] if 0 <= idx < len(taco) else 'N/A'
                score = 1.0
            else:
                idx, score, tdesc = match_tfidf(norm(dpof))
                if idx is None:
                    idx = -1; tdesc = 'FALLBACK'; fonte = 'FALLBACK'; n_f += 1
                else:
                    fonte = 'TFIDF'; n_t += 1
            row = dict(V9001=v9001, GRUPO=GRUPO, DESC_POF=dpof,
                       TACO_IDX=idx, TACO_DESC=tdesc,
                       SCORE=round(score, 3), FONTE=fonte)
            for nut in NUT_COLS:
                row[nut] = taco.loc[idx, f'_{nut}'] if idx >= 0 else 0.0
            res.append(row)

        mapa = pd.DataFrame(res)
        mapas_completos.append(mapa)
        nut_maps = {nut: dict(zip(mapa['V9001'], mapa[nut])) for nut in NUT_COLS}
        pct_cob = 100 * len(mapa) / max(len(v9001_grupo), 1)
        print(f"  [4] MANUAL:{n_m} | TFIDF:{n_t} | FALLBACK:{n_f} | "
              f"Cobertura: {len(mapa)}/{len(v9001_grupo)} ({pct_cob:.0f}%)")

        # [5] Filtra caderneta para este grupo
        cad = cad_raw[
            (cad_raw['GRUPO'] == GRUPO) &
            cad_raw['V8000'].gt(0) & cad_raw['V8000'].lt(9_999_999) &
            cad_raw['PESO_FINAL'].gt(0)
        ].copy()
        print(f"  [5] {len(cad):,} registros na caderneta")

        # [6] Conversao de unidades -> gramas
        cad['QTD_GRAMAS'] = np.nan
        cad.loc[cad['V9007'].isin(KG), 'QTD_GRAMAS'] = cad.loc[cad['V9007'].isin(KG), 'QTD_FINAL'] * 1000
        cad.loc[cad['V9007'].isin(G),  'QTD_GRAMAS'] = cad.loc[cad['V9007'].isin(G),  'QTD_FINAL']
        cad.loc[cad['V9007'].isin(L),  'QTD_GRAMAS'] = cad.loc[cad['V9007'].isin(L),  'QTD_FINAL'] * 1000
        cad.loc[cad['V9007'].isin(ML), 'QTD_GRAMAS'] = cad.loc[cad['V9007'].isin(ML), 'QTD_FINAL']
        mask_u = cad['QTD_GRAMAS'].isna()
        cad.loc[mask_u, 'QTD_GRAMAS'] = (
            cad.loc[mask_u, 'QTD_FINAL'] * cad.loc[mask_u, 'V9007'].map(peso_map_u)
        )
        n_val = cad['QTD_GRAMAS'].gt(0).sum()
        n_sem = cad['QTD_GRAMAS'].isna().sum()
        pct   = 100 * n_val / len(cad) if len(cad) > 0 else 0
        print(f"  [6] Conversao valida: {n_val:,} ({pct:.1f}%) | "
              f"Sem conversao: {n_sem:,}")

        # [7] Preco mediano por V9001 com trimming IQR
        validos = cad[cad['QTD_GRAMAS'].gt(0)].copy()
        validos['PRECO_G']  = validos['V8000_DEFLA'] / validos['QTD_GRAMAS']
        validos = validos[validos['PRECO_G'].gt(0)].copy()
        validos['LN_PRECO'] = np.log(validos['PRECO_G'])
        validos_trim  = validos.groupby('V9001', group_keys=False).apply(trimming_iqr)
        preco_mediano = validos_trim.groupby('V9001')['PRECO_G'].median()
        if validos_trim.empty:
            preco_grupo = validos['PRECO_G'].median() if not validos.empty else 1.0
        else:
            preco_grupo = validos_trim['PRECO_G'].median()

        # [8] Imputa QTD_GRAMAS quando conversao de unidade falha
        cad['QTD_GRAMAS_CORR'] = cad['QTD_GRAMAS']
        mask_sem = cad['QTD_GRAMAS'].isna() | cad['QTD_GRAMAS'].le(0)
        cad.loc[mask_sem, 'QTD_GRAMAS_CORR'] = (
            cad.loc[mask_sem, 'V8000_DEFLA'] /
            cad.loc[mask_sem, 'V9001'].map(preco_mediano).fillna(preco_grupo)
        )
        cad = cad[cad['QTD_GRAMAS_CORR'].gt(0)].copy()
        print(f"  [8] Imputados: {mask_sem.sum():,} | Finais: {len(cad):,}")

        for nut in NUT_COLS:
            cad[nut] = cad['V9001'].map(nut_maps[nut])

        # [9] C_nj por nutriente
        cnj = {nut: c_nj_calc(cad, nut) for nut in NUT_COLS}
        resultados_cnj[GRUPO] = cnj
        resultados_qtd[GRUPO] = (cad['QTD_GRAMAS_CORR'] * cad['PESO_FINAL']).sum()

        # [10] Validacao de sanidade
        print(f"\n  [10] C_nj — {GRUPO}:")
        print(f"  {'Nutriente':<20} {'C_nj':>10}  Sanidade")
        print(f"  {'-'*50}")
        tudo_ok = True
        for nut in NUT_COLS:
            val    = cnj[nut]
            lo, hi = SANIDADE[nut]
            ok     = 'OK' if lo <= val <= hi else f'ATENCAO [{lo}-{hi}]'
            if 'ATENCAO' in ok:
                tudo_ok = False
            print(f"  {nut:<20} {val:>10.3f}  {ok}")
        print(f"  -> {'[OK]' if tudo_ok else '[VERIFICAR]'}")

    # -- Monta Matriz C (10 x 14) ---------------------------------------------

    print(f"\n{SEP}")
    print("MONTANDO MATRIZ C e Omega")
    print(SEP)

    mat_C = pd.DataFrame(index=NUT_COLS, columns=GRUPOS_QUAIDS, dtype=float).fillna(0.0)
    for grupo in GRUPOS_C:
        if grupo in resultados_cnj:
            for nut in NUT_COLS:
                mat_C.loc[nut, grupo] = resultados_cnj[grupo][nut]

    Q_g = pd.Series(0.0, index=GRUPOS_QUAIDS)
    for grupo in GRUPOS_C:
        if grupo in resultados_qtd:
            Q_g[grupo] = resultados_qtd[grupo]

    # Omega_{ng} = C_{ng} * Q_g / sum_k(C_{nk} * Q_k)
    C_times_Q = mat_C.multiply(Q_g, axis=1)
    row_sums   = C_times_Q.sum(axis=1)
    mat_Omega  = C_times_Q.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)

    print("\n  Verificacao das linhas de Omega (deve somar 1.0):")
    row_check = mat_Omega.sum(axis=1)
    all_ok = True
    for nut in NUT_COLS:
        s  = row_check[nut]
        ok = 'OK' if abs(s - 1.0) < 1e-9 else f'ATENCAO: {s:.8f}'
        if 'ATENCAO' in ok: all_ok = False
        print(f"    {nut:<20} {s:.8f}  {ok}")
    print(f"  -> {'[OK]' if all_ok else '[VERIFICAR]'}")

    # -- Salva outputs --------------------------------------------------------

    mat_C.to_parquet(OUT / 'matriz_C.parquet')
    mat_Omega.to_parquet(OUT / 'matriz_omega.parquet')
    pd.concat(mapas_completos, ignore_index=True).to_parquet(
        OUT / 'v9001_taco_map_FINAL.parquet'
    )

    print(f"\n{SEP}")
    print("OUTPUTS SALVOS:")
    print(f"  {OUT}/matriz_C.parquet")
    print(f"  {OUT}/matriz_omega.parquet")
    print(f"  {OUT}/v9001_taco_map_FINAL.parquet")
    print(SEP)

    print("\nMATRIZ C (nutriente x grupo, C_nj):")
    print(mat_C.to_string(float_format=lambda x: f'{x:8.2f}'))

    print("\nMATRIZ Omega (participacao nutricional por grupo):")
    print(mat_Omega.map(lambda x: f'{x:.4f}').to_string())

    print("\n[OK] 02_matriz_C_FINAL.py concluido.")


if __name__ == "__main__":
    main()
