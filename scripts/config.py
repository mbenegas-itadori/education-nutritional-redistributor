"""
config.py
=========
Configuração central dos scripts POF 2017-18.
Edite apenas este arquivo para ajustar caminhos.
Todos os scripts (01, 02, 03) importam daqui.
"""

from pathlib import Path

# ============================================================
# CAMINHOS — ajuste conforme sua estrutura no Drive
# ============================================================

# Pasta raiz do projeto no Google Drive
DRIVE_ROOT = Path('/content/drive/MyDrive/POF')

# Subpasta com os arquivos brutos .txt da POF
DATA_DIR = DRIVE_ROOT / 'dados'

# Subpasta de saída para parquets e base final
OUT_DIR = DRIVE_ROOT / 'output'

# Caminho do arquivo consumo_quaids.xlsx
CONSUMO_PATH = DRIVE_ROOT / 'consumo_quaids.xlsx'

# ============================================================
# CRIA PASTAS SE NÃO EXISTIREM
# ============================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CÓDIGO DO BOLSA FAMÍLIA no dicionário de fontes (Quadro 54)
# ============================================================
COD_BOLSA_FAMILIA = 5400101

# ============================================================
# VERIFICAÇÃO RÁPIDA
# ============================================================
if __name__ == '__main__':
    print("=== CONFIGURAÇÃO ===")
    print(f"DRIVE_ROOT:    {DRIVE_ROOT}")
    print(f"DATA_DIR:      {DATA_DIR}  | existe: {DATA_DIR.exists()}")
    print(f"OUT_DIR:       {OUT_DIR}   | existe: {OUT_DIR.exists()}")
    print(f"CONSUMO_PATH:  {CONSUMO_PATH} | existe: {CONSUMO_PATH.exists()}")

    if DATA_DIR.exists():
        txts = list(DATA_DIR.glob('*.TXT')) + list(DATA_DIR.glob('*.txt'))
        print(f"\nArquivos .txt encontrados em DATA_DIR ({len(txts)}):")
        for t in sorted(txts):
            print(f"  {t.name}")
    else:
        print(f"\nAVISO: DATA_DIR não encontrado — verifique o caminho.")
