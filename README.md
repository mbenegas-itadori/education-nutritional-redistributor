# Education as a Nutritional Redistributor
### Income, Diet Quality, and the Demand for Nutrients in Brazil

Replication package for the paper by Mauricio Benegas (CAEN/UFC), submitted
.

[![DOI](https://zenodo.org/badge/DOI/PENDING.svg)](https://doi.org/PENDING)

---

## What's in this repository

```
.
├── README.md                 — this file
├── REPRODUCIBILITY.md        — full pipeline documentation: data sources,
│                                script order, validation checkpoints
├── LICENSE                   — code license (MIT)
├── requirements.txt          — Python dependencies
├── CITATION.cff              — machine-readable citation metadata
└── scripts/                  — the 11-script estimation pipeline
    ├── 01_leitura_pof.py
    ├── 02_variaveis_socio.py
    ├── 00_consumo_quaids_v2.py
    ├── 01b_base_analitica_v2.py
    ├── 02_matriz_C_FINAL.py
    ├── 03_probit_copula_v3.py
    ├── 04_quaids_estimacao.py
    ├── 05_elasticidades_nutricionais.py
    ├── 06_engel_nutricional_FINAL.py
    ├── 08_engel_grupos_FINAL.py
    └── config.py
```

See **`REPRODUCIBILITY.md`** for the complete pipeline map (execution
order, inputs/outputs at every stage, and numerical anchors to confirm a
reproduction is correct).

## Data availability

This repository does **not** redistribute raw microdata. The pipeline uses:

- **POF 2017–18** (Pesquisa de Orçamentos Familiares), IBGE — public microdata,
  available at <https://www.ibge.gov.br/estatisticas/sociais/saude/24786-pesquisa-de-orcamentos-familiares-2.html>.
  The collective acquisition diary (`CADERNETA_COLETIVA.txt`) and individual
  expenditure module (`DESPESA_INDIVIDUAL.txt`), plus household/person
  records (`DOMICILIO.TXT`, `MORADOR.TXT`) and income modules
  (`RENDIMENTO_DO_TRABALHO.TXT`, `OUTROS_RENDIMENTOS.TXT`), are required.
- **TACO** (Tabela Brasileira de Composição de Alimentos), 4th ed.,
  NEPA-UNICAMP, 2011 — used to build the nutritional content matrix.

`REPRODUCIBILITY.md` documents exactly which files are needed and how each
script consumes them.

## Quick start

1. Download the POF 2017–18 microdata and TACO table from the sources above.
2. Set up the folder structure described in `REPRODUCIBILITY.md` (§2).
3. Install dependencies: `pip install -r requirements.txt`
4. Run the 11 scripts in `scripts/` in the order documented in
   `REPRODUCIBILITY.md` (§5) — each prints its own progress and a summary
   on completion.
5. Cross-check your output against the numerical anchors in
   `REPRODUCIBILITY.md` (§7) to confirm a correct reproduction.

Scripts were written for Google Colab (hardcoded `/content/drive/MyDrive/POF/...`
paths) but run on any environment once paths are adjusted — see
`REPRODUCIBILITY.md` for details.

## Known limitations of this replication package

- Software versions (Python, pandas, numpy, statsmodels, etc.) are not yet
  pinned to exact numbers — see `REPRODUCIBILITY.md` (§4).

## Citation

If you use this code, please cite the paper:

> Benegas, M. (2026). Education as a Nutritional Redistributor: Income,
> Diet Quality, and the Demand for Nutrients in Brazil, (subbmitted).

See `CITATION.cff` for machine-readable citation metadata.

## License

Code is released under the MIT License (see `LICENSE`). This license
covers the code only — it does not apply to the POF or TACO data, which
are governed by their respective sources' terms of use.

## Contact

Mauricio Benegas — Graduate Program in Economics, CAEN/UFC
mbenegas@ufc.br
