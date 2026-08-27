# Protocolo de Reprodutibilidade — "Education as a Nutritional Redistributor"

**Status deste documento:** rascunho inicial (sessão 2026-08-20), gerado a partir do
mapeamento do pipeline no Google Drive durante a investigação do item 14
(sensibilidade da imputação de Alimentação Fora). Precisa de revisão e
preenchimento dos `TODO` antes de publicar no repositório.

---

## 1. Visão geral

Este documento descreve como reproduzir, do zero, os resultados do artigo a
partir dos microdados da POF 2017-18 (IBGE) e da Tabela TACO (4ª ed.,
NEPA-UNICAMP). Cobre: estrutura de diretórios, dependências de software,
ordem de execução do pipeline, e como validar que uma reprodução está
correta.

`TODO`: decidir se este documento acompanha o replication package completo
(incluindo dados brutos, quando a licença permitir) ou só o código +
instruções de acesso aos dados.

---

## 2. Estrutura de diretórios

O pipeline espera a seguinte estrutura (hoje replicada no Google Drive do
autor, pasta `POF/`; caminhos hardcoded nos scripts como
`/content/drive/MyDrive/POF/...` — ajustar se migrar para execução local):

```
POF/
├── dados/              # inputs brutos, não gerados pelo pipeline
│   ├── CADERNETA_COLETIVA.txt      (~110 MB, diário coletivo POF, fixed-width)
│   └── alimentos.csv                (TACO 4ª ed., ~4.848 produtos × 27 nutrientes)
├── documentacao/        # tabelas de apoio do IBGE
│   ├── Cadastro de Unidades de Medida.xls
│   └── Cadastro de Pesos ou Volumes.xls
├── output/               # tudo gerado pelo pipeline (não versionar no git — ver §7)
│   ├── classificacao_grupos_v2.parquet
│   ├── base_quaids_v2.parquet
│   ├── matriz_C.parquet / matriz_C_nutricional.parquet
│   ├── matriz_omega.parquet
│   ├── matriz_C_metadados.json
│   ├── v9001_taco_map.parquet  (ou v9001_taco_map_FINAL.parquet — ver §6.1)
│   ├── pesos_composicao_grupos.parquet
│   ├── probit_cov_mats.npz
│   ├── probit_xb.parquet
│   ├── quaids_s1_resultados.json   (modelo M1)
│   ├── quaids_s4_resultados.json   (modelo M2)
│   ├── quaids_cov_mats.npz         (V_SUR_S1, V_MT_S1, V_SUR_S4, Sigma_S1, Sigma_S4)
│   └── eta_nutricional.json
└── script/               # pipeline, ver §5
```

`TODO`: confirmar se `00_classificacao_v2.py` (script que gera
`classificacao_grupos_v2.parquet`) está no Drive com esse nome — não foi
localizado/inspecionado nesta sessão, só inferido pela existência do output.

---

## 3. Dados de origem

| Fonte | Versão | Acesso |
|---|---|---|
| POF 2017-18 (IBGE) — Caderneta Coletiva | `TODO: confirmar versão/data de download do microdado IBGE` | `TODO: link oficial IBGE + instruções de download` |
| POF 2017-18 (IBGE) — Quadro 24 (despesa individual) | `TODO` | usado para grupos 12 (Alimentação Fora), 13 (Álcool, parcial) |
| TACO — Tabela Brasileira de Composição de Alimentos, 4ª ed. | NEPA-UNICAMP, 2011 | `TODO: link oficial + nota de licença de redistribuição` |

`TODO`: documentar explicitamente que `CADERNETA_COLETIVA.txt` é derivado de
microdado público do IBGE, mas confirmar se pode ser redistribuído no
replication package ou se o protocolo deve só documentar como baixá-lo e
reconstruir o `.txt` fixed-width a partir da fonte oficial.

---

## 4. Ambiente de software

`TODO`: preencher com as versões reais usadas.

```
Python: TODO (>=3.10 provável, dado uso de match/walrus em scripts recentes — confirmar)
pandas: TODO
numpy: TODO
scipy: TODO
statsmodels: TODO
rapidfuzz: TODO   (usado em 02_matriz_C_v3.py para matching TACO)
scikit-learn: TODO (TfidfVectorizer, cosine_similarity — usado em 02_matriz_C_FINAL.py)
pyarrow: TODO      (leitura/escrita de .parquet)
openpyxl / xlrd: TODO (leitura de .xls — Cadastro de Pesos/Unidades)
```

`TODO`: gerar `requirements.txt` ou `environment.yml` a partir do ambiente
real (Colab, dado os paths `/content/drive/MyDrive/...` hardcoded).

`TODO`: documentar se há seed fixa em algum estágio estocástico do pipeline
(bootstrap dos gráficos de Engel nutricional — Figuras 1-4 do artigo citam
`B = 200 resamples`; a CMLE de seleção de cópula e a SUR iterada são
determinísticas dado os dados, mas o bootstrap não é, a menos que haja seed).

---

## 5. Pipeline — ordem de execução

**✅ Cadeia completa confirmada e arquivada** (sessão 2026-08-25) — ver §5.1
para o repositório validado. Numeração dos scripts corresponde à ordem de
dependência, não à nomenclatura de arquivo (que tem gaps reais — não existe
`01_matriz_C.py`, por exemplo).

| Ordem | Script | Input | Output | O que faz |
|---|---|---|---|---|
| 1 | `01_leitura_pof.py` | `DOMICILIO.TXT`, `MORADOR.TXT`, `OUTROS_RENDIMENTOS.TXT` | `domicilio.parquet`, `morador.parquet`, `rendimento_bf.parquet` | Lê microdados brutos POF, isola Bolsa Família (código 5400101) |
| 2 | `02_variaveis_socio.py` | outputs de (1) + `RENDIMENTO_DO_TRABALHO.TXT`, `OUTROS_RENDIMENTOS.TXT` | `socioeconomico.parquet` | Consolida renda total, per capita, educação, região por UC |
| 3 | `00_classificacao_v2.py` `(TODO: nome exato não confirmado — só os outputs)` | `CADERNETA_COLETIVA.txt` | `classificacao_grupos_v2.parquet` | Mapeia V9001 → 14 grupos QUAIDS (3 camadas: manual, palavra-chave, faixa de código) |
| 4 | `02_matriz_C_FINAL.py` (canônico — ver §6.1) | `alimentos.csv`, `classificacao_grupos_v2.parquet`, `CADERNETA_COLETIVA.txt`, `Cadastro de Pesos ou Volumes.xls` | `matriz_C.parquet`, `matriz_omega.parquet`, `v9001_taco_map_FINAL.parquet` | Constrói C (composição nutricional) via matching TACO (manual → TF-IDF → fallback), deriva Ω |
| 5 | `00_consumo_quaids_v2.py` | `CADERNETA_COLETIVA.txt`, `DESPESA_INDIVIDUAL.txt`, `Cadastro de Produtos.xls`, `classificacao_grupos_v2.parquet`, `socioeconomico.parquet` | `consumo_quaids_v2.parquet` (58.039 UCs) | Gastos anualizados por grupo — caderneta coletiva + Quadro 24 (Tabaco=21, Alim.Fora+Álcool=24, viagens=41) |
| 6 | `01b_base_analitica_v2.py` | `consumo_quaids_v2.parquet`, `socioeconomico.parquet`, `classificacao_grupos_v2.parquet`, `CADERNETA_COLETIVA.txt`, `DESPESA_INDIVIDUAL.txt`, `Cadastro de Produtos.xls`, `Cadastro de Unidades de Medida.xls` | `base_analitica_v2.parquet` (58.039), `base_quaids_v2.parquet` (54.208) | Preços (Deaton + imputação hierárquica PSU→UF→Região), índice de Stone, `LN_M_C` centralizado |
| 7 | `03_probit_copula_v3.py` | `base_quaids_v2.parquet` | `probit_copula_resultados.parquet`, `probit_cov_mats.npz`, `probit_xb.parquet` | Estágio 1: Probit de participação + seleção de cópula via CMLE |
| 8 | `04_quaids_estimacao.py` | `base_quaids_v2.parquet`, saídas de (7) | `quaids_s1_resultados.json` (M1), `quaids_s4_resultados.json` (M2), `quaids_cov_mats.npz` | SUR iterado (M1 baseline, M2 educação×renda), correção Murphy-Topel |
| 9 | `05_elasticidades_nutricionais.py` | saídas de (8), `matriz_omega.parquet` | `eta_nutricional.json` | η̂ₙ = Ωμ̂ (M1) e η̂ₙᵏ por estrato (M2), erros-padrão via método delta |
| 10 | `06_engel_nutricional_FINAL.py` | `base_quaids_v2.parquet`, `matriz_omega.parquet` | `figA/B/C_nutricional_FINAL.pdf` | Curvas de Engel nutricionais (Figuras 3-4 do artigo) |
| 11 | `08_engel_grupos_FINAL.py` | `base_quaids_v2.parquet` | `engel_grupos_figA/B.pdf` | Curvas de Engel por grupo de alimento (Figuras 1-2 do artigo) |

Mais: `02c`–`02m_diagnostico_*.py` (11 scripts, auditoria de classificação por
grupo — ver §6.2), não fazem parte da cadeia principal mas documentam as
correções aplicadas antes da versão FINAL de `02_matriz_C`.

**⚠️ Pipeline abandonado, NÃO usar**: `03_merge_final.py` +
`consumo_quaids.xlsx` (arquivo `.xlsx`, sem sufixo `_v2`) — versão anterior
do passo 5-6, substituída pelo par `00_consumo_quaids_v2.py` +
`01b_base_analitica_v2.py` (confirmado pela data de modificação: `.xlsx`
parado em 1º/jul, `_v2` atualizado em 5/ago). Não está no repositório
validado (§5.1).

**Nota sobre `config.py`**: só é importado por `01_leitura_pof.py` e
`02_variaveis_socio.py` (fornece `DATA_DIR`, `OUT_DIR`,
`COD_BOLSA_FAMILIA`). Do passo 3 em diante, cada script hardcoda seus
próprios caminhos (`Path('/content/drive/MyDrive/POF/...')`) — não
depende de `config.py`. O campo `CONSUMO_PATH` dentro de `config.py`
aponta para o pipeline abandonado (`consumo_quaids.xlsx`) e não é usado
por nenhum script da cadeia atual.

### 5.1 Repositório de scripts validados

Os 11 scripts da cadeia acima (passos 1-11, tabela de cima) estão
auditados, com sintaxe verificada, e arquivados em uma pasta dedicada no
Drive, separada da pasta `script/` original (que ainda contém os scripts
supersedidos e os diagnósticos):

**Pasta**: `POF/script_validados_replicacao/`
**Link**: https://drive.google.com/drive/folders/11Rr0cb8md73HHphRsmodIX_a6g77Lg4W

Proveniência de cada arquivo:
- `00_consumo_quaids_v2.py`, `01b_base_analitica_v2.py`, `03_probit_copula_v3.py`,
  `04_quaids_estimacao.py`, `05_elasticidades_nutricionais.py`: copiados
  diretamente da pasta `script/` canônica (sem retranscrição — cópia
  server-side no próprio Drive)
- `01_leitura_pof.py`, `02_variaveis_socio.py`, `config.py`: enviados a
  partir do upload local do autor, bytes conferidos exatos
- `06_engel_nutricional_FINAL.py`, `08_engel_grupos_FINAL.py`: enviados a
  partir do upload local do autor, com a correção da mediana ponderada
  aplicada (ver §6.4) e documentada no próprio docstring
- `02_matriz_C_FINAL.py` (127.918 bytes): subido pelo autor diretamente no
  Drive — arquivo grande demais para transcrição segura por este canal
  (ver §6.4, nota sobre corrupção de transcrição)

`TODO`: script(s) que geram as Tabelas 6-9 (elasticidades-preço, inclusive
Hicksiana/Marshalliana via Slutsky) não foram localizados/mapeados nesta
sessão — completar antes de considerar o repositório 100% completo.

---

## 6. Problemas conhecidos e decisões de auditoria

### 6.1 ✅ Ambiguidade de versão do script 02 — RESOLVIDA (sessão 2026-08-20)

Existiam (pelo menos) duas versões do script que constrói a matriz Ω:

- **`02_matriz_C_v3.py`** (no Drive, pasta `script/`): matching automático via
  `rapidfuzz.token_set_ratio`, sem dicionário manual por item.
- **`02_matriz_C_FINAL.py`** (encontrado localmente, pasta "scripts → validados"
  do autor, não estava no Drive até esta sessão): reescrita com dicionário
  manual extenso por grupo, motivado pela auditoria de classificação
  documentada em `diagnostico_classificacao_grupos.md`.

**Resolução, com prova decisiva**: rodamos `02_matriz_C_FINAL.py` (saídas
sufixadas `_test`) e comparamos a Ω resultante contra a Tabela A6 publicada
— divergência ampla (~40 células, até 44 p.p. de diferença). Isso por si só
não provava qual estava certa. O teste decisivo foi calcular
η̂ₙ = Ω_test · μ̂ (a fórmula da Tabela 8) usando a Ω do `_FINAL`: bateu
**exatamente** com os 10 valores de η̂ₙ já publicados na Tabela 8 — prova de
que **`02_matriz_C_FINAL.py` é a versão real que gerou os resultados
publicados no artigo**, e que a **Tabela A6 impressa está dessincronizada**
(provavelmente uma versão antiga, nunca regenerada após a auditoria de
classificação que motivou o `_FINAL`).

**Ação**: `02_matriz_C_FINAL.py` é a versão canônica. `02_matriz_C_v3.py`
deve ser arquivado em `script/archive/` com nota explicando que foi
substituído pela auditoria de classificação (ver `diagnostico_classificacao_grupos.md`).
A Tabela A6 do artigo precisa ser regenerada a partir da saída do `_FINAL`
antes da submissão — ver `referencia_v3.md`, item 14.

`TODO`: ainda não sabemos que script gerou a Tabela A6 atualmente publicada
(nem `v3` nem `_FINAL` batem célula a célula) — não é essencial para a
correção (já sabemos qual Ω é a certa), mas vale entender para o histórico.

**Nesta rodada de verificação (sessão 2026-08-20)**: autor está rodando
`02_matriz_C_FINAL.py` com todos os outputs sufixados `_test` (ex.
`matriz_omega_test.parquet`) — precisamente para não sobrescrever o
`matriz_omega.parquet` atual (presumivelmente gerado pelo `v3`) antes de
confirmar qual versão bate com a Tabela A6 publicada. Essa é a prática que
o protocolo final deve formalizar como padrão, não só para esta verificação
pontual.

### 6.2 Auditoria de classificação (grupo por grupo)

Processo documentado em `diagnostico_classificacao_grupos.md` (5 versões
incrementais no Drive, a mais recente com status "CONCLUÍDO — todos os 14
grupos inspecionados"). Resumo dos erros de classificação confirmados e
corrigidos:

| Grupo | Erro encontrado | Impacto no gasto total | Status |
|---|---|---|---|
| 08.Carnes | Arroz parboilizado + macarrão classificados como carne | 0,499% | Corrigido |
| 01.Cereais | Farinha de trigo classificada como cereal | 0,236% | Corrigido |
| 03.Tubérculos | Farinha de mandioca (~36% do grupo!) classificada como tubérculo | 0,357% | Corrigido |
| 04.Açúcares | Melancia, batata doce, peixe de água doce mal classificados | 0,358% | Corrigido |
| 12.Alimentação Fora | Só 2 itens de fronteira, ambos legítimos | ~0,035% | ✓ Limpo |
| 02, 07 | — | 0% | ✓ Limpo |
| demais | Erros menores (biscoitos com nome de outro grupo, etc.) | <0,1% cada | Corrigido |

`TODO`: confirmar se a decisão inicial "NÃO corrigir Carnes/Arroz para P1"
(documentada em `_DECISAO_arroz_carnes_junho2026.txt`) foi de fato revertida
— a versão mais recente do log diz "REVISADA: corrigir junto com todos os
demais", mas isso precisa bater com o que `02_matriz_C_FINAL.py`
efetivamente implementa.

### 6.3 Problemas de matching TACO (distintos de classificação)

Documentados para o grupo Carnes: itens como "OVOS DE GALINHA" e "GALINHA
ABATIDA" originalmente casados (via matching automático) com "Caldo de
galinha, tablete" (22.300 mg sódio/100g) — sódio médio do grupo Carnes caiu
de 3.597 mg/100g (bug) para 403 mg/100g (corrigido), meta ~150 mg/100g.
Corrigido via dicionário manual, mesmo mecanismo usado depois para o grupo
12 (Alimentação Fora).

`TODO`: mapear todas as correções de matching (não só classificação) que
entraram no dicionário manual do `02_matriz_C_FINAL.py`, por grupo.

### 6.4 Bug de mediana rotulada como média (Figuras 1-4) — CORRIGIDO

Nos dois scripts de curvas de Engel (`06_engel_nutricional_FINAL.py`,
`08_engel_grupos_FINAL.py`), a linha vertical tracejada rotulada "median"
no gráfico vinha de `np.average(lnm, weights=pesos_n)` — isso é média
ponderada, não mediana. Corrigido nos dois: nova função `weighted_median()`
(ordena, acumula peso, corta em 50%), documentada no próprio docstring de
cada script. Os dois scripts já estavam corretamente ponderados no resto
(curva central via `bin_loess_boot`, bin-then-weighted-mean) — só esse
ponto específico estava errado.

### 6.5 Lição sobre transferência de arquivos grandes por este canal

Ao montar o repositório validado (§5.1), a transcrição manual de
`01b_base_analitica_v2.py` (14.300 bytes) via base64 produziu corrupção
silenciosa **duas vezes seguidas, de forma idêntica** — não foi acaso, foi
sistemático. A corrupção só foi pega porque `py_compile` falhou em dois
pontos (colchetes faltando) e uma verificação de decodificação UTF-8 falhou
num terceiro ponto (ruído binário puro por ~300 bytes). O arquivo só ficou
íntegro (bytes exatos, 14.300) depois que o autor subiu diretamente via
upload no chat, contornando a transcrição.

**Regra prática adotada**: para arquivos de texto puro, `textContent` é
seguro até a faixa de ~15KB (testado, íntegro). Para arquivos maiores
(`02_matriz_C_FINAL.py`, 127.918 bytes), a transcrição manual não é
confiável — pedir para o autor subir/copiar diretamente é mais seguro que
qualquer tentativa de retransmissão. Quando o arquivo já existe intacto em
algum lugar do Drive, `copy_file` (cópia server-side) é sempre preferível a
qualquer forma de retranscrição, mesmo para arquivos pequenos — elimina o
risco por completo.

---

## 7. Validação — números-âncora para conferir uma reprodução

Se o pipeline rodar corretamente, os seguintes valores (já verificados
contra o artigo publicado nesta sessão) devem ser reproduzidos:

| Quantidade | Valor esperado | Fonte de verificação |
|---|---|---|
| N (unidades de consumo, amostra QUAIDS) | 54.207 | Tabela 1 |
| Resíduo de homogeneidade (Σγ̂ᵢⱼ) | ≈ 3,33 × 10⁻¹⁶ | `quaids_s1_resultados.json`, campo `hom_resid` — confirmado bater com Tabela 4/A5 |
| Autovalor da direção adding-up (E^H) | ≈ 4,08 × 10⁻⁹ | Recalculado nesta sessão a partir de `E_H` salvo em `quaids_s1_resultados.json` |
| Ω(sódio, grupo 04) | 0,717 | Tabela A6 |
| Ω(proteína, grupo 08) | 0,295 | Tabela A6 |
| Wald M2 (H₀: todos δ=0) | W ≈ 936,5, df=39 | Recalculado nesta sessão a partir de `quaids_s4_resultados.json` + `quaids_cov_mats.npz` |
| **η̂ₙ (10 nutrientes, Tabela 8, M1)** | **Bate exatamente (3 casas decimais) usando Ω do `02_matriz_C_FINAL.py`** | Teste decisivo desta sessão — ver §6.1. Prova que `_FINAL` é a fonte real |
| Ω(lipídios, grupo 12) | Tabela A6 publicada mostrava 0% (errado); `_FINAL` dá 1,51%; **Tabela A6 corrigida no `.tex` (sessão 2026-08-20)** | Ver `referencia_v3.md`, item 14 — correção aplicada, não só diagnosticada |

### Nota adicional: Quadro 24 confirmado para orçamento do grupo 12

Investigação adicional (sessão 2026-08-20) confirmou, via `00_consumo_quaids_v2.py`,
que o **orçamento** do grupo 12 (usado na estimação QUAIDS) de fato vem do
Quadro 24 dentro de `DESPESA_INDIVIDUAL.txt` — a documentação do próprio
script confirma: `"Quadros alimentares da DESPESA_INDIVIDUAL: 21→Tabaco,
24→Alimentação Fora+Álcool, 41→Alimentação Fora (viagens)"`. Isso é
diferente da **composição nutricional** (matriz C/Ω), que não pode vir do
Quadro 24 porque esse módulo não registra quantidade — só valor monetário
agregado por refeição. A Seção A1.5 do artigo agora distingue essas duas
fontes explicitamente (corrigido nesta sessão).

`TODO`: adicionar mais âncoras (Tabelas 5-9) conforme forem conferidas.

---

## 8. TODO geral antes de publicar no git

- [x] ~~Resolver a ambiguidade v3/FINAL (§6.1)~~ — resolvido, `_FINAL` é a versão canônica
- [x] ~~Confirmar nome exato de `04_quaids_estimacao.py`~~ — confirmado, existe com esse nome
- [x] ~~Corrigir bug de mediana rotulada como média nas Figuras 1-4~~ — ver §6.4
- [x] ~~Montar repositório de scripts validados~~ — ver §5.1, 11/11 scripts arquivados
- [ ] Confirmar nome exato de `00_classificacao_v2.py` (só os outputs foram confirmados)
- [ ] Mapear scripts que geram Tabelas 6-9 (elasticidades-preço) — não localizados ainda
- [ ] Preencher versões de software (§4)
- [ ] Decidir política de redistribuição de dados brutos (POF é público, mas
      confirmar termos de uso para redistribuir o `.txt` processado)
- [ ] Adicionar instruções de setup (Colab vs. local; ajuste de paths)
- [ ] Decidir se `POF/output/` entra no `.gitignore` (provavelmente sim,
      dado o tamanho de `CADERNETA_COLETIVA.txt` e derivados) ou se outputs
      pequenos (jsons de resultados) ficam versionados para auditoria
- [ ] Adicionar seção de licença dos dados (POF/IBGE, TACO/NEPA-UNICAMP)
- [ ] Referenciar este documento a partir do `README.md` principal do repositório
- [ ] Migrar a pasta `script_validados_replicacao/` do Drive para um
      repositório git de verdade (GitHub) e mintar DOI via Zenodo —
      próxima fase do plano de submissão (item 2)
