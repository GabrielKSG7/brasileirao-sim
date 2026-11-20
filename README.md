# brasileirao-sim

Simulador do Brasileirão Série A com **backtest rodada a rodada**: para cada rodada da temporada 2026,
o modelo roda usando apenas o que se sabia até aquele momento, e as probabilidades de título,
Libertadores, Sul-Americana e rebaixamento são comparadas com o que aconteceu.

> Status: **em construção** (v0.4: backtest, métricas e gerador do site prontos; publicação no Pages em seguida).
> Metodologia e resultados completos em [docs/METODOLOGIA.md](docs/METODOLOGIA.md).

## Resultados até agora

- **2026, rodada a rodada:** o favorito ao título trocou 8 vezes. Na rodada 20, o modelo dava
  70% ao Palmeiras e 27% ao Flamengo; após 35 rodadas, Flamengo 98%.
- **Jogo a jogo (log loss, 2019–2025 / 2026):** mercado 1,001 / 0,970 · Dixon-Coles 1,021 / 1,003 ·
  Elo 1,021 / 0,995 · tabela 1,052 / 1,053.
- **Leitura honesta:** Dixon-Coles e Elo empatam estatisticamente; os dois batem com folga o
  "se continuar assim" e perdem para o mercado de apostas, que sabe de lesões e escalações.
- **Calibração:** das previsões entre 40% e 50%, 43,9% aconteceram.

## Ideia

1. **Dados:** resultados e odds de mercado do [football-data.co.uk](https://www.football-data.co.uk/brazil.php),
   tabela de jogos da temporada corrente via [football-data.org](https://www.football-data.org/) (plano gratuito).
2. **Modelo:** Dixon-Coles (Poisson de gols por time, mando de campo e decaimento temporal), com Elo e
   "extrapolar a tabela" como baselines.
3. **Simulação:** Monte Carlo do restante do campeonato (10 mil temporadas por rodada).
4. **Backtest:** corte por data (nunca por número de rodada), parâmetros ajustados só com 2019–2025 e
   congelados antes de tocar em 2026.
5. **Avaliação:** Brier, log loss, RPS e calibração, contra baselines e contra as odds de mercado.
6. **Site:** estático, gerado pelo pipeline e publicado no GitHub Pages; atualização semanal em 2027.

Custo zero: roda localmente e no GitHub Actions.

## Zonas consideradas (regulamento 2026)

Título; G4 (fase de grupos da Libertadores); 5º (fase preliminar); 6º ao 11º (Sul-Americana); Z4 (rebaixamento).
Na v1 as zonas são fixas por posição, sem a redistribuição de vagas que depende das copas.

## Arquitetura

```
ingestão (Python)  ->  data/raw/*.parquet + manifesto (hash, data da coleta)
dbt + DuckDB       ->  staging -> intermediate -> marts
Python (src/)      ->  data.py (corte por data) -> models/ -> simulate.py
results/backtest/  ->  previsões versionadas (Parquet)
dbt (evaluation/)  ->  métricas, calibração, precisão por rodada
python -m brasileirao_sim.cli site  ->  site/index.html (SVG + JS mínimo, sem CDN) -> GitHub Pages
```

## Como rodar (qualquer máquina)

```bash
git clone https://github.com/GabrielKSG7/brasileirao-sim.git
cd brasileirao-sim
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m brasileirao_sim.ingest.footballdata_uk     # baixa o BRA.csv e gera o Parquet
cd dbt && dbt build --profiles-dir .                 # staging + testes
cd .. && pytest

python -m brasileirao_sim.cli site --out site        # gera o site em site/index.html (não precisa dos dados brutos)
```

Para usar um CSV já baixado: `python -m brasileirao_sim.ingest.footballdata_uk --csv caminho/BRA.csv`.

### Simular a partir de uma data

```bash
python -m brasileirao_sim.cli simulate --season 2026 --as-of 2026-11-20
python -m brasileirao_sim.cli simulate --season 2026 --as-of 2026-07-29 --model elo --out out/r20.csv
```

Modelos: `dixon_coles` (principal), `elo` e `tabela` (baselines). Usa os parâmetros congelados
em `config/model_params.json`.

### Backtest e ajuste

```bash
python -m brasileirao_sim.tune --stage matches    # grade jogo a jogo em 2019–2025 (~10 min)
python -m brasileirao_sim.tune --stage season     # incerteza da simulação (~5 min)
python -m brasileirao_sim.cli backtest --seasons 2019-2026   # ~2 min, grava results/backtest/
cd dbt && dbt build --profiles-dir .              # métricas e calibração (modelos eval_*)
```

Os resultados do backtest (`results/`) são versionados: são o produto do projeto e permitem
conferir os números sem rodar nada.

## Proteção contra vazamento de dados futuros

- Os modelos só recebem resultados por `TrainingData`, criada por `as_of(corte)`, que filtra e
  **verifica** que nenhum jogo é posterior ao corte.
- Os jogos que faltam chegam ao simulador só como pares mandante x visitante, **sem placar**.
- Um teste de perturbação altera todos os placares posteriores ao corte e exige que a previsão
  saia idêntica, para os três modelos.
- Os cortes por rodada usam a quantidade de jogos disputados (10 por rodada), não o número
  oficial da rodada, porque jogos adiados embaralham a ordem.

Os dados brutos e o `.duckdb` não vão para o Git: são regerados pela ingestão.

## Fontes e créditos

- Resultados e odds: [football-data.co.uk](https://www.football-data.co.uk/) (dados gratuitos; conferir os termos do site).
- Calendário da temporada corrente: [football-data.org](https://www.football-data.org/).

## Roteiro

- [x] Estrutura, ingestão e staging
- [x] Dixon-Coles, Elo, baseline da tabela e simulação Monte Carlo
- [x] Ajuste de hiperparâmetros em 2019–2025 (congelados em `config/model_params.json`)
- [x] Backtest 2019–2026 e métricas (RPS, log loss, Brier, calibração, bootstrap) no dbt
- [ ] Avaliação de temporada de 2026 após a última rodada (02/12/2026)
- [ ] Site estático
- [ ] v1.0.0
- [ ] Atualização semanal em 2027

## Licença

MIT
