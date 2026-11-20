# brasileirao-sim

Simulador do Brasileirão Série A com **backtest rodada a rodada**: para cada rodada da temporada 2026,
o modelo roda usando apenas o que se sabia até aquele momento, e as probabilidades de título,
Libertadores, Sul-Americana e rebaixamento são comparadas com o que aconteceu.

> Status: **em construção** (v0.2: ingestão, modelos e simulação). O roteiro completo está abaixo.

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
dbt                ->  predições, probabilidades por time, métricas, calibração
site/              ->  páginas estáticas -> GitHub Pages
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
```

Para usar um CSV já baixado: `python -m brasileirao_sim.ingest.footballdata_uk --csv caminho/BRA.csv`.

### Simular a partir de uma data

```bash
python -m brasileirao_sim.cli simulate --season 2026 --as-of 2026-11-20
python -m brasileirao_sim.cli simulate --season 2026 --as-of 2026-07-29 --model elo --out out/r20.csv
```

Modelos: `dixon_coles` (principal), `elo` e `tabela` (baselines).

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
- [ ] Ajuste de hiperparâmetros em 2019–2025
- [ ] Backtest 2026 (parâmetros congelados)
- [ ] Métricas e calibração
- [ ] Site estático
- [ ] v1.0.0
- [ ] Atualização semanal em 2027

## Licença

MIT
