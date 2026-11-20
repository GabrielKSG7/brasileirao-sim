# Metodologia e resultados do backtest

Documento de referência do projeto: como o modelo foi construído, como foi avaliado sem
vazamento de informação e o que os números mostram, inclusive onde ele perde.

## 1. Dados

- Resultados e odds de fechamento do Brasileirão Série A, 2012–2026, do
  [football-data.co.uk](https://www.football-data.co.uk/brazil.php) (`BRA.csv`).
- Conferência: a classificação calculada a partir do arquivo reproduz campeões, pontuações e
  rebaixados reais de 2019 a 2025 (ex.: Flamengo 90 pontos em 2019, Botafogo campeão em 2024).
- Datas convertidas de UTC para o horário de Brasília antes dos cortes do backtest.
- 2026 está em andamento (35 rodadas, último jogo em 19/11/2026); o campeonato termina em 02/12.

## 2. Modelos

| Modelo | Ideia | Papel |
|---|---|---|
| **Dixon-Coles** | Gols de cada time ~ Poisson, com ataque, defesa, mando de campo e correção para placares baixos; jogos antigos pesam menos (decaimento exponencial) | principal |
| **Elo** | Rating atualizado jogo a jogo; logit ordenado converte diferença de rating em vitória/empate/derrota | baseline forte |
| **Tabela** | "Se continuar assim": taxas de vitória/empate/derrota de cada time na temporada | baseline ingênuo |
| **Mercado** | Probabilidades implícitas na odd média de fechamento, sem a margem da casa | régua (só jogo a jogo) |

A simulação Monte Carlo sorteia o restante do campeonato milhares de vezes e conta em quantas
cada time termina em cada posição. Desempate: pontos, vitórias, saldo, gols pró, sorteio.

**Incerteza de parâmetro.** Simular com as forças estimadas como se fossem exatas deixa as
probabilidades de longo prazo confiantes demais. Em cada simulação, a força de cada time recebe
um desvio aleatório (Dixon-Coles: desvio-padrão de 0,20 no ataque e na defesa; Elo: 75 pontos
de rating), escolhido na validação. Isso reduziu o Brier das probabilidades de temporada em
cerca de 2% (Dixon-Coles) e 3% (Elo).

## 3. Como o backtest evita vazamento

1. **Corte por data.** Para cada temporada há um corte na véspera do primeiro jogo e depois um a
   cada 10 jogos disputados ("rodada" = 10 jogos, porque jogos adiados embaralham a numeração
   oficial).
2. **O modelo só vê o passado.** Os resultados chegam aos modelos apenas por `TrainingData`,
   criada por `as_of(corte)`, que filtra e verifica que nenhum jogo é posterior ao corte.
3. **Jogos futuros sem placar.** O simulador recebe os confrontos que faltam só como pares
   mandante x visitante (em pontos corridos, eles são conhecidos antes do campeonato).
4. **Testes automáticos.** Um teste de perturbação altera todos os placares posteriores ao corte
   e exige previsão idêntica nos três modelos; um teste no dbt garante que nenhuma previsão tem
   corte igual ou posterior à data do jogo.
5. **2026 como teste puro.** Todos os hiperparâmetros foram escolhidos só com 2019–2025 e
   congelados em [`config/model_params.json`](../config/model_params.json) antes de rodar 2026.

## 4. Ajuste de hiperparâmetros (2019–2025)

- **Etapa 1, jogo a jogo:** grade de parâmetros escolhida pela menor log loss das previsões da
  janela seguinte a cada corte (2.660 jogos).
  - Dixon-Coles (64 combinações): decaimento 0,0019/dia (meia-vida ~1 ano), ridge 5, prior de
    −0,1 no ataque e na defesa dos promovidos. O ótimo ficou no interior da grade.
  - Elo (81 + 54 combinações): K = 11, regressão de 20% à média entre temporadas, sem penalidade
    extra para promovidos. A primeira grade teve o ótimo na borda e foi estendida; as duas estão
    em `results/tuning/`.
- **Etapa 2, temporada:** desvio da incerteza de parâmetro escolhido pelo Brier médio das
  probabilidades por zona. A grade do Dixon-Coles também foi estendida após o ótimo cair na borda.

Ressalva: as métricas de temporada de 2019–2025 são *dentro da amostra* para o desvio da
simulação (etapa 2). As métricas jogo a jogo de 2019–2025 também orientaram a escolha dos
parâmetros. Só 2026 é avaliação totalmente fora da amostra.

## 5. Resultados

### 5.1 Jogo a jogo (menor é melhor)

| Modelo | Log loss 2019–25 | RPS 2019–25 | Log loss 2026 (teste) | RPS 2026 (teste) |
|---|---|---|---|---|
| Mercado | **1,0007** | **0,2013** | **0,9698** | **0,1918** |
| Dixon-Coles | 1,0208 | 0,2075 | 1,0034 | 0,2019 |
| Elo | 1,0214 | 0,2077 | 0,9954 | 0,2005 |
| Tabela | 1,0522 | 0,2171 | 1,0526 | 0,2172 |

Diferenças de log loss com intervalo de 95% (bootstrap pareado, 5.000 reamostragens):

| Comparação | 2019–25 | 2026 |
|---|---|---|
| Dixon-Coles − Elo | −0,0005 [−0,0051; +0,0038] | +0,0081 [−0,0021; +0,0179] |
| Dixon-Coles − Mercado | +0,0202 [+0,0134; +0,0269] | +0,0337 [+0,0160; +0,0516] |
| Dixon-Coles − Tabela | −0,0314 [−0,0397; −0,0233] | −0,0492 [−0,0731; −0,0249] |

**Leitura honesta:**
- Dixon-Coles e Elo estão **estatisticamente empatados**: o intervalo inclui zero nos dois períodos,
  e a vantagem alterna de temporada para temporada. Em 2026, o Elo foi nominalmente melhor.
- Os dois **batem com folga** o baseline "se continuar assim".
- Os dois **perdem com folga para o mercado**, que incorpora informação que um modelo só de
  placares não tem (lesões, escalações, viagens, rodízio por causa das copas). Bater o mercado
  com um modelo de placares seria sinal de erro, não de mérito.

### 5.2 Probabilidades de temporada (2019–2025, Brier, menor é melhor)

| Modelo | Título | G4 | Libertadores (G5) | Sul-Americana (6º–11º) | Rebaixamento |
|---|---|---|---|---|---|
| Dixon-Coles | 0,0285 | **0,0711** | **0,0833** | 0,1638 | 0,0903 |
| Elo | **0,0281** | 0,0732 | 0,0855 | 0,1636 | **0,0886** |
| Tabela | 0,0323 | 0,0756 | 0,0894 | **0,1621** | 0,0921 |

A precisão melhora ao longo da temporada (Dixon-Coles, média 2019–2025): o Brier do rebaixamento
cai de 0,139 na pré-temporada para 0,092 na rodada 19 e 0,036 na rodada 35. A faixa 6º–11º
(Sul-Americana) é quase imprevisível para todos os modelos: é o meio da tabela, onde poucos
pontos separam muitos times.

### 5.3 Calibração (Dixon-Coles, todas as zonas, 2019–2025)

| Faixa prevista | Previsões | Prob. média | Frequência observada |
|---|---|---|---|
| 10–20% | 2.584 | 14,8% | 15,7% |
| 20–30% | 2.040 | 24,8% | 27,2% |
| 30–40% | 2.001 | 35,2% | 33,5% |
| 40–50% | 1.593 | 44,6% | 43,9% |
| 50–60% | 990 | 54,5% | 52,3% |
| 60–70% | 669 | 64,6% | 58,1% |
| 70–80% | 556 | 74,6% | 79,3% |
| 80–90% | 487 | 84,8% | 88,3% |

Bem calibrado na maior parte da escala; a faixa de 60–70% mostra leve excesso de confiança.
As previsões não são independentes (o mesmo time aparece em vários cortes), então as faixas
com poucas centenas de casos têm margem de erro de vários pontos.

### 5.4 Temporada 2026 (teste)

Probabilidade de título pelo Dixon-Coles com os parâmetros congelados:

| Rodada | Flamengo | Palmeiras |
|---|---|---|
| 0 (pré-temporada) | 28,6% | 24,8% |
| 10 | 25,4% | 56,4% |
| 15 | 50,5% | 42,2% |
| 20 | 27,1% | 69,8% |
| 25 | 60,9% | 37,7% |
| 30 | 60,2% | 39,8% |
| 35 | 98,4% | 1,6% |

O favorito trocou **8 vezes** ao longo da temporada. Um erro que vale registrar: na
pré-temporada, o modelo dava 58% de rebaixamento ao Coritiba, que chegou à rodada 35 em 16º, com
14% de risco.

A avaliação das probabilidades de temporada de 2026 (Brier, calibração) só pode ser feita depois
da última rodada, em 02/12/2026: basta rodar de novo a ingestão, o backtest de 2026 e o `dbt build`.

## 5b. Site

`brasileirao_sim.site` é uma função pura de `results/backtest/` e do seed de nomes: não lê dados
brutos, então roda no CI. Gera uma página única com gráficos em SVG (sem biblioteca externa) e um
explorador por rodada em JavaScript puro. Os números exibidos são recalculados dos Parquets, e um
teste confere as métricas do site contra um recálculo independente. Em 2026 (temporada em curso) a
tabela mostra a classificação no corte escolhido, não um resultado final. O workflow de publicação
está em `ci_templates/pages.yml` (copiar para `.github/workflows/`).

## 6. Limitações conhecidas

- Só placares: sem escalações, lesões, mercado de transferências ou calendário das copas.
- Zonas fixas por posição; a redistribuição de vagas da Libertadores pelas copas fica fora.
- Confronto direto não é usado no desempate (sorteio no lugar).
- O teste de 2026 tem 350 jogos e uma temporada: pouco para separar modelos parecidos.
