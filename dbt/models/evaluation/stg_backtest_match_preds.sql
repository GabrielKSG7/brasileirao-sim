-- Previsões jogo a jogo do backtest, incluindo o mercado (odd média de fechamento).
select * from read_parquet('{{ var("results_dir") }}/match_preds.parquet')
