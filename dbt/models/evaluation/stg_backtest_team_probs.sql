-- Probabilidades por time, corte e modelo geradas pelo backtest (Python).
select * from read_parquet('{{ var("results_dir") }}/team_probs.parquet')
