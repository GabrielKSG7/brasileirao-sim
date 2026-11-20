-- Classificação final das temporadas completas.
select * from read_parquet('{{ var("results_dir") }}/final_positions.parquet')
