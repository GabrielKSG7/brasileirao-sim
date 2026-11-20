"""Gerador do site estático (GitHub Pages).

Lê só o que está versionado em ``results/backtest/`` e o seed de nomes de times do dbt, então roda
em qualquer máquina e no CI sem precisar dos dados brutos. Saída: ``site/index.html`` (uma página
autocontida, sem CDN) e ``site/data.json`` com os números exibidos.

    python -m brasileirao_sim.site --out site
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brasileirao_sim.metrics import calibration_table, match_scores
from brasileirao_sim.tune import HOLDOUT_SEASON as HOLDOUT

RESULTS = Path("results/backtest")
ALIASES = Path("dbt/seeds/team_aliases.csv")
MODEL_LABEL = {
    "dixon_coles": "Dixon-Coles",
    "elo": "Elo",
    "tabela": "Tabela atual",
    "mercado": "Mercado (odds)",
}
MODEL_ORDER = ["mercado", "dixon_coles", "elo", "tabela"]
MODEL_COLOR = {
    "mercado": "#fbbf24",
    "dixon_coles": "#34d399",
    "elo": "#60a5fa",
    "tabela": "#94a3b8",
}
LINE_COLORS = ["#34d399", "#fbbf24", "#60a5fa", "#f472b6", "#a78bfa", "#fb923c"]
EVENTS = [
    ("titulo", "final_titulo"),
    ("g4", "final_g4"),
    ("sul_americana", "final_sul_americana"),
    ("rebaixamento", "final_rebaixamento"),
]


# ---------------------------------------------------------------- dados


def team_names(path: Path = ALIASES) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["team_id"], df["team_name"], strict=False))


def load_results(results: Path = RESULTS):
    teams = pd.read_parquet(results / "team_probs.parquet")
    games = pd.read_parquet(results / "match_preds.parquet")
    finals = pd.read_parquet(results / "final_positions.parquet")
    meta = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    return teams, games, finals, meta


def match_metrics(games: pd.DataFrame, holdout: int = HOLDOUT) -> pd.DataFrame:
    """Média de RPS, log loss e Brier por modelo, em validação (antes de 2026) e em 2026."""
    rows = []
    for model, g in games.groupby("model"):
        for period, sel in (
            ("validacao", g[g["season"] < holdout]),
            ("holdout", g[g["season"] == holdout]),
        ):
            if sel.empty:
                continue
            s = match_scores(sel[["p_home", "p_draw", "p_away"]].to_numpy(), sel["result"])
            rows.append({"model": model, "periodo": period, "jogos": len(sel), **s.mean().to_dict()})
    return pd.DataFrame(rows)


def season_brier_by_round(teams: pd.DataFrame, finals: pd.DataFrame) -> pd.DataFrame:
    """Brier médio das quatro zonas, por rodada, nas temporadas completas."""
    m = teams.merge(finals.drop(columns=["final_position", "final_points"]),
                    on=["season", "team_id"], how="inner")
    brier = sum((m[p] - m[f].astype(float)) ** 2 for p, f in EVENTS) / len(EVENTS)
    return (
        m.assign(brier=brier).groupby(["model", "round"], as_index=False)["brier"].mean()
    )


def pooled_calibration(games: pd.DataFrame, model: str) -> pd.DataFrame:
    g = games[games["model"] == model]
    p = np.concatenate([g["p_home"], g["p_draw"], g["p_away"]])
    o = np.concatenate([g["result"] == "H", g["result"] == "D", g["result"] == "A"])
    return calibration_table(p, o.astype(float), bins=10)


# ---------------------------------------------------------------- SVG


def _fmt(x: float) -> str:
    return f"{x:.1f}"


def line_chart(series, x_max, *, width=860, height=360, y_max=1.0, y_fmt="{:.0%}",
               x_label="rodada", y_ticks=None, label_end=True, vline=None):
    """series: lista de (rótulo, cor, [(x, y), ...]). Devolve <svg>."""
    ml, mr, mt, mb = 52, 120 if label_end else 20, 14, 36
    w, h = width - ml - mr, height - mt - mb

    def sx(x):
        return ml + w * x / x_max

    def sy(y):
        return mt + h * (1 - y / y_max)

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    ticks = y_ticks or [i * y_max / 4 for i in range(5)]
    for t in ticks:
        out.append(f'<line x1="{ml}" x2="{ml + w}" y1="{_fmt(sy(t))}" y2="{_fmt(sy(t))}" class="grid"/>')
        out.append(f'<text x="{ml - 8}" y="{_fmt(sy(t) + 4)}" class="axis" text-anchor="end">'
                   f"{y_fmt.format(t)}</text>")
    step = max(1, x_max // 8)
    for x in range(0, x_max + 1, step):
        out.append(f'<text x="{_fmt(sx(x))}" y="{height - 14}" class="axis" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{ml + w / 2}" y="{height - 1}" class="axis" text-anchor="middle">{x_label}</text>')
    if vline is not None:
        out.append(f'<line id="vmarker" x1="{_fmt(sx(vline))}" x2="{_fmt(sx(vline))}" y1="{mt}" '
                   f'y2="{mt + h}" class="vline" data-ml="{ml}" data-w="{w}" data-xmax="{x_max}"/>')
    ends = []
    for label, color, pts in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{_fmt(sx(x))},{_fmt(sy(y))}" for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" '
                   f'stroke-linejoin="round"><title>{html.escape(label)}</title></path>')
        if label_end and pts and pts[-1][1] >= 0.01 * y_max:
            ends.append([sy(pts[-1][1]), label, color])
    # rótulos finais: ordena por altura e afasta quem ficou a menos de 14px
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for y, label, color in ends:
        out.append(f'<text x="{_fmt(ml + w + 8)}" y="{_fmt(y + 4)}" fill="{color}" '
                   f'class="endlabel">{html.escape(label)}</text>')
    out.append("</svg>")
    return "".join(out)


def calibration_chart(tables: dict[str, pd.DataFrame], *, size=420):
    ml, mb, mt, mr = 46, 40, 10, 14
    w, h = size - ml - mr, size - mt - mb

    def sx(x):
        return ml + w * x

    def sy(y):
        return mt + h * (1 - y)

    out = [f'<svg viewBox="0 0 {size} {size}" role="img" class="chart">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line x1="{ml}" x2="{ml + w}" y1="{_fmt(sy(t))}" y2="{_fmt(sy(t))}" class="grid"/>')
        out.append(f'<text x="{ml - 8}" y="{_fmt(sy(t) + 4)}" class="axis" text-anchor="end">{t:.0%}</text>')
        out.append(f'<text x="{_fmt(sx(t))}" y="{size - 22}" class="axis" text-anchor="middle">{t:.0%}</text>')
    out.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" class="diag"/>')
    out.append(f'<text x="{ml + w / 2}" y="{size - 4}" class="axis" text-anchor="middle">'
               "probabilidade prevista</text>")
    for model, tab in tables.items():
        pts = " ".join(f"{_fmt(sx(a))},{_fmt(sy(b))}" for a, b in zip(tab["prob_media"],
                                                                         tab["freq_observada"], strict=True))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{MODEL_COLOR[model]}" stroke-width="2"/>')
        for a, b, n in zip(tab["prob_media"], tab["freq_observada"], tab["n"], strict=True):
            out.append(f'<circle cx="{_fmt(sx(a))}" cy="{_fmt(sy(b))}" r="3.5" fill="{MODEL_COLOR[model]}">'
                       f"<title>{MODEL_LABEL[model]}: previsto {a:.1%}, observado {b:.1%} (n={n})</title></circle>")
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- conteúdo


def _pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def _num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def title_story(teams: pd.DataFrame, names: dict[str, str], season: int, ref_round: int = 20) -> dict:
    t = teams[(teams["season"] == season) & (teams["model"] == "dixon_coles")]
    last = int(t["round"].max())
    ref_round = min(ref_round, last)
    ref = t[t["round"] == ref_round].sort_values("titulo", ascending=False)
    now = t[t["round"] == last].sort_values("titulo", ascending=False)
    leader_by_round = t.loc[t.groupby("round")["titulo"].idxmax(), ["round", "team_id"]]
    switches = int((leader_by_round["team_id"] != leader_by_round["team_id"].shift()).sum() - 1)
    return {
        "ref_round": ref_round, "last_round": last, "switches": switches,
        "ref": [(names.get(r.team_id, r.team_id), float(r.titulo)) for r in ref.head(2).itertuples()],
        "now": [(names.get(r.team_id, r.team_id), float(r.titulo)) for r in now.head(2).itertuples()],
        "cutoff_ref": str(ref["cutoff"].iloc[0]), "cutoff_now": str(now["cutoff"].iloc[0]),
    }


def explorer_payload(teams: pd.DataFrame, names: dict[str, str], season: int) -> dict:
    out = {}
    t = teams[teams["season"] == season]
    for model, g in t.groupby("model"):
        rounds = {}
        for r, gr in g.groupby("round"):
            gr = gr.sort_values(["pontos_atuais", "posicao_media"], ascending=[False, True])
            rounds[int(r)] = {
                "cutoff": str(gr["cutoff"].iloc[0]),
                "rows": [
                    [names.get(x.team_id, x.team_id), int(x.pontos_atuais), int(x.jogos),
                     round(float(x.titulo), 4), round(float(x.g4), 4),
                     round(float(x.sul_americana), 4), round(float(x.rebaixamento), 4)]
                    for x in gr.itertuples()
                ],
            }
        out[model] = rounds
    return out


CSS = """
:root{--bg:#0a110e;--panel:#111b16;--line:#1e2d26;--text:#e7f0ea;--muted:#8da598;
--green:#34d399;--gold:#fbbf24;--red:#f87171;--blue:#60a5fa}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,-apple-system,"Segoe UI",
Roboto,sans-serif}
a{color:var(--green)}
.wrap{max-width:980px;margin:0 auto;padding:0 20px}
header.hero{padding:64px 0 36px;background:radial-gradient(900px 300px at 20% -10%,#12382a 0,transparent 70%);
border-bottom:1px solid var(--line)}
.eyebrow{color:var(--green);font-weight:600;letter-spacing:.12em;font-size:.78rem;text-transform:uppercase}
h1{font-size:clamp(2rem,5vw,3.1rem);line-height:1.1;margin:.3em 0}
h2{font-size:1.5rem;margin:0 0 .3em}h3{font-size:1.05rem;margin:1.2em 0 .4em}
.lead{color:var(--muted);max-width:680px;font-size:1.08rem}
section{padding:44px 0;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 20px;max-width:720px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:28px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card .big{font-size:1.9rem;font-weight:700;line-height:1.15}
.card .lbl{color:var(--muted);font-size:.86rem}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}
svg.chart{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}.axis{fill:var(--muted);font-size:11px}
.endlabel{font-size:12px;font-weight:600}.diag{stroke:#476357;stroke-dasharray:5 5}
.vline{stroke:#e7f0ea;stroke-opacity:.45;stroke-dasharray:3 4}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);font-size:.93rem}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600;font-size:.8rem;
text-transform:uppercase;letter-spacing:.05em}
td.best{color:var(--green);font-weight:700}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 0;color:var(--muted);font-size:.86rem}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.story{border-left:3px solid var(--gold);padding:4px 0 4px 16px;margin:0 0 22px}
.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.controls input[type=range]{flex:1;min-width:200px;accent-color:var(--green)}
select{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
.bar{position:relative;height:18px;min-width:90px;background:#0d1612;border-radius:4px;overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;display:block}
.bar span{position:relative;z-index:1;padding:0 6px;font-size:.8rem;line-height:18px}
.g4 i{background:#1d6b4f}.sul i{background:#25508a}.tit i{background:#8a6a14}.reb i{background:#8a3535}
.note{color:var(--muted);font-size:.88rem}
footer{padding:30px 0 60px;color:var(--muted);font-size:.9rem}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.two{grid-template-columns:1fr}.bar{min-width:60px}}
"""

JS = """
const D=JSON.parse(document.getElementById('data').textContent);
const sel=document.getElementById('model'),rng=document.getElementById('round'),
 body=document.getElementById('tbody'),lab=document.getElementById('rlabel');
function bar(cls,p){return `<div class="bar ${cls}"><i style="width:${(p*100).toFixed(1)}%"></i>`+
 `<span>${(p*100).toFixed(p>0&&p<0.1?1:0).replace('.',',')}%</span></div>`}
function draw(){
 const r=rng.value,m=D.explorer[sel.value][r];if(!m)return;
 lab.textContent=`Rodada ${r} · corte em ${m.cutoff.split('-').reverse().join('/')}`;
 body.innerHTML=m.rows.map((x,i)=>`<tr><td>${i+1}. ${x[0]}</td><td>${x[1]}</td><td>${x[2]}</td>`+
  `<td>${bar('tit',x[3])}</td><td>${bar('g4',x[4])}</td><td>${bar('sul',x[5])}</td>`+
  `<td>${bar('reb',x[6])}</td></tr>`).join('');
 const v=document.getElementById('vmarker');
 if(v){const ml=+v.dataset.ml,w=+v.dataset.w,xm=+v.dataset.xmax;
  const x=ml+w*r/xm;v.setAttribute('x1',x);v.setAttribute('x2',x)}
}
sel.onchange=draw;rng.oninput=draw;rng.value=D.story.ref_round;draw();
"""


def build_html(teams, games, finals, meta, names) -> tuple[str, dict]:
    holdout = HOLDOUT
    t26 = teams[(teams["season"] == holdout) & (teams["model"] == "dixon_coles")]
    last = int(t26["round"].max())
    complete = holdout in set(finals["season"])
    story = title_story(teams, names, holdout)
    metrics = match_metrics(games, holdout)

    # --- gráfico de título
    top = (t26.groupby("team_id")["titulo"].max().sort_values(ascending=False))
    top_ids = [i for i in top.index if top[i] >= 0.05][:6]
    series = []
    for k, tid in enumerate(top_ids):
        d = t26[t26["team_id"] == tid].sort_values("round")
        series.append((names.get(tid, tid), LINE_COLORS[k % len(LINE_COLORS)],
                       list(zip(d["round"], d["titulo"], strict=True))))
    title_svg = line_chart(series, last, vline=story["ref_round"])

    # --- Brier por rodada (validação)
    bb = season_brier_by_round(teams[teams["season"] < holdout], finals[finals["season"] < holdout])
    bser = []
    ymax = float(bb["brier"].max())
    for m in ("tabela", "elo", "dixon_coles"):
        d = bb[bb["model"] == m].sort_values("round")
        bser.append((MODEL_LABEL[m], MODEL_COLOR[m], list(zip(d["round"], d["brier"], strict=True))))
    brier_svg = line_chart(bser, int(bb["round"].max()), width=480, height=420, label_end=False, y_max=round(ymax * 1.1 + 0.005, 2),
                           y_fmt="{:.2f}")

    # --- calibração
    cal = {m: pooled_calibration(games, m) for m in ("mercado", "dixon_coles", "elo")}
    cal_svg = calibration_chart(cal)

    # --- tabela de métricas
    def metric_table(period: str) -> str:
        sub = metrics[metrics["periodo"] == period].set_index("model")
        best = {c: sub[c].min() for c in ("log_loss", "rps", "brier")}
        rows = []
        for m in MODEL_ORDER:
            if m not in sub.index:
                continue
            r = sub.loc[m]
            cells = "".join(
                f'<td class="{"best" if r[c] == best[c] else ""}">{_num(r[c])}</td>'
                for c in ("log_loss", "rps", "brier")
            )
            rows.append(f"<tr><td>{MODEL_LABEL[m]}</td>{cells}</tr>")
        n = int(sub["jogos"].iloc[0])
        return (f"<table><thead><tr><th>Modelo</th><th>Log loss</th><th>RPS</th><th>Brier</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table><p class='note'>{n} jogos</p>")

    mk_v = metrics[(metrics.model == "mercado") & (metrics.periodo == "holdout")]["log_loss"].iloc[0]
    dc_v = metrics[(metrics.model == "dixon_coles") & (metrics.periodo == "holdout")]["log_loss"].iloc[0]
    tb_v = metrics[(metrics.model == "tabela") & (metrics.periodo == "holdout")]["log_loss"].iloc[0]

    cards = [
        (f"{story['switches']}", f"trocas de favorito ao título em {holdout}"),
        (_pct(story["now"][0][1], 1), f"{story['now'][0][0]}: chance de título na rodada {story['last_round']}"),
        (_num(dc_v), f"log loss do Dixon-Coles em {holdout} (tabela: {_num(tb_v)}, mercado: {_num(mk_v)})"),
        ("0", "jogos futuros usados em qualquer previsão (testado)"),
    ]
    cards_html = "".join(f'<div class="card"><div class="big">{a}</div><div class="lbl">{b}</div></div>'
                         for a, b in cards)

    ref = story["ref"]
    now = story["now"]
    story_html = (
        f"<p class='story'><b>Rodada {story['ref_round']}</b> (corte em "
        f"{story['cutoff_ref'][8:]}/{story['cutoff_ref'][5:7]}): o modelo dava "
        f"<b>{_pct(ref[0][1])}</b> de título ao {ref[0][0]} e <b>{_pct(ref[1][1])}</b> ao {ref[1][0]}. "
        f"Na rodada {story['last_round']}, {now[0][0]} aparece com <b>{_pct(now[0][1], 1)}</b>. "
        f"Mova o controle abaixo para ver o que o modelo sabia em cada momento.</p>"
    )
    status = ("A temporada terminou: colunas de probabilidade comparam com o resultado final."
              if complete else
              f"A temporada {holdout} ainda não terminou: a tabela mostra a classificação no corte "
              "escolhido, e as barras são o que o modelo previa naquela data.")

    meta_line = (f"Dados até {story['cutoff_now']} · {meta['simulacoes_holdout']:,} simulações por corte "
                 f"em {holdout} · gerado em {meta['gerado_em'][:10]}").replace(",", ".")

    data = {
        "story": {"ref_round": story["ref_round"], "last_round": last},
        "explorer": explorer_payload(teams, names, holdout),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    legend = "".join(
        f'<span><span class="dot" style="background:{c}"></span>{html.escape(n)}</span>'
        for n, c, _ in series
    )
    model_opts = "".join(f'<option value="{k}">{MODEL_LABEL[k]}</option>'
                         for k in ("dixon_coles", "elo", "tabela"))

    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>brasileirao-sim · simulador do Brasileirão com backtest</title>
<meta name="description" content="Probabilidades de título, G4, Sul-Americana e rebaixamento do Brasileirão, rodada a rodada, com backtest sem vazamento de dados.">
<style>{CSS}</style></head><body>
<header class="hero"><div class="wrap">
<div class="eyebrow">Brasileirão Série A · {holdout}</div>
<h1>O que o modelo dizia em cada rodada,<br>e o que aconteceu.</h1>
<p class="lead">Simulação de Monte Carlo com Dixon-Coles, avaliada em backtest rodada a rodada:
cada previsão usa apenas o que se sabia até a data de corte. Nada de olhar o futuro.</p>
<div class="cards">{cards_html}</div>
<p class="note" style="margin-top:18px">{meta_line}</p>
</div></header>

<section><div class="wrap">
<h2>Chance de título, rodada a rodada</h2>
<p class="sub">Dixon-Coles, {holdout}. Cada ponto é uma simulação de {meta['simulacoes_holdout']:,}
temporadas completas a partir do corte.</p>
{story_html}
<div class="panel">{title_svg}<div class="legend">{legend}</div></div>
</div></section>

<section><div class="wrap">
<h2>Explorador: o modelo na rodada X</h2>
<p class="sub">{status} A linha tracejada do gráfico acima acompanha a rodada escolhida.</p>
<div class="panel">
<div class="controls">
<label>Modelo <select id="model">{model_opts}</select></label>
<input id="round" type="range" min="0" max="{last}" step="1" value="{story['ref_round']}" aria-label="rodada">
<strong id="rlabel"></strong></div>
<div style="overflow-x:auto"><table><thead><tr><th>Time (por pontos)</th><th>Pts</th><th>J</th>
<th>Título</th><th>G4</th><th>Sul-Am.</th><th>Z4</th></tr></thead><tbody id="tbody"></tbody></table></div>
</div></div></section>

<section><div class="wrap">
<h2>Quão bom é o modelo?</h2>
<p class="sub">Métricas jogo a jogo (menor é melhor; em verde, o melhor). O mercado de apostas é o teto
realista: ele sabe de lesões e escalações, que o modelo não vê. Contra o baseline "a tabela de hoje",
o Dixon-Coles ganha com folga. Contra o Elo, é empate estatístico (intervalos de bootstrap em
<a href="https://github.com/GabrielKSG7/brasileirao-sim/blob/main/docs/METODOLOGIA.md">METODOLOGIA</a>).</p>
<div class="two">
<div class="panel"><h3 style="margin-top:0">Validação (2019–{holdout - 1})</h3>{metric_table('validacao')}</div>
<div class="panel"><h3 style="margin-top:0">Holdout ({holdout})</h3>{metric_table('holdout')}</div>
</div>
<p class="note">Parâmetros ajustados só com 2019–{holdout - 1} e congelados antes de olhar {holdout}.</p>
</div></section>

<section><div class="wrap">
<h2>Calibração e aprendizado ao longo da temporada</h2>
<div class="two">
<div><h3 style="margin-top:0">Quando diz 40%, acontece 40%?</h3>
<div class="panel">{cal_svg}
<div class="legend">{"".join(f'<span><span class="dot" style="background:{MODEL_COLOR[m]}"></span>{MODEL_LABEL[m]}</span>' for m in cal)}</div></div>
<p class="note">Vitória do mandante, empate e vitória do visitante agrupados por faixa. Quanto mais
perto da diagonal, melhor.</p></div>
<div><h3 style="margin-top:0">Erro nas zonas (Brier) por rodada</h3>
<div class="panel">{brier_svg}<div class="legend">{"".join(f'<span><span class="dot" style="background:{c}"></span>{n}</span>' for n, c, _ in bser)}</div></div>
<p class="note">Média das zonas título, G4, Sul-Americana e rebaixamento, temporadas 2019–{holdout - 1}.
O erro cai conforme a temporada avança. O modelo supera a tabela simples no começo; da metade em diante, a tabela empata ou fica ligeiramente à frente.</p></div>
</div>
</div></section>

<footer><div class="wrap">
<p>Projeto de portfólio de engenharia de dados: ingestão em Python, dbt + DuckDB, simulação em NumPy,
CI no GitHub Actions. Resultados de um backtest, não conselho de aposta.
<a href="https://github.com/GabrielKSG7/brasileirao-sim">Código no GitHub</a> ·
<a href="https://github.com/GabrielKSG7/brasileirao-sim/blob/main/docs/METODOLOGIA.md">Metodologia</a>.
Dados: <a href="https://www.football-data.co.uk/brazil.php">football-data.co.uk</a>.</p>
</div></footer>
<script id="data" type="application/json">{payload}</script>
<script>{JS}</script>
</body></html>"""
    summary = {
        "temporada": holdout, "ultima_rodada": last, "trocas_de_favorito": story["switches"],
        "metricas_jogo": metrics.round(5).to_dict(orient="records"),
        "gerado_em": meta["gerado_em"],
    }
    return page, summary


def build_site(out: Path, results: Path = RESULTS, aliases: Path = ALIASES) -> Path:
    teams, games, finals, meta = load_results(results)
    page, summary = build_html(teams, games, finals, meta, team_names(aliases))
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "data.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out / "index.html"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="brasileirao-sim-site")
    p.add_argument("--out", default="site")
    p.add_argument("--results", default=str(RESULTS))
    args = p.parse_args(argv)
    path = build_site(Path(args.out), Path(args.results))
    print(f"site gerado em {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
