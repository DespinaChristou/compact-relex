# -*- coding: utf-8 -*-
# HTML template for the findings dashboard. __DATA__ is replaced with JSON.
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sub-Billion, Super-Frontier — Relation Extraction Findings</title>
<meta name="description" content="Interactive findings dashboard: fine-tuned small language models (360M-3B) rival zero-shot frontier LLMs on general and literary relation extraction.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#f6f7fb; --panel:#ffffff; --ink:#0d1424; --ink2:#3a465c; --muted:#6b7688;
  --line:#e7eaf1; --line2:#eef1f7;
  --brand:#4f46e5; --brand2:#7c3aed; --accent:#06b6d4;
  --slm:#10b981; --rob:#6366f1; --gpt:#f59e0b; --claude:#ef4444;
  --gen:#3b82f6; --lit:#ec4899;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 8px 24px -12px rgba(16,24,40,.18);
  --shadow-lg:0 24px 60px -24px rgba(16,24,40,.30);
  --r:16px; --r-sm:10px;
  --maxw:1180px;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:78px}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--ink);line-height:1.5;-webkit-font-smoothing:antialiased;font-feature-settings:"cv02","cv03","cv04","tnum"}
.tnum{font-variant-numeric:tabular-nums}
a{color:inherit;text-decoration:none}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 24px}
section{padding:66px 0;scroll-margin-top:70px}
.eyebrow{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--brand);background:linear-gradient(90deg,rgba(79,70,229,.10),rgba(124,58,237,.10));padding:6px 12px;border-radius:100px;border:1px solid rgba(79,70,229,.16)}
h2.sec{font-size:clamp(26px,3.4vw,38px);font-weight:800;letter-spacing:-.02em;margin:16px 0 10px;line-height:1.08}
.sec-lead{font-size:16.5px;color:var(--ink2);max-width:760px;margin-bottom:30px}
.sec-lead b{color:var(--ink);font-weight:700}

/* NAV */
nav{position:sticky;top:0;z-index:50;background:rgba(246,247,251,.82);backdrop-filter:saturate(180%) blur(14px);border-bottom:1px solid var(--line)}
.nav-in{max-width:var(--maxw);margin:0 auto;padding:11px 24px;display:flex;align-items:center;gap:16px}
.brand{display:flex;align-items:center;gap:10px;font-weight:800;letter-spacing:-.02em;font-size:15px;white-space:nowrap}
.brand .dot{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,var(--brand),var(--brand2));display:grid;place-items:center;color:#fff;font-size:13px;font-weight:900;box-shadow:0 4px 12px -2px rgba(79,70,229,.5)}
.nav-links{display:flex;gap:2px;margin-left:auto;flex-wrap:wrap}
.nav-links a{font-size:13px;font-weight:600;color:var(--muted);padding:7px 12px;border-radius:100px;transition:.16s;white-space:nowrap}
.nav-links a:hover{color:var(--ink);background:#eef0f6}
.nav-links a.active{color:var(--brand);background:#fff;box-shadow:var(--shadow)}
.nav-cta{font-size:13px;font-weight:700;color:#fff!important;background:var(--ink);padding:8px 15px!important;border-radius:100px!important;box-shadow:var(--shadow)}
.nav-cta:hover{background:#000!important}
@media(max-width:1080px){.nav-links{display:none}}

/* HERO */
.hero{position:relative;overflow:hidden;background:#0b1020;color:#fff;padding:82px 0 90px}
.hero:before{content:"";position:absolute;inset:0;background:
  radial-gradient(60% 90% at 82% -10%,rgba(124,58,237,.55),transparent 60%),
  radial-gradient(50% 80% at 8% 0%,rgba(6,182,212,.35),transparent 60%),
  radial-gradient(60% 120% at 50% 120%,rgba(79,70,229,.45),transparent 55%);}
.hero:after{content:"";position:absolute;inset:0;opacity:.5;
  background-image:linear-gradient(rgba(255,255,255,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.045) 1px,transparent 1px);
  background-size:44px 44px;mask-image:radial-gradient(80% 80% at 50% 30%,#000,transparent 80%)}
.hero .wrap{position:relative;z-index:2}
.hero .tag{display:inline-flex;align-items:center;gap:8px;font-size:12.5px;font-weight:600;letter-spacing:.02em;color:#c7d2fe;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);padding:7px 14px;border-radius:100px;margin-bottom:22px}
.hero .tag .pip{width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 0 4px rgba(52,211,153,.22)}
.hero h1{font-size:clamp(34px,6vw,68px);line-height:1.02;font-weight:900;letter-spacing:-.035em;max-width:16ch}
.hero h1 .grad{background:linear-gradient(100deg,#a5b4fc,#67e8f9 55%,#6ee7b7);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero .sub{margin-top:22px;font-size:clamp(16px,2vw,20px);color:#cdd5e6;max-width:60ch;font-weight:400}
.hero .sub b{color:#fff;font-weight:600}
.hero .authors{margin-top:26px;font-size:14px;color:#93a0bd}
.hero .authors b{color:#e7ecf7;font-weight:600}
.hero-cta{margin-top:30px;display:flex;gap:12px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:8px;font-size:14px;font-weight:700;padding:12px 20px;border-radius:100px;transition:.18s;border:1px solid transparent;cursor:pointer}
.btn-primary{background:#fff;color:#0b1020}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 12px 28px -8px rgba(255,255,255,.35)}
.btn-ghost{background:rgba(255,255,255,.06);color:#fff;border-color:rgba(255,255,255,.22)}
.btn-ghost:hover{background:rgba(255,255,255,.14)}
.hero-strip{position:relative;z-index:2;margin-top:52px;display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.hs{background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.12);border-radius:14px;padding:16px 18px;backdrop-filter:blur(4px)}
.hs .v{font-size:26px;font-weight:800;letter-spacing:-.02em;color:#fff}
.hs .k{font-size:12px;color:#9fb0d0;margin-top:2px;font-weight:500}
@media(max-width:760px){.hero-strip{grid-template-columns:repeat(2,1fr)}}

/* KPI */
.kpi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:22px 22px 20px;box-shadow:var(--shadow);position:relative;overflow:hidden;transition:.2s}
.kpi:hover{transform:translateY(-3px);box-shadow:var(--shadow-lg)}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(var(--brand),var(--brand2))}
.kpi .v{font-size:38px;font-weight:900;letter-spacing:-.03em;line-height:1;background:linear-gradient(120deg,var(--ink),#334155);-webkit-background-clip:text;background-clip:text;color:transparent}
.kpi .u{font-size:12.5px;font-weight:700;color:var(--brand);margin-top:8px;text-transform:uppercase;letter-spacing:.05em}
.kpi .l{font-size:14.5px;font-weight:700;color:var(--ink);margin-top:10px}
.kpi .s{font-size:12.5px;color:var(--muted);margin-top:3px;line-height:1.45}
@media(max-width:820px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:520px){.kpi-grid{grid-template-columns:1fr}}

/* CARD / PANEL */
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow)}
.card-pad{padding:26px}
.card h3{font-size:17px;font-weight:800;letter-spacing:-.01em}
.card .desc{font-size:13.5px;color:var(--muted);margin-top:4px;max-width:70ch}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:900px){.grid-2{grid-template-columns:1fr}}

/* legend */
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:6px}
.legend .li{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;font-weight:600;color:var(--ink2)}
.legend .sw{width:12px;height:12px;border-radius:4px}
.chart-note{font-size:12px;color:var(--muted);margin-top:14px;line-height:1.5}
.chart-note code{background:#eef1f7;padding:1px 5px;border-radius:5px;font-size:11.5px}

/* svg chart */
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart text{font-family:'Inter',sans-serif}
.axis{stroke:var(--line);stroke-width:1}
.grid-l{stroke:var(--line2);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}
.bar{transition:opacity .15s}
.bar:hover{opacity:.82}
.blab{fill:var(--ink);font-size:10.5px;font-weight:700;text-anchor:middle}
.catlab{fill:var(--ink2);font-size:12.5px;font-weight:700;text-anchor:middle}

/* findings */
.find-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.find{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:24px 22px;box-shadow:var(--shadow);transition:.2s;position:relative}
.find:hover{transform:translateY(-3px);box-shadow:var(--shadow-lg);border-color:#dfe3ee}
.find .ic{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,rgba(79,70,229,.12),rgba(124,58,237,.12));display:grid;place-items:center;color:var(--brand);margin-bottom:15px}
.find .ic svg{width:22px;height:22px}
.find h4{font-size:16px;font-weight:800;letter-spacing:-.01em}
.find p{font-size:13.6px;color:var(--ink2);margin-top:8px;line-height:1.55}
.find .stat{margin-top:14px;display:inline-block;font-size:13px;font-weight:800;color:var(--slm);background:rgba(16,185,129,.10);padding:5px 11px;border-radius:8px;font-variant-numeric:tabular-nums}
@media(max-width:900px){.find-grid{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.find-grid{grid-template-columns:1fr}}

/* matrix */
.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
.ctl{display:flex;align-items:center;gap:8px}
.ctl .lbl{font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.segbtns{display:inline-flex;background:#eef0f6;border-radius:100px;padding:3px;gap:2px}
.segbtns button{border:0;background:transparent;font:inherit;font-size:12.5px;font-weight:700;color:var(--ink2);padding:6px 13px;border-radius:100px;cursor:pointer;transition:.15s}
.segbtns button.on{background:#fff;color:var(--brand);box-shadow:var(--shadow)}
.resetbtn{border:1px solid var(--line);background:#fff;font:inherit;font-size:12.5px;font-weight:700;color:var(--ink2);padding:7px 14px;border-radius:100px;cursor:pointer;transition:.15s}
.resetbtn:hover{border-color:var(--brand);color:var(--brand)}
.matrix-scroll{overflow-x:auto;border-radius:12px;border:1px solid var(--line)}
table.matrix{border-collapse:separate;border-spacing:0;width:100%;font-size:12px;min-width:900px}
table.matrix th{position:sticky;top:0;background:#fbfcfe;z-index:1;font-size:10.5px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);font-weight:800;padding:10px 6px;text-align:center;border-bottom:2px solid var(--line);white-space:nowrap;cursor:pointer;user-select:none}
table.matrix th.lft{text-align:left;padding-left:14px}
table.matrix th.sortable:hover{color:var(--brand)}
table.matrix th .arw{opacity:.35;font-size:9px}
table.matrix th.avg{background:#f2f0fb;color:#5b4bd6}
table.matrix td{padding:0;text-align:center;border-bottom:1px solid var(--line2)}
table.matrix td.meta{text-align:left;padding:8px 6px 8px 14px;white-space:nowrap;background:#fff;position:sticky;left:0;z-index:1;border-right:1px solid var(--line)}
.mdl{font-weight:700;color:var(--ink)}
.rg{display:inline-block;font-size:10px;font-weight:700;padding:1px 6px;border-radius:5px;margin-left:5px}
.rg.gen{background:#e0edff;color:#1d4ed8}.rg.lit{background:#fde4f1;color:#be185d}.rg.mix{background:#e6e2fb;color:#5b4bd6}
.shot{color:var(--muted);font-size:11px;font-weight:600}
.cell{height:34px;line-height:34px;font-weight:700;font-variant-numeric:tabular-nums;color:#0d1424;position:relative}
.cell.empty{color:#c3cad6;font-weight:500;background:repeating-linear-gradient(45deg,#fafbfd,#fafbfd 5px,#f1f3f8 5px,#f1f3f8 10px)}
.cell.avg{font-weight:800}
.cell .fl{position:absolute;top:2px;right:3px;font-size:8px;color:#b91c1c;font-weight:900}
tr.rrow:hover td{outline:2px solid rgba(79,70,229,.16);outline-offset:-2px}
.mx-caption{font-size:12px;color:var(--muted);margin-top:12px;line-height:1.55}
.flagbox{margin-top:12px;display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink2)}
.flagbox b{color:#b91c1c}

/* efficiency table */
.etable{width:100%;border-collapse:collapse;font-size:13px;margin-top:4px}
.etable th{text-align:right;padding:10px 12px;border-bottom:2px solid var(--line);font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:800}
.etable th:first-child,.etable td:first-child{text-align:left}
.etable td{padding:10px 12px;border-bottom:1px solid var(--line2);text-align:right;font-variant-numeric:tabular-nums}
.etable tr:hover td{background:#f8f9fc}
.etable .pill{font-size:11px;font-weight:800;padding:2px 8px;border-radius:100px}
.pill.sub{background:rgba(16,185,129,.12);color:#047857}
.pill.big{background:rgba(99,102,241,.12);color:#4338ca}
.pill.fr{background:rgba(239,68,68,.10);color:#b91c1c}
.f1b-bar{display:inline-block;height:8px;border-radius:4px;background:linear-gradient(90deg,#34d399,#10b981);vertical-align:middle;margin-left:8px}

/* two-col insight */
.insight{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;align-items:stretch}
@media(max-width:900px){.insight{grid-template-columns:1fr}}
.callout{background:linear-gradient(160deg,#0b1020,#161f3a);color:#fff;border-radius:var(--r);padding:26px;position:relative;overflow:hidden}
.callout:before{content:"";position:absolute;right:-40px;top:-40px;width:180px;height:180px;border-radius:50%;background:radial-gradient(circle,rgba(124,58,237,.5),transparent 70%)}
.callout .big{font-size:44px;font-weight:900;letter-spacing:-.03em;background:linear-gradient(100deg,#a5b4fc,#6ee7b7);-webkit-background-clip:text;background-clip:text;color:transparent}
.callout h4{font-size:16px;font-weight:800;margin-top:6px}
.callout p{font-size:13.5px;color:#c1cadf;margin-top:8px;line-height:1.6}

/* significance */
.sig-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
@media(max-width:760px){.sig-grid{grid-template-columns:1fr}}
.sig{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:20px 22px;box-shadow:var(--shadow);display:flex;gap:16px;align-items:flex-start}
.sig .badge{flex:none;font-size:12px;font-weight:800;padding:6px 12px;border-radius:100px;white-space:nowrap;font-variant-numeric:tabular-nums}
.sig .badge.pos{background:rgba(16,185,129,.12);color:#047857}
.sig .badge.neutral{background:#eef0f6;color:var(--ink2)}
.sig h4{font-size:15px;font-weight:800}
.sig p{font-size:13px;color:var(--ink2);margin-top:4px}

/* dapt table */
.dapt-t{width:100%;border-collapse:collapse;font-size:13.5px}
.dapt-t th{text-align:center;padding:9px 10px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:800;border-bottom:2px solid var(--line)}
.dapt-t th:first-child,.dapt-t td:first-child{text-align:left}
.dapt-t td{padding:9px 10px;text-align:center;border-bottom:1px solid var(--line2);font-variant-numeric:tabular-nums}
.dapt-t tr.dapt-row td{background:#f7f6fd}
.d0{color:var(--muted);font-weight:700}

/* about / cite */
.about{display:grid;grid-template-columns:1.3fr 1fr;gap:20px}
@media(max-width:860px){.about{grid-template-columns:1fr}}
.cite-box{background:#0b1020;color:#cdd6ea;border-radius:var(--r);padding:22px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.65;overflow-x:auto;position:relative}
.cite-box .cp{position:absolute;top:12px;right:12px;font-family:'Inter';font-size:11px;font-weight:700;color:#0b1020;background:#6ee7b7;border:0;padding:5px 11px;border-radius:8px;cursor:pointer}
.art-list{display:flex;flex-direction:column;gap:10px;margin-top:16px}
.art{display:flex;align-items:center;gap:12px;padding:13px 16px;border:1px solid var(--line);border-radius:12px;background:var(--panel);transition:.15s}
.art:hover{border-color:var(--brand);transform:translateX(3px);box-shadow:var(--shadow)}
.art .ai{width:34px;height:34px;border-radius:9px;background:#eef0f6;display:grid;place-items:center;flex:none;font-size:16px}
.art .at{font-size:13.5px;font-weight:700}
.art .as{font-size:12px;color:var(--muted);font-family:ui-monospace,monospace}
.art .go{margin-left:auto;color:var(--muted)}

footer{background:#0b1020;color:#8896b3;padding:38px 0;font-size:13px}
footer .wrap{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;align-items:center}
footer a{color:#c7d2fe;font-weight:600}
footer .fbrand{color:#fff;font-weight:800;font-size:15px}

.reveal{opacity:0;transform:translateY(16px);transition:opacity .6s cubic-bezier(.2,.7,.2,1),transform .6s cubic-bezier(.2,.7,.2,1)}
.reveal.in{opacity:1;transform:none}
.tip{cursor:help;border-bottom:1px dotted var(--muted)}
::selection{background:rgba(124,58,237,.22)}
</style>
</head>
<body>

<nav>
  <div class="nav-in">
    <a class="brand" href="#top"><span class="dot">RE</span><span>Compact&nbsp;RelEx</span></a>
    <div class="nav-links">
      <a href="#headline">Headline</a>
      <a href="#findings">Findings</a>
      <a href="#explorer">Explorer</a>
      <a href="#frontier">Frontier</a>
      <a href="#efficiency">Efficiency</a>
      <a href="#scale">Scale &amp; Prompt</a>
      <a href="#prompting">Prompt format</a>
      <a href="#baselines">Baselines</a>
      <a href="#stats">Significance</a>
    </div>
    <a class="nav-cta" id="ghlink" href="#about">Paper &amp; code</a>
  </div>
</nav>

<a id="top"></a>
<header class="hero">
  <div class="wrap">
    <div class="tag"><span class="pip"></span> 30 tuned configurations · 9 RE benchmarks · 3 frontier LLMs</div>
    <h1 id="heroTitle"></h1>
    <p class="sub" id="heroSub"></p>
    <div class="authors" id="heroAuthors"></div>
    <div class="hero-cta">
      <a class="btn btn-primary" href="#headline">Explore the findings ↓</a>
      <a class="btn btn-ghost" id="heroArxiv" target="_blank" rel="noopener">Read the paper</a>
      <a class="btn btn-ghost" id="heroGit" target="_blank" rel="noopener">GitHub repo</a>
    </div>
    <div class="hero-strip" id="heroStrip"></div>
  </div>
</header>

<main>

<!-- KPIs -->
<section id="overview">
  <div class="wrap reveal">
    <span class="eyebrow">The bottom line</span>
    <h2 class="sec">Fine-tuned small models close, then cross, the frontier gap</h2>
    <p class="sec-lead">Under a minimal zero-shot protocol, targeted task adaptation lets <b>4-bit models that fit on one consumer GPU</b> outperform general-purpose frontier systems on relation extraction, in both general-domain and literary text. Every figure below is scored with positive-class micro-F1 (no-relation excluded).</p>
    <div class="kpi-grid" id="kpiGrid"></div>
  </div>
</section>

<!-- HEADLINE -->
<section id="headline">
  <div class="wrap reveal">
    <span class="eyebrow">Headline result</span>
    <h2 class="sec">Small &amp; in-domain beats large &amp; zero-shot</h2>
    <p class="sec-lead">The strongest tuned SLM (best per benchmark) and an in-domain <b>RoBERTa</b> encoder both clear <b>GPT-5.4</b> and <b>Claude Sonnet 4.6</b> on both domain averages. That two very different in-domain systems win says the advantage is <b>task adaptation</b>, not generative decoding or scale.</p>
    <div class="card card-pad">
      <div class="grid-2" style="align-items:center;gap:32px">
        <div>
          <div id="headlineChart"></div>
          <div class="legend" id="headlineLegend"></div>
        </div>
        <div>
          <div class="callout">
            <div class="big">0.83&nbsp;<span style="font-size:20px;color:#93a0bd;-webkit-text-fill-color:#93a0bd">vs&nbsp;0.69</span></div>
            <h4>A 0.5B model tops a frontier LLM on general RE</h4>
            <p>Qwen2.5-0.5B fine-tuned on pooled general data reaches <b>0.83</b> General-Avg F1, above GPT-5.4 (0.69) and Claude Sonnet&nbsp;4.6 (0.66) evaluated zero-shot on the same full test sets — with roughly one-sixth the parameters of a 3B model and an undisclosed fraction of a frontier system's.</p>
          </div>
        </div>
      </div>
      <div class="chart-note">Frontier scores are the paper's <b>full-test-set</b> zero-shot numbers (GPT-5.4 at its default <code>reasoning=none</code>), not an earlier subsampled run. "Best tuned SLM" is the per-benchmark maximum over the 30 configurations, so its average is an upper envelope rather than one model. Gemini&nbsp;2.5&nbsp;Pro is omitted (schema-valid rate &lt; 0.11 on general RE).</div>
    </div>
  </div>
</section>

<!-- FINDINGS -->
<section id="findings">
  <div class="wrap reveal">
    <span class="eyebrow">Six things we learned</span>
    <h2 class="sec">What the 30-configuration study shows</h2>
    <p class="sec-lead">Five base models (360M–3B) × three domain-composition regimes (GenTune / LitTune / MixTune) × two prompt-conditioned tuning styles (0-shot / 2-shot), each fine-tuned with QLoRA and evaluated on nine benchmarks.</p>
    <div class="find-grid" id="findGrid"></div>
  </div>
</section>

<!-- EXPLORER -->
<section id="explorer">
  <div class="wrap reveal">
    <span class="eyebrow">Interactive</span>
    <h2 class="sec">Full results explorer</h2>
    <p class="sec-lead">Every tuned configuration on every benchmark. Filter by scale, regime, or prompt style; click a column header to sort. Cells are colored by F1 — <span style="color:#047857;font-weight:700">green is strong</span>, <span style="color:#d97706;font-weight:700">amber mid</span>, <span style="color:#e11d48;font-weight:700">red weak</span>. Specialists are only evaluated in their own domain (hatched cells).</p>
    <div class="card card-pad">
      <div class="controls">
        <div class="ctl"><span class="lbl">Scale</span><div class="segbtns" data-group="scale">
          <button class="on" data-v="all">All</button><button data-v="sub-billion">Sub-billion</button><button data-v="3B">3B</button></div></div>
        <div class="ctl"><span class="lbl">Regime</span><div class="segbtns" data-group="regime">
          <button class="on" data-v="all">All</button><button data-v="GenTune">GenTune</button><button data-v="LitTune">LitTune</button><button data-v="MixTune">MixTune</button></div></div>
        <div class="ctl"><span class="lbl">Prompt</span><div class="segbtns" data-group="shot">
          <button class="on" data-v="all">All</button><button data-v="0s">0-shot</button><button data-v="2s">2-shot</button></div></div>
        <div class="ctl" style="margin-left:auto"><button id="resetSort" class="resetbtn">↺ Reset sort</button></div>
      </div>
      <div class="matrix-scroll"><table class="matrix" id="matrixTable"></table></div>
      <div class="flagbox">
        <span><b>†</b> SmolLM3-3B MixTune 0-shot emits <code style="font-size:11px">&lt;think&gt;</code> tokens instead of a label and scores 0 under the default protocol (a post-hoc rescue recovers ~0.18; 2-shot removes it entirely).</span>
        <span><b>‡</b> Qwen2.5-3B GenTune 0-shot generates wrong-schema labels without demonstrations (0.28).</span>
      </div>
      <div class="mx-caption">Positive-class micro-F1, schema-enumerated prompting with matched prompt shots. <b>Gen</b>/<b>Lit</b>/<b>Overall</b> are dataset-macro averages over the 7 general, 2 literary, and all applicable benchmarks.</div>
    </div>
  </div>
</section>

<!-- FRONTIER -->
<section id="frontier">
  <div class="wrap reveal">
    <span class="eyebrow">Head to head</span>
    <h2 class="sec">Per-benchmark: tuned SLM vs frontier</h2>
    <p class="sec-lead">The best tuned SLM leads on <b>all nine</b> benchmarks. The margin is widest where task-specific supervision matters most — schema-heavy REBEL and Re-DocRED, and narrative PG-Fiction — and narrowest on near-saturated closed-schema sets.</p>
    <div class="card card-pad">
      <div id="frontierChart"></div>
      <div class="legend" id="frontierLegend"></div>
      <div class="chart-note">All models scored identically on full test sets. Best-SLM values are the per-benchmark maximum across the 30 configurations. On <b>CoNLL04</b>, relations are a one-to-one function of the ordered entity-type pair, so it is a type-determined ceiling for every system (a lookup scores 1.000).</div>
    </div>
  </div>
</section>

<!-- EFFICIENCY -->
<section id="efficiency">
  <div class="wrap reveal">
    <span class="eyebrow">Performance per watt-hour</span>
    <h2 class="sec">The efficiency frontier</h2>
    <p class="sec-lead">Accuracy is only half the story. Sub-billion models sit far above the frontier baselines while extracting many times more F1 per billion parameters, and they run in interactive time on commodity hardware — <b>~18–22 ms</b> on an RTX 4090, or CPU-only at <b>~120–180 ms</b>.</p>
    <div class="grid-2" style="gap:18px;align-items:stretch">
      <div class="card card-pad">
        <h3>F1 vs model size</h3>
        <div class="desc">Bubble area ∝ F1 per billion parameters. Dashed lines mark the two frontier baselines (undisclosed size, plotted as reference levels).</div>
        <div id="effScatter" style="margin-top:8px"></div>
        <div class="legend" id="effLegend"></div>
      </div>
      <div class="card card-pad">
        <h3>Deployment trade-offs</h3>
        <div class="desc">Representative configurations. F1/B is the sharpest lens on efficiency.</div>
        <div style="overflow-x:auto"><table class="etable" id="effTable"></table></div>
      </div>
    </div>
  </div>
</section>

<!-- SCALE & PROMPT -->
<section id="scale">
  <div class="wrap reveal">
    <span class="eyebrow">Why it works</span>
    <h2 class="sec">Scale barely moves the needle — prompt conditioning does</h2>
    <p class="sec-lead">Read within a model family, a large scale-up buys little. The lever that transforms the smallest models is <b>2-shot prompt-conditioned tuning</b>, whose gain is concentrated exactly where capacity is scarcest.</p>
    <div class="insight">
      <div class="card card-pad">
        <h3>Within-family scaling</h3>
        <div class="desc">Best overall F1 vs parameters (log axis). Lines connect same-family sizes only — cross-family gaps are confounded by tokenizer, data, and generation.</div>
        <div id="scaleChart" style="margin-top:10px"></div>
        <div class="chart-note">Cleanest same-generation contrast (Qwen2.5&nbsp;0.5B→3B): only <b>+0.037</b> overall F1 <span class="tnum">[+0.009,&nbsp;+0.067]</span>, and <b>−0.004</b> on general RE. SmolLM's larger <b>+0.132</b> also crosses a model generation, so it conflates size with better pretraining.</div>
      </div>
      <div class="card card-pad">
        <h3>0→2-shot F1 gain, by regime</h3>
        <div class="desc">Sub-billion models (top) gain sharply; 3B models (bottom) are already schema-saturated.</div>
        <div id="promptChart" style="margin-top:10px"></div>
        <div class="chart-note">Mean gain <b style="color:#047857">+{PSUB} for sub-billion</b> vs <b style="color:#4338ca">+{PBIG} for 3B</b> (2 decoding artifacts, marked †/‡, excluded from the 3B mean). Sub-billion gain is significant at <b>p&lt;0.001</b>.</div>
      </div>
    </div>
  </div>
</section>

<!-- PROMPT FORMAT: constrained vs open -->
<section id="prompting">
  <div class="wrap reveal">
    <span class="eyebrow">Prompt format</span>
    <h2 class="sec">Generic prompts beat schema-enumerated ones</h2>
    <p class="sec-lead">Each model is evaluated under two inference prompt formats: <b>generic</b> (a plain system prompt) and <b>schema-enumerated</b> (the allowed label set injected into the prompt). Counter-intuitively, generic prompting wins on <b id="coNPos">8 of 9</b> datasets with no loss of output well-formedness, so the schema-enumerated scores used in the headline tables are <b>conservative lower bounds</b>.</p>
    <div class="card card-pad">
      <div class="grid-2" style="align-items:center;gap:32px">
        <div>
          <div id="coChart"></div>
          <div class="legend" id="coLegend"></div>
        </div>
        <div class="callout">
          <div class="big" id="coOverall">+3.2 pp</div>
          <h4 id="coHeadline">Generic beats schema-enumerated on average</h4>
          <p>The gain is largest on GIDS (+12.9) and the small knowledge-oriented schemas; the sole exception is <b>CoNLL04</b>, whose relations are already fixed by the ordered entity-type pair. Smaller models, with less schema absorbed during tuning, benefit most.</p>
          <div style="margin-top:18px;display:flex;gap:12px">
            <div style="flex:1;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:12px 14px">
              <div style="font-size:23px;font-weight:800;color:#6ee7b7" id="coSub">+4.7</div>
              <div style="font-size:11.5px;color:#9fb0d0;margin-top:2px">Sub-billion (pp)</div></div>
            <div style="flex:1;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:12px 14px">
              <div style="font-size:23px;font-weight:800;color:#a5b4fc" id="coBig">+2.1</div>
              <div style="font-size:11.5px;color:#9fb0d0;margin-top:2px">3B (pp)</div></div>
          </div>
        </div>
      </div>
      <div class="chart-note">&Delta; = positive-class micro-F1 under generic minus schema-enumerated prompting (percentage points); matched prompt shots; the two decoding-artifact 0-shot configs excluded; error bars are SEM across the up-to-<span id="coMaxN">19</span> configurations per dataset; dashed line = overall mean. This is the axis on which the paper's headline numbers are deliberately the lower of the two.</div>
    </div>
  </div>
</section>

<!-- BASELINES -->
<section id="baselines">
  <div class="wrap reveal">
    <span class="eyebrow">Controls</span>
    <h2 class="sec">Two controls that pin down the cause</h2>
    <p class="sec-lead">If the SLM win came from generative decoding, a discriminative encoder shouldn't also beat the frontier. It does. And if it came from literary <i>exposure</i>, domain-adaptive pretraining should help. It doesn't.</p>
    <div class="grid-2" style="gap:18px;align-items:stretch">
      <div class="card card-pad">
        <h3>Discriminative encoder baseline</h3>
        <div class="desc">Entity-marker RoBERTa fine-tuned per benchmark clears both frontier systems on every dataset — General-Avg 0.826 vs 0.69 / 0.66.</div>
        <div id="encChart" style="margin-top:8px"></div>
        <div class="legend" id="encLegend"></div>
      </div>
      <div class="card card-pad">
        <h3>DAPT case study — a clean null result</h3>
        <div class="desc">Continued LitBank pretraining on Llama-3.2-3B before QLoRA, a 2×2 design (±DAPT × LitTune/MixTune).</div>
        <div style="overflow-x:auto;margin-top:12px"><table class="dapt-t" id="daptTable"></table></div>
        <div class="chart-note">Continued literary pretraining adds at most <b>+0.001</b> Literature-Avg F1 over supervised fine-tuning. Supervised task adaptation, not unsupervised domain exposure, closes the literary gap (a ~9% verbatim PG-Fiction/LitBank overlap likely contributes to the null).</div>
      </div>
    </div>
    <div class="card card-pad" style="margin-top:18px">
      <h3>Difficulty is about schema, not domain or label count</h3>
      <div class="desc">Mean F1 across all configs vs number of relation labels (log axis). More labels does <b>not</b> mean harder: 219-label REBEL scores 0.81 while 41-label TACRED is the hardest at 0.58.</div>
      <div id="diffScatter" style="margin-top:8px"></div>
      <div class="legend" id="diffLegend"></div>
    </div>
  </div>
</section>

<!-- STATS -->
<section id="stats">
  <div class="wrap reveal">
    <span class="eyebrow">Is it real?</span>
    <h2 class="sec">Statistical significance</h2>
    <p class="sec-lead">Paired bootstrap tests on positive-class F1 (10,000 iterations) over tens of thousands of aligned examples. All runs are single-seed, so sub-three-point differences are read as suggestive.</p>
    <div class="sig-grid" id="sigGrid"></div>
  </div>
</section>

<!-- ABOUT -->
<section id="about">
  <div class="wrap reveal">
    <span class="eyebrow">Paper &amp; artifacts</span>
    <h2 class="sec">Reproduce everything</h2>
    <div class="about">
      <div>
        <p class="sec-lead" style="margin-bottom:18px">The full pipeline is config-driven and open. The best sub-billion checkpoint, the processed benchmarks (8 of 9; TACRED is LDC-licensed), and the collected frontier generations are released on the Hugging Face Hub.</p>
        <div class="art-list" id="artList"></div>
      </div>
      <div>
        <div class="cite-box" id="citeBox"><button class="cp" id="copyCite">Copy</button><span id="citeText"></span></div>
      </div>
    </div>
  </div>
</section>

</main>

<footer>
  <div class="wrap">
    <div>
      <div class="fbrand">Sub-Billion, Super-Frontier</div>
      <div style="margin-top:6px">Christou, D. &amp; Tsoumakas, G. (2026) · Aristotle University of Thessaloniki</div>
    </div>
    <div style="text-align:right">
      <a id="footArxiv" target="_blank" rel="noopener">arXiv</a> &nbsp;·&nbsp;
      <a id="footGit" target="_blank" rel="noopener">GitHub</a> &nbsp;·&nbsp;
      <a id="footHf" target="_blank" rel="noopener">Hugging Face</a>
      <div style="margin-top:6px;color:#5b6884">Interactive findings dashboard · positive-class micro-F1 throughout</div>
    </div>
  </div>
</footer>

<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById('data').textContent);
const $ = (s,r=document)=>r.querySelector(s);
const $$ = (s,r=document)=>[...r.querySelectorAll(s)];
const NS="http://www.w3.org/2000/svg";
const f3=v=>v==null?"–":v.toFixed(3);
const f2=v=>v==null?"–":v.toFixed(2);

/* ---------- color scale for F1 ---------- */
function hx(h){h=h.replace('#','');return[parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}
function mix(a,b,t){return a.map((x,i)=>Math.round(x+(b[i]-x)*t));}
function toHex(c){return '#'+c.map(x=>x.toString(16).padStart(2,'0')).join('');}
const STOPS=[[0.40,hx('#fb7185')],[0.62,hx('#fca5a5')],[0.70,hx('#fcd34d')],[0.82,hx('#bef264')],[1.00,hx('#34d399')]];
function f1color(v){
  if(v==null) return null;
  if(v<=0.001) return '#e9edf3';
  let t=Math.max(0.40,Math.min(1,v));
  for(let i=0;i<STOPS.length-1;i++){
    const[a,ca]=STOPS[i],[b,cb]=STOPS[i+1];
    if(t<=b){const f=(t-a)/(b-a);return toHex(mix(ca,cb,Math.max(0,f)));}
  }
  return toHex(STOPS[STOPS.length-1][1]);
}

/* ---------- svg helpers ---------- */
function E(tag,attrs={},kids=[]){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);(Array.isArray(kids)?kids:[kids]).forEach(c=>{if(c!=null)e.appendChild(typeof c==='string'?document.createTextNode(c):c);});return e;}
function svg(w,h){const s=E('svg',{viewBox:`0 0 ${w} ${h}`,class:'chart',preserveAspectRatio:'xMidYMid meet'});s.style.maxHeight=(h+4)+'px';return s;}

/* ---------- vertical grouped bars ---------- */
function groupedBars(mount,{categories,series,max=1,H=330,unit=''}){
  const W=680,mL=44,mR=14,mT=18,mB=44;
  const pw=W-mL-mR,ph=H-mT-mB;
  const s=svg(W,H);
  const y=v=>mT+ph-(v/max)*ph;
  for(let i=0;i<=5;i++){const val=max*i/5,yy=y(val);
    s.appendChild(E('line',{x1:mL,x2:W-mR,y1:yy,y2:yy,class:'grid-l'}));
    s.appendChild(E('text',{x:mL-8,y:yy+3.5,class:'tick','text-anchor':'end'},val.toFixed(1)));}
  s.appendChild(E('line',{x1:mL,x2:mL,y1:mT,y2:mT+ph,class:'axis'}));
  const nC=categories.length,nS=series.length,gw=pw/nC,pad=gw*0.16,bw=(gw-2*pad)/nS;
  categories.forEach((cat,ci)=>{
    const gx=mL+ci*gw+pad;
    series.forEach((se,si)=>{
      const v=se.values[ci];if(v==null)return;
      const bx=gx+si*bw,bh=(v/max)*ph,by=y(v);
      const r=E('rect',{x:bx+1,y:by,width:Math.max(1,bw-2),height:bh,rx:3,fill:se.color,class:'bar'});
      r.appendChild(E('title',{},`${se.name} — ${cat}: ${f3(v)}`));
      s.appendChild(r);
      s.appendChild(E('text',{x:bx+bw/2,y:by-5,class:'blab','font-size':nC>6?9:10.5},v.toFixed(2)));
    });
    s.appendChild(E('text',{x:mL+ci*gw+gw/2,y:mT+ph+22,class:'catlab','font-size':nC>6?10:12.5},cat));
  });
  mount.innerHTML='';mount.appendChild(s);
}

/* ---------- horizontal grouped bars (per-dataset) ---------- */
function hGroupedBars(mount,{rows,series,max=1}){
  const W=680,mL=118,mR=40,mT=8,mB=24;
  const rowH=44,gap=14;
  const H=mT+mB+rows.length*rowH+(rows.length-1)*gap;
  const pw=W-mL-mR;
  const s=svg(W,H);
  const x=v=>mL+(v/max)*pw;
  for(let i=0;i<=5;i++){const val=max*i/5,xx=x(val);
    s.appendChild(E('line',{x1:xx,x2:xx,y1:mT,y2:H-mB,class:'grid-l'}));
    s.appendChild(E('text',{x:xx,y:H-mB+15,class:'tick','text-anchor':'middle'},val.toFixed(1)));}
  const nS=series.length;
  rows.forEach((row,ri)=>{
    const ry=mT+ri*(rowH+gap);
    const bh=(rowH)/nS;
    s.appendChild(E('text',{x:mL-10,y:ry+rowH/2+4,class:'catlab','text-anchor':'end'},row.label));
    series.forEach((se,si)=>{
      const v=row.values[si];if(v==null)return;
      const by=ry+si*bh;
      const r=E('rect',{x:mL,y:by+1.5,width:Math.max(1,x(v)-mL),height:bh-3,rx:3,fill:se.color,class:'bar'});
      r.appendChild(E('title',{},`${se.name} — ${row.label}: ${f3(v)}`));
      s.appendChild(r);
      s.appendChild(E('text',{x:x(v)+5,y:by+bh/2+3.5,class:'blab','text-anchor':'start',fill:'#0d1424'},v.toFixed(2)));
    });
  });
  mount.innerHTML='';mount.appendChild(s);
}

/* ---------- scatter (log x, bubbles, hlines) ---------- */
function scatter(mount,{points,xDomain,yDomain,H=320,xLabel,yLabel,logX=true,hlines=[],xticks}){
  const W=680,mL=48,mR=20,mT=16,mB=46;
  const pw=W-mL-mR,ph=H-mT-mB;
  const s=svg(W,H);
  const lx=v=>logX?Math.log10(v):v;
  const x0=lx(xDomain[0]),x1=lx(xDomain[1]);
  const X=v=>mL+((lx(v)-x0)/(x1-x0))*pw;
  const Y=v=>mT+ph-((v-yDomain[0])/(yDomain[1]-yDomain[0]))*ph;
  // y grid
  const ys=5;for(let i=0;i<=ys;i++){const val=yDomain[0]+(yDomain[1]-yDomain[0])*i/ys,yy=Y(val);
    s.appendChild(E('line',{x1:mL,x2:W-mR,y1:yy,y2:yy,class:'grid-l'}));
    s.appendChild(E('text',{x:mL-8,y:yy+3.5,class:'tick','text-anchor':'end'},val.toFixed(2)));}
  // x ticks
  (xticks||[]).forEach(t=>{const xx=X(t.v);
    s.appendChild(E('line',{x1:xx,x2:xx,y1:mT,y2:mT+ph,class:'grid-l'}));
    s.appendChild(E('text',{x:xx,y:mT+ph+16,class:'tick','text-anchor':'middle'},t.l));});
  s.appendChild(E('line',{x1:mL,x2:mL,y1:mT,y2:mT+ph,class:'axis'}));
  s.appendChild(E('line',{x1:mL,x2:W-mR,y1:mT+ph,y2:mT+ph,class:'axis'}));
  // hlines
  hlines.forEach(h=>{const yy=Y(h.v);
    s.appendChild(E('line',{x1:mL,x2:W-mR,y1:yy,y2:yy,stroke:h.color,'stroke-width':1.6,'stroke-dasharray':'6 4',opacity:.85}));
    s.appendChild(E('text',{x:W-mR,y:yy-5,class:'tick','text-anchor':'end',fill:h.color,'font-weight':700},h.label));});
  if(xLabel)s.appendChild(E('text',{x:mL+pw/2,y:H-8,class:'tick','font-weight':700},xLabel));
  if(yLabel)s.appendChild(E('text',{x:14,y:mT+ph/2,class:'tick','font-weight':700,transform:`rotate(-90 14 ${mT+ph/2})`,'text-anchor':'middle'},yLabel));
  // points
  points.forEach(p=>{
    const cx=X(p.x),cy=Y(p.y),r=p.r||6;
    const c=E('circle',{cx,cy,r,fill:p.color,'fill-opacity':.28,stroke:p.color,'stroke-width':2});
    c.appendChild(E('title',{},p.title||''));
    s.appendChild(c);
    s.appendChild(E('circle',{cx,cy,r:2.4,fill:p.color}));
    if(p.label)s.appendChild(E('text',{x:cx+(p.ldx||0),y:cy+(p.ldy!=null?p.ldy:-r-6),class:'blab',fill:p.color,'text-anchor':p.lanchor||'middle'},p.label));
  });
  mount.innerHTML='';mount.appendChild(s);
}

/* ---------- multi-series lines ---------- */
function lines(mount,{series,xDomain,yDomain,H=300,xticks,logX=true,xLabel,yLabel}){
  const W=680,mL=48,mR=64,mT=16,mB=42;
  const pw=W-mL-mR,ph=H-mT-mB;
  const s=svg(W,H);
  const lx=v=>logX?Math.log10(v):v;
  const x0=lx(xDomain[0]),x1=lx(xDomain[1]);
  const X=v=>mL+((lx(v)-x0)/(x1-x0))*pw;
  const Y=v=>mT+ph-((v-yDomain[0])/(yDomain[1]-yDomain[0]))*ph;
  for(let i=0;i<=5;i++){const val=yDomain[0]+(yDomain[1]-yDomain[0])*i/5,yy=Y(val);
    s.appendChild(E('line',{x1:mL,x2:W-mR,y1:yy,y2:yy,class:'grid-l'}));
    s.appendChild(E('text',{x:mL-8,y:yy+3.5,class:'tick','text-anchor':'end'},val.toFixed(2)));}
  (xticks||[]).forEach(t=>{const xx=X(t.v);s.appendChild(E('text',{x:xx,y:mT+ph+16,class:'tick','text-anchor':'middle'},t.l));});
  s.appendChild(E('line',{x1:mL,x2:mL,y1:mT,y2:mT+ph,class:'axis'}));
  s.appendChild(E('line',{x1:mL,x2:W-mR,y1:mT+ph,y2:mT+ph,class:'axis'}));
  if(xLabel)s.appendChild(E('text',{x:mL+pw/2,y:H-6,class:'tick','font-weight':700},xLabel));
  if(yLabel)s.appendChild(E('text',{x:14,y:mT+ph/2,class:'tick','font-weight':700,transform:`rotate(-90 14 ${mT+ph/2})`,'text-anchor':'middle'},yLabel));
  series.forEach(se=>{
    let d='';se.points.forEach((p,i)=>{d+=(i?'L':'M')+X(p.x)+' '+Y(p.y)+' ';});
    s.appendChild(E('path',{d,fill:'none',stroke:se.color,'stroke-width':3,'stroke-linecap':'round','stroke-linejoin':'round'}));
    se.points.forEach(p=>{const c=E('circle',{cx:X(p.x),cy:Y(p.y),r:5,fill:'#fff',stroke:se.color,'stroke-width':3});c.appendChild(E('title',{},`${se.name}: ${p.label} → ${f3(p.y)}`));s.appendChild(c);
      s.appendChild(E('text',{x:X(p.x),y:Y(p.y)-11,class:'blab',fill:se.color},p.y.toFixed(3)));});
    const last=se.points[se.points.length-1];
    s.appendChild(E('text',{x:X(last.x)+9,y:Y(last.y)+4,class:'blab','text-anchor':'start',fill:se.color},se.name));
  });
  mount.innerHTML='';mount.appendChild(s);
}

/* ---------- diverging horizontal bars (prompt deltas) ---------- */
function divBars(mount,{items,maxAbs=0.25}){
  const W=680,mL=150,mR=44,mT=6,mB=26,rowH=26,gap=6;
  const H=mT+mB+items.length*(rowH+gap);
  const pw=W-mL-mR;const midx=mL+pw*0; // left-aligned baseline at mL (all positive except a few)
  const s=svg(W,H);
  const X=v=>mL+(v/maxAbs)*pw;
  // zero line + scale
  for(let i=0;i<=5;i++){const val=maxAbs*i/5,xx=X(val);
    s.appendChild(E('line',{x1:xx,x2:xx,y1:mT,y2:H-mB,class:'grid-l'}));
    s.appendChild(E('text',{x:xx,y:H-mB+15,class:'tick','text-anchor':'middle'},'+'+val.toFixed(2)));}
  s.appendChild(E('line',{x1:mL,x2:mL,y1:mT,y2:H-mB,class:'axis'}));
  items.forEach((it,i)=>{
    const ry=mT+i*(rowH+gap);
    const col=it.scale==='sub-billion'?'#10b981':'#6366f1';
    const over=it.delta>maxAbs;                 // decoding artifacts overflow the axis
    const bx=Math.min(X(it.delta),mL+pw);
    s.appendChild(E('text',{x:mL-10,y:ry+rowH/2+4,class:'blab','text-anchor':'end',fill:'#3a465c'},it.label));
    const r=E('rect',{x:mL,y:ry+2,width:Math.max(2,bx-mL),height:rowH-4,rx:4,fill:it.flag?'#f3c982':col,class:'bar'});
    r.appendChild(E('title',{},`${it.label}: delta ${it.delta>=0?'+':''}${it.delta.toFixed(3)}${it.flag?' (decoding artifact, excluded from means)':''}`));
    s.appendChild(r);
    const lbl=(it.delta>=0?'+':'')+it.delta.toFixed(3)+(it.flag?(it.flag==='think'?' †':' ‡'):'');
    if(over) s.appendChild(E('text',{x:bx-8,y:ry+rowH/2+3.5,class:'blab','text-anchor':'end',fill:'#8a5a00'},lbl));
    else s.appendChild(E('text',{x:bx+6,y:ry+rowH/2+3.5,class:'blab','text-anchor':'start'},lbl));
  });
  mount.innerHTML='';mount.appendChild(s);
}

/* ---------- signed delta bars (constrained vs open), zero baseline + SEM ---------- */
function deltaBars(mount,{items,mean}){
  const W=680,mL=120,mR=54,mT=8,mB=30,rowH=27,gap=9;
  const H=mT+mB+items.length*(rowH+gap);
  const pw=W-mL-mR;
  const vmax=Math.max(...items.map(d=>d.delta+(d.sem||0)));
  const vmin=Math.min(0,...items.map(d=>d.delta-(d.sem||0)));
  const lo=vmin-1,hi=vmax+2.5;
  const X=v=>mL+((v-lo)/(hi-lo))*pw;
  const s=svg(W,H);
  for(let t=Math.ceil(lo/4)*4;t<=hi;t+=4){const xx=X(t);
    s.appendChild(E('line',{x1:xx,x2:xx,y1:mT,y2:H-mB,class:'grid-l'}));
    s.appendChild(E('text',{x:xx,y:H-mB+15,class:'tick','text-anchor':'middle'},(t>0?'+':'')+t));}
  const zx=X(0);
  s.appendChild(E('line',{x1:zx,x2:zx,y1:mT,y2:H-mB,stroke:'#94a3b8','stroke-width':1.4}));
  const mx=X(mean);
  s.appendChild(E('line',{x1:mx,x2:mx,y1:mT,y2:H-mB,stroke:'#6d28d9','stroke-width':1.6,'stroke-dasharray':'5 4'}));
  s.appendChild(E('text',{x:mx+4,y:mT+9,class:'tick',fill:'#6d28d9','font-weight':700},`mean ${mean>=0?'+':''}${mean.toFixed(1)}`));
  items.forEach((it,i)=>{
    const ry=mT+i*(rowH+gap);
    const pos=it.delta>=0,col=pos?'#10b981':'#ef4444';
    const xv=X(it.delta),bx=Math.min(zx,xv),bw=Math.max(2,Math.abs(xv-zx));
    s.appendChild(E('text',{x:mL-10,y:ry+rowH/2+4,class:'blab','text-anchor':'end',fill:'#3a465c'},it.label));
    const r=E('rect',{x:bx,y:ry+3,width:bw,height:rowH-6,rx:3,fill:col,class:'bar'});
    r.appendChild(E('title',{},`${it.label}: ${pos?'+':''}${it.delta.toFixed(2)} pp (generic - schema-enumerated)`));
    s.appendChild(r);
    if(it.sem){const e1=X(it.delta-it.sem),e2=X(it.delta+it.sem),ey=ry+rowH/2;
      s.appendChild(E('line',{x1:e1,x2:e2,y1:ey,y2:ey,stroke:'#475569','stroke-width':1}));
      s.appendChild(E('line',{x1:e1,x2:e1,y1:ey-3,y2:ey+3,stroke:'#475569','stroke-width':1}));
      s.appendChild(E('line',{x1:e2,x2:e2,y1:ey-3,y2:ey+3,stroke:'#475569','stroke-width':1}));}
    const lx=pos?X(it.delta+(it.sem||0))+6:zx+6;   // negative label sits just right of the zero baseline
    s.appendChild(E('text',{x:lx,y:ry+rowH/2+3.5,class:'blab','text-anchor':'start'},(pos?'+':'')+it.delta.toFixed(1)));
  });
  mount.innerHTML='';mount.appendChild(s);
}

function legend(mount,items){
  mount.innerHTML='';
  items.forEach(it=>{const d=document.createElement('span');d.className='li';
    d.innerHTML=`<span class="sw" style="background:${it.color}"></span>${it.name}`;mount.appendChild(d);});
}

/* =====================  RENDER  ===================== */
// hero + meta
$('#heroTitle').innerHTML = `<span class="grad">${D.meta.title}.</span> <br>${D.meta.subtitle}`;
$('#heroSub').innerHTML = 'We fine-tune five small language models (360M–3B) for relation extraction across general and literary text, and benchmark them against zero-shot frontier LLMs. <b>Targeted task adaptation lets 4-bit models on a single consumer GPU outperform general-purpose frontier systems.</b>';
$('#heroAuthors').innerHTML = `<b>${D.meta.authors}</b> · ${D.meta.affil}`;
$('#heroArxiv').href=D.meta.arxiv; $('#heroGit').href=D.meta.github; $('#ghlink').href=D.meta.github;
$('#footArxiv').href=D.meta.arxiv; $('#footGit').href=D.meta.github; $('#footHf').href=D.meta.hf;
const strip=[['0.83','Best sub-billion General-Avg F1'],['+26–30','Literary F1 lead over frontier'],['30','Tuned configurations'],['~18 ms','Per extraction on one GPU']];
$('#heroStrip').innerHTML=strip.map(x=>`<div class="hs"><div class="v">${x[0]}</div><div class="k">${x[1]}</div></div>`).join('');

// KPIs
$('#kpiGrid').innerHTML=D.kpis.map(k=>`<div class="kpi"><div class="v">${k.value}</div><div class="u">${k.unit}</div><div class="l">${k.label}</div><div class="s">${k.sub}</div></div>`).join('');

// headline chart
groupedBars($('#headlineChart'),{categories:D.headline.categories,series:D.headline.series,max:1,H:340});
legend($('#headlineLegend'),D.headline.series.map(s=>({name:s.name,color:s.color})));

// findings
const ICONS={
 target:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>',
 book:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"/><path d="M4 20.5A2.5 2.5 0 0 1 6.5 18H20"/></svg>',
 mix:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="8.5" cy="12" r="5.5"/><circle cx="15.5" cy="12" r="5.5"/></svg>',
 prompt:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H8l-4 4V5a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2z"/></svg>',
 scale:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 20h18"/><rect x="5" y="12" width="4" height="8"/><rect x="10" y="7" width="4" height="13"/><rect x="15" y="3" width="4" height="17"/></svg>',
 check:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6 9 17l-5-5"/></svg>'};
$('#findGrid').innerHTML=D.findings.map(f=>`<div class="find"><div class="ic">${ICONS[f.icon]||''}</div><h4>${f.title}</h4><p>${f.body}</p><span class="stat">${f.stat}</span></div>`).join('');

// per-dataset frontier
(function(){
  const ds=[...D.genDs,...D.litDs];
  const rows=ds.map(d=>({label:D.dsLabel[d],values:[
     D.bestSlm[d],
     (D.frontierGeneral['GPT-5.4'][d]??D.frontierLiterary['GPT-5.4'][d]),
     (D.frontierGeneral['Claude Sonnet 4.6'][d]??D.frontierLiterary['Claude Sonnet 4.6'][d])
  ]}));
  const series=[{name:'Best tuned SLM',color:'#10b981'},{name:'GPT-5.4 (0-shot)',color:'#f59e0b'},{name:'Claude Sonnet 4.6 (0-shot)',color:'#ef4444'}];
  hGroupedBars($('#frontierChart'),{rows,series,max:1});
  legend($('#frontierLegend'),series);
})();

// efficiency scatter + table
(function(){
  const EPLACE={
    'SmolLM2-360M MixTune 2s':{lab:'SmolLM2-360M',ldy:-16},
    'Qwen2.5-0.5B GenTune 2s':{lab:'Qwen2.5-0.5B (Gen)',ldy:-16},
    'Qwen2.5-0.5B MixTune 2s':{lab:'Qwen2.5-0.5B (Mix)',ldy:24},
    'SmolLM3-3B LitTune 0s':{lab:'SmolLM3-3B',ldy:4,ldx:-12,lanchor:'end'},
    'Llama-3.2-3B MixTune 2s':{lab:'Llama-3B (Mix)',ldy:26},
    'Llama-3.2-3B GenTune 2s':{lab:'Llama-3B (Gen)',ldy:-16},
  };
  const pts=D.efficiency.map(e=>{const p=EPLACE[e.cfg]||{};return {x:e.params,y:e.avg,r:6+Math.sqrt(e.f1b)*9,color:e.scale==='sub-billion'?'#10b981':'#6366f1',
     label:p.lab||e.cfg.split(' ')[0],ldy:p.ldy,ldx:p.ldx,lanchor:p.lanchor,
     title:`${e.cfg} — Avg F1 ${f3(e.avg)} · ${e.f1b} F1/B · ${e.gpu}ms GPU`};});
  scatter($('#effScatter'),{points:pts,xDomain:[0.3,3.6],yDomain:[0.5,0.9],logX:true,
    xticks:[{v:0.36,l:'360M'},{v:0.5,l:'0.5B'},{v:1,l:'1B'},{v:3,l:'3B'}],
    xLabel:'Parameters (log scale)',yLabel:'Avg F1',
    hlines:[{v:D.efficiencyFrontier[0].avg,color:'#f59e0b',label:'GPT-5.4 0.667'},{v:D.efficiencyFrontier[1].avg,color:'#ef4444',label:'Claude 0.632'}]});
  legend($('#effLegend'),[{name:'Sub-billion',color:'#10b981'},{name:'3B',color:'#6366f1'},{name:'bubble ∝ F1 / billion params',color:'#cbd5e1'}]);
  const maxF1b=Math.max(...D.efficiency.map(e=>e.f1b));
  const rowsHtml=D.efficiency.map(e=>`<tr>
     <td><span class="pill sub">${e.scale==='sub-billion'?'sub-B':'3B'}</span> ${e.cfg}</td>
     <td>${e.params}B</td><td>${e.size}</td><td>${e.gpu} ms</td><td>${e.cpu} ms</td>
     <td>${f3(e.avg)}</td>
     <td><b>${e.f1b.toFixed(2)}</b><span class="f1b-bar" style="width:${(e.f1b/maxF1b*54).toFixed(0)}px"></span></td></tr>`).join('');
  const frHtml=D.efficiencyFrontier.map(e=>`<tr>
     <td><span class="pill fr">API</span> ${e.cfg}</td><td>—</td><td>API</td><td>—</td><td>—</td><td>${f3(e.avg)}</td><td>N/A</td></tr>`).join('');
  $('#effTable').innerHTML=`<thead><tr><th>Configuration</th><th>Params</th><th>4-bit size</th><th>GPU</th><th>CPU</th><th>Avg F1</th><th>F1 / B</th></tr></thead><tbody>${rowsHtml}${frHtml}</tbody>`;
})();

// scaling lines
(function(){
  const series=[
    {name:'Qwen2.5',color:'#7c3aed',points:D.scaling['Qwen2.5'].map(p=>({x:p.params,y:p.overall,label:p.label}))},
    {name:'SmolLM',color:'#06b6d4',points:D.scaling['SmolLM'].map(p=>({x:p.params,y:p.overall,label:p.label}))},
  ];
  lines($('#scaleChart'),{series,xDomain:[0.3,3.4],yDomain:[0.7,0.87],logX:true,
    xticks:[{v:0.36,l:'360M'},{v:0.5,l:'0.5B'},{v:3,l:'3B'}],xLabel:'Parameters (log)',yLabel:'Best overall F1'});
})();

// prompt delta bars
(function(){
  const order={'sub-billion':0,'3B':1};
  const items=D.promptDeltas.slice().sort((a,b)=> (order[a.scale]-order[b.scale]) || (b.delta-a.delta))
    .map(d=>({label:`${d.model} ${d.regime.replace('Tune','')}`,delta:d.delta,scale:d.scale,flag:d.flag}));
  divBars($('#promptChart'),{items,maxAbs:0.25});
})();

// constrained vs open (prompt format)
(function(){
  const CO=D.constrainedOpen;
  const fp=v=>(v>=0?'+':'')+v.toFixed(1);
  $('#coOverall').textContent=fp(CO.overall)+' pp';
  $('#coHeadline').textContent=`Generic beats schema-enumerated on ${CO.nPos} of ${CO.n} datasets`;
  $('#coNPos').textContent=`${CO.nPos} of ${CO.n}`;
  $('#coSub').textContent=fp(CO.sub);
  $('#coBig').textContent=fp(CO.big);
  $('#coMaxN').textContent=CO.maxN;
  const items=CO.perDataset.slice().sort((a,b)=>b.delta-a.delta);
  deltaBars($('#coChart'),{items,mean:CO.overall});
  legend($('#coLegend'),[{name:'Generic better',color:'#10b981'},{name:'Schema-enumerated better',color:'#ef4444'},{name:'error bars = SEM',color:'#94a3b8'}]);
})();

// encoder chart
(function(){
  const ds=[...D.genDs,...D.litDs];
  const cats=ds.map(d=>D.dsLabel[d]);
  const series=[
    {name:'RoBERTa-base',color:'#6366f1',values:D.encoder.map(e=>e.rob_base)},
    {name:'Best tuned SLM',color:'#10b981',values:D.encoder.map(e=>e.best_slm)},
    {name:'GPT-5.4',color:'#f59e0b',values:D.encoder.map(e=>e.gpt)},
  ];
  groupedBars($('#encChart'),{categories:cats,series,max:1,H:300});
  legend($('#encLegend'),series);
})();

// dapt table
(function(){
  const r=D.dapt.map(d=>`<tr class="${d.dapt?'dapt-row':''}">
     <td>${d.model}</td><td>${f3(d.bio)}</td><td>${f3(d.pg)}</td><td><b>${f3(d.lit)}</b></td>
     <td>${d.delta==null?'<span class="d0">—</span>':'<b style="color:#047857">+'+d.delta.toFixed(3)+'</b>'}</td></tr>`).join('');
  $('#daptTable').innerHTML=`<thead><tr><th>Model</th><th>Biographical</th><th>PG-Fiction</th><th>Lit-Avg F1</th><th>Δ vs base</th></tr></thead><tbody>${r}</tbody>`;
})();

// difficulty scatter
(function(){
  const pts=D.difficulty.map(d=>({x:d.n_labels,y:d.f1,r:8,color:d.domain==='general'?'#3b82f6':'#ec4899',
     label:d.label,title:`${d.label}: ${d.n_labels} labels · mean F1 ${f3(d.f1)} (range ${f2(d.min)}–${f2(d.max)})`}));
  scatter($('#diffScatter'),{points:pts,xDomain:[4,240],yDomain:[0.5,1.0],logX:true,
    xticks:[{v:5,l:'5'},{v:13,l:'13'},{v:41,l:'41'},{v:96,l:'96'},{v:219,l:'219'}],
    xLabel:'Number of relation labels (log)',yLabel:'Mean F1 across configs'});
  legend($('#diffLegend'),[{name:'General-domain',color:'#3b82f6'},{name:'Literary',color:'#ec4899'}]);
})();

// significance
$('#sigGrid').innerHTML=D.significance.map(s=>`<div class="sig"><span class="badge ${s.kind==='pos'?'pos':'neutral'}">${s.stat}</span><div><h4>${s.title}</h4><p>${s.detail}</p></div></div>`).join('');

// artifacts + cite
const arts=[
 {i:'🤖',t:'Best sub-billion checkpoint',s:'Despina/Qwen2.5-0.5B-Instruct-re_gentune-2-shot',u:D.meta.hf},
 {i:'📊',t:'Frontier-model generations (GPT-5.4 + Claude)',s:'Despina/frontier-re-generations',u:'https://huggingface.co/datasets/Despina/frontier-re-generations'},
 {i:'📚',t:'Processed RE benchmarks (8 of 9)',s:'huggingface.co/Despina/*',u:'https://huggingface.co/Despina'},
 {i:'💻',t:'Reproduction code & configs',s:'github.com/DespinaChristou/compact-relex',u:D.meta.github},
];
$('#artList').innerHTML=arts.map(a=>`<a class="art" href="${a.u}" target="_blank" rel="noopener"><span class="ai">${a.i}</span><div><div class="at">${a.t}</div><div class="as">${a.s}</div></div><span class="go">→</span></a>`).join('');
const bib=`@article{christou2026subbillion,
  title   = {Sub-Billion, Super-Frontier: Small Language Models
             Rival Zero-Shot Frontier LLMs on General and
             Literary Relation Extraction},
  author  = {Christou, Despina and Tsoumakas, Grigorios},
  journal = {arXiv preprint arXiv:2606.22606},
  year    = {2026}
}`;
$('#citeText').textContent=bib;
$('#copyCite').addEventListener('click',()=>{navigator.clipboard.writeText(bib).then(()=>{const b=$('#copyCite');b.textContent='Copied!';setTimeout(()=>b.textContent='Copy',1400);});});

/* =====================  MATRIX  ===================== */
(function(){
  const ds=[...D.genDs,...D.litDs];
  const table=$('#matrixTable');
  const filters={scale:'all',regime:'all',shot:'all'};
  let sortCol=null,sortDir=-1; // default: original order
  const RG={GenTune:'gen',LitTune:'lit',MixTune:'mix'};

  function head(){
    let h='<thead><tr>';
    h+='<th class="lft sortable" data-sort="model">Configuration</th>';
    ds.forEach((d,i)=>{h+=`<th class="sortable${i===7?'':''}" data-sort="${d}" title="${D.dsLabel[d]}">${D.dsLabel[d]}</th>`;});
    h+='<th class="avg sortable" data-sort="gen">Gen</th><th class="avg sortable" data-sort="lit">Lit</th><th class="avg sortable" data-sort="overall">Overall ▾</th>';
    h+='</tr></thead>';
    return h;
  }
  function passes(r){
    return (filters.scale==='all'||r.scale===filters.scale)
        && (filters.regime==='all'||r.regime===filters.regime)
        && (filters.shot==='all'||r.shot===filters.shot);
  }
  function rowsData(){
    let rows=D.matrix.filter(passes);
    if(sortCol){
      rows=rows.slice().sort((a,b)=>{
        let va,vb;
        if(sortCol==='model'){va=a.model+a.regime+a.shot;vb=b.model+b.regime+b.shot;return (va<vb?-1:1)*sortDir;}
        if(['gen','lit','overall'].includes(sortCol)){va=a[sortCol];vb=b[sortCol];}
        else{va=a.cells[sortCol];vb=b.cells[sortCol];}
        va=(va==null?-1:va);vb=(vb==null?-1:vb);
        return (va-vb)*sortDir;
      });
    }
    return rows;
  }
  function cell(v,extra=''){
    if(v==null) return `<td><div class="cell empty ${extra}">–</div></td>`;
    return `<td><div class="cell ${extra}" style="background:${f1color(v)}">${v.toFixed(3)}</div></td>`;
  }
  function render(){
    let h=head()+'<tbody>';
    rowsData().forEach(r=>{
      const shotN=r.shot==='0s'?'0-shot':'2-shot';
      const mark=r.flag==='think'?' <span class="fl" style="position:static">†</span>':r.flag==='schema'?' <span class="fl" style="position:static">‡</span>':'';
      h+='<tr class="rrow">';
      h+=`<td class="meta"><span class="mdl">${r.model}</span><span class="rg ${RG[r.regime]}">${r.regime.replace('Tune','')}</span> <span class="shot">${shotN}</span>${mark}</td>`;
      ds.forEach(d=>{ h+=cell(r.cells[d]); });
      h+=cell(r.gen,'avg')+cell(r.lit,'avg')+cell(r.overall,'avg');
      h+='</tr>';
    });
    h+='</tbody>';
    table.innerHTML=h;
    // header arrows + sort handlers
    $$('#matrixTable th.sortable').forEach(th=>{
      const c=th.dataset.sort;
      if(c===sortCol)th.innerHTML=th.textContent.replace(/[▾▴]/g,'').trim()+(sortDir<0?' ▾':' ▴');
      th.onclick=()=>{ if(sortCol===c){sortDir*=-1;}else{sortCol=c;sortDir=-1;} render(); };
    });
  }
  // filter buttons
  $$('.segbtns[data-group]').forEach(g=>{
    g.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
      g.querySelectorAll('button').forEach(x=>x.classList.remove('on'));
      b.classList.add('on');filters[g.dataset.group]=b.dataset.v;render();
    }));
  });
  $('#resetSort').addEventListener('click',()=>{sortCol='overall';sortDir=-1;render();});
  // default sort by overall desc
  sortCol='overall';sortDir=-1;
  render();
})();

/* =====================  UX: scrollspy + reveal + countup ===================== */
(function(){
  const links=$$('.nav-links a');
  const map={};links.forEach(a=>{const id=a.getAttribute('href').slice(1);map[id]=a;});
  const secs=Object.keys(map).map(id=>document.getElementById(id)).filter(Boolean);
  const spy=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){links.forEach(l=>l.classList.remove('active'));if(map[e.target.id])map[e.target.id].classList.add('active');}});},{rootMargin:'-45% 0px -50% 0px'});
  secs.forEach(s=>spy.observe(s));
  // entrance animation on natural scroll, but never leave jumped-to content blank
  const revs=$$('.reveal');
  const revealAll=()=>revs.forEach(r=>r.classList.add('in'));
  const rev=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');rev.unobserve(e.target);}});},{rootMargin:'0px 0px -6% 0px',threshold:.03});
  revs.forEach(r=>rev.observe(r));
  $$('a[href^="#"]').forEach(a=>a.addEventListener('click',revealAll));
  window.addEventListener('hashchange',revealAll);
  if(location.hash&&location.hash.length>1) revealAll();
})();
</script>
</body>
</html>"""
