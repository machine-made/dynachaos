"""Presentation shell for the interactive paper: fonts, tokens, CSS and JS.

Kept separate from ``build_paper.py`` so the design can be edited without
touching the conversion pipeline. Everything here is inlined into the generated
page except the webfonts, which are served from ``site/fonts/``.

Two decisions worth stating, because they drive the rest:

* The three accent hues encode the Lyapunov classification used throughout the
  study --- locked, quasiperiodic, chaotic. The hero canvas colours the logistic
  attractor by that same rule, so the legend teaches the palette before the
  reader meets a figure.
* Figures render compact by default. A reader scanning the argument should not
  wade through full-bleed plots; interaction and magnification are opt-in, one
  click away, and the heavy chart payload is never fetched until asked for.
"""

# ruff: noqa: E501 -- inlined CSS and JS are kept in their authored form.

from __future__ import annotations

FONT_FACES = (
    (
        "TeX Gyre Pagella",
        "pagella-regular.woff2",
        "normal",
        "400",
        "U+0000-024F,U+1E00-1EFF,U+2000-206F,U+2070-209F,U+20A0-20CF,U+2100-214F,U+2190-21FF,U+2200-22FF,U+FEFF,U+FFFD",
    ),
    (
        "TeX Gyre Pagella",
        "pagella-italic.woff2",
        "italic",
        "400",
        "U+0000-024F,U+1E00-1EFF,U+2000-206F,U+2070-209F,U+20A0-20CF,U+2100-214F,U+2190-21FF,U+2200-22FF,U+FEFF,U+FFFD",
    ),
    (
        "TeX Gyre Pagella",
        "pagella-bold.woff2",
        "normal",
        "700",
        "U+0000-024F,U+1E00-1EFF,U+2000-206F,U+2070-209F,U+20A0-20CF,U+2100-214F,U+2190-21FF,U+2200-22FF,U+FEFF,U+FFFD",
    ),
    (
        "TeX Gyre Pagella",
        "pagella-bolditalic.woff2",
        "italic",
        "700",
        "U+0000-024F,U+1E00-1EFF,U+2000-206F,U+2070-209F,U+20A0-20CF,U+2100-214F,U+2190-21FF,U+2200-22FF,U+FEFF,U+FFFD",
    ),
)


def _font_css() -> str:
    return "\n".join(
        f"@font-face{{font-family:'{family}';src:url('fonts/{file}') format('woff2');"
        f"font-style:{style};font-weight:{weight};font-display:optional;unicode-range:{ur};}}"
        for family, file, style, weight, ur in FONT_FACES
    )


CSS = (
    _font_css()
    + r"""
:root{
  --locked:#a8620a;   /* lambda < 0  mode-locked / periodic */
  --locked-hero:#c07410; /* hero canvas only: same gold, one notch brighter */
  --torus:#54489f;    /* lambda ~ 0  quasiperiodic */
  --chaotic:#0d6f6a;  /* lambda > 0  chaotic */

  --ground:#faf8f3;
  --raised:#ffffff;
  --sunken:#efece3;
  --ink:#14161c;
  --ink-mid:#454b59;
  --ink-low:#6c7284; /* WCAG AA: 4.52:1 on --ground, 4.80:1 on --raised */
  --slate:#2E3440;    /* the paper's ink: attractor point clouds */
  --vermilion:#D1495B;/* the paper's single signal accent */
  --rule:#e0dccf;
  --rule-soft:#ebe7dc;

  --serif:'TeX Gyre Pagella', Palatino, 'Palatino Linotype', Georgia, serif;
  --math:'TeX Gyre Pagella Math', 'TeX Gyre Pagella', serif;
  --mono:ui-monospace,'SF Mono',SFMono-Regular,Menlo,Consolas,monospace;
  --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;

  --base:1.155rem;
  --measure:clamp(30rem,16.5rem + 24vw,42rem);
  --rail:clamp(12rem,14vw,17rem);
  --bleed:clamp(34rem,6rem + 44vw,66rem);
  --gutter:clamp(1.15rem,3.5vw,2.75rem);
  --shadow:0 1px 2px rgba(20,22,28,.05), 0 8px 28px rgba(20,22,28,.07);
}

:root[data-size="s"]{--base:1.0rem;--measure:clamp(28rem,16.75rem + 20vw,38rem);}
:root[data-size="m"]{--base:1.075rem;--measure:clamp(29rem,16.6rem + 22vw,40rem);}
:root[data-size="l"]{--base:1.155rem;--measure:clamp(30rem,16.5rem + 24vw,42rem);}
:root[data-size="xl"]{--base:1.26rem;--measure:clamp(31rem,16.4rem + 26vw,44rem);}

@media (prefers-color-scheme: dark){
  :root{
    --locked:#f0a830; --locked-hero:#f9bb46; --torus:#9086dc; --chaotic:#3fd0c8;
    --ground:#0a0d15; --raised:#111624; --sunken:#070911;
    --ink:#e9edf7; --ink-mid:#a6b0c7; --ink-low:#767f98; /* WCAG AA: 4.86:1 on --ground, 4.52:1 on --raised */
    --slate:#a6b0c7; --vermilion:#e06a7c;
    --rule:#1e2537; --rule-soft:#161c2b;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 30px rgba(0,0,0,.45);
  }
}
:root[data-theme="dark"]{
  --locked:#f0a830; --locked-hero:#f9bb46; --torus:#9086dc; --chaotic:#3fd0c8;
  --ground:#0a0d15; --raised:#111624; --sunken:#070911;
  --ink:#e9edf7; --ink-mid:#a6b0c7; --ink-low:#767f98; /* WCAG AA: 4.86:1 on --ground, 4.52:1 on --raised */
  --slate:#a6b0c7; --vermilion:#e06a7c;
  --rule:#1e2537; --rule-soft:#161c2b;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 30px rgba(0,0,0,.45);
}
:root[data-theme="light"]{
  --locked:#a8620a; --locked-hero:#c07410; --torus:#54489f; --chaotic:#0d6f6a;
  --ground:#faf8f3; --raised:#ffffff; --sunken:#efece3;
  --ink:#14161c; --ink-mid:#454b59; --ink-low:#6c7284; /* WCAG AA: 4.52:1 on --ground, 4.80:1 on --raised */
  --slate:#2E3440; --vermilion:#D1495B;
  --rule:#e0dccf; --rule-soft:#ebe7dc;
  --shadow:0 1px 2px rgba(20,22,28,.05), 0 8px 28px rgba(20,22,28,.07);
}

.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
  clip:rect(0,0,0,0);white-space:nowrap;border:0;}

.skip-link{position:absolute;left:0.6rem;top:-3.2rem;z-index:100;
  background:var(--raised);color:var(--ink);border:1px solid var(--rule);border-radius:4px;
  padding:0.6rem 1rem;font-family:var(--sans);font-size:0.85rem;
  transition:top .15s ease;}
.skip-link:focus{top:0.6rem;}

*{box-sizing:border-box;}
html{scroll-behavior:smooth;scroll-padding-top:3.5rem;-webkit-text-size-adjust:100%;}
body{
  margin:0;background:var(--ground);color:var(--ink);
  font-family:var(--serif);font-size:var(--base);line-height:1.6;
  text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased;
  overflow-x:hidden;
  transition:background-color .35s ease,color .35s ease;
}
@media (prefers-reduced-motion:reduce){body{transition:none;}}

math{font-family:var(--math);}
math[display="block"]{display:block;margin:0;overflow-x:auto;overflow-y:hidden;max-width:100%;}
/* the raw-LaTeX annotation is stripped at build time; belt and braces */
annotation,annotation-xml{display:none;}
.eqn{display:grid;grid-template-columns:1fr auto;align-items:center;gap:1rem;
  margin:1.65em 0;padding:0.15em 0;}
.eqn > math{min-width:0;}
.eqn .eqno{font-variant-numeric:tabular-nums;color:var(--ink-low);font-size:0.92em;flex:none;}
math[display="block"]:not(.eqn > math){margin:1.5em auto;}

/* ------------------------------ type ------------------------------ */
.eyebrow{font-family:var(--mono);font-size:0.66rem;letter-spacing:0.26em;text-transform:uppercase;color:var(--ink-low);margin:0;}
h1{font-size:clamp(2.1rem,0.9rem+4.6vw,4.6rem);line-height:1.0;letter-spacing:-0.032em;font-weight:700;
  margin:0.38em 0 0;max-width:17ch;text-wrap:balance;}
h2{font-size:clamp(1.35rem,1.1rem+0.9vw,1.75rem);line-height:1.2;letter-spacing:-0.014em;font-weight:700;margin:0 0 0.6em;text-wrap:balance;}
h3{font-size:1.06rem;font-weight:700;margin:2em 0 0.35em;text-wrap:balance;}
p{margin:0 0 0.95em;}
article ul,article ol{margin:0 0 1em;padding-left:1.35em;}
article li{margin-bottom:0.4em;}
article li > p{margin:0 0 0.4em;}
article li > p:last-child{margin-bottom:0;}
article li > ul,article li > ol{margin-top:0.4em;}
/* pandoc leaves these wrappers behind for LaTeX environments it half-converts */
.center,.minipage,.tabularx{margin:1.2rem 0;}
.tabularx{overflow-x:auto;font-size:0.85em;}
a{color:var(--chaotic);text-decoration:none;border-bottom:1px solid color-mix(in oklab,var(--chaotic) 35%,transparent);}
a:hover{border-bottom-color:currentColor;}
:focus-visible{outline:2px solid var(--chaotic);outline-offset:2px;border-radius:2px;}
code,.num{font-family:var(--mono);font-size:0.85em;font-variant-numeric:tabular-nums;}
pre{background:var(--sunken);border:1px solid var(--rule);border-radius:4px;padding:0.8rem 0.95rem;overflow-x:auto;font-size:0.78rem;line-height:1.55;}

/* ------------------------------ hero ------------------------------ */
.hero{position:relative;min-height:100svh;display:grid;grid-template-rows:1fr auto;overflow:hidden;border-bottom:1px solid var(--rule);contain:layout style;}
#bifurcation{position:absolute;inset:-8% 0 -8% 0;width:100%;height:116%;display:block;will-change:transform;}
/* the beacon plate paints over the attractor, both under the readability fade */
#hero-motes{position:absolute;inset:-8% 0 -8% 0;width:100%;height:116%;display:block;pointer-events:none;will-change:transform;}
.hero::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(100deg,var(--ground) 0%,color-mix(in oklab,var(--ground) 92%,transparent) 38%,color-mix(in oklab,var(--ground) 22%,transparent) 68%,transparent 100%);}
.hero-inner{position:relative;z-index:2;align-self:center;width:100%;max-width:min(112rem,96vw);
  margin:0 auto;padding:clamp(2rem,8vh,6rem) var(--gutter);}
.hero-inner > *{max-width:min(52rem,90%);}
.hero-rise{animation:hero-rise .6s cubic-bezier(.22,.68,.28,1) both;}
@keyframes hero-rise{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:none;}}
@media (prefers-reduced-motion:reduce){.hero-rise{animation:none;opacity:1;transform:none;}}
.byline{margin:1.6em 0 0;font-size:1rem;color:var(--ink);}
.byline .affil{display:block;color:var(--ink-low);font-size:0.86rem;margin-top:0.25em;max-width:44ch;}
.lede{font-size:clamp(1.06rem,0.9rem+0.62vw,1.4rem);line-height:1.55;color:var(--ink-mid);max-width:48ch;margin-top:1.5em;}
.stats{display:flex;flex-wrap:wrap;gap:2.25rem;margin-top:2.4rem;padding:0;list-style:none;}
.stats li{margin:0;}
.stats b{display:block;font-size:clamp(1.6rem,1.1rem+1.5vw,2.6rem);line-height:1.1;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;}
.stats span{font-family:var(--mono);font-size:0.62rem;letter-spacing:0.16em;text-transform:uppercase;color:var(--ink-low);}
.legend{position:relative;z-index:2;display:flex;flex-wrap:wrap;gap:0.3rem clamp(1rem,2.5vw,2.4rem);
  padding:0.9rem var(--gutter) 1.5rem;max-width:min(112rem,96vw);margin:0 auto;width:100%;font-family:var(--mono);font-size:0.66rem;letter-spacing:0.04em;color:var(--ink-mid);border-top:1px solid var(--rule-soft);}
.legend b{font-weight:400;color:var(--ink);}
.swatch{display:inline-block;width:0.62rem;height:0.62rem;margin-right:0.4rem;vertical-align:-1px;border-radius:2px;}

/* ------------------------------ controls ------------------------------ */
.controls{
  position:fixed;top:0;right:0;z-index:60;display:flex;align-items:center;gap:0.15rem;
  padding:0.4rem 0.55rem;background:color-mix(in oklab,var(--ground) 88%,transparent);
  backdrop-filter:blur(10px);border-left:1px solid var(--rule);border-bottom:1px solid var(--rule);
  border-bottom-left-radius:8px;
}
.controls button{
  appearance:none;background:none;border:1px solid transparent;border-radius:5px;
  color:var(--ink-mid);font-family:var(--mono);font-size:0.66rem;letter-spacing:0.06em;
  padding:0.34rem 0.5rem;cursor:pointer;line-height:1;transition:color .15s,background .15s,border-color .15s;
}
.controls button:hover{color:var(--ink);background:var(--sunken);}
.controls button[aria-pressed="true"]{color:var(--chaotic);border-color:color-mix(in oklab,var(--chaotic) 40%,transparent);}
.controls .sep{width:1px;height:1.1rem;background:var(--rule);margin:0 0.25rem;}
@media print{.controls,.progress{display:none;}}
/* Phones: 22-24 px buttons are below the 44 px comfortable touch target, and a
   top-right fixed bar can end up under iOS Safari's floating bottom chrome
   when the page is scrolled. Grow padding (not font-size) and reserve the
   safe-area inset so the bar always clears it. */
@media (max-width:36rem){
  html{scroll-padding-top:7rem;}
  .controls{
    left:0;right:0;max-width:100%;flex-wrap:wrap;justify-content:flex-end;
    border-left:none;border-bottom-left-radius:0;
    padding:0.35rem 0.55rem calc(0.35rem + env(safe-area-inset-bottom,0px));
    gap:0.2rem 0.15rem;
  }
  .controls button{padding:0.5rem 0.6rem;min-width:44px;min-height:44px;}
}

.progress{position:fixed;top:0;left:0;height:2px;background:var(--chaotic);z-index:61;width:0;transition:width .1s linear;}

/* ------------------------------ layout ------------------------------ */
.shell{display:grid;grid-template-columns:var(--rail) minmax(0,1fr);gap:clamp(1.5rem,3.5vw,4rem);
  max-width:min(112rem,96vw);margin:0 auto;padding:clamp(2rem,4vw,4rem) var(--gutter) 6rem;align-items:start;
  transition:grid-template-columns .45s cubic-bezier(.22,.68,.28,1),max-width .45s cubic-bezier(.22,.68,.28,1);}
:root[data-rail="off"] .shell{grid-template-columns:0 minmax(0,1fr);max-width:min(96rem,94vw);}
:root[data-rail="off"] .spine{opacity:0;pointer-events:none;overflow:hidden;}

.spine{position:sticky;top:3.6rem;max-height:calc(100svh - 5.5rem);overflow-y:auto;overscroll-behavior:contain;
  font-family:var(--sans);font-size:0.775rem;line-height:1.35;scrollbar-width:thin;
  scrollbar-color:var(--rule) transparent;padding-right:0.4rem;}
.spine::-webkit-scrollbar{width:6px;}
.spine::-webkit-scrollbar-thumb{background:var(--rule);border-radius:3px;}
.toc,.toc ul{list-style:none;margin:0;padding:0;}
.toc > li.top{margin-bottom:0.1rem;}
.toc a{display:block;padding:0.26rem 0 0.26rem 0.75rem;border-left:1.5px solid var(--rule-soft);
  color:var(--ink-low);text-decoration:none;border-bottom:none;
  transition:color .16s ease,border-color .16s ease;}
.toc a:hover{color:var(--ink);border-left-color:var(--ink-low);}
.toc a i{font-style:normal;font-variant-numeric:tabular-nums;color:var(--ink-low);
  margin-right:0.4em;font-size:0.92em;letter-spacing:0.01em;}
.toc > li.top > a{color:var(--ink-mid);font-weight:700;letter-spacing:-0.005em;}
.toc > li.top > a.on{color:var(--ink);border-left-color:var(--chaotic);}
.toc > li.top > a.on i{color:var(--chaotic);}
.toc .sub{max-height:0;overflow:hidden;opacity:0;transition:max-height .3s ease,opacity .2s ease;}
.toc > li.top.open > .sub{max-height:40rem;opacity:1;}
.toc .sub a{padding-left:1.55rem;font-size:0.735rem;}
.toc .sub a.on{color:var(--ink);border-left-color:var(--chaotic);}
@media (max-width:66rem){
  .shell{grid-template-columns:minmax(0,1fr);}
  .spine{display:none;}
  /* Drawer: the very same .spine/.toc markup, not a duplicate list. The
     "contents" control toggles html.toc-open at this same breakpoint. */
  html.toc-open .spine{
    display:block;position:fixed;inset:0 auto 0 0;z-index:90;
    width:min(20rem,84vw);height:100svh;max-height:none;
    background:var(--raised);box-shadow:var(--shadow);
    padding:4.5rem 1.2rem 2rem;overflow-y:auto;
    animation:drawer-in .22s ease forwards;
  }
  html.toc-open .toc-backdrop{display:block;}
}
@keyframes drawer-in{from{transform:translateX(-100%);}to{transform:translateX(0);}}
@media (prefers-reduced-motion:reduce){html.toc-open .spine{animation:none;}}
.toc-backdrop{
  display:none;position:fixed;inset:0;z-index:89;
  background:color-mix(in oklab,var(--sunken) 70%,transparent);backdrop-filter:blur(4px);
}
.toc-backdrop[hidden]{display:none;}

.secno{color:var(--ink-low);font-variant-numeric:tabular-nums;font-weight:400;margin-right:0.5em;letter-spacing:0.01em;}
h2 .secno{font-size:0.78em;}
h3 .secno{font-size:0.85em;}

details.backmatter{margin:0;}
details.backmatter > summary{list-style:none;cursor:pointer;display:block;}
details.backmatter > summary::-webkit-details-marker{display:none;}
details.backmatter > summary h2{display:flex;align-items:baseline;gap:0.5rem;margin-bottom:0.9em;}
details.backmatter > summary h2::after{
  content:"show";font-family:var(--mono);font-size:0.58rem;letter-spacing:0.14em;
  text-transform:uppercase;color:var(--ink-low);border:1px solid var(--rule);
  border-radius:4px;padding:0.18rem 0.42rem;margin-left:auto;flex:none;transition:all .15s;}
details.backmatter[open] > summary h2::after{content:"hide";}
details.backmatter > summary:hover h2::after{color:var(--chaotic);border-color:var(--chaotic);}

main{display:block;min-width:0;}
article{max-width:var(--measure);margin:0 auto;}
article > section{scroll-margin-top:3.5rem;content-visibility:auto;contain-intrinsic-size:auto 600px;}
article > section:first-child{content-visibility:visible;}
article > section > h2,
article > section > details > summary > h2{border-top:1px solid var(--rule);padding-top:1.4rem;margin-top:2.8rem;}
article > section:first-child > h2{margin-top:0;border-top:none;padding-top:0;}

/* ------------------------------ figures ------------------------------ */
figure{margin:2.6rem 0;width:var(--bleed);max-width:calc(100vw - 2*var(--gutter));
  position:relative;left:50%;transform:translateX(-50%);background:var(--raised);border:1px solid var(--rule);border-radius:6px;overflow:hidden;
  box-shadow:var(--shadow);transition:box-shadow .3s ease,border-color .3s ease,transform .3s ease;}
figure:hover{border-color:color-mix(in oklab,var(--chaotic) 30%,var(--rule));
  box-shadow:0 2px 4px rgba(20,22,28,.05),0 18px 50px rgba(20,22,28,.12);
  transform:translateX(-50%) translateY(-2px);}
@media (prefers-reduced-motion:reduce){figure,figure:hover{transition:none;transform:translateX(-50%);}}
.fig-head{display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0.7rem;border-bottom:1px solid var(--rule-soft);}
.fig-head .name{font-family:var(--mono);font-size:0.63rem;color:var(--ink-low);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.fig-head .acts{margin-left:auto;display:flex;gap:0.25rem;flex:none;}
.fig-head button{appearance:none;background:none;border:1px solid var(--rule);border-radius:4px;color:var(--ink-mid);
  font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;padding:0.24rem 0.45rem;cursor:pointer;transition:all .15s;}
.fig-head button:hover{color:var(--chaotic);border-color:var(--chaotic);}
/* always-visible: a reader who followed a shared link must never be silently
   looking at a non-published view without a way to tell and undo it. */
.fig-badge{appearance:none;background:none;border:1px solid var(--chaotic);border-radius:4px;
  color:var(--chaotic);font-family:var(--mono);font-size:0.6rem;letter-spacing:0.06em;
  text-transform:uppercase;padding:0.24rem 0.5rem;cursor:pointer;white-space:nowrap;flex:none;}
.fig-badge:hover{background:var(--chaotic);color:var(--raised);}
.fig-badge[hidden]{display:none;}
.fig-body{position:relative;background:var(--raised);}
figure img{display:block;width:100%;height:auto;max-height:clamp(14rem,34vh,26rem);object-fit:contain;object-position:center;
  cursor:zoom-in;padding:0.5rem;transition:max-height .35s ease;}
:root[data-figsize="tall"] figure img{max-height:min(78vh,46rem);}
figcaption{padding:0.6rem 0.8rem 0.75rem;font-size:0.82rem;line-height:1.5;color:var(--ink-mid);border-top:1px solid var(--rule-soft);}
figcaption .num{color:var(--ink);font-weight:700;}
/* ------------------------------ figure code snippets ------------------------------ */
details.fig-code{margin:0;border-top:1px solid var(--rule-soft);background:var(--sunken);}
details.fig-code > summary{list-style:none;cursor:pointer;display:none;}
details.fig-code > summary::-webkit-details-marker{display:none;}
details.fig-code pre{margin:0;padding:0.7rem 0.8rem;overflow-x:auto;background:var(--sunken);}
details.fig-code code.fig-snippet{font-family:var(--mono);font-size:0.72rem;line-height:1.5;color:var(--ink);white-space:pre;}
.fig-code-actions{display:flex;justify-content:flex-end;padding:0 0.8rem 0.5rem;}
.fig-code-copy{appearance:none;background:none;border:1px solid var(--rule);border-radius:4px;color:var(--ink-mid);
  font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;padding:0.22rem 0.45rem;cursor:pointer;transition:all .15s;}
.fig-code-copy:hover{color:var(--chaotic);border-color:var(--chaotic);}
.fig-code-copy.copied{color:var(--chaotic);border-color:var(--chaotic);}
.fig-code-copy.copied::after{content:" copied";}
.fig-code-footer{margin:0;padding:0 0.8rem 0.7rem;font-family:var(--mono);font-size:0.62rem;letter-spacing:0.02em;color:var(--ink-low);}
.fig-code-footer code{color:var(--ink-mid);}
.tok-comment{color:var(--ink-low);font-style:italic;}
.tok-string{color:var(--locked);}
.tok-kw{color:var(--chaotic);}
.tok-num{color:var(--torus);}
.tok-fn{color:var(--ink);font-weight:700;}
.tok-mod{color:var(--ink);font-weight:700;}
.plot-wrap{position:relative;padding:0.35rem;}
/* text-transform would fold rho/eps/alpha into P/E/A, so app.js applies the
   same caps to the title text one ASCII letter at a time (titleCaps). */
.plot-title{margin:0 6rem 0.15rem;text-align:center;font-family:var(--mono);font-size:0.62rem;
  letter-spacing:0.1em;color:var(--ink-low);}
canvas.plot{width:100%;display:block;touch-action:pan-y;}
canvas.plot.live{touch-action:none;}
.hint{margin:0;padding:0 0.8rem 0.6rem;font-family:var(--mono);font-size:0.6rem;letter-spacing:0.06em;color:var(--ink-low);}
.live-params{display:flex;flex-wrap:wrap;gap:0.4rem 1.2rem;padding:0.2rem 0.8rem 0.4rem;
  font-family:var(--mono);font-size:0.66rem;color:var(--ink-low);}
.live-param{display:flex;align-items:center;gap:0.45rem;}
.live-param-name{letter-spacing:0.08em;}
.live-param input[type="range"]{width:9rem;accent-color:var(--chaotic);}
.live-param input[type="number"]{width:4.2rem;font:inherit;color:var(--ink);
  background:var(--sunken);border:1px solid var(--rule);border-radius:4px;padding:0.1rem 0.3rem;}
.live-presets{display:flex;flex-wrap:wrap;gap:0.4rem;padding:0 0.8rem 0.4rem;}
.live-presets button{appearance:none;background:none;border:1px solid var(--rule);border-radius:4px;
  color:var(--ink-mid);font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;
  text-transform:uppercase;padding:0.24rem 0.45rem;cursor:pointer;transition:all .15s;}
.live-presets button:hover{color:var(--chaotic);border-color:var(--chaotic);}
.live-debug-report{display:flex;align-items:center;justify-content:space-between;gap:0.6rem;
  margin:0;padding:0.3rem 0.8rem 0.6rem;font-family:var(--mono);font-size:0.62rem;letter-spacing:0.04em;
  color:var(--ink-low);border-top:1px dashed var(--rule);}
.live-debug-line{word-break:break-all;white-space:normal;flex:1;}
.live-debug-copy{appearance:none;background:none;border:1px solid var(--rule);border-radius:3px;
  color:var(--ink-mid);font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;
  padding:0.15rem 0.4rem;cursor:pointer;flex-shrink:0;}
.live-debug-copy:hover{color:var(--chaotic);border-color:var(--chaotic);}
/* research-arc overview: a timeline, not a table */
figure.arc{background:var(--raised);}
.arc-track{list-style:none;margin:0;padding:1.4rem 1rem 1.2rem;display:grid;
  grid-template-columns:repeat(6,minmax(0,1fr));gap:0;position:relative;}
.arc-track::before{content:"";position:absolute;left:1.6rem;right:1.6rem;top:2.55rem;height:1.5px;
  background:linear-gradient(90deg,var(--locked),var(--torus),var(--chaotic));opacity:.45;}
.arc-track li{display:flex;flex-direction:column;gap:0.18rem;padding:0 0.45rem;margin:0;
  text-align:left;position:relative;}
.arc-track li::before{content:"";position:absolute;left:0.45rem;top:1.72rem;width:7px;height:7px;
  border-radius:50%;border:1.5px solid var(--raised);z-index:1;
  background:color-mix(in oklab,var(--locked) calc((5 - var(--i)) * 20%),var(--chaotic));}
.arc-track .yr{font-family:var(--mono);font-size:0.63rem;letter-spacing:0.1em;color:var(--ink-low);}
.arc-track .topic{margin-top:1.35rem;font-weight:700;font-size:0.9rem;line-height:1.2;}
.arc-track .mech{font-size:0.78rem;color:var(--ink-mid);line-height:1.25;}
@media (max-width:44rem){
  .arc-track{grid-template-columns:repeat(2,minmax(0,1fr));gap:1.1rem 0;}
  .arc-track::before{display:none;}
  .arc-track li::before{display:none;}
  .arc-track .topic{margin-top:0.15rem;}
}

.readout{position:absolute;pointer-events:none;z-index:4;background:var(--sunken);border:1px solid var(--rule);
  border-radius:4px;padding:0.3rem 0.45rem;font-family:var(--mono);font-size:0.66rem;line-height:1.45;
  color:var(--ink);white-space:pre;opacity:0;transition:opacity .1s;box-shadow:var(--shadow);}
.readout.on{opacity:1;}
/* touch has no dblclick, so a zoomed plot needs a visible way back; shown only
   once the domain has actually moved away from its base extent. */
.plot-reset{position:absolute;top:0.45rem;right:0.45rem;z-index:5;appearance:none;
  background:var(--raised);color:var(--ink-mid);border:1px solid var(--rule);border-radius:4px;
  font-family:var(--mono);font-size:0.62rem;letter-spacing:0.06em;text-transform:uppercase;
  padding:0.3rem 0.5rem;cursor:pointer;}
.plot-reset:hover{color:var(--chaotic);border-color:var(--chaotic);}
.fetch-err{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.02em;color:var(--locked);margin-left:0.4rem;}

/* ------------------------------ tables + refs ------------------------------ */
.fig-note{margin:0 0 0.8rem;font-size:0.82rem;line-height:1.5;color:var(--ink-mid);}
.table-wrap{overflow-x:auto;margin:2.2rem 0;border:1px solid var(--rule);border-radius:6px;
  background:var(--raised);box-shadow:var(--shadow);
  width:var(--bleed);max-width:calc(100vw - 2*var(--gutter));position:relative;left:50%;transform:translateX(-50%);}
table{border-collapse:collapse;width:100%;font-size:0.9rem;line-height:1.45;
  font-variant-numeric:tabular-nums;}
caption{caption-side:top;text-align:left;padding:0.85rem 1rem 0.75rem;
  font-size:0.86rem;line-height:1.45;color:var(--ink-mid);border-bottom:1px solid var(--rule);}
caption .num{color:var(--ink);font-weight:700;}
thead th{
  position:sticky;top:0;z-index:1;
  background:var(--sunken);
  font-family:var(--sans);font-size:0.7rem;font-weight:700;
  letter-spacing:0.07em;text-transform:uppercase;color:var(--ink-mid);
  padding:0.6rem 1rem;white-space:nowrap;
  border-bottom:1px solid var(--rule);}
tbody td{padding:0.52rem 1rem;border-bottom:1px solid var(--rule-soft);vertical-align:baseline;}
tbody tr:last-child td{border-bottom:none;}
tbody tr:hover td{background:color-mix(in oklab,var(--chaotic) 5%,transparent);}
/* first column of a reference table is the symbol being defined */
tbody td:first-child{font-weight:700;white-space:nowrap;}
tbody td math{font-size:1em;}
table th[style],table td[style]{text-align:left !important;}

/* 144 citations: they must be reachable without shouting over the prose */
.citation{white-space:nowrap;}
.citation a,a[role="doc-biblioref"]{
  color:inherit;border-bottom:1px solid color-mix(in oklab,var(--chaotic) 30%,transparent);
  transition:color .15s ease,border-color .15s ease;}
.citation a:hover,a[role="doc-biblioref"]:hover{color:var(--chaotic);border-bottom-color:currentColor;}
/* the entry you jumped to should be findable when you land on it */
.csl-entry:target{background:color-mix(in oklab,var(--chaotic) 9%,transparent);
  border-radius:4px;box-shadow:0 0 0 0.5rem color-mix(in oklab,var(--chaotic) 9%,transparent);}
.csl-entry a{color:var(--ink-mid);border-bottom-color:color-mix(in oklab,var(--ink-low) 40%,transparent);
  overflow-wrap:anywhere;}
.csl-entry a:hover{color:var(--chaotic);}
#refs,.references{margin-top:1.2rem;font-size:0.82rem;}
.csl-entry{margin:0 0 0.6em;padding-left:1.5em;text-indent:-1.5em;color:var(--ink-mid);line-height:1.45;}

/* ------------------------------ lightbox ------------------------------ */
.lb{position:fixed;inset:0;z-index:70;display:none;place-items:center;background:color-mix(in oklab,var(--sunken) 94%,transparent);backdrop-filter:blur(8px);padding:2.5rem 1.5rem;}
.lb.on{display:grid;animation:lbin .22s ease forwards;}
@keyframes lbin{from{opacity:0;}to{opacity:1;}}
.lb.on img{animation:lbimg .28s cubic-bezier(.22,.68,.28,1) forwards;}
@keyframes lbimg{from{opacity:0;transform:scale(.975);}to{opacity:1;transform:none;}}
@media (prefers-reduced-motion:reduce){.lb.on,.lb.on img{animation:none;}}
.lb img{max-width:100%;max-height:88svh;object-fit:contain;border-radius:3px;box-shadow:0 24px 70px rgba(0,0,0,.45);}
.lb-cap{position:absolute;left:0;right:0;bottom:2.4rem;padding:0.9rem 1.5rem 0;font-size:0.8rem;color:var(--ink-mid);text-align:center;max-width:60rem;margin:0 auto;}
.lb-actions{position:absolute;left:0;right:0;bottom:0;margin:0;padding:0 1.5rem 0.7rem;text-align:center;}
.lb-goto{font-family:var(--mono);font-size:0.7rem;letter-spacing:0.03em;color:var(--ink-mid);
  border-bottom:1px solid color-mix(in oklab,var(--ink-mid) 40%,transparent);}
.lb-goto:hover{color:var(--chaotic);}
.lb-close{position:absolute;top:0.9rem;right:1.1rem;background:var(--raised);color:var(--ink);border:1px solid var(--rule);border-radius:5px;font-family:var(--mono);font-size:0.66rem;letter-spacing:0.08em;text-transform:uppercase;padding:0.4rem 0.65rem;cursor:pointer;}
.lb-nav{
  position:absolute;top:50%;transform:translateY(-50%);
  appearance:none;background:var(--raised);color:var(--ink);
  border:1px solid var(--rule);border-radius:5px;
  font-family:var(--mono);font-size:1rem;line-height:1;
  width:2.2rem;height:2.2rem;padding:0;cursor:pointer;
  opacity:.72;transition:opacity .15s,border-color .15s,color .15s;
}
.lb-nav:hover{opacity:1;border-color:var(--chaotic);color:var(--chaotic);}
.lb-prev{left:0.9rem;}
.lb-next{right:0.9rem;}
/* A position in the sequence rather than a figure number. Kept visually apart
   from the caption so neither can be read as the other. */
.lb-pos{position:absolute;top:1rem;left:50%;transform:translateX(-50%);margin:0;
  font-family:var(--mono);font-size:0.68rem;letter-spacing:0.12em;
  color:var(--ink-low);pointer-events:none;}
/* Cheapest possible shortcut-discovery fix: the keys are already wired, they
   were just never mentioned. Decorative only -- .lb-pos already carries its
   own aria-label, so this is not read twice by a screen reader. */
.lb-pos::after{content:" \00b7 \2190 \2192 esc";opacity:.7;}

/* ------------------------------ search ------------------------------ */
.search-overlay{position:fixed;inset:0;z-index:80;display:none;place-items:start center;
  background:color-mix(in oklab,var(--sunken) 88%,transparent);backdrop-filter:blur(6px);
  padding:4.5rem 1.2rem 2rem;}
.search-overlay.on{display:grid;animation:lbin .18s ease forwards;}
@media (prefers-reduced-motion:reduce){.search-overlay.on{animation:none;}}
.search-panel{width:100%;max-width:38rem;background:var(--raised);color:var(--ink);
  border:1px solid var(--rule);border-radius:8px;box-shadow:var(--shadow);
  overflow:hidden;display:flex;flex-direction:column;max-height:calc(100svh - 6rem);}
.search-input{border:0;border-bottom:1px solid var(--rule);background:none;color:var(--ink);
  font-family:var(--serif);font-size:1.05rem;padding:0.9rem 1rem;width:100%;outline:none;flex:none;}
.search-results{list-style:none;margin:0;padding:0.3rem;overflow-y:auto;}
.search-results li{margin:0;}
.search-results button{appearance:none;display:block;width:100%;text-align:left;background:none;
  border:0;border-radius:5px;padding:0.55rem 0.7rem;cursor:pointer;color:var(--ink);
  font-family:var(--serif);}
.search-results button:hover,.search-results button.sel{background:var(--sunken);}
.search-hit-sec{display:block;font-family:var(--mono);font-size:0.62rem;letter-spacing:0.08em;
  text-transform:uppercase;color:var(--ink-low);margin-bottom:0.2rem;}
.search-hit-snip{display:block;font-size:0.85rem;line-height:1.4;color:var(--ink-mid);}
.search-hit-snip mark{background:color-mix(in oklab,var(--chaotic) 32%,transparent);
  color:var(--ink);border-radius:2px;}
.search-empty{margin:0;padding:1rem;font-size:0.85rem;color:var(--ink-low);}
.search-overlay:not(.empty) .search-empty{display:none;}
@media (max-width:36rem){
  .search-overlay{padding:0;}
  .search-panel{max-width:none;height:100svh;max-height:none;border-radius:0;border:0;}
  .search-input{font-size:16px;padding:1.1rem 1rem;}
  .citation{white-space:normal;}
}

/* ------------------------------ help dialog ------------------------------ */
.help-overlay{position:fixed;inset:0;z-index:80;display:none;place-items:center;
  background:color-mix(in oklab,var(--sunken) 88%,transparent);backdrop-filter:blur(6px);
  padding:2rem 1.2rem;}
.help-overlay.on{display:grid;animation:lbin .18s ease forwards;}
@media (prefers-reduced-motion:reduce){.help-overlay.on{animation:none;}}
.help-panel{width:100%;max-width:30rem;max-height:calc(100svh - 4rem);overflow-y:auto;
  background:var(--raised);color:var(--ink);border:1px solid var(--rule);
  border-radius:8px;box-shadow:var(--shadow);padding:1rem 1.2rem 1.3rem;}
.help-head{display:flex;align-items:baseline;gap:1rem;margin-bottom:0.5rem;}
.help-head h2{margin:0;flex:1;font-size:1.05rem;}
.help-close{appearance:none;background:none;border:1px solid var(--rule);border-radius:5px;
  color:var(--ink-mid);font-family:var(--mono);font-size:0.66rem;letter-spacing:0.08em;
  text-transform:uppercase;padding:0.34rem 0.55rem;cursor:pointer;flex:none;}
.help-close:hover{color:var(--chaotic);border-color:var(--chaotic);}
.help-table{width:100%;border-collapse:collapse;font-size:0.85rem;}
.help-table th{text-align:left;font-family:var(--sans);font-size:0.68rem;
  letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-low);
  padding:0.3rem 0.5rem;border-bottom:1px solid var(--rule);}
.help-table td{padding:0.4rem 0.5rem;border-bottom:1px solid var(--rule-soft);vertical-align:top;}
.help-table td:first-child{font-family:var(--mono);white-space:nowrap;color:var(--ink);}

/* ------------------------------ heading copy-link ------------------------------ */
h2 .head-anchor,h3 .head-anchor{
  display:inline-block;opacity:0;margin-left:0.45em;font-family:var(--mono);
  font-size:0.75em;font-weight:400;color:var(--ink-low);text-decoration:none;
  border-bottom:none;transition:opacity .15s ease;
}
h2:hover .head-anchor,h2:focus-within .head-anchor,
h3:hover .head-anchor,h3:focus-within .head-anchor{opacity:1;}
.head-anchor.copied::after,.act-copylink.copied::after{content:" copied";font-size:0.85em;color:var(--chaotic);}
@media (prefers-reduced-motion:reduce){h2 .head-anchor,h3 .head-anchor{transition:none;}}

/* ------------------------------ xref popover + return pill ------------------------------ */
.xref-pop{
  position:fixed;z-index:65;max-width:min(32rem,calc(100vw - 2rem));
  background:var(--raised);color:var(--ink);border:1px solid var(--rule);
  border-radius:6px;box-shadow:var(--shadow);padding:0.7rem 0.85rem;
  font-size:0.82rem;line-height:1.45;
  opacity:0;transform:scale(.98) translateY(4px);visibility:hidden;
  transition:opacity .12s ease,transform .12s ease,visibility .12s;
  pointer-events:none;
}
.xref-pop.on{opacity:1;transform:none;visibility:visible;pointer-events:auto;}
.xref-pop .xref-body{max-height:min(40vh,22rem);overflow:auto;}
.xref-pop .xref-body .csl-entry{margin:0;padding-left:0;text-indent:0;color:var(--ink-mid);}
.xref-pop .xref-body .eqn{margin:0.35em 0;}
.xref-pop .xref-label{font-family:var(--mono);font-size:0.66rem;letter-spacing:0.08em;
  text-transform:uppercase;color:var(--ink-low);margin:0 0 0.35rem;}
.xref-pop .xref-figimg{display:block;max-width:100%;height:auto;max-height:12rem;
  object-fit:contain;margin:0 0 0.4rem;border-radius:3px;}
.xref-pop .xref-cap{color:var(--ink-mid);font-size:0.78rem;margin:0;line-height:1.45;}

.return-pill{
  position:fixed;left:1rem;bottom:1.2rem;z-index:55;
  appearance:none;background:var(--raised);color:var(--ink);
  border:1px solid var(--rule);border-radius:999px;box-shadow:var(--shadow);
  font-family:var(--mono);font-size:0.7rem;letter-spacing:0.02em;
  padding:0.55rem 0.95rem;cursor:pointer;line-height:1;
  opacity:0;transform:translateY(8px);visibility:hidden;pointer-events:none;
  transition:opacity .18s ease,transform .18s ease,visibility .18s;
}
.return-pill.on{opacity:1;transform:none;visibility:visible;pointer-events:auto;}
.return-pill:hover{border-color:var(--chaotic);color:var(--chaotic);}

.cited-in{
  margin:0.2em 0 0.65em;padding-left:1.5em;text-indent:0;
  font-family:var(--mono);font-size:0.72em;color:var(--ink-low);line-height:1.4;
}
.cited-in a{color:var(--ink-low);border-bottom-color:color-mix(in oklab,var(--ink-low) 35%,transparent);}
.cited-in a:hover{color:var(--chaotic);}

html.js .reveal{opacity:0;transform:translateY(14px);transition:opacity .6s cubic-bezier(.22,.68,.28,1),transform .6s cubic-bezier(.22,.68,.28,1);}
html.js figure{opacity:0;transform:translateX(-50%) translateY(22px) scale(.985);}
html.js figure.seen{opacity:1;transform:translateX(-50%) translateY(0) scale(1);
  transition:opacity .7s cubic-bezier(.22,.68,.28,1),transform .7s cubic-bezier(.22,.68,.28,1),box-shadow .3s ease,border-color .3s ease;}
@media (prefers-reduced-motion:reduce){html.js figure{opacity:1;transform:translateX(-50%);}}
html.js .reveal.in{opacity:1;transform:none;}
@media (prefers-reduced-motion:reduce){
  html.js .reveal{opacity:1;transform:none;transition:none;}
  html{scroll-behavior:auto;}.progress{transition:none;}
  .xref-pop,.return-pill{transition:none;transform:none;}
}

@media print{
  :root{--ground:#fff;--raised:#fff;--ink:#000;--ink-mid:#333;--ink-low:#666;--rule:#bbb;--rule-soft:#ddd;}
  .hero{min-height:auto;border-bottom:2px solid #000;page-break-after:avoid;}
  #bifurcation,#hero-motes,.hero::after,.legend,.spine,.controls,.progress,.lb,.fig-head .acts,
  .xref-pop,.return-pill,.lb-nav,.lb-pos,.search-overlay{display:none !important;}
  .shell{display:block;max-width:none;padding:0;}
  article{max-width:none;}
  figure{break-inside:avoid;box-shadow:none;border:1px solid #bbb;}
  figure img{max-height:none;}
  details.backmatter > summary h2::after{display:none;}
  details.backmatter{display:block;}
  details.backmatter > summary{list-style:none;}
  h2,h3{page-break-after:avoid;}
  a{color:#000;border-bottom:none;}
  .foot{border-top:1px solid #bbb;}
}

.foot{border-top:1px solid var(--rule);padding:2.2rem var(--gutter);max-width:72rem;margin:0 auto;color:var(--ink-mid);font-size:0.83rem;}
"""
)


JS = r"""
"use strict";
document.documentElement.classList.add("js");
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
// prefers-reduced-data has patchy support; matchMedia on an unrecognised media
// feature simply never matches, so testing it is always safe.
const reducedData=matchMedia("(prefers-reduced-data: reduce)").matches;
const root=document.documentElement;
const store=(k,v)=>{try{v===null?localStorage.removeItem(k):localStorage.setItem(k,v);}catch(e){}};
const load=k=>{try{return localStorage.getItem(k);}catch(e){return null;}};

// Shared by every dialog-like overlay (lightbox, search, help, the mobile TOC
// drawer): a Tab keydown handler that cycles focus among `getFocusables()`
// instead of letting it escape into the page behind the overlay.
function makeFocusTrap(getFocusables){
  return function(e){
    if(e.key!=="Tab") return;
    const els=getFocusables();
    if(!els.length) return;
    const first=els[0],last=els[els.length-1];
    const active=document.activeElement;
    if(e.shiftKey&&(active===first||!els.includes(active))){
      e.preventDefault();last.focus();
    }else if(!e.shiftKey&&(active===last||!els.includes(active))){
      e.preventDefault();first.focus();
    }
  };
}

/* ---------------- reading controls ---------------- */
(function(){
  const SIZES=["s","m","l","xl"];
  let size=load("dc-size")||"l";
  let theme=load("dc-theme")||"auto";
  let rail=load("dc-rail")||"on";
  let fig=load("dc-fig")||"short";

  const apply=()=>{
    root.setAttribute("data-size",size);
    if(theme==="auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme",theme);
    root.setAttribute("data-rail",rail);
    root.setAttribute("data-figsize",fig);
  };
  apply();

  const bar=document.querySelector(".controls");
  if(!bar) return;
  const $=s=>bar.querySelector(s);
  const sync=()=>{
    $(".c-theme").textContent=theme==="auto"?"auto":theme;
    $(".c-theme").setAttribute("aria-pressed",theme!=="auto");
    $(".c-rail").setAttribute("aria-pressed",rail==="on");
    $(".c-fig").setAttribute("aria-pressed",fig==="tall");
    $(".c-smaller").disabled=size==="s";
    $(".c-larger").disabled=size==="xl";
  };
  $(".c-theme").addEventListener("click",()=>{
    theme=theme==="auto"?"light":theme==="light"?"dark":"auto";
    store("dc-theme",theme==="auto"?null:theme);apply();sync();repaint();
  });
  $(".c-rail").addEventListener("click",()=>{
    rail=rail==="on"?"off":"on";store("dc-rail",rail);apply();sync();
    setTimeout(()=>MOUNTED.forEach(p=>p.resize()),60);
  });
  $(".c-fig").addEventListener("click",()=>{
    fig=fig==="short"?"tall":"short";store("dc-fig",fig);apply();sync();
  });
  const step=d=>{
    const i=Math.max(0,Math.min(SIZES.length-1,SIZES.indexOf(size)+d));
    size=SIZES[i];store("dc-size",size);apply();sync();
    setTimeout(()=>MOUNTED.forEach(p=>p.resize()),60);
  };
  $(".c-smaller").addEventListener("click",()=>step(-1));
  $(".c-larger").addEventListener("click",()=>step(1));
  sync();
})();

/* ---------------- reading progress ---------------- */
(function(){
  const bar=document.querySelector(".progress");
  const art=document.querySelector("article");
  if(!bar||!art) return;
  let ticking=false;
  const upd=()=>{
    ticking=false;
    const top=art.offsetTop, h=art.offsetHeight-innerHeight;
    const p=h>0?Math.min(1,Math.max(0,(scrollY-top)/h)):0;
    bar.style.width=(p*100).toFixed(2)+"%";
  };
  const req=()=>{if(!ticking){ticking=true;requestAnimationFrame(upd);}};
  addEventListener("scroll",req,{passive:true});
  addEventListener("resize",req,{passive:true});
  if("requestIdleCallback" in window){requestIdleCallback(upd);}
  else setTimeout(upd,150);
})();

/* ---------------- reading position: persist + restore scroll ---------------- */
(function(){
  const KEY="dc-scroll:"+location.pathname;
  let pending=null;
  addEventListener("scroll",()=>{
    if(pending) return;
    pending=setTimeout(()=>{
      pending=null;
      try{sessionStorage.setItem(KEY,String(Math.round(scrollY)));}catch(e){}
    },200);
  },{passive:true});

  function restore(){
    // A hash already names a place to land; do not fight it with a second
    // scroll target.
    if(location.hash) return;
    let y=null;
    try{y=sessionStorage.getItem(KEY);}catch(e){}
    y=y?parseInt(y,10):0;
    if(!y) return;
    scrollTo(0,y);
  }
  function afterSettle(){
    const idle=window.requestIdleCallback||(cb=>setTimeout(cb,0));
    idle(restore);
  }
  if(document.readyState==="complete") afterSettle();
  else addEventListener("load",afterSettle);
})();

/* ---------------- hero: logistic attractor coloured by sign of lambda ----------------
   After the initial sweep the map keeps iterating: the resonance columns gain a
   few more points every frame while a faint wash of the ground colour holds the
   density at equilibrium. The shimmer is the computation continuing, not an
   effect layered on top of it. A second canvas carries slow-breathing beacons,
   each one iterated onto the attractor first so none of them float in the gaps
   between branches. Both plates stop when the hero scrolls out of view. */
function hero(){
  const cv=document.getElementById("bifurcation");
  const mcv=document.getElementById("hero-motes");
  if(!cv) return ()=>{};
  const ctx=cv.getContext("2d",{alpha:false});
  const mctx=mcv?mcv.getContext("2d",{alpha:true}):null;
  const still=reduced||reducedData;
  let raf=null,col=0,W=0,H=0,tick=0;
  let sweeping=false,shimmerOn=false,starsOn=false;
  let seedOffset=0;

  const R0=2.85, R1=4.0, DR=R1-R0;
  let lam=new Float32Array(0);

  function column(px,iters,alpha){
    const r=R0+DR*(px/W);
    let x=0.35+0.3*((px*2654435761+seedOffset*17)%1000)/1000,l=0;
    for(let i=0;i<380;i++) x=r*x*(1-x);
    for(let i=0;i<160;i++){x=r*x*(1-x);l+=Math.log(Math.abs(r*(1-2*x))+1e-12);}
    l/=160;
    if(lam.length>px) lam[px]=l;
    ctx.fillStyle=l>0.005?css("--chaotic"):l<-0.005?css("--locked-hero"):css("--torus");
    ctx.globalAlpha=alpha;
    for(let i=0;i<iters;i++){x=r*x*(1-x);ctx.fillRect(px,(1-x)*H,1.15,1.15);}
    ctx.globalAlpha=1;
  }

  // ---------------- starlight beacons, pinned to the attractor ----------------
  const N=80;
  const starPx=new Float32Array(N), starPy=new Float32Array(N);
  const starR=new Float32Array(N);
  const starPhase=new Float32Array(N), starFreq=new Float32Array(N);

  function lamAt(rv){
    if(!lam.length||!W) return 0;
    const ix=Math.max(0,Math.min(W-1,Math.round(((rv-R0)/DR)*W)));
    return lam[ix];
  }

  // Spread the beacons over the features worth looking at, then iterate each
  // one 400 steps so it lands on a point the map really visits.
  function seedStars(){
    for(let i=0;i<N;i++){
      let rv;
      if(i<10)      rv=2.90+Math.random()*0.50; // period 1 and 2
      else if(i<25) rv=3.45+Math.random()*0.12; // period 4, 8, cascade
      else if(i<40) rv=3.82+Math.random()*0.04; // period-3 window
      else          rv=3.57+Math.random()*0.42; // chaotic bands
      starR[i]=rv;

      let x=0.4+0.2*Math.random();
      for(let k=0;k<400;k++) x=rv*x*(1-x);
      const extra=Math.floor(Math.random()*32);
      for(let k=0;k<extra;k++) x=rv*x*(1-x);

      starPx[i]=((rv-R0)/DR)*W;
      starPy[i]=(1-x)*H;
      starPhase[i]=Math.random()*Math.PI*2;
      starFreq[i]=0.006+0.014*Math.random(); // one breath every 8 to 15 s
    }
  }

  function drawStars(){
    if(!mctx||!W||!H) return;
    mctx.clearRect(0,0,W,H);
    const colC=css("--chaotic"), colL=css("--locked-hero"), colT=css("--torus");
    for(let i=0;i<N;i++){
      if(starPx[i]>col) continue; // never render ahead of the sweep line
      starPhase[i]+=starFreq[i];
      const twinkle=Math.pow(0.5+0.5*Math.sin(starPhase[i]),3);
      const ly=lamAt(starR[i]);
      mctx.fillStyle=ly>0.005?colC:ly<-0.005?colL:colT;
      mctx.globalAlpha=0.20+0.65*twinkle;
      const size=1.1+1.8*twinkle;
      mctx.fillRect(starPx[i]-size/2,starPy[i]-size/2,size,size);
      if(twinkle>0.82){
        mctx.globalAlpha=(0.20+0.65*twinkle)*0.55;
        const flare=3.0+3.0*twinkle;
        mctx.fillRect(starPx[i]-flare/2,starPy[i]-0.4,flare,0.8);
        mctx.fillRect(starPx[i]-0.4,starPy[i]-flare/2,0.8,flare);
      }
    }
    mctx.globalAlpha=1;
  }

  function needFrame(){ return sweeping||shimmerOn||starsOn; }
  function schedule(){ if(!raf&&needFrame()) raf=requestAnimationFrame(frame); }
  function frame(){
    raf=null;
    tick++;
    if(sweeping) stepSweep();
    if(starsOn) drawStars();
    if(shimmerOn) stepShimmer();
    if(needFrame()) raf=requestAnimationFrame(frame);
  }

  // Reduced motion gets the finished plate in one pass instead of a sweep.
  function stepSweep(){
    const end=still?W:Math.min(W,col+5);
    for(;col<end;col++){
      ctx.fillStyle=css("--ground");ctx.fillRect(col,0,1.2,H);
      column(col,280,0.46);
    }
    if(col>=W){
      sweeping=false;
      if(!still) shimmerOn=true;
    }
  }

  function stepShimmer(){
    if(tick%5===0){
      ctx.globalAlpha=0.006;
      ctx.fillStyle=css("--ground");
      ctx.fillRect(0,0,W,H);
      ctx.globalAlpha=1;
    }
    if(tick%2===0){
      const resonance=[3.0,3.2,3.449,3.544,3.5699,3.63,3.738,3.8284,3.845,3.905,3.96];
      const targetR=resonance[Math.floor((tick/2)%resonance.length)];
      const targetPx=Math.round(((targetR-R0)/DR)*W);
      if(targetPx>=0&&targetPx<W) column(targetPx,120,0.45+0.35*Math.sin(tick*0.18));
    }
  }

  // Resizing used to clear to the ground colour and redraw, which flashed.
  // Stretch the previous frame into the new size first so the plate deforms
  // continuously, then refine it column by column underneath.
  function reset(carry){
    const dpr=Math.min(devicePixelRatio||1,2);
    const nw=Math.round(cv.clientWidth*dpr),nh=Math.round(cv.clientHeight*dpr);
    if(!nw||!nh||(nw===W&&nh===H&&carry)) return;
    let prev=null;
    if(carry&&W&&H){
      prev=document.createElement("canvas");prev.width=W;prev.height=H;
      prev.getContext("2d").drawImage(cv,0,0);
    }
    W=nw;H=nh;cv.width=W;cv.height=H;
    if(mcv){mcv.width=W;mcv.height=H;}
    lam=new Float32Array(W);
    ctx.fillStyle=css("--ground");ctx.fillRect(0,0,W,H);
    if(mctx) mctx.clearRect(0,0,W,H);
    if(prev){ctx.globalAlpha=0.85;ctx.drawImage(prev,0,0,W,H);ctx.globalAlpha=1;}
    col=0;sweeping=true;shimmerOn=false;starsOn=!still;
    seedOffset=Math.floor(Math.random()*10000);
    seedStars();
    if(raf){cancelAnimationFrame(raf);raf=null;}
    schedule();
  }

  // pause when off-screen; an attractor nobody can see should not burn a core
  new IntersectionObserver(es=>{
    for(const e of es){
      if(e.isIntersecting){
        if(col>=W&&!still){shimmerOn=true;starsOn=true;schedule();}
      } else {
        shimmerOn=false;starsOn=false;
      }
    }
  },{threshold:0.01}).observe(cv);

  // slow parallax: the plate drifts against the type as the reader leaves
  if(!reduced){
    addEventListener("scroll",()=>{
      const y=Math.min(scrollY,innerHeight);
      const ty="translate3d(0,"+(y*0.16).toFixed(1)+"px,0)";
      cv.style.transform=ty;
      if(mcv) mcv.style.transform=ty;
    },{passive:true});
  }

  let rrq=false;
  addEventListener("resize",()=>{
    if(rrq) return;
    rrq=true;
    requestAnimationFrame(()=>{rrq=false;reset(true);});
  });
  requestAnimationFrame(()=>reset());
  return ()=>reset(false);
}

/* ---------------- canvas plotting ---------------- */
// Tick text is formatted against the axis *span*, not the value: an axis
// running 1.4..2.5 wants one decimal, not the four that "1.4000" wasted.
function fmtSpan(v,span){
  const a=Math.abs(v);
  if((a!==0&&(a<1e-3||a>=1e5))||(span&&span<1e-3))return v.toExponential(2);
  const d=span?Math.max(0,Math.min(6,1-Math.floor(Math.log10(span/4)))):3;
  const s=v.toFixed(d);
  return s==="-0"?"0":s;
}
function fmt(v){return fmtSpan(v,Math.abs(v)||1);}
function ticks(lo,hi,n){
  const raw=(hi-lo||1)/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),nm=raw/mag;
  const st=(nm<1.5?1:nm<3?2:nm<7?5:10)*mag,out=[];
  for(let t=Math.ceil(lo/st)*st;t<=hi+1e-9;t+=st) out.push(t);
  return out;}
const PAL=["--chaotic","--locked","--torus"];
const MOUNTED=[];
// Shared by extent()/draw()/renderReadout(): a decimated export can carry a
// non-finite sample as JSON null (see export_figure_data.py's _round_to()),
// and a bare numeric compare against null silently coerces it to 0 instead
// of skipping it.
const finite=v=>typeof v==="number"&&Number.isFinite(v);
// Plot titles get their small-caps look here instead of CSS text-transform:
// uppercasing ASCII a-z leaves rho/eps/alpha alone.
function titleCaps(s){return s.replace(/[a-z]/g,c=>c.toUpperCase());}

const LEGEND_ROWH=14;

function Plot(canvas,panel,meta){
  const ctx=canvas.getContext("2d");
  const live=meta.live||null;
  const multi=panel.traces&&panel.traces.length>1;
  const pad={l:58,r:12,t:multi?24:10,b:34};
  const wrap=canvas.parentElement;
  const tip=document.createElement("div");tip.className="readout";wrap.appendChild(tip);
  // Only shown once the domain has moved away from its base extent -- the
  // one always-visible touch affordance back to the full view (no dblclick
  // on touch, and the keyboard "0"/Escape reset is invisible to a tap reader).
  const resetBtn=document.createElement("button");
  resetBtn.type="button";resetBtn.className="plot-reset";resetBtn.hidden=true;
  resetBtn.textContent="reset view";
  wrap.appendChild(resetBtn);
  let W=0,H=0,dom=null,base=null,anchor=null,drag=null,kx=null,dpr=1,legendLayout=null;
  // Set by afterDomain() -- every real view change funnels through it -- so
  // the next arrow key reads the snapped position instead of stepping off it.
  let viewChanged=false;
  let liveColorCache=new Map(),liveColorGen=-1;
  const heat=meta.kind==="heatmap";
  // Offscreen cache for the heatmap raster: the per-cell fillRect loop below
  // is the ~150-900ms cost a drag used to re-pay on every pointermove. It is
  // keyed on everything that can change what the raster looks like (visible
  // sub-domain, pixel size, and -- only for the "signed" cmap, which reads
  // theme custom properties -- the current theme), so a zoom/reset/resize or
  // a light/dark toggle invalidates it, while a same-domain pointermove
  // during a drag hits the cache and just blits.
  let heatCanvas=null,heatCtx=null,heatCacheKey=null;
  // Every place the visible domain actually changes funnels through here, so
  // the URL-state hook (meta.onDomainChange, wired in mountInteractive) fires
  // exactly once per real change instead of being sprinkled at each call site.
  function afterDomain(){
    viewChanged=true;
    if(meta.onDomainChange)meta.onDomainChange();
    if(live&&live.onView)live.onView(dom);
  }
  function resetDom(){kx=null;dom={...base};anchor={...base};draw();afterDomain();}
  function applyDom(d){dom=d;draw();afterDomain();}
  resetBtn.addEventListener("click",resetDom);
  function syncResetBtn(){
    resetBtn.hidden=!dom||!base||(dom.x0===base.x0&&dom.x1===base.x1&&dom.y0===base.y0&&dom.y1===base.y1);
  }

  function extent(){
    if(live&&live.base) return {...live.base};
    if(heat) return {x0:Math.min(...panel.x),x1:Math.max(...panel.x),y0:Math.min(...panel.y),y1:Math.max(...panel.y)};
    let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
    for(const s of panel.traces) for(let i=0;i<s.x.length;i++){
      // A NaN/Inf sample round-trips through JSON as null; skip it rather
      // than let a bare compare coerce it to 0 and drag the extent there.
      if(!finite(s.x[i])||!finite(s.y[i])) continue;
      if(s.x[i]<x0)x0=s.x[i]; if(s.x[i]>x1)x1=s.x[i];
      const e=s.yerr&&finite(s.yerr[i])?s.yerr[i]:0;
      const lo=s.y[i]-e, hi=s.y[i]+e;
      if(lo<y0)y0=lo; if(hi>y1)y1=hi;
    }
    const p=(y1-y0)*0.08||0.5;
    y0-=p;y1+=p;
    // Reference lines widen the axis, exactly as matplotlib's axvline/axhline
    // do -- otherwise a threshold the caption depends on falls off the edge.
    for(const m of (meta.marks||[])){
      if(m.axis==="x"){x0=Math.min(x0,m.value);x1=Math.max(x1,m.value);}
      else{y0=Math.min(y0,m.value);y1=Math.max(y1,m.value);}
    }
    return {x0,x1,y0,y1};
  }
  const sx=v=>pad.l+(v-dom.x0)/(dom.x1-dom.x0)*(W-pad.l-pad.r);
  const sy=v=>H-pad.b-(v-dom.y0)/(dom.y1-dom.y0)*(H-pad.t-pad.b);
  const ux=p=>dom.x0+(p-pad.l)/(W-pad.l-pad.r)*(dom.x1-dom.x0);

  let zmin=0,zmax=1;
  if(heat&&!live){zmin=Infinity;zmax=-Infinity;
    for(const row of panel.z) for(const v of row){
      if(v===null||Number.isNaN(v))continue;
      if(v<zmin)zmin=v;if(v>zmax)zmax=v;}}
  const hex=c=>[parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)];
  // Same viridis the matplotlib figures use, so toggling to the interactive
  // view never changes what a colour means. Magma is the space-time
  // figures' cmap (CMAP_SPACETIME in utils/style.py), sampled at the same
  // eleven stops.
  const VIRIDIS=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
                 [31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];
  const MAGMA=[[0,0,4],[20,14,54],[59,15,112],[100,26,128],[140,41,129],
               [183,55,121],[222,73,104],[247,112,92],[254,159,109],
               [254,207,146],[252,253,191]];
  const STOPS=(panel.cmap==="magma")?MAGMA:VIRIDIS;
  const signed=(panel.cmap||"viridis")==="signed"&&zmin<0&&zmax>0;
  function lerp3(a,b,f){return [0,1,2].map(j=>Math.round(a[j]+(b[j]-a[j])*f));}
  function rampRGB(t){
    t=Math.max(0,Math.min(1,t));
    if(signed){
      const stops=[hex(css("--locked")),hex(css("--torus")),hex(css("--chaotic"))];
      const k=t*2,i=Math.min(1,Math.floor(k));
      return lerp3(stops[i],stops[i+1],k-i);
    }
    const k=t*(STOPS.length-1),i=Math.min(STOPS.length-2,Math.floor(k));
    return lerp3(STOPS[i],STOPS[i+1],k-i);
  }
  // The colour limits: the panel's own extent for a static heatmap, or the
  // live figure's current window limits (the CML field's 1st/99th
  // percentiles, recomputed each publish). draw() refreshes zlo/zhi so a
  // rolling window's colours track the field it shows.
  let zlo=zmin,zhi=zmax;
  function syncZlim(){
    if(live&&typeof live.zlim==="function"){
      const z=live.zlim();
      if(z&&Number.isFinite(z[0])&&Number.isFinite(z[1])){zlo=z[0];zhi=z[1];}
    }
  }
  // Two-slope norm, as matplotlib's TwoSlopeNorm: z = 0 sits at the ramp's
  // midpoint (the locked/chaotic boundary) while each side still spans its half
  // of the ramp, so a lopsided field does not collapse into one hue.
  const znorm=v=>signed
    ? (v<0 ? 0.5*(1-v/zlo) : 0.5+0.5*(v/zhi))
    : (v-zlo)/((zhi-zlo)||1);
  function ramp(t){const c=rampRGB(t);return `rgb(${c[0]},${c[1]},${c[2]})`;}

  const CBAR=13;   // colourbar strip width
  // Reserve exactly the room the tick text needs, so the rotated axis label
  // stops landing on top of it.
  // Wraps the legend into as many rows as the plot width needs (measured
  // against the font actually used to draw it), instead of letting entries
  // run off the right edge past ~4-5 traces.
  function layoutLegend(){
    if(heat||!multi) return null;
    ctx.font="10px "+css("--mono");
    const avail=Math.max(40,W-pad.l-pad.r-4);
    const items=panel.traces.map((s,k)=>({
      name:s.name||"",color:s.color||PAL[k%PAL.length],
      w:17+ctx.measureText(s.name||"").width+8}));
    const rows=[[]];
    let lx=0;
    for(const it of items){
      if(lx+it.w>avail&&rows[rows.length-1].length){rows.push([]);lx=0;}
      rows[rows.length-1].push(it);lx+=it.w;
    }
    return rows;
  }
  // Reserve exactly the room the tick text needs, so the rotated axis label
  // stops landing on top of it.
  function measurePads(){
    ctx.font="10px "+css("--mono");
    let w=0;
    const yspan=dom.y1-dom.y0;
    for(const t of ticks(dom.y0,dom.y1,4)) w=Math.max(w,ctx.measureText(fmtSpan(t,yspan)).width);
    pad.l=Math.ceil(w)+(meta.ylabel?26:12);
    pad.r=12;
    if(heat){
      let zw=0;
      for(const t of ticks(zlo,zhi,4)) zw=Math.max(zw,ctx.measureText(fmtSpan(t,zhi-zlo)).width);
      pad.r=12+CBAR+8+Math.ceil(zw)+(zlabel?16:4);
    }
    // Legend rows need pad.l/pad.r settled first (they bound the available
    // width); pad.t is then sized to whatever the legend actually needs
    // instead of a fixed guess, so traces are never drawn under it.
    legendLayout=layoutLegend();
    pad.t=legendLayout?10+legendLayout.length*LEGEND_ROWH+4:10;
  }
  const zlabel=panel.zlabel||"";

  function drawColorbar(){
    const x=W-pad.r+12,y0=pad.t,y1=H-pad.b,h=y1-y0;
    if(h<20) return;
    // Painted linearly in value, not in ramp position, so a two-slope norm
    // still reads against evenly spaced ticks.
    for(let p=0;p<h;p++){
      ctx.fillStyle=ramp(znorm(zlo+(1-p/(h-1))*(zhi-zlo)));
      ctx.fillRect(x,y0+p,CBAR,1);
    }
    ctx.strokeStyle=css("--rule");ctx.lineWidth=1;
    ctx.strokeRect(x+0.5,y0+0.5,CBAR-1,h-1);
    ctx.fillStyle=css("--ink-low");ctx.textAlign="left";ctx.textBaseline="middle";
    const zspan=zhi-zlo;
    for(const t of ticks(zlo,zhi,4)){
      if(t<zlo-1e-12||t>zhi+1e-12)continue;
      const y=y1-(t-zlo)/(zspan||1)*h;
      ctx.fillText(fmtSpan(t,zspan),x+CBAR+5,y);
    }
    if(zlabel){
      ctx.save();ctx.fillStyle=css("--ink-mid");
      ctx.translate(W-3,y0+h/2);ctx.rotate(Math.PI/2);
      ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(zlabel,0,0);ctx.restore();
    }
  }

  // Cache key for the heatmap raster: the visible sub-domain, the pixel size
  // (CSS px and device-pixel ratio), and -- only relevant to the "signed"
  // cmap, which reads --locked/--torus/--chaotic -- the resolved theme
  // colours. Anything else (marks, legend, drag overlay) is drawn fresh every
  // frame on top and does not need to be in this key.
  function heatDomainKey(){
    const themeBit=signed?css("--locked")+"|"+css("--torus")+"|"+css("--chaotic"):"v";
    return [dom.x0,dom.x1,dom.y0,dom.y1,W,H,dpr,themeBit].join(",");
  }
  function renderHeatRaster(){
    if(!heatCanvas){heatCanvas=document.createElement("canvas");heatCtx=heatCanvas.getContext("2d");}
    const pw=Math.max(1,Math.round(W*dpr)),ph=Math.max(1,Math.round(H*dpr));
    if(heatCanvas.width!==pw||heatCanvas.height!==ph){heatCanvas.width=pw;heatCanvas.height=ph;}
    heatCtx.setTransform(dpr,0,0,dpr,0,0);
    heatCtx.clearRect(0,0,W,H);
    const nx=panel.x.length,ny=panel.y.length;
    // Fill to the next sample's edge rather than a fixed +1 fudge, so cells
    // tile exactly instead of smearing over one another.
    const edgeX=i=>i<nx-1?(sx(panel.x[i])+sx(panel.x[i+1]))/2:sx(panel.x[i])+(sx(panel.x[i])-sx(panel.x[i-1]))/2;
    const edgeY=j=>j<ny-1?(sy(panel.y[j])+sy(panel.y[j+1]))/2:sy(panel.y[j])+(sy(panel.y[j])-sy(panel.y[j-1]))/2;
    for(let j=0;j<ny;j++){
      const yb=j>0?edgeY(j-1):sy(panel.y[0])-(edgeY(0)-sy(panel.y[0]));
      const ya=edgeY(j);
      const yTop=Math.min(ya,yb),yh=Math.abs(ya-yb)+1;
      if(yTop>H-pad.b||yTop+yh<pad.t)continue;
      const row=panel.z[j];
      for(let i=0;i<nx;i++){
        const v=row[i];if(v===null||Number.isNaN(v))continue;
        const xb=i>0?edgeX(i-1):sx(panel.x[0])-(edgeX(0)-sx(panel.x[0]));
        const xa=edgeX(i);
        const xL=Math.min(xa,xb),xw=Math.abs(xa-xb)+1;
        if(xL>W-pad.r||xL+xw<pad.l)continue;
        heatCtx.fillStyle=ramp(znorm(v));heatCtx.fillRect(xL,yTop,xw,yh);
      }
    }
    heatCacheKey=heatDomainKey();
  }

  function renderLiveHeat(){
    const tiles=live.tiles||[];
    const gen=live.generation?live.generation():0;
    const plot={x0:dom.x0,x1:dom.x1,y0:dom.y0,y1:dom.y1,left:pad.l,top:pad.t,width:W-pad.l-pad.r,height:H-pad.t-pad.b};
    const rectOf=live.tilePixelRect;
    if(!rectOf) return;
    if(liveColorGen!==gen){liveColorCache.clear();liveColorGen=gen;}
    const keyOf=live.tileColorKey;
    const ordered=tiles.slice().filter(t=>t&&t.generation===gen).sort((a,b)=>a.level-b.level);
    const prevSmooth=ctx.imageSmoothingEnabled;
    ctx.imageSmoothingEnabled=false;
    for(const t of ordered){
      const nOmega=Math.max(1,Math.floor(Number(t.header&&t.header[0])));
      const nK=Math.max(1,Math.floor(Number(t.header&&t.header[1])));
      const data=t.data;
      if(!data||data.length<4+nOmega*nK) continue;
      const rect=rectOf(t,plot);
      if(!(rect.w>0&&rect.h>0)) continue;
      const key=keyOf?keyOf(t):t.generation+"|"+t.id;
      let sprite=liveColorCache.get(key);
      if(!sprite){
        sprite=document.createElement("canvas");
        sprite.width=nOmega;sprite.height=nK;
        const sctx=sprite.getContext("2d");
        const img=sctx.createImageData(nOmega,nK);
        const pix=img.data;
        for(let iy=0;iy<nK;iy++){
          const dstRow=nK-1-iy;
          for(let ix=0;ix<nOmega;ix++){
            const v=data[4+iy*nOmega+ix];
            const p=(dstRow*nOmega+ix)*4;
            if(v===null||Number.isNaN(v)){pix[p+3]=0;continue;}
            const c=rampRGB(znorm(v));
            pix[p]=c[0];pix[p+1]=c[1];pix[p+2]=c[2];pix[p+3]=255;
          }
        }
        sctx.putImageData(img,0,0);
        liveColorCache.set(key,sprite);
      }
      ctx.drawImage(sprite,rect.x,rect.y,rect.w,rect.h);
    }
    ctx.imageSmoothingEnabled=prevSmooth;
  }

  // Reference lines the captions promise (critical lines, bifurcation
  // thresholds). Values come from the payload, never recomputed here.
  function drawMarks(){
    const marks=meta.marks||[];
    if(!marks.length) return;
    ctx.save();ctx.setLineDash([5,4]);ctx.lineWidth=1;
    ctx.font="10px "+css("--mono");ctx.textBaseline="bottom";
    for(const m of marks){
      const vertical=m.axis==="x";
      const p=vertical?sx(m.value):sy(m.value);
      if(vertical?(p<pad.l||p>W-pad.r):(p<pad.t||p>H-pad.b))continue;
      ctx.strokeStyle=css("--ink-mid");ctx.globalAlpha=.75;
      ctx.beginPath();
      if(vertical){ctx.moveTo(p+0.5,pad.t);ctx.lineTo(p+0.5,H-pad.b);}
      else{ctx.moveTo(pad.l,p+0.5);ctx.lineTo(W-pad.r,p+0.5);}
      ctx.stroke();
      if(m.label){
        ctx.globalAlpha=1;ctx.fillStyle=css("--ink-mid");
        if(vertical){ctx.textAlign="left";ctx.fillText(m.label,p+4,pad.t+11);}
        else{ctx.textAlign="right";ctx.fillText(m.label,W-pad.r-4,p-3);}
      }
    }
    ctx.restore();
  }

  function draw(){
    if(!W||!H) return;
    syncZlim();
    measurePads();
    ctx.clearRect(0,0,W,H);
    const rule=css("--rule"),low=css("--ink-low"),mid=css("--ink-mid");
    ctx.font="10px "+css("--mono");ctx.strokeStyle=rule;ctx.lineWidth=1;ctx.fillStyle=low;
    ctx.textAlign="right";ctx.textBaseline="middle";
    const yspan=dom.y1-dom.y0,xspan=dom.x1-dom.x0;
    for(const t of ticks(dom.y0,dom.y1,4)){
      const y=Math.round(sy(t))+0.5;if(y<pad.t-1||y>H-pad.b+1)continue;
      if(!heat){ctx.globalAlpha=.4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.globalAlpha=1;}
      ctx.fillText(fmtSpan(t,yspan),pad.l-7,y);
    }
    ctx.textAlign="center";ctx.textBaseline="top";
    for(const t of ticks(dom.x0,dom.x1,5)){
      const x=Math.round(sx(t))+0.5;if(x<pad.l-1||x>W-pad.r+1)continue;
      if(!heat){ctx.globalAlpha=.4;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,H-pad.b);ctx.stroke();ctx.globalAlpha=1;}
      ctx.fillText(fmtSpan(t,xspan),x,H-pad.b+7);
    }
    ctx.fillStyle=mid;ctx.textAlign="center";ctx.textBaseline="bottom";
    // meta.labelFont swaps the axis labels' mono for the paper's math
    // italic (the delayed-logistic panels label their axes x and y).
    ctx.save();
    if(meta.labelFont) ctx.font=meta.labelFont;
    ctx.fillText(meta.xlabel||"",pad.l+(W-pad.l-pad.r)/2,H-2);
    ctx.translate(11,pad.t+(H-pad.t-pad.b)/2);ctx.rotate(-Math.PI/2);
    ctx.textBaseline="top";ctx.fillText(meta.ylabel||"",0,0);
    ctx.restore();

    ctx.save();ctx.beginPath();ctx.rect(pad.l,pad.t,W-pad.l-pad.r,H-pad.t-pad.b);ctx.clip();
    if(heat){
      if(live) renderLiveHeat();
      else{
        if(heatCacheKey!==heatDomainKey()) renderHeatRaster();
        ctx.drawImage(heatCanvas,0,0,W,H);
      }
    }else{
      panel.traces.forEach((s,k)=>{
        const c=css(s.color||PAL[k%PAL.length]);
        if(s.yerr){
          ctx.fillStyle=c;ctx.globalAlpha=.15;ctx.beginPath();
          for(let i=0;i<s.x.length;i++) ctx.lineTo(sx(s.x[i]),sy(s.y[i]+(finite(s.yerr[i])?s.yerr[i]:0)));
          for(let i=s.x.length-1;i>=0;i--) ctx.lineTo(sx(s.x[i]),sy(s.y[i]-(finite(s.yerr[i])?s.yerr[i]:0)));
          ctx.closePath();ctx.fill();ctx.globalAlpha=1;
        }
        if(meta.kind==="scatter"){
          ctx.fillStyle=c;ctx.globalAlpha=.72;
          for(let i=0;i<s.x.length;i++){
            if(!finite(s.x[i])||!finite(s.y[i]))continue;
            ctx.fillRect(sx(s.x[i])-1,sy(s.y[i])-1,2,2);
          }
          ctx.globalAlpha=1;
        }else{
          // A non-finite sample breaks the polyline (moveTo on the next
          // finite point) instead of drawing straight through the gap --
          // otherwise sx()/sy() of a null coerces to (0-domain) and a stray
          // segment shoots across the whole plot.
          ctx.strokeStyle=c;ctx.lineWidth=1.5;ctx.lineJoin="round";
          // A dashed trace (the double staircase's rho_theta = D
          // reference) carries its dash pattern; every other trace resets
          // to solid so the pattern never leaks between traces.
          ctx.setLineDash(s.dash||[]);ctx.beginPath();
          let open=false;
          for(let i=0;i<s.x.length;i++){
            if(!finite(s.x[i])||!finite(s.y[i])){open=false;continue;}
            const px=sx(s.x[i]),py=sy(s.y[i]);
            if(open) ctx.lineTo(px,py); else {ctx.moveTo(px,py);open=true;}
          }
          ctx.stroke();
        }
      });
    }
    drawMarks();
    if(drag&&drag.cur!==null){
      // Heatmaps rubber-band a rectangle (x and y both narrow); line/scatter
      // plots keep the vertical band, since their y-domain is refit from the
      // visible data on release rather than dragged directly.
      ctx.fillStyle=css("--chaotic");ctx.globalAlpha=.12;
      const a=Math.min(drag.start,drag.cur),b=Math.max(drag.start,drag.cur);
      if(heat&&drag.py!==null){
        const c=Math.min(drag.py0,drag.py),d=Math.max(drag.py0,drag.py);
        ctx.fillRect(a,c,b-a,d-c);
      }else{
        ctx.fillRect(a,pad.t,b-a,H-pad.t-pad.b);
      }
      ctx.globalAlpha=1;
    }
    ctx.restore();
    ctx.strokeStyle=css("--rule");ctx.lineWidth=1;
    ctx.strokeRect(pad.l+0.5,pad.t+0.5,W-pad.l-pad.r-1,H-pad.t-pad.b-1);
    if(heat) drawColorbar();
    if(!heat&&multi&&legendLayout){
      ctx.font="10px "+css("--mono");ctx.textAlign="left";ctx.textBaseline="middle";
      legendLayout.forEach((row,ri)=>{
        let lx=pad.l+4;
        const cy=10+ri*LEGEND_ROWH+LEGEND_ROWH/2-2;
        row.forEach(it=>{
          ctx.fillStyle=css(it.color);ctx.fillRect(lx,cy-1,12,2);
          ctx.fillStyle=mid;ctx.fillText(it.name,lx+17,cy);
          lx+=it.w;
        });
      });
    }
    syncResetBtn();
  }

  function resize(){
    dpr=Math.min(devicePixelRatio||1,2);
    const cw=canvas.clientWidth;if(!cw) return;
    const chh=Math.round(Math.min(340,Math.max(200,cw*(heat?0.5:0.42))));
    canvas.style.height=chh+"px";
    canvas.width=cw*dpr;canvas.height=chh*dpr;
    ctx.setTransform(dpr,0,0,dpr,0,0);W=cw;H=chh;draw();
  }

  // Shared by pointer hover and keyboard-panned reads, so both paths report
  // the same values the same way.
  const SCATTER_MAXR=18; // px cutoff beyond which no point is "under" the pointer
  function renderReadout(px,py){
    if(px<pad.l||px>W-pad.r||py<pad.t||py>H-pad.b){tip.classList.remove("on");return;}
    const xv=ux(px);const lines=[];
    if(heat){
      const yv=dom.y0+(H-pad.b-py)/(H-pad.t-pad.b)*(dom.y1-dom.y0);
      if(live){
        lines.push((meta.xlabel||"x")+" = "+fmt(xv));
        lines.push((meta.ylabel||"y")+" = "+fmt(yv));
        const rho=live.sample?live.sample(xv,yv):null;
        lines.push((zlabel||"value")+" = "+(rho===null?"…":fmt(rho)));
      }else{
        const near=(arr,v)=>{let b=0,d=Infinity;
          for(let i=0;i<arr.length;i++){const q=Math.abs(arr[i]-v);if(q<d){d=q;b=i;}}return b;};
        const i=near(panel.x,xv),j=near(panel.y,yv);
        lines.push((meta.xlabel||"x")+" = "+fmt(panel.x[i]));
        lines.push((meta.ylabel||"y")+" = "+fmt(panel.y[j]));
        lines.push((zlabel||"value")+" = "+fmt(panel.z[j][i]));
      }
    }else if(meta.kind==="scatter"){
      // A scatter cloud has no natural x-ordering, so "nearest in x" can
      // report a y value from a completely different part of the cloud.
      // Nearest in screen-space distance (with a cutoff) is what the cursor
      // actually looks like it is pointing at.
      let bestD2=Infinity,bestS=null,bestI=-1;
      for(const s of panel.traces) for(let i=0;i<s.x.length;i++){
        if(!finite(s.x[i])||!finite(s.y[i]))continue;
        const dx=sx(s.x[i])-px,dy=sy(s.y[i])-py,d2=dx*dx+dy*dy;
        if(d2<bestD2){bestD2=d2;bestS=s;bestI=i;}
      }
      if(!bestS||Math.sqrt(bestD2)>SCATTER_MAXR){tip.classList.remove("on");return;}
      lines.push((meta.xlabel||"x")+" = "+fmt(bestS.x[bestI]));
      lines.push((bestS.name||meta.ylabel||"y")+" = "+fmt(bestS.y[bestI]));
    }else{
      lines.push((meta.xlabel||"x")+" = "+fmt(xv));
      if(live&&live.sample){
        // The 1-D live figure quotes the point kernel at (D, x), the same
        // value its e2e test recomputes; the tile-derived curve is only the
        // fallback while the module loads.
        const rho=live.sample(xv);
        lines.push((meta.ylabel||"y")+" = "+(rho===null?"…":fmt(rho)));
      }else{
        for(const s of panel.traces){
          let b=0,d=Infinity;
          for(let i=0;i<s.x.length;i++){
            if(!finite(s.x[i])||!finite(s.y[i]))continue;
            const q=Math.abs(s.x[i]-xv);if(q<d){d=q;b=i;}
          }
          lines.push((s.name||"y")+" = "+(d===Infinity?"n/a":fmt(s.y[b])));
        }
      }
    }
    tip.textContent=lines.join("\n");tip.classList.add("on");
    const tw=tip.offsetWidth,th=tip.offsetHeight;
    tip.style.left=Math.min(px+12,W-tw-4)+"px";
    tip.style.top=Math.max(4,Math.min(py-th-8,H-th-4))+"px";
  }

  // uy() is renderReadout's heat y-pixel-to-value inverse, factored out so
  // the drag rectangle can convert its y edges the same way.
  function uy(py){return dom.y0+(H-pad.b-py)/(H-pad.t-pad.b)*(dom.y1-dom.y0);}
  const pointers=new Map();
  let pan=null,pinch=null;
  function clampDom(d){
    let {x0,x1,y0,y1}=d;
    let xs=x1-x0,ys=y1-y0;
    const bx=base.x1-base.x0,by=base.y1-base.y0;
    if(xs<1e-9) xs=1e-9;
    if(ys<1e-9) ys=1e-9;
    if(xs>bx){x0=base.x0;x1=base.x1;} else {
      if(x0<base.x0){x0=base.x0;x1=x0+xs;}
      if(x1>base.x1){x1=base.x1;x0=x1-xs;}
    }
    if(ys>by){y0=base.y0;y1=base.y1;} else {
      if(y0<base.y0){y0=base.y0;y1=y0+ys;}
      if(y1>base.y1){y1=base.y1;y0=y1-ys;}
    }
    return {x0,x1,y0,y1};
  }
  function zoomAbout(px,py,factor){
    return applyZoom(ux(px),uy(py),px,py,factor,dom);
  }
  // Place world (xv,yv) at screen (px,py) with the span of `from` times factor.
  // Do not write `from` into `dom` first. A refused zoom must leave the view.
  function applyZoom(xv,yv,px,py,factor,from){
    let xs=(from.x1-from.x0)*factor,ys=(from.y1-from.y0)*factor;
    xs=Math.max(1e-9,Math.min(base.x1-base.x0,xs));
    ys=Math.max(1e-9,Math.min(base.y1-base.y0,ys));
    const plotW=W-pad.l-pad.r,plotH=H-pad.t-pad.b;
    const fx=(px-pad.l)/plotW,fy=(H-pad.b-py)/plotH;
    let next={x0:xv-fx*xs,x1:xv-fx*xs+xs,y0:yv-fy*ys,y1:yv-fy*ys+ys};
    next=clampDom(next);
    if(next.x0===dom.x0&&next.x1===dom.x1&&next.y0===dom.y0&&next.y1===dom.y1) return false;
    applyDom(next);
    return true;
  }
  function pointerXY(e){
    const r=canvas.getBoundingClientRect();
    return {x:e.clientX-r.left,y:e.clientY-r.top};
  }
  function pinchNow(){
    const pts=[...pointers.values()];
    if(pts.length<2) return null;
    const dx=pts[1].x-pts[0].x,dy=pts[1].y-pts[0].y;
    return {cx:(pts[0].x+pts[1].x)/2,cy:(pts[0].y+pts[1].y)/2,dist:Math.hypot(dx,dy)||1};
  }
  canvas.addEventListener("pointermove",e=>{
    const r=canvas.getBoundingClientRect(),px=e.clientX-r.left,py=e.clientY-r.top;
    if(live&&pointers.has(e.pointerId)){
      pointers.set(e.pointerId,{x:px,y:py});
      if(pointers.size>=2&&pinch){
        const now=pinchNow();
        if(now){
          applyZoom(pinch.xv,pinch.yv,now.cx,now.cy,pinch.dist/now.dist,pinch.dom);
        }
        return;
      }
      if(pan&&pointers.size===1){
        const plotW=W-pad.l-pad.r,plotH=H-pad.t-pad.b;
        const xspan=pan.dom.x1-pan.dom.x0,yspan=pan.dom.y1-pan.dom.y0;
        const dx=-(px-pan.x)/plotW*xspan;
        const dy=(py-pan.y)/plotH*yspan;
        applyDom(clampDom({x0:pan.dom.x0+dx,x1:pan.dom.x1+dx,y0:pan.dom.y0+dy,y1:pan.dom.y1+dy}));
        return;
      }
    }
    if(drag){
      drag.cur=Math.max(pad.l,Math.min(W-pad.r,px));
      if(heat) drag.py=Math.max(pad.t,Math.min(H-pad.b,py));
      draw();return;
    }
    renderReadout(px,py);
  });
  canvas.addEventListener("pointerleave",()=>tip.classList.remove("on"));
  canvas.addEventListener("pointerdown",e=>{
    const r=canvas.getBoundingClientRect(),px=e.clientX-r.left,py=e.clientY-r.top;
    if(px<pad.l||px>W-pad.r)return;
    if(live&&e.pointerType==="touch"){
      pointers.set(e.pointerId,{x:px,y:py});
      canvas.setPointerCapture(e.pointerId);tip.classList.remove("on");
      if(pointers.size>=2){
        const now=pinchNow();
        if(now) pinch={...now,dom:{...dom},xv:ux(now.cx),yv:uy(now.cy)};
        pan=null;drag=null;
      }else{
        pan={x:px,y:py,dom:{...dom}};
      }
      return;
    }
    // Heatmaps rubber-band a full rectangle, so the drag needs a y anchor
    // too; line/scatter plots only ever read drag.start/drag.cur (x).
    drag={start:px,cur:null,py0:py,py:null};canvas.setPointerCapture(e.pointerId);tip.classList.remove("on");
  });
  canvas.addEventListener("pointerup",e=>{
    if(live&&pointers.has(e.pointerId)){
      pointers.delete(e.pointerId);
      if(pointers.size>=2){
        const now=pinchNow();
        if(now) pinch={...now,dom:{...dom},xv:ux(now.cx),yv:uy(now.cy)};
      }else if(pointers.size===1){
        pinch=null;
        const remaining=[...pointers.values()][0];
        pan={x:remaining.x,y:remaining.y,dom:{...dom}};
      }else{
        pinch=null;pan=null;
      }
      return;
    }
    let zoomed=false;
    if(drag&&drag.cur!==null&&Math.abs(drag.cur-drag.start)>10){
      const a=ux(Math.min(drag.start,drag.cur)),b=ux(Math.max(drag.start,drag.cur));
      let next=null;
      if(heat){
        let y0=dom.y0,y1=dom.y1;
        if(drag.py!==null){
          const yA=uy(Math.min(drag.py0,drag.py)),yB=uy(Math.max(drag.py0,drag.py));
          y0=Math.min(yA,yB);y1=Math.max(yA,yB);
        }
        if(b-a>=1e-9&&y1-y0>=1e-9) next={...dom,x0:a,x1:b,y0,y1};
      }else{
        const ys=[];
        for(const s of panel.traces) for(let i=0;i<s.x.length;i++)
          if(s.x[i]>=a&&s.x[i]<=b&&finite(s.y[i])) ys.push(s.y[i]);
        if(ys.length>1){const y0=Math.min(...ys),y1=Math.max(...ys),p=(y1-y0)*0.08||0.5;
          next={x0:a,x1:b,y0:y0-p,y1:y1+p};}
      }
      if(next){dom=next;zoomed=true;}
    }else if(drag){
      // No drag happened -- on a coarse pointer this is the only way a tap
      // can ever produce the value readout, since there is no hover to fall
      // back on. Reuses the exact same pointermove readout path.
      const r=canvas.getBoundingClientRect();
      renderReadout(e.clientX-r.left,e.clientY-r.top);
    }
    drag=null;draw();
    if(zoomed) afterDomain();
  });
  canvas.addEventListener("pointercancel",e=>{
    if(!(live&&pointers.has(e.pointerId))) {pointers.delete(e.pointerId);return;}
    pointers.delete(e.pointerId);
    if(pointers.size>=2){
      const now=pinchNow();
      if(now) pinch={...now,dom:{...dom},xv:ux(now.cx),yv:uy(now.cy)};
    }else if(pointers.size===1){
      pinch=null;
      const remaining=[...pointers.values()][0];
      pan={x:remaining.x,y:remaining.y,dom:{...dom}};
    }else{
      pinch=null;pan=null;
    }
  });
  canvas.addEventListener("dblclick",resetDom);
  canvas.addEventListener("wheel",e=>{
    if(!live) return;
    e.preventDefault();
    const r=canvas.getBoundingClientRect(),px=e.clientX-r.left,py=e.clientY-r.top;
    if(px<pad.l||px>W-pad.r||py<pad.t||py>H-pad.b) return;
    zoomAbout(px,py,e.deltaY<0?0.8:1/0.8);
  },{passive:false});

  // Keyboard path: arrow keys pan a stepped x-index cursor through the same
  // readout the pointer uses, +/- (or Up/Down) zoom around it, 0 or Escape
  // resets -- the same domain reset as dblclick.
  function xValues(){
    if(live){
      if(!dom) return [];
      const n=32;
      const xs=new Array(n);
      const span=dom.x1-dom.x0;
      for(let i=0;i<n;i++) xs[i]=dom.x0+(i/(n-1))*span;
      return xs;
    }
    return heat?panel.x:(panel.traces[0]?panel.traces[0].x:[]);
  }
  function nearestIndex(xs,v){
    let b=0,d=Infinity;
    for(let i=0;i<xs.length;i++){const q=Math.abs(xs[i]-v);if(q<d){d=q;b=i;}}
    return b;
  }
  function zoomAt(factor){
    const center=kx===null?(dom.x0+dom.x1)/2:kx;
    const half=(dom.x1-dom.x0)*factor/2;
    const a=Math.max(base.x0,center-half),b=Math.min(base.x1,center+half);
    if(b-a<1e-9) return;
    if(heat){
      // No keyboard cursor exists on the y axis, so zoom it the same factor
      // around its own current midpoint -- keeps keyboard zoom a true 2-D
      // zoom, consistent with the drag-rectangle behaviour above.
      const ycenter=(dom.y0+dom.y1)/2,yhalf=(dom.y1-dom.y0)*factor/2;
      const c=Math.max(base.y0,ycenter-yhalf),d=Math.min(base.y1,ycenter+yhalf);
      if(d-c<1e-9) return;
      dom={...dom,x0:a,x1:b,y0:c,y1:d};
    }
    else{
      const ys=[];
      for(const s of panel.traces) for(let i=0;i<s.x.length;i++)
        if(s.x[i]>=a&&s.x[i]<=b) ys.push(s.y[i]);
      if(ys.length>1){const y0=Math.min(...ys),y1=Math.max(...ys),p=(y1-y0)*0.08||0.5;
        dom={x0:a,x1:b,y0:y0-p,y1:y1+p};}
      else dom={...dom,x0:a,x1:b};
    }
    draw();
    afterDomain();
    // A shown tip was laid out against the old pixel mapping; refresh it in
    // place so its K and rho are not stale until the next arrow key.
    if(kx!==null&&tip.classList.contains("on"))
      renderReadout(sx(kx),(pad.t+H-pad.b)/2);
  }
  canvas.addEventListener("keydown",e=>{
    const xs=xValues();
    if(e.key==="ArrowLeft"||e.key==="ArrowRight"){
      if(!xs.length) return;
      e.preventDefault();
      if(kx===null) kx=xs[Math.floor(xs.length/2)];
      let i=nearestIndex(xs,kx);
      // The first press after a zoom, pan, reset or setDomain reads the
      // snapped position; only later presses step one index.
      if(!viewChanged) i=Math.max(0,Math.min(xs.length-1,i+(e.key==="ArrowRight"?1:-1)));
      viewChanged=false;
      kx=xs[i];
      renderReadout(sx(kx),(pad.t+H-pad.b)/2);
    }else if(e.key==="+"||e.key==="="||e.key==="ArrowUp"){
      e.preventDefault();zoomAt(0.6);
    }else if(e.key==="-"||e.key==="_"||e.key==="ArrowDown"){
      e.preventDefault();zoomAt(1/0.6);
    }else if(e.key==="0"||e.key==="Escape"){
      e.preventDefault();resetDom();tip.classList.remove("on");
    }
  });

  base=extent();dom={...base};anchor={...base};
  const ro=new ResizeObserver(resize);ro.observe(canvas);resize();
  // destroy() lets the toggle-off path in mountInteractive fully retire this
  // instance: without disconnecting the observer here, an unmounted canvas
  // stays watched forever and MOUNTED would otherwise only ever grow.
  return {redraw:draw,resize,destroy:()=>{ro.disconnect();},
    // URL-state hooks (see the IIFE near the end of the file): read-only
    // access to the current/published domain, and a way in for a domain
    // that came from a shared link rather than a live drag or keypress.
    getDomain:()=>({...dom}),
    getBase:()=>({...base}),
    // setDomain's silent option is for the automatic staircase y-fit: the
    // view moves but no reader change happened, so onDomainChange does not
    // fire (no modified badge, no ?fig= write) and live.onView is skipped
    // for the same reason afterDomain skips it. The anchor folds the
    // machine-moved keys into the unmodified reference so isModified()
    // still sees the fit as stock while a reader change elsewhere stays
    // flagged.
    setDomain:(d,opts)=>{const n={...base,...d};
      // Same floor zoomAt applies: a degenerate span would make sx()/sy()
      // divide by zero and the readout show a non-finite value.
      n.x1=n.x0+Math.max(1e-9,n.x1-n.x0);
      n.y1=n.y0+Math.max(1e-9,n.y1-n.y0);
      if(opts&&opts.silent){
        for(const k of ["x0","x1","y0","y1"]) if(n[k]!==dom[k]) anchor[k]=n[k];
        dom=n;viewChanged=true;draw();
      }else{
        dom=n;draw();afterDomain();
      }},
    // rebase recomputes the base extent from live.base and resets the view
    // onto it: the CML figure's site-count control changes the field's
    // width, so the published domain itself moves, not just the view.
    rebase:()=>{base=extent();dom={...base};anchor={...base};draw();afterDomain();},
    isModified:()=>!!dom&&!!anchor&&(dom.x0!==anchor.x0||dom.x1!==anchor.x1||dom.y0!==anchor.y0||dom.y1!==anchor.y1),
    reset:resetDom};
}

/* ---------------- figures: static by default, interaction on demand ---------------- */
function unmountPlot(fig){
  (fig._plots||[]).forEach(p=>{
    p.destroy();
    const i=MOUNTED.indexOf(p);
    if(i>=0) MOUNTED.splice(i,1);
  });
  fig._plots=[];
  if(fig._live&&fig._live.destroy){try{fig._live.destroy();}catch(e){}}
  if(fig._pool){fig._pool.destroy();fig._pool=null;}
  if(fig._debugTracker){try{fig._debugTracker.destroy();}catch(e){}fig._debugTracker=null;}
  fig._live=null;
  const body=fig.querySelector(".fig-body");
  if(body) body.querySelectorAll(".plot-wrap,.hint,.live-debug-report").forEach(n=>n.remove());
}

function liveModuleUrl(name){
  return new URL("live/"+name, document.baseURI).href;
}

async function mountLive(fig, opts){
  const body=fig.querySelector(".fig-body");
  const [poolMod, raster, point, liveFigure, paramsMod]=await Promise.all([
    import(liveModuleUrl("pool.js")),
    import(liveModuleUrl("raster.js")),
    import(liveModuleUrl("point.js")),
    import(liveModuleUrl("live-figure.js")),
    import(liveModuleUrl("params.js")),
  ]);
  const interactT0=(opts&&Number.isFinite(opts.interactT0))?opts.interactT0:performance.now();
  const debugTracker=liveFigure.createDebugTracker({
    fig,
    interactT0,
    debugEnabled:poolMod.debugEnabled()
  });
  fig._debugTracker=debugTracker;
  const fc=fig.querySelector("figcaption");
  let capText=fc?fc.textContent.trim():"";
  capText=capText.replace(/^Figure\s*\d+\.\s*/i,"");
  if(fig.dataset.live==="devils_staircase"){
    await mountStaircase(fig,body,{poolMod,raster,point,liveFigure,paramsMod,capText,debugTracker});
    return;
  }
  if(fig.dataset.live==="delayed_logistic_attractors"){
    await mountAttractors(fig,body,{liveFigure,paramsMod,capText,debugTracker});
    return;
  }
  if(fig.dataset.live==="torus_doubling_attractors"){
    await mountTorus(fig,body,{liveFigure,paramsMod,capText,debugTracker});
    return;
  }
  if(fig.dataset.live==="double_staircase"){
    await mountDoubleStaircase(fig,body,{liveFigure,paramsMod,capText,debugTracker});
    return;
  }
  if(fig.dataset.live==="spacetime_diagrams"){
    await mountSpacetime(fig,body,{liveFigure,paramsMod,raster,capText,debugTracker});
    return;
  }
  const title="Rotation number over the circle-map parameter plane";
  // Five levels keep the finest tile small. A zoom then waits on less in-flight work.
  const liveLevels=5;
  try{
    body.querySelector("img").style.display="none";
    const w=document.createElement("div");w.className="plot-wrap";
    const c=document.createElement("canvas");c.className="plot live";
    c.setAttribute("role","img");
    c.setAttribute("tabindex","0");
    c.setAttribute("aria-label",capText+". "+title);
    w.appendChild(c);body.appendChild(w);
    const h=document.createElement("p");h.className="plot-title";
    h.textContent=titleCaps(title);w.insertBefore(h,c);
    const store=liveFigure.createStore();
    const params=liveFigure.LIVE_ARNOLD;
    let pool=null,plot=null;
    point.ensureLoaded().catch(()=>{});
    const sampleReadout=liveFigure.createReadout({store,point,raster,params});
    const live={
      base:{x0:0,x1:1,y0:0,y1:0.3},
      tiles:store.tiles,
      generation:()=>store.generation,
      tilePixelRect:raster.tilePixelRect,
      tileColorKey:raster.liveTileColorKey,
      sample:sampleReadout,
      onView:liveFigure.createViewHandler({store,getPool:()=>pool,getPlot:()=>plot})
    };
    const panel={
      title,
      zlabel:"Rotation number \u03c1",
      cmap:"viridis",
      x:[0,1],y:[0,0.3],z:[[0]]
    };
    plot=Plot(c,panel,{
      kind:"heatmap",
      live,
      marks:[{axis:"y",value:1/(2*Math.PI),label:"K_c = 1/2\u03c0"}],
      xlabel:"Bare frequency \u03a9",
      ylabel:"Nonlinearity K",
      onDomainChange:()=>{if(window.figState)window.figState.notify(fig);}
    });
    MOUNTED.push(plot);fig._plots=[plot];
    const dpr=Math.min(devicePixelRatio||1,2);
    const tileCells=raster.tileCellsFor((c.clientWidth||1)*dpr,(c.clientHeight||1)*dpr,{
      levels:liveLevels,nIter:params.nIter,nTransient:params.nTransient
    });
    const hint=document.createElement("p");hint.className="hint";
    hint.textContent="tap to read values · drag to zoom · scroll or pinch to zoom · one finger to pan · reset view button to restore · focus the plot and use +/- to zoom, 0 or Esc to reset · computed live in this browser";
    body.appendChild(hint);
    pool=poolMod.createPool({
      workerUrl:new URL("live/tile-worker.js", document.baseURI),
      scheduler:liveFigure.createSchedulerOptions({levels:liveLevels,tileCells,params}),
      onCapacityLost(){
        hint.textContent="live figure incomplete: every worker failed. Reload the page to retry";
      },
      onPaint:liveFigure.createPaintHandler({
        store,
        raster,
        getPool:()=>pool,
        getPlot:()=>plot,
        onFirstPaintComplete:({tileTimes})=>{
          if(debugTracker) debugTracker.onFirstPaintComplete({workerCount:pool.workerCount,tileTimes});
        }
      })
    });
    fig._pool=pool;
    const d0=plot.getDomain();
    pool.setViewport({omegaMin:d0.x0,omegaMax:d0.x1,kMin:d0.y0,kMax:d0.y1});
    store.generation=pool.getState().generation;
    fig._live={
      sample:sampleReadout,
      rasterSample:liveFigure.createRasterSample({store,raster}),
      pointReady:()=>point.isReady(),
      setView(v){
        plot.setDomain({x0:v.omegaMin,x1:v.omegaMax,y0:v.kMin,y1:v.kMax});
      },
      stats:()=>({
        generation:store.generation,
        painted:store.painted,
        workerCount:pool.workerCount,
        liveWorkers:pool.liveWorkers,
        nIter:store.nIter
      })
    };
    fig.dataset.state="live";
  }catch(err){
    // Unmount the live figure if this setup throws. Then the caller reports the error.
    unmountPlot(fig);
    const img=body.querySelector("img");
    if(img) img.style.display="";
    throw err;
  }
}

/* One row of range + number inputs per parameter spec, shared by the live
   figures that carry sliders. Pure DOM: the decisions (clamp, debounce,
   hash) live in live-figure.js's parameter wiring. */
function buildParamControls(specs){
  const ctrls=document.createElement("div");ctrls.className="live-params";
  const inputs={};
  for(const spec of specs){
    const lab=document.createElement("label");lab.className="live-param";
    const name=document.createElement("span");name.className="live-param-name";
    name.textContent=spec.name;
    if(spec.options){
      // A discrete parameter (the torus figure's map selector) is a select,
      // not a slider pair; the wiring still sees a number.
      const sel=document.createElement("select");
      sel.setAttribute("aria-label",spec.label);
      for(const opt of spec.options){
        const o=document.createElement("option");
        o.value=String(opt.value);o.textContent=opt.label;
        sel.appendChild(o);
      }
      sel.value=String(spec.default);
      lab.appendChild(name);lab.appendChild(sel);
      ctrls.appendChild(lab);
      inputs[spec.name]={select:sel};
      continue;
    }
    if(spec.numberOnly){
      // A range the reader types rather than slides (the double
      // staircase's D window endpoints): a number field alone, no slider.
      const num=document.createElement("input");
      num.type="number";num.min=spec.min;num.max=spec.max;num.step=spec.step;
      num.value=spec.default;num.setAttribute("aria-label",spec.label);
      lab.appendChild(name);lab.appendChild(num);
      ctrls.appendChild(lab);
      inputs[spec.name]={num};
      continue;
    }
    const range=document.createElement("input");
    range.type="range";range.min=spec.min;range.max=spec.max;range.step=spec.step;
    range.value=spec.default;range.setAttribute("aria-label",spec.label);
    const num=document.createElement("input");
    num.type="number";num.min=spec.min;num.max=spec.max;num.step=spec.step;
    num.value=spec.default;num.setAttribute("aria-label",spec.label+" (number)");
    lab.appendChild(name);lab.appendChild(range);lab.appendChild(num);
    ctrls.appendChild(lab);
    inputs[spec.name]={range,num};
  }
  return {ctrls,inputs};
}

/* The devil's staircase: rho(A) live at a reader-chosen D, above the PNG.
   The PNG's lower panel (the Lyapunov exponent) has no wasm kernel yet, so
   the image stays and a note says which part is live. This function only
   builds DOM and hands collaborators to live-figure.js; the decisions
   (viewport mapping, trace, debounce, hash) live there for node tests. */
async function mountStaircase(fig,body,{poolMod,raster,point,liveFigure,paramsMod,capText,debugTracker}){
  const cfg=liveFigure.LIVE_STAIRCASE;
  const title="Rotation number \u03c1(A) at drive frequency D";
  try{
    const w=document.createElement("div");w.className="plot-wrap";
    const c=document.createElement("canvas");c.className="plot live";
    c.setAttribute("role","img");
    c.setAttribute("tabindex","0");
    c.setAttribute("aria-label",capText+". "+title);
    const h=document.createElement("p");h.className="plot-title";
    h.textContent=titleCaps(title);w.appendChild(h);
    w.appendChild(c);
    const {ctrls,inputs}=buildParamControls(cfg.paramSpecs);
    w.appendChild(ctrls);
    const hint=document.createElement("p");hint.className="hint";
    hint.textContent="tap to read values · drag to zoom · scroll or pinch to zoom · one finger to pan · reset view button to restore · focus the plot and use +/- to zoom, 0 or Esc to reset · computed live in this browser";
    w.appendChild(hint);
    body.insertBefore(w,body.firstChild);
    const note=document.createElement("p");note.className="hint";
    note.textContent="the image below is the published figure; its lower panel (Lyapunov exponent) is not live; only the \u03c1(A) curve above is computed in your browser";
    body.insertBefore(note,w.nextSibling);
    const store=liveFigure.createStore(cfg);
    let pool=null,plot=null;
    point.ensureLoaded().catch(()=>{});
    const trace=store.trace;
    const panel={title,traces:[{name:"\u03c1",x:trace.x,y:trace.y,color:"--ink"}]};
    const live={
      base:{x0:0,x1:0.25,y0:0,y1:1},
      tiles:store.tiles,
      generation:()=>store.generation,
      onView:null
    };
    plot=Plot(c,panel,{
      kind:"line",
      live,
      marks:[{axis:"x",value:1/(2*Math.PI),label:"K_c = 1/2\u03c0"}],
      xlabel:"Nonlinearity A",
      ylabel:"Rotation number \u03c1",
      onDomainChange:()=>{if(window.figState)window.figState.notify(fig);}
    });
    MOUNTED.push(plot);fig._plots=[plot];
    const dpr=Math.min(devicePixelRatio||1,2);
    const tileCells=raster.tileCellsFor((c.clientWidth||1)*dpr,(c.clientHeight||1)*dpr,{
      levels:5,nIter:cfg.nIter,nTransient:cfg.nTransient
    });
    const view=liveFigure.createStaircaseView({
      store,
      getPool:()=>pool,
      getPlot:()=>plot,
      getD:()=>wiring.state.D
    });
    live.onView=view.onView;
    const fitY=liveFigure.createLineYFit({store,getPlot:()=>plot});
    const writeHash=()=>{
      const frag=paramsMod.writeParamsIntoHash(location.hash,fig.id,wiring.state);
      history.replaceState(null,"",location.pathname+location.search+(frag?"#"+frag:""));
    };
    const wiring=liveFigure.createParamWiring({
      specs:cfg.paramSpecs,
      debounceMs:150,
      echo:(name,v)=>{const f=inputs[name];if(f){f.range.value=v;f.num.value=v;}},
      apply:view.applyD,
      writeHash
    });
    for(const spec of cfg.paramSpecs){
      const f=inputs[spec.name];
      f.range.addEventListener("input",()=>wiring.onInput(spec.name,Number(f.range.value)));
      f.num.addEventListener("change",()=>wiring.onInput(spec.name,Number(f.num.value)));
    }
    pool=poolMod.createPool({
      workerUrl:new URL("live/tile-worker.js", document.baseURI),
      scheduler:liveFigure.createSchedulerOptions({levels:5,tileCells,params:cfg,lockOmega:true}),
      onCapacityLost(){
        hint.textContent="live figure incomplete: every worker failed. Reload the page to retry";
      },
      onPaint:liveFigure.createPaintHandler({store,raster,getPool:()=>pool,getPlot:()=>plot,onStore:(s)=>{
        // Fold the new tile into the polyline in place: panel.traces[0]
        // holds these two arrays, so replacing them would orphan the plot.
        const t=liveFigure.tilesToTrace(s.tiles,s.generation);
        s.trace.x.splice(0,s.trace.x.length,...t.x);
        s.trace.y.splice(0,s.trace.y.length,...t.y);
        fitY();
      },
      onFirstPaintComplete:({tileTimes})=>{
        if(debugTracker) debugTracker.onFirstPaintComplete({workerCount:pool.workerCount,tileTimes});
      }})
    });
    fig._pool=pool;
    const d0=plot.getDomain();
    pool.setViewport({omegaMin:wiring.state.D,omegaMax:wiring.state.D,kMin:d0.x0,kMax:d0.x1});
    store.generation=pool.getState().generation;
    const readout=liveFigure.createStaircaseReadout({store,point,raster,getD:()=>wiring.state.D});
    live.sample=readout;
    fig._live={
      sample:readout,
      rasterSample:(a)=>raster.sampleAt(store.tiles,store.generation,wiring.state.D,a),
      pointReady:()=>point.isReady(),
      setParams:(values)=>{wiring.setParams(values);},
      trace:()=>({x:trace.x.slice(),y:trace.y.slice()}),
      setView(v){
        const cur=plot.getDomain();
        plot.setDomain({x0:v.x0!==undefined?v.x0:v.kMin,x1:v.x1!==undefined?v.x1:v.kMax,y0:cur.y0,y1:cur.y1});
      },
      stats:()=>({
        generation:store.generation,
        painted:store.painted,
        points:trace.x.length,
        workerCount:pool.workerCount,
        liveWorkers:pool.liveWorkers,
        nIter:store.nIter
      })
    };
    fig.dataset.state="live";
  }catch(err){
    unmountPlot(fig);
    const img=body.querySelector("img");
    if(img) img.style.display="";
    throw err;
  }
}

/* The delayed-logistic attractors: the (x, y) point cloud live at a
   reader-chosen D, above the published twelve-panel PNG. One direct kernel
   call per parameter set — no tile pool: a burst of slider moves is
   collapsed by the debounce in live-figure.js and stale results drop by
   sequence number. This function only builds DOM; the request shape, the
   trace fold and the hash live in live-figure.js for node tests. */
async function mountAttractors(fig,body,{liveFigure,paramsMod,capText,debugTracker}){
  const cfg=liveFigure.LIVE_ATTRACTORS;
  const title="Delayed logistic map, \u03b1 = 0.3";
  try{
    const w=document.createElement("div");w.className="plot-wrap";
    const c=document.createElement("canvas");c.className="plot live";
    c.setAttribute("role","img");
    c.setAttribute("tabindex","0");
    c.setAttribute("aria-label",capText+". "+title);
    const h=document.createElement("p");h.className="plot-title";
    h.textContent=titleCaps(title);w.appendChild(h);
    w.appendChild(c);
    const {ctrls,inputs}=buildParamControls(cfg.paramSpecs);
    w.appendChild(ctrls);
    const hint=document.createElement("p");hint.className="hint";
    hint.textContent="tap to read values · drag to zoom · scroll or pinch to zoom · one finger to pan · reset view button to restore · focus the plot and use +/- to zoom, 0 or Esc to reset · computed live in this browser";
    w.appendChild(hint);
    body.insertBefore(w,body.firstChild);
    const note=document.createElement("p");note.className="hint";
    note.textContent="the image below is the published figure; twelve panels at fixed D values; the cloud above is computed in your browser at the D you choose";
    body.insertBefore(note,w.nextSibling);
    let plot=null;
    const writeHash=()=>{
      const frag=paramsMod.writeParamsIntoHash(location.hash,fig.id,af.state);
      history.replaceState(null,"",location.pathname+location.search+(frag?"#"+frag:""));
    };
    let firstTraceDone=false;
    const af=liveFigure.createAttractorFigure({
      params:cfg,
      debounceMs:150,
      echo:(name,v)=>{
        const f=inputs[name];if(f){f.range.value=v;f.num.value=v;}
        if(name==="D") h.textContent=titleCaps(title+", D = "+v);
      },
      onTrace:(store,trace,req,computeMs)=>{
        // panel.traces holds these arrays; replacing them would orphan the
        // plot. The second trace is the analytic fixed point the orbit
        // starts beside — the figure's one vermilion mark.
        const fp=liveFigure.attractorFixedPoint(req.dMin);
        fpTrace.x.splice(0,fpTrace.x.length,fp);
        fpTrace.y.splice(0,fpTrace.y.length,fp);
        if(plot) plot.redraw();
        if(!firstTraceDone){
          firstTraceDone=true;
          if(debugTracker) debugTracker.onFirstPaintComplete({workerCount:0,tileTimes:Number.isFinite(computeMs)?[computeMs]:[]});
        }
      },
      onError:()=>{
        hint.textContent="live figure failed: the wasm kernel did not answer. Reload the page to retry";
      },
      writeHash
    });
    const fpTrace={x:[],y:[]};
    const panel={title,traces:[
      {name:"orbit",x:af.store.trace.x,y:af.store.trace.y,color:"--slate"},
      {name:"fixed point",x:fpTrace.x,y:fpTrace.y,color:"--vermilion"}
    ]};
    const live={
      base:{...cfg.domain},
      generation:()=>af.store.generation
    };
    plot=Plot(c,panel,{
      kind:"scatter",
      live,
      labelFont:"italic 11px 'TeX Gyre Pagella',Georgia,serif",
      xlabel:"x",
      ylabel:"y",
      onDomainChange:()=>{if(window.figState)window.figState.notify(fig);}
    });
    MOUNTED.push(plot);fig._plots=[plot];
    for(const spec of cfg.paramSpecs){
      const f=inputs[spec.name];
      f.range.addEventListener("input",()=>af.onInput(spec.name,Number(f.range.value)));
      f.num.addEventListener("change",()=>af.onInput(spec.name,Number(f.num.value)));
    }
    af.refresh();
    fig._live={
      setParams:(values)=>{af.setParams(values);},
      params:()=>({...af.state}),
      trace:()=>({x:af.store.trace.x.slice(),y:af.store.trace.y.slice()}),
      ready:()=>af.ready(),
      stats:()=>({
        generation:af.store.generation,
        painted:af.store.painted,
        points:af.store.trace.x.length
      })
      ,destroy:()=>af.destroy()
    };
    fig.dataset.state="live";
  }catch(err){
    unmountPlot(fig);
    const img=body.querySelector("img");
    if(img) img.style.display="";
    throw err;
  }
}

/* The torus-doubling attractors: the (X, Y) projection of map (I) or map
   (IV) live at a reader-chosen D, above the published three-panel PNG. One
   direct kernel call per parameter set — no tile pool, same as the
   delayed-logistic figure. The map selector is a <select>; switching maps
   moves the D slider's range and default to the chosen map's doubling
   window. This function only builds DOM; the request shape, the trace
   fold, the map switch and the hash live in live-figure.js for node
   tests. */
async function mountTorus(fig,body,{liveFigure,paramsMod,capText,debugTracker}){
  const cfg=liveFigure.LIVE_TORUS;
  try{
    const w=document.createElement("div");w.className="plot-wrap";
    const c=document.createElement("canvas");c.className="plot live";
    c.setAttribute("role","img");
    c.setAttribute("tabindex","0");
    const h=document.createElement("p");h.className="plot-title";
    w.appendChild(h);
    w.appendChild(c);
    const {ctrls,inputs}=buildParamControls(cfg.paramSpecs);
    w.appendChild(ctrls);
    const hint=document.createElement("p");hint.className="hint";
    hint.textContent="tap to read values · drag to zoom · scroll or pinch to zoom · one finger to pan · reset view button to restore · focus the plot and use +/- to zoom, 0 or Esc to reset · computed live in this browser";
    w.appendChild(hint);
    body.insertBefore(w,body.firstChild);
    const note=document.createElement("p");note.className="hint";
    note.textContent="the image below is the published figure; three panels at fixed D values; the cloud above is computed in your browser at the map and D you choose";
    body.insertBefore(note,w.nextSibling);
    let plot=null;
    const setTitle=()=>{h.textContent=titleCaps(cfg.maps[liveFigure.torusMapKind(tf.state.map)].title+", D = "+tf.state.D);};
    const setDBounds=(m)=>{
      const f=inputs.D;
      f.range.min=m.dMin;f.range.max=m.dMax;f.range.step=m.dStep;
      f.num.min=m.dMin;f.num.max=m.dMax;f.num.step=m.dStep;
    };
    const applyMap=(m)=>{
      inputs.map.select.value=String(m.kind);
      setDBounds(m);
      c.setAttribute("aria-label",capText+". "+m.title);
    };
    const writeHash=()=>{
      const frag=paramsMod.writeParamsIntoHash(location.hash,fig.id,tf.state);
      history.replaceState(null,"",location.pathname+location.search+(frag?"#"+frag:""));
    };
    let firstTraceDone=false;
    const tf=liveFigure.createTorusFigure({
      params:cfg,
      debounceMs:150,
      echo:(name,v)=>{
        const f=inputs[name];
        if(name==="map"){applyMap(cfg.maps[v]);setTitle();return;}
        if(f){f.range.value=v;f.num.value=v;}
        if(name==="D") setTitle();
      },
      onTrace:(store,trace,req,computeMs)=>{
        // panel.traces holds these arrays; replacing them would orphan the
        // plot. The second trace is the map's unstable fixed point — the
        // figure's one vermilion mark, as on the delayed-logistic figure.
        const fp=liveFigure.attractorFixedPoint(req.d);
        fpTrace.x.splice(0,fpTrace.x.length,fp);
        fpTrace.y.splice(0,fpTrace.y.length,fp);
        if(plot) plot.redraw();
        if(!firstTraceDone){
          firstTraceDone=true;
          if(debugTracker) debugTracker.onFirstPaintComplete({workerCount:0,tileTimes:Number.isFinite(computeMs)?[computeMs]:[]});
        }
      },
      onError:()=>{
        hint.textContent="live figure failed: the wasm kernel did not answer. Reload the page to retry";
      },
      writeHash
    });
    const fpTrace={x:[],y:[]};
    const panel={title:cfg.maps[1].title,traces:[
      {name:"orbit",x:tf.store.trace.x,y:tf.store.trace.y,color:"--slate"},
      {name:"fixed point",x:fpTrace.x,y:fpTrace.y,color:"--vermilion"}
    ]};
    const live={
      base:{...cfg.domain},
      generation:()=>tf.store.generation
    };
    plot=Plot(c,panel,{
      kind:"scatter",
      live,
      labelFont:"italic 11px 'TeX Gyre Pagella',Georgia,serif",
      xlabel:"X",
      ylabel:"Y",
      onDomainChange:()=>{if(window.figState)window.figState.notify(fig);}
    });
    MOUNTED.push(plot);fig._plots=[plot];
    applyMap(cfg.maps[liveFigure.torusMapKind(tf.state.map)]);
    setTitle();
    for(const spec of cfg.paramSpecs){
      const f=inputs[spec.name];
      if(f.select){f.select.addEventListener("change",()=>tf.onInput(spec.name,Number(f.select.value)));continue;}
      f.range.addEventListener("input",()=>tf.onInput(spec.name,Number(f.range.value)));
      f.num.addEventListener("change",()=>tf.onInput(spec.name,Number(f.num.value)));
    }
    tf.refresh();
    fig._live={
      setParams:(values)=>{tf.setParams(values);},
      params:()=>({...tf.state}),
      trace:()=>({x:tf.store.trace.x.slice(),y:tf.store.trace.y.slice()}),
      ready:()=>tf.ready(),
      stats:()=>({
        generation:tf.store.generation,
        painted:tf.store.painted,
        points:tf.store.trace.x.length
      })
      ,destroy:()=>tf.destroy()
    };
    fig.dataset.state="live";
  }catch(err){
    unmountPlot(fig);
    const img=body.querySelector("img");
    if(img) img.style.display="";
    throw err;
  }
}

/* The double devil's staircase: rho_theta and rho_phi against D for the
   modulated circle map, live above the published PNG. One direct
   modulated_circle_rotation_tile call per parameter set — no tile pool,
   same as the attractor figures. The D window is the compute range and
   the plot's x domain at once: the min/max fields, the two zoom presets
   and a drag zoom all move the same pair through live-figure.js's range
   view. This function only builds DOM; the request shape, the trace fold,
   the window sync and the hash live in live-figure.js for node tests. */
async function mountDoubleStaircase(fig,body,{liveFigure,paramsMod,capText,debugTracker}){
  const cfg=liveFigure.LIVE_MODULATED;
  const title="Double devil's staircase";
  try{
    const w=document.createElement("div");w.className="plot-wrap";
    const c=document.createElement("canvas");c.className="plot live";
    c.setAttribute("role","img");
    c.setAttribute("tabindex","0");
    c.setAttribute("aria-label",capText+". "+title);
    const h=document.createElement("p");h.className="plot-title";
    const setTitle=()=>{h.textContent=titleCaps(title+", \u03b5 = "+df.state.eps);};
    w.appendChild(h);
    w.appendChild(c);
    const {ctrls,inputs}=buildParamControls(cfg.paramSpecs);
    w.appendChild(ctrls);
    const hint=document.createElement("p");hint.className="hint";
    hint.textContent="tap to read values · drag to zoom · scroll or pinch to zoom · one finger to pan · reset view button to restore · focus the plot and use +/- to zoom, 0 or Esc to reset · computed live in this browser";
    w.appendChild(hint);
    body.insertBefore(w,body.firstChild);
    const note=document.createElement("p");note.className="hint";
    note.textContent="the image below is the published figure; the curves above are computed in your browser at the \u03b5 and D window you choose; the two preset buttons reproduce the zoom panels' windows";
    body.insertBefore(note,w.nextSibling);
    let plot=null;
    const writeHash=()=>{
      const frag=paramsMod.writeParamsIntoHash(location.hash,fig.id,df.state);
      history.replaceState(null,"",location.pathname+location.search+(frag?"#"+frag:""));
    };
    let firstTraceDone=false;
    const df=liveFigure.createModulatedFigure({
      params:cfg,
      debounceMs:150,
      echo:(name,v)=>{
        const f=inputs[name];
        if(f){if(f.range)f.range.value=v;if(f.num)f.num.value=v;}
        if(name==="eps") setTitle();
      },
      onTrace:(store,trace,req,computeMs)=>{
        if(plot) plot.redraw();
        if(!firstTraceDone){
          firstTraceDone=true;
          if(debugTracker) debugTracker.onFirstPaintComplete({workerCount:0,tileTimes:Number.isFinite(computeMs)?[computeMs]:[]});
        }
      },
      onError:()=>{
        hint.textContent="live figure failed: the wasm kernel did not answer. Reload the page to retry";
      },
      writeHash,
      getPlot:()=>plot
    });
    // The preset row: the two zoom-panel windows from the config plus a
    // reset to the full [0, 1] sweep. A click is a discrete choice, so it
    // applies through setParams — no debounce.
    const presets=document.createElement("div");presets.className="live-presets";
    for(const p of cfg.presets){
      const b=document.createElement("button");
      b.type="button";b.textContent=p.name;
      b.addEventListener("click",()=>df.setParams({dMin:p.dMin,dMax:p.dMax}));
      presets.appendChild(b);
    }
    const reset=document.createElement("button");
    reset.type="button";reset.textContent="full D range";
    reset.addEventListener("click",()=>df.setParams({dMin:0,dMax:1}));
    presets.appendChild(reset);
    ctrls.appendChild(presets);
    const trace=df.store.trace;
    const panel={title,traces:[
      {name:"\u03c1\u03b8",x:trace.x,y:trace.y,color:"--slate"},
      {name:"\u03c1\u03c6",x:trace.x,y:trace.y2,color:"--vermilion"},
      {name:"\u03c1\u03b8 = D",x:[0,1],y:[0,1],color:"--ink-low",dash:[5,4]}
    ]};
    const live={
      base:{x0:0,x1:1,y0:0,y1:1},
      generation:()=>df.store.generation,
      onView:df.view.onView
    };
    plot=Plot(c,panel,{
      kind:"line",
      live,
      xlabel:"Bare frequency D",
      ylabel:"Observed rotation \u03c1\u03b8",
      onDomainChange:()=>{if(window.figState)window.figState.notify(fig);}
    });
    MOUNTED.push(plot);fig._plots=[plot];
    setTitle();
    for(const spec of cfg.paramSpecs){
      const f=inputs[spec.name];
      if(f.range) f.range.addEventListener("input",()=>df.onInput(spec.name,Number(f.range.value)));
      if(f.num) f.num.addEventListener("change",()=>df.onInput(spec.name,Number(f.num.value)));
    }
    df.refresh();
    const readout=liveFigure.createModulatedReadout({store:df.store,getEps:()=>df.state.eps});
    live.sample=readout;
    fig._live={
      setParams:(values)=>{df.setParams(values);},
      params:()=>({...df.state}),
      trace:()=>({x:df.store.trace.x.slice(),y:df.store.trace.y.slice(),y2:df.store.trace.y2.slice()}),
      sample:readout,
      ready:()=>df.ready(),
      stats:()=>({
        generation:df.store.generation,
        painted:df.store.painted,
        points:df.store.trace.x.length
      })
      ,destroy:()=>df.destroy()
    };
    fig.dataset.state="live";
  }catch(err){
    unmountPlot(fig);
    const img=body.querySelector("img");
    if(img) img.style.display="";
    throw err;
  }
}

/* The CML space-time diagrams: the field x_i^n of the chosen model live
   above the published nine-panel PNG — site i across, time n up, magma
   colour at the shown field's 1st/99th percentiles, the same rule plot()
   uses per row. One direct cml_spacetime_tile call per parameter set —
   no tile pool, same as the attractor figures. The play button appends
   chunks computed from the window's last row and drops the oldest rows;
   the initial field comes from live/cml-x0.json (the paper's seed-42
   uniform x0). This function only builds DOM; the request shape, the
   field fold, the window trim, the model switch and the hash live in
   live-figure.js for node tests. */
async function mountSpacetime(fig,body,{liveFigure,paramsMod,raster,capText,debugTracker}){
  const cfg=liveFigure.LIVE_SPACETIME;
  const title="Coupled-map lattice space-time";
  try{
    const res=await fetch(new URL("live/cml-x0.json",document.baseURI).href);
    if(!res.ok) throw new Error("HTTP "+res.status);
    const x0=await res.json();
    const w=document.createElement("div");w.className="plot-wrap";
    const c=document.createElement("canvas");c.className="plot live";
    c.setAttribute("role","img");
    c.setAttribute("tabindex","0");
    c.setAttribute("aria-label",capText+". "+title);
    const h=document.createElement("p");h.className="plot-title";
    w.appendChild(h);
    w.appendChild(c);
    const {ctrls,inputs}=buildParamControls(cfg.paramSpecs);
    w.appendChild(ctrls);
    const transport=document.createElement("div");transport.className="live-presets";
    const playBtn=document.createElement("button");
    playBtn.type="button";playBtn.className="live-play";playBtn.textContent="play";
    transport.appendChild(playBtn);
    w.appendChild(transport);
    const hint=document.createElement("p");hint.className="hint";
    hint.textContent="tap to read values · drag to zoom · scroll or pinch to zoom · one finger to pan · reset view button to restore · focus the plot and use +/- to zoom, 0 or Esc to reset · play appends new rows and drops the oldest · computed live in this browser";
    w.appendChild(hint);
    body.insertBefore(w,body.firstChild);
    const note=document.createElement("p");note.className="hint";
    note.textContent="the image below is the published figure; nine panels at fixed \u03b5 values; the field above is computed in your browser at the model, \u03b5 and site count you choose";
    body.insertBefore(note,w.nextSibling);
    let plot=null;
    const setTitle=()=>{h.textContent=titleCaps(cfg.models[liveFigure.cmlModelKind(cf.state.model)].title+", \u03b5 = "+cf.state.eps);};
    const setEpsBounds=(m)=>{
      const f=inputs.eps;
      f.range.min=m.epsMin;f.range.max=m.epsMax;
      f.num.min=m.epsMin;f.num.max=m.epsMax;
    };
    const applyModel=(m)=>{
      inputs.model.select.value=String(m.kind);
      setEpsBounds(m);
      c.setAttribute("aria-label",capText+". "+m.title);
    };
    const writeHash=()=>{
      const frag=paramsMod.writeParamsIntoHash(location.hash,fig.id,cf.state);
      history.replaceState(null,"",location.pathname+location.search+(frag?"#"+frag:""));
    };
    let firstTraceDone=false;
    const cf=liveFigure.createSpacetimeFigure({
      params:cfg,
      x0,
      debounceMs:150,
      echo:(name,v)=>{
        const f=inputs[name];
        if(name==="model"){applyModel(cfg.models[v]);setTitle();return;}
        if(f){if(f.range)f.range.value=v;if(f.num)f.num.value=v;}
        if(name==="eps") setTitle();
      },
      onField:(store,field,req,computeMs)=>{
        // The site-count control changes the field's width: the published
        // domain itself moves, so the plot rebases instead of zooming.
        if(live.base.x1!==field.nSites||live.base.y1!==field.nRows){
          live.base={x0:0,x1:field.nSites,y0:0,y1:field.nRows};
          if(plot) plot.rebase();
        }
        if(plot) plot.redraw();
        if(!firstTraceDone){
          firstTraceDone=true;
          if(debugTracker) debugTracker.onFirstPaintComplete({workerCount:0,tileTimes:Number.isFinite(computeMs)?[computeMs]:[]});
        }
      },
      onError:()=>{
        hint.textContent="live figure failed: the wasm kernel did not answer. Reload the page to retry";
      },
      writeHash
    });
    const live={
      base:{x0:0,x1:cfg.paramSpecs[2].default,y0:0,y1:cfg.nRecord},
      tiles:cf.store.tiles,
      generation:()=>cf.store.generation,
      tilePixelRect:raster.tilePixelRect,
      tileColorKey:raster.liveTileColorKey,
      zlim:()=>cf.store.zlim,
      sample:liveFigure.createRasterSample({store:cf.store,raster})
    };
    const panel={title,cmap:"magma",zlabel:"x_i^n"};
    plot=Plot(c,panel,{
      kind:"heatmap",
      live,
      xlabel:"Site i",
      ylabel:"Time n",
      onDomainChange:()=>{if(window.figState)window.figState.notify(fig);}
    });
    MOUNTED.push(plot);fig._plots=[plot];
    applyModel(cfg.models[liveFigure.cmlModelKind(cf.state.model)]);
    setTitle();
    for(const spec of cfg.paramSpecs){
      const f=inputs[spec.name];
      if(f.select){f.select.addEventListener("change",()=>cf.onInput(spec.name,Number(f.select.value)));continue;}
      f.range.addEventListener("input",()=>cf.onInput(spec.name,Number(f.range.value)));
      f.num.addEventListener("change",()=>cf.onInput(spec.name,Number(f.num.value)));
    }
    playBtn.addEventListener("click",()=>{
      if(cf.playing()){cf.pause();playBtn.textContent="play";}
      else{cf.play();playBtn.textContent="pause";}
    });
    cf.refresh();
    fig._live={
      setParams:(values)=>{cf.setParams(values);},
      params:()=>({...cf.state}),
      field:()=>cf.store.field?{values:Array.from(cf.store.field.values),nSites:cf.store.field.nSites,nRows:cf.store.field.nRows}:null,
      sample:live.sample,
      play:()=>{cf.play();playBtn.textContent="pause";},
      pause:()=>{cf.pause();playBtn.textContent="play";},
      playing:()=>cf.playing(),
      ready:()=>cf.ready(),
      stats:()=>({
        generation:cf.store.generation,
        painted:cf.store.painted,
        rows:cf.store.field?cf.store.field.nRows:0,
        sites:cf.store.field?cf.store.field.nSites:0
      })
      ,destroy:()=>cf.destroy()
    };
    fig.dataset.state="live";
  }catch(err){
    unmountPlot(fig);
    const img=body.querySelector("img");
    if(img) img.style.display="";
    throw err;
  }
}

function applyPendingLive(fig){
  // A narrow viewport keeps the PNG, but the shared link's parameters must
  // still be here when the reader later presses interact. The restore paths
  // stash them; this applies the stash once a mount has actually happened.
  if(fig._pendingLive&&fig._live&&fig._live.setParams){
    fig._live.setParams(fig._pendingLive);
    fig._pendingLive=null;
  }
  if(fig._pendingDomain&&fig._plots){
    fig._pendingDomain.forEach((d,idx)=>{
      const p=fig._plots[idx];
      if(p) p.setDomain(d);
    });
    fig._pendingDomain=null;
    if(window.figState) window.figState.notify(fig);
  }
}

async function mountInteractive(fig, opts){
  const requested=!!(opts&&opts.requested);
  const interactT0=(opts&&Number.isFinite(opts.interactT0))?opts.interactT0:performance.now();
  const body=fig.querySelector(".fig-body");
  const src=fig.dataset.src;
  const btn=fig.querySelector(".act-interact");
  if(fig.dataset.state==="live"){                     // toggle back to the image
    fig.dataset.state="static";
    unmountPlot(fig);
    body.querySelector("img").style.display="";
    btn.textContent="interact";
    return;
  }
  const oldErr=fig.querySelector(".fetch-err");
  if(oldErr) oldErr.remove();
  btn.textContent="loading";btn.disabled=true;
  try{
    if(fig.dataset.live&&!reducedData){
      // A shared link may mount live mode on its own. Below LIVE_MIN_WIDTH
      // (a phone in portrait) keep the PNG until the reader presses interact.
      if(!requested){
        const liveFigure=await import(liveModuleUrl("live-figure.js"));
        if(!liveFigure.liveAutoStarts(innerWidth)){
          fig.dataset.liveHold="narrow";
          btn.textContent="interact";
          return;
        }
      }
      delete fig.dataset.liveHold;
      await mountLive(fig, { interactT0 });
      applyPendingLive(fig);
      btn.textContent="image";
      return;
    }
    if(!src){
      btn.textContent="interact";
      return;
    }
    const res=await fetch(src);
    if(!res.ok) throw new Error("HTTP "+res.status);
    const spec=await res.json();
    body.querySelector("img").style.display="none";
    // The figcaption is the figure's only alt text; the canvas replacing the
    // <img> must carry the same description forward, not drop it.
    const fc=fig.querySelector("figcaption");
    let capText=fc?fc.textContent.trim():"";
    capText=capText.replace(/^Figure\s*\d+\.\s*/i,"");
    fig._plots=[];
    spec.panels.forEach(panel=>{
      const w=document.createElement("div");w.className="plot-wrap";
      const c=document.createElement("canvas");c.className="plot";
      c.setAttribute("role","img");
      c.setAttribute("tabindex","0");
      c.setAttribute("aria-label",panel.title?capText+". "+panel.title:capText);
      w.appendChild(c);body.appendChild(w);
      if(spec.panels.length>1&&panel.title){
        const h=document.createElement("p");h.className="plot-title";
        h.textContent=titleCaps(panel.title);w.insertBefore(h,c);
      }
      const p=Plot(c,panel,{
        kind:spec.kind,
        marks:spec.marks||[],
        xlabel:(spec.axes&&spec.axes.x&&spec.axes.x.label)||"",
        ylabel:(spec.axes&&spec.axes.y&&spec.axes.y.label)||"",
        onDomainChange:()=>{if(window.figState)window.figState.notify(fig);}});
      MOUNTED.push(p);fig._plots.push(p);
    });
    const d=spec.decimation,n=document.createElement("p");n.className="hint";
    n.textContent="tap to read values · drag to zoom · reset view button to restore · focus the plot and use arrow keys to pan, +/- to zoom, 0 or Esc to reset"
      +(d&&d.method!=="none"?" · resampled to "+d.to.toLocaleString()+" of "+d.from.toLocaleString()+" grid points; the static image is at full resolution":"");
    body.appendChild(n);
    fig.dataset.state="live";applyPendingLive(fig);btn.textContent="image";
  }catch(err){
    console.warn("figure data fetch failed:",src,err);
    btn.textContent="retry";
    const msg=document.createElement("span");
    msg.className="fetch-err";
    msg.textContent="couldn't load the figure data. Tap retry";
    fig.querySelector(".fig-head").appendChild(msg);
  }finally{btn.disabled=false;}
}

document.querySelectorAll("figure[data-src],figure[data-live]").forEach(fig=>{
  const btn=fig.querySelector(".act-interact");
  if(btn) btn.addEventListener("click",()=>mountInteractive(fig,{requested:true,interactT0:performance.now()}));
});

/* ---------------- figure code snippets ---------------- */
document.querySelectorAll("figure").forEach(fig=>{
  const codeBtn=fig.querySelector(".act-code");
  const details=fig.querySelector("details.fig-code");
  if(codeBtn&&details) codeBtn.addEventListener("click",()=>{
    details.open=!details.open;
    codeBtn.textContent=details.open?"hide":"code";
  });
  const copyBtn=fig.querySelector(".fig-code-copy");
  if(copyBtn) copyBtn.addEventListener("click",()=>{
    const code=fig.querySelector(".fig-snippet");
    const text=code?code.textContent:"";
    const flash=()=>{
      copyBtn.classList.add("copied");
      clearTimeout(copyBtn._copiedTimer);
      copyBtn._copiedTimer=setTimeout(()=>copyBtn.classList.remove("copied"),1400);
    };
    if(navigator.clipboard&&navigator.clipboard.writeText){
      navigator.clipboard.writeText(text).then(flash,()=>{});
    }else{
      const ta=document.createElement("textarea");
      ta.value=text;ta.style.position="fixed";ta.style.opacity="0";
      document.body.appendChild(ta);ta.select();
      try{document.execCommand("copy");flash();}catch(e){/* clipboard unavailable */}
      ta.remove();
    }
  });
});

/* ---------------- interactive-figure state in the URL ----------------
   One search param, "fig", carries every figure that has drifted from its
   published domain: entries are "figid:panel:x0:x1:y0:y1" joined by commas
   (numbers at short precision; panel is the index within a multi-panel
   figure's own spec.panels, so a bare per-figure key never has to guess
   which chart moved). A figid can itself contain a colon (pandoc labels
   read "fig:name"), so decoding always takes the last five colon-fields as
   panel+domain and rejoins whatever is left as the id -- never the other
   way around.

   This never touches location.hash: the scroll-spy above (search for
   "history.replaceState" near the bottom of this file) rewrites only the
   hash from location.pathname+location.search, and everything here rewrites
   only location.search from location.pathname+location.hash. Each side
   reads the other's current value before replacing its own half, so a
   scroll and a zoom in the same moment cannot clobber each other. */
(function(){
  const PARAM="fig";

  function packNum(v){
    if(!isFinite(v)) return "0";
    return Number(v.toPrecision(5)).toString();
  }
  function parseParam(){
    const out=new Map();               // figid -> Map(panelIndex -> domain)
    const raw=new URLSearchParams(location.search).get(PARAM);
    if(!raw) return out;
    raw.split(",").forEach(tok=>{
      const parts=tok.split(":");
      if(parts.length<6) return;       // figid(>=1) + panel + 4 numbers
      const nums=parts.slice(-4).map(Number);
      const panel=parseInt(parts[parts.length-5],10);
      const figid=parts.slice(0,-5).join(":");
      if(!figid||Number.isNaN(panel)||nums.some(Number.isNaN)) return;
      if(!out.has(figid)) out.set(figid,new Map());
      out.get(figid).set(panel,{x0:nums[0],x1:nums[1],y0:nums[2],y1:nums[3]});
    });
    return out;
  }
  function serialize(){
    const entries=[];
    document.querySelectorAll("figure[data-src]").forEach(fig=>{
      (fig._plots||[]).forEach((p,i)=>{
        if(!p.isModified()) return;
        const d=p.getDomain();
        entries.push([fig.id,i,packNum(d.x0),packNum(d.x1),packNum(d.y0),packNum(d.y1)].join(":"));
      });
    });
    return entries.join(",");
  }
  function writeURL(){
    const usp=new URLSearchParams(location.search);
    const s=serialize();
    if(s) usp.set(PARAM,s); else usp.delete(PARAM);
    const qs=usp.toString();
    history.replaceState(null,"",location.pathname+(qs?"?"+qs:"")+location.hash);
  }
  function figModified(fig){return (fig._plots||[]).some(p=>p.isModified());}
  function syncBadge(fig){
    const b=fig.querySelector(".fig-badge");
    if(b) b.hidden=!figModified(fig);
  }

  document.querySelectorAll("figure[data-src]").forEach(fig=>{
    const badge=fig.querySelector(".fig-badge");
    if(badge) badge.addEventListener("click",()=>{
      (fig._plots||[]).forEach(p=>p.reset());
      syncBadge(fig);writeURL();
    });
    const copy=fig.querySelector(".act-copylink");
    if(copy) copy.addEventListener("click",()=>{
      const url=location.origin+location.pathname+location.search+location.hash;
      const flash=()=>{
        copy.classList.add("copied");
        clearTimeout(copy._copiedTimer);
        copy._copiedTimer=setTimeout(()=>copy.classList.remove("copied"),1400);
      };
      if(navigator.clipboard&&navigator.clipboard.writeText){
        navigator.clipboard.writeText(url).then(flash,()=>{});
      }else{
        const ta=document.createElement("textarea");
        ta.value=url;ta.style.position="fixed";ta.style.opacity="0";
        document.body.appendChild(ta);ta.select();
        try{document.execCommand("copy");flash();}catch(e){/* clipboard unavailable */}
        ta.remove();
      }
    });
  });

  // Exposed so mountInteractive's per-panel onDomainChange callback (wired at
  // Plot construction, above) can reach this module without a circular
  // top-level import -- there is no module system here, just script order.
  window.figState={notify:fig=>{syncBadge(fig);writeURL();}};

  // Restore from a shared link: mount whichever figures the URL names, the
  // same path the "interact" button uses, then apply the stored domain. A
  // figure that fails to fetch (mountInteractive already reports that error
  // itself) must not stop the rest of the URL from being honoured.
  parseParam().forEach((panels,figid)=>{
    const fig=document.getElementById(figid);
    if(!fig||!fig.dataset||!fig.dataset.src) return;
    fig._pendingDomain=panels;
    const mounted=fig.dataset.state==="live"?Promise.resolve():mountInteractive(fig);
    mounted.then(()=>{
      if(fig.dataset.state!=="live"||!fig._plots) return;   // fetch failed
      panels.forEach((d,idx)=>{
        const p=fig._plots[idx];
        if(p) p.setDomain(d);
      });
      syncBadge(fig);
    });
  });

  // Live-figure parameters ride the hash (#section&fig:<id>.<name>=<value>),
  // not the ?fig= search param: live figures carry no data-src, so the block
  // above never sees them. Read the hash captured at parse time — the
  // scroll-spy may already have rewritten it — then mount through the same
  // interact path and push the parsed values in without the debounce.
  (function(){
    const frag=location.hash;
    if(frag.indexOf("=")<0) return;
    document.querySelectorAll("figure[data-live]").forEach(fig=>{
      import(liveModuleUrl("params.js")).then(paramsMod=>
        import(liveModuleUrl("live-figure.js")).then(liveFigure=>{
          const cfg=liveFigure.LIVE_FIGURES[fig.dataset.live];
          if(!cfg||!cfg.paramSpecs) return;
          const values=paramsMod.parseHash(frag,cfg.paramSpecs,fig.id);
          if(!Object.keys(values).length) return;
          fig._pendingLive=values;
          const mounted=fig.dataset.state==="live"?Promise.resolve():mountInteractive(fig);
          mounted.then(()=>{
            if(fig._live&&fig._live.setParams) fig._live.setParams(values);
          });
        })
      ).catch(()=>{});
    });
  })();
})();

/* ---------------- lightbox + figure nav ---------------- */
(function(){
  const lb=document.querySelector(".lb");if(!lb)return;
  const img=lb.querySelector("img"),cap=lb.querySelector(".lb-cap");
  const announce=lb.querySelector(".lb-announce");
  const goto=lb.querySelector(".lb-goto");
  const figs=[...document.querySelectorAll("figure")].filter(f=>f.querySelector(".fig-body img, img"));
  let idx=-1,opener=null;
  const prev=document.createElement("button");
  prev.type="button";prev.className="lb-nav lb-prev";
  prev.setAttribute("aria-label","Previous figure");prev.textContent="\u2190";
  const next=document.createElement("button");
  next.type="button";next.className="lb-nav lb-next";
  next.setAttribute("aria-label","Next figure");next.textContent="\u2192";
  const pos=document.createElement("p");
  pos.className="lb-pos";pos.setAttribute("aria-label","Position in the figure sequence");
  lb.appendChild(prev);lb.appendChild(next);lb.appendChild(pos);

  function focusable(){
    return [lb.querySelector(".lb-close"),prev,next,goto].filter(el=>el&&!el.disabled&&!el.hidden);
  }

  function show(i){
    if(!figs.length) return;
    idx=((i%figs.length)+figs.length)%figs.length;
    const f=figs[idx],t=f.querySelector("img");
    img.src=t.dataset.full||t.src;img.alt=t.alt||"";
    const c=f.querySelector("figcaption");
    // "Figure N of M" would invent a second numbering authority: N is the
    // figure's own number, M the size of the enlargeable set, and the two only
    // agree while the one figure without an image happens to sit last. Name the
    // figure exactly as its caption does, and report the position separately as
    // a position, so neither can be read as the other.
    const n=f.dataset.fignum||"";
    const head=n?("Figure "+n):("Figure "+(idx+1));
    let body=c?c.textContent.trim():"";
    body=body.replace(/^Figure\s*\d+\.\s*/i,"");
    if(body.length>220) body=body.slice(0,217)+"\u2026";
    const capText=body?head+". "+body:head;
    cap.textContent=capText;
    pos.textContent=(idx+1)+" / "+figs.length;
    if(announce) announce.textContent=head+", position "+(idx+1)+" of "+figs.length+". "+body;
    lb.setAttribute("aria-label",head);
    if(goto){goto.href="#"+(f.id||"");goto.dataset.figTarget=f.id||"";}
    lb.classList.add("on");
    // Arrow-key/next-button navigation is the common path once the lightbox
    // is open, so warm the browser cache for both neighbours -- the reader
    // never waits on a full-res decode while stepping through the sequence.
    if(figs.length>1){
      const neighbourSrc=j=>{
        const t=figs[((j%figs.length)+figs.length)%figs.length].querySelector("img");
        return t&&(t.dataset.full||t.src);
      };
      [idx-1,idx+1].forEach(j=>{
        const s=neighbourSrc(j);
        if(s) new Image().src=s;
      });
    }
  }
  const open=t=>{
    opener=document.activeElement;
    const f=t.closest("figure");
    const i=f?figs.indexOf(f):-1;
    show(i>=0?i:0);
    lb.querySelector(".lb-close").focus();
  };
  const close=()=>{
    lb.classList.remove("on");
    if(opener&&typeof opener.focus==="function") opener.focus();
    opener=null;
  };
  document.addEventListener("click",e=>{
    const t=e.target;
    if(t.closest&&t.closest(".lb-nav")) return;
    if(t.tagName==="IMG"&&t.closest("figure")&&!t.closest(".xref-pop")){open(t);return;}
    if(t.classList.contains("act-zoom")){
      const i=t.closest("figure").querySelector("img");if(i)open(i);return;}
    if(t.closest&&t.closest(".lb-goto")) return;   // handled below, closes to a different place than opener
    if(t.closest(".lb")) close();
  });
  prev.addEventListener("click",e=>{e.stopPropagation();show(idx-1);});
  next.addEventListener("click",e=>{e.stopPropagation();show(idx+1);});
  // A reader who arrived via the gallery or an xref preview, not the article,
  // has no "back to the trigger" to return to -- send them to the figure's own
  // place in the text instead of wherever close() would otherwise land them.
  if(goto) goto.addEventListener("click",e=>{
    e.preventDefault();e.stopPropagation();
    const id=goto.dataset.figTarget;
    const target=id&&document.getElementById(id);
    opener=null;close();
    if(!target) return;
    let d=target.closest("details");
    while(d){d.open=true;d=d.parentElement&&d.parentElement.closest("details");}
    if(!target.hasAttribute("tabindex")) target.setAttribute("tabindex","-1");
    target.scrollIntoView({block:"start",behavior:reduced?"auto":"smooth"});
    target.focus({preventScroll:true});
  });
  const trapTab=makeFocusTrap(focusable);
  addEventListener("keydown",e=>{
    if(!lb.classList.contains("on")) return;
    if(e.key==="Escape"){close();return;}
    if(e.key==="ArrowLeft"){show(idx-1);return;}
    if(e.key==="ArrowRight"){show(idx+1);return;}
    trapTab(e);
  });
})();

/* ---------------- in-page search ---------------- */
(function(){
  const overlay=document.querySelector(".search-overlay");
  const btn=document.querySelector(".c-search");
  if(!overlay||!btn) return;
  const input=overlay.querySelector(".search-input");
  const list=overlay.querySelector(".search-results");
  let INDEX=null,indexLoading=null;

  function loadIndex(){
    if(INDEX) return Promise.resolve(INDEX);
    if(indexLoading) return indexLoading;
    indexLoading=fetch("search-index.json")
      .then(r=>r.json())
      .then(data=>{INDEX=data;return INDEX;})
      .catch(()=>{INDEX=[];return INDEX;});
    return indexLoading;
  }
  btn.addEventListener("pointerenter",()=>loadIndex(),{once:true});
  btn.addEventListener("focus",()=>loadIndex(),{once:true});
  setTimeout(()=>{if("requestIdleCallback" in window)requestIdleCallback(loadIndex);},4000);

  let opener=null,items=[],sel=-1;

  function escapeHtml(s){
    return s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  }
  function escapeRe(s){return s.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");}

  // One snippet window around the earliest matching term, then every term
  // highlighted inside it -- not just the one the window was centred on.
  function snippet(text,terms){
    const low=text.toLowerCase();
    let pos=-1;
    for(const t of terms){const p=low.indexOf(t);if(p>=0&&(pos<0||p<pos))pos=p;}
    if(pos<0) pos=0;
    const start=Math.max(0,pos-40),end=Math.min(text.length,pos+140);
    let out=text.slice(start,end);
    if(start>0) out="…"+out;
    if(end<text.length) out+="…";
    let html=escapeHtml(out);
    for(const t of terms){
      if(!t) continue;
      html=html.replace(new RegExp("("+escapeRe(t)+")","ig"),"<mark>$1</mark>");
    }
    return html;
  }

  function runSearch(q){
    const terms=q.toLowerCase().split(/\s+/).filter(Boolean);
    if(!terms.length) return [];
    const hits=[];
    for(const unit of INDEX){
      const hay=(unit.text+" "+(unit.section||"")).toLowerCase();
      if(!terms.every(t=>hay.includes(t))) continue;
      hits.push({unit,terms,heading:unit.tag==="h2"||unit.tag==="h3"||unit.tag==="h4"});
    }
    // headings first, otherwise keep the document's own order
    hits.sort((a,b)=>(b.heading?1:0)-(a.heading?1:0));
    return hits.slice(0,30);
  }

  function highlight(i){
    const buttons=list.querySelectorAll("button");
    buttons.forEach((b,k)=>b.classList.toggle("sel",k===i));
    if(buttons[i]) buttons[i].scrollIntoView({block:"nearest"});
    sel=i;
  }

  function render(q){
    if(!INDEX){
      loadIndex().then(()=>render(q));
      return;
    }
    items=runSearch(q);sel=-1;
    list.replaceChildren();
    overlay.classList.toggle("empty",q.trim().length>0&&items.length===0);
    items.forEach((hit,i)=>{
      const li=document.createElement("li");
      const b=document.createElement("button");
      b.type="button";b.dataset.index=String(i);
      b.innerHTML='<span class="search-hit-sec">'+escapeHtml(hit.unit.section||"")+"</span>"
        +'<span class="search-hit-snip">'+snippet(hit.unit.text,hit.terms)+"</span>";
      li.appendChild(b);list.appendChild(li);
    });
  }

  // Same jump idiom as the lightbox go-to-figure handler: open every enclosing
  // <details>, scroll, then move focus onto the target itself.
  function jump(unit){
    if(!unit||!unit.id) return;
    const target=document.getElementById(unit.id);
    if(!target) return;
    close();
    let d=target.closest("details");
    while(d){d.open=true;d=d.parentElement&&d.parentElement.closest("details");}
    if(!target.hasAttribute("tabindex")) target.setAttribute("tabindex","-1");
    target.scrollIntoView({block:"start",behavior:reduced?"auto":"smooth"});
    target.focus({preventScroll:true});
  }

  function open(){
    opener=document.activeElement;
    overlay.classList.add("on");
    input.value="";loadIndex().then(()=>render(""));
    input.focus();
  }
  function close(){
    overlay.classList.remove("on");overlay.classList.remove("empty");
    if(opener&&typeof opener.focus==="function") opener.focus();
    opener=null;
  }

  btn.addEventListener("pointerenter",()=>loadIndex(),{once:true});
  btn.addEventListener("click",()=>{overlay.classList.contains("on")?close():open();});
  input.addEventListener("input",()=>render(input.value));
  list.addEventListener("click",e=>{
    const b=e.target.closest("button");
    if(!b) return;
    const hit=items[parseInt(b.dataset.index,10)];
    if(hit) jump(hit.unit);
  });

  const trapTab=makeFocusTrap(()=>[input,...list.querySelectorAll("button")]);
  overlay.addEventListener("keydown",e=>{
    if(e.key==="Escape"){e.preventDefault();close();return;}
    if(e.key==="ArrowDown"){e.preventDefault();if(items.length)highlight(Math.min(items.length-1,sel+1));return;}
    if(e.key==="ArrowUp"){e.preventDefault();if(items.length)highlight(Math.max(0,sel-1));return;}
    if(e.key==="Enter"){
      e.preventDefault();
      const hit=items[sel>=0?sel:0];
      if(hit) jump(hit.unit);
      return;
    }
    trapTab(e);
  });

  // "/" opens search from anywhere on the page, except while typing in a
  // field -- a reader filling in an input should get a literal "/", not a
  // stolen keystroke.
  addEventListener("keydown",e=>{
    if(e.key!=="/"||overlay.classList.contains("on")) return;
    const t=e.target,tag=t&&t.tagName;
    if(tag==="INPUT"||tag==="TEXTAREA"||(t&&t.isContentEditable)) return;
    e.preventDefault();open();
  });
})();

/* ---------------- mobile TOC drawer ---------------- */
// Reuses the existing .spine/.toc markup (no second list): at the same
// max-width:66rem breakpoint the sidebar CSS already switches to, the
// "contents" button now opens it as an overlay drawer instead of doing
// nothing. A wide-viewport click still just toggles the desktop rail, via the
// unrelated data-rail handler set up in the reading-controls IIFE above.
(function(){
  const spine=document.querySelector(".spine");
  const btn=document.querySelector(".c-rail");
  const backdrop=document.querySelector(".toc-backdrop");
  if(!spine||!btn||!backdrop) return;
  const mq=matchMedia("(max-width:66rem)");
  let opener=null;

  function isOpen(){return root.classList.contains("toc-open");}
  function openDrawer(){
    if(!mq.matches||isOpen()) return;
    opener=document.activeElement;
    root.classList.add("toc-open");
    spine.setAttribute("role","dialog");
    spine.setAttribute("aria-modal","true");
    spine.setAttribute("aria-label","Contents");
    const first=spine.querySelector("a");
    if(first) first.focus(); else spine.focus();
  }
  function closeDrawer(){
    if(!isOpen()) return;
    root.classList.remove("toc-open");
    spine.removeAttribute("role");
    spine.removeAttribute("aria-modal");
    spine.removeAttribute("aria-label");
    if(opener&&typeof opener.focus==="function") opener.focus();
    opener=null;
  }

  btn.addEventListener("click",()=>{
    if(!mq.matches) return;
    isOpen()?closeDrawer():openDrawer();
  });
  backdrop.addEventListener("click",closeDrawer);
  spine.addEventListener("click",e=>{
    if(isOpen()&&e.target.closest("a")) closeDrawer();
  });
  const trapTab=makeFocusTrap(()=>[...spine.querySelectorAll("a")]);
  spine.addEventListener("keydown",e=>{
    if(!isOpen()) return;
    if(e.key==="Escape"){e.preventDefault();closeDrawer();return;}
    trapTab(e);
  });
  const onMqChange=e=>{if(!e.matches) closeDrawer();};
  if(mq.addEventListener) mq.addEventListener("change",onMqChange);
  else if(mq.addListener) mq.addListener(onMqChange);
})();

/* ---------------- keyboard shortcuts help dialog ---------------- */
(function(){
  // Single source of truth for the "?" help dialog: add a new shortcut here
  // and it appears in the table below automatically. This list must only ever
  // name keys something on the page actually listens for.
  const SHORTCUTS=[
    ["/","Open search"],
    ["?","Open this help dialog"],
    ["Esc","Close whichever dialog is open (search, help, or the lightbox)"],
    ["← →","In the lightbox: previous / next figure"],
    ["Tab / Shift+Tab","Move focus within an open dialog"],
    ["← →","Focused plot: pan the value readout"],
    ["+ / -","Focused plot: zoom in / out"],
    ["0","Focused plot: reset zoom"],
  ];
  const overlay=document.querySelector(".help-overlay");
  const btn=document.querySelector(".c-help");
  if(!overlay||!btn) return;
  const tbody=overlay.querySelector(".help-table tbody");
  const closeBtn=overlay.querySelector(".help-close");
  tbody.replaceChildren(...SHORTCUTS.map(([key,action])=>{
    const tr=document.createElement("tr");
    const kd=document.createElement("td");kd.textContent=key;
    const ad=document.createElement("td");ad.textContent=action;
    tr.append(kd,ad);return tr;
  }));

  let opener=null;
  function open(){
    opener=document.activeElement;
    overlay.classList.add("on");
    closeBtn.focus();
  }
  function close(){
    overlay.classList.remove("on");
    if(opener&&typeof opener.focus==="function") opener.focus();
    opener=null;
  }
  btn.addEventListener("click",()=>{overlay.classList.contains("on")?close():open();});
  closeBtn.addEventListener("click",close);
  overlay.addEventListener("click",e=>{if(e.target===overlay) close();});
  const trapTab=makeFocusTrap(()=>[closeBtn]);
  overlay.addEventListener("keydown",e=>{
    if(e.key==="Escape"){e.preventDefault();close();return;}
    trapTab(e);
  });
  // "?" opens help from anywhere, same guard as "/" for search: not while
  // typing in a field, and not if another overlay already owns the key.
  addEventListener("keydown",e=>{
    if(e.key!=="?"||overlay.classList.contains("on")) return;
    const t=e.target,tag=t&&t.tagName;
    if(tag==="INPUT"||tag==="TEXTAREA"||(t&&t.isContentEditable)) return;
    e.preventDefault();open();
  });
})();

/* ---------------- heading copy-link ---------------- */
// The anchor is a real <a href="#id">: Enter/click both navigate normally
// (updates location.hash, works with JS disabled) *and* trigger a clipboard
// copy of the full URL -- deliberately both, not a choice between them.
(function(){
  document.querySelectorAll(".head-anchor").forEach(a=>{
    a.addEventListener("click",()=>{
      const url=location.origin+location.pathname+location.search+a.getAttribute("href");
      const flash=()=>{
        a.classList.add("copied");
        clearTimeout(a._copiedTimer);
        a._copiedTimer=setTimeout(()=>a.classList.remove("copied"),1400);
      };
      if(navigator.clipboard&&navigator.clipboard.writeText){
        navigator.clipboard.writeText(url).then(flash,()=>{});
      }else{
        const ta=document.createElement("textarea");
        ta.value=url;ta.style.position="fixed";ta.style.opacity="0";
        document.body.appendChild(ta);ta.select();
        try{document.execCommand("copy");flash();}catch(e){/* clipboard unavailable */}
        ta.remove();
      }
    });
  });
})();

/* ---------------- hover / focus / tap previews ---------------- */
(function(){
  const coarse=matchMedia("(pointer:coarse)").matches;
  const pop=document.createElement("div");
  pop.className="xref-pop";pop.id="xref-pop";pop.setAttribute("role","tooltip");
  pop.hidden=true;document.body.appendChild(pop);
  let cur=null,hideT=null,armed=null;

  /* The equation anchor is an empty span that pandoc emits *inside* the
     paragraph, immediately before the display equation. A <div> cannot live in
     a <p>, so the parser closes the paragraph and hoists the .eqn out: in the
     DOM the anchor is the paragraph's last child and the equation is the
     paragraph's next sibling, one level up. Checking nextElementSibling alone
     finds nothing and the preview comes up blank -- which is how this shipped
     looking correct. Climb until an .eqn turns up, bounded so a stray anchor
     cannot drag in an unrelated equation from further down the section. */
  function eqnAfter(anchor){
    let node=anchor;
    for(let up=0;node&&up<3;up++,node=node.parentElement){
      for(let s=node.nextElementSibling;s;s=s.nextElementSibling){
        if(s.classList&&s.classList.contains("eqn")) return s;
        if(s.querySelector){const inner=s.querySelector(".eqn");if(inner) return inner;}
        if(s.textContent&&s.textContent.trim()) return null;   // real prose intervened
      }
    }
    return null;
  }

  function classify(a){
    const href=a.getAttribute("href")||"";
    if(href[0]!=="#") return null;
    let id;try{id=decodeURIComponent(href.slice(1));}catch(e){id=href.slice(1);}
    if(!id) return null;
    const el=document.getElementById(id);
    if(!el) return null;
    if(a.getAttribute("role")==="doc-biblioref"||el.classList.contains("csl-entry")||id.startsWith("ref-"))
      return {kind:"cite",el,href};
    if(a.dataset.referenceType==="eqref"||el.hasAttribute("data-eqno")||el.classList.contains("eqn")
       ||id.startsWith("eq:")||id.startsWith("eq-u")){
      let content=el;
      if(el.hasAttribute("data-eqno")) content=eqnAfter(el)||el;
      return {kind:"eq",el:content,href};
    }
    if((el.matches&&el.matches("figure"))||id.startsWith("fig:"))
      return {kind:"fig",el:el.matches&&el.matches("figure")?el:(el.closest("figure")||el),href};
    return null;
  }

  function fill(info){
    pop.replaceChildren();
    const body=document.createElement("div");body.className="xref-body";
    if(info.kind==="cite"){
      // No jump link here: role="tooltip" promises non-interactive content,
      // and a keyboard user reaches the same target by pressing Enter on the
      // trigger itself, so the link would only ever be reachable by a mouse
      // that could have clicked the trigger directly instead.
      const c=info.el.cloneNode(true);
      c.removeAttribute("id");
      c.querySelectorAll(".cited-in").forEach(n=>n.remove());
      body.appendChild(c);pop.appendChild(body);
    }else if(info.kind==="eq"){
      const c=info.el.cloneNode(true);
      if(c.removeAttribute) c.removeAttribute("id");
      body.appendChild(c);pop.appendChild(body);
    }else{
      const lab=document.createElement("p");lab.className="xref-label";
      lab.textContent="Figure "+(info.el.dataset.fignum||"");
      body.appendChild(lab);
      const im=info.el.querySelector("img");
      if(im){
        const c=document.createElement("img");
        c.className="xref-figimg";c.src=im.getAttribute("src")||im.src;c.alt=im.alt||"";
        body.appendChild(c);
      }
      const fc=info.el.querySelector("figcaption");
      if(fc){
        // The caption opens with its own "Figure N." span; keeping it here
        // would print the number twice under the label we just added.
        const c=fc.cloneNode(true);
        c.querySelectorAll(".num").forEach(n=>n.remove());
        const p=document.createElement("p");p.className="xref-cap";
        let t=c.textContent.trim();
        if(t.length>200) t=t.slice(0,197)+"\u2026";
        p.textContent=t;body.appendChild(p);
      }
      pop.appendChild(body);
    }
  }

  function place(a){
    const r=a.getBoundingClientRect();
    const pw=pop.offsetWidth,ph=pop.offsetHeight;
    let top=r.top-ph-10;
    if(top<8) top=Math.min(r.bottom+10,innerHeight-ph-8);
    top=Math.max(8,Math.min(top,innerHeight-ph-8));
    let left=r.left+r.width/2-pw/2;
    left=Math.max(8,Math.min(left,innerWidth-pw-8));
    pop.style.top=top+"px";pop.style.left=left+"px";
  }

  function hide(){
    clearTimeout(hideT);hideT=null;
    pop.classList.remove("on");
    if(cur){cur.removeAttribute("aria-describedby");cur=null;}
    const done=()=>{if(!pop.classList.contains("on")) pop.hidden=true;};
    if(reduced) done(); else setTimeout(done,120);
  }
  function scheduleHide(){clearTimeout(hideT);hideT=setTimeout(hide,120);}
  function cancelHide(){clearTimeout(hideT);hideT=null;}

  function open(a,info){
    cancelHide();
    fill(info);
    pop.hidden=false;
    if(cur&&cur!==a) cur.removeAttribute("aria-describedby");
    cur=a;a.setAttribute("aria-describedby","xref-pop");
    // measure then fade in
    pop.style.left="0";pop.style.top="0";
    place(a);
    if(reduced) pop.classList.add("on");
    else requestAnimationFrame(()=>{place(a);pop.classList.add("on");});
  }

  document.querySelectorAll('a[href^="#"]').forEach(a=>{
    if(!classify(a)) return;
    a.addEventListener("pointerenter",()=>{if(!coarse) open(a,classify(a));});
    a.addEventListener("pointerleave",()=>{if(!coarse) scheduleHide();});
    a.addEventListener("focus",()=>open(a,classify(a)));
    a.addEventListener("blur",scheduleHide);
  });
  pop.addEventListener("pointerenter",cancelHide);
  pop.addEventListener("pointerleave",scheduleHide);

  // first tap on coarse pointers opens the preview instead of navigating
  document.addEventListener("click",e=>{
    const a=e.target.closest&&e.target.closest('a[href^="#"]');
    if(!a||a.closest(".xref-pop")) return;
    const info=classify(a);
    if(!info){armed=null;return;}
    if(coarse){
      if(armed!==a){
        e.preventDefault();
        armed=a;open(a,info);return;
      }
      armed=null;hide();
    }
  },true);

  // A tap-armed popover has no hover to dismiss it and no keyboard on touch,
  // so without this a reader is stuck until the next scroll or link tap:
  // dismiss on any pointerdown/touchstart landing outside both the popover
  // and the link that opened it.
  function dismissIfOutside(e){
    if(!coarse||!armed) return;
    const t=e.target;
    if(t===armed||(t.closest&&(t.closest(".xref-pop")||t.closest("a")===armed))) return;
    armed=null;hide();
  }
  document.addEventListener("pointerdown",dismissIfOutside,true);
  document.addEventListener("touchstart",dismissIfOutside,true);

  addEventListener("keydown",e=>{if(e.key==="Escape") hide();});
  addEventListener("scroll",()=>{if(cur) hide();},{passive:true});
  addEventListener("resize",()=>{if(cur) hide();});
})();

/* ---------------- return pill ---------------- */
(function(){
  const pill=document.createElement("button");
  pill.type="button";pill.className="return-pill";pill.hidden=true;
  document.body.appendChild(pill);
  let origin=null,landY=null,hideTimer=null,landTimer=null;

  function currentSecno(){
    const on=document.querySelector(".spine a.on i");
    if(on&&on.textContent.trim()) return on.textContent.trim();
    const secs=[...document.querySelectorAll("article section[id]")];
    let cur=null;
    for(const s of secs){
      if(s.getBoundingClientRect().top<=innerHeight*0.35) cur=s;
    }
    if(!cur) return "";
    const h=cur.querySelector("h2, h3");
    const sn=h&&h.querySelector(".secno");
    return sn?sn.textContent.trim():"";
  }

  function hide(){
    clearTimeout(hideTimer);hideTimer=null;
    clearTimeout(landTimer);landTimer=null;
    origin=null;landY=null;
    pill.classList.remove("on");
    const done=()=>{if(!pill.classList.contains("on")) pill.hidden=true;};
    if(reduced) done(); else setTimeout(done,180);
  }
  function show(y,sec){
    origin={y,sec};landY=null;
    pill.textContent=sec?("\u21A9 back to \u00A7"+sec):"\u21A9 back";
    pill.hidden=false;
    // Reading offsetWidth flushes layout, which gives the transition the frame
    // boundary it needs. requestAnimationFrame would do the same but does not
    // run in a hidden tab -- a reader who jumps and immediately switches tabs
    // would come back to a pill that never appeared and never armed its timeout.
    void pill.offsetWidth;
    pill.classList.add("on");
    clearTimeout(hideTimer);
    hideTimer=setTimeout(hide,45000);
    // snapshot landing position after smooth-scroll settles
    clearTimeout(landTimer);
    landTimer=setTimeout(()=>{landY=scrollY;},reduced?0:550);
  }

  pill.addEventListener("click",()=>{
    if(!origin) return;
    const y=origin.y;hide();
    scrollTo({top:y,behavior:reduced?"auto":"smooth"});
  });

  document.addEventListener("click",e=>{
    const a=e.target.closest&&e.target.closest('a[href^="#"]');
    if(!a) return;
    // skip the coarse first-tap that only opens a preview
    if(e.defaultPrevented) return;
    const href=a.getAttribute("href")||"";
    if(href[0]!=="#") return;
    let id;try{id=decodeURIComponent(href.slice(1));}catch(err){id=href.slice(1);}
    if(!id||!document.getElementById(id)) return;
    show(scrollY,currentSecno());
  });

  addEventListener("scroll",()=>{
    if(!origin||landY===null) return;
    if(Math.abs(scrollY-landY)>innerHeight*0.7) hide();
  },{passive:true});
  // never cover the lightbox
  const lb=document.querySelector(".lb");
  if(lb) new MutationObserver(()=>{
    if(lb.classList.contains("on")) pill.style.visibility="hidden";
    else if(pill.classList.contains("on")) pill.style.visibility="";
  }).observe(lb,{attributes:true,attributeFilter:["class"]});
})();

/* ---------------- cited-in back-links ---------------- */
(function(){
  function buildBacklinks(){
    function numberedSection(from){
      let sec=from&&from.closest("section");
      while(sec){
        if(sec.id){
          let h=null;
          for(const c of sec.children){
            if(c.matches("h2, h3")){h=c;break;}
            if(c.matches("details")){
              const s=c.querySelector(":scope > summary > h2");
              if(s){h=s;break;}
            }
          }
          const sn=h&&h.querySelector(".secno");
          const label=sn&&sn.textContent.trim();
          if(label) return {id:sec.id,label};
        }
        sec=sec.parentElement&&sec.parentElement.closest("section");
      }
      return null;
    }
    const map=new Map();
    document.querySelectorAll('a[role="doc-biblioref"]').forEach(a=>{
      const href=a.getAttribute("href")||"";
      if(href[0]!=="#") return;
      const key=href.slice(1);
      const where=numberedSection(a);
      if(!where) return;
      if(!map.has(key)) map.set(key,[]);
      const list=map.get(key);
      if(!list.some(x=>x.id===where.id)) list.push(where);
    });
    const rank=s=>s.split(".").map(n=>parseInt(n,10)||0);
    const cmp=(a,b)=>{
      const A=rank(a.label),B=rank(b.label);
      for(let i=0;i<Math.max(A.length,B.length);i++){
        const d=(A[i]||0)-(B[i]||0);if(d) return d;
      }
      return 0;
    };
    for(const [key,cites] of map){
      const el=document.getElementById(key);
      if(!el) continue;
      cites.sort(cmp);
      const cap=cites.slice(0,6),rest=cites.length-cap.length;
      const line=document.createElement("div");
      line.className="cited-in";
      line.appendChild(document.createTextNode("Cited in "));
      cap.forEach((c,i)=>{
        if(i) line.appendChild(document.createTextNode(", "));
        const a=document.createElement("a");
        a.href="#"+c.id;a.textContent="\u00A7"+c.label;
        line.appendChild(a);
      });
      if(rest) line.appendChild(document.createTextNode(" + "+rest+" more"));
      el.appendChild(line);
    }
  }
  if("requestIdleCallback" in window){requestIdleCallback(buildBacklinks);}
  else setTimeout(buildBacklinks,200);
})();

/* ---------------- print: open closed details ---------------- */
(function(){
  // CSS display:block does not open a closed <details>; force the open attribute.
  let opened=null;
  const openAll=()=>{
    if(opened) return;
    opened=[];
    document.querySelectorAll("details").forEach(d=>{
      if(!d.open){d.open=true;opened.push(d);}
    });
  };
  const restore=()=>{
    if(!opened) return;
    opened.forEach(d=>{d.open=false;});
    opened=null;
  };
  addEventListener("beforeprint",openAll);
  addEventListener("afterprint",restore);
  const mq=matchMedia("print");
  const onMq=e=>{if(e.matches) openAll(); else restore();};
  if(mq.addEventListener) mq.addEventListener("change",onMq);
  else if(mq.addListener) mq.addListener(onMq);
})();

/* ---------------- cross-references into folded back matter ---------------- */
function revealTarget(){
  const id=decodeURIComponent(location.hash.slice(1)).split("&")[0];
  if(!id) return;
  const el=document.getElementById(id);
  if(!el) return;
  let d=el.closest("details");
  while(d){d.open=true;d=d.parentElement&&d.parentElement.closest("details");}
  requestAnimationFrame(()=>el.scrollIntoView({block:"start",behavior:"auto"}));
}
addEventListener("hashchange",revealTarget);
if(location.hash) addEventListener("load",revealTarget);
document.addEventListener("click",e=>{
  const a=e.target.closest&&e.target.closest('a[href^="#"]');
  if(!a) return;
  const el=document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1)));
  if(!el) return;
  let d=el.closest("details");
  while(d){d.open=true;d=d.parentElement&&d.parentElement.closest("details");}
});

/* ---------------- reveal + spine ---------------- */
const io=new IntersectionObserver(es=>{
  for(const e of es) if(e.isIntersecting){e.target.classList.add("in");io.unobserve(e.target);}
},{rootMargin:"-20px"});
document.querySelectorAll(".reveal").forEach(n=>io.observe(n));

const figIn=new IntersectionObserver(es=>{
  for(const e of es) if(e.isIntersecting){e.target.classList.add("seen");figIn.unobserve(e.target);}
},{rootMargin:"0px 0px -8% 0px"});
document.querySelectorAll("figure").forEach(n=>figIn.observe(n));

const links=[...document.querySelectorAll(".spine a")];
function openBranch(a){
  if(!a) return;
  const li=a.closest("li.top");
  document.querySelectorAll(".toc > li.top").forEach(n=>{if(n!==li)n.classList.remove("open");});
  if(li) li.classList.add("open");
}
// Reuses this same observer's notion of "the active section" for the address
// bar, throttled, instead of standing up a second observer just to watch
// scroll position for the hash.
let hashTimer=null;
const spy=new IntersectionObserver(es=>{
  for(const e of es){
    if(!e.isIntersecting)continue;
    const id=e.target.id;
    links.forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+id));
    const on=links.find(a=>a.classList.contains("on"));
    openBranch(on);
    if(on&&on.offsetParent) on.scrollIntoView({block:"nearest"});
    if(id){
      clearTimeout(hashTimer);
      hashTimer=setTimeout(()=>{
        // Live-figure params share the hash as &key=value tokens; keep them
        // when the spy rewrites the section id or a scroll would silently
        // drop a shared view's parameters. #live-debug is a bare token, not
        // a key=value pair, and has to be kept the same way or telemetry
        // turns off as soon as the reader scrolls.
        const cur=location.hash.slice(1);
        const parts=cur.split("&");
        const params=parts.filter(t=>t.indexOf("=")>=0);
        const debug=parts.filter(t=>t==="live-debug");
        const extra=debug.concat(params).join("&");
        const next=id+(extra?"&"+extra:"");
        if(cur!==next){
          history.replaceState(null,"",location.pathname+location.search+"#"+next);
        }
      },500);
    }
  }
},{rootMargin:"-20% 0px -70%"});
if("requestIdleCallback" in window){
  requestIdleCallback(()=>{document.querySelectorAll("article section[id]").forEach(s=>spy.observe(s));});
} else {
  setTimeout(()=>{document.querySelectorAll("article section[id]").forEach(s=>spy.observe(s));},100);
}
links.forEach(a=>a.addEventListener("click",()=>openBranch(a)));
openBranch(links[0]);

document.querySelectorAll(".hero-inner > *").forEach((n,i)=>{
  n.classList.add("hero-rise");
  n.style.animationDelay=(0.06*i+0.05).toFixed(2)+"s";
});

const rebuildHero=hero();
function repaint(){rebuildHero();MOUNTED.forEach(p=>p.redraw());}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",repaint);
new MutationObserver(repaint).observe(root,{attributes:true,attributeFilter:["data-theme"]});

/* ---------------- lazy-load math webfont ---------------- */
(function(){
  if(!("fonts" in document)||typeof FontFace==="undefined") return;
  const mf=new FontFace("TeX Gyre Pagella Math","url('fonts/pagella-math.woff2') format('woff2')",{style:"normal",weight:"400",display:"swap"});
  let loaded=false;
  const loadMath=()=>{if(loaded)return;loaded=true;mf.load().then(f=>document.fonts.add(f)).catch(()=>{});};
  const firstMath=document.querySelector("math");
  if(firstMath&&"IntersectionObserver" in window){
    const io=new IntersectionObserver(es=>{
      for(const e of es){if(e.isIntersecting){loadMath();io.disconnect();break;}}
    },{rootMargin:"600px"});
    io.observe(firstMath);
  } else {
    if("requestIdleCallback" in window) requestIdleCallback(loadMath);
    else setTimeout(loadMath,1500);
  }
})();
"""
