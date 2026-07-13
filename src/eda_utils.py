"""Shared EDA helpers for the Yelp case study — validated palette, chart style, text utils."""
import re
import numpy as np

# validated data-viz palette (light surface)
SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
VIOLET = "#4a3aa7"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def style(fig, title, ytitle="", xtitle="", ymax=None, h=360, legend=False):
    """Apply the shared chart chrome (recessive grid/axes, palette ink, title)."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK, family=FONT), x=0.02),
        paper_bgcolor=SURF, plot_bgcolor=SURF,
        font=dict(family=FONT, color=INK2, size=12),
        margin=dict(l=60, r=24, t=48, b=48), height=h, bargap=0.28, showlegend=legend)
    if legend:
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    fig.update_xaxes(title=xtitle, showgrid=False, zeroline=False, linecolor=BASE,
                     ticks="outside", tickcolor=BASE, color=MUTED)
    fig.update_yaxes(title=ytitle, showgrid=True, gridcolor=GRID, zeroline=True,
                     zerolinecolor=BASE, linecolor=SURF, color=MUTED,
                     range=[0, ymax] if ymax else None)
    return fig


_TOKEN = re.compile(r"[a-z']+")


def simple_tokens(text):
    """Lowercase word tokenizer for length/vocab EDA (subword count ≈ 1.3× these)."""
    return _TOKEN.findall(text.lower())


def logodds_z(ci, cj, prior):
    """Monroe et al. (2008) weighted log-odds with an informative Dirichlet prior.

    ci, cj  : aligned per-term count arrays for the two groups being contrasted.
    prior   : pooled background counts over the whole corpus (the informative prior).
    Returns z-scores — large positive ⇒ distinctive to group i, large negative ⇒ group j.
    Preferable to raw frequency, which is dominated by common words.
    """
    ni = ci.sum()
    nj = cj.sum()
    a0 = prior.sum()
    li = np.log((ci + prior) / (ni + a0 - ci - prior))
    lj = np.log((cj + prior) / (nj + a0 - cj - prior))
    delta = li - lj
    var = 1.0 / (ci + prior) + 1.0 / (cj + prior)
    return delta / np.sqrt(var)
