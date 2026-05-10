"""Chart functions used by report sections.

Each chart function returns a :class:`ChartArtifact`. When ``use_plotly`` is
True the function attempts to render a Plotly figure (returning an inline
HTML fragment); on any failure it falls back to a matplotlib PNG. Sections
inline ``artifact.to_html()`` directly into their HTML output and add
``artifact.files`` to their ``SectionRecord.files`` list.

The Plotly path uses ``include_plotlyjs=False`` and ``full_html=False``
so a single ``plotly.js`` is loaded once at the top of ``index.html``
(see template ``include_plotly`` flag).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


# =============================================================================
# Public artifact shape
# =============================================================================

@dataclass
class ChartArtifact:
    """A rendered chart returned to a section emitter.

    Attributes:
        kind:    "plotly" | "png" | "empty"
        html:    when kind=="plotly", the rendered ``<div>``-with-script HTML
                 fragment; ignored otherwise. Already escape-safe — sections
                 emit it via ``|safe`` (or directly into the html_fragment).
        png_path: when kind=="png", path to the saved PNG (POSIX paths are
                 used relative to the report root by ``ChartArtifact.to_html``).
        caption: optional caption rendered under the figure.
        div_id:  Plotly element id (used for unique anchoring).
        rel_png: POSIX path to the PNG, relative to the report root —
                 populated by sections via ``set_relative_png``.
        files:   POSIX-relative paths the section should record in its
                 ``SectionRecord.files`` list (orphan-cleanup safe).
    """

    kind: str = "empty"
    html: str = ""
    png_path: Optional[Path] = None
    caption: str = ""
    div_id: str = ""
    rel_png: str = ""
    files: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.kind == "empty"

    def to_html(self) -> str:
        """Render the artifact as an HTML fragment.

        The PNG variant uses ``rel_png`` if it has been populated by the
        caller (typical: the section passes its ``ctx.rel(png_path)`` in
        before producing the fragment); otherwise it falls back to a
        bare ``<img src>`` whose ``src`` is the basename — sections
        should always set ``rel_png``.
        """
        if self.kind == "plotly":
            wrapper = f'<div class="plotly-fig" id="wrap-{self.div_id}">{self.html}</div>'
            if self.caption:
                wrapper += f'<div class="caption">{_html_escape(self.caption)}</div>'
            return wrapper
        if self.kind == "png":
            src = self.rel_png or (self.png_path.name if self.png_path else "")
            cap = (
                f'<figcaption>{_html_escape(self.caption)}</figcaption>'
                if self.caption else ""
            )
            return (
                f'<figure><img src="{_html_escape(src)}" '
                f'alt="{_html_escape(self.caption or "figure")}">{cap}</figure>'
            )
        return '<div class="empty"><p><em>no data</em></p></div>'


def _html_escape(s: str) -> str:
    import html as _h
    return _h.escape("" if s is None else str(s), quote=True)


# =============================================================================
# Plotly fallback wrapper
# =============================================================================

def render_with_fallback(
    plotly_fn,
    matplotlib_fn,
    *,
    use_plotly: bool,
    png_path: Path,
    div_id: str,
    caption: str = "",
) -> ChartArtifact:
    """Try the Plotly version; on any failure fall back to matplotlib PNG.

    ``plotly_fn() -> str`` must return an HTML fragment built with
    ``plotly.io.to_html(fig, include_plotlyjs=False, full_html=False,
    div_id=div_id)`` (or empty string to indicate "no data").

    ``matplotlib_fn(out_path: Path) -> Optional[Path]`` writes the PNG
    and returns its path, or None if there's nothing to render.
    """
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if use_plotly:
        try:
            html = plotly_fn()
            if html:
                return ChartArtifact(
                    kind="plotly", html=html, caption=caption, div_id=div_id,
                )
            # If plotly returned empty intentionally (no data), don't bother
            # the matplotlib path with a PNG either — return empty.
            return ChartArtifact(kind="empty", caption=caption)
        except Exception as e:
            logger.warning(
                "plotly render failed (%s); falling back to matplotlib for %s",
                e, png_path.name,
            )

    try:
        out = matplotlib_fn(png_path)
        if out is None:
            return ChartArtifact(kind="empty", caption=caption)
        return ChartArtifact(
            kind="png", png_path=Path(out), caption=caption, div_id=div_id,
        )
    except Exception as e:
        logger.exception("matplotlib render failed for %s: %s", png_path.name, e)
        return ChartArtifact(kind="empty", caption=caption)


# =============================================================================
# Plotly common helpers
# =============================================================================

# Keep the visual identity aligned with wagie.charts.theme.
PLOTLY_PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "warning": "#d62728",
    "muted": "#7f7f7f",
    "background": "#ffffff",
    "grid": "#e6e6e6",
}


def plotly_layout_defaults(title: str = "") -> dict:
    """Common ``layout=`` kwargs aligned with the matplotlib theme."""
    return {
        "title": {"text": title, "font": {"size": 13}},
        "paper_bgcolor": PLOTLY_PALETTE["background"],
        "plot_bgcolor": PLOTLY_PALETTE["background"],
        "font": {"family": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
                 "size": 11, "color": "#1a1a1a"},
        "margin": {"l": 56, "r": 16, "t": 38, "b": 44},
        "xaxis": {"gridcolor": PLOTLY_PALETTE["grid"], "zerolinecolor": PLOTLY_PALETTE["muted"]},
        "yaxis": {"gridcolor": PLOTLY_PALETTE["grid"], "zerolinecolor": PLOTLY_PALETTE["muted"]},
        "hovermode": "closest",
        "showlegend": False,
    }


def plotly_to_div(fig, *, div_id: str) -> str:
    """Render a Plotly figure as an inline ``<div>`` fragment.

    ``include_plotlyjs=False`` because the template loads plotly.js once at
    the top of the report (see ``include_plotly`` in the template).
    """
    import plotly.io as pio
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False, div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


__all__ = [
    "ChartArtifact",
    "render_with_fallback",
    "PLOTLY_PALETTE",
    "plotly_layout_defaults",
    "plotly_to_div",
]
