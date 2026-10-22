"""Visualization tools for dynamical systems.

Requires the ``viz`` optional dependency group::

    pip install dynachaos[viz]
"""

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "dynachaos.viz requires matplotlib. Install with: pip install dynachaos[viz]"
    ) from exc

from dynachaos.viz.bifurcation import bifurcation_diagram
from dynachaos.viz.cobweb import cobweb_diagram
from dynachaos.viz.poincare import poincare_section_plot
from dynachaos.viz.return_map import return_map_plot

__all__ = [
    "bifurcation_diagram",
    "cobweb_diagram",
    "poincare_section_plot",
    "return_map_plot",
]
