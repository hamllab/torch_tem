from importlib import resources
import matplotlib.pyplot as plt


def set_style(style_path=None):
    """Set default plot style."""
    if style_path is None:
        style_path = resources.files('tem').joinpath('figures.mplstyle')
    plt.style.use(style_path)
