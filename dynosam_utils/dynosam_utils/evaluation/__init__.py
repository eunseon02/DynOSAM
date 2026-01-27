try:
    from .core.plotting import startup_plotting
    font_size = 20
    startup_plotting(font_size)
except ImportError as e:
    # gtsam may not be available, plotting will fail if needed
    import warnings
    warnings.warn(f"Could not import plotting module: {e}. Plotting features will be unavailable.")
