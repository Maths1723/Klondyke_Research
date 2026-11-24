"""Klondike placer gold prospectivity: data, validation and models.

Core modules (units, qc, valley, cv, baselines, models, metrics, synthetic)
depend only on numpy / pandas / scipy / scikit-learn. Geospatial I/O lives in
``ingest`` and needs geopandas, which is imported lazily.
"""

__version__ = "0.1.0"
