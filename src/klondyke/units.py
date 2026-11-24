"""Unit conversions for YCGC historic placer data.

YCGC recorded grade as cents of gold per cubic yard, valued at a fixed
historic price of $20.67 per fine troy ounce. Dollars at a fixed price are a
mass proxy, so we convert to milligrams of fine gold per cubic metre, which is
price-independent and is what a modern model should use.

    1 cent/yd3 @ $20.67/oz
      = (0.01 / 20.67) oz/yd3
      = 4.8379e-4 oz * 31103.4768 mg/oz / 0.764554858 m3
      ~ 19.68 mg/m3
"""

from __future__ import annotations

import numpy as np

TROY_OZ_MG = 31_103.4768
YD3_M3 = 0.764_554_857_984
FT_M = 0.3048
HISTORIC_PRICE_PER_OZ = 20.67

# Value YCGC / van Loon used for "trace" (TR) assays, in cents per yd3.
TRACE_CENTS = 0.1


def cents_per_yd3_to_mg_per_m3(cents, price_per_oz: float = HISTORIC_PRICE_PER_OZ):
    """Convert grade in cents/yd3 at a fixed gold price to mg fine Au per m3."""
    cents = np.asarray(cents, dtype=float)
    oz_per_yd3 = (cents / 100.0) / price_per_oz
    return oz_per_yd3 * TROY_OZ_MG / YD3_M3


def mg_per_m3_to_g_per_t(mg_m3, bulk_density_t_m3: float = 2.0):
    """Approximate g/t for comparison with modern reports (bulk density is an assumption)."""
    return np.asarray(mg_m3, dtype=float) / 1000.0 / bulk_density_t_m3


def ft_to_m(ft):
    return np.asarray(ft, dtype=float) * FT_M


def m_to_ft(m):
    return np.asarray(m, dtype=float) / FT_M


def accumulation_mg_per_m2(grade_mg_m3, thickness_m):
    """Gold per unit area of ground: grade times the thickness it applies to.

    For YCGC drill data the grade applies to the 'dredge section'
    (pay gravel plus the bedrock dug), so thickness_m is that section.
    """
    return np.asarray(grade_mg_m3, dtype=float) * np.asarray(thickness_m, dtype=float)


def modern_value_per_m3(grade_mg_m3, price_per_oz: float):
    """Gross in-situ value per m3 at a modern price (no recovery, no fineness)."""
    return np.asarray(grade_mg_m3, dtype=float) / TROY_OZ_MG * price_per_oz
