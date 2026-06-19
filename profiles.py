# ======================
# functions for fitting basic sersic, mass profiles, etc.
# ======================

import numpy as np
import matplotlib.pyplot as plt


# Create a mask (from Gottemoller+26 methodology)
def mask_center_2d(center_x, center_y, r, x_grid, y_grid):
    x_shift = x_grid - center_x
    y_shift = y_grid - center_y
    R = np.sqrt(x_shift * x_shift + y_shift * y_shift)
    mask = np.empty_like(R, dtype="int")
    mask[R > r] = 1
    mask[R <= r] = 0
    return mask