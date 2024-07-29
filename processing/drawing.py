import float_raster
import numpy as np
import numpy.typing as npt

def draw_circle(image_shape: tuple[int, int],
                center: tuple[float, float],
                radius: float,
                max_error: float = 1e-3,
                renderer: str = 'float_raster') -> npt.NDArray:
  # Determine required number of points to achieved required error.
  dtheta = 2 * np.arccos(1 - max_error / radius)
  num_theta = int(np.ceil(2 * np.pi / dtheta))

  # Approximate circle as a polygon.
  theta = - np.linspace(0, 2 * np.pi, num_theta + 1)
  x = center[0] + radius * np.cos(theta)
  y = center[1] + radius * np.sin(theta)
  vertices = np.stack((x, y))

  if renderer == 'float_raster':
    return float_raster.raster(
        vertices,
        np.arange(image_shape[0] + 1),
        np.arange(image_shape[1] + 1)
    )
  else:
    raise ValueError(f'Unexpected renderer "{renderer}"')

