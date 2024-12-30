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

def draw_tapered_disc(image_shape: tuple[int, int],
                      center: tuple[float, float],
                      inner_radius: float,
                      outer_radius: float) -> npt.NDArray:
  """Draws a filled disc with a tapered edge.

  The pixels are set to 1 within the inner radius, and set to 0 outside the
  outer radius. The values linearly taper from the inner to outer radius.
  """
  x = np.arange(image_shape[0]) - center[0]
  y = np.arange(image_shape[1]) - center[1]
  x, y = np.meshgrid(x, y, indexing='ij')
  r = np.sqrt(x**2 + y**2)
  return np.clip(
      (outer_radius - r) / (outer_radius - inner_radius),
      a_min=0,
      a_max=1
  )



