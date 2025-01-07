from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import interpolate
from scipy import ndimage

import drawing
import image_loader


def apply_radial_filter_to_raw(
    function: Callable[[npt.NDArray, tuple[float, float], ...], npt.NDArray],
    image: npt.NDArray,
    center: tuple[float, float],
    *args) -> npt.NDArray:
  new_image = np.zeros(image.shape, dtype=float)
  for offset in image_loader.BAYER_MASK_OFFSET.values():
    current_center = (
        center[0] / 2 + offset[0] / 2,
        center[1] / 2 + offset[1] / 2,
    )
    new_image[offset[0]::2, offset[1]::2] = function(
        image[offset[0]::2, offset[1]::2],
        current_center,
        *args
    )
  return new_image


def radial_gaussian_filter(image: npt.NDArray,
                           center: tuple[float, float],
                           sigma_r: float,
                           sigma_theta: float,
                           min_radius: float = 0,
                           taper_width: float = 0) -> npt.NDArray:
  """Applies a radial Gaussian filter to the image.

  Args:
    image: image to be filtered.
    center: center of the radial Gaussian filter.
    sigma_r: Gaussian filter standard deviation in the radial direction.
    sigma_theta: Gaussian filter standard deviation in the tangential
      direction.
    min_radius: minimum radius for applying the Gaussian filter.
    taper_width: distance over which to taper on the filter.
  """
  # Pixel coordinates for Euclidean grid.
  x = np.arange(image.shape[0]) - center[0]
  y = np.arange(image.shape[1]) - center[1]
  x0, y0 = np.meshgrid(x, y, indexing='ij')
  r0 = np.sqrt(x0**2 + y0**2)
  theta0 = np.arctan2(y0, x0)

  # Pixel coordinates for polar grid.
  dtheta = 0.6 / np.max(r0)
  theta = np.linspace(-np.pi, np.pi, int(2 * np.pi / dtheta))
  r = np.arange(int(np.ceil(np.max(r0))) + 1)
  r1, theta1 = np.meshgrid(r, theta, indexing='ij')
  x1 = r1 * np.cos(theta1)
  y1 = r1 * np.sin(theta1)

  # Transform image to polar grid.
  image_polar = interpolate.interpn(
      (x, y),
      image,
      np.stack((x1, y1), axis=2),
      method='linear',
      bounds_error=False,
      fill_value=0,
  )

  # Filter in polar coordinates.
  filtered_polar = ndimage.gaussian_filter(image_polar, (sigma_r, sigma_theta))

  # Transform back to the Euclidean grid.
  filtered = interpolate.interpn(
      (r, theta),
      filtered_polar,
      np.stack((r0, theta0), axis=2),
      method='linear'
  )

  # Blend the original and filtered images to preserve the original image
  # inside the minimum specified radius.
  mask = drawing.draw_tapered_disc(
      image.shape, center, min_radius, min_radius + taper_width)
  return mask * image + (1 - mask) * filtered
