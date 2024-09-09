from collections.abc import Sequence
import copy
import os

import numpy as np
import numpy.typing as npt

import constants
import filepaths
import image_loader


def split_into_exposure_stacks(images: Sequence[image_loader.RawImage]):
  stacks = []
  current_stack = None
  prev_ind = 0
  for image in images:
    ind = constants.EXPOSURES.index(image.get_exposure())
    if ind == 0:
      if current_stack is not None:
        stacks.append(current_stack)
      current_stack = []
    elif ind != prev_ind + 1:
      raise ValueError(
          'Non-sequential exposure in exposure stack. Expected '
          f'exposure {prev_ind + 1:d} but got exposure {ind:d}.'
      )
    current_stack.append(image)
    prev_ind = ind
  stacks.append(current_stack)
  return stacks


def _exposure_ind(exposure: image_loader.Exposure):
  return constants.EXPOSURES.index(exposure)


class RawProcessor:
  shape: tuple[int, int]
  background: npt.NDArray
  stdev: npt.NDArray
  hot_pixel_indices: list[tuple[int, int]]

  def __init__(self, hot_pixel_threshold: int = 5000,
               saturated_pixel_value: int = 14_000):
    self.hot_pixel_threshold = hot_pixel_threshold
    self.saturated_pixel_value = saturated_pixel_value

  def save_calibration_to_file(self, filename: str):
    np.savez(
        filepaths.calibration(filename),
        shape=self.shape,
        background=self.background,
        stdev=self.stdev,
        hot_pixel_indices=np.asarray(self.hot_pixel_indices)
    )

  def load_calibration_from_file(self, filename: str):
    data = np.load(filepaths.calibration(filename))

    self.shape = tuple(data['shape'])
    self.background = data['background']
    self.stdev = data['stdev']
    self.hot_pixel_indices = [(entry[0], entry[1]) for entry in
                              data['hot_pixel_indices']]

  def process_black_frames(self, images: Sequence[image_loader.RawImage]):
    images = list(images)

    background = []
    stdev = []
    shapes = []
    for exposure in constants.EXPOSURES:
      images_at_exposure = []
      for image in images:
        if image.get_exposure() == exposure:
          images_at_exposure.append(image)

      black_frames = np.stack(
          [image.raw_image for image in images_at_exposure], axis=0)
      background.append(np.mean(black_frames, axis=0))
      stdev.append(np.std(black_frames, axis=0))
      shapes.append(black_frames.shape[1:])
    self.background = np.stack(background, axis=0)
    self.stdev = np.stack(stdev, axis=0)

    for shape in shapes:
      if shape != shapes[0]:
        raise ValueError('Not all images are the same size.')
    self.shape = shapes[0]

    # Determine hot pixels using the longest exposure. Ignore border pixels
    # for convenience since we typically crop our images anyways.
    hot_pixels = (background[constants.LONGEST_EXPOSURE_IND] >
                  self.hot_pixel_threshold)
    hot_pixels[:1, :] = False
    hot_pixels[-2:, :] = False
    hot_pixels[:, :1] = False
    hot_pixels[:, -2:] = False
    self.hot_pixel_indices = np.argwhere(hot_pixels)

  def subtract_background(self, image: image_loader.RawImage
                          ) -> image_loader.RawImage:
    if image.raw_image.shape != self.shape:
      raise ValueError('Shape of image does not match black frames.')

    new_image = copy.copy(image)
    new_image.bw_image = None
    background = self.background[_exposure_ind(image.get_exposure())]
    new_image.raw_image = np.asarray(image.raw_image, dtype=float) - background
    return new_image

  def remove_hot_pixels(self, image: image_loader.RawImage
                        ) -> image_loader.RawImage:
    if image.raw_image.shape != self.shape:
      raise ValueError('Shape of image does not match black frames.')

    new_image = copy.copy(image)
    new_image.bw_image = None

    # Replace hot pixels with average of the 4 neighbouring pixels from the
    # same Bayer subimage.
    corrected =  np.copy(image.raw_image)
    for row, col in self.hot_pixel_indices:
      corrected[row, col] = np.mean((
          corrected[row - 2, col],
          corrected[row + 2, col],
          corrected[row, col - 2],
          corrected[row, col + 2]
      ))
    new_image.raw_image = corrected
    return new_image

  def stack_hdr_image(self, images: Sequence[image_loader.RawImage]
                      ) -> npt.NDArray:
    weights = np.zeros(images[0].raw_image.shape, dtype=float)
    weighted_pixels = np.zeros(images[0].raw_image.shape, dtype=float)
    for image in images:
      corrected_image = self.remove_hot_pixels(self.subtract_background(image))

      # Normalize image to an F-number of 1.0 and ISO 100.
      effective_exposure_s = (
          image.exposure_s / image.f_number**2 * image.iso / 100)
      scale_factor = 1 / effective_exposure_s
      norm_image = scale_factor * corrected_image.raw_image

      # Determine weight per pixel.
      not_saturated = image.raw_image < self.saturated_pixel_value
      stdev = scale_factor * np.maximum(
          1e-2,
          self.stdev[_exposure_ind(image.get_exposure())]
      )
      weight = not_saturated * (1 / stdev**2)

      # Perform weighted average of pixel values from each image.
      weighted_pixels += weight * norm_image
      weights += weight

    # Pixels with zero total weight are marked as NaN.
    bad_pixels = weights == 0
    weights[bad_pixels] = 1
    weighted_pixels[bad_pixels] = np.nan

    # Normalize pixel values by weights.
    return weighted_pixels / weights
