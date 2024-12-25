from collections.abc import Sequence
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy import special

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


def _remove_specified_pixels(image: npt.NDArray,
                             pixel_indices: Sequence[tuple[int, int]]
                             ) -> npt.NDArray:
  corrected =  np.copy(image)

  # Replace hot pixels with average of the 4 neighbouring pixels from the
  # same Bayer subimage.
  for row, col in pixel_indices:
    corrected[row, col] = np.mean((
        corrected[row - 2, col],
        corrected[row + 2, col],
        corrected[row, col - 2],
        corrected[row, col + 2]
    ))
  return corrected


class RawProcessor:
  shape: tuple[int, int]
  background: npt.NDArray
  stdev: npt.NDArray
  hot_pixel_indices: list[tuple[int, int]]

  def __init__(self, hot_pixel_threshold: int = 3000,
               saturated_pixel_value: int = 15_000):
    self.hot_pixel_threshold = hot_pixel_threshold
    self.saturated_pixel_value = saturated_pixel_value

  def save_calibration_to_file(self, filename: str):
    np.savez(
        filepaths.calibration(filename),
        shape=self.shape,
        background=self.background,
        stdev=self.stdev,
        hot_pixel_indices=np.asarray(self.hot_pixel_indices),
        effective_exposures_s=np.asarray(self.effective_exposures_s),
    )

  def load_calibration_from_file(self, filename: str):
    data = np.load(filepaths.calibration(filename))

    self.shape = tuple(data['shape'])
    self.background = data['background']
    self.stdev = data['stdev']
    self.hot_pixel_indices = [(entry[0], entry[1]) for entry in
                              data['hot_pixel_indices']]
    self.effective_exposures_s = data['effective_exposures_s']

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

  def process_grey_frames(self, images: Sequence[image_loader.RawImage],
                          verbose: bool = False):
    def maybe_print(message: str = ''):
      if verbose:
        print(message)

    images = [
        self.remove_hot_pixels(self.subtract_background(image))
        for image in images
    ]
    stacks = split_into_exposure_stacks(images)

    maybe_print('Comparing expected vs actual ratios in pixel values for '
                'different exposures...')
    maybe_print()
    num_ratios = int(np.max([len(stack) for stack in stacks])) - 1
    actual_ratios = {}
    expected_ratios = [[] for _ in range(num_ratios)]
    for colour in image_loader.BAYER_MASK_OFFSET:
      actual_ratios[colour] = [[] for _ in range(num_ratios)]

      offset = image_loader.BAYER_MASK_OFFSET[colour]
      maybe_print(f'Bayer mask: {colour} {offset}')

      for stack_num, stack in enumerate(stacks):
        images = [image.raw_image[offset[0]::2, offset[1]::2]
                  for image in stack]
        invalid_pixels = []

        invalid = []
        for image in images:
          num_pixels = np.prod(image.shape)

          # Pixels that are too saturated or too dark are considered invalid.
          current_invalid_pixels = np.logical_or(image > 12e3, image < 40)
          invalid_pixels.append(current_invalid_pixels)

          # Do not use images that have too many invalid pixels.
          if np.sum(current_invalid_pixels) / num_pixels > 0.7:
            current_invalid = True
          else:
            current_invalid = False
          invalid.append(current_invalid)

        maybe_print(f'  Stack {stack_num:d}:')
        for ind in range(len(stack) - 1):
          if invalid[ind] or invalid[ind + 1]:
            continue
          actual_ratio = images[ind] / np.maximum(images[ind + 1], 1e-3)
          actual_ratio[invalid_pixels[ind]] = np.nan
          actual_ratio[invalid_pixels[ind + 1]] = np.nan
          actual_ratios[colour][ind].append(np.nanmean(actual_ratio))

          expected_ratio = (stack[ind].get_exposure().norm_exposure_s() /
                            stack[ind + 1].get_exposure().norm_exposure_s())
          expected_ratios[ind] = expected_ratio

          maybe_print(
              f'    Image {ind:d} vs {ind + 1:d}: '
              f'{expected_ratio:5.2f} (expected), '
              f'{np.nanmean(actual_ratio):5.2f} +/- '
              f'{np.nanstd(actual_ratio):4.2f} (actual)'
          )
        maybe_print()

    maybe_print('Summary of ratios:')
    for ind in range(len(expected_ratios)):
      reports = []
      for colour in image_loader.BAYER_MASK_OFFSET:
        current_ratios = actual_ratios[colour][ind]
        if len(current_ratios) == 1:
          reports.append(
              f'{np.mean(current_ratios):5.2f} +/- ____ ({colour})'
          )
        else:
          reports.append(
              f'{np.mean(current_ratios):5.2f} +/- '
              f'{np.std(current_ratios):4.2f} ({colour})'
          )
      report = ', '.join(reports)

      maybe_print(
          f'  Image {ind:d} vs {ind + 1:d}: '
          f'{expected_ratios[ind]:5.2f} (expected), ' + report
      )
    maybe_print()

    # Averages exposure ratios across colours.
    avg_ratios = []
    for ind in range(len(expected_ratios)):
      avg_ratio = np.mean(
          np.concatenate([
            actual_ratios[colour][ind] for colour in
            image_loader.BAYER_MASK_OFFSET
        ])
      )
      avg_ratios.append(avg_ratio)
    if len(avg_ratios) != len(constants.EXPOSURES) - 1:
      raise ValueError(
          f'Expected {len(constants.EXPOSURES) - 1} exposure ratios, but '
          f'actually have {len(avg_ratios)}'
      )

    # Computes equivalent exposure times relative to F1.0 and ISO100.
    self.effective_exposures_s = [constants.EXPOSURES[0].norm_exposure_s()]
    for avg_ratio in avg_ratios:
      self.effective_exposures_s.append(
          self.effective_exposures_s[-1] / avg_ratio
      )

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
    new_image.raw_image = _remove_specified_pixels(
        image.raw_image,
        self.hot_pixel_indices
    )
    return new_image

  def stack_hdr_image(self, images: Sequence[image_loader.RawImage],
                      smoothing_radius: float = 5) -> npt.NDArray:
    stack_shape = (len(images),) + images[0].raw_image.shape
    norm_images = np.zeros(stack_shape, dtype=float)
    weights = np.zeros(stack_shape, dtype=float)
    inv_stdev = np.zeros(stack_shape, dtype=float)

    for ind, image in enumerate(images):
      corrected_image = self.remove_hot_pixels(self.subtract_background(image))

      # Normalize image to an F-number of 1.0 and ISO 100.
      effective_exposure_s = self.effective_exposures_s[
          constants.EXPOSURES.index(image.get_exposure())
      ]
      scale_factor = 1 / effective_exposure_s
      norm_images[ind] = scale_factor * corrected_image.raw_image

      # Determines non-saturated subset of image and weights to use for merging.
      saturated = ndimage.maximum_filter(
          image.raw_image > self.saturated_pixel_value,
          size=3,
      )
      dist_to_saturated = ndimage.distance_transform_edt(
          np.logical_not(saturated))
      weights[ind] = np.minimum(1,  dist_to_saturated / smoothing_radius)

      # Determine inverse standard deviation per pixel.
      stdev = scale_factor * np.maximum(
          1e-2,
          self.stdev[_exposure_ind(image.get_exposure())]
      )
      inv_stdev[ind] = np.logical_not(saturated)  / stdev

    # Take weighted average of the best two exposures for each pixel.
    pixel_ranking = np.argsort(-inv_stdev, axis=0)
    weight = np.take_along_axis(weights, pixel_ranking[:1], axis=0)
    weighted_average = (
        weight * np.take_along_axis(norm_images, pixel_ranking[:1], axis=0) +
        (1 - weight) * np.take_along_axis(
            norm_images, pixel_ranking[1:2], axis=0)
    )[0]

    # Pixels with zero total weight are saturated at all exposures.
    # Remove these in the same way as we remove hot pixels.
    saturated_pixel_indices = np.argwhere(np.sum(weights, axis=0) == 0)
    return _remove_specified_pixels(weighted_average, saturated_pixel_indices)
