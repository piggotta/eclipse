from collections.abc import Sequence
import copy
import dataclasses
import json
import os
import time

import cattrs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import optimize
from scipy import signal

import constants
import drawing
import eclipse_tracker
import filepaths
import image_loader


@dataclasses.dataclass
class PartialEclipseImageMetadata:
  common: eclipse_tracker.CroppedBwImageMetadata
  black_mean: float
  black_std: float
  black_threshold: float
  sun_mean: float


def _draw_partial_eclipses(image_shape: tuple[int, int],
                           track: eclipse_tracker.EclipseTrack) -> npt.NDArray:
  images = np.zeros((len(track.unix_time_s), *image_shape))
  for ind, time_s in enumerate(track.unix_time_s):
    # Draw sun in each frame.
    sun_center = track.sun_centers[ind]
    sun_image = drawing.draw_circle(image_shape, sun_center, track.sun_radius)

    # Draw moon in each frame.
    t = time_s - track.moon_zero_time_s
    moon_center = (
        track.moon_r0 + t * track.moon_dr_dt + t**2 * track.moon_d2r_dt2,
        track.moon_c0 + t * track.moon_dc_dt + t**2 * track.moon_d2c_dt2,
    )
    moon_image = drawing.draw_circle(
        image_shape,
        tuple(np.asarray(moon_center) + np.asarray(sun_center)),
        track.moon_radius
    )

    # Use moon image as mask for sun image.
    images[ind, :, :] = sun_image * (1 - moon_image)

  return images


@dataclasses.dataclass
class PartialEclipseTrackerOptions:
  guess_sun_radius: float = 0
  guess_moon_dr_dt: float = 0
  guess_moon_dc_dt: float = 0


class PartialEclipseTracker(eclipse_tracker.EclipseTracker):
  """Image tracker for partial eclipse images."""
  def __init__(self, options: PartialEclipseTrackerOptions):
    super().__init__(renderer=_draw_partial_eclipses)
    self.options = options

  def approx_unix_time_of_index(self, index: int) -> float:
    photo_indices = np.asarray([entry.index for entry in self.image_metadata])
    ind = np.argmin(np.abs(photo_indices - index))
    return self.image_metadata[ind].unix_time_s

  def approx_total_eclipse_start_unix_time_s(self) -> float:
    return self.approx_unix_time_of_index(constants.IND_FIRST_TOTAL)

  def approx_total_eclipse_end_unix_time_s(self) -> float:
    return self.approx_unix_time_of_index(constants.IND_LAST_TOTAL)

  def approx_partial_eclipse_start_unix_time_s(self) -> float:
    return self.approx_unix_time_of_index(constants.IND_FIRST_PARTIAL)

  def approx_partial_eclipse_end_unix_time_s(self) -> float:
    return self.approx_unix_time_of_index(constants.IND_LAST_PARTIAL)

  def fit_sun_radius(self) -> float:
    photo_indices = np.asarray([entry.index for entry in self.image_metadata])
    mask = np.logical_or(
        photo_indices < constants.IND_FIRST_PARTIAL - 1,
        photo_indices > constants.IND_LAST_PARTIAL + 1)

    unix_time_s = [entry.unix_time_s for entry in self.image_metadata]
    unix_time_s = list(np.asarray(unix_time_s)[mask])

    num_images = np.sum(mask)
    image_shape = self.bw_images.shape[1:]
    sun_centers = [(image_shape[0] // 2, image_shape[1] // 2)] * num_images

    initial_track = eclipse_tracker.EclipseTrack(
        sun_radius=self.options.guess_sun_radius,
        moon_radius=1,
        unix_time_s=unix_time_s,
        sun_centers=sun_centers,
        moon_zero_time_s=0,
        moon_r0=-1e4,
        moon_c0=-1e4,
        moon_dr_dt=0,
        moon_dc_dt=0,
        moon_d2r_dt2=0,
        moon_d2c_dt2=0,
    )

    track = self.optimize_track(
        initial_track,
        ['sun_radius', 'sun_centers'],
        image_mask=mask,
        verbose=True,
    )
    self.plot_track(track, [0])
    return track.sun_radius

  def track_sun_and_moon(self) -> eclipse_tracker.EclipseTrack:
    # Fit solar radius.
    sun_radius = self.fit_sun_radius()

    # Get general information.
    num_images = self.bw_images.shape[0]
    image_shape = self.bw_images.shape[1:]

    unix_time_s = np.asarray(
        [entry.unix_time_s for entry in self.image_metadata])
    moon_zero_time_s = 0.5 * (
        self.approx_total_eclipse_start_unix_time_s() +
        self.approx_total_eclipse_end_unix_time_s())

    partial_eclipse_time_s = (
        self.approx_partial_eclipse_end_unix_time_s() -
        self.approx_partial_eclipse_start_unix_time_s()
    )
    exclude_time_length_s = 0.1 * partial_eclipse_time_s
    exclude_time_start_s = moon_zero_time_s - exclude_time_length_s
    exclude_time_end_s = moon_zero_time_s + exclude_time_length_s

    # Initial fit to small subset of images.
    indices = np.asarray(
        np.linspace(0, self.bw_images.shape[0] - 1, 60),
        dtype=int
    )
    mask = np.logical_or(
        unix_time_s[indices] < exclude_time_start_s,
        unix_time_s[indices] > exclude_time_end_s
    )
    indices = indices[mask]

    track = eclipse_tracker.EclipseTrack(
        sun_radius=sun_radius,
        moon_radius=sun_radius,
        unix_time_s=None,
        sun_centers=None,
        moon_zero_time_s=moon_zero_time_s,
        moon_r0=0,
        moon_c0=0,
        moon_dr_dt=self.options.guess_moon_dr_dt,
        moon_dc_dt=self.options.guess_moon_dc_dt,
        moon_d2r_dt2=0,
        moon_d2c_dt2=0,
    )

    for _ in range(2):
      track = self.align_images(track, indices)

      track = self.optimize_track(
          track,
          ['moon_radius', 'moon_r0', 'moon_c0', 'moon_dr_dt', 'moon_dc_dt'],
          image_mask=indices,
          method='BFGS',
          verbose=True,
      )

    for _ in range(2):
      track = self.align_images(track, indices)

      track = self.optimize_track(
          track,
          ['moon_radius', 'moon_r0', 'moon_c0', 'moon_dr_dt', 'moon_dc_dt',
           'moon_d2r_dt2', 'moon_d2c_dt2'],
          image_mask=indices,
          method='BFGS',
          verbose=True,
      )

    # Fit image offset for each partial eclipse image.
    track.sun_centers = None
    track = self.align_images(track, np.arange(num_images))
    return track


@dataclasses.dataclass
class PreprocessedImage:
  metadata: PartialEclipseImageMetadata
  cropped_image: npt.NDArray
  is_sun: npt.NDArray


def _preprocess_image(
    image: image_loader.RawImage,
    cropped_shape: tuple[int, int],
    bayer_offset: tuple[int, int] = image_loader.BAYER_MASK_OFFSET['red']
    ) -> PreprocessedImage:
  # Compute black level of image.
  is_black = image.bw_image < np.quantile(image.bw_image, 0.90)
  black_mean = np.mean(image.bw_image[is_black])
  black_std = np.std(image.bw_image[is_black])
  black_threshold = black_mean + 5 * black_std

  # Determine mean sun level. If we don't have enough pixels (i.e. very thin
  # partial sun), use a manually chosen value instead.
  sun_mask = image.bw_image > black_threshold
  if np.sum(sun_mask) > 4000:
    sun_mean = np.mean(image.bw_image[sun_mask])
  else:
    sun_mean = black_mean + 600

  # Adjust the white and black levels to generate a high-contrast greyscale
  # image of the sun.
  delta = sun_mean - black_mean
  black_level = black_mean + 0.1 * delta
  white_level = sun_mean - 0.4 * delta
  scaled = (image.bw_image - black_level) / (white_level - black_level)
  is_sun = np.minimum( np.maximum(scaled, 0), 1)

  # Determine rectangular subset of image containing sun.
  offset = [500, 1000]
  rows = np.arange(image.bw_image.shape[0])
  cols = np.arange(image.bw_image.shape[1])
  rows, cols = np.meshgrid(rows, cols, indexing='ij')

  sun_mask = is_sun > 0.2
  min_row = np.quantile(rows[sun_mask], 0.1)
  max_row = np.quantile(rows[sun_mask], 0.9)
  min_col = np.quantile(cols[sun_mask], 0.1)
  max_col = np.quantile(cols[sun_mask], 0.9)

  center = np.asarray((
      int(round((min_row + max_row) / 2)),
      int(round((min_col + max_col) / 2))))
  offset = center - np.asarray(cropped_shape) // 2

  # Crop image to region containing sun.
  cropped_image = image.bw_image[
      offset[0]:offset[0] + cropped_shape[0],
      offset[1]:offset[1] + cropped_shape[1]
  ]
  is_sun = is_sun[
      offset[0]:offset[0] + cropped_shape[0],
      offset[1]:offset[1] + cropped_shape[1]
  ]

  common = eclipse_tracker.CroppedBwImageMetadata(
      index=image.index,
      unix_time_s=image.time.timestamp(),
      bayer_offset=bayer_offset,
      orig_shape= image.bw_image.shape,
      cropped_shape=tuple([int(n) for n in cropped_shape]),
      offset=tuple([int(n) for n in offset]),
  )
  metadata = PartialEclipseImageMetadata(
      common=common,
      black_mean=black_mean,
      black_std=black_std,
      black_threshold=black_threshold,
      sun_mean=sun_mean,
  )
  return PreprocessedImage(
      metadata=metadata,
      cropped_image=cropped_image,
      is_sun=is_sun,
  )


def is_partial_eclipse(image: image_loader.RawImage):
  # Filter based on image index.
  ind = image.index
  if ind < constants.IND_FIRST_SUN or ind > constants.IND_LAST_SUN:
    return False
  if ind >= constants.IND_FIRST_TOTAL and ind <= constants.IND_LAST_TOTAL:
    return False

  # Filter based on image exposure settings.
  if image.iso != 100 or image.exposure_s != 0.0005 or image.f_number != 8:
    return False
  return True


class PartialEclipsePreprocessor:
  def __init__(self):
    self.image_metadata = None
    self.cropped_images = None
    self.is_sun = None

  def save_to_file(self, filename: str):
    dicts = [cattrs.unstructure(m) for m in self.image_metadata]
    with open(filepaths.metadata(filename), 'w') as f:
      f.write(json.dumps(dicts))

    np.savez(filepaths.cropped_partials(filename),
             cropped_images=self.cropped_images,
             is_sun=self.is_sun)

  def load_from_file(self, filename: str):
    with open(filepaths.metadata(filename), 'r') as f:
      dicts = json.loads(f.read())
    self.image_metadata = [
        cattrs.structure(d, PartialEclipseImageMetadata) for d in dicts
    ]

    data = np.load(filepaths.cropped_partials(filename))
    self.cropped_images = data['cropped_images']
    self.is_sun = data['is_sun']

  def preprocess_images(self,
                        cropped_shape: tuple[int, int] = (500, 500),
                        start_ind: int = constants.IND_FIRST_SUN,
                        max_images: int = None):
    # Determine image filepaths.
    max_index = constants.IND_LAST_SUN
    if max_images is not None:
      max_index = min(max_index, start_ind + max_images)
    indices = list(range(start_ind, max_index))

    # First pass to determine which images are valid partial eclipse images.
    print('Counting number of partial eclipse images...')
    valid_indices = []
    is_first_partial = True
    for index in indices:
      try:
        attr = image_loader.maybe_read_image_attributes_by_index(index)
      except ValueError:
        continue

      filename = os.path.split(attr.filepath)[1]

      if is_partial_eclipse(attr):
        valid_indices.append(index)
        if is_first_partial:
          is_first_partial = False
          first_image = image_loader.maybe_read_image_by_index(index)
        print(f'  {filename:s}: partial eclipse')
      else:
        print(f'  {filename:s}: invalid')
    num_partials = len(valid_indices)
    print(f'Number of partial eclipse images: {num_partials:d}')
    print()

    # Load images from file.
    print('Processing images...')
    self.cropped_images = np.zeros((num_partials, *cropped_shape))
    self.is_sun = np.zeros((num_partials, *cropped_shape))
    self.image_metadata = []
    for ind, index in enumerate(valid_indices):
      image = image_loader.maybe_read_image_by_index(index)
      filename = os.path.split(image.filepath)[1]
      print(f'  {filename:s}')

      result = _preprocess_image(image, cropped_shape)
      self.image_metadata.append(result.metadata)
      self.cropped_images[ind, :, :] = result.cropped_image
      self.is_sun[ind, :, :] = result.is_sun

      del image
      del result

  def init_partial_eclipse_tracker(self, tracker: PartialEclipseTracker):
    tracker.init(
        image_metadata=[entry.common for entry in self.image_metadata],
        bw_images=self.is_sun,
    )

