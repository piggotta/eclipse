import matplotlib.pyplot as plt

import dataclasses
import glob
import json
import os

import cattrs
import numpy as np
import numpy.typing as npt
from scipy import optimize
from scipy import signal

import constants
import drawing
import image_loader


@dataclasses.dataclass
class ImageMetadata:
  index: int
  unix_time_s: float
  black_mean: float
  black_std: float
  black_threshold: float
  sun_mean: float
  orig_shape: tuple[int, int]
  cropped_shape: tuple[int, int]
  offset: tuple[int, int]


@dataclasses.dataclass
class PreprocessedImage:
  metadata: ImageMetadata
  cropped_image: npt.NDArray
  is_sun: npt.NDArray


def _preprocess_image(image: image_loader.RawImage,
                      cropped_shape: tuple[int, int]
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

  # Generate a soft thresholded image of the sun.
  delta = 0.1 * (sun_mean - black_mean)
  threshold = black_mean + 3 * delta
  is_sun = 0.5 * (1 + np.tanh((image.bw_image  - threshold) / delta))

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

  metadata = ImageMetadata(
      index= image.index,
      unix_time_s = image.time.timestamp(),
      black_mean=black_mean,
      black_std=black_std,
      black_threshold=black_threshold,
      sun_mean=sun_mean,
      orig_shape= image.bw_image.shape,
      cropped_shape=tuple([int(n) for n in cropped_shape]),
      offset=tuple([int(n) for n in offset]))

  return PreprocessedImage(
      metadata=metadata,
      cropped_image=cropped_image,
      is_sun=is_sun)


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


@dataclasses.dataclass
class PartialEclipseTrack:
  sun_radius: float
  moon_radius: float
  unix_time_s: list[float]
  sun_centers: list[tuple[float, float]]

  # Moon position is relative to sun.
  moon_zero_time_s: float
  moon_r0: float
  moon_c0: float
  moon_dr_dt: float
  moon_dc_dt: float
  moon_dr2_dt2: float
  moon_dc2_dt2: float


def _draw_partial_eclipses(image_shape: tuple[int, int],
                           track: PartialEclipseTrack) -> npt.NDArray:
  images = np.zeros((len(track.unix_time_s), *image_shape))
  for ind, time_s in enumerate(track.unix_time_s):
    # Draw sun in each frame.
    sun_center = track.sun_centers[ind]
    sun_image = drawing.draw_circle(image_shape, sun_center, track.sun_radius)

    # Draw moon in each frame.
    t = time_s - track.moon_zero_time_s
    moon_center = (
        track.moon_r0 + t * track.moon_dr_dt + t**2 * track.moon_dr2_dt2,
        track.moon_c0 + t * track.moon_dc_dt + t**2 * track.moon_dc2_dt2,
    )
    moon_image = drawing.draw_circle(
        image_shape,
        tuple(np.asarray(moon_center) + np.asarray(sun_center)),
        track.moon_radius
    )

    # Use moon image as mask for sun image.
    images[ind, :, :] = sun_image * (1 - moon_image)

  return images


def _align_with_cross_correlation(image_orig, image_new
                                  ) -> tuple[float, float]:
  corr = signal.correlate(image_orig, image_new, method='fft')
  opt_coord = np.asarray(np.unravel_index(np.argmax(corr), corr.shape))
  center_coord = np.asarray(corr.shape) // 2
  return tuple(opt_coord - center_coord)


def _plot_image_diff(image_orig, image_new):
  fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(2 * 6.4, 4.8))

  pcm = ax[0].imshow(image_orig, cmap='gray')
  fig.colorbar(pcm, ax=ax[0])
  ax[0].set_title('Original')

  delta = image_new - image_orig
  delta_lim = np.max(np.abs(delta))
  pcm = ax[1].imshow(delta, vmin=-delta_lim, vmax=delta_lim, cmap='seismic')
  fig.colorbar(pcm, ax=ax[1])
  ax[1].set_title('Difference')


@dataclasses.dataclass
class PartialEclipseTrackerOptions:
  guess_sun_radius: float = 0
  guess_moon_dr_dt: float = 0
  guess_moon_dc_dt: float = 0

  show_plots: bool = True
  verbose: bool = True
  decimate_for_debug: bool = False


class PartialEclipseTracker:

  def __init__(self, options: PartialEclipseTrackerOptions):
    self.options = options
    self.image_metadata = []
    self.cropped_images = None
    self.is_sun = None

  def _metadata_filepath(self, filename):
    return os.path.join(constants.OUTPUTS_PATH, filename + '.cattr')

  def _image_npz_filepath(self, filename):
    return os.path.join(constants.OUTPUTS_PATH, filename + '.npz')

  def preprocess_images(self,
                        cropped_shape: tuple[int, int] = (500, 500),
                        start_ind: int = constants.IND_FIRST_SUN,
                        max_images: int = None):
    # Determine image filepaths.
    max_index = constants.IND_LAST_SUN
    if max_images is not None:
      max_index = min(max_index, start_ind + max_images)
    filenames  = [f'IMG_{ind:04d}.CR2' for ind in
                  range(start_ind, max_index)]
    filepaths = [os.path.join(constants.PHOTOS_PATH, filename) for filename
                 in filenames]

    # First pass to determine which images are valid partial eclipse images.
    print('Counting number of partial eclipse images...')
    valid_filepaths = []
    is_first_partial = True
    for filepath in filepaths:
      attr = image_loader.read_attributes(filepath)
      filename = os.path.split(filepath)[1]

      if is_partial_eclipse(attr):
        valid_filepaths.append(filepath)
        if is_first_partial:
          is_first_partial = False
          first_image = image_loader.read_image(filepath)
        print(f'  {filename:s}: partial eclipse')
      else:
        print(f'  {filename:s}: invalid')
    num_partials = len(valid_filepaths)
    print(f'Number of partial eclipse images: {num_partials:d}')
    print()

    # Load images from file.
    print('Processing images...')
    self.cropped_images = np.zeros((num_partials, *cropped_shape))
    self.is_sun = np.zeros((num_partials, *cropped_shape))
    for ind, filepath in enumerate(valid_filepaths):
      filename = os.path.split(filepath)[1]
      print(f'  {filename:s}')

      image = image_loader.read_image(filepath)
      result = _preprocess_image(image, cropped_shape)
      self.image_metadata.append(result.metadata)
      self.cropped_images[ind, :, :] = result.cropped_image
      self.is_sun[ind, :, :] = result.is_sun

      del image
      del result

  def save_preprocess_to_file(self, filename: str):
    dicts = [cattrs.unstructure(m) for m in self.image_metadata]
    with open(self._metadata_filepath(filename), 'w') as f:
      f.write(json.dumps(dicts))

    np.savez(self._image_npz_filepath(filename),
             cropped_images=self.cropped_images,
             is_sun=self.is_sun)

  def load_preprocess_from_file(self, filename: str):
    with open(self._metadata_filepath(filename), 'r') as f:
      dicts = json.loads(f.read())
    self.image_metadata = [cattrs.structure(d, ImageMetadata) for d in dicts]

    data = np.load(self._image_npz_filepath(filename))
    self.cropped_images = data['cropped_images']
    self.is_sun = data['is_sun']

  def approx_total_eclipse_start_unix_time_s(self) -> float:
    photo_indices = np.asarray([entry.index for entry in self.image_metadata])
    ind = np.argmin(np.abs(photo_indices - constants.IND_FIRST_TOTAL))
    return self.image_metadata[ind].unix_time_s

  def approx_total_eclipse_end_unix_time_s(self) -> float:
    photo_indices = np.asarray([entry.index for entry in self.image_metadata])
    ind = np.argmin(np.abs(photo_indices - constants.IND_LAST_TOTAL))
    return self.image_metadata[ind].unix_time_s

  def fit_sun_radius(self) -> float:
    photo_indices = np.asarray([entry.index for entry in self.image_metadata])
    mask = np.logical_or(
        photo_indices < constants.IND_FIRST_PARTIAL - 1,
        photo_indices > constants.IND_LAST_PARTIAL + 1)
    full_sun_images = self.is_sun[mask, :, :]

    num_images = full_sun_images.shape[0]
    image_shape = self.is_sun.shape[1:]

    def images(x):
      sun_radius = x[0]
      center_row = x[1:num_images + 1]
      center_col = x[num_images + 1:]
      sun_centers = [(center_row[ind], center_col[ind]) for ind in
                     range(num_images)]

      track = PartialEclipseTrack(
          sun_radius=sun_radius,
          moon_radius=1,
          unix_time_s=np.zeros(num_images),
          sun_centers=sun_centers,
          moon_zero_time_s=0,
          moon_r0=-1e4,
          moon_c0=-1e4,
          moon_dr_dt=0,
          moon_dc_dt=0,
          moon_dr2_dt2=0,
          moon_dc2_dt2=0,
      )
      return _draw_partial_eclipses(image_shape, track)

    def f_per_image(x):
      delta = (images(x) - full_sun_images)**2
      return np.sum(np.sum(delta, axis=2), axis=1) / np.prod(full_sun_images.shape)

    def f(x):
      return np.sum(f_per_image(x))

    def df_dx(x):
      f_per_image_x = f_per_image(x)
      f_x = np.sum(f_per_image_x)

      delta_r = 1e-2
      delta_x = np.zeros(x.shape, dtype=float)
      delta_x[0] = delta_r
      df_dr = (f(x + delta_x) - f_x) / delta_r

      delta_cr = 1e-2
      delta_x = np.zeros(x.shape, dtype=float)
      delta_x[1:num_images + 1] = delta_cr
      df_dcr = (f_per_image(x + delta_x) - f_per_image_x) / delta_cr

      delta_cc = 1e-2
      delta_x = np.zeros(x.shape, dtype=float)
      delta_x[num_images + 1:] = delta_cc
      df_dcc = (f_per_image(x + delta_x) - f_per_image_x) / delta_cc

      return np.concatenate(((df_dr,), df_dcr, df_dcc))

    x0 = np.concatenate(((self.options.guess_sun_radius,),
                         (image_shape[0] // 2) * np.ones(num_images),
                         (image_shape[1] // 2) * np.ones(num_images)))

    res = optimize.minimize(f, x0, jac=df_dx, method='L-BFGS-B',
                            options={'disp': self.options.verbose})
    sun_radius = res.x[0]

    if self.options.show_plots:
      ind = 0
      _plot_image_diff(full_sun_images[ind, :, :], images(res.x)[ind, :, :])
      plt.suptitle(f'IMG_{photo_indices[ind]}', fontsize=16)

    return sun_radius

  def track_sun_and_moon(self) -> PartialEclipseTrack:
    is_sun = self.is_sun
    unix_time_s = [entry.unix_time_s for entry in self.image_metadata]
    photo_indices = [entry.index for entry in self.image_metadata]

    if self.options.decimate_for_debug:
      stride = 20
      is_sun = self.is_sun[::stride, :, :]
      unix_time_s = unix_time_s[::stride]
      photo_indices = photo_indices[::stride]

    num_images = is_sun.shape[0]
    image_shape = is_sun.shape[1:]

    # Determine fixed parameters.
    sun_radius = self.fit_sun_radius()
    moon_zero_time_s = 0.5 * (
        self.approx_total_eclipse_start_unix_time_s() +
        self.approx_total_eclipse_end_unix_time_s())

    def track(x):
      center_row = x[7:num_images + 7]
      center_col = x[num_images + 7:]
      sun_centers = [(center_row[ind], center_col[ind]) for ind in
                     range(num_images)]

      return PartialEclipseTrack(
          sun_radius=sun_radius,
          moon_radius=x[0],
          unix_time_s=unix_time_s,
          sun_centers=sun_centers,
          moon_zero_time_s=moon_zero_time_s,
          moon_r0=x[1],
          moon_c0=x[2],
          moon_dr_dt=x[3],
          moon_dc_dt=x[4],
          moon_dr2_dt2=0 * x[5],
          moon_dc2_dt2=0 * x[6],
      )

    def images(x):
      return _draw_partial_eclipses(image_shape, track(x))

    def f_per_image(x):
      delta = (images(x) - is_sun)**2
      return np.sum(np.sum(delta, axis=2), axis=1) / np.prod(is_sun.shape)

    def f(x):
      return np.sum(f_per_image(x))

    def df_dx(x):
      f_per_image_x = f_per_image(x)
      f_x = np.sum(f_per_image_x)

      df_dp = np.zeros(7)
      delta_per_global_param = (1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-8, 1e-8)
      for ind, delta in enumerate(delta_per_global_param):
        delta_x = np.zeros(x.shape, dtype=float)
        delta_x[ind] = delta
        df_dp[ind] = (f(x + delta_x) - f_x) / delta

      delta_cr = 1e-2
      delta_x = np.zeros(x.shape, dtype=float)
      delta_x[1:num_images + 1] = delta_cr
      df_dcr = (f_per_image(x + delta_x) - f_per_image_x) / delta_cr

      delta_cc = 1e-2
      delta_x = np.zeros(x.shape, dtype=float)
      delta_x[num_images + 1:] = delta_cc
      df_dcc = (f_per_image(x + delta_x) - f_per_image_x) / delta_cc

      return np.concatenate((df_dp, df_dcr, df_dcc))

    # Initial guess for global parameters.
    guess_global_params = (sun_radius,
                           0,
                           0,
                           self.options.guess_moon_dr_dt,
                           self.options.guess_moon_dc_dt,
                           0,
                           0)

    # Align each image using cross-correlation to generate starting guess.
    sun_centers_r = (image_shape[0] // 2) * np.ones(num_images)
    sun_centers_c = (image_shape[1] // 2) * np.ones(num_images)
    x0 = np.concatenate((guess_global_params,
                         sun_centers_r,
                         sun_centers_c))
    initial_images = images(x0)

    for ind in range(num_images):
      offsets = _align_with_cross_correlation(is_sun[ind, :, :],
                                              initial_images[ind, :, :])
      sun_centers_r[ind] += offsets[0]
      sun_centers_c[ind] += offsets[1]
    x1 = np.concatenate((guess_global_params,
                         sun_centers_r,
                         sun_centers_c))

    if False:
      for ind in range(num_images):
        _plot_image_diff(is_sun[ind, :, :], images(x1)[ind, :, :])
        plt.suptitle(f'IMG_{photo_indices[ind]}', fontsize=16)
      return

    res = optimize.minimize(f, x1, jac=df_dx,
                            method='L-BFGS-B',
                            #method='BFGS',
                            options={'disp': self.options.verbose})
    print(res.x[:7])

    if self.options.show_plots:
      for ind in range(num_images):
        _plot_image_diff(is_sun[ind, :, :], images(res.x)[ind, :, :])
        plt.suptitle(f'IMG_{photo_indices[ind]}', fontsize=16)

    return track(res.x)
