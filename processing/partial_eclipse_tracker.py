from collections.abc import Sequence
import copy
import dataclasses
import glob
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
  unix_time_s: list[float] | None

  sun_radius: float
  moon_radius: float
  sun_centers: list[tuple[float, float]] | None

  # Moon position is relative to sun.
  moon_zero_time_s: float
  moon_r0: float
  moon_c0: float
  moon_dr_dt: float
  moon_dc_dt: float
  moon_d2r_dt2: float
  moon_d2c_dt2: float

def _metadata_filepath(filename):
  return os.path.join(constants.OUTPUTS_PATH, filename + '.cattr')


def save_track(filename: str, track: PartialEclipseTrack):
  serialized = cattrs.unstructure(track)
  with open(_metadata_filepath(filename), 'w') as f:
    f.write(json.dumps(dicts))


def load_track(filename: str) -> PartialEclipseTrack:
  with open(_metadata_filepath(filename), 'r') as f:
    serialized = json.loads(f.read())
  return cattrs.structure(serialized, PartialEclipseTrack)


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


class PartialEclipseTracker:

  def __init__(self, options: PartialEclipseTrackerOptions):
    self.options = options
    self.image_metadata = []
    self.cropped_images = None
    self.is_sun = None


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
    with open(_metadata_filepath(filename), 'w') as f:
      f.write(json.dumps(dicts))

    np.savez(self._image_npz_filepath(filename),
             cropped_images=self.cropped_images,
             is_sun=self.is_sun)

  def load_preprocess_from_file(self, filename: str):
    with open(_metadata_filepath(filename), 'r') as f:
      dicts = json.loads(f.read())
    self.image_metadata = [cattrs.structure(d, ImageMetadata) for d in dicts]

    data = np.load(self._image_npz_filepath(filename))
    self.cropped_images = data['cropped_images']
    self.is_sun = data['is_sun']

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

  def plot_track(self,
                 track: PartialEclipseTrack,
                 ind_to_plot: Sequence[int]):
    # Render only the images we want to plot.
    track = copy.deepcopy(track)
    ind_to_plot = np.asarray(ind_to_plot)
    track.unix_time_s = list(np.asarray(track.unix_time_s)[ind_to_plot])
    track.sun_centers = [tuple(entry) for entry in
                         np.asarray(track.sun_centers)[ind_to_plot, :]]

    image_shape = self.is_sun.shape[1:]
    new_images = _draw_partial_eclipses(image_shape, track)

    # Plot images.
    unix_time_s = [entry.unix_time_s for entry in self.image_metadata]
    photo_indices = np.asarray([entry.index for entry in self.image_metadata])
    for ind, time in enumerate(track.unix_time_s):
      global_ind = unix_time_s.index(time)
      _plot_image_diff(self.is_sun[global_ind], new_images[ind])
      plt.suptitle(f'IMG_{photo_indices[global_ind]}', fontsize=16)

  def optimize_track(self,
                     initial_track: PartialEclipseTrack,
                     params_to_fit: list[str],
                     image_mask: npt.NDArray,
                     method: str = 'L-BFGS-B',
                     verbose: bool = False
                     ) -> PartialEclipseTrack:
    if verbose:
      print('Optimizing track...')
      print('  Variables: ' + ', '.join(params_to_fit))
      print(f'  Number of images: {self.is_sun[image_mask, :, :].shape[0]}')
      print()

    # Select subset of images to fit.
    unix_time_s = np.asarray([entry.unix_time_s for entry in self.image_metadata])
    photo_indices = np.asarray([entry.index for entry in self.image_metadata])

    is_sun = self.is_sun[image_mask, :, :]
    unix_time_s = unix_time_s[image_mask]
    photo_indices = photo_indices[image_mask]

    if len(initial_track.sun_centers) != is_sun.shape[0]:
      raise ValueError('Number of provided sun_centers does not match number '
                       'of images to optimize over.')

    num_images = is_sun.shape[0]
    image_shape = is_sun.shape[1:]

    # Order in which we store global parameters in our state vector x.
    all_global_params = [
        'sun_radius',
        'moon_radius',
        'moon_r0',
        'moon_c0',
        'moon_dr_dt',
        'moon_dc_dt',
        'moon_d2r_dt2',
        'moon_d2c_dt2',
    ]

    # Determine which parameters to optimize over.
    for param in params_to_fit:
      if param not in all_global_params and param != 'sun_centers':
        raise ValueError(f'Unexpected parameter to fit: {param}')

    global_params = []
    for param in all_global_params:
      if param in params_to_fit:
        global_params.append(param)

    if 'sun_centers' in params_to_fit:
      fit_sun_centers = True
    else:
      fit_sun_centers = False

    def track(x):
      current_track = copy.deepcopy(initial_track)
      for ind, param in enumerate(global_params):
        setattr(current_track, param, x[ind])

      if fit_sun_centers:
        center_row = x[len(global_params):len(global_params) + num_images]
        center_col = x[len(global_params) + num_images:]
        current_track.sun_centers = [(center_row[ind], center_col[ind])
                                       for ind in range(num_images)]
      return current_track

    def images(x):
      return _draw_partial_eclipses(image_shape, track(x))

    def f_per_image(x):
      delta = (images(x) - is_sun)**2
      return np.sum(np.sum(delta, axis=2), axis=1) / np.prod(is_sun.shape)

    def f(x):
      return np.sum(f_per_image(x))

    def df_dx(x):
      all_df_dx = []

      f_per_image_x = f_per_image(x)
      f_x = np.sum(f_per_image_x)

      delta_per_global_param = {
        'sun_radius': 1e-3,
        'moon_radius': 1e-3,
        'sun_centers': 1e-3,
        'moon_r0': 1e-3,
        'moon_c0': 1e-3,
        'moon_dr_dt': 1e-7,
        'moon_dc_dt': 1e-7,
        'moon_d2r_dt2': 1e-9,
        'moon_d2c_dt2': 1e-9,
      }

      for ind, param in enumerate(global_params):
        delta = delta_per_global_param[param]
        delta_x = np.zeros(x.shape, dtype=float)
        delta_x[ind] = delta
        df_dp = (f(x + delta_x) - f_x) / delta
        all_df_dx.append([df_dp])

      if fit_sun_centers:
        delta_cr = 1e-2
        delta_x = np.zeros(x.shape, dtype=float)
        delta_x[len(global_params):num_images + len(global_params)] = delta_cr
        df_dcr = (f_per_image(x + delta_x) - f_per_image_x) / delta_cr
        all_df_dx.append(df_dcr)

        delta_cc = 1e-2
        delta_x = np.zeros(x.shape, dtype=float)
        delta_x[num_images + len(global_params):] = delta_cc
        df_dcc = (f_per_image(x + delta_x) - f_per_image_x) / delta_cc
        all_df_dx.append(df_dcc)

      return np.concatenate(all_df_dx)

    # Get initial starting point for state vector.
    x0 = []
    for param in global_params:
      x0.append([getattr(initial_track, param)])

    if fit_sun_centers:
      center_row = [entry[0] for entry in initial_track.sun_centers]
      center_col = [entry[1] for entry in initial_track.sun_centers]
      x0 += [center_row, center_col]
    x0 = np.concatenate(x0)

    # Find optimal solution.
    start_time_s = time.time()
    res = optimize.minimize(f, x0, jac=df_dx,
                            method=method,
                            options={'disp': verbose})
    elapsed_time_s = time.time() - start_time_s

    # Print initial and final function values.
    if verbose:
      print(f'Initial f(x): {f(x0):e}')
      print(f'Final f(x): {res.fun:e}')
      print(f'Elapsed time: {elapsed_time_s:.3f} s')
      print()

    return track(res.x)

  def align_images(self,
                   initial_track: PartialEclipseTrack,
                   image_mask: npt.NDArray,
                   ) -> PartialEclipseTrack:
    """Aligns images to the provided track.

    If sun_centers is None, cross-correlations are used to provide the initial
    guess for alignment. Otherwise, sun_centers is used as the initial guess.
    """
    track = copy.deepcopy(initial_track)

    num_images = self.is_sun.shape[0]
    image_shape = self.is_sun.shape[1:]
    indices = np.arange(num_images)[image_mask]

    unix_time_s = [entry.unix_time_s for entry in self.image_metadata]
    track.unix_time_s = list(np.asarray(unix_time_s)[image_mask])

    if track.sun_centers is None:
      track.sun_centers = [(image_shape[0] // 2, image_shape[1] // 2)
                           ] * len(indices)
      initial_images = _draw_partial_eclipses(image_shape, track)

      print('Aligning images with cross-correlations...')
      for ind, global_ind in enumerate(indices):
        print(f'  {ind} / {len(indices)}')
        offset = _align_with_cross_correlation(
            self.is_sun[global_ind, :, :],
            initial_images[ind, :, :]
        )
        track.sun_centers[ind] = tuple(
            np.asarray(track.sun_centers[ind]) + np.asarray(offset)
        )
      print()

    if len(track.sun_centers) != len(track.unix_time_s):
      raise ValueError('Number of sun_centers entries should be equal to the '
                       'number of images selected by image_mask.')

    print('Optimizing image alignment...')
    for ind, global_ind in enumerate(indices):
      print(f'  {ind} / {len(indices)}')

      current_track = copy.copy(track)
      current_track.unix_time_s = [track.unix_time_s[ind]]
      current_track.sun_centers = [track.sun_centers[ind]]

      current_track = self.optimize_track(
          current_track,
          ['sun_centers'],
          image_mask=[global_ind],
          method='L-BFGS-B',
      )

      track.sun_centers[ind] = current_track.sun_centers[0]
    print()

    return track

  def fit_sun_radius(self) -> float:
    photo_indices = np.asarray([entry.index for entry in self.image_metadata])
    mask = np.logical_or(
        photo_indices < constants.IND_FIRST_PARTIAL - 1,
        photo_indices > constants.IND_LAST_PARTIAL + 1)

    unix_time_s = [entry.unix_time_s for entry in self.image_metadata]
    unix_time_s = list(np.asarray(unix_time_s)[mask])

    num_images = np.sum(mask)
    image_shape = self.is_sun.shape[1:]
    sun_centers = [(image_shape[0] // 2, image_shape[1] // 2)] * num_images

    initial_track = PartialEclipseTrack(
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

  def track_sun_and_moon(self) -> PartialEclipseTrack:
    # Fit solar radius.
    sun_radius = self.fit_sun_radius()

    # Get general information.
    num_images = self.is_sun.shape[0]
    image_shape = self.is_sun.shape[1:]

    unix_time_s = np.asarray(
        [entry.unix_time_s for entry in self.image_metadata])
    moon_zero_time_s = 0.5 * (
        self.approx_total_eclipse_start_unix_time_s() +
        self.approx_total_eclipse_end_unix_time_s())

    partial_eclipse_time_s = (
        self.approx_partial_eclipse_end_unix_time_s() -
        self.approx_partial_eclipse_start_unix_time_s()
    )
    exclude_time_length_s = 0.2 * partial_eclipse_time_s
    exclude_time_start_s = moon_zero_time_s - exclude_time_length_s
    exclude_time_end_s = moon_zero_time_s + exclude_time_length_s

    # Initial fit to small subset of images.
    indices = np.asarray(
        np.linspace(0, self.is_sun.shape[0] - 1, 60),
        dtype=int
    )
    mask = np.logical_or(
        unix_time_s[indices] < exclude_time_start_s,
        unix_time_s[indices] > exclude_time_end_s
    )
    indices = indices[mask]

    track = PartialEclipseTrack(
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
