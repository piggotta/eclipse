from collections.abc import Callable, Sequence
import copy
import enum
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
import filepaths
import image_loader


@dataclasses.dataclass
class CroppedBwImageMetadata:
  index: int
  unix_time_s: float
  bayer_offset: tuple[int, int]
  orig_shape: tuple[int, int]
  cropped_shape: tuple[int, int]
  offset: tuple[int, int]


@dataclasses.dataclass
class EclipseTrack:
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


def save_track(filename: str, track: EclipseTrack):
  serialized = cattrs.unstructure(track)
  with open(filepaths.metadata(filename), 'w') as f:
    f.write(json.dumps(serialized))


def load_track(filename: str) -> EclipseTrack:
  with open(filepaths.metadata(filename), 'r') as f:
    serialized = json.loads(f.read())
  return cattrs.structure(serialized, EclipseTrack)


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


class EclipseTracker:
  """Generalized image tracker for eclipse images."""
  def __init__(
      self,
      renderer: Callable[[tuple[int, int], EclipseTrack], npt.NDArray]):
    self.renderer = renderer
    self.image_metadata = None
    self.bw_images = None

  def init(self,
           image_metadata: Sequence[CroppedBwImageMetadata],
           bw_images: npt.NDArray):
    self.image_metadata = list(image_metadata)
    self.bw_images = np.asarray(bw_images)

  def save_to_file(self, filename: str):
    dicts = [cattrs.unstructure(m) for m in self.image_metadata]
    with open(filepaths.metadata(filename), 'w') as f:
      f.write(json.dumps(dicts))

    np.savez(self.filepaths.image_stack(filename), bw_images=self.bw_images)

  def load_from_file(self, filename: str):
    with open(filepaths.metadata(filename), 'r') as f:
      dicts = json.loads(f.read())
    self.image_metadata = [cattrs.structure(d, CroppedBwImageMetadata) for d in dicts]

    data = np.load(self.filepaths.image_stack(filename))
    self.bw_images = data['bw_images']

  def plot_track(self,
                 track: EclipseTrack,
                 ind_to_plot: Sequence[int]):
    # Render only the images we want to plot.
    track = copy.deepcopy(track)
    ind_to_plot = np.asarray(ind_to_plot)
    track.unix_time_s = list(np.asarray(track.unix_time_s)[ind_to_plot])
    track.sun_centers = [tuple(entry) for entry in
                         np.asarray(track.sun_centers)[ind_to_plot, :]]

    image_shape = self.bw_images.shape[1:]
    new_images = self.renderer(image_shape, track)

    # Plot images.
    unix_time_s = [entry.unix_time_s for entry in self.image_metadata]
    photo_indices = np.asarray([entry.index for entry in self.image_metadata])
    for ind, time in enumerate(track.unix_time_s):
      global_ind = unix_time_s.index(time)
      _plot_image_diff(self.bw_images[global_ind], new_images[ind])
      plt.suptitle(f'IMG_{photo_indices[global_ind]}', fontsize=16)

  def optimize_track(self,
                     initial_track: EclipseTrack,
                     params_to_fit: list[str],
                     image_mask: npt.NDArray,
                     method: str = 'L-BFGS-B',
                     verbose: bool = False
                     ) -> EclipseTrack:
    if verbose:
      print('Optimizing track...')
      print('  Variables: ' + ', '.join(params_to_fit))
      print(f'  Number of images: {self.bw_images[image_mask, :, :].shape[0]}')
      print()

    # Select subset of images to fit.
    unix_time_s = np.asarray([entry.unix_time_s for entry in self.image_metadata])

    bw_images = self.bw_images[image_mask, :, :]
    unix_time_s = unix_time_s[image_mask]

    if len(initial_track.sun_centers) != bw_images.shape[0]:
      raise ValueError('Number of provided sun_centers does not match number '
                       'of images to optimize over.')

    num_images = bw_images.shape[0]
    image_shape = bw_images.shape[1:]

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
      return self.renderer(image_shape, track(x))

    def f_per_image(x):
      delta = (images(x) - bw_images)**2
      return np.sum(np.sum(delta, axis=2), axis=1) / np.prod(bw_images.shape)

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
                   initial_track: EclipseTrack,
                   image_mask: npt.NDArray,
                   ) -> EclipseTrack:
    """Aligns images to the provided track.

    If sun_centers is None, cross-correlations are used to provide the initial
    guess for alignment. Otherwise, sun_centers is used as the initial guess.
    """
    track = copy.deepcopy(initial_track)

    num_images = self.bw_images.shape[0]
    image_shape = self.bw_images.shape[1:]
    indices = np.arange(num_images)[image_mask]

    unix_time_s = [entry.unix_time_s for entry in self.image_metadata]
    track.unix_time_s = list(np.asarray(unix_time_s)[image_mask])

    if track.sun_centers is None:
      track.sun_centers = [(image_shape[0] // 2, image_shape[1] // 2)
                           ] * len(indices)
      initial_images = self.renderer(image_shape, track)

      print('Aligning images with cross-correlations...')
      for ind, global_ind in enumerate(indices):
        print(f'  {ind} / {len(indices)}')
        offset = _align_with_cross_correlation(
            self.bw_images[global_ind, :, :],
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

