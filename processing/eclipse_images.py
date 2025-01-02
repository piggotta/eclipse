from collections.abc import Sequence
import dataclasses
import enum
import json
import gc
import os

import cattrs
import colour_demosaicing
import numpy as np
import numpy.typing as npt
from scipy import special
from scipy import ndimage

import constants
import eclipse_image_loader
import eclipse_tracker
import filepaths
import filtering
import image_loader
import partial_eclipse_tracker
import raw_processor
import total_eclipse_tracker


# TODO: delete me
import matplotlib.pyplot as plt


# Internal constants.
_CROP_PIXELS = 200


class ImageType(enum.Enum):
  RAW = 0
  HDR = 1
  EMULATED = 2


@dataclasses.dataclass
class ImageMetadata:
  index: int
  unix_time_s: float
  image_type: ImageType
  sun_center: tuple[float, float]


@dataclasses.dataclass
class EclipseSequence:
  """Eclipse track information for full raw images."""
  image_metadata: list[ImageMetadata]

  sun_radius: float
  moon_radius: float

  # Moon position is relative to sun.
  moon_zero_time_s: float
  moon_r0: float
  moon_c0: float
  moon_dr_dt: float
  moon_dc_dt: float
  moon_d2r_dt2: float
  moon_d2c_dt2: float


def save_sequence(filename: str, sequence: EclipseSequence):
  serialized = cattrs.unstructure(sequence)
  with open(filepaths.metadata(filename), 'w') as f:
    f.write(json.dumps(serialized))


def load_sequence(filename: str) -> EclipseSequence:
  with open(filepaths.metadata(filename), 'r') as f:
    serialized = json.loads(f.read())
  return cattrs.structure(serialized, EclipseSequence)


class TotalEclipseProcessor:
  def __init__(
      self,
      processor: raw_processor.RawProcessor,
      partials_preprocessor: partial_eclipse_tracker.PartialEclipsePreprocessor,
      partials_track: eclipse_tracker.EclipseTrack):
    """Initalize EclipseImages instance.

    Args:
      processor: RawProcessor instance for handling image corrections. Should
        have pre-loaded black frame calibration.
      partials_tracker: PartialEclipseTracker instance for tracking partial
        eclipses. Should already have pre-processed images.
      partials_track: fitted track for partial eclipse.
    """
    self.processor = processor
    self.partials_metadata = partials_preprocessor.image_metadata
    self.partials_track = partials_track
    self.totals_track = None
    self.totals_metadata = None
    self.corr_images_metadata = []

  def _stack_totals(self) -> list[int]:
    print('Read total eclipse image attributes...')
    attributes = eclipse_image_loader.maybe_read_images_attributes_by_index(
        range(constants.IND_FIRST_TOTAL, constants.IND_LAST_TOTAL + 1)
    )

    print('Generating HDR total eclipse images...')
    stacks = raw_processor.split_into_exposure_stacks(attributes)
    totals_indices = []
    for stack in stacks:
      if len(stack) != len(constants.EXPOSURES):
        continue

      metadata = stack[-1]
      index = metadata.index
      totals_indices.append(index)
      #continue
      unix_time_s = metadata.time.timestamp()
      npz_filepath = filepaths.hdr_total(index)
      print('  ' + npz_filepath)

      indices = [entry.index for entry in stack]
      images = eclipse_image_loader.maybe_read_images_by_index(
          indices, verbose=False)
      hdr_image = self.processor.stack_hdr_image(images)

      np.savez(npz_filepath,
               index=index,
               unix_time_s=unix_time_s,
               raw=hdr_image
      )

      del images
      gc.collect()

    print()
    return totals_indices

  def _track_totals(self, indices: Sequence[int]):
    preprocessor = total_eclipse_tracker.TotalEclipsePreprocessor()
    preprocessor.preprocess_images(indices)
    self.totals_metadata = preprocessor.image_metadata

    tracker = total_eclipse_tracker.TotalEclipseTracker()
    preprocessor.init_total_eclipse_tracker(tracker)
    self.totals_track = tracker.align_totals_to_partials_track(
        self.partials_track)

    tracker.plot_track(self.totals_track,
                       range(len(self.totals_track.unix_time_s)))

  def stack_and_track_totals(self):
    indices = self._stack_totals()
    self._track_totals(indices)

  def build_sequence(self) -> EclipseSequence:
    if not self.totals_track or not self.totals_metadata:
      raise RuntimeError('stack_and_track_totals() must be called before '
                         'generate_full_track()')

    all_metadata = self.partials_metadata + self.totals_metadata
    unix_time_s = [entry.common.unix_time_s for entry in all_metadata]

    image_metadata = []
    for ind in np.argsort(unix_time_s):
      metadata = all_metadata[ind].common

      if ind < len(self.partials_metadata):
        image_type = ImageType.RAW
        track = self.partials_track
        track_ind = ind
      else:
        image_type = ImageType.HDR
        track = self.totals_track
        track_ind = ind - len(self.partials_metadata)

      sun_center = tuple(
          2 * np.asarray(track.sun_centers[track_ind]) +
          np.asarray(metadata.bayer_offset) + 2 * np.asarray(metadata.offset)
      )

      image_metadata.append(
          ImageMetadata(
              index=metadata.index,
              unix_time_s=metadata.unix_time_s,
              image_type=image_type,
              sun_center=sun_center
          )
      )

    return EclipseSequence(
        image_metadata=image_metadata,
        sun_radius=2 * self.partials_track.sun_radius,
        moon_radius=2 * self.partials_track.moon_radius,
        moon_zero_time_s=self.partials_track.moon_zero_time_s,
        moon_r0 = 2 * self.partials_track.moon_r0,
        moon_c0 = 2 * self.partials_track.moon_c0,
        moon_dr_dt = 2 * self.partials_track.moon_dr_dt,
        moon_dc_dt = 2 * self.partials_track.moon_dc_dt,
        moon_d2r_dt2 = 2 * self.partials_track.moon_d2r_dt2,
        moon_d2c_dt2 = 2 * self.partials_track.moon_d2c_dt2,
    )


def _demosaic_image(raw: npt.NDArray,
                    white_level: float) -> npt.NDArray:
  scaled_raw = np.copy(raw)

  # Scale image to white level.
  scaled_raw = scaled_raw / white_level

  # Clip image to white and black levels. Assume np.nan pixels are saturated.
  #scaled_raw[np.isnan(scaled_raw)] = 1
  #scaled_raw = np.minimum(1, np.maximum(1e-3, scaled_raw))

  # Apply demosaicing.
  return colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(scaled_raw)


@dataclasses.dataclass
class RendererOptions:
  shape: tuple[int, int]

  # Angles must be between -180 and +180 deg.
  bright_corona_deg_lims: tuple[float, float]
  dark_corona_deg_lims: tuple[float, float]


class Renderer:
  def __init__(
      self,
      processor: raw_processor.RawProcessor,
      sequence: EclipseSequence,
      options: RendererOptions,
  ):
    self.processor = processor
    self.sequence = sequence
    self.options = options
    self.seq_ind_from_unix_time_s = {entry.unix_time_s: ind for ind, entry in
                                     enumerate(sequence.image_metadata)}
    self.index_from_unix_time_s = {entry.unix_time_s: entry.index for entry in
                                   sequence.image_metadata}
    self.unix_time_s_from_index = {entry.index: entry.unix_time_s for entry in
                                   sequence.image_metadata}

  def _read_total(self, unix_time_s: float) -> npt.NDArray:
    ind = self.seq_ind_from_unix_time_s[unix_time_s]
    metadata = self.sequence.image_metadata[ind]
    if metadata.image_type != ImageType.HDR:
      raise ValueError('Expected image type to be HDR.')
    data = np.load(filepaths.hdr_total(metadata.index))
    return data['raw']

  def _get_metadata(self, unix_time_s: float) -> ImageMetadata:
    seq_ind = self.seq_ind_from_unix_time_s[unix_time_s]
    return self.sequence.image_metadata[seq_ind]

  def _coord_about_sun(self, unix_time_s: float
                       ) -> tuple[npt.NDArray, npt.NDArray]:
    raw = self._read_total(unix_time_s)
    metadata = self._get_metadata(unix_time_s)
    row = np.arange(raw.shape[0]) - metadata.sun_center[0]
    col = np.arange(raw.shape[1]) - metadata.sun_center[1]
    return np.meshgrid(row, col, indexing='ij')

  def _radius_from_sun(self, unix_time_s: float) -> npt.NDArray:
    row, col = self._coord_about_sun(unix_time_s)
    return np.sqrt(row**2 + col**2)

  def _angle_from_sun_deg(self, unix_time_s: float) -> npt.NDArray:
    """Zero degrees is to the right (east) of the sun."""
    row, col = self._coord_about_sun(unix_time_s)
    x = col
    y = -row
    return np.rad2deg(np.arctan2(y, x))

  def _read_and_correct_partial(self, unix_time_s) -> npt.NDArray:
    index = self.index_from_unix_time_s[unix_time_s]
    image = eclipse_image_loader.maybe_read_image_by_index(index)
    corrected_image = self.processor.remove_hot_pixels(
        self.processor.subtract_background(image))
    return corrected_image.raw_image

  def fit_corona(self, unix_times_s: Sequence[float], show_plots: bool = False):
    # Collect all total eclipse images to be processed.
    image_all = []
    radius_all = []
    theta_deg_all = []
    for unix_time_s in unix_times_s:
      raw = self._read_total(unix_time_s)
      image = image_loader.correct_white_balance(raw, constants.WB_TOTAL)
      radius = self._radius_from_sun(unix_time_s)
      theta_deg = self._angle_from_sun_deg(unix_time_s)

      image_all.append(image)
      radius_all.append(radius)
      theta_deg_all.append(theta_deg)

    image = np.stack(image_all, axis=0)
    radius = np.stack(radius_all, axis=0)
    theta_deg = np.stack(theta_deg_all, axis=0)

    # Crop out edges of images, which have known artifacts in bright images.
    crop = raw_processor.CROP_PIXELS
    image = image[:, crop:-crop, crop:-crop]
    radius = radius[:, crop:-crop, crop:-crop]
    theta_deg = theta_deg[:, crop:-crop, crop:-crop]

    # Determine radii limits for fit.
    min_radius = 1.1 * self.sequence.sun_radius
    max_radius = 0.98 * np.max(image.shape[1:]) / 2

    def fit_corona_with_theta_lims(lims: tuple[float, float], label: str
                                   ) -> dict[str, list[float]]:
      lims = np.asarray(lims)
      if np.any(lims > 180) or np.any(lims < -180):
        raise ValueError(
            'Angle limits should be within +/- 180 deg, but recieved angle '
            f'limits of {lims[0]:f} and {lims[1]:f}'
        )
      mask = np.logical_and(theta_deg > lims[0], theta_deg < lims[1])
      masked_image = np.copy(image)
      masked_image[np.logical_not(mask)] = np.nan

      # Process each Bayer color subgrid independently.
      p_fit = {}
      mean_image = {}
      radius_bins = np.linspace(min_radius, max_radius, 50)
      for color in image_loader.BAYER_MASK_OFFSET:
        # Get pixels from Bayer subgrid.
        r0, c0 = image_loader.BAYER_MASK_OFFSET[color]
        radius_color = radius[:, r0::2, c0::2]
        image_color = masked_image[:, r0::2, c0::2]

        # Compute mean pixel value versus radius.
        mean_image[color] = []
        for ind in range(radius_bins.size - 1):
          mask = np.logical_and(radius_color >= radius_bins[ind],
                                radius_color < radius_bins[ind + 1])
          mean_image[color].append(np.nanmean(image_color[mask]))
        mean_image[color] = np.asarray(mean_image[color])

        # Fit a polylogarithmic function to pixel values versus radius.
        mask = np.logical_and(radius_bins[:-1] > min_radius,
                              radius_bins[:-1] < max_radius)
        radius_fit = radius_bins[:-1][mask]
        image_fit = mean_image[color][mask]
        p_fit[color] = np.polyfit(radius_fit, np.log(image_fit), 6)

      if show_plots:
        ind = 0
        unix_time_s = unix_times_s[ind]
        color = image_loader.DEFAULT_BAYER_COLOR
        r0, c0 = image_loader.BAYER_MASK_OFFSET[color]
        image_color = masked_image[ind, r0::2, c0::2]
        radius_color = radius[ind, r0::2, c0::2]
        leveled_image = image_color * np.exp(
            -np.polyval(p_fit[color], radius_color))
        leveled_image[radius_color > max_radius] = np.nan

        plt.figure()
        plt.imshow(leveled_image, cmap='inferno', vmin=0, vmax=2)
        plt.colorbar()
        plt.title(f'{label}: image used for fit after leveling '
                  f'(t = {unix_time_s:.1f} s, {color})')

        plt.figure()
        for color in image_loader.BAYER_MASK_OFFSET:
          plot_color = image_loader.BAYER_MASK_PLOT_COLORS[color]
          plt.semilogy(
              radius_bins[:-1],
              mean_image[color],
              '.',
              color=plot_color,
              label=f'{color} - data'
          )
          plt.semilogy(
              radius_bins,
              np.exp(np.polyval(p_fit[color], radius_bins)),
              '-',
              color=plot_color,
              label=f'{color} - fit'
          )

        plt.xlim(min_radius, max_radius)
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Pixel value')
        plt.title(f'{label}: fits')
        plt.legend()

      return p_fit

    dark_fit = fit_corona_with_theta_lims(
        self.options.dark_corona_deg_lims,
        'Dark corona',
    )
    bright_fit = fit_corona_with_theta_lims(
        self.options.bright_corona_deg_lims,
        'Bright corona',
    )

    # Save corona fits to a .npz file.
    fits = {}
    for color in dark_fit:
      fits['dark-corona_' + color] = dark_fit[color]
    for color in bright_fit:
      fits['bright-corona_' + color] = bright_fit[color]
    np.savez(filepaths.corona_fit(), **fits)

  def render_total(self, unix_time_s: float, show_plots: bool=False):
    metadata = self._get_metadata(unix_time_s)
    raw = self._read_total(unix_time_s)
    fits = np.load(filepaths.corona_fit())

    # Correct white balance.
    balanced_raw = image_loader.correct_white_balance(raw, constants.WB_TOTAL)

    # Pre-compute useful quantities.
    radius = self._radius_from_sun(unix_time_s)
    raw_shape = np.asarray(balanced_raw.shape, dtype=int)
    sun_center = np.asarray(metadata.sun_center)

    # Crop image to speed up image processing.
    buffer = 20
    crop_shape = 2 * ((np.asarray(self.options.shape) + 2 * buffer) // 2)
    crop_offset = 2 * (np.asarray(sun_center, dtype=int) // 2) - crop_shape // 2
    crop_sun_center = metadata.sun_center - crop_offset
    cropped_raw = balanced_raw[
        crop_offset[0]:crop_offset[0] + crop_shape[0],
        crop_offset[1]:crop_offset[1] + crop_shape[1],
    ]
    cropped_radius = radius[
        crop_offset[0]:crop_offset[0] + crop_shape[0],
        crop_offset[1]:crop_offset[1] + crop_shape[1],
    ]

    # Apply corona levelling fit.
    min_radius = 0.8 * self.sequence.sun_radius
    leveled_raw = np.zeros(cropped_raw.shape, dtype=float)
    for color in image_loader.BAYER_MASK_OFFSET:
      r0, c0 = image_loader.BAYER_MASK_OFFSET[color]
      image_color = cropped_raw[r0::2, c0::2]
      radius_color = np.maximum(min_radius, cropped_radius[r0::2, c0::2])

      dark_corona = np.exp(
          np.polyval(fits['dark-corona_' + color], radius_color))
      bright_corona  = np.exp(
          np.polyval(fits['bright-corona_' + color], radius_color))

      black_level = dark_corona
      white_level = 1.4 * bright_corona
      leveled_raw[r0::2, c0::2] = (image_color - black_level) / (white_level - black_level)

    # Demosaic image in linear space.
    linear_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(leveled_raw)

    # Render corona using nonlinear compression of values.
    def one_sided_sigmoid(x):
      y = np.copy(x)
      mask = x > 0
      y[mask] = x[mask] / (1 + x[mask])
      return y

    #compressed_raw = 0.8 * (0.5 + 0.5 * one_sided_sigmoid((leveled_raw - 2)))
    compressed_raw = np.clip(leveled_raw, a_min=0, a_max=1)
    corona_rgb = _demosaic_image(compressed_raw, 1)

    if False:
      # Perform initial denoising, filtering each Bayer subgrid independently.
      denoised_raw = np.zeros(raw.shape, dtype=float)
      for color in image_loader.BAYER_MASK_OFFSET:
        r0, c0 = image_loader.BAYER_MASK_OFFSET[color]
        center = (
            (metadata.sun_center[0] + r0)/ 2,
            (metadata.sun_center[1] + c0)/ 2,
        )
        denoised_raw[r0::2, c0::2] = filtering.radial_gaussian_filter(
            leveled_raw[r0::2, c0::2],
            center,
            sigma_r=10,
            sigma_theta=5,
            min_radius=200,
            taper_width=100,
        )

    # TODO: implement proper combination of linear and compressed corona images.
    final_rgb = corona_rgb

    # Shift image to center on the sun, and crop to final dimensions.
    shift = -sun_center + np.asarray(self.options.shape) / 2 + crop_offset
    print(shift)
    shifted = ndimage.shift(final_rgb, tuple(shift) + (0,))
    rgb = shifted[:self.options.shape[0], :self.options.shape[1], :]

    if show_plots:
      for color in image_loader.BAYER_MASK_OFFSET:
        r0, c0 = image_loader.BAYER_MASK_OFFSET[color]
        plt.figure()
        plt.imshow(leveled_raw[r0::2, c0::2], cmap='inferno')
        plt.title(color)
        plt.clim(0, 2)
        plt.colorbar()

      plt.figure()
      plt.imshow(linear_rgb / np.nanmax(linear_rgb))
      plt.title('Linear')

      plt.figure()
      plt.imshow(corona_rgb)
      plt.title('Corona')

      plt.figure()
      plt.imshow(rgb)
      plt.title('Final')

    # Save rendered image to file.
    index = self.index_from_unix_time_s[unix_time_s]
    np.savez(
        filepaths.rendered(index),
        rgb=rgb
    )


  def _correct_partials(self):
    """We cannot use this function because it requires too much disk space."""
    print('Applying corrections to partial images...')
    for metadata in self.partials_metadata:
      index = metadata.common.index
      print(f'  IMG_{index}.CR2')

      try:
        image = eclipse_image_loader.maybe_read_image_by_index(index)
        corrected_image = self.processor.remove_hot_pixels(
            self.processor.subtract_background(image))
        np.savez(filepaths.corrected_raw(index),
                 raw=corrected_image.raw_image)

        new_metadata = ProcessedImage(
            index=index,
            unix_time_s=metadata.common.unix_time_s,
            is_total=False
        )
        self.corr_images_metadata.append(new_metadata)
      except ValueError:
        pass
    print()

