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


CORONA_RADIUS_MIN = 300
CORONA_RADIUS_MAX = 3000

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
        sun_radius=self.partials_track.sun_radius,
        moon_radius=self.partials_track.moon_radius,
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
  scaled_raw[np.isnan(scaled_raw)] = 1
  scaled_raw = np.minimum(1, np.maximum(1e-3, scaled_raw))

  # Apply demosaicing.
  return colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(scaled_raw)


class Renderer:
  def __init__(
      self,
      processor: raw_processor.RawProcessor,
      sequence: EclipseSequence
  ):
    self.processor = processor
    self.sequence = sequence
    self.seq_ind_from_unix_time_s = {entry.unix_time_s: ind for ind, entry in
                                     enumerate(sequence.image_metadata)}
    self.index_from_unix_time_s = {entry.unix_time_s: entry.index for entry in
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

  def _radius_from_sun(self, unix_time_s: float) -> npt.NDArray:
    raw = self._read_total(unix_time_s)
    metadata = self._get_metadata(unix_time_s)
    r = np.arange(raw.shape[0])
    c = np.arange(raw.shape[1])
    r, c = np.meshgrid(r, c, indexing='ij')
    return np.sqrt((r - metadata.sun_center[0])**2 +
                   (c - metadata.sun_center[1])**2)

  def _read_and_correct_partial(self, unix_time_s) -> npt.NDArray:
    index = self.index_from_unix_time_s[unix_time_s]
    image = eclipse_image_loader.maybe_read_image_by_index(index)
    corrected_image = self.processor.remove_hot_pixels(
        self.processor.subtract_background(image))
    return corrected_image.raw_image

  def fit_corona(self, unix_time_s: float) -> npt.NDArray:
    raw = self._read_total(unix_time_s)
    raw = image_loader.correct_white_balance(raw, constants.WB_TOTAL)
    radius = self._radius_from_sun(unix_time_s)

    raw = raw[_CROP_PIXELS:-_CROP_PIXELS, _CROP_PIXELS:-_CROP_PIXELS]
    radius = radius[_CROP_PIXELS:-_CROP_PIXELS, _CROP_PIXELS:-_CROP_PIXELS]

    # Computes mean pixel value versus radius for each Bayer subgrid.
    mean_raw = {}
    radius_bins = np.linspace(0, np.max(radius), 100)
    for color in image_loader.BAYER_MASK_OFFSET:
      r0, c0 = image_loader.BAYER_MASK_OFFSET[color]
      radius_color = radius[r0::2, c0::2]
      raw_color = raw[r0::2, c0::2]

      mean_raw[color] = []
      for ind in range(radius_bins.size - 1):
        mask = np.logical_and(radius_color >= radius_bins[ind],
                              radius_color < radius_bins[ind + 1])
        mean_raw[color].append(np.nanmean(raw_color[mask]))
      mean_raw[color] = np.asarray(mean_raw[color])

    # Computes mean pixel value versus radius over all Bayer subgrids.
    all_color_mean = 0
    for color in image_loader.BAYER_MASK_OFFSET:
      all_color_mean += mean_raw[color]
    all_color_mean *= 1 / len(image_loader.BAYER_MASK_OFFSET)

    # Fits a polylogarithmic function to pixel values versus radius.
    mask = np.logical_and(radius_bins[:-1] > CORONA_RADIUS_MIN,
                          radius_bins[:-1] < CORONA_RADIUS_MAX)
    radius_fit = radius_bins[:-1][mask]
    raw_fit = all_color_mean[mask]
    p = np.polyfit(radius_fit, np.log(raw_fit), 6)
    return p


  def render_total(self, unix_time_s: float, corona_polylog_fit: npt.NDArray,
                   show_plots: bool=True):
    metadata = self._get_metadata(unix_time_s)
    raw = self._read_total(unix_time_s)

    balanced_raw = image_loader.correct_white_balance(raw, constants.WB_TOTAL)

    # Apply corona levelling fit.
    radius = self._radius_from_sun(unix_time_s)
    leveled_raw = balanced_raw * np.exp(
        -np.polyval(corona_polylog_fit, radius))

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

    # Use linear rendering for brightest parts of image (prominences, etc).
    linear_rgb = _demosaic_image(denoised_raw, np.nanmax(denoised_raw))

    # Render corona using nonlinear compression of values.
    def one_sided_sigmoid(x):
      y = np.copy(x)
      mask = x > 0
      y[mask] = x[mask] / (1 + x[mask])
      return y
    radial_gain = 1 #+ radius / 10e3

    compressed_raw = 0.5 + 0.5 * one_sided_sigmoid(2 * radial_gain * (denoised_raw - 1))
    compressed_raw = np.clip(compressed_raw, a_min=0, a_max=1)
    corona_rgb = _demosaic_image(compressed_raw, 1)

    if show_plots:
      plt.figure()
      plt.imshow(linear_rgb)

      plt.figure()
      plt.imshow(corona_rgb)


    # TODO: implement proper combination of linear and compressed corona images.
    final_rgb = corona_rgb

    # Save rendered image to file.
    index = self.index_from_unix_time_s[unix_time_s]
    np.savez(
        filepaths.rendered(index),
        rgb=final_rgb
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

