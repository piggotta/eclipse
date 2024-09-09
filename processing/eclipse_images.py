from collections.abc import Sequence
import dataclasses
import enum
import json
import gc
import os

import cattrs
import colour
import numpy as np
import numpy.typing as npt

from colour_demosaicing import (
    ROOT_RESOURCES_EXAMPLES,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

import constants
import eclipse_tracker
import filepaths
import image_loader
import partial_eclipse_tracker
import raw_processor
import total_eclipse_tracker

# DELETE ME
import matplotlib.pyplot as plt


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
    attributes = image_loader.maybe_read_images_attributes_by_index(
        range(constants.IND_FIRST_TOTAL, constants.IND_LAST_TOTAL + 1)
    )

    print('Generating HDR total eclipse images...')
    stacks = raw_processor.split_into_exposure_stacks(attributes)
    totals_indices = []
    for stack in stacks:
      if len(stack) != len(constants.EXPOSURES):
        continue

      metadata = stack[0]
      index = metadata.index
      totals_indices.append(index)
      unix_time_s = metadata.time.timestamp()
      npz_filepath = filepaths.hdr_total(index)
      print('  ' + npz_filepath)

      indices = [entry.index for entry in stack]
      images = image_loader.maybe_read_images_by_index(indices, verbose=False)
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
