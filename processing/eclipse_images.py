import dataclasses
import os

import numpy as np
import numpy.typing as npt

import constants
import filepaths
import image_loader
import raw_processor
import eclipse_tracker
import partial_eclipse_tracker


@dataclasses.dataclass
class ProcessedImage:
  index: int
  unix_time_s: float
  is_total: bool


class EclipseImages:
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

    if partials_track.image_type != eclipse_tracker.ImageType.BW:
      raise ValueError(
          'Expected image type for partial eclipse track to be BW')
    self.partials_track = partials_track

    self.corr_images_metadata = []

  def _correct_partials(self):
    print('Applying corrections to partial images...')
    for metadata in self.partials_metadata:
      index = metadata.common.index
      print(f'  IMG_{index}.CR2')

      image = image_loader.maybe_load_image_by_index(index)
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
    print()

  def _stack_hdr_totals(self):
    print('Loading total eclipse images...')
    images = []
    for ind in range(constants.IND_FIRST_TOTAL, constants.IND_LAST_TOTAL + 1):
      print(f'  IMG{ind:04d}.CR2')
      images.append(image_loader.maybe_load_image_by_index(ind))
    print()

    print('Generating HDR total eclipse images...')
    stacks = raw_processor.split_into_exposure_stacks(images)
    for stack in stacks:
      if len(stack) != len(constants.EXPOSURES):
        continue

      index = stack[0].index
      print('  ' + filepaths.corrected_raw(index))

      hdr_image = self.processor.stack_hdr_image(stack)
      np.savez(filepaths.corrected_raw(index),
               raw=hdr_image)

      new_metadata = ProcessedImage(
          index=index,
          unix_time_s=stack[0].time.timestamp(),
          is_total=True
      )
      self.corr_images_metadata.append(new_metadata)
    print()

  def correct_images(self):
    self._correct_partials()
    self._stack_hdr_totals()

    # Sort image metadata by timestamp.
    unix_time_s = [entry.unix_time_s for entry in self.corr_images_metadata]
    self.corr_images_metadata = [self.corr_images_metadata[ind] for ind in
                                 np.argsort(unix_time_s)]

  def track_totals(self):
    pass

