from collections.abc import Sequence
import copy
import dataclasses
import os

import numpy as np
import numpy.typing as npt
from skimage import morphology
from skimage import segmentation

import constants
import drawing
import eclipse_tracker
import filepaths
import image_loader


@dataclasses.dataclass
class TotalEclipseImageMetadata:
  common: eclipse_tracker.CroppedBwImageMetadata
  black_level: float
  white_level: float


def _draw_total_eclipses(image_shape: tuple[int, int],
                         track: eclipse_tracker.EclipseTrack) -> npt.NDArray:
  images = np.zeros((len(track.unix_time_s), *image_shape))
  for ind, time_s in enumerate(track.unix_time_s):
    sun_center = track.sun_centers[ind]

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

    # Flip polarity of moon image since the moon is blocking light.
    images[ind, :, :] = 1 - moon_image

  return images


class TotalEclipseTracker(eclipse_tracker.EclipseTracker):
  def __init__(self):
    super().__init__(renderer=_draw_total_eclipses)

  def align_totals_to_partials_track(
      self, partials_track: eclipse_tracker.EclipseTrack
  ) -> eclipse_tracker.EclipseTrack:
    track = copy.copy(partials_track)
    track.unix_time_s = [entry.unix_time_s for entry in self.image_metadata]
    track.sun_centers = None

    all_indices = np.asarray([True for _ in self.image_metadata])
    return self.align_images(track, all_indices)


@dataclasses.dataclass
class PreprocessedImage:
  metadata: TotalEclipseImageMetadata
  cropped_image: npt.NDArray
  not_moon: npt.NDArray


def _preprocess_image(
    filepath: str,
    cropped_shape: tuple[int, int],
    bayer_offset: tuple[int, int] = image_loader.BAYER_MASK_OFFSET['red'],
    black_level: float = 2e6,
    white_level: float = 7e6,
    ) -> PreprocessedImage:
  data = np.load(filepath)

  # Generate greyscale image by taking a single Bayer subimage.
  bw_image = data['raw'][bayer_offset[0]::2, bayer_offset[1]::2]
  bw_image[np.isnan(bw_image)] = 1

  # Adjust the white and black levels to generate a high-contrast greyscale
  # image of the moon shadow.
  scaled = (bw_image - black_level) / (white_level - black_level)
  scaled = np.minimum(1, np.maximum(0, scaled))

  # Crop only center of image.
  center = (bw_image.shape[0] // 2, bw_image.shape[1] // 2)
  offset = center - np.asarray(cropped_shape) // 2

  cropped_image = bw_image[
      offset[0]:offset[0] + cropped_shape[0],
      offset[1]:offset[1] + cropped_shape[1]
  ]
  not_moon = scaled[
      offset[0]:offset[0] + cropped_shape[0],
      offset[1]:offset[1] + cropped_shape[1]
  ]

  # Fill in region outside of moon area.
  is_moon = segmentation.flood(
      not_moon,
      (cropped_shape[0] // 2, cropped_shape[1] // 2),
      tolerance=0.2
  )
  is_moon = morphology.binary_dilation(
      is_moon,
      footprint=morphology.disk(3)
  )
  not_moon[np.logical_not(is_moon)] = 1

  common = eclipse_tracker.CroppedBwImageMetadata(
      index=int(data['index']),
      unix_time_s=float(data['unix_time_s']),
      bayer_offset=bayer_offset,
      orig_shape=bw_image.shape,
      cropped_shape=tuple([int(n) for n in cropped_shape]),
      offset=tuple([int(n) for n in offset]),
  )
  metadata = TotalEclipseImageMetadata(
      common=common,
      black_level=black_level,
      white_level=white_level
  )
  return PreprocessedImage(
      metadata=metadata,
      cropped_image=cropped_image,
      not_moon=not_moon,
  )


class TotalEclipsePreprocessor:
  def __init__(self):
    self.image_metadata = None
    self.cropped_images = None
    self.not_moon = None

  def save_to_file(self, filename: str):
    dicts = [cattrs.unstructure(m) for m in self.image_metadata]
    with open(filepaths.metadata(filename), 'w') as f:
      f.write(json.dumps(dicts))

    np.savez(filepaths.cropped_partials(filename),
             cropped_images=self.cropped_images,
             not_moon=not_moon)

  def load_from_file(self, filename: str):
    with open(filepaths.metadata(filename), 'r') as f:
      dicts = json.loads(f.read())
    self.image_metadata = [
        cattrs.structure(d, PartialEclipseImageMetadata) for d in dicts
    ]

    data = np.load(filepaths.cropped_partials(filename))
    self.cropped_images = data['cropped_images']
    self.not_moon = data['not_moon']

  def preprocess_images(self,
                        totals_indices: Sequence[int],
                        cropped_shape: tuple[int, int] = (500, 500)):
    totals_indices = list(totals_indices)
    num_totals = len(totals_indices)

    print('Processing images...')
    self.cropped_images = np.zeros((num_totals, *cropped_shape))
    self.not_moon = np.zeros((num_totals, *cropped_shape))
    self.image_metadata = []
    for ind, index in enumerate(totals_indices):
      filepath = filepaths.hdr_total(index)
      filename = os.path.split(filepath)[1]
      print(f'  {filename:s}')

      result = _preprocess_image(filepath, cropped_shape)
      self.image_metadata.append(result.metadata)
      self.cropped_images[ind, :, :] = result.cropped_image
      self.not_moon[ind, :, :] = result.not_moon

      del result
    print()

  def init_total_eclipse_tracker(self, tracker: TotalEclipseTracker):
    tracker.init(
        image_metadata=[entry.common for entry in self.image_metadata],
        bw_images=self.not_moon,
    )

