import dataclasses
import glob
import json
import os

import cattrs
import numpy as np
import numpy.typing as npt

import constants
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



class PartialEclipseTracker:

  def __init__(self):
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
