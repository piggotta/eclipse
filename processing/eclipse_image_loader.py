from collections.abc import Sequence
import os
import datetime

import constants
import filepaths
import image_loader


def maybe_read_image_attributes_by_index(index: int) -> image_loader.RawImage:
  filepath = filepaths.raw(index)
  if not os.path.isfile(filepath):
    raise ValueError(f'{filepath} does not exist.')
  return image_loader.read_attributes(filepath, constants.TIME_ZONE)


def maybe_read_images_attributes_by_index(
    indices: Sequence[int],
    verbose: bool = True) -> list[image_loader.RawImage]:
  """Reads camera images by index. Skips any missing files."""
  if verbose:
    print('Loading...')

  attributes = []
  for index in indices:
    try:
      attributes.append(
          maybe_read_image_attributes_by_index(index))
      if verbose:
        print(f'  {filepaths.raw(index)}')
    except ValueError:
      pass

  if verbose:
    print()

  return attributes


def maybe_read_image_by_index(
    index: int,
    bayer_offset: tuple[int, int] = image_loader.DEFAULT_BAYER_OFFSET
    ) -> image_loader.RawImage:
  filepath = filepaths.raw(index)
  if not os.path.isfile(filepath):
    raise ValueError(f'{filepath} does not exist.')
  return image_loader.read_image(
      filepath,
      constants.TIME_ZONE,
      bayer_offset=bayer_offset
  )


def maybe_read_images_by_index(
    indices: Sequence[int],
    bayer_offset: tuple[int, int] = image_loader.DEFAULT_BAYER_OFFSET,
    verbose: bool = True) -> list[image_loader.RawImage]:
  """Reads camera images by index. Skips any missing files."""
  if verbose:
    print('Loading...')

  images = []
  for index in indices:
    try:
      images.append(
          maybe_read_image_by_index(index, bayer_offset=bayer_offset))
      if verbose:
        print(f'  {filepaths.raw(index)}')
    except ValueError:
      pass

  if verbose:
    print()

  return images
