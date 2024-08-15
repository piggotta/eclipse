from collections.abc import Sequence
import dataclasses
import datetime
import os

import exifread
import numpy as np
import numpy.typing as npt
import rawpy

import constants


BAYER_MASK_OFFSET = {
    'red': (0, 0),
    'green0': (0, 1),
    'green1': (1, 0),
    'blue': (1, 1),
}


@dataclasses.dataclass
class Exposure:
  exposure_s: float
  f_number: float
  iso: int

  def norm_exposure_s(self) -> float:
    """Returns exposure time normalized to f/1.0 and ISO 100."""
    return self.exposure_s * (1 / self.f_number**2) * (self.iso / 100)


@dataclasses.dataclass
class RawImage:
  filepath: str | None
  index: int | None
  time: datetime.datetime | None
  exposure_s: float
  f_number: float
  iso: int
  raw_image: npt.NDArray | None
  bw_image: npt.NDArray | None

  def get_exposure(self) -> Exposure:
    return Exposure(
        exposure_s=self.exposure_s,
        f_number=self.f_number,
        iso=self.iso
    )


def read_attributes(filepath) -> RawImage:
  # Read image index from filename.
  filename = os.path.split(filepath)[1]
  if filename[:4] != 'IMG_' or filename[-4:] != '.CR2':
    raise ValueError(f'Unexpected filename {filename}. Filename should be of '
                     'form IMG_*.CR2')
  index = int(filename[4:-4])

  # Read EXIF tag information.
  with open(filepath, 'rb') as f:
    tags = exifread.process_file(f)
  exposure_s = float(tags['EXIF ExposureTime'].values[0])
  f_number = float(tags['EXIF FNumber'].values[0])
  iso = int(tags['EXIF ISOSpeedRatings'].values[0])

  time_str = (tags['EXIF DateTimeOriginal'].values + '.' +
              tags['EXIF SubSecTimeOriginal'].values)
  date, time = time_str.split(' ')
  date = date.replace(':', '-')
  image_time = datetime.datetime.fromisoformat(date + ' ' + time + '0')

  return RawImage(
      filepath=filepath,
      index=index,
      time=image_time,
      exposure_s=exposure_s,
      f_number=f_number,
      iso=iso,
      raw_image=None,
      bw_image=None)


def read_image(filepath: str,
               bayer_offset: tuple[int, int] = BAYER_MASK_OFFSET['red']
               ) -> RawImage:
  image = read_attributes(filepath)

  # Read raw image.
  with rawpy.imread(filepath) as raw:
    image.raw_image = np.asarray(np.copy(raw.raw_image), dtype=int)

  # Use the selected Bayer subimage as our black and white image.
  image.bw_image = image.raw_image[bayer_offset[0]::2, bayer_offset[1]::2]
  return image


def maybe_load_images_by_index(
    indices: Sequence[int],
    bayer_offset: tuple[int, int] = BAYER_MASK_OFFSET['red'],
    verbose: bool = True) -> list[RawImage]:
  """Loads camera images by index. Skips any missing files."""
  if verbose:
    print('Loading...')

  images = []
  for index in indices:
    filepath = os.path.join(constants.PHOTOS_PATH, f'IMG_{index}.CR2')
    if not os.path.isfile(filepath):
      continue
    if verbose:
      print(f'  {filepath}')
    images.append(read_image(filepath, bayer_offset=bayer_offset))

  if verbose:
    print()

  return images
