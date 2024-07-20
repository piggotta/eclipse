import dataclasses
import datetime
import glob
import os

import exifread
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rawpy

@dataclasses.dataclass
class RawImage:
  filepath: str
  index: int
  time: datetime.datetime
  exposure_s: float
  f_number: float
  iso: int
  raw_image: npt.NDArray
  bw_image: npt.NDArray

def load_image(filepath: str) -> npt.NDArray:
  # Read image index from filename.
  filename = os.path.split(filepath)[1]
  if filename[:4] != 'IMG_' or filename[-4:] != '.CR2':
    raise ValueError(f'Unexpected filename {filename}. Filename should be of '
                     'form IMG_*.CR2')
  index = int(filename.strip('IMG_').strip('.CR2'))

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

  # Read raw image.
  with rawpy.imread(filepath) as raw:
    raw_image = np.copy(raw.raw_image)

  # Use (0, 0) Bayer subimage (green) as our black and white image.
  green_image = raw_image[::2, ::2]

  return RawImage(
      filepath=filepath,
      index=index,
      time=image_time,
      exposure_s=exposure_s,
      f_number=f_number,
      iso=iso,
      raw_image=raw_image,
      bw_image=green_image)
