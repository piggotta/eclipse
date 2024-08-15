import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import exifread
import rawpy

import image_loader

PHOTO_PATH = 'photos'

# Demosaicing
# 1. For tracking algorithms, either use:
#    a. A single color channel (perhaps ideal)
#    b. Any cheap demosaicing algorithm (luminance noise should be smaller than
#    chromatic noise).
#   Global demosaicing with fixed constants does not work perfectly, even for partial eclipse
#   photos, because color appears to change across sun.
#
# 2. Consider doing HDR stacking first, and then demosaicing.
#   Demosaic libraries:
#      https://pypi.org/project/colour-demosaicing/
#
# 3. Algorithm=1 and median_filter_passes=3 seems best for demosaicing (less
#   color noise than default 3 on flat areas). With no median filtering, the
#   default 3 is essentially impossible to beat.

def compare_demosaicing():
  for filename in ('IMG_1001.CR2', 'IMG_1404.CR2'):
    for alg in range(13):
      with rawpy.imread(os.path.join(PHOTO_PATH, filename)) as raw:
        print(raw)
        print(dir(raw))
        print(raw.raw_pattern)

        #plt.figure()
        #plt.imshow(raw.raw_image)

        demosaic_algorithm = rawpy.DemosaicAlgorithm(alg)
        if not demosaic_algorithm.isSupported:
          continue
        rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16,
                              demosaic_algorithm=demosaic_algorithm,
                              median_filter_passes=3)
        print('Shapes')
        print(raw.raw_image.shape)
        print(rgb.shape)

        scale = 0.02
        r_frac = 1.0 * scale
        c_frac = 0.7 * scale
        r_offset = 0.07
        c_offset = 0.015
        r_start = int((0.5 + r_offset - r_frac) * rgb.shape[0])
        r_stop = int((0.5 + r_offset + r_frac) * rgb.shape[0])
        c_start = int((0.5 + c_offset - c_frac) * rgb.shape[1])
        c_stop = int((0.5 + c_offset + c_frac) * rgb.shape[1])
        rgb_crop = rgb[r_start:r_stop, c_start:c_stop] / np.max(rgb)

        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_crop)
        plt.title(f'{alg}')

def convert_to_bw():
  for filename in ('IMG_1001.CR2',):
    with rawpy.imread(os.path.join(PHOTO_PATH, filename)) as raw:
      raw_image = np.asarray(raw.raw_image)
      bayer_offsets = ((0, 0), (0, 1), (1, 0), (1, 1))
      bw_image = np.zeros(raw_image.shape, dtype=float)
      for offset_r, offset_c in bayer_offsets:
        sub_image = raw_image[offset_r::2, offset_c::2]
        black_level = np.quantile(sub_image, 0.2)
        print(np.quantile(sub_image, 0.990),
              np.quantile(sub_image, 0.999))
        mask = np.logical_and(sub_image > np.quantile(sub_image, 0.9990),
                              sub_image < np.quantile(sub_image, 0.9995))
        black_corr_image = sub_image - black_level
        norm_image = black_corr_image / np.mean(black_corr_image[mask])
        bw_image[offset_r::2, offset_c::2] = norm_image

        plt.figure()
        plt.imshow(mask)
        plt.title(f'offset = str(offset)')

      plt.figure()
      plt.imshow(raw.raw_image)
      plt.colorbar()
      plt.title(f'Raw - {filename}')

      plt.figure()
      plt.imshow(bw_image)
      plt.colorbar()
      plt.title(f'BW - {filename}')

def main():
  start = time.time()

  compare_demosaicing()
  convert_to_bw()

  print(f'Time elapsed: {time.time() - start:.3f} s')

  plt.show()

if __name__ == '__main__':
  main()
