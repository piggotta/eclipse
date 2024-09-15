import numpy as np
import matplotlib.pyplot as plt

import eclipse_images
import raw_processor


def main():
  # Load raw image processor configuration from file.
  processor = raw_processor.RawProcessor()
  processor.load_calibration_from_file('calibration')

  # Load full eclipse image sequence from file
  sequence = eclipse_images.load_sequence('full_sequence')

  # Create renderer object.
  renderer = eclipse_images.Renderer(processor, sequence)

  # Find total eclipse images.
  totals_unix_time_s = [entry.unix_time_s for entry in sequence.image_metadata
                        if entry.image_type == eclipse_images.ImageType.HDR]

  #for ind in (0, 1, 2, 5, 10, 18):
  for ind in (2,):
    renderer.analyze_total(totals_unix_time_s[ind])

  plt.show()

if __name__ == '__main__':
  main()
