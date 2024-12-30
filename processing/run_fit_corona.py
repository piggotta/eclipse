import numpy as np
import matplotlib.pyplot as plt

import eclipse_images
import filepaths
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

  # Obtain typical corona fit.
  p_fits = []
  for unix_time_s in totals_unix_time_s[2:-2]:
    p_fits.append(renderer.fit_corona(unix_time_s))
  p_fit = np.mean(np.asarray(p_fits), axis=0)

  # Save fit to file.
  np.savez(filepaths.corona_fit(), p_fit=p_fit)
  p_fit = np.load(filepaths.corona_fit())['p_fit']

  # Plot the fit.
  r = np.linspace(
      eclipse_images.CORONA_RADIUS_MIN,
      eclipse_images.CORONA_RADIUS_MAX,
      100
  )

  plt.figure()
  plt.semilogy(r, np.exp(np.polyval(p_fit, r)), 'k')
  plt.xlim(np.min(r), np.max(r))
  plt.xlabel('Radius (pixels)')
  plt.ylabel('HDR pixel value')
  plt.title('Corona fit')

  plt.show()

if __name__ == '__main__':
  main()
