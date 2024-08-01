import matplotlib.pyplot as plt
import numpy as np

import partial_eclipse_tracker

# sun radius = 139

def main():
  options = partial_eclipse_tracker.PartialEclipseTrackerOptions(
      guess_sun_radius=100,
      # Moon movement rate estimate
      #  * Radius of sun is approximately 130 pixels
      #  * Takes approximately 9000 seconds for moon to cross sun
      #  * Moon moves from bottom-left corner of sun to top-left corner of sun
      guess_moon_dr_dt=-0.7 * 4 * 130 / 9000,
      guess_moon_dc_dt=0.7 * 4 * 130 / 9000,
      decimate_for_debug=True,
  )
  tracker = partial_eclipse_tracker.PartialEclipseTracker(options)
  tracker.load_preprocess_from_file('preprocess_partials')
  #tracker.track_sun_and_moon()
  tracker.fit_sun_radius()
  plt.show()
  return

  print(tracker.approx_total_eclipse_start_unix_time_s())
  print(tracker.approx_total_eclipse_end_unix_time_s())

  inds = np.asarray(
      np.linspace(0, tracker.is_sun.shape[0] - 1, 30),
      dtype=int)

  start_time_s = tracker.image_metadata[0].unix_time_s
  for ind in inds:
    time_s = tracker.image_metadata[ind].unix_time_s - start_time_s
    plt.figure()
    plt.imshow(tracker.is_sun[ind, :, :], cmap='gray')
    plt.colorbar()
    plt.title(f'{time_s:.1f} s')
  plt.show()
  return




if __name__ == '__main__':
  main()
