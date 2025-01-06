import run_dark_and_grey_frame_analysis
import run_fit_corona
import run_preprocess_partials
import run_preprocess_totals
import run_render_totals
import run_track_partials


def main():
  run_dark_and_grey_frame_analysis.main()
  run_preprocess_partials.main()
  run_track_partials.main(show_plots=False, plot_only=False)
  run_fit_corona.main(show_plots=False)
  run_preprocess_totals.main(show_plots=False)
  run_render_totals.main(show_plots=False)


if __name__ == '__main__':
  main()
