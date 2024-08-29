import eclipse_images
import eclipse_tracker
import partial_eclipse_tracker
import raw_processor

def main():
  processor = raw_processor.RawProcessor()
  processor.load_calibration_from_file('calibration')

  partials_preprocessor = partial_eclipse_tracker.PartialEclipsePreprocessor()
  partials_preprocessor.load_from_file('preprocess_partials')

  partials_track = eclipse_tracker.load_track('partial_track')

  image_gen = eclipse_images.EclipseImages(
      processor, partials_preprocessor, partials_track)
  image_gen.correct_images()

if __name__ == '__main__':
  main()
