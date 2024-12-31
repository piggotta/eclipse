import os
import pathlib


PHOTOS_PATH = 'photos'
OUTPUTS_PATH = 'outputs'
HDR_TOTALS_PATH = 'outputs/hdr'
RENDERED_PATH = 'outputs/rendered'


def _make_recursive_dir_if_not_exist(path: str):
  dirs = pathlib.Path(path).parts
  for ind in range(len(dirs)):
    current_path = os.path.join(*dirs[:ind + 1])
    if not os.path.isdir(current_path):
      os.mkdir(current_path)


def _make_folder_and_return_path(folder: str, filename: str):
  _make_recursive_dir_if_not_exist(folder)
  return os.path.join(folder, filename)


def raw(index: int) -> str:
  return os.path.join(PHOTOS_PATH, f'IMG_{index:04d}.CR2')


def metadata(filename: str) -> str:
  return _make_folder_and_return_path(OUTPUTS_PATH, filename + '.cattr')


def calibration(filename: str) -> str:
  return _make_folder_and_return_path(OUTPUTS_PATH, filename + '.npz')


def cropped_partials(filename: str) -> str:
  return _make_folder_and_return_path(OUTPUTS_PATH, filename + '.npz')


def corona_fit() -> str:
  return _make_folder_and_return_path(OUTPUTS_PATH, 'corona_fit.npz')


def hdr_total(index: int) -> str:
  return _make_folder_and_return_path(HDR_TOTALS_PATH,
                                      f'total_{index:04d}.npz')


def rendered(index: int) -> str:
  return _make_folder_and_return_path(RENDERED_PATH, f'img_{index:04d}.npz')

