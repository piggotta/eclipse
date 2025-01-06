def print_title(title: str, width=80):
  if len(title) > width - 10:
    raise ValueError(
        f'Length of title must be =< {width - 10:d} characters, but is '
        f'{len(title):d} characters'
    )
  padding = width - len(title) - 2
  left_padding = padding // 2
  right_padding = padding - left_padding
  title_row = '#' + left_padding * ' ' + title + right_padding * ' ' + '#'

  print()
  print(width * '#')
  print('#' + (width - 2) * ' ' + '#')
  print(title_row)
  print('#' + (width - 2) * ' ' + '#')
  print(width * '#')
  print()
