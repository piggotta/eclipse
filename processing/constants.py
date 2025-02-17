import dataclasses
import zoneinfo

import image_loader

EXPOSURES = [
    image_loader.Exposure(exposure_s=5, f_number=5.6, iso=2000),
    image_loader.Exposure(exposure_s=0.6, f_number=5.6, iso=2000),
    image_loader.Exposure(exposure_s=1/13, f_number=5.6, iso=2000),
    image_loader.Exposure(exposure_s=1/100, f_number=5.6, iso=2000),
    image_loader.Exposure(exposure_s=1/800, f_number=5.6, iso=2000),
    image_loader.Exposure(exposure_s=1/800, f_number=8, iso=500),
    image_loader.Exposure(exposure_s=1/2000, f_number=8, iso=100),
]
LONGEST_EXPOSURE_IND = 0

TIME_ZONE = zoneinfo.ZoneInfo('America/Los_Angeles')

IND_FIRST_SUN = 978
IND_FIRST_PARTIAL = 994
IND_FIRST_TOTAL = 1343
IND_LAST_TOTAL = 1468
IND_LAST_PARTIAL = 1805
IND_LAST_SUN = 1812
IND_FIRST_GREY = 1918
IND_LAST_GREY = 1981
IND_FIRST_BLACK = 1982
IND_LAST_BLACK = 2023

WB_TOTAL = image_loader.WhiteBalance(
    blue_scaling = 1 / 0.65,
    red_scaling = 1 / 0.49,
)

WB_PARTIAL = image_loader.WhiteBalance(
    blue_scaling = 1 / 0.65,
    red_scaling = 1 / 0.49,
)

WHITE_LEVEL_TOTAL = 1e8

RENDER_SHAPE = (3000, 3000)
RENDER_BRIGHT_CORONA_DEG_LIMS = (39, 43)
RENDER_DARK_CORONA_DEG_LIMS = (-20, 20)

