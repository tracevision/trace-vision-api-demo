#
#  Copyright Alpinereplay Inc., 2025. All rights reserved.
#  Authors: Claire Roberts-Thomson
#
"""
Coordinate conversions.

This module contains functions for converting between latitude/longitude and
x/y coordinates.
"""

import math


def ll2xy(lat, lon, LAT0, LON0):
    """
    Convert lat, lon to x, y.

    Args:
        lat     Latitude value (degrees)
        lon     Longitude value (degrees)
        LAT0    Reference latitude (degrees)
        LON0    Reference longitude (degrees)
    Returns:
        X       N-S location(s) corresponding to (lat, lon) relative to LAT0, LON0 (m)
        Y       E-W location(s) corresponding to (lat, lon) relative to LAT0, LON0 (m)
    """
    DG2M = 40 * 1.0e6 / 360
    s = math.cos(LAT0 * math.pi / 180)
    X = s * DG2M * (lon - LON0)
    Y = DG2M * (lat - LAT0)

    return X, Y


def xy2ll(X, Y, LAT0, LON0):
    """
    Convert X, Y to lat, lon.

    Args:
        X       N-S location(s) relative to LAT0, LON0 (m)
        Y       E-W location(s) relative to LAT0, LON0 (m)
        LAT0    Reference latitude (degrees)
        LON0    Reference longitude (degrees)
    Returns:
        lat     Latitude value(s) corresponding to (x, y) (degrees)
        lon     Longitude value(s) corresponding to (x, y) (degrees)
    """
    DG2M = 40 * 1.0e6 / 360
    s = math.cos(LAT0 * math.pi / 180)
    lon = X / s / DG2M + LON0
    lat = Y / DG2M + LAT0

    return lat, lon
