import numpy as np
from everybeam import load_telescope, LOFAR, Options, thetaphi2cart

def allsky_beam(ms_path, time, frequency, station_id, npix, element_id=0, mode='station'):
    mode = "station"  # Change to "element" to obtain the element response

    # Same station0 direction as in python/test
    station0 = np.array([0.655743, -0.0670973, 0.751996])
    is_local = True  # use local coords.
    rotate = False
    response_model = "hamaker"

    # Load telescope
    telescope = load_telescope(ms_path, element_response_model=response_model)
    station_name = telescope.station_name(station_id)

    # Make coordinates for mesh
    x_v = np.linspace(-1.0, 1.0, npix)
    y_v = np.linspace(-1.0, 1.0, npix)

    response = np.empty((y_v.size, x_v.size, 2, 2), dtype=np.cdouble)

    response.fill(np.nan)

    for i, x in enumerate(x_v):
        for j, y in enumerate(y_v):
            if (x**2 + y**2) <= 1.0:
                # Compute theta/phi and resulting direction vector
                theta = np.arcsin(np.sqrt(x * x + y * y))
                phi = np.arctan2(y, x)
                direction = thetaphi2cart(theta, phi)
                if mode == "element":
                    response[j,i,:,:] = telescope.element_response(
                        time,
                        station_id,
                        element_id,
                        frequency,
                        direction,
                        is_local,
                        rotate=rotate,
                    )
                elif mode == "station":
                    response[j,i,:,:] = telescope.station_response(
                        time,
                        station_id,
                        frequency,
                        direction,
                        station0,
                        rotate=rotate,
                    )
                else:
                    raise Exception(
                        "Unrecognized response mode. Must be either station or element"
                    )
    return response

# telescope = eb.load_telescope(ms_path)
# assert type(telescope) == eb.LOFAR

# # Set time and freq for which beam should be evaluated
# time = startime
# freq = fmin

# # Specify the settings of the grid (/image)
# gs = eb.GridSettings()
# gs.width = gs.height = 128
# gs.ra = ra;
# gs.dec = 60. * pi/180.

# gs.dl = gs.dm = 0.3 * np.pi / 180.0

# print(0.3*gs.width)

# gs.l_shift = gs.m_shift = 0.0

# # Get the gridded response for all stations at once
# J_field = telescope.gridded_response(gs, time, freq, station_id)

# J_field.shape
