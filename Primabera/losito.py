from numpy import pi
import subprocess
import sys
import os

dec_deg = 52.9

# if synthms_path is None:
#     raise FileNotFoundError("Could not find synthms in known locations.")
    
def generate_ms(
    synthms_path,
    startime=4.92183348e09,
    intime=600,
    minute=10,
    ra_deg=180,
    dec_deg=dec_deg,
    station="HBA",
    fmin=120e6,
    fmax=120e6,
    nchan=4
):
    """
    Generate a Measurement Set using LoSiTo's synthms tool.
    
    Parameters
    ----------
    startime : float
        Start time in seconds since epoch (default: 4.92183348e09).
    intime : int
        Total observation time in seconds.
    minute : int
        Integration time per step (minutes).
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    station : str
        LOFAR station ("HBA" or "LBA").
    fmin, fmax : float
        Frequency range in Hz.
    nchan : int
        Number of channels per subband.
    synthms_path : str
        Path to the synthms binary.
    """

    # Output MS name auto-generated from dec_deg
    name = f"synthms_{station.lower()}_dec{dec_deg}"

    # Get Python interpreter
    python_bin = sys.executable

    # Build command
    cmd = [
        python_bin,
        synthms_path,
        "--name", name,
        "--start", str(startime),
        "--tobs", str(intime),
        "--ra", str(ra_deg * pi / 180),
        "--dec", str(dec_deg * pi / 180),
        "--station", station,
        "--tres", str(minute * 60),
        "--minfreq", str(fmin),
        "--maxfreq", str(fmax),
        "--chanpersb", str(nchan),
    ]

    # Run
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("❌ synthm file not found. Breaking function.")
        return 
    except subprocess.CalledProcessError as e:
        print(f"❌ synthm command failed with exit code {e.returncode}")
        return
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return

    print(f"Measurement Set '{name}' generated successfully!")

    return name
