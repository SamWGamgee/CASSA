import numpy as np
import math
fac = math.factorial

# Calculate the radial component of Zernike polynomial (m, n) given a grid of radial coordinates rho.
def zernike_rad(m, n, rho):
    if (n < 0 or m < 0 or abs(m) > n): raise ValueError
    if ((n-m) % 2): return rho*0.0
    pre_fac = lambda k: (-1)**k * fac(n-k) / ( fac(k) * fac( int((n+m)/2) - k ) * fac( int((n-m)/2) - k ) )
    return sum(pre_fac(k) * rho**(n-2.0*k) for k in np.arange((n-m)/2+1).astype(int))

# Calculate Zernike polynomial (m, n) given a grid of radial coordinates rho and azimuthal coordinates phi.
def zernike(m, n, rho, phi):
    if (m > 0): return zernike_rad(m, n, rho) * np.cos(m * phi)
    if (m < 0): return zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return zernike_rad(0, n, rho)

"""
Convert linear Noll index to tuple of Zernike indices. j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike index.
@param [in] j Zernike mode Noll index
@return (n, m) tuple of Zernike indices
@see <https://oeis.org/A176988>.
"""

def noll_to_zern(j):
# add 1 to start from 1
    j += 1
    n = 0
    j1 = j-1
    while (j1 > n):
      n += 1
      j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    return (n, m)

# Calculate Zernike polynomial with Noll coordinate j given a grid of radial coordinates rho and azimuthal coordinates phi.
def zernikel(j, rho, phi):
    nm = noll_to_zern(j)
    m, n = nm[1], nm[0]
    return zernike(m, n, rho, phi)

#Create an unit disk and convert to rho, phi   
def unit_disk(npix=None):
    if npix: nx, ny = npix, npix
    else: nx, ny = img.shape
    grid = (np.indices((nx, ny), dtype=float) - nx/2) / (nx*1./2) # create unit grid [-1,1]
    grid_rho = (grid**2.0).sum(0)**0.5 # rho = sqrt(x^2+y^2)
    grid_phi = np.arctan2(grid[0], grid[1]) # phi = itan(x/y)
    grid_mask = grid_rho <= 1 # boolean array specifying where rho<=1
    return grid_rho, grid_phi, grid_mask #returned grid_rho, grid_phi, grid_mask

def decompose(img, N):
   """Decompose using SVD"""
   grid_rho,grid_phi, grid_mask = unit_disk(img.shape[0]) #defined 

    # Caculate Zernike bases given the maximum Noll index, N
   basis = [zernikel(i,grid_rho, grid_phi)*grid_mask for i in range(N)]

   # Calculate covariance between all Zernike polynomials
   cov_mat = np.array([[np.sum(zerni * zernj) for zerni in basis] for zernj in basis])

   # Invert covariance matrix using SVD (  A x = b  ==>  x =>  x= A^{pseudoinv} b)
   cov_mat_in = np.linalg.pinv(cov_mat)
   
   # Inner product between the img and the Zernike bases
   innerprod = np.array([np.sum(img * zerni) for zerni in basis])
        
   # Dot product between inverse covariance matrix and the innerprod to get the coeffs
   coeffs = np.dot(cov_mat_in, innerprod)
   return coeffs

def truncate(coeffs, thresh):
  """Truncate the coefficients upto the given threshold"""
  sortedindex = np.argsort(np.abs(coeffs))[::-1]
  Ncoeff = coeffs.shape[-1]
  cutoff = int(np.round(Ncoeff*thresh/100.))
        
  #print "Keeping %2.0f %% (N=%s) of the biggest coefficients"%(thresh,cutoff)

  coeffs_trunc = coeffs.copy() # copy of all coeff
  coeffs_trunc[sortedindex[cutoff:]] = 0 # put coeff below threshold to 0

  return coeffs_trunc

# def best_coeffs(C, I):
#    idx = np.argsort(np.abs(C))[::-1][thresh]
#    return C[idx], I[idx]

def reconstruct(dcom, thresh, img):
    grid_rho,grid_phi, grid_mask = unit_disk(img.shape[0])#defined 
    coeffs_trunc = truncate(dcom, thresh)
    recon = np.sum(val * zernikel(i, grid_rho, grid_phi)*grid_mask for (i, val) in enumerate(coeffs_trunc))
    return recon
