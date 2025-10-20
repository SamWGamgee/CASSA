def dct_decom(x, y, thresh):
    dctcoeff = dct(y, norm='ortho')
    co = abs(dctcoeff)
    sorti = np.argsort(co)[::-1]
    if isinstance(thresh, float):
        noise = np.std(co[sorti][15:])
        ind = np.where(co>thresh*noise)
    elif isinstance(thresh, int): ind = sorti[:thresh]
    coeffs = dctcoeff[ind]
    return coeffs, ind

def dct_recon(coeffs, ind, nchan):
    co = np.zeros(nchan)
    co[ind] = coeffs
    return idct(co, norm='ortho')

def dct_recon_all(Co):
    F,C,I = Co['nu'], Co['dctc'], Co['dcti']
    recons = np.zeros((2,len(F),2,2,C.shape[-1]))
    for i in range(2):
        for j in range(2):
            for k in range(C.shape[-1]):
                for p in range(2):
                    recons[p,:,i,j,k] = dct_recon(C[p,:,i,j,k], list(map(int,I[p,:,i,j,k])), len(F))
    return recons

def dctise_all(data, nu, thresh):
    recons = np.zeros((2,len(nu),2,2,data.shape[-1]))
    coeffs = np.zeros((2,thresh,2,2,data.shape[-1]))
    inds = np.zeros((2,thresh,2,2,data.shape[-1]))
    for i in range(2):
        for j in range(2):
            for k in range(data.shape[-1]):
                for p in range(2):
                    d = data[p,:,i,j,k]
                    coeff, ind = dct_decom(nu, d, thresh)
                    recon = dct_recon(coeff, ind, len(nu))
                    coeffs[p,:,i,j,k] = coeff
                    inds[p,:,i,j,k] = ind
                    recons[p,:,i,j,k] = recon
    return recons, coeffs, indsS