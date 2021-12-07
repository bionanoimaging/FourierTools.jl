export czt, iczt

"""
    czt_1d(xin , scaled , d)

Chirp z transform along a single direction d of an ND array `xin` into the ND array 'xout'.
Note that xin and xout can be the same array for inplace operations.
Note that the result type is defined by `eltype(xin)` and not by `scales`.
This code is based on a 2D Matlab version of the CZT, written by H. Gross et al.
    
#References: Rabiner, Schafer, Rader, The Cirp z-Transform Algorithm, IEEE Trans AU 17(1969) p. 86
"""
function czt_1d(xin, scaled, d)
    sz=size(xin)
    # returns the real datatype
    rtype = real(eltype(xin))  
    scaled = rtype(scaled)
    dsize = sz[d]
    nn = (0:dsize-1)
    # 2N +1 new k-space positions
    kk = ((-dsize+1):(dsize-1)) 
    kk2 = rtype.((kk .^ 2) ./ 2)
    
    # scalar factor  
    w = cispi(-2/(dsize*scaled)) 
    # this is needed to center the array according to the center pixel convention
    half_pix_shift = (2*(dsize÷2))/dsize 
    # scalar factor. The correction factor to the right was introduced to be centered correctly on the pixel as ft is.
    a = cispi(-1/scaled * half_pix_shift) 
    ww = w .^ kk2
    aa = a .^ (-nn)
    # is a 1d list of factors. This defines the shift in Fourier space (centering of frequencies)
    aa = aa .* ww[dsize .+ nn]
    to_fft = NDTools.select_region(1 ./ ww[1:(2*dsize-1)], new_size=(2*dsize,), center=(dsize+1,))
    # return tofft
    # is always 1d (small array)
    fv = fft(to_fft)

    y = xin .* reorient(aa,d)
    # twice the size along direction d
    nsz = sz .* NDTools.single_dim_size(d,2,length(sz)) 
    to_fft = NDTools.select_region(y, new_size=nsz, center=nsz.÷2 .+1)
    # convolve on a larger grid along one dimension
    g = ifft(fft(to_fft, d) .* reorient(fv,d), d) 
    # return g
    oldctr = sz[d]÷2 + 1
    newctr = size(g) .÷ 2 .+1
    ctr = ntuple(md -> (md == d) ? newctr[md] + oldctr - 2 : newctr[md], length(newctr))

    # This is to deal with a strange phase shift appearing for odd-sized arrays
    if isodd(dsize) 
        extra_phase = (2*dsize-2)/(2*dsize) # 5: 12 / 15, 7: 12/14, 9: 16/18, 11: 20/22
    else
        extra_phase = 1
    end
    # is a 1d list of factors
    fak =  ww[dsize:(2*dsize-1)] .* cispi.(ramp(rtype,1,dsize, scale=1/scaled * extra_phase))
    # return select_region(g, new_size=sz,center=ctr)
    xout = select_region(g, new_size=sz,center=ctr) .* reorient(fak,d)

    # this is a fix to deal with the problem that imaginary numbers are appearing for even-sized arrays, caused by the first entry
    if iseven(dsize) && (scaled>1.0) 
        # a `scaled` of one meas this is an ordinary Fourier-transformation without zoom, which needs to keep the highes frequency value
        midp = dsize÷2+1
        for o in 1+mod(midp,2):2:dsize
            slice(xout,d, o) .-= slice(xin,d, 1) .* (1im).^mod(o-midp,4)
        end
    end
    return xout
    # xout .= g[dsize:(2*dsize-1)] .* reorient(fak,d)
end

"""
    czt(xin , scale, dims=1:length(size(xin)))
Chirp z transform of the ND array `xin`
This code is based on a 2D Matlab version of the CZT, written by H. Gross.
The tuple `scale` defines the zoom factors in the Fourier domain. Each has to be bigger than one.

#See also: iczt, czt_1d
    
#References: Rabiner, Schafer, Rader, The Cirp z-Transform Algorithm, IEEE Trans AU 17(1969) p. 86

#Example:
```jdoctest
julia> using IndexFunArrays

julia> sz = (10,10);

julia> xin = disc(sz,4)
10×10 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0

julia> xft = czt(xin,(1.2,1.3));

julia> zoomed = real.(ift(xft))
10×10 Matrix{Float64}:
 -0.0197423    0.0233008  -0.0449251   0.00295724   0.205593  -0.166546   0.205593   0.00295724  -0.0449251   0.0233008
  0.0239759   -0.028264    0.0541186  -0.0116475   -0.261294   0.312719  -0.261294  -0.0116475    0.0541186  -0.028264
 -0.0569       0.0666104  -0.122277    0.140354     0.78259    1.34381    0.78259    0.140354    -0.122277    0.0666104
  0.00540611  -0.0117886   0.0837357   1.30651      1.8283     1.50127    1.8283     1.30651      0.0837357  -0.0117886
  0.11892     -0.147731    0.368046    1.76537      1.33218    1.66119    1.33218    1.76537      0.368046   -0.147731
 -0.00389861   0.0145979   1.21842     1.52989      1.67375    1.543      1.67375    1.52989      1.21842     0.0145979
  0.11892     -0.147731    0.368046    1.76537      1.33218    1.66119    1.33218    1.76537      0.368046   -0.147731
  0.00540611  -0.0117886   0.0837357   1.30651      1.8283     1.50127    1.8283     1.30651      0.0837357  -0.0117886
 -0.0569       0.0666104  -0.122277    0.140354     0.78259    1.34381    0.78259    0.140354    -0.122277    0.0666104
  0.0239759   -0.028264    0.0541186  -0.0116475   -0.261294   0.312719  -0.261294  -0.0116475    0.0541186  -0.028264
```
"""
function czt(xin::Array{T,N}, scale, dims=1:length(size(xin)))::Array{complex(T),N} where {T,N}
    xout = xin
    for d in dims
        xout = czt_1d(xout, scale[d], d)
    end
    return xout
end

"""
    iczt(xin , scale, dims=1:length(size(xin)))
Inverse chirp z transform of the ND array `xin`
This code is based on a 2D Matlab version of the CZT, written by H. Gross.
The tuple `scale` defines the zoom factors in the Fourier domain. Each has to be bigger than one.
    
#References: Rabiner, Schafer, Rader, The Cirp z-Transform Algorithm, IEEE Trans AU 17(1969) p. 86

#See also: czt, czt_1d

#Example: 
```jdoctest
julia> using IndexFunArrays

julia> sz = (10,10);

julia> xin = disc(sz,4)
10×10 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0

julia> xft = ft(xin);

julia> iczt(xft,(1.2,1.3))
10×10 Matrix{ComplexF64}:
 0.00648614+0.0213779im  0.0165456+0.0357733im  0.0389356+0.0482465im  -0.235491-0.156509im    …  0.178484-0.0730099im  -0.245418-5.88331e-5im  0.0471654-0.0814548im  0.0141525+0.0734892im
  -0.104602-0.160481im   -0.163859-0.125535im    0.495205+0.135059im    0.660335+0.00736904im     0.764045-0.0497007im    0.67753+0.263814im      0.48095-0.0775406im  -0.159713-0.0637132im
   0.139304+0.111348im    0.454973+0.106869im    0.970263-0.0249785im    1.25999-0.166495im        1.07328-0.0481437im    1.24013-0.14664im      0.986722-0.0414382im   0.450186+0.111656im
  -0.035645-0.0311352im    1.03899-0.0589268im     1.1463-0.0940003im   0.790545+0.283668im       0.994255+0.134865im     0.80774-0.0124851im     1.13205+0.151519im     1.04314-0.130321im
   0.292575+0.0853233im   0.929883+0.0687029im    1.06514-0.0649952im   0.989483-0.019913im        1.02311+0.018235im    0.979555-0.136654im      1.07337+0.0317868im    0.92749+0.0405597im
    1.12254-0.0464723im    1.03467-0.0239316im    0.92709+0.0822984im     1.0521-0.0992709im   …  0.983655-0.0663123im     1.0521+0.0992709im     0.92709-0.0822984im    1.03467+0.0239316im
   0.287928-0.0306724im    0.92749-0.0405597im    1.07337-0.0317868im   0.979555+0.136654im        1.01648+0.0597475im   0.989483+0.019913im      1.06514+0.0649952im   0.929883-0.0687029im
 -0.0275957+0.169775im     1.04314+0.130321im     1.13205-0.151519im     0.80774+0.0124851im       1.00574+0.0629632im   0.790545-0.283668im       1.1463+0.0940003im    1.03899+0.0589268im
   0.130009-0.120643im    0.450186-0.111656im    0.986722+0.0414382im    1.24013+0.14664im         1.06002+0.0348813im    1.25999+0.166495im     0.970263+0.0249785im   0.454973-0.106869im
 -0.0965531+0.0404296im  -0.159713+0.0637132im    0.48095+0.0775406im    0.67753-0.263814im        0.77553-0.121603im    0.660335-0.00736904im   0.495205-0.135059im   -0.163859+0.125535im
 ```
"""
function iczt( xin , scale, dims=1:length(size(xin)))
    conj(czt(conj(xin), scale, dims)) / prod(size(xin))
end
