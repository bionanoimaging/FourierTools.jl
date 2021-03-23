## for testing
using FourierTools
using FFTW, TestImages, Colors # , PaddedViews
img = testimage("resolution_test_512.tif"); # fullname
# imgg = Gray.(img);
mat0 = convert(Array{Float64}, img);
mat=mat0[1:511,1:512];

vv(mat)
newsize=(1024,500)
@time res = ft_resize(mat, newsize; keep_complex=true);
maximum(imag(res))
vv(res)
@time res = rft_resize(mat, newsize);
vv(res)

# w=ft(mat)
# q=ft_pad(w,newsize)
# r=ift(q)
