### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a696290a-0122-11ed-01e5-a39256aed683
begin
	using Pkg;
	Pkg.activate(".")
	using Revise
end

# ╔═╡ cc0f07d5-9192-48cd-9a90-50eebb06783f
Pkg.add("IndexFunArrays")

# ╔═╡ 459649ed-ca70-426e-8273-97b146b5bcd5
using FourierTools, FFTW, NDTools, TestImages, Colors, ImageShow, PlutoUI, Napari, IndexFunArrays, Plots

# ╔═╡ 55894157-a2d1-4567-99a8-a052d5335dd1
begin
	"""
	    complex_show(arr)
	Displays a complex array. Color encodes phase, brightness encodes magnitude.
	Works within Jupyter and Pluto.
	"""
	function complex_show(cpx::AbstractArray{T, N}) where {T<:Complex, N}
	    Tr = real(T)
		ac = abs.(cpx)
	    HSV.(angle.(cpx)./Tr(2pi)*256, ones(Tr,size(cpx)), ac./maximum(ac))
	end
	
	
	"""
	    gray_show(arr; set_one=false, set_zero=false)
	Displays a real gray color array. Brightness encodes magnitude.
	Works within Jupyter and Pluto.
	## Keyword args
	* `set_one=false` divides by the maximum to set maximum to 1
	* `set_zero=false` subtracts the minimum to set minimum to 1
	"""
	function gray_show(arr; set_one=true, set_zero=false)
	    arr = set_zero ? arr .- minimum(arr) : arr
	    arr = set_one ? arr ./ maximum(arr) : arr
	    Gray.(arr)
	end
	
end

# ╔═╡ 18b0700c-20fc-4e58-8950-ca09fe34ea19
import FractionalTransforms

# ╔═╡ 4371cfbf-a3b3-45dc-847b-019994fbb234
md"## Fractional Fourier Transform on a Image"

# ╔═╡ d90b7f67-4166-44fa-aab7-de2c4f38fc00
img = Float32.(testimage("resolution_test_512"));

# ╔═╡ 24901666-4cc4-497f-a6ff-68c3e7ead629
md"
Fractional order

$(@bind s Slider(-1.5:0.01:2, show_value=true))
"

# ╔═╡ 7c445baa-d970-4954-a3dc-df828971bfd7
[gray_show(log1p.(abs.(ft(img)))) gray_show(log1p.(abs.(sqrt(length(img)) .* frfft(img, s))))]

# ╔═╡ 1915c023-69cf-4d18-90cb-b47465dbef69
begin
	plot(log1p.(abs.(ft(img)[(end+begin)÷2+1,:] ./ sqrt(length(img)))))
	plot!(log1p.(abs.(frfft(img, s)[(end+begin)÷2+1,:])))
end

# ╔═╡ abff911a-e10d-4311-955a-7afc4e0d344c
md"## Fractional Fourier Transform on Vector
Comparison with [FractionalTransforms.jl](https://github.com/SciFracX/FractionalTransforms.jl) roughly matches.
"

# ╔═╡ bae3c5b7-8964-493b-9e7b-d343e092219c
r = box(Float64, (101,), (50,)).+ 0.1 .* randn((101,))

# ╔═╡ 07d2b3b6-3584-4c64-9c4a-138beb3d6b88
@bind s2 Slider(-5:0.1:5, show_value=true)

# ╔═╡ 1839f03e-6add-4c85-b6fd-9035656ed86c
begin
	plot(imag.(frfft(r, s2, shift=true)))
	plot!(real.(frfft(r, s2, shift=true)))

	plot!(imag.(FractionalTransforms.frft(r, s2)))
	plot!(real.(FractionalTransforms.frft(r, s2)))
end

# ╔═╡ Cell order:
# ╠═a696290a-0122-11ed-01e5-a39256aed683
# ╟─55894157-a2d1-4567-99a8-a052d5335dd1
# ╠═cc0f07d5-9192-48cd-9a90-50eebb06783f
# ╠═18b0700c-20fc-4e58-8950-ca09fe34ea19
# ╠═459649ed-ca70-426e-8273-97b146b5bcd5
# ╟─4371cfbf-a3b3-45dc-847b-019994fbb234
# ╠═d90b7f67-4166-44fa-aab7-de2c4f38fc00
# ╟─24901666-4cc4-497f-a6ff-68c3e7ead629
# ╠═7c445baa-d970-4954-a3dc-df828971bfd7
# ╠═1915c023-69cf-4d18-90cb-b47465dbef69
# ╟─abff911a-e10d-4311-955a-7afc4e0d344c
# ╠═bae3c5b7-8964-493b-9e7b-d343e092219c
# ╠═07d2b3b6-3584-4c64-9c4a-138beb3d6b88
# ╠═1839f03e-6add-4c85-b6fd-9035656ed86c
