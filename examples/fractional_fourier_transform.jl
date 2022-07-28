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

$(@bind s Slider(-3:0.001:3, show_value=true))
"

# ╔═╡ 7c445baa-d970-4954-a3dc-df828971bfd7
[gray_show(log1p.(abs.(ft(img)))) gray_show(log1p.(abs.(sqrt(length(img)) .* frfft(img, s, shift=true))))]

# ╔═╡ 50a798b3-e3b6-47bc-98a5-fffe47053976
[gray_show(log1p.(abs.(angle.(ft(img))))) gray_show(log1p.(abs.(angle.(sqrt(length(img)) .* frfft(img, s, shift=true)))))]

# ╔═╡ 1915c023-69cf-4d18-90cb-b47465dbef69
begin
	plot(log1p.(abs.(ft(img)[(end+begin)÷2+1,:] ./ sqrt(length(img)))))
	plot!(log1p.(abs.(frfft(img, s)[(end+begin)÷2+1,:])))
end

# ╔═╡ 3109fc21-50c6-46e6-850d-add6f54872d7
begin
	plot(abs.(imag.(ft(img)[(end+begin)÷2+1,:] ./ sqrt(length(img)))) .|> log1p)
	plot!(abs.(imag.(frfft(img, 0.99999999999)[(end+begin)÷2+1,:])) .|> log1p)
end

# ╔═╡ 227ae9a3-9387-4ac3-b391-e2a78ce40d49
begin
	plot((real.(ft(img)[(end+begin)÷2+1,200:300] ./ sqrt(length(img)))))
	plot!((real.(frfft(img, s)[(end+begin)÷2+1,200:300])))
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

# ╔═╡ 37ebf4d8-28fa-4d0b-929c-5df4c9f418e0
md"## Gaussian Propagation"

# ╔═╡ fab2b38f-7a93-438e-a1f9-9e58709aec2e
x = -256:256

# ╔═╡ 02708a88-14ce-45cc-8d40-71a74bc5a56d
amp = exp.(-(x.^2 .+ x'.^2) ./ 100);

# ╔═╡ 77807bbe-a33a-4d65-8e06-446ad368784f
phase_term = exp.(1im .* x .* 2π ./ 5)

# ╔═╡ 696a77b2-a904-4cf8-805e-b66621dbbb8f
field = amp .* phase_term;

# ╔═╡ e556fd79-00b1-468b-85eb-79a08edfc5bf
gray_show(amp)

# ╔═╡ 4e53efc4-de25-4b97-8dc8-985d56b8bc67
complex_show(field)

# ╔═╡ 14d21206-ba85-485c-b1c1-2fca106a7169
complex_show(ft(field))

# ╔═╡ 4dcf3db5-6d37-4a09-a161-4af53ffc91ec
@bind f2 Slider(-1:0.01:2, show_value=true)

# ╔═╡ 1fe0d80f-664b-4b9f-9ff3-95f0d00e32d5
complex_show(frfft(field, f2))

# ╔═╡ Cell order:
# ╠═a696290a-0122-11ed-01e5-a39256aed683
# ╟─55894157-a2d1-4567-99a8-a052d5335dd1
# ╠═cc0f07d5-9192-48cd-9a90-50eebb06783f
# ╠═18b0700c-20fc-4e58-8950-ca09fe34ea19
# ╠═459649ed-ca70-426e-8273-97b146b5bcd5
# ╟─4371cfbf-a3b3-45dc-847b-019994fbb234
# ╠═d90b7f67-4166-44fa-aab7-de2c4f38fc00
# ╠═24901666-4cc4-497f-a6ff-68c3e7ead629
# ╠═7c445baa-d970-4954-a3dc-df828971bfd7
# ╠═50a798b3-e3b6-47bc-98a5-fffe47053976
# ╠═1915c023-69cf-4d18-90cb-b47465dbef69
# ╠═3109fc21-50c6-46e6-850d-add6f54872d7
# ╠═227ae9a3-9387-4ac3-b391-e2a78ce40d49
# ╟─abff911a-e10d-4311-955a-7afc4e0d344c
# ╠═bae3c5b7-8964-493b-9e7b-d343e092219c
# ╠═07d2b3b6-3584-4c64-9c4a-138beb3d6b88
# ╠═1839f03e-6add-4c85-b6fd-9035656ed86c
# ╠═37ebf4d8-28fa-4d0b-929c-5df4c9f418e0
# ╠═fab2b38f-7a93-438e-a1f9-9e58709aec2e
# ╠═02708a88-14ce-45cc-8d40-71a74bc5a56d
# ╠═77807bbe-a33a-4d65-8e06-446ad368784f
# ╠═696a77b2-a904-4cf8-805e-b66621dbbb8f
# ╠═e556fd79-00b1-468b-85eb-79a08edfc5bf
# ╠═4e53efc4-de25-4b97-8dc8-985d56b8bc67
# ╠═14d21206-ba85-485c-b1c1-2fca106a7169
# ╠═4dcf3db5-6d37-4a09-a161-4af53ffc91ec
# ╠═1fe0d80f-664b-4b9f-9ff3-95f0d00e32d5
