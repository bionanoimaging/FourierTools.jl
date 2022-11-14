### A Pluto.jl notebook ###
# v0.19.13

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

# ╔═╡ 459649ed-ca70-426e-8273-97b146b5bcd5
using FourierTools, FFTW, NDTools, TestImages, Colors, ImageShow, PlutoUI, IndexFunArrays, Plots

# ╔═╡ 55894157-a2d1-4567-99a8-a052d5335dd1
begin
	
	"""
	    simshow(arr; set_one=false, set_zero=false,
	                f=nothing, γ=1)
	Displays a real valued array . Brightness encodes magnitude.
	Works within Jupyter and Pluto.
	# Keyword args
	The transforms are applied in that order.
	* `set_zero=false` subtracts the minimum to set minimum to 1
	* `set_one=false` divides by the maximum to set maximum to 1
	* `f` applies an arbitrary function to the abs array
	* `γ` applies a gamma correction to the abs 
	* `cmap=:gray` applies a colormap provided by ColorSchemes.jl. If `cmap=:gray` simply `Colors.Gray` is used
	    and with different colormaps the result is an `Colors.RGB` element type
	"""
	function simshow(arr::AbstractArray{T};
	                 set_one=true, set_zero=false,
	                 f = nothing,
	                 γ = one(T),
	                 cmap=:gray) where {T<:Real}
	    arr = set_zero ? arr .- minimum(arr) : arr
	
	    if set_one
	        m = maximum(arr)
	        if !iszero(m)
	            arr = arr ./ maximum(arr)
	        end
	    end
	
	    arr = isnothing(f) ? arr : f(arr)
	
	    if !isone(γ)
	        arr = arr .^ γ
	    end
	
	    if cmap == :gray
	        Gray.(arr)
	    else
	        get(colorschemes[cmap], arr)
	    end
	end
	
	
	"""
	    simshow(arr)
	Displays a complex array. Color encodes phase, brightness encodes magnitude.
	Works within Jupyter and Pluto.
	# Keyword args
	The transforms are applied in that order.
	* `f` applies a function `f` to the array.
	* `absf` applies a function `absf` to the absolute of the array
	* `absγ` applies a gamma correction to the abs 
	"""
	function simshow(arr::AbstractArray{T};
	                 f=nothing,
	                 absγ=one(T),
	                 absf=nothing) where (T<:Complex)
	
	    if !isnothing(f)
	        arr = f(arr)
	    end
	
	    Tr = real(T)
	    # scale abs to 1
	    absarr = abs.(arr)
	    absarr ./= maximum(absarr)
	
	    if !isnothing(absf)
	        absarr .= absf(absarr)
	    end
	
	    if !isone(absγ)
	        absarr .= absarr .^ absγ
	    end
	
	    angarr = angle.(arr) ./ Tr(2pi) * Tr(360)
	
	    HSV.(angarr, one(Tr), absarr)
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
[simshow(abs.(ft(img)), γ=0.2) simshow(sqrt(length(img)) .* abs.(frfft(img, s, shift=true)), γ=0.2)]

# ╔═╡ 1915c023-69cf-4d18-90cb-b47465dbef69
begin
	plot(log1p.(abs.(ft(img)[(end+begin)÷2+1,:] ./ sqrt(length(img)))))
	plot!(log1p.(abs.(frfft(img, s)[(end+begin)÷2+1,:])))
end

# ╔═╡ 3109fc21-50c6-46e6-850d-add6f54872d7
begin
	plot((imag.(ft(img)[(end+begin)÷2+1,:] ./ sqrt(length(img)))))
	plot!((imag.(frfft(img, 0.99999999999)[(end+begin)÷2+1,:])))
end

# ╔═╡ 284cd6f2-1ee3-4923-afa6-ea57e93b28a7
begin
	plot((angle.(ft(img)[(end+begin)÷2+1,:] ./ sqrt(length(img)))))
	plot!((angle.(frfft(img, 0.99999999999)[(end+begin)÷2+1,:])))
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
r = box(Float64, (301,), (201,))#.+ 0.4 .* randn((301,))

# ╔═╡ 5655dc10-f4e9-4765-9a89-ac9702864de1
plot(abs.(ft(r)))

# ╔═╡ 07d2b3b6-3584-4c64-9c4a-138beb3d6b88
@bind s2 Slider(-5:0.001:5, show_value=true)

# ╔═╡ 1839f03e-6add-4c85-b6fd-9035656ed86c
begin
	plot(real.(frfft(r, s2, shift=true)), label="FourierTools")
	#plot!(imag.(frfft(frfft(r, s2/2, shift=true), s2/2, shift=true)), label="FourierTools 2 Step")

	plot!(real.(FractionalTransforms.frft(r, s2)), label="FractionalTransforms")
	#plot!(imag.(FractionalTransforms.frft(ft(r) ./ sqrt(length(r)), -1+s2)), label="FractionalTransforms")

	#plot!(imag.(FractionalTransforms.frft(r, s2)))

	#plot!(real.(FractionalTransforms.frft(r, s2)))
end

# ╔═╡ f3cb2153-a7b3-46ed-adbb-038a812b6a81
begin

	#plot(abs.(frfft(r, s2, shift=true, p_change=false)))
	plot(abs.(frfft(r, s2, shift=true, p_change=true)))

	#plot!(imag.(FractionalTransforms.frft(r, s2)), label="FractionalTransforms")
	#plot!(imag.(FractionalTransforms.frft(ft(r) ./ sqrt(length(r)), -1+s2)), label="FractionalTransforms")
	plot!(abs.(FractionalTransforms.frft(r, s2)))

	#plot!(real.(FractionalTransforms.frft(r, s2)))
end

# ╔═╡ 37ebf4d8-28fa-4d0b-929c-5df4c9f418e0
md"## Gaussian Propagation"

# ╔═╡ fab2b38f-7a93-438e-a1f9-9e58709aec2e
x = -256:256

# ╔═╡ 02708a88-14ce-45cc-8d40-71a74bc5a56d
amp = exp.(-(x.^2 .+ x'.^2) ./ 3000);

# ╔═╡ 77807bbe-a33a-4d65-8e06-446ad368784f
phase_term = exp.(1im .* x .* 2π ./ 5 .+ 1im .* x'.^2);

# ╔═╡ 696a77b2-a904-4cf8-805e-b66621dbbb8f
field = amp .* phase_term;

# ╔═╡ 4e53efc4-de25-4b97-8dc8-985d56b8bc67
simshow(field)

# ╔═╡ 3fa82b96-e701-40d1-89c2-8f71038b6d05
simshow(ft(field))

# ╔═╡ 4dcf3db5-6d37-4a09-a161-4af53ffc91ec
@bind f2 Slider(-4:0.001:4, show_value=true)

# ╔═╡ e4db42df-cfe9-4ae0-91f6-5672707d87d5
simshow(frfft(field, f2))

# ╔═╡ 67e0e7dc-4692-451c-97d0-742ab5df3853
rev2(x) = ifftshift(reverse(fftshift(x)))

# ╔═╡ a52deb8f-64e6-46ab-b77e-13b35a20f17c
rev2([simshow(frfft(field, f2));;; simshow(frfft(ift(field), f2-1))][:, :, 2])

# ╔═╡ ce65aa2c-e558-434b-aa8a-2268c47f5684
md"### Comparison with two step FRFT with half the order"

# ╔═╡ e0529213-f2c9-49e4-b2fe-96bbd16a77b7
@bind f3 Slider(-4:0.001:4, show_value=true)

# ╔═╡ 1fe0d80f-664b-4b9f-9ff3-95f0d00e32d5
simshow(frfft(field, f3, p_change=false))

# ╔═╡ e5f32874-5f98-4825-824a-780764e8ef91
simshow(frfft(frfft(field, f3/2),f3/2))

# ╔═╡ Cell order:
# ╠═a696290a-0122-11ed-01e5-a39256aed683
# ╟─55894157-a2d1-4567-99a8-a052d5335dd1
# ╠═18b0700c-20fc-4e58-8950-ca09fe34ea19
# ╠═459649ed-ca70-426e-8273-97b146b5bcd5
# ╟─4371cfbf-a3b3-45dc-847b-019994fbb234
# ╠═d90b7f67-4166-44fa-aab7-de2c4f38fc00
# ╟─24901666-4cc4-497f-a6ff-68c3e7ead629
# ╠═7c445baa-d970-4954-a3dc-df828971bfd7
# ╠═1915c023-69cf-4d18-90cb-b47465dbef69
# ╠═3109fc21-50c6-46e6-850d-add6f54872d7
# ╠═284cd6f2-1ee3-4923-afa6-ea57e93b28a7
# ╠═227ae9a3-9387-4ac3-b391-e2a78ce40d49
# ╟─abff911a-e10d-4311-955a-7afc4e0d344c
# ╠═bae3c5b7-8964-493b-9e7b-d343e092219c
# ╠═5655dc10-f4e9-4765-9a89-ac9702864de1
# ╠═1839f03e-6add-4c85-b6fd-9035656ed86c
# ╠═07d2b3b6-3584-4c64-9c4a-138beb3d6b88
# ╠═f3cb2153-a7b3-46ed-adbb-038a812b6a81
# ╟─37ebf4d8-28fa-4d0b-929c-5df4c9f418e0
# ╠═fab2b38f-7a93-438e-a1f9-9e58709aec2e
# ╠═02708a88-14ce-45cc-8d40-71a74bc5a56d
# ╠═77807bbe-a33a-4d65-8e06-446ad368784f
# ╠═696a77b2-a904-4cf8-805e-b66621dbbb8f
# ╠═4e53efc4-de25-4b97-8dc8-985d56b8bc67
# ╠═3fa82b96-e701-40d1-89c2-8f71038b6d05
# ╟─4dcf3db5-6d37-4a09-a161-4af53ffc91ec
# ╠═e4db42df-cfe9-4ae0-91f6-5672707d87d5
# ╠═67e0e7dc-4692-451c-97d0-742ab5df3853
# ╠═a52deb8f-64e6-46ab-b77e-13b35a20f17c
# ╟─ce65aa2c-e558-434b-aa8a-2268c47f5684
# ╟─e0529213-f2c9-49e4-b2fe-96bbd16a77b7
# ╠═1fe0d80f-664b-4b9f-9ff3-95f0d00e32d5
# ╠═e5f32874-5f98-4825-824a-780764e8ef91
