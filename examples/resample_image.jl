### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 64af5a38-490a-11eb-34bc-37017a602974
using Revise, FourierTools, Colors, PlutoUI, ImageShow

# ╔═╡ aec408dc-490a-11eb-0d5f-b167a9599994
md"#### Image Interpolation with FFT based sinc interpolation"

# ╔═╡ 18d15d82-50ee-11eb-1a86-abae54ad0050
begin
	N = 129
	x = range(-10.0, 10.0, length=33)[1:end-1]
	x_exact = range(-10.0, 10.0, length=513)[1:end-1]
	y = x'
	y_exact = x_exact'
	arr = sinc.(sqrt.(x .^2 .+ y .^2))
	arr_exact = sinc.(sqrt.(x_exact .^2 .+ y_exact .^2))
	arr_interp512 = resample(arr[1:end, 1:end], (512, 512));
end

# ╔═╡ 0a52fc98-76eb-11eb-2981-19c696ccd106
Gray.(arr)

# ╔═╡ fd0a8c24-504b-11eb-3e44-75fb8475f9da
md"### Slider to switch between images
"

# ╔═╡ 9245616e-490b-11eb-1648-c56a0155a111
img_all = cat(arr_exact, arr_interp512, dims=3);

# ╔═╡ 4c3e9134-490b-11eb-03a4-4d58d4d942da
md"
$(@bind index Slider(1:2))
"

# ╔═╡ 6a0cbf10-490b-11eb-10bd-359a3425da42
Gray.(img_all[:, :, index])#[100:300, 100:300, index])

# ╔═╡ Cell order:
# ╠═64af5a38-490a-11eb-34bc-37017a602974
# ╟─aec408dc-490a-11eb-0d5f-b167a9599994
# ╠═18d15d82-50ee-11eb-1a86-abae54ad0050
# ╠═0a52fc98-76eb-11eb-2981-19c696ccd106
# ╟─fd0a8c24-504b-11eb-3e44-75fb8475f9da
# ╠═9245616e-490b-11eb-1648-c56a0155a111
# ╟─4c3e9134-490b-11eb-03a4-4d58d4d942da
# ╠═6a0cbf10-490b-11eb-10bd-359a3425da42
