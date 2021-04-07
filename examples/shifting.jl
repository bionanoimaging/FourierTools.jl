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

# ╔═╡ f66e452c-97db-11eb-23fc-bd95da10e2d6
using Revise, FourierTools, Plots, FFTW, TestImages, PlutoUI

# ╔═╡ 8ea180a2-97dc-11eb-2276-43d5c21fc168
md"## Minimal Example"

# ╔═╡ 3b40d714-97dc-11eb-1d06-d3077fede734
begin
	arr = [1.0 2.0 3.0; 4.0 5.0 6.0]
	shift!(arr, (1, 2))
end

# ╔═╡ 48e01ee8-97dc-11eb-2330-93089d4b14b4
begin
	arr2 = [0, 1.0, 0.0, 1.0]
	shift!(arr2, 0.5)
end

# ╔═╡ 882699d8-97dc-11eb-1d71-131242e03e5f
md"# Subpixel shift"

# ╔═╡ 94433620-97dc-11eb-34df-6b93b05824cf
begin
	f(x) = cos(4π * x / 30)
	x1 = 1:30
	x2 = x1 .+ 3
end

# ╔═╡ f682339c-97dc-11eb-14c6-59570cc41c0f
@bind offset Slider(-5:0.01:5)

# ╔═╡ 3610c3aa-97dd-11eb-0053-733c17d8e7c3
md"offset=$offset"

# ╔═╡ 942ab232-97dc-11eb-18d2-d33374a89efa
begin
	y1 = f.(x1)
	y2 = f.(x2)
	y3 = shift(y2, tuple(offset))
end

# ╔═╡ d7d7fe18-97dc-11eb-1800-d1dc9ba78574
begin
	plot(y1, label="Original signal")
	plot!(y2, label="Shifted signal")
	plot!(y3, label="Fourier shifted with $offset")
end

# ╔═╡ 880650ec-97dc-11eb-37ef-451f1431c32f
md"## Image Shifting"

# ╔═╡ 5167d98e-97dc-11eb-38ef-ebe0005635d5
img = testimage("fabio_512_gray")

# ╔═╡ 5a1a8d06-97dc-11eb-3dec-452bd1fb2d65
Gray.(shift(Float64.(img), (100.12, 0.1)))

# ╔═╡ 7af79a8c-97dc-11eb-107e-2f25d880eb16
Gray.(shift(Float64.(img), (100.12, -100)))

# ╔═╡ Cell order:
# ╠═f66e452c-97db-11eb-23fc-bd95da10e2d6
# ╟─8ea180a2-97dc-11eb-2276-43d5c21fc168
# ╠═3b40d714-97dc-11eb-1d06-d3077fede734
# ╠═48e01ee8-97dc-11eb-2330-93089d4b14b4
# ╠═882699d8-97dc-11eb-1d71-131242e03e5f
# ╠═94433620-97dc-11eb-34df-6b93b05824cf
# ╠═3610c3aa-97dd-11eb-0053-733c17d8e7c3
# ╠═f682339c-97dc-11eb-14c6-59570cc41c0f
# ╠═942ab232-97dc-11eb-18d2-d33374a89efa
# ╠═d7d7fe18-97dc-11eb-1800-d1dc9ba78574
# ╟─880650ec-97dc-11eb-37ef-451f1431c32f
# ╠═5167d98e-97dc-11eb-38ef-ebe0005635d5
# ╠═5a1a8d06-97dc-11eb-3dec-452bd1fb2d65
# ╠═7af79a8c-97dc-11eb-107e-2f25d880eb16
