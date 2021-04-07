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

# ╔═╡ 82467e1e-97e0-11eb-35bc-b70fe47f7296
using Revise, FourierTools, ImageCore, TestImages, PlutoUI, ImageShow

# ╔═╡ 8f0854f4-97e0-11eb-14b8-7958b0ab243f
begin
	img = Float32.(testimage("fabio_512_gray"))
	z = zeros(Float32, (768, 768))
	FourierTools.center_set!(z, img)
end

# ╔═╡ e43962ba-97e0-11eb-274f-fd6ed85b03ff
@bind ϕ Slider(-180:1:180)

# ╔═╡ dc4917a8-97e0-11eb-341d-25ec7207b748
md"ϕ=$ϕ"

# ╔═╡ a9e968a8-97e0-11eb-0486-3f75f68be3c7
Gray.(FourierTools.rotate(z, ϕ))

# ╔═╡ f1fecc74-97e1-11eb-203e-2949bafae27d
Gray.(FourierTools.rotate(img, ϕ))

# ╔═╡ Cell order:
# ╠═82467e1e-97e0-11eb-35bc-b70fe47f7296
# ╠═8f0854f4-97e0-11eb-14b8-7958b0ab243f
# ╠═dc4917a8-97e0-11eb-341d-25ec7207b748
# ╠═e43962ba-97e0-11eb-274f-fd6ed85b03ff
# ╠═a9e968a8-97e0-11eb-0486-3f75f68be3c7
# ╠═f1fecc74-97e1-11eb-203e-2949bafae27d
