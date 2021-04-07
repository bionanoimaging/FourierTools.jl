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

# ╔═╡ 0ad6e310-97e2-11eb-28bc-99ff80496ecb
using Revise, FourierTools, ImageCore, TestImages, PlutoUI, ImageShow

# ╔═╡ 19339dd6-97e2-11eb-2449-05f0a5ca2f9d
begin
	img = Float32.(testimage("fabio_512_gray"))
	z = zeros(Float32, (768, 768))
	FourierTools.center_set!(z, img)
end

# ╔═╡ 1e72338e-97e2-11eb-0ca1-374f5ab2d2da
@bind Δ Slider(-1024:1:1024)

# ╔═╡ 1aeea42c-97e2-11eb-3af3-d377949794ed
md"Δ=$Δ"

# ╔═╡ 3752a30c-97e2-11eb-3469-dbf857775870
Gray.(FourierTools.shear(z, Δ))

# ╔═╡ 39b3ba50-97e2-11eb-3153-9757657de483
Gray.(FourierTools.shear(img, Δ, 2, 1))

# ╔═╡ Cell order:
# ╠═0ad6e310-97e2-11eb-28bc-99ff80496ecb
# ╠═19339dd6-97e2-11eb-2449-05f0a5ca2f9d
# ╠═1aeea42c-97e2-11eb-3af3-d377949794ed
# ╠═1e72338e-97e2-11eb-0ca1-374f5ab2d2da
# ╠═3752a30c-97e2-11eb-3469-dbf857775870
# ╠═39b3ba50-97e2-11eb-3153-9757657de483
