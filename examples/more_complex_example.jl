### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 09637070-a4f8-11eb-3e6d-ab2106c891d3
using TestImages, FourierTools, ImageShow, Colors, IndexFunArrays

# ╔═╡ b0b0700c-a8f0-4f13-b9d2-799db5904b69
md"## More complex example using FourierTools"

# ╔═╡ 2888c718-8300-47be-8716-ca67908e3f32
img = Float32.(testimage("fabio_gray_512"));

# ╔═╡ c77dab69-ea8b-4cd6-8169-f7572e600126
Gray.(img)

# ╔═╡ 2e879e20-bb7b-41f7-8c72-833e959587fd
md"## Put a low pass filter in Fourier space"

# ╔═╡ e1239869-34fb-4984-b72d-0b1eb8d765c8
begin
	low_pass = IndexFunArrays.rr(size(img)) .< 30;
	low_pass_psf = real(iffts(low_pass));
end

# ╔═╡ 0d190942-5888-42b2-85cf-1522b8a2b8c4
Gray.(low_pass)

# ╔═╡ e299a827-50c2-4fee-89b2-96ac95d4f081
md"## Now take only frequencies passing the low pass filter"

# ╔═╡ 56445c4f-69b4-43ce-81a1-809216e79ff5
img_filtered = real(iffts(ffts(img) .* low_pass))

# ╔═╡ e3b63443-5234-4e5b-a074-b91957cffc98
Gray.(img_filtered)

# ╔═╡ 4e82292b-39c2-4cae-a052-2bc2c7be4100
md"## Since the image is clearly bandlimited, we can downsample without loosing data"

# ╔═╡ 72626a88-b95f-45ac-85cc-622195b136ed
img_downsample = resample(img_filtered, (150, 150));

# ╔═╡ 197d57ec-d89e-4d7d-a209-53535f6e8045
Gray.(img_downsample)

# ╔═╡ 1f06fbca-39a2-4c08-84ca-c5a00a524b0e
md"## Downsampling preserves all the information!"

# ╔═╡ d5bae0f3-55d1-41c2-846e-03b4e5c47529
resample(img_downsample, size(img_filtered)) ≈ img_filtered

# ╔═╡ 8eaac2d0-a698-4c16-809f-4c8f61a99675
md"## Use convolution instead of handcrafted convolution"

# ╔═╡ de770699-474d-42f1-a9aa-d5bfdeb2261e
img_filtered2 = conv(img, low_pass_psf)

# ╔═╡ e83ab89a-ea25-4623-a198-503676f445ef
Gray.(img_filtered2)

# ╔═╡ 53546c5d-f718-4618-bcf1-2f6e9cd2c118
img_filtered2 ≈ img_filtered

# ╔═╡ Cell order:
# ╠═b0b0700c-a8f0-4f13-b9d2-799db5904b69
# ╠═09637070-a4f8-11eb-3e6d-ab2106c891d3
# ╠═2888c718-8300-47be-8716-ca67908e3f32
# ╠═c77dab69-ea8b-4cd6-8169-f7572e600126
# ╠═2e879e20-bb7b-41f7-8c72-833e959587fd
# ╠═e1239869-34fb-4984-b72d-0b1eb8d765c8
# ╠═0d190942-5888-42b2-85cf-1522b8a2b8c4
# ╠═e299a827-50c2-4fee-89b2-96ac95d4f081
# ╠═56445c4f-69b4-43ce-81a1-809216e79ff5
# ╠═e3b63443-5234-4e5b-a074-b91957cffc98
# ╠═4e82292b-39c2-4cae-a052-2bc2c7be4100
# ╠═72626a88-b95f-45ac-85cc-622195b136ed
# ╠═197d57ec-d89e-4d7d-a209-53535f6e8045
# ╠═1f06fbca-39a2-4c08-84ca-c5a00a524b0e
# ╠═d5bae0f3-55d1-41c2-846e-03b4e5c47529
# ╠═8eaac2d0-a698-4c16-809f-4c8f61a99675
# ╠═de770699-474d-42f1-a9aa-d5bfdeb2261e
# ╠═e83ab89a-ea25-4623-a198-503676f445ef
# ╠═53546c5d-f718-4618-bcf1-2f6e9cd2c118
