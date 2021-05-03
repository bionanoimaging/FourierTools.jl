
"""
    hfft(U, ρ, ψ)

`U` being the electrical field.
`rho` being the coordinates.
`ψ` is a function taking a vector of length 2 as input as coordinates.
`ψ` is the slowly varying phase. (`ψ.(ρ)` would return the value of the phase on a grid.


"""
function hfft(U, ρ, ψ; ϕ=nothing, λ0=550f-9, Δz=1f-3)
    # note the . to apply gradient and hessian elementwise
    κ = ForwardDiff.gradient.(Ref(ψ), ρ)
    Hψ = ForwardDiff.hessian.(Ref(ψ), ρ)
    a = _hfft_a.(Hψ)
    ψ_of_ρ = ψ.(ρ)
    if isnothing(ϕ)
        Ṽ = a .* U .* exp.(1im .* ψ_of_ρ .- 1im .* dot.(κ, ρ))
    else
        full_phase = ψ_of_ρ .- dot.(κ, ρ) .+ Δz .* sqrt.(complex.(eltype(U)(2π)^2 ./ λ0^2 .- dot.(κ, κ)))
        Ṽ = a .* U .* exp.(1im .* full_phase)
    end
    return Ṽ, κ
end

function _hfft_a(Hψ)
    if iszero(Hψ[2, 2])
        return 1 / abs(Hψ[2, 1])
    else
        c1 = 1im / Hψ[2, 2]
        c2_num = -1im * Hψ[2, 2]
        c2_denom = Hψ[2, 1]^2 - Hψ[2,2] * Hψ[1, 1]
        return √(c1) * √(c2_num / c2_denom)
    end
end
