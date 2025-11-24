using InteractiveUtils  # for code_warntype

# --- NEW: report store (full output + location) ---
mutable struct FTReport
    file::String
    line::Int
    call::String
    output::String
    red_lines::Vector{String}
    yellow_lines::Vector{String}
end
const _FT_REPORTS = FTReport[]

# --- counters and helpers ---
mutable struct _FTCounters
    ok::Int
    bad::Int
end
const _FT_STACK  = _FTCounters[]
const _FT_GLOBAL = _FTCounters(0, 0)

function _ft_record(is_bad::Bool)
    ctr = isempty(_FT_STACK) ? _FT_GLOBAL : _FT_STACK[end]
    if is_bad
        ctr.bad += 1
    else
        ctr.ok  += 1
    end
    nothing
end

# Include unicode approximate operators too (and a few arithmetic ops if used in tests)
# const COMP_OPS = Set([:(.+), :(.-), :(.*), :(==), :(!=), :(===), :(!==), :(<), :(<=), :(>), :(>=), :isapprox, :≈, :≉])
const COMP_OPS = Set([:.+, :.-, :.*, :(==), :(!=), :(===), :(!==), :(<), :(<=), :(>), :(>=), :isapprox, :≈, :≉])

# ANSI color detection
const ANSI_YELLOW_RX      = r"\x1b\[[0-9;]*9?3m"         # 33m or 93m
const ANSI_EXT_YELLOW_RX  = r"\x1b\[[0-9;]*38;5;11m"     # common 256-color yellow
const ANSI_RED_RX         = r"\x1b\[[0-9;]*9?1m"         # 31m or 91m
const ANSI_EXT_RED_RX     = r"\x1b\[[0-9;]*38;5;(9|196)m"  # bright red (common)
# Optional fallback if terminal/theme changes colors:
const FALLBACK_ANY_SCAN = true
const ANY_RX   = r"::Any(\b|[^A-Za-z0-9_])"
const UNION_RX = r"::Union\{"
const BOX_RX   = r"::Core\.Box\b"

# Collect function-call subexpressions from a @test expression
function calls_to_check(ex)
    calls = Expr[]
    if ex isa Expr
        if ex.head == :call
            if ex.args[1] in COMP_OPS
                # ==(lhs, rhs) / ≈(lhs, rhs): check callable sides
                for arg in ex.args[2:end]
                    (arg isa Expr && arg.head == :call) && push!(calls, arg)
                end
            else
                push!(calls, ex)  # plain f(args...)
            end
        elseif ex.head == :comparison
            # a < f(x) < g(y): check call operands
            for arg in ex.args
                (arg isa Expr && arg.head == :call) && push!(calls, arg)
            end
        end
    end
    return calls
end

# Build a block that evaluates the call and runs code_warntype(f, Tuple{argtypes...})
function emit_warntype_block(call::Expr, ln::Int, file::AbstractString)
    fex = call.args[1]
    aex = call.args[2:end]
    :(begin
        # Evaluate function and arguments in caller’s scope
        local _f = $(esc(fex))
        local _vals = tuple($(map(a -> :( $(esc(a)) ), aex)...))  # robust 0/1/N args
        # Tuple type of argument types (e.g. Tuple{Int64,Float32})
        local _TT = Base.typesof(_vals...)

        # Run colored code_warntype into a buffer
        local _buf = IOBuffer()
        local _io  = IOContext(_buf, :color => true)
        InteractiveUtils.code_warntype(_io, _f, _TT; optimize=true, debuginfo=:none)
        local _txt_col = String(take!(_buf))

        # Decide if we print (yellow/red or fallback patterns)
        local _is_yellow = occursin(ANSI_YELLOW_RX, _txt_col) || occursin(ANSI_EXT_YELLOW_RX, _txt_col)
        local _is_red    = occursin(ANSI_RED_RX, _txt_col)    || occursin(ANSI_EXT_RED_RX, _txt_col)
        local _print = (_is_yellow || _is_red)
        if !_print && FALLBACK_ANY_SCAN
            # Also scan a non-colored rendering for known problematic patterns
            local _scanbuf = IOBuffer()
            InteractiveUtils.code_warntype(_scanbuf, _f, _TT; optimize=true, debuginfo=:none)
            local _plain = String(take!(_scanbuf))
            _print = occursin(ANY_RX, _plain) || occursin(UNION_RX, _plain) || occursin(BOX_RX, _plain)
        end

        # Record result regardless of print
        _ft_record(_print)

        if _print
            # Only show red/yellow lines
            local _lines = split(_txt_col, '\n')
            local _ylines = [l for l in _lines if occursin(ANSI_YELLOW_RX, l) || occursin(ANSI_EXT_YELLOW_RX, l)]
            local _rlines = [l for l in _lines if occursin(ANSI_RED_RX, l)    || occursin(ANSI_EXT_RED_RX, l)]

            # println("code_warntype (potential instability) at ", $(file), ":", $(ln), " for: ", $(string(call)))
            # for l in _rlines
            #     println(l)
            # end
            # for l in _ylines
            #     println(l)
            # end
            flush(stdout)

            # Store full output + metadata for later inspection
            push!(_FT_REPORTS, FTReport($(file), $(ln), $(string(call)), _txt_col, _rlines, _ylines))
        end
    end)
end

macro test(ex)
    calls = calls_to_check(ex)
    if isempty(calls)
        return :(begin _ft_record(false); nothing end)  # literal-only test → count as ok
    end
    # Capture call site
    local ln = __source__.line
    local file = String(__source__.file)
    blocks = Any[ emit_warntype_block(c, ln, file) for c in calls ]
    return :(begin
        $(blocks...)
    end)
end

# --- custom @testset that prints name and counts ok/bad ---
macro testset(name, block)
    return :(begin
        local _name = $(esc(name))
        local _ctr = _FTCounters(0, 0)
        push!(_FT_STACK, _ctr)
        println("\nTestset: ", _name)
        $(esc(block))
        pop!(_FT_STACK)
        println("Summary for '", _name, "': ok=", _ctr.ok, " bad=", _ctr.bad)
    end)
end

# @test_throws just routes through @test for scanning
macro test_throws(errtype, expr)
    return :( @test $(esc(expr)) )
end

# Support @testset begin ... end (no name)
macro testset(block)
    return :( @testset "unnamed" $(esc(block)) )
end

# You can inspect all captured reports afterwards:
# _FT_REPORTS  # Vector{FTReport}

# Examples
function test_examples()
    @test stable(2) == 4
    @test unstable(false) == 0
    @test 0 ≈ unstable(false)   # RHS call only
    @test unstable(false) == unstable(true)
end

function opt_cu(img, use_cuda=false)
        img
end

use_cuda = false
using FourierTools
using ImageTransformations
using IndexFunArrays
using Zygote
using NDTools
using LinearAlgebra # for the assigned nfft function LinearAlgebra.mul!
using FractionalTransforms
using TestImages
using Random, FFTW
Random.seed!(42)

function test_all()
    include("fft_helpers.jl");
    include("fftshift_alternatives.jl");
    include("utils.jl");
    include("fourier_shifting.jl");
    include("fourier_shear.jl");
    include("fourier_rotate.jl");
    include("resampling_tests.jl");
    include("convolutions.jl");
    include("correlations.jl");
    include("custom_fourier_types.jl");
    include("damping.jl");
    include("czt.jl");
    include("nfft_tests.jl");
    include("fractional_fourier_transform.jl");
    include("fourier_filtering.jl");
    include("sdft.jl");
end
