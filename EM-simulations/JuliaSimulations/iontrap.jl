using ForwardDiff


# Constants (later will be matrix A)
Vrf = 100.0        # RF voltage amplitude [V]
Udc = 10.0         # DC voltage amplitude [V]
Ω   = 2π * 1e6     # RF angular frequency [rad/s]
κ   = 0.5          # geometry factor (dimensionless)
r0  = 1.0          # characteristic radius [m]
z0  = 1.0          # characteristic length [m]

# Variables (later will be vectors X, T)
t = 1e-6                   # time [s]
r = (0.1, 0.2, 0.3)         # position [m] as tuple (x,y,z)

function rf_term(Vrf, Ω, t, r, r0)
    x, y, z = r
    return Vrf * cos(Ω*t) * (y^2 - x^2) / (2*r0^2)
end

function dc_term(Udc, κ, r, z0)
    x, y, z = r
    return κ * Udc * (2z^2 - x^2 - y^2) / (2*z0^2)
end

# Total electric potential
function potential(Vrf, Udc, Ω, κ, r0, z0, t, r)
    return rf_term(Vrf, Ω, t, r, r0) + dc_term(Udc, κ, r, z0)
end

Φ = potential(Vrf, Udc, Ω, κ, r0, z0, t, r)

#convert scalar tuple to vector
function potential_vec(Vrf, Udc, Ω, κ, r0, z0, t, r::AbstractVector{<:Real})
    return potential(Vrf, Udc, Ω, κ, r0, z0, t, (r[1], r[2], r[3]))
end

# Electric field = -∇Φ
function electric_field(Vrf, Udc, Ω, κ, r0, z0, t, r::NTuple{3,<:Real})
    rvec = collect(r)  # convert tuple -> vector
    gradΦ = ForwardDiff.gradient(rr -> potential_vec(Vrf, Udc, Ω, κ, r0, z0, t, rr), rvec)
    return -gradΦ
end

E = electric_field(Vrf, Udc, Ω, κ, r0, z0, t, r)

println("At time t=$t s and position r=$r, the potential is Φ = $Φ V")
println("The electric field is E = $E V/m")
