#Should run 1_datafactory before ruunning this to generate velocity values

module FullMethod

using Pkg
Pkg.activate("/scratch/bt62/cm8825")

using Gridap
import Gridap: ∇
using GridapEmbedded
using ForwardDiff
using GridapEmbedded.LevelSetCutters
using Test
using LinearAlgebra: tr
using Gridap.ReferenceFEs
using Gridap.Arrays
using LinearAlgebra
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using LineSearches: BackTracking
import GridapODEs.TransientFETools: ∂t
using GridapEmbedded.Interfaces: cut
using DataFrames
using Plots
using CSV
using JSON
using WriteVTK
using Gridap.Algebra: NewtonRaphsonSolver
using GridapPardiso
using Gridap.CellData

#Laws
conv(u, ∇u) = (∇u') ⋅ u
dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u)

# Physical constants
u_max = 150 #150# 150#  150 #cm/s
L = 1 #cm
ρ =  1.06e-3 #kg/cm^3 
μ =  3.50e-5 #kg/cm.s
ν = μ/ρ 
Δt =  0.046 / 1000 # * 0.1 / ( u_max )   # 0.046  #s \\

θ = 1
n_t = 1 # 0 # 5 #1 #n of time steps

t0 = 0.0
dt = Δt
tF = dt*n_t

## Importing 
#import Level_Set
Level_Set = JSON.parsefile("Data/Distance_map.json")

#Storing data into variables
dimensions=Level_Set["dimensions"].-1 #-1 for conversion from nodes to number of cells
spacing=Level_Set["spacing"] #see definition of h for conversion to mm 
Level_Set = Level_Set["scalars"]
Level_Set = convert(Vector{Float64},Level_Set)

#Forcing terms
f(t) = VectorValue(0.0,0.0,0.0)
g(t) = 0.0

#defining background grid
domain = (0.0, spacing[1]*dimensions[1], 0.0, spacing[2]*dimensions[2], 0.0 ,spacing[3]*dimensions[3]) 
partition=(dimensions[1],dimensions[2],dimensions[3])

bgmodel  = simplexify(CartesianDiscreteModel(domain,partition))
D=length(dimensions)
h = maximum(spacing) 
h = h/10 # converting to mm 

# Setup model from level set
point_to_coords = collect1d(get_node_coordinates(bgmodel))
geo_aorta = DiscreteGeometry(Level_Set,point_to_coords,name="")

# Cut Neumann EmbeddedBoundary
geo_notcube =  ! cube(;L=30,x0=Point(105,325,25),name="notcube")
geo_notcube_x = discretize(geo_notcube,bgmodel)

# intersect aorta and not_cube
cutter=LevelSetCutter()
geo_sliced_aorta = intersect(geo_aorta,geo_notcube_x) 
cutgeo_sliced_aorta = cut(cutter,bgmodel,geo_sliced_aorta)
model = DiscreteModel(cutgeo_sliced_aorta)

order = 1 

reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

#interpolation space
V2 = FESpace(
  bgmodel,
  reffeᵤ,
  conformity=:H1
  )

#importing velocity data
#u_MRI_import(t) = CSV.read("Data/u_MRI_$(t)")

u_MRI_import(i) = CSV.read("Data/u_MRI_$(i)") 

u_MRI_values(i) = convert(Array,u_MRI_import(i).u_MRI)
u_MRI(i) = FEFunction(V2,u_MRI_values(0))

u_MRI_Ω(t) = u_MRI(t)
u_MRI_Γ(t) = u_MRI(t)

#writevtk(Ω,"u_MRI_0_Test",cellfields=["uh"=>u_MRI_Ω(0)])














# Setup integration meshes
Ω = Triangulation(cutgeo_sliced_aorta)
Γ = EmbeddedBoundary(cutgeo_sliced_aorta, geo_sliced_aorta, geo_aorta) # nitsche
Γn = EmbeddedBoundary(cutgeo_sliced_aorta, geo_sliced_aorta, geo_notcube_x) #neumann
Γg = GhostSkeleton(cutgeo_sliced_aorta)

# Setup normal vectors
n_Γ = get_normal_vector(Γ)
n_Γn = get_normal_vector(Γn)
n_Γg = get_normal_vector(Γg)

# Setup cuadratures
order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓn = Measure(Γn,degree)
dΓg = Measure(Γg,degree)
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

#Spaces
V0 = FESpace(
  model,
  reffeᵤ,
  conformity=:H1
  )

reffeₚ = ReferenceFE(lagrangian,Float64,order)

Q = TestFESpace(
  model,
  reffeₚ,
  conformity=:H1,
  constraint=:zeromean)

U = TrialFESpace(V0)
P = TrialFESpace(Q)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

## Coefficiants

α = 1 #for dimension conversion in the SP- should have dim 1/L^2

#NITSCHE
α_γ = 100
γ_SP = α_γ * ( 1 / h )
γ(u) =  α_γ * ( μ / h + h*ρ/(12*θ*dt) ) #+  ρ * normInf( u ) / 6 ) # Nitsche Penalty parameter ( γ / h ) 

#STABILISATION
α_τ = 1 #Tunable coefficiant (0,1)
τ_PSPG_SP = α_τ * 0.1 * h^2
τ_SUPG(u) = α_τ * inv( sqrt( ( 2 / Δt )^2 + ( 2 * normInf(u) / h )*( 2 * normInf(u) / h ) + 9 * ( 4*ν / h^2 )^2 )) # SUPG Stabilisation - convection stab ( τ_SUPG(u )
τ_PSPG(u) = τ_SUPG(u) # PSPG stabilisation - inf-sup stab  ( ρ^-1 * τ_PSPG(u) )

#GHOST PENALTY
# Ghost Penalty parameters  
α_B = 0.01 
α_u = 0.01 
α_p = 0.01 

#NS Paper ( DOI 10.1007/s00211-007-0070-5)
γ_uSP = α_u * h  #visc diffusion 
γ_pSP = α_p * h^3

γ_BINS(u)   = α_B * ρ * abs(u.⁺ ⋅ n_Γg.⁺ ) * h^2    #conv
γ_uINS     = α_u * μ * h  #visc diffusion 
γ_pINS(u)   = α_p * inv( (μ/h) + (ρ*normInf(u.⁺)/6) ) * h^2  #pressure

## Terms
#Stokes Projector
m_ΩSP(u, v) = u ⊙ v
a_ΩSP(u, v) = ∇(u) ⊙ ∇(v)
b_ΩSP(v, p) = -(∇ ⋅ v) * p

sm_ΩSP(u, q) = ρ^(-1) * τ_PSPG_SP  * (u ⋅ ∇(q)) 
sb_ΩSP(p, q) = ρ^(-1) *τ_PSPG_SP  * ∇(p) ⋅ ∇(q)
ϕ_ΩSP(q, t) = α * ρ^(-1) * τ_PSPG_SP  * ∇(q) ⋅ u_MRI_Ω(t)

a_ΓSP(u, v) = ( -(n_Γ ⋅ ∇(u)) ⋅ v - u ⋅ (n_Γ ⋅ ∇(v)) + (γ_SP / h) * u ⋅ v )
b_ΓSP(v, p) = (n_Γ ⋅ v) * p

i_ΓgSP(u,v) = ( γ_uSP ) * jump(n_Γg⋅∇(u))⋅jump(n_Γg⋅∇(v))
j_ΓgSP(p,q) = ( γ_pSP ) * jump(n_Γg⋅∇(p))*jump(n_Γg⋅∇(q))

#Inc Navier-Stokes

#Interior terms
m_ΩINS(ut,v) = ρ * ut⊙v
a_ΩINS(u,v) = μ * ∇(u)⊙∇(v) 
b_ΩINS(v,p) = - (∇⋅v)*p
c_ΩINS(u, v) = ρ *  v ⊙ conv(u, ∇(u))
dc_ΩINS(u, du, v) = ρ * v ⊙ dconv(du, ∇(du), u, ∇(u))

#Boundary terms 
a_ΓINS(u,v) = μ* ( - (n_Γ⋅∇(u))⋅v - u⋅(n_Γ⋅∇(v)) ) + ( γ(u)/h )*u⋅v 
b_ΓINS(v,p) = (n_Γ⋅v)*p

#PSPG 
sp_ΩINS(w,p,q)    = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ ∇(p)
st_ΩINS(w,ut,q)   = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ ut
sc_ΩINS(w,u,q)    = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ conv(u, ∇(u))
dsc_ΩINS(w,u,du,q)= (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ dconv(du, ∇(du), u, ∇(u))
ϕ_ΩINS(w,q,t)     = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ f(t)

#SUPG
sp_sΩINS(w,p,v)    = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ ∇(p)
st_sΩINS(w,ut,v)   = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ ut
sc_sΩINS(w,u,v)    = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ conv(u, ∇(u)) 
dsc_sΩINS(w,u,du,v)= τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ dconv(du, ∇(du), u, ∇(u)) 
ϕ_sΩINS(w,v,t)     = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ f(t)  

#Ghost Penalty terms
i_ΓgINS(w,u,v) = ( γ_BINS(w) + γ_uINS )*jump(n_Γg⋅∇(u))⋅jump(n_Γg⋅∇(v))
j_ΓgINS(w,p,q) = ( γ_pINS(w) ) * jump(n_Γg⋅∇(p))*jump(n_Γg⋅∇(q))


function SolveStokes(i)

  a((u,p),(v,q)) = 
  ∫( α * m_ΩSP(u, v) + a_ΩSP(u, v) + b_ΩSP(u, q) + b_ΩSP(v, p) - α * sm_ΩSP(u, q) - sb_ΩSP(p, q) )dΩ +  #+ 
  ∫(a_ΓSP(u, v) + b_ΓSP(u, q) + b_ΓSP(v, p) )dΓ + 
  ∫( i_ΓgSP(u, v) - j_ΓgSP(p, q) )dΓg

l((v,q)) = 
  ∫( α * m_ΩSP(u_MRI_Ω(i), v)  + a_ΩSP(u_MRI_Ω(i), v) - ϕ_ΩSP(q, i )  )dΩ +#+ 
  ∫( u_MRI_Γ(i) ⊙ ( (γ_SP / h) * v - n_Γ ⋅ ∇(v) + q * n_Γ)  - v⋅(n_Γ⋅∇(u_MRI_Γ(i))) )dΓ #+ 
  #∫( ν * v⋅(n_Γn⋅∇(u_Γn(i))) - (n_Γn⋅v)*p_Γn(i) )dΓn

  # FE problem
  op = AffineFEOperator(a,l,X,Y)
  uh, ph = solve(op)

  (uh,ph)
  
  end # function Stokes
  
  function SolveNavierStokes(u_projΓ,uh_0,ph_0)
  
  #Interior term collection
  res(t,(u,p),(ut,pt),(v,q)) = 
  ∫( m_ΩINS(ut,v) + a_ΩINS(u,v) + b_ΩINS(v,p) + b_ΩINS(u,q) - v⋅f(t) + q*g(t) + c_ΩINS(u,v)  # + ρ * 0.5 * (∇⋅u) * u ⊙ v  
  - sp_ΩINS(u,p,q)  -  st_ΩINS(u,ut,q)  + ϕ_ΩINS(u,q,t)     - sc_ΩINS(u,u,q) 
  - sp_sΩINS(u,p,v) - st_sΩINS(u,ut,v)  + ϕ_sΩINS(u,v,t)    - sc_sΩINS(u,u,v) )dΩ + 
  
  ∫( a_ΓINS(u,v)+b_ΓINS(u,q)+b_ΓINS(v,p) - u_projΓ(t) ⊙ (  ( γ(u)/h )*v - μ * n_Γ⋅∇(v) + q*n_Γ ) )dΓ + 
  #∫( a_ΓINS(u,v)+b_ΓINS(u,q)+b_ΓINS(v,p) - ud(t) ⊙ (  ( γ(u)/h )*v - μ * n_Γ⋅∇(v) + q*n_Γ ) )dΓ + 
  
  #∫( μ * - v⋅(n_Γn⋅ ∇( u_Γn(t) ) ) + (n_Γn⋅v)* p_Γn(t) )dΓn + 
  
  ∫( i_ΓgINS(u,u,v) - j_ΓgINS(u,p,q) )dΓg
  
  
  jac(t,(u,p),(ut,pt),(du,dp),(v,q))=
  ∫( a_ΩINS(du,v) + b_ΩINS(v,dp) + b_ΩINS(du,q)  + dc_ΩINS(u, du, v) # + ρ * 0.5 * (∇⋅u) * du ⊙ v 
  - sp_ΩINS(u,dp,q)  - dsc_ΩINS(u,u,du,q) 
  - sp_sΩINS(u,dp,v) - dsc_sΩINS(u,u,du,v) )dΩ + 
  
  ∫( a_ΓINS(du,v)+b_ΓINS(du,q)+b_ΓINS(v,dp) )dΓ + 
  
  ∫( i_ΓgINS(u,du,v) - j_ΓgINS(u,dp,q) )dΓg
  
  
  jac_t(t,(u,p),(ut,pt),(dut,dpt),(v,q)) = 
    ∫( m_ΩINS(dut,v) 
    - st_ΩINS(u,dut,q) 
    - st_sΩINS(u,dut,v) )dΩ
  
  X0 = X(0.0)
  xh0 = interpolate_everywhere([uh_0,ph_0],X(0.0))
  
  op = TransientFEOperator(res,jac,jac_t,X,Y)
  
  ls = LUSolver()
  #ls = PardisoSolver(op.assem_t.matrix_type)
  
  #nls = NewtonRaphsonSolver(ls,1e-5,30)

  nls = NewtonRaphsonSolver(ls,1e99,1) #debugging SI version
  #nls = NewtonRaphsonSolver(ls,1e-1,10) #debugging SI version
  
 #= 
  nls = NLSolver(
    ls,
    show_trace = true,
    method = :newton,
    linesearch = BackTracking(),
    ftol = 1e-3,
    iterations=20
)
=#
  
  
  odes = ThetaMethod(nls, dt, θ)
  solver = TransientFESolver(odes)
  sol_t = solve(solver, op, xh0, t0, tF)
  
  (sol_t)
  
  end #function INS

function writePVD(filePath, Ω, sol; append=false)
    outfiles = paraview_collection(filePath, append=append) do pvd
        for (i, (xh, t)) in enumerate(sol)
            @show i
            uh = xh[1]
            ph = xh[2]
            pvd[t] = createvtk(
                Ω,
                filePath * "_$i.vtu",
                cellfields = ["uh" => uh, "ph" => ph],
            )
        end
    end
end

println("Solving Stokes Projector")
u_projΓ_vector = []
uh_0,ph_0 = SolveStokes(0)
push!(u_projΓ_vector,uh_0)

dtss = []
for i in 1:n_t
  u_proj_t,p_proj_t = SolveStokes(i)
  push!(u_projΓ_vector,u_proj_t)
end

uh = u_projΓ_vector
u_projΓ(t) = u_projΓ_vector[Int(round((t/dt)+1,digits=7))]

### Initialize Paraview files
folderName = "ins-results_TESTING"
fileName = "fields"
if !isdir(folderName)
    mkdir(folderName)
end
filePath = join([folderName, fileName], "/")

println("Solving Navier Stokes")
sol_t = SolveNavierStokes(u_projΓ,uh_0,ph_0)

println("Writing Solution")
writePVD(filePath, Ω, sol_t, append=true)

end #module