

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
using Gridap.Algebra: NewtonRaphsonSolver
using Gridap.CellData
using GridapPardiso

#Laws
conv(u, ∇u) = (∇u') ⋅ u
dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u)

# Physical constants
u_max = 100 #150# 150#  150 #cm/s
L = 1 #cm
ρ =  1.06e-3 #kg/cm^3 
μ =  3.50e-5 #kg/cm.s
ν = μ/ρ 
Δt =  0.046 / 1000 # * 0.1 / ( u_max )   # 0.046  #s \\

 #=
#Space Convergence Parameters
θ = 1
ns = [50,60,80,120]#[12,16,24,40]#,48]#,64]
n_ts = 1*1 #[0.2,0.1,0.05]
tF = Δt*n_ts
SNRs = [1e10,20,5]
n_tests = 20
 =#

# #=
#Time Convergence Parameters
θ = 1
ns = 32
n_ts = [8,12,20,36]
tF = Δt
SNRs = [1e10,100,50,20,5]
n_tests = 20# 1 #50 #30
# =#

#Manufactured solution
k=2*pi

if length(n_ts) > 1
 @assert ( length(ns) == 1 )
 dimδ = "n_t"
 dimδs = n_ts
 dimname = "Time"

 u(x,t) = u_max * VectorValue(x[2],x[1]) * sin( k* (t/Δt) )
 p(x,t) = (x[1]-x[2])*sin(k* (t/Δt) )

elseif length(ns) > 1
 @assert ( length(n_ts) == 1 )
 dimδ = "n"
 dimδs = ns
 dimname = "Space"

 u(x,t) = u_max * VectorValue( cos(k*x[1])*sin(k*x[2]), -sin(k*x[1])*cos(k*x[2]) ) * (t/tF)
 p(x,t) = k* ( sin(k*x[1]) - sin(k*x[2]) ) * (t/tF)
else
 throw(InputError("choose 1 dimension to test (dt or h) and set that to a lengh > 1 vector and the other dim to a length = 1 vector"))
end

ud(t) = u(t)

#= BOTH LINEAR
u(x,t) = VectorValue(x[1],x[2])*t
∂tu(t) = x -> VectorValue(x[1],x[2])
p(x,t) = (x[1] + x[2])
=#



u(t::Real) = x -> u(x,t)
#∂tu(x,t) = ∂tu(t)(x)
#∂t(::typeof(u)) = ∂tu
#ud(t) = x -> u(t)(x)
p(t::Real) = x -> p(x,t)
#q(x) = t -> p(x,t)
#∂tp(t) = x -> ForwardDiff.derivative(q(x),t)
#∂tp(x,t) = ∂tp(t)(x)
#∂t(::typeof(p)) = ∂tp

α = 1
f(t) = x -> ρ * ∂t(u)(t)(x) + ρ * conv(u(t)(x),∇(u(t))(x)) - μ * Δ(u(t))(x) + ∇(p(t))(x)
#f(t) = x -> α*u(t)(x)-Δ(u(t))(x)+∇(p(t))(x) #FOR STOKES PROJECTOR TEST
g(t) = x -> (∇⋅u(t))(x)

order = 1

function run_test(n,n_t,tF,SNR,θ)

h=L/n

# Select geometry
R = 0.4
#geom = square(L=0.999,x0=Point(0.5,0.5))
#geom = disk(R,x0=Point(0.75,0.75))
geom = disk(R,x0=Point(0.5,0.5))

n = n
partition = (n,n)
D=length(partition)

# Setup background model
domain = (0,L,0,L)
bgmodel = simplexify(CartesianDiscreteModel(domain,partition))
#const h = L/n

# Cut the background model
cutdisc = cut(bgmodel,geom)
model = DiscreteModel(cutdisc)

# Setup integration meshes
Ω = Triangulation(cutdisc)
#writevtk(Ω,"_Ω")

Γ = EmbeddedBoundary(cutdisc)
Γg = GhostSkeleton(cutdisc)
#writevtk(Γ,"_Γ")

cutter = LevelSetCutter()
cutgeo_facets = cut_facets(cutter,bgmodel,geom)
Γn = BoundaryTriangulation(cutgeo_facets,"boundary",geom)
#writevtk(_Γn,"ntrian")

# Setup normal vectors
n_Γ = get_normal_vector(Γ)
n_Γn = get_normal_vector(Γn)
n_Γg = get_normal_vector(Γg)

# Setup Lebesgue measures
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

A = (0.35/(0.5*SNR)) # Max value for random numbers "SIGNAL LEVEL IS 0.35 (RMS OF SINXCOSX) and the 0.5 is avg of rand(0,1)"
x0 = A*randn(Float64,num_free_dofs(V0)) # Vector of DOFs for the FE space
noise = FEFunction(V0,x0) # Create a FE function with these DOFs
u_MRI(t) = interpolate_everywhere(u(t)+noise,V0)
u_MRI_Ω(t) = u_MRI(t)#restrict(u_MRI(t), trian_Ω)
u_MRI_Γ(t) = u_MRI(t)#restrict(u_MRI(t), trian_Γ)

u_Γn(t) = u(t)
p_Γn(t) = p(t)

function SolveStokes(t)

a((u,p),(v,q)) = 
  ∫( α * m_ΩSP(u, v) + a_ΩSP(u, v) + b_ΩSP(u, q) + b_ΩSP(v, p) - α * sm_ΩSP(u, q) - sb_ΩSP(p, q) )dΩ +  #+ 
  ∫(a_ΓSP(u, v) + b_ΓSP(u, q) + b_ΓSP(v, p) )dΓ + 
  ∫( i_ΓgSP(u, v) - j_ΓgSP(p, q) )dΓg

l((v,q)) = 
  ∫( α * m_ΩSP(u_MRI_Ω(t), v)  + a_ΩSP(u_MRI_Ω(t), v) - ϕ_ΩSP(q, t) - q * g(t))dΩ +#+ 
  ∫( u_MRI_Γ(t) ⊙ ( (γ_SP / h) * v - n_Γ ⋅ ∇(v) + q * n_Γ)  - v⋅(n_Γ⋅∇(u_MRI_Γ(t))) )dΓ #+ 
  #∫( ν * v⋅(n_Γn⋅∇(u_Γn(t))) - (n_Γn⋅v)*p_Γn(t) )dΓn

# FE problem
op = AffineFEOperator(a,l,X,Y)
uh, ph = solve(op)

#=
# Postprocess
uh_Ω = restrict(uh,trian_Ω)
ph_Ω = restrict(ph,trian_Ω)
eu_Ω = u(t) - uh_Ω
ep_Ω = p(t) - ph_Ω
l2(v) = v⊙v
h1(v) = inner(∇(v),∇(v))
eu_l2 = sqrt(sum(integrate(l2(eu_Ω),trian_Ω,quad_Ω)))
ep_l2 = sqrt(sum(integrate(l2(ep_Ω),trian_Ω,quad_Ω)))
=#

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
xh0 = interpolate_everywhere([u(0.0),p(0.0)],X0)

op = TransientFEOperator(res,jac,jac_t,X,Y)

#ls = LUSolver()
ls = PardisoSolver(op.assem_t.matrix_type)

nls = NewtonRaphsonSolver(ls,1e-5,30)

#=

nls = NLSolver(
    show_trace = true,
    method = :newton,
    linesearch = BackTracking(),

#    ftol = 1e-2,

=#

odes = ThetaMethod(nls, dt, θ)
solver = TransientFESolver(odes)
sol_t = solve(solver, op, xh0, t0, tF)

l2(w) = w⋅w

#tol = 1.0e-5
_t_n = t0

result = Base.iterate(sol_t)

eul2=[]
epl2=[]

for (xh_tn, tn) in sol_t
  _t_n += dt
  uh_tn = xh_tn[1]
  ph_tn = xh_tn[2]
  e_u = u(tn) - uh_tn
  e_p = p(tn) - ph_tn
  u_ex = u(tn) - 0*uh_tn
  p_ex = p(tn) - 0*ph_tn
  @show eul2i = sqrt(sum( ∫(l2(e_u))dΩ ))
  @show epl2i = sqrt(sum( ∫(l2(e_p))dΩ ))
  writevtk(Ω,"results_fm",cellfields=["e_u"=>e_u,"uh_Ω"=>uh_tn,"u_ex"=>u_ex,"e_p"=>e_p,"ph_Ω"=>ph_tn,"p_ex"=>p_ex])
  push!(eul2,eul2i)
  push!(epl2,epl2i)
  println("$(tn*100/Δt)% of timesteps complete")
  @show Int(round(((_t_n-dt)/dt)+1,digits=7))
end

eul2=last(eul2)
epl2=last(epl2)

(eul2, epl2)

end #function INS

#running solvers
t0 = 0.0
tF = tF
dt = tF/n_t

println("Solving Stokes Projector (n = $(n)) (n_t = $(n_t))")
u_projΓ_vector = []
uh_0,ph_0 = SolveStokes(t0)
push!(u_projΓ_vector,uh_0)

dtss=[]
for i in 1:n_t
  t=i*dt
  u_proj_t,p_proj_t = SolveStokes(t)
  push!(u_projΓ_vector,u_proj_t)
  push!(dtss,t)
  @show length(u_projΓ_vector)
end

uh = u_projΓ_vector
u_projΓ(t) = u_projΓ_vector[Int(round((t/dt)+1,digits=7))]

println("Solving Navier Stokes")
(eul2, epl2) = SolveNavierStokes( u_projΓ,uh_0,ph_0 )

(eul2, epl2, h)

end #function (whole method)

function conv_test(ns,n_ts,tF,SNR,θ)

  eul2s = Float64[]
  epl2s = Float64[]
  hs = Float64[]

  for n in ns
    for n_t in n_ts

      eul2, epl2, h = run_test(n,n_t,tF,SNR,θ)

      push!(eul2s,eul2)
      push!(epl2s,epl2)
      push!(hs,h)
    end
  end

  (eul2s, epl2s,  hs)

end

#Initialising save data
OutputData = DataFrame(dimδ = [ _ for _ in dimδs])
plot()

#Running Method for different SNR values
for SNR in SNRs
  println(SNR)
  #Running tests
  conv_test_results = [conv_test(ns,n_ts,tF,SNR,θ) for _ in 1:n_tests]
  eul2s = (1/n_tests)*(sum( [conv_test_results[i][1] for i in 1:n_tests] ) )
  epl2s = (1/n_tests)*(sum( [conv_test_results[i][2] for i in 1:n_tests] ) )
  global hs = conv_test_results[1][3]

  if dimδ == "n"
    x_plot = hs
    x_plot_name = "h"
  else
    x_plot = n_ts
    x_plot_name = "dt"
  end

  #Plotting Data
  plot!(x_plot, #hs, #dts
      [eul2s, epl2s],
      xaxis=:log, yaxis=:log,
      label=["L2U_$(SNR)" "L2P_$(SNR)"],
      shape=:auto,
      xlabel=x_plot_name,
      ylabel="L2 error norm",
      title = "Method_Convergence_$(x_plot)")

  #Tabulating Data
  df_i = DataFrame(eu = eul2s, ep = epl2s)
  df_i = rename(df_i, Dict("eu" => "eu SNR=$(SNR)", "ep" => "ep SNR=$(SNR)")) #making column name include the SNR value
  global OutputData = hcat(OutputData,df_i,makeunique=true)
  OutputData
end

#adding h col to table if δ=h
if dimδ == "n"
  df_h = DataFrame(h = hs)
  OutputData = hcat(df_h,OutputData,makeunique=true)
end

#Saving data
folderName = "Results"
PlotFileName = "$(dimname)ConvergencePlot"
DataFileName = "$(dimname)ConvergenceData"

if !isdir(folderName)
    mkdir(folderName)
end
filePath_Plot = join([folderName, PlotFileName], "/")
filePath_Data = join([folderName, DataFileName], "/")

CSV.write(filePath_Data, OutputData)
savefig(filePath_Plot)

end #module


