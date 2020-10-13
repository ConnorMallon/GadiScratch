
module FullMethod

using Pkg
Pkg.activate(".")

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

# #=
#Space Convergence Parameters
θ = 1
ns = [16,20,28,42]#,74]
dts = 0.0001#[0.2,0.1,0.05,0.025,0.0125]
tF= 0.0001
SNRs = [1e10,100]#,50,20,5]
n_tests = 10 #20
# =#

 #=
#Time Convergence Parameters
θ = 1
ns = 16
dts = [0.2,0.1,0.05,0.025,0.0125]
SNRs = [1e10,100,50,20,5]
n_tests = 20 #30
 =#

#Manufactured solution
k=2*pi

if length(dts) > 1
 @assert ( length(ns) == 1 )
 dimδ = "dt"
 dimδs = dts
 dimname = "Time"

 u(x,t) = VectorValue(x[1]+x[2],-x[1]-x[2])*cos(k*t)
 p(x,t) = (x[1]-x[2])*cos(k*t)

elseif length(ns) > 1
 @assert ( length(dts) == 1 )
 dimδ = "n"
 dimδs = ns
 dimname = "Space"

 u(x,t) = VectorValue(-cos(k*x[1])*sin(k*x[2]),sin(k*x[1])*cos(k*x[2]))*t
 p(x,t) = k*(sin(k*x[1])-sin(k*x[2]))*t
else
 throw(InputError("choose 1 dimension to test (dt or h) and set that to a lengh > 1 vector and the other dim to a length = 1 vector"))
end

#= BOTH LINEAR
u(x,t) = VectorValue(x[1],x[2])*t
∂tu(t) = x -> VectorValue(x[1],x[2])
p(x,t) = (x[1] + x[2])
=#

#Laws
@law conv(u, ∇u) = (∇u') ⋅ u
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u)

#Physical constants
ρ = 1.06e-3 #kg/cm^3 
μ = 3.50e-5  #kg/cm.s
ν = μ/ρ 

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
f(t) = x -> ∂t(u)(t)(x) + conv(u(t)(x),∇(u(t))(x)) - ν*Δ(u(t))(x) + ∇(p(t))(x) #FOR INS TEST
#f(t) = x -> α*u(t)(x)-Δ(u(t))(x)+∇(p(t))(x) #FOR STOKES PROJECTOR TEST
g(t) = x -> (∇⋅u(t))(x)

order = 1

function run_test(n,dt,tF,SNR,θ)

# Select geometry
R = 0.7
geo1 = disk(R)
box = get_metadata(geo1)

# Cut the background model
partition = (n,n)
D = length(partition)
bgmodel = simplexify(CartesianDiscreteModel(box.pmin,box.pmax,partition))
cutgeo = cut(bgmodel,geo1)

# Generate the "active" model
model = DiscreteModel(cutgeo)

tempvar=discretize(geo1,bgmodel)

tree1=tempvar.tree
level_set=tree1.data[1]

point_to_coords=collect1d(get_node_coordinates(bgmodel))
DG1=DiscreteGeometry(level_set,point_to_coords,name="")
cutter=LevelSetCutter()

DGcutgeo=cut(cutter,bgmodel,DG1)
DGmodel=DiscreteModel(DGcutgeo)

model=DGmodel
cutgeo=DGcutgeo

# Setup integration meshes
trian_Ω = Triangulation(cutgeo)
trian_Γ = EmbeddedBoundary(cutgeo)
trian_Γg = GhostSkeleton(cutgeo)

# Setup normal vectors
n_Γ = get_normal_vector(trian_Γ)
n_Γg = get_normal_vector(trian_Γg)

# Setup cuadratures
order = 1
quad_Ω = CellQuadrature(trian_Ω,2*order)
quad_Γ = CellQuadrature(trian_Γ,2*order)
quad_Γg = CellQuadrature(trian_Γg,2*order)

# Setup FESpace
V = TestFESpace(
  model=model,valuetype=VectorValue{D,Float64},reffe=:PLagrangian,
  order=order,conformity=:H1)

Q = TestFESpace(
  model=model,valuetype=Float64,reffe=:PLagrangian,
  order=order,conformity=:H1,constraint=:zeromean, zeromean_trian=trian_Ω)

U = TrialFESpace(V)
P = TrialFESpace(Q)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

# Stabilization parameters

β1 = 0.2
β2 = 0.1
β3 = 0.05
γ = 10.0
h = (box.pmax-box.pmin)[1]/partition[1]

# Terms
m_Ω(u, v) = u ⊙ v
a_ΩSP(u, v) = ∇(u) ⊙ ∇(v)
a_ΩINS(u, v) = ν* a_ΩSP(u, v)
b_Ω(v, p) = -(∇ ⋅ v) * p
c_Ω(u, v) = v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = v ⊙ dconv(du, ∇(du), u, ∇(u))

sm_Ω(u, q) = (β1 * h^2) * (u ⋅ ∇(q))
sb_Ω(p, q) = (β1 * h^2) * ∇(p) ⋅ ∇(q)
sc_Ω(u,q) = (β1*h^2) * conv(u, ∇(u))⋅∇(q)
dsc_Ω(u,du,q) = (β1*h^2) * ∇(q)⋅dconv(du, ∇(du), u, ∇(u))

a_ΓSP(u, v) = ( -(n_Γ ⋅ ∇(u)) ⋅ v - u ⋅ (n_Γ ⋅ ∇(v)) + (γ / h) * u ⋅ v )
a_ΓINS(u, v) = ν * a_ΓSP(u, v)
b_Γ(v, p) = (n_Γ ⋅ v) * p

i_ΓgSP(u, v) = (β2 * h) * jump(n_Γg ⋅ ∇(u)) ⋅ jump(n_Γg ⋅ ∇(v))
i_ΓgINS(u, v) = ν * i_ΓgSP(u, v)
j_Γg(p, q) = (β3 * h^3) * jump(n_Γg ⋅ ∇(p)) * jump(n_Γg ⋅ ∇(q))

ϕ_ΩINS(q, t) = (β1 * h^2) * ∇(q) ⋅ f(t) # TO BE DELETED ONCE USING THE REAL RHS FOR INS (WHICH IS ZERO)
ϕ_ΩSP(q, t) = α * (β1 * h^2) * ∇(q) ⋅ u_MRI_Ω(t)

A = (0.35/(0.5*SNR)) # Max value for random numbers "SIGNAL LEVEL IS 0.35 (RMS OF SINXCOSX) and the 0.5 is avg of rand(0,1)"
x0 = A*randn(Float64,num_free_dofs(V)) # Vector of DOFs for the FE space
noise = FEFunction(V,x0) # Create a FE function with these DOFs
u_MRI(t) = interpolate_everywhere(V,u(t)+noise)
u_MRI_Ω(t) = restrict(u_MRI(t), trian_Ω)
u_MRI_Γ(t) = restrict(u_MRI(t), trian_Γ)

function SolveStokes(t)

function A_Ω(X, Y)
  u, p = X
  v, q = Y
  α * m_Ω(u, v) + a_ΩSP(u, v) + b_Ω(u, q) + b_Ω(v, p) - α * sm_Ω(u, q) - sb_Ω(p, q)
end

function A_Γ(X, Y)
  u, p = X
  v, q = Y
  a_ΓSP(u, v) + b_Γ(u, q) + b_Γ(v, p)
end

function J_Γg(X, Y)
  u, p = X
  v, q = Y
  i_ΓgSP(u, v) - j_Γg(p, q)
end

function L_Ω(Y)
  v, q = Y
  α * m_Ω(u_MRI_Ω(t), v)  + a_ΩSP(u_MRI_Ω(t), v) - ϕ_ΩSP(q, t) - q * g(t)
end

function L_Γ(Y)
  v, q = Y
  u_MRI_Γ(t) ⊙ ((γ / h) * v - n_Γ ⋅ ∇(v) + q * n_Γ)  - v⋅(n_Γ⋅∇(u_MRI_Γ(t)))
end

# FE problem
t_Ω = AffineFETerm(A_Ω,L_Ω,trian_Ω,quad_Ω)
t_Γ = AffineFETerm(A_Γ,L_Γ,trian_Γ,quad_Γ)
t_Γg = LinearFETerm(J_Γg,trian_Γg,quad_Γg)
op = AffineFEOperator(X,Y,t_Ω,t_Γ,t_Γg)
uh, ph = solve(op)

# Postprocess
uh_Ω = restrict(uh,trian_Ω)
ph_Ω = restrict(ph,trian_Ω)
eu_Ω = u(t) - uh_Ω
ep_Ω = p(t) - ph_Ω
l2(v) = v⊙v
h1(v) = inner(∇(v),∇(v))
eu_l2 = sqrt(sum(integrate(l2(eu_Ω),trian_Ω,quad_Ω)))
ep_l2 = sqrt(sum(integrate(l2(ep_Ω),trian_Ω,quad_Ω)))

(uh,ph)

end # function Stokes

function SolveNavierStokes(u_projΓ,uh_0,ph_0)

#Interior term collection
function res_Ω(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  m_Ω(ut,v) + a_ΩINS(u,v) + b_Ω(v,p) + b_Ω(u,q) + c_Ω(u,v) - v⋅f(t) + q*g(t) - sm_Ω(ut,q) - sb_Ω(p,q) - sc_Ω(u,q) + ϕ_ΩINS(q,t)
end

function jac_Ω(t,x,xt,dx,y)
  u, p = x
  du,dp = dx
  v,q = y
  dc_Ω(u, du, v) + a_ΩINS(du,v) + b_Ω(v,dp) + b_Ω(du,q) - sb_Ω(dp,q) - dsc_Ω(u,du,q)
end

function jac_tΩ(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  m_Ω(dut,v) - sm_Ω(dut,q)
end

#Boundary term collection
function res_Γ(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  a_ΓINS(u,v)+b_Γ(u,q)+b_Γ(v,p) - restrict(u_projΓ(t),trian_Γ) ⊙( ν*(γ/h)*v - ν*n_Γ⋅∇(v) + q*n_Γ )
end

function jac_Γ(t,x,xt,dx,y)
  du,dp = dx
  v,q = y
  a_ΓINS(du,v)+b_Γ(du,q)+b_Γ(v,dp)
end

function jac_tΓ(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  0*m_Ω(dut,v)
end

#Skeleton term collection
function res_Γg(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  i_ΓgINS(u,v) - j_Γg(p,q)
end

function jac_Γg(t,x,xt,dx,y)
  du,dp = dx
  v,q = y
  i_ΓgINS(du,v) - j_Γg(dp,q)
end

function jac_tΓg(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  0*i_ΓgINS(dut,v)
end

xh0 = interpolate_everywhere([uh_0,ph_0],X(0.0))

t_Ω = FETerm(res_Ω,jac_Ω,jac_tΩ,trian_Ω,quad_Ω)
t_Γ = FETerm(res_Γ,jac_Γ,jac_tΓ,trian_Γ,quad_Γ)
t_Γg = FETerm(res_Γg,jac_Γg,jac_tΓg,trian_Γg,quad_Γg)

op = TransientFEOperator(X,Y,t_Ω,t_Γ,t_Γg)

# #=
nls = NLSolver(
    show_trace = false,
    method = :newton,
    linesearch = BackTracking(),
)
# =#

ls=LUSolver()
odes = ThetaMethod(nls, dt, θ)
solver = TransientFESolver(odes)
sol_t = solve(solver, op, xh0, t0, tF)

l2(w) = w⋅w

tol = 1.0e-6
_t_n = t0

result = Base.iterate(sol_t)

eul2=[]
epl2=[]

for (xh_tn, tn) in sol_t
  _t_n += dt
  uh_tn = xh_tn[1]
  ph_tn = xh_tn[2]
  uh_Ω = restrict(uh_tn,trian_Ω)
  ph_Ω = restrict(ph_tn,trian_Ω)
  e = u(tn) - uh_Ω
  eul2i = sqrt(sum( integrate(l2(e),trian_Ω,quad_Ω) ))
  e = p(tn) - ph_Ω
  epl2i = sqrt(sum( integrate(l2(e),trian_Ω,quad_Ω) ))
  push!(eul2,eul2i)
  push!(epl2,epl2i)
  @show _t_n
end

eul2=last(eul2)
epl2=last(epl2)

(eul2, epl2)

end #function INS

#running solvers
t0 = 0.0
tF = tF
dt = dt

println("Solving Stokes Projector (n = $(n)) (dt = $(dt))")
u_projΓ_vector = []
uh_0,ph_0 = SolveStokes(t0)
push!(u_projΓ_vector,uh_0)

dtss = []
for t in t0+dt:dt:tF
  u_proj_t,p_proj_t = SolveStokes(t)
  push!(u_projΓ_vector,u_proj_t)
  push!(dtss,t)
end
uh = u_projΓ_vector
u_projΓ(t) = u_projΓ_vector[Int(round((t/dt)+1,digits=7))]

println("Solving Navier Stokes")
(eul2, epl2) = SolveNavierStokes(u_projΓ,uh_0,ph_0)

(eul2, epl2, h)

end #function (whole method)

function conv_test(ns,dts,tF,SNR,θ)

  eul2s = Float64[]
  epl2s = Float64[]
  hs = Float64[]

  for n in ns
    for dt in dts

      eul2, epl2, h = run_test(n,dt,tF,SNR,θ)

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
  conv_test_results = [conv_test(ns,dts,tF,SNR,θ) for _ in 1:n_tests]
  eul2s = (1/n_tests)*(sum( [conv_test_results[i][1] for i in 1:n_tests] ) )
  epl2s = (1/n_tests)*(sum( [conv_test_results[i][2] for i in 1:n_tests] ) )
  global hs = conv_test_results[1][3]

  if dimδ == "n"
    x_plot = hs
    x_plot_name = "h"
  else
    x_plot = dts
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
