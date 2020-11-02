#Should run 1_datafactory before ruunning this to generate velocity values

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
using JSON
using WriteVTK
using Gridap.Algebra: NewtonRaphsonSolver

#Laws
@law conv(u, ∇u) = (∇u') ⋅ u
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u)

#Physical constants
ρ = 1.06e-3 #kg/cm^3 
μ = 3.50e-5  #kg/cm.s
ν = μ/ρ 

θ = 1
n_t = 2#19

t0 = 0.0
dt = 46.08e-3 #ms
tF = dt*n_t

#import Level_Set
Level_Set = JSON.parsefile("Data/Distance_map.json")

#Storing data into variables
dimensions=Level_Set["dimensions"].-1 #-1 for conversion from nodes to number of cells
spacing=Level_Set["spacing"]
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

# Setup integration meshes
trian_Ω = Triangulation(cutgeo_sliced_aorta)
trian_Γ = EmbeddedBoundary(cutgeo_sliced_aorta, geo_sliced_aorta, geo_aorta) # nitsche
trian_Γn = EmbeddedBoundary(cutgeo_sliced_aorta, geo_sliced_aorta, geo_notcube_x) #neumann
trian_Γg = GhostSkeleton(cutgeo_sliced_aorta)

# Setup normal vectors
n_Γ = get_normal_vector(trian_Γ)
n_Γn = get_normal_vector(trian_Γn)
n_Γg = get_normal_vector(trian_Γg)

# Setup cuadratures
order = 1
quad_Ω = CellQuadrature(trian_Ω,2*order)
quad_Γ = CellQuadrature(trian_Γ,2*order)
quad_Γn = CellQuadrature(trian_Γn,2*order)
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
α = 1 # h^-2

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

#u_MRI(t) = interpolate_everywhere(V,u(t))

V2 = TestFESpace(
  model=bgmodel,valuetype=VectorValue{D,Float64},reffe=:PLagrangian,
  order=order,conformity=:H1)

#importing velocity data
u_MRI_import(t) = CSV.read("Data/u_MRI_$(t)")
u_MRI_values(t) = convert(Array,u_MRI_import(t).u_MRI)
u_MRI(t) = FEFunction(V2,u_MRI_values(t))

u_MRI_Ω(t) = restrict(u_MRI(t), trian_Ω)
u_MRI_Γ(t) = restrict(u_MRI(t), trian_Γ)

#writevtk(trian_Ω,"u_MRI_0_Test",cellfields=["uh"=>u_MRI_Ω(0)])

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

function l_Γn(y)
  v,q = y
  0* q * g(t) #dummy 
end

# FE problem
t_Ω = AffineFETerm(A_Ω,L_Ω,trian_Ω,quad_Ω)
t_Γ = AffineFETerm(A_Γ,L_Γ,trian_Γ,quad_Γ)
t_Γn = FESource(l_Γn,trian_Γn,quad_Γn)
t_Γg = LinearFETerm(J_Γg,trian_Γg,quad_Γg)
op = AffineFEOperator(X,Y,t_Ω,t_Γ,t_Γn,t_Γg)
uh, ph = solve(op)

(uh,ph)

end # function Stokes

function SolveNavierStokes(u_projΓ,uh_0,ph_0)

#Interior term collection
function res_Ω(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  m_Ω(ut,v) + a_ΩINS(u,v) + b_Ω(v,p) + b_Ω(u,q) + c_Ω(u,v) - v⋅f(t) + q*g(t) - sm_Ω(ut,q) - sb_Ω(p,q) - sc_Ω(u,q) + ϕ_ΩINS(q,t) #+ 0.5 * (∇⋅u) * u ⊙ v
end

function jac_Ω(t,x,xt,dx,y)
  u, p = x
  du,dp = dx
  v,q = y
  dc_Ω(u, du, v) + a_ΩINS(du,v) + b_Ω(v,dp) + b_Ω(du,q) - sb_Ω(dp,q) - dsc_Ω(u,du,q) #+ 0.5 * (∇⋅u) * du ⊙ v 
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

#Neumann term collection
function res_Γn(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  0*inner(u,v) 
end

function jac_Γn(t,x,xt,dx,y)
  u, p = x
  du,dp = dx
  v,q = y
  0*inner(du,v)
end
  
function jac_tΓn(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  0*inner(dut,v)
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
t_Γn = FETerm(res_Γn,jac_Γn,jac_tΓn,trian_Γn,quad_Γn)
t_Γg = FETerm(res_Γg,jac_Γg,jac_tΓg,trian_Γg,quad_Γg)

op = TransientFEOperator(X,Y,t_Ω,t_Γ,t_Γn,t_Γg)

ls=LUSolver() 
nls = NewtonRaphsonSolver(ls,1e-5,3)

odes = ThetaMethod(nls, dt, θ)
solver = TransientFESolver(odes)
sol_t = solve(solver, op, xh0, t0, tF)

(sol_t)

end #function INS

function writePVD(filePath, trian_Ω, sol; append=false)
    outfiles = paraview_collection(filePath, append=append) do pvd
        for (i, (xh, t)) in enumerate(sol)
            @show i
            uh = restrict(xh[1],trian_Ω)
            ph = restrict(xh[2],trian_Ω)
            pvd[t] = createvtk(
                trian_Ω,
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
writePVD(filePath, trian_Ω, sol_t, append=true)

end #module
