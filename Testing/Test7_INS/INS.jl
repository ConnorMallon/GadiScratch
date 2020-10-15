#INS eq test when u ∈ V

module INS

using Pkg
Pkg.activate(".")

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using GridapEmbedded
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t
using LineSearches: BackTracking
using Gridap.Algebra: NewtonRaphsonSolver

@law conv(u, ∇u) = (∇u') ⋅ u
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u) #Changing to using the linear solver

#Physical constants
ρ = 1.06e-3 #kg/cm^3 
μ = 3.50e-5  #kg/cm.s
ν = μ/ρ 

θ = 1

k=2*pi

u(x,t) = VectorValue(x[1]+x[2],-x[1]-x[2])*t
u(t::Real) = x -> u(x,t)
ud(t) = x -> u(t)(x)

p(x,t) = (x[1]-x[2])*t
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)

f(t) = x -> ∂t(u)(t)(x) + conv(u(t)(x),∇(u(t))(x)) - ν*Δ(u(t))(x) + ∇(p(t))(x)
g(t) = x -> (∇⋅u(t))(x)

order = 1

#Background Model
R = 0.7
geom = disk(R)
n=3
partition = (n,n)
D=length(partition)
box = get_metadata(geom)
bgmodel = simplexify(CartesianDiscreteModel(box.pmin,box.pmax,partition))
dp = box.pmax - box.pmin
h = dp[1]/n

cutgeo = cut(bgmodel,geom)
model = DiscreteModel(cutgeo)

#Triangulation
trian_Ω = Triangulation(cutgeo)
trian_Γ = EmbeddedBoundary(cutgeo)
trian_Γg = GhostSkeleton(cutgeo)

n_Γ = get_normal_vector(trian_Γ)
n_Γg = get_normal_vector(trian_Γg)

quad_Ω = CellQuadrature(trian_Ω,2*order)
quad_Γ = CellQuadrature(trian_Γ,2*order)
quad_Γg = CellQuadrature(trian_Γg,2*order)

#Spaces
V0 = FESpace(
  reffe=:PLagrangian,
  order=order,
  valuetype=VectorValue{D,Float64},
  conformity=:H1,
  model=model)

Q = TestFESpace(
  model=model,
  order=order,
  reffe=:PLagrangian,
  valuetype=Float64,
  conformity=:H1,
  constraint=:zeromean,
  zeromean_trian=trian_Ω)

U = TrialFESpace(V0)
P = TrialFESpace(Q)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

#Stabilisation parameters
β1 = 0.2
β2 = 0.1
β3 = 0.05
γ = 10

# Weak form
m_Ω(ut,v) = ut⊙v
a_Ω(u,v) = ν * ∇(u)⊙∇(v)
b_Ω(v,p) = - (∇⋅v)*p

c_Ω(u, v) = v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = v ⊙ dconv(du, ∇(du), u, ∇(u))

sp_Ω(p,q) = (β1*h^2)*∇(p)⋅∇(q)
st_Ω(ut,q) = (β1*h^2)*(ut⋅∇(q))
ϕ_Ω(q,t) = (β1*h^2)*∇(q)⋅f(t)

a_Γ(u,v) = ν* ( - (n_Γ⋅∇(u))⋅v - u⋅(n_Γ⋅∇(v)) + (γ/h)*u⋅v )
b_Γ(v,p) = (n_Γ⋅v)*p

i_Γg(u,v) = ν * (β2*h)*jump(n_Γg⋅∇(u))⋅jump(n_Γg⋅∇(v))
j_Γg(p,q) = (β3*h^3)*jump(n_Γg⋅∇(p))*jump(n_Γg⋅∇(q))

#Interior term collection
function res_Ω(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  m_Ω(ut,v) + c_Ω(u,v) + a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) - sp_Ω(p,q) - st_Ω(ut,q) - v⋅f(t) + q*g(t) + ϕ_Ω(q,t)
end

function jac_Ω(t,x,xt,dx,y)
  u, p = x
  du,dp = dx
  v,q = y
  dc_Ω(u, du, v) + a_Ω(du,v) + b_Ω(v,dp) + b_Ω(du,q) - sp_Ω(dp,q)
end

function jac_tΩ(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  m_Ω(dut,v) - st_Ω(dut,q)
end

#Boundary term collection
function res_Γ(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  a_Γ(u,v)+b_Γ(u,q)+b_Γ(v,p) - ud(t) ⊙( ν * (γ/h)*v - ν * n_Γ⋅∇(v) + q*n_Γ )
end

function jac_Γ(t,x,xt,dx,y)
  du,dp = dx
  v,q = y
  a_Γ(du,v)+b_Γ(du,q)+b_Γ(v,dp)
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
  i_Γg(u,v) - j_Γg(p,q)
end

function jac_Γg(t,x,xt,dx,y)
  du,dp = dx
  v,q = y
  i_Γg(du,v) - j_Γg(dp,q)
end

function jac_tΓg(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  0*i_Γg(dut,v)
end

X0 = X(0.0)
xh0 = interpolate_everywhere(X0,[u(0.0),p(0.0)])

t_Ω = FETerm(res_Ω,jac_Ω,jac_tΩ,trian_Ω,quad_Ω)
t_Γ = FETerm(res_Γ,jac_Γ,jac_tΓ,trian_Γ,quad_Γ)
t_Γg = FETerm(res_Γg,jac_Γg,jac_tΓg,trian_Γg,quad_Γg)

t0 = 0.0
tF = 1.0
dt = 1.0

ls = LUSolver()
nls = NewtonRaphsonSolver(ls,1e-5,99)

op = TransientFEOperator(X,Y,t_Ω,t_Γ,t_Γg)

#ls=LUSolver()
#nls = NewtonRaphsonSolver(ls,1e-5,2)

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
  global _t_n += dt
  uh_tn = xh_tn[1]
  ph_tn = xh_tn[2]
  uh_Ω = restrict(uh_tn,trian_Ω)
  ph_Ω = restrict(ph_tn,trian_Ω)
  e = u(tn) - uh_Ω
  eul2i = sqrt(sum( integrate(l2(e),trian_Ω,quad_Ω) ))
  @test eul2i < tol
  e = p(tn) - ph_Ω
  epl2i = sqrt(sum( integrate(l2(e),trian_Ω,quad_Ω) ))
  @test epl2i < tol
  push!(eul2,eul2i)
  push!(epl2,epl2i)
end

@show "something"

end #module
