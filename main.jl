#!/usr/bin/env julia

"""
Agile multi-obstacle navigation comparison. A single evading agent must
traverse a spatially constrained slalom using one of three controllers:
distance HOCBF, artificial potential field (APF), or vector-CBF (VCBF).

Examples:
  julia main.jl --scenario slalom
  julia main.jl --scenario slalom --plot
  julia main.jl --scenario slalom --plot --out slalom_comparison.png

Dependency: Plots.jl (the simulations themselves otherwise use Julia stdlib).
"""

using LinearAlgebra
using Printf
import Plots

const EPS = 1e-12

wrap_angle(a) = mod(a + pi, 2pi) - pi
perp(v) = [-v[2], v[1]]
smoothstep(x) = (z = clamp(x, 0.0, 1.0); z*z*(3 - 2z))
function clipnorm(v, vmax)
    n = norm(v)
    n > vmax ? v .* (vmax / (n + EPS)) : copy(v)
end

Base.@kwdef struct RunResult
    strategy::String
    status::String
    time::Float64
    path::Matrix{Float64}
    controls::Matrix{Float64}
    min_clearance::Float64
    infeasible_steps::Int = 0
    extra::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

# ============================================================================
# Multi-obstacle agile navigation benchmark, ported from main.py
# ============================================================================

Base.@kwdef mutable struct ObstacleParams
    margin::Float64 = 0.06; sigma::Float64 = 0.55; beta::Float64 = 1.0
    s_cap::Float64 = 0.12; k1_E::Float64 = 3.0; k2_E::Float64 = 3.0
    gamma_perp::Float64 = 2.5; alpha_gov::Float64 = 2.5
    dtheta_pre::Float64 = 1.3; xi_pre::Float64 = 0.5; d0_pre::Float64 = 0.35
    w_pre::Float64 = 0.4; lane_pre::Float64 = 0.35
    c_t::Float64 = 1.0; c_r::Float64 = 0.6; sigma_r::Float64 = 0.7
    commit::Float64 = 0.35; alpha_dist::Float64 = 2.0; k_hocbf::Float64 = 3.0
    brake_frac::Float64 = 0.7; alpha_brake::Float64 = 1.5
    apf_zeta::Float64 = 1.6; apf_eta::Float64 = 2.2
    apf_sig::Float64 = 0.6; apf_dswitch::Float64 = 1.0
    amax::Float64 = 3.0; wmax::Float64 = 2.5; alat::Float64 = 4.0
    dt::Float64 = 0.028; steps::Int = 1800; k_att::Float64 = 1.6
    activate_dist::Float64 = 2.6; cbf_brake::Bool = false
end

const SCENARIOS = Dict(
    # Alternating disks overlap the straight start-to-goal corridor. The
    # controller must repeatedly redirect laterally, producing the desired
    # sine-wave/slalom motion instead of taking one isolated avoidance turn.
    "slalom" => ([-4.65,0.0], [4.65,0.0],
                 [([-3.10, 0.45],0.58), ([-1.55,-0.45],0.58),
                  ([ 0.00, 0.45],0.58), ([ 1.55,-0.45],0.58),
                  ([ 3.10, 0.45],0.58)], 2.2, 0.50),
    "headon" => ([-2.0,-1.1], [2.0,1.2], [([0.2,0.0],0.85)], 2.5, 0.6),
    "wall"   => ([-2.4,0.0], [2.3,0.0], [([0.2,0.0],1.25)], 2.5, 0.6),
    "gap"    => ([-2.2,0.0], [2.3,0.0], [([0.2,0.9],0.6),([0.2,-0.9],0.6)], 2.5, 0.6),
    "trap"   => ([-2.2,0.8], [2.3,0.8], [([0.2,0.9],0.6),([0.2,-0.9],0.6)], 2.5, 0.6),
    "sprint" => ([-2.3,0.0], [2.4,0.0], [([0.4,0.0],0.6)], 4.0, 1.0))

mutable struct Car
    p::Vector{Float64}; theta::Float64; speed::Float64; vmax::Float64
end
heading(c::Car) = [cos(c.theta), sin(c.theta)]
left(c::Car) = [-sin(c.theta), cos(c.theta)]
omega_cap(c, prm) = min(1.0, prm.alat/(max(c.speed, 0.3)*prm.wmax))
clamp_input(c, u, prm) = [clamp(u[1], -1.0, 1.0), clamp(u[2], -omega_cap(c,prm), omega_cap(c,prm))]

function track_velocity(c, vdes, prm; commit=0.0)
    vref = clipnorm(vdes, c.vmax); sd = norm(vref)
    sd < 1e-6 && return clamp_input(c, [-3.2c.speed/prm.amax, 0.0], prm)
    err = wrap_angle(atan(vref[2], vref[1]) - c.theta)
    sref = sd * max(cos(err), commit)
    clamp_input(c, [3.2(sref-c.speed)/prm.amax, 3err/prm.wmax], prm)
end

field_profiles(s, prm) = begin
    E = 0.5prm.beta^2 * exp(-s^2/prm.sigma^2)
    (prm.beta*exp(-s^2/(2prm.sigma^2)), E,
     -2s/prm.sigma^2*E, 2E*(2s^2/prm.sigma^4 - 1/prm.sigma^2))
end

goal_field(p, goal, prm, vmax) = clipnorm(prm.k_att .* (goal-p), vmax)
function apf_reference(p, goal, obstacles, prm, vmax)
    tg = goal-p; dg = norm(tg)+EPS
    F = dg > prm.apf_dswitch ? prm.apf_zeta*prm.apf_dswitch.*tg/dg : prm.apf_zeta.*tg
    for (center,R) in obstacles
        rel=p-center; d=norm(rel)+EPS; s=max(d-(R+prm.margin),0.0)
        F += prm.apf_eta*exp(-s^2/(2prm.apf_sig^2)).*rel/d
    end
    clipnorm(F,vmax)
end

function preload_horizon(speed, prm)
    weff=min(prm.wmax, prm.alat/max(speed,0.3))
    speed*prm.dtheta_pre/weff + prm.xi_pre*speed^2/(2prm.amax) + prm.d0_pre
end

function vcbf_reference(p,speed,goal,obstacles,prm,vmax,latch)
    tg=goal-p; dg=norm(tg)+EPS; gh=tg/dg
    v=goal_field(p,goal,prm,vmax); engage=0.0; Dpre=preload_horizon(speed,prm)
    for (i,(center,R)) in enumerate(obstacles)
        rel=p-center; d=norm(rel)+EPS; n=rel/d; t=perp(n); s=d-(R+prm.margin)
        oc=center-p; along=dot(gh,oc); off=norm(oc-along.*gh)
        lane=R+prm.margin+prm.lane_pre
        front=smoothstep(along/0.3)*smoothstep((dg+R-along)/0.3)*smoothstep((lane-off)/0.3)
        w=smoothstep((Dpre-s)/prm.w_pre); proj=dot(t,tg); tie=abs(proj)<1e-9 ? 1.0 : sign(proj)
        haskey(latch,i) && front<0.05 && delete!(latch,i)
        !haskey(latch,i) && w>0.35 && front>0.5 && (latch[i]=tie)
        side=get(latch,i,tie); kr=exp(-max(s,0)^2/(2prm.sigma_r^2))
        v += prm.c_r*vmax*kr*(0.1+0.9front).*n + prm.c_t*vmax*w*front*side.*t
        engage=max(engage,front*w)
    end
    for (center,R) in obstacles
        rel=p-center; d=norm(rel)+EPS; n=rel/d; s=d-(R+prm.margin)
        blend=exp(-max(s,0)^2/(2*0.35^2)); inward=min(0.0,dot(v,n)); v-=blend*inward.*n
    end
    clipnorm(v,vmax), engage
end

function barrier_row(c, B, gradB, hessB, rho, gradrho, prm, kho)
    pd=c.speed.*heading(c)
    a=[dot(gradB,heading(c))*prm.amax, c.speed*dot(gradB,left(c))*prm.wmax]
    b=-kho*(dot(gradB,pd)+rho)-dot(pd,hessB*pd)-dot(gradrho,pd)
    a,b
end

function distance_row(c,center,Reff,prm)
    rel=c.p-center; d=norm(rel)+EPS; n=rel/d; P=I-n*n'; s=d-Reff
    barrier_row(c,s,n,P/d,prm.alpha_dist*s,prm.alpha_dist.*n,prm,prm.k_hocbf)
end

function energy_row(c,center,Reff,prm)
    rel=c.p-center; d=norm(rel)+EPS; n=rel/d; P=I-n*n'; s=d-Reff
    _,E,dE,ddE=field_profiles(s,prm); _,Ecap,_,_=field_profiles(prm.s_cap,prm)
    B=Ecap-E; grad=-dE.*n; hess=-(ddE.*(n*n')+(dE/d).*P)
    barrier_row(c,B,grad,hess,prm.k1_E*B,prm.k1_E.*grad,prm,prm.k2_E)
end

function brake_row(c,center,Reff,prm)
    rel=c.p-center; d=norm(rel)+EPS; n=rel/d; s=d-Reff; vr=c.speed*dot(n,heading(c))
    abrk=prm.brake_frac*prm.amax; hb=2abrk*(s-prm.s_cap)-c.speed^2
    [-2c.speed*prm.amax,0.0], -prm.alpha_brake*hb-2abrk*vr
end

function governor_row(c,center,Reff,prm)
    rel=c.p-center; d=norm(rel)+EPS; n=rel/d; t=perp(n); s=d-Reff
    e=heading(c); mu=dot(n,e); te=dot(t,e); vr=c.speed*mu; vp=c.speed*te
    vrm=max(-vr,0.0); chi=vr<0 ? 1.0 : 0.0; abrk=prm.brake_frac*prm.amax
    hg=2abrk*(s-prm.s_cap)-vrm^2+prm.gamma_perp*vp^2
    ca=chi*2vrm*mu+2prm.gamma_perp*vp*te
    cw=chi*2vrm*(-vp)+2prm.gamma_perp*vp*mu*c.speed
    con=2abrk*vr+chi*2vrm*vp^2/d-2prm.gamma_perp*vp^2*vr/d
    [ca*prm.amax,cw*prm.wmax], -prm.alpha_gov*hg-con
end

function project_rows(u0,A,b; passes=14)
    u=copy(u0)
    for _ in 1:passes, k in eachindex(b)
        gap=dot(A[k],u)-b[k]
        gap<0 && (u += (-gap/(dot(A[k],A[k])+EPS)).*A[k])
    end
    u
end

max_violation(A,b,u)=isempty(b) ? 0.0 : maximum(b[k]-dot(A[k],u) for k in eachindex(b))
function bounded_projection(c,u0,A,b,prm)
    u=copy(u0)
    for _ in 1:5
        u=project_rows(u,A,b); u=clamp_input(c,u,prm)
    end
    u, max_violation(A,b,u)
end

function obstacle_control(strategy,c,goal,obstacles,prm,latch)
    if strategy=="apf"
        u=track_velocity(c,apf_reference(c.p,goal,obstacles,prm,c.vmax),prm)
        return u,0.0
    end
    if strategy=="cbf"
        unom=track_velocity(c,goal_field(c.p,goal,prm,c.vmax),prm); engage=0.0
    else
        vd,engage=vcbf_reference(c.p,c.speed,goal,obstacles,prm,c.vmax,latch)
        unom=track_velocity(c,vd,prm;commit=prm.commit*smoothstep(engage/0.3))
    end
    A=Vector{Vector{Float64}}(); b=Float64[]
    actdyn=max(prm.activate_dist,c.speed^2/(2prm.brake_frac*prm.amax)+0.4)
    for (center,R) in obstacles
        s=norm(c.p-center)-(R+prm.margin)
        if s<=prm.activate_dist
            row=strategy=="cbf" ? distance_row(c,center,R+prm.margin,prm) : energy_row(c,center,R+prm.margin,prm)
            push!(A,row[1]); push!(b,row[2])
        end
        if strategy=="vcbf" && s<=actdyn
            row=governor_row(c,center,R+prm.margin,prm); push!(A,row[1]); push!(b,row[2])
        elseif strategy=="cbf" && prm.cbf_brake && s<=actdyn
            row=brake_row(c,center,R+prm.margin,prm); push!(A,row[1]); push!(b,row[2])
        end
    end
    bounded_projection(c,unom,A,b,prm)
end

function run_obstacle(strategy, scenario="headon"; prm=ObstacleParams())
    haskey(SCENARIOS,scenario) || error("Unknown scenario: $scenario")
    start,goal,obstacles,vmax,v0frac=SCENARIOS[scenario]
    c=Car(copy(start),atan((goal-start)[2],(goal-start)[1]),v0frac*vmax,vmax)
    path=[copy(c.p)]; controls=Vector{Vector{Float64}}(); latch=Dict{Int,Float64}()
    infeasible=0; minclear=Inf; status="timeout"; stall=0
    for _ in 1:prm.steps
        clearances=[norm(c.p-center)-R for (center,R) in obstacles]
        minclear=min(minclear,minimum(clearances)); nearest=argmin(clearances)
        center,R=obstacles[nearest]; d=norm(c.p-center)
        if d<=R; status="COLLISION"; break
        elseif d<=R+prm.margin+1e-9; status=c.speed>=0.15 ? "BOUNDARY" : "PINNED"; break
        elseif norm(c.p-goal)<0.18; status="goal"; break
        end
        stall = c.speed<0.03 ? stall+1 : 0
        stall>60 && (status="STALLED"; break)
        u,viol=obstacle_control(strategy,c,goal,obstacles,prm,latch)
        infeasible += viol>1e-6; push!(controls,copy(u))
        a,w=u[1]*prm.amax,u[2]*prm.wmax
        c.speed=clamp(c.speed+prm.dt*a,0.0,c.vmax)
        c.theta=wrap_angle(c.theta+prm.dt*w)
        c.p += prm.dt*c.speed.*heading(c)
        push!(path,copy(c.p))
    end
    P=reduce(hcat,path)'; U=isempty(controls) ? zeros(0,2) : reduce(hcat,controls)'
    path_length=sum(norm(path[i]-path[i-1]) for i in 2:length(path))
    control_effort=sum(sum(abs2,u) for u in controls)*prm.dt
    RunResult(strategy=uppercase(strategy),status=status,time=(length(path)-1)*prm.dt,
              path=P,controls=U,min_clearance=minclear-prm.margin,
              infeasible_steps=infeasible,
              extra=Dict(:scenario=>scenario,:obstacles=>obstacles,:goal=>goal,
                         :start=>start,:margin=>prm.margin,
                         :path_length=>path_length,:control_effort=>control_effort))
end

function print_results(results)
    println("\nstrategy  status       time(s)  path(m)  avg speed  min safe gap  effort  infeasible")
    println("-"^91)
    for r in results
        L=r.extra[:path_length]; avg=L/max(r.time,EPS); effort=r.extra[:control_effort]
        @printf("%-9s %-11s %8.2f %8.2f %10.2f %13.3f %7.2f %11d\n",
                r.strategy,r.status,r.time,L,avg,r.min_clearance,effort,r.infeasible_steps)
    end
end

function plot_results(results, outfile)
    scenario=results[1].extra[:scenario]
    plt=Plots.plot(aspect_ratio=:equal,xlabel="x [m]",ylabel="y [m]",
                   title="Agile navigation comparison — $scenario",
                   legend=:outerright,size=(1100,650))
    obstacles=results[1].extra[:obstacles]; margin=results[1].extra[:margin]
    for (c,R) in obstacles
        θ=range(0,2pi,length=160)
        Plots.plot!(plt,c[1].+R*cos.(θ),c[2].+R*sin.(θ),
                    color=:black,fill=(0,:gray85),label=false,lw=1.5)
        Plots.plot!(plt,c[1].+(R+margin)*cos.(θ),c[2].+(R+margin)*sin.(θ),
                    color=:gray45,ls=:dot,label=false,lw=1)
    end
    colors=Dict("CBF"=>:magenta,"APF"=>:orange,"VCBF"=>:cyan3)
    for r in results
        label="$(r.strategy): $(r.status), $(round(r.time,digits=2)) s"
        Plots.plot!(plt,r.path[:,1],r.path[:,2],label=label,lw=2.7,
                    color=colors[r.strategy])
    end
    start=results[1].extra[:start]; goal=results[1].extra[:goal]
    Plots.scatter!(plt,[start[1]],[start[2]],marker=:diamond,color=:green,
                   ms=7,label="Start")
    Plots.scatter!(plt,[goal[1]],[goal[2]],marker=:star5,color=:red,
                   ms=9,label="Goal")
    Plots.savefig(plt,outfile); println("Saved $outfile")
end

function parse_args(args)
    opts=Dict("scenario"=>"slalom","plot"=>"false","out"=>"slalom_comparison.png","brake"=>"false")
    i=1
    while i<=length(args)
        if args[i] in ("--plot","--brake")
            opts[args[i][3:end]]="true"
        elseif args[i] in ("--scenario","--out")
            i==length(args) && error("Missing value after $(args[i])")
            opts[args[i][3:end]]=args[i+1]; i+=1
        elseif args[i] in ("-h","--help")
            println("Usage: julia main.jl [--scenario slalom|headon|wall|gap|trap|sprint] [--brake] [--plot] [--out FILE]")
            exit()
        else
            error("Unknown argument: $(args[i])")
        end
        i+=1
    end
    opts
end

function main(args=ARGS)
    o=parse_args(args)
    prm=ObstacleParams(cbf_brake=o["brake"]=="true")
    results=[run_obstacle(s,o["scenario"];prm=prm) for s in ("cbf","apf","vcbf")]
    print_results(results)
    o["plot"]=="true" && plot_results(results,o["out"])
    results
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
