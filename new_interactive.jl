#!/usr/bin/env julia

"""
Stationary-obstacle adaptation of the multi-directional PE1 HOCBF.

A single kinematic-bicycle pursuer/agent travels to a goal through five fixed
circular obstacles. At every step it evaluates left, straight, and right
maneuvers. Each candidate is filtered by all active obstacle CBFs, predicted
over a short horizon, and scored for safety, goal progress, and traversal cost.

State:   [x, y, psi, v]
Control: [a, kappa], where delta = atan(wheelbase*kappa)
Dynamics: xdot=v*cos(psi), ydot=v*sin(psi), psidot=v*kappa, vdot=a

Compared controllers:
  MDCBF  left/straight/right predictive selection + distance HOCBF
  ACBF   main.py distance HOCBF + acceleration/braking-distance row
  VCBF   main.py Gaussian energy CBF + preload + momentum governor

`main.py` calls its acceleration-aware baseline `cbf`; this file labels that
combination `ACBF` to match the requested comparison terminology.

Run: julia new_interactive.jl             # interactive editor/dashboard
Video export: julia new_interactive.jl --video
Static plot: julia new_interactive.jl --static
"""

using CairoMakie
using LinearAlgebra
using OSQP
using Printf
using SparseArrays

const DT, T_MAX = 0.03, 35.0
const N_STEPS = round(Int, T_MAX / DT)
const AGENT_RADIUS = 0.15
const SAFETY_MARGIN = 0.08
const OBSTACLE_RADIUS = 0.58
const SAFE_DISTANCE = AGENT_RADIUS + SAFETY_MARGIN + OBSTACLE_RADIUS
const K1, K2 = 3.0, 2.0
const CBF_ACTIVATION_DISTANCE = 1.15
const V_MAX = 1.8
const A_MIN, A_MAX = -3.0, 3.0
const WHEELBASE = 0.55
const STEER_MAX = deg2rad(40.0)
const KAPPA_MAX = tan(STEER_MAX)/WHEELBASE
const A_LAT_MAX = 8.0
const START, GOAL = [-5.50, 0.0], [5.50, 0.0]
const MANEUVER_ANGLE = 0.72
const PREDICTION_HORIZON = 1.35
const PREDICTION_DT = 0.075

# main.py acceleration-aware CBF baseline (called ACBF in this comparison).
const BRAKE_FRACTION = 0.70
const BRAKE_ALPHA = 1.50
const BRAKE_CLEARANCE_FLOOR = 0.05

# main.py Gaussian vector-CBF, preload, and momentum-governor parameters.
const FIELD_SIGMA = 0.55
const FIELD_BETA = 1.0
const VCBF_CLEARANCE_FLOOR = 0.12
const VCBF_K1, VCBF_K2 = 3.0, 3.0
const GOVERNOR_GAMMA, GOVERNOR_ALPHA = 2.5, 2.5
const PRELOAD_ANGLE, PRELOAD_BRAKE_WEIGHT = 1.30, 0.50
const PRELOAD_FLOOR, PRELOAD_WIDTH = 0.35, 0.40
const PRELOAD_LANE = 0.35
const FIELD_RADIAL_GAIN, FIELD_TANGENTIAL_GAIN = 0.60, 1.00
const FIELD_RADIAL_SIGMA = 0.70
const VCBF_COMMIT = 0.35

const CONTROLLERS = (:mdcbf,:acbf,:vcbf)
const CONTROLLER_LABEL = Dict(:mdcbf=>"MDCBF",:acbf=>"ACBF",:vcbf=>"VCBF")
const CONTROLLER_COLOR = Dict(:mdcbf=>:dodgerblue,:acbf=>:orange,:vcbf=>:seagreen3)

const OBSTACLES = [
    [-4.00,  0.65],
    [-2.00, -0.65],
    [ 0.00,  0.65],
    [ 2.00, -0.65],
    [ 4.00,  0.65],
]
const OBSTACLE_RADII = fill(OBSTACLE_RADIUS,length(OBSTACLES))
safe_distance(index::Integer)=AGENT_RADIUS+SAFETY_MARGIN+OBSTACLE_RADII[index]
obstacle_index(center::AbstractVector)=something(
    findfirst(candidate->candidate===center,OBSTACLES),
    findfirst(candidate->candidate==center,OBSTACLES))
safe_distance(center::AbstractVector)=safe_distance(obstacle_index(center))
obstacle_radius(center::AbstractVector)=OBSTACLE_RADII[obstacle_index(center)]

wrap_angle(a) = mod(a + pi, 2pi) - pi
vehicle_dynamics(x,u) = [x[4]*cos(x[3]),x[4]*sin(x[3]),x[4]*u[2],u[1]]
steering_angle(curvature) = atan(WHEELBASE*curvature)

"""Curvature bound from front-steering travel and lateral acceleration."""
function curvature_limit(speed)
    friction_limit=A_LAT_MAX/max(speed^2,0.30^2)
    min(KAPPA_MAX,friction_limit)
end

const MANEUVERS = ((:right,-1.0),(:straight,0.0),(:left,1.0))

"""Nominal control for one of the three directional motion hypotheses."""
function maneuver_nominal(state, direction)
    to_goal = GOAL-state[1:2]
    goal_heading = atan(to_goal[2],to_goal[1])
    desired_heading = goal_heading + direction*MANEUVER_ANGLE
    heading_error = wrap_angle(desired_heading-state[3])
    desired_speed = V_MAX*max(0.48,1-0.55*abs(heading_error)/pi)
    # Convert a heading-rate request into bicycle curvature: omega=v*kappa.
    kappa_limit = curvature_limit(state[4])
    desired_yaw_rate = 3.8*heading_error
    desired_curvature = desired_yaw_rate/max(state[4],0.30)
    [clamp(2.6*(desired_speed-state[4]),A_MIN,A_MAX),
     clamp(desired_curvature,-kappa_limit,kappa_limit)]
end

perp(v)=[-v[2],v[1]]
smoothstep(x)=(z=clamp(x,0.0,1.0);z*z*(3-2z))
function clipnorm(v,limit)
    magnitude=norm(v)
    magnitude>limit ? v.*(limit/(magnitude+1e-12)) : copy(v)
end

"""Convert a desired world velocity into bicycle acceleration and curvature."""
function track_velocity(state,desired_velocity;commit=0.0)
    desired=clipnorm(desired_velocity,V_MAX)
    desired_speed=norm(desired)
    desired_speed<1e-8 && return [-2.6state[4],0.0]
    desired_heading=atan(desired[2],desired[1])
    heading_error=wrap_angle(desired_heading-state[3])
    speed_reference=desired_speed*max(cos(heading_error),commit)
    acceleration=clamp(2.6*(speed_reference-state[4]),A_MIN,A_MAX)
    curvature=3.8*heading_error/max(state[4],0.30)
    [acceleration,clamp(curvature,-curvature_limit(state[4]),curvature_limit(state[4]))]
end

function goal_velocity(state)
    clipnorm(1.6.*(GOAL-state[1:2]),V_MAX)
end

"""
For h=||p-c||²-safe_distance(c)², construct
    Arow*[a,kappa] >= brow
so h_ddot + K1*h_dot + K2*h >= 0.
The obstacle velocity and acceleration are both zero.
"""
function obstacle_cbf_row(state, center)
    p, psi, speed = state[1:2], state[3], state[4]
    dp = p-center
    e = [cos(psi), sin(psi)]
    eleft = [-sin(psi), cos(psi)]
    velocity = speed.*e
    h = dot(dp,dp)-safe_distance(center)^2
    hdot = 2dot(dp,velocity)
    # p_ddot = a*e + v^2*kappa*e_left for the bicycle model.
    row = [2dot(dp,e), 2speed^2*dot(dp,eleft)]
    bound = -2dot(velocity,velocity)-K1*hdot-K2*h
    row, bound, h, hdot
end

"""PE1-style minimum-adjustment QP shared by all three safety filters."""
function solve_control_qp(state,nominal,rows,bounds;weights=[12.0,0.35])
    Acbf = isempty(rows) ? zeros(0,2) : reduce(vcat,permutedims.(rows))
    A = sparse(vcat(Acbf,Matrix{Float64}(I,2,2)))
    kappa_limit = curvature_limit(state[4])
    lower = vcat(bounds,[A_MIN,-kappa_limit])-A*nominal
    upper = vcat(fill(Inf,length(bounds)),[A_MAX,kappa_limit])-A*nominal
    model = OSQP.Model()
    OSQP.setup!(model;P=sparse(2diagm(weights)),q=zeros(2),A=A,
                l=lower,u=upper,verbose=false,eps_abs=1e-6,eps_rel=1e-6,
                polish=true)
    solution = OSQP.solve!(model)
    solved = occursin("solved",lowercase(string(solution.info.status)))
    safe = solved ? nominal+solution.x : nominal
    safe[1] = clamp(safe[1],A_MIN,A_MAX)
    safe[2] = clamp(safe[2],-kappa_limit,kappa_limit)
    safe,solved
end

geometric_h(state)=minimum(norm(state[1:2]-c)^2-safe_distance(i)^2
                           for (i,c) in enumerate(OBSTACLES))

"""Distance-HOCBF filter used by the multi-directional controller."""
function cbf_safe_control(state,nominal)
    rows=Vector{Vector{Float64}}();bounds=Float64[]
    for (index,center) in enumerate(OBSTACLES)
        row,bound,_,_=obstacle_cbf_row(state,center)
        if norm(state[1:2]-center)-OBSTACLE_RADII[index]<=CBF_ACTIVATION_DISTANCE
            push!(rows,row);push!(bounds,bound)
        end
    end
    safe,solved=solve_control_qp(state,nominal,rows,bounds)
    safe,geometric_h(state),solved
end

"""Acceleration-aware braking-distance CBF row ported from main.py."""
function braking_cbf_row(state,center)
    relative=state[1:2]-center;distance=norm(relative)+1e-12
    normal=relative/distance;gap=distance-safe_distance(center)
    radial_speed=state[4]*dot(normal,[cos(state[3]),sin(state[3])])
    braking=BRAKE_FRACTION*A_MAX
    barrier=2braking*(gap-BRAKE_CLEARANCE_FLOOR)-state[4]^2
    [-2state[4],0.0],-BRAKE_ALPHA*barrier-2braking*radial_speed
end

"""Choose a directional nominal without applying a safety filter yet."""
function directional_nominal_choice(state,previous_mode)
    best=nothing
    for (mode,direction) in MANEUVERS
        nominal=maneuver_nominal(state,direction)
        score=maneuver_score(state,direction,nominal,mode,previous_mode)
        if best===nothing||score<best.score
            best=(;mode,direction,control=nominal,score)
        end
    end
    best
end

"""main.py baseline: distance HOCBF plus acceleration/braking CBF."""
function acbf_control(state,previous_mode)
    nominal_choice=directional_nominal_choice(state,previous_mode)
    nominal=nominal_choice.control
    rows=Vector{Vector{Float64}}();bounds=Float64[]
    dynamic_reach=max(CBF_ACTIVATION_DISTANCE,
                      state[4]^2/(2BRAKE_FRACTION*A_MAX)+0.40)
    for (index,center) in enumerate(OBSTACLES)
        gap=norm(state[1:2]-center)-safe_distance(index)
        if gap<=CBF_ACTIVATION_DISTANCE
            row,bound,_,_=obstacle_cbf_row(state,center)
            push!(rows,row);push!(bounds,bound)
        end
        if gap<=dynamic_reach
            row,bound=braking_cbf_row(state,center)
            push!(rows,row);push!(bounds,bound)
        end
    end
    safe,solved=solve_control_qp(state,nominal,rows,bounds)
    safe,geometric_h(state),solved,nominal_choice.mode
end

function field_profiles(gap)
    energy=0.5FIELD_BETA^2*exp(-gap^2/FIELD_SIGMA^2)
    denergy=-2gap/FIELD_SIGMA^2*energy
    ddenergy=2energy*(2gap^2/FIELD_SIGMA^4-1/FIELD_SIGMA^2)
    energy,denergy,ddenergy
end

function preload_horizon(speed)
    max_curvature=curvature_limit(max(speed,0.30))
    effective_yaw=max(speed,0.30)*max_curvature
    speed*PRELOAD_ANGLE/max(effective_yaw,1e-6)+
        PRELOAD_BRAKE_WEIGHT*speed^2/(2A_MAX)+PRELOAD_FLOOR
end

"""Gaussian preload reference ported from main.py and mapped to bicycle inputs."""
function vcbf_reference(state,latch)
    p=state[1:2];to_goal=GOAL-p;goal_distance=norm(to_goal)+1e-12
    goal_direction=to_goal/goal_distance
    desired=goal_velocity(state);engagement=0.0
    horizon=preload_horizon(state[4])
    for (i,center) in enumerate(OBSTACLES)
        relative=p-center;distance=norm(relative)+1e-12
        normal=relative/distance;tangent=perp(normal);gap=distance-safe_distance(i)
        obstacle_vector=center-p
        along=dot(goal_direction,obstacle_vector)
        off=norm(obstacle_vector-along.*goal_direction)
        lane=safe_distance(i)+PRELOAD_LANE
        front=smoothstep(along/0.30)*smoothstep((goal_distance+OBSTACLE_RADII[i]-along)/0.30)*
              smoothstep((lane-off)/0.30)
        gate=smoothstep((horizon-gap)/PRELOAD_WIDTH)
        projection=dot(tangent,to_goal)
        tie=abs(projection)<1e-9 ? 1.0 : sign(projection)
        haskey(latch,i)&&front<0.05&&delete!(latch,i)
        !haskey(latch,i)&&gate>0.35&&front>0.50&&(latch[i]=tie)
        side=get(latch,i,tie)
        radial=exp(-max(gap,0.0)^2/(2FIELD_RADIAL_SIGMA^2))
        desired+=FIELD_RADIAL_GAIN*V_MAX*radial*(0.1+0.9front).*normal
        desired+=FIELD_TANGENTIAL_GAIN*V_MAX*gate*front*side.*tangent
        engagement=max(engagement,front*gate)
    end
    for (index,center) in enumerate(OBSTACLES)
        relative=p-center;distance=norm(relative)+1e-12;normal=relative/distance
        gap=distance-safe_distance(index);blend=exp(-max(gap,0.0)^2/(2*0.35^2))
        inward=min(0.0,dot(desired,normal));desired-=blend*inward.*normal
    end
    clipnorm(desired,V_MAX),engagement
end

function energy_cbf_row(state,center)
    p=state[1:2];relative=p-center;distance=norm(relative)+1e-12
    normal=relative/distance;projector=I-normal*normal';gap=distance-safe_distance(center)
    energy,denergy,ddenergy=field_profiles(gap)
    energy_cap=field_profiles(VCBF_CLEARANCE_FLOOR)[1]
    barrier=energy_cap-energy
    gradient=-denergy.*normal
    hessian=-(ddenergy.*(normal*normal')+(denergy/distance).*projector)
    velocity=state[4].*[cos(state[3]),sin(state[3])]
    e=[cos(state[3]),sin(state[3])];eleft=perp(e)
    row=[dot(gradient,e),state[4]^2*dot(gradient,eleft)]
    rho=VCBF_K1*barrier;grad_rho=VCBF_K1.*gradient
    bound=-VCBF_K2*(dot(gradient,velocity)+rho)-
          dot(velocity,hessian*velocity)-dot(grad_rho,velocity)
    row,bound
end

"""Momentum governor from main.py, with omega=v*kappa for the bicycle."""
function governor_row(state,center)
    p=state[1:2];relative=p-center;distance=norm(relative)+1e-12
    normal=relative/distance;tangent=perp(normal);gap=distance-safe_distance(center)
    e=[cos(state[3]),sin(state[3])];mu=dot(normal,e);te=dot(tangent,e)
    radial=state[4]*mu;transverse=state[4]*te;inward=max(-radial,0.0)
    active=radial<0 ? 1.0 : 0.0;braking=BRAKE_FRACTION*A_MAX
    barrier=2braking*(gap-VCBF_CLEARANCE_FLOOR)-inward^2+
            GOVERNOR_GAMMA*transverse^2
    coef_a=active*2inward*mu+2GOVERNOR_GAMMA*transverse*te
    coef_yaw=active*2inward*(-transverse)+
             2GOVERNOR_GAMMA*transverse*mu*state[4]
    constant=2braking*radial+active*2inward*transverse^2/distance-
             2GOVERNOR_GAMMA*transverse^2*radial/distance
    [coef_a,coef_yaw*state[4]],-GOVERNOR_ALPHA*barrier-constant
end

function vcbf_control(state,latch)
    desired,engagement=vcbf_reference(state,latch)
    nominal=track_velocity(state,desired;
                           commit=VCBF_COMMIT*smoothstep(engagement/0.30))
    energy_rows=Vector{Vector{Float64}}();energy_bounds=Float64[]
    governor_rows=Vector{Vector{Float64}}();governor_bounds=Float64[]
    dynamic_reach=max(CBF_ACTIVATION_DISTANCE,
                      state[4]^2/(2BRAKE_FRACTION*A_MAX)+0.40)
    for (index,center) in enumerate(OBSTACLES)
        gap=norm(state[1:2]-center)-safe_distance(index)
        if gap<=CBF_ACTIVATION_DISTANCE
            row,bound=energy_cbf_row(state,center)
            push!(energy_rows,row);push!(energy_bounds,bound)
        end
        if gap<=dynamic_reach
            row,bound=governor_row(state,center)
            push!(governor_rows,row);push!(governor_bounds,bound)
        end
    end
    safe,solved=solve_control_qp(state,nominal,vcat(energy_rows,governor_rows),
                                 vcat(energy_bounds,governor_bounds))
    # The energy certificate is the hard safety constraint. If the auxiliary
    # momentum governor conflicts with actuator bounds, retain the certificate
    # and retry without only the advisory governor rows.
    if !solved
        safe,solved=solve_control_qp(state,nominal,energy_rows,energy_bounds)
    end
    safe,geometric_h(state),solved
end

"""
Predict one directional hypothesis. The first step uses its CBF-filtered
control; later prediction steps keep tracking the same left/straight/right
hypothesis. A lower score means faster goal progress with more clearance.
"""
function maneuver_score(state, direction, first_safe, mode, previous_mode)
    predicted = copy(state)
    min_clearance = Inf
    path_cost = 0.0
    n_predict = round(Int,PREDICTION_HORIZON/PREDICTION_DT)
    for k in 1:n_predict
        u = k==1 ? first_safe : maneuver_nominal(predicted,direction)
        predicted = predicted + PREDICTION_DT.*vehicle_dynamics(predicted,u)
        predicted[3] = wrap_angle(predicted[3])
        predicted[4] = clamp(predicted[4],0.0,V_MAX)
        clearance = minimum(norm(predicted[1:2]-c)-safe_distance(i)
                            for (i,c) in enumerate(OBSTACLES))
        min_clearance = min(min_clearance,clearance)
        path_cost += PREDICTION_DT*predicted[4]
    end
    collision_cost = min_clearance<0 ? 1.0e5+1.0e4*abs(min_clearance) : 0.0
    proximity_cost = 22.0*exp(-max(min_clearance,0.0)/0.28)
    goal_cost = 5.0*norm(predicted[1:2]-GOAL)
    # The humanoid is assumed to operate in a spatially constrained corridor;
    # leaving the center band is expensive, so going around the entire obstacle
    # field cannot beat weaving through its alternating openings.
    corridor_cost = 12.0*abs(predicted[2])
    turn_cost = 0.10*abs(direction)+0.025*abs(first_safe[2])
    switching_cost = previous_mode==:none || previous_mode==mode ? 0.0 : 0.22
    collision_cost+proximity_cost+goal_cost+corridor_cost+turn_cost+
        switching_cost+0.04*path_cost
end

"""
The actual multi-directional controller: filter and score all three candidate
maneuvers, then return the best safe action and its diagnostic information.
"""
function multidirectional_control(state,previous_mode)
    best = nothing
    scores = Dict{Symbol,Float64}()
    for (mode,direction) in MANEUVERS
        nominal = maneuver_nominal(state,direction)
        safe,min_h,solved = cbf_safe_control(state,nominal)
        score = maneuver_score(state,direction,safe,mode,previous_mode)
        !solved && (score += 1.0e6)
        scores[mode] = score
        if best===nothing || score<best.score
            best=(;mode,direction,control=safe,min_h,solved,score)
        end
    end
    best,scores
end

function run_controller(controller)
    controller in CONTROLLERS||error("Unknown controller: $controller")
    state = [START[1],START[2],0.0,0.7V_MAX]
    states = [copy(state)]
    controls = Vector{Vector{Float64}}()
    min_h_history = Float64[]
    maneuver_history = Symbol[]
    score_history = Vector{Dict{Symbol,Float64}}()
    fallback_steps = 0
    status = "timeout"
    previous_mode = :none
    latch=Dict{Int,Float64}()
    stall_steps=0

    for _ in 1:N_STEPS
        if norm(state[1:2]-GOAL)<0.30
            status="goal"; break
        end
        geometric_h(state)<-1e-6&&(status="BOUNDARY";break)
        stall_steps=state[4]<0.03 ? stall_steps+1 : 0
        stall_steps>80&&(status="STALLED";break)
        if controller==:mdcbf
            choice,scores=multidirectional_control(state,previous_mode)
            control,min_h,solved,mode=choice.control,choice.min_h,choice.solved,choice.mode
        elseif controller==:acbf
            control,min_h,solved,mode=acbf_control(state,previous_mode);scores=Dict{Symbol,Float64}()
        else
            control,min_h,solved=vcbf_control(state,latch);mode=:vcbf;scores=Dict{Symbol,Float64}()
        end
        fallback_steps += !solved
        state = state + DT.*vehicle_dynamics(state,control)
        state[3] = wrap_angle(state[3])
        state[4] = clamp(state[4],0.0,V_MAX)
        previous_mode = mode
        push!(states,copy(state)); push!(controls,copy(control))
        push!(min_h_history,min_h); push!(maneuver_history,mode)
        push!(score_history,scores)
    end

    positions = reduce(hcat,(s[1:2] for s in states))'
    U = isempty(controls) ? zeros(0,2) : reduce(hcat,controls)'
    clearance = minimum(norm(states[k][1:2]-c)-OBSTACLE_RADII[i]-AGENT_RADIUS
                        for k in eachindex(states),(i,c) in enumerate(OBSTACLES))
    elapsed = (length(states)-1)*DT
    path_length = sum(norm(positions[k,:]-positions[k-1,:]) for k in 2:size(positions,1))
    label=CONTROLLER_LABEL[controller]
    println("\n--- $label result ---")
    println("Status:             $status")
    @printf("Traversal time:     %.2f s\n",elapsed)
    @printf("Path length:        %.2f m\n",path_length)
    @printf("Minimum clearance:  %.3f m (requested margin %.3f m)\n",clearance,SAFETY_MARGIN)
    println("QP fallback steps:  $fallback_steps")
    counts=controller==:mdcbf ? Dict(m=>count(==(m),maneuver_history) for (m,_) in MANEUVERS) : Dict{Symbol,Int}()
    controller==:mdcbf&&println("Maneuver selections: $counts")
    (;controller,label,status,states,positions,controls=U,min_h_history,maneuver_history,
      score_history,elapsed,path_length,physical_clearance=clearance,
      fallback_steps,maneuver_counts=counts)
end

run_simulation()=run_controller(:mdcbf)
run_comparison()=[run_controller(controller) for controller in CONTROLLERS]

function plot_result(result;output="stationary_obstacle_cbf.png")
    ts=(0:size(result.positions,1)-1).*DT
    tu=(0:size(result.controls,1)-1).*DT
    fig=Figure(size=(1300,1000))
    ax=Axis(fig[1:2,1:2],aspect=DataAspect(),
            title="Five stationary obstacles — sine/slalom HOCBF path",
            xlabel="x [m]",ylabel="y [m]")
    for (i,c) in enumerate(OBSTACLES)
        poly!(ax,Circle(Point2f(c...),OBSTACLE_RADII[i]),color=(:gray,0.65),
              strokecolor=:black,strokewidth=2)
        lines!(ax,Circle(Point2f(c...),safe_distance(i)),color=:gray35,linestyle=:dot)
        text!(ax,c[1],c[2];text="$i",align=(:center,:center))
    end
    lines!(ax,result.positions[:,1],result.positions[:,2],color=:dodgerblue,
           linewidth=3,label="Pursuer/agent")
    scatter!(ax,[Point2f(START...)],marker=:diamond,markersize=18,color=:green,label="Start")
    scatter!(ax,[Point2f(GOAL...)],marker=:star5,markersize=22,color=:red,label="Goal")
    axislegend(ax,position=:rt)
    axs=Axis(fig[1,3],title="Forward speed",xlabel="time [s]",ylabel="m/s")
    lines!(axs,ts,[s[4] for s in result.states],color=:blue); hlines!(axs,[V_MAX],color=:red,linestyle=:dash)
    axu=Axis(fig[2,3],title="Bicycle controls",xlabel="time [s]")
    steering_deg=rad2deg.(steering_angle.(result.controls[:,2]))
    lines!(axu,tu,result.controls[:,1],label="a [m/s²]",color=:orange)
    lines!(axu,tu,steering_deg,label="delta [deg]",color=:purple); axislegend(axu)
    axm=Axis(fig[3,1:3],title="Multi-directional decision selected before CBF application",
             xlabel="time [s]",ylabel="maneuver",yticks=([-1,0,1],["right","straight","left"]))
    mode_value=Dict(:right=>-1.0,:straight=>0.0,:left=>1.0)
    stairs!(axm,tu,[mode_value[m] for m in result.maneuver_history],color=:purple,linewidth=2)
    ylims!(axm,-1.35,1.35)
    axh=Axis(fig[4,1:3],title="Minimum barrier across all five obstacles",xlabel="time [s]",ylabel="min h")
    lines!(axh,tu,result.min_h_history,color=:darkgreen,linewidth=2)
    hlines!(axh,[0.0],color=:red,linestyle=:dash,label="safety boundary"); axislegend(axh)
    save(output,fig); println("Saved $output")
end

function animate_result(result;output="stationary_obstacle_cbf.mp4")
    nstates=size(result.positions,1)
    ncontrols=size(result.controls,1)
    ts=(0:nstates-1).*DT
    tu=(0:ncontrols-1).*DT
    speed=[s[4] for s in result.states]
    vx=[s[4]*cos(s[3]) for s in result.states]
    vy=[s[4]*sin(s[3]) for s in result.states]
    along=result.controls[:,1]
    curvature=result.controls[:,2]
    steering=steering_angle.(curvature)
    lateral=[result.states[k][4]^2*curvature[k] for k in 1:ncontrols]
    mode_value=Dict(:right=>-1.0,:straight=>0.0,:left=>1.0)
    mode_series=[mode_value[m] for m in result.maneuver_history]

    fig=Figure(size=(1500,1050))
    ax=Axis(fig[1:3,1:2],aspect=DataAspect(),
            title="Live bicycle-model multi-directional CBF",
            xlabel="x [m]",ylabel="y [m]")
    for (i,c) in enumerate(OBSTACLES)
        poly!(ax,Circle(Point2f(c...),OBSTACLE_RADII[i]),color=(:gray,0.65),
              strokecolor=:black,strokewidth=2)
        lines!(ax,Circle(Point2f(c...),safe_distance(i)),color=:gray35,
               linestyle=:dot,linewidth=1.3)
        text!(ax,c[1],c[2];text="$i",align=(:center,:center))
    end
    scatter!(ax,[Point2f(START...)],marker=:diamond,markersize=17,color=:green)
    scatter!(ax,[Point2f(GOAL...)],marker=:star5,markersize=22,color=:red)
    xlims!(ax,START[1]-0.4,GOAL[1]+0.4); ylims!(ax,-1.5,1.5)

    frame=Observable(1)
    trail=@lift(Point2f.(eachrow(result.positions[1:$frame,:])))
    pos=@lift(Point2f(result.positions[$frame,1],result.positions[$frame,2]))
    heading_line=@lift(begin
        k=$frame; p=Point2f(result.positions[k,1],result.positions[k,2])
        psi=result.states[k][3]
        [p,Point2f(p[1]+0.42cos(psi),p[2]+0.42sin(psi))]
    end)
    lines!(ax,trail,color=:dodgerblue,linewidth=3)
    lines!(ax,heading_line,color=:navy,linewidth=3)
    scatter!(ax,pos,color=:dodgerblue,markersize=2AGENT_RADIUS,markerspace=:data)

    hud=@lift(begin
        k=$frame; j=min(k,ncontrols); s=result.states[k]
        mode=j<=length(result.maneuver_history) ? result.maneuver_history[j] : :done
        a=j<=ncontrols ? along[j] : 0.0
        kap=j<=ncontrols ? curvature[j] : 0.0
        delta=j<=ncontrols ? rad2deg(steering[j]) : 0.0
        alat=j<=ncontrols ? lateral[j] : 0.0
        h=j<=length(result.min_h_history) ? result.min_h_history[j] : result.min_h_history[end]
        @sprintf("t = %5.2f s\nmode = %-8s\npos = (%5.2f, %5.2f) m\nv = %5.2f m/s\nvx = %5.2f, vy = %5.2f m/s\na_long = %6.2f m/s²\na_lat  = %6.2f m/s²\nkappa = %6.3f 1/m\ndelta = %6.2f deg\nmin h = %6.3f",
                 ts[k],string(mode),s[1],s[2],s[4],vx[k],vy[k],a,alat,kap,delta,h)
    end)
    text!(ax,0.02,0.97;text=hud,space=:relative,align=(:left,:top),
          fontsize=15,color=:black)

    axv=Axis(fig[1,3],title="Live velocities",xlabel="time [s]",ylabel="m/s")
    lines!(axv,@lift(ts[1:$frame]),@lift(speed[1:$frame]),label="v",color=:blue,linewidth=2)
    lines!(axv,@lift(ts[1:$frame]),@lift(vx[1:$frame]),label="vx",color=:teal)
    lines!(axv,@lift(ts[1:$frame]),@lift(vy[1:$frame]),label="vy",color=:magenta)
    xlims!(axv,0,result.elapsed); ylims!(axv,-V_MAX-0.2,V_MAX+0.2); axislegend(axv,position=:lb)

    axa=Axis(fig[2,3],title="Live accelerations",xlabel="time [s]",ylabel="m/s²")
    live_tu=@lift(tu[1:min($frame,ncontrols)])
    lines!(axa,live_tu,@lift(along[1:min($frame,ncontrols)]),label="longitudinal",color=:orange,linewidth=2)
    lines!(axa,live_tu,@lift(lateral[1:min($frame,ncontrols)]),label="lateral",color=:red)
    xlims!(axa,0,result.elapsed); ylims!(axa,-A_LAT_MAX-0.5,A_LAT_MAX+0.5); axislegend(axa,position=:lb)

    axu=Axis(fig[3,3],title="Live normalized controls",xlabel="time [s]",ylabel="command")
    lines!(axu,live_tu,@lift(along[1:min($frame,ncontrols)]./A_MAX),
           label="a / amax",color=:orange,linewidth=2)
    lines!(axu,live_tu,@lift(steering[1:min($frame,ncontrols)]./STEER_MAX),
           label="delta / delta_max",color=:purple,linewidth=2)
    xlims!(axu,0,result.elapsed); ylims!(axu,-1.1,1.1); axislegend(axu,position=:lb)

    axm=Axis(fig[4,1:3],title="Selected multi-directional maneuver",
             xlabel="time [s]",ylabel="maneuver",
             yticks=([-1,0,1],["right","straight","left"]))
    stairs!(axm,live_tu,@lift(mode_series[1:min($frame,ncontrols)]),color=:purple,linewidth=2)
    xlims!(axm,0,result.elapsed); ylims!(axm,-1.3,1.3)

    axh=Axis(fig[5,1:3],title="Live minimum CBF value",xlabel="time [s]",ylabel="min h")
    lines!(axh,live_tu,@lift(result.min_h_history[1:min($frame,ncontrols)]),
           color=:darkgreen,linewidth=2)
    hlines!(axh,[0.0],color=:red,linestyle=:dash)
    xlims!(axh,0,result.elapsed); ylims!(axh,-0.1,max(2.1,maximum(result.min_h_history)*1.05))

    record(fig,output,1:nstates;framerate=round(Int,1/DT)) do k
        frame[]=k
    end
    println("Saved $output")
end

function telemetry(result)
    nstates=size(result.positions,1);ncontrols=size(result.controls,1)
    speed=[s[4] for s in result.states]
    vx=[s[4]*cos(s[3]) for s in result.states]
    vy=[s[4]*sin(s[3]) for s in result.states]
    acceleration=result.controls[:,1]
    curvature=result.controls[:,2]
    steering=steering_angle.(curvature)
    lateral=[result.states[k][4]^2*curvature[k] for k in 1:ncontrols]
    (;nstates,ncontrols,ts=(0:nstates-1).*DT,tu=(0:ncontrols-1).*DT,
      speed,vx,vy,acceleration,curvature,steering,lateral)
end

function draw_obstacles!(axis)
    for (i,center) in enumerate(OBSTACLES)
        poly!(axis,Circle(Point2f(center...),OBSTACLE_RADII[i]),
              color=(:gray,0.65),strokecolor=:black,strokewidth=2)
        lines!(axis,Circle(Point2f(center...),safe_distance(i)),
               color=:gray35,linestyle=:dot,linewidth=1.2)
        text!(axis,center[1],center[2];text="$i",align=(:center,:center))
    end
end

"""Static summary of all three controllers under identical conditions."""
function plot_comparison(results;output="three_controller_comparison.png")
    fig=Figure(size=(1550,1150))
    path_axis=Axis(fig[1:2,1:3],aspect=DataAspect(),
                   title="MDCBF vs ACBF vs VCBF — bicycle-model slalom",
                   xlabel="x [m]",ylabel="y [m]")
    draw_obstacles!(path_axis)
    scatter!(path_axis,[Point2f(START...)],marker=:diamond,markersize=17,
             color=:green,label="Start")
    scatter!(path_axis,[Point2f(GOAL...)],marker=:star5,markersize=22,
             color=:red,label="Goal")
    for result in results
        lines!(path_axis,result.positions[:,1],result.positions[:,2],
               color=CONTROLLER_COLOR[result.controller],linewidth=3,
               label="$(result.label): $(result.status), $(round(result.elapsed,digits=2)) s")
    end
    axislegend(path_axis,position=:rt)
    for (column,result) in enumerate(results)
        tel=telemetry(result)
        axv=Axis(fig[3,column],title="$(result.label) velocity",xlabel="time [s]",ylabel="m/s")
        lines!(axv,tel.ts,tel.speed,label="v",color=:blue)
        lines!(axv,tel.ts,tel.vx,label="vx",color=:teal)
        lines!(axv,tel.ts,tel.vy,label="vy",color=:magenta);axislegend(axv,position=:lb)
        axa=Axis(fig[4,column],title="$(result.label) acceleration",xlabel="time [s]",ylabel="m/s²")
        lines!(axa,tel.tu,tel.acceleration,label="longitudinal",color=:orange)
        lines!(axa,tel.tu,tel.lateral,label="lateral",color=:red);axislegend(axa,position=:lb)
        axh=Axis(fig[5,column],title="$(result.label) safety",xlabel="time [s]",ylabel="min h")
        lines!(axh,tel.tu,result.min_h_history,color=:darkgreen,linewidth=2)
        hlines!(axh,[0.0],color=:red,linestyle=:dash)
        axu=Axis(fig[6,column],title="$(result.label) controls",xlabel="time [s]",ylabel="normalized")
        lines!(axu,tel.tu,tel.acceleration./A_MAX,label="a/amax",color=:orange)
        lines!(axu,tel.tu,tel.steering./STEER_MAX,label="delta/delta_max",color=:purple)
        ylims!(axu,-1.1,1.1);axislegend(axu,position=:lb)
    end
    save(output,fig);println("Saved $output")
end

"""Synchronized live comparison video for all controller states and metrics."""
function animate_comparison(results;output="three_controller_comparison.mp4")
    data=telemetry.(results)
    max_states=maximum(d.nstates for d in data)
    max_time=(max_states-1)*DT
    frame=Observable(1)
    cursor_time=@lift(min(($frame-1)*DT,max_time))
    fig=Figure(size=(1550,1200))
    path_axis=Axis(fig[1:2,1:3],aspect=DataAspect(),
                   title="Live three-controller bicycle-model comparison",
                   xlabel="x [m]",ylabel="y [m]")
    draw_obstacles!(path_axis)
    scatter!(path_axis,[Point2f(START...)],marker=:diamond,markersize=17,color=:green)
    scatter!(path_axis,[Point2f(GOAL...)],marker=:star5,markersize=22,color=:red)
    xlims!(path_axis,START[1]-0.4,GOAL[1]+0.5);ylims!(path_axis,-1.5,1.5)

    for (column,result) in enumerate(results)
        tel=data[column];color=CONTROLLER_COLOR[result.controller]
        live_state_index=@lift(min($frame,tel.nstates))
        live_control_index=@lift(min($frame,tel.ncontrols))
        live_ts=@lift(tel.ts[1:min($frame,tel.nstates)])
        live_tu=@lift(tel.tu[1:min($frame,tel.ncontrols)])
        trail=@lift(Point2f.(eachrow(result.positions[1:min($frame,tel.nstates),:])))
        position=@lift(begin
            k=min($frame,tel.nstates)
            Point2f(result.positions[k,1],result.positions[k,2])
        end)
        lines!(path_axis,trail,color=color,linewidth=3,label=result.label)
        scatter!(path_axis,position,color=color,markersize=2AGENT_RADIUS,markerspace=:data)

        live_title=@lift(begin
            ks=min($frame,tel.nstates);kc=min($frame,tel.ncontrols)
            a=tel.acceleration[kc];delta=rad2deg(tel.steering[kc])
            h=result.min_h_history[kc]
            @sprintf("%s | v %.2f | a %.2f | delta %.1f° | h %.2f",
                     result.label,tel.speed[ks],a,delta,h)
        end)
        axv=Axis(fig[3,column],title=live_title,xlabel="time [s]",ylabel="velocity [m/s]")
        lines!(axv,live_ts,@lift(tel.speed[1:min($frame,tel.nstates)]),label="v",color=:blue)
        lines!(axv,live_ts,@lift(tel.vx[1:min($frame,tel.nstates)]),label="vx",color=:teal)
        lines!(axv,live_ts,@lift(tel.vy[1:min($frame,tel.nstates)]),label="vy",color=:magenta)
        xlims!(axv,0,max_time);ylims!(axv,-V_MAX-0.2,V_MAX+0.2);axislegend(axv,position=:lb)
        vlines!(axv,cursor_time,color=:black,linestyle=:dash,linewidth=1.5)

        axa=Axis(fig[4,column],title="Accelerations",xlabel="time [s]",ylabel="m/s²")
        lines!(axa,live_tu,@lift(tel.acceleration[1:min($frame,tel.ncontrols)]),
               label="longitudinal",color=:orange)
        lines!(axa,live_tu,@lift(tel.lateral[1:min($frame,tel.ncontrols)]),
               label="lateral",color=:red)
        xlims!(axa,0,max_time);ylims!(axa,-A_LAT_MAX-0.5,A_LAT_MAX+0.5);axislegend(axa,position=:lb)
        vlines!(axa,cursor_time,color=:black,linestyle=:dash,linewidth=1.5)

        axh=Axis(fig[5,column],title="Safety",xlabel="time [s]",ylabel="min h")
        lines!(axh,live_tu,@lift(result.min_h_history[1:min($frame,tel.ncontrols)]),
               color=:darkgreen,linewidth=2)
        hlines!(axh,[0.0],color=:red,linestyle=:dash)
        xlims!(axh,0,max_time);ylims!(axh,-0.1,2.2)
        vlines!(axh,cursor_time,color=:black,linestyle=:dash,linewidth=1.5)

        axu=Axis(fig[6,column],title="Controls",xlabel="time [s]",ylabel="normalized")
        lines!(axu,live_tu,@lift(tel.acceleration[1:min($frame,tel.ncontrols)]./A_MAX),
               label="a/amax",color=:orange)
        lines!(axu,live_tu,@lift(tel.steering[1:min($frame,tel.ncontrols)]./STEER_MAX),
               label="delta/delta_max",color=:purple)
        xlims!(axu,0,max_time);ylims!(axu,-1.1,1.1);axislegend(axu,position=:lb)
        vlines!(axu,cursor_time,color=:black,linestyle=:dash,linewidth=1.5)
    end
    axislegend(path_axis,position=:rt)
    clock=@lift(@sprintf("comparison time: %.2f s",$cursor_time))
    Label(fig[0,1:3],clock,fontsize=24,tellwidth=false)
    record(fig,output,1:max_states;framerate=round(Int,1/DT)) do k
        frame[]=k
    end
    println("Saved $output")
end

function activate_interactive_backend!()
    if !isdefined(Main,:GLMakie)
        Base.eval(Main,:(import GLMakie))
    end
    Base.invokelatest(getfield(Main,:GLMakie).activate!)
end

function separated_circle_points(centers,radii;extra=0.0)
    points=Point2f[]
    angles=range(0,2pi,length=100)
    for (center,radius) in zip(centers,radii)
        append!(points,[Point2f(center[1]+(radius+extra)*cos(angle),
                                center[2]+(radius+extra)*sin(angle)) for angle in angles])
        push!(points,Point2f(NaN,NaN))
    end
    points
end

"""
Open a GLMakie editor and synchronized comparison dashboard.

Interaction:
  * drag the start, goal, any current agent, or any obstacle center;
  * click an obstacle and use the radius slider;
  * add an obstacle at the last mouse position or delete the selected one;
  * rerun all controllers after geometry changes;
  * scrub or play one shared timeline. Every black vertical cursor represents
    exactly the same timestamp as all three agent markers.
"""
function interactive_gui()
    activate_interactive_backend!()
    results_ref=Ref(run_comparison())
    telemetry_ref=Ref(telemetry.(results_ref[]))
    current_time=Observable(0.0)
    playing=Observable(false)
    selected_obstacle=Observable(0)

    center_points=Observable(Point2f.(OBSTACLES))
    radius_values=Observable(copy(OBSTACLE_RADII))
    start_point=Observable(Point2f(START...))
    goal_point=Observable(Point2f(GOAL...))
    obstacle_shapes=@lift([Circle(center,radius) for (center,radius) in zip($center_points,$radius_values)])
    safety_points=@lift(separated_circle_points($center_points,$radius_values;
                                                extra=AGENT_RADIUS+SAFETY_MARGIN))
    selected_point=@lift(begin
        index=$selected_obstacle
        index in eachindex($center_points) ? [$center_points[index]] : Point2f[]
    end)

    fig=Figure(size=(1750,1250))
    path_axis=Axis(fig[1:2,1:3],aspect=DataAspect(),
                   title="Interactive MDCBF / ACBF / VCBF comparison",
                   xlabel="x [m]",ylabel="y [m]")
    poly!(path_axis,obstacle_shapes,color=(:gray,0.65),strokecolor=:black,strokewidth=2)
    lines!(path_axis,safety_points,color=:gray35,linestyle=:dot,linewidth=1.2)
    scatter!(path_axis,center_points,color=:gray20,markersize=5)
    scatter!(path_axis,selected_point,color=:yellow,strokecolor=:black,
             strokewidth=2,markersize=20)
    scatter!(path_axis,start_point,marker=:diamond,markersize=18,color=:green,label="Start")
    scatter!(path_axis,goal_point,marker=:star5,markersize=23,color=:red,label="Goal")
    xlims!(path_axis,-7,7);ylims!(path_axis,-3,3)

    full_paths=Observable[];trails=Observable[];agent_points=Observable[]
    velocity_time=Observable[];speed_data=Observable[];vx_data=Observable[];vy_data=Observable[]
    control_time=Observable[];long_data=Observable[];lat_data=Observable[]
    safety_data=Observable[];accel_control_data=Observable[];steer_control_data=Observable[]
    telemetry_axes=Axis[]

    for (column,result) in enumerate(results_ref[])
        tel=telemetry_ref[][column];color=CONTROLLER_COLOR[result.controller]
        full=Observable(Point2f.(eachrow(result.positions)))
        trail=Observable([Point2f(result.positions[1,:]...)])
        agent=Observable(Point2f(result.positions[1,:]...))
        push!(full_paths,full);push!(trails,trail);push!(agent_points,agent)
        lines!(path_axis,full,color=(color,0.25),linewidth=2)
        lines!(path_axis,trail,color=color,linewidth=3,label=result.label)
        scatter!(path_axis,agent,color=color,markersize=2AGENT_RADIUS,markerspace=:data)

        vt=Observable(copy(tel.ts));sp=Observable(copy(tel.speed));vxo=Observable(copy(tel.vx));vyo=Observable(copy(tel.vy))
        ct=Observable(copy(tel.tu));alo=Observable(copy(tel.acceleration));lao=Observable(copy(tel.lateral))
        sho=Observable(copy(result.min_h_history));aco=Observable(tel.acceleration./A_MAX)
        sco=Observable(tel.steering./STEER_MAX)
        append!(velocity_time,[vt]);append!(speed_data,[sp]);append!(vx_data,[vxo]);append!(vy_data,[vyo])
        append!(control_time,[ct]);append!(long_data,[alo]);append!(lat_data,[lao])
        append!(safety_data,[sho]);append!(accel_control_data,[aco]);append!(steer_control_data,[sco])

        axv=Axis(fig[3,column],title="$(result.label) velocity",xlabel="time [s]",ylabel="m/s")
        lines!(axv,vt,sp,label="v",color=:blue);lines!(axv,vt,vxo,label="vx",color=:teal)
        lines!(axv,vt,vyo,label="vy",color=:magenta);axislegend(axv,position=:lb)
        axa=Axis(fig[4,column],title="Acceleration",xlabel="time [s]",ylabel="m/s²")
        lines!(axa,ct,alo,label="longitudinal",color=:orange)
        lines!(axa,ct,lao,label="lateral",color=:red);axislegend(axa,position=:lb)
        axh=Axis(fig[5,column],title="Safety",xlabel="time [s]",ylabel="min h")
        lines!(axh,ct,sho,color=:darkgreen,linewidth=2);hlines!(axh,[0.0],color=:red,linestyle=:dash)
        axu=Axis(fig[6,column],title="Controls",xlabel="time [s]",ylabel="normalized")
        lines!(axu,ct,aco,label="a/amax",color=:orange)
        lines!(axu,ct,sco,label="delta/delta_max",color=:purple);axislegend(axu,position=:lb)
        ylims!(axu,-1.1,1.1)
        append!(telemetry_axes,[axv,axa,axh,axu])
    end
    axislegend(path_axis,position=:rt)
    initial_end=maximum(result.elapsed for result in results_ref[])
    for axis in telemetry_axes
        # Identical x limits make every cursor a true vertical time slice through
        # velocity, acceleration, safety, controls, and the trajectory animation.
        xlims!(axis,0,initial_end)
        vlines!(axis,current_time,color=:black,linestyle=:dash,linewidth=2)
    end

    play_button=Button(fig[7,1],label="Play")
    rerun_button=Button(fig[7,2],label="Rerun controllers")
    export_button=Button(fig[7,3],label="Export synchronized MP4")
    add_button=Button(fig[8,1],label="Add obstacle")
    delete_button=Button(fig[8,2],label="Delete selected")
    radius_slider=Slider(fig[8,3],range=0.15:0.01:1.25,
                         startvalue=OBSTACLE_RADII[1])
    Label(fig[9,1],"Shared time")
    time_slider=Slider(fig[9,2:3],range=0.0:DT:T_MAX,startvalue=0.0)
    status_text=Observable("Drag start/goal/agents/obstacles, edit radius, then rerun.")
    Label(fig[10,1:3],status_text,tellwidth=false)

    function comparison_end_time()
        maximum(result.elapsed for result in results_ref[])
    end
    function update_time!(requested)
        time=clamp(Float64(requested),0.0,comparison_end_time())
        current_time[]=time
        for i in eachindex(results_ref[])
            result=results_ref[][i]
            state_index=clamp(floor(Int,time/DT)+1,1,size(result.positions,1))
            trails[i][]=Point2f.(eachrow(result.positions[1:state_index,:]))
            agent_points[i][]=Point2f(result.positions[state_index,:]...)
        end
    end
    on(time_slider.value) do value
        update_time!(value)
    end

    # The cursor itself is draggable from any telemetry panel. All panels use
    # the same Observable, so one horizontal mouse movement updates every
    # vertical line and all three vehicle positions together.
    timeline_dragging=Ref(false)
    function scrub_from_mouse!()
        for axis in telemetry_axes
            if is_mouseinside(axis.scene)
                set_close_to!(time_slider,clamp(mouseposition(axis)[1],
                                                0.0,comparison_end_time()))
                return true
            end
        end
        false
    end
    on(events(fig).mousebutton) do event
        if event.button==Mouse.left&&event.action==Mouse.press
            timeline_dragging[]=scrub_from_mouse!()
        elseif event.button==Mouse.left&&event.action==Mouse.release
            timeline_dragging[]=false
        end
    end
    on(events(fig).mouseposition) do _
        timeline_dragging[]&&scrub_from_mouse!()
    end

    on(play_button.clicks) do _
        playing[]=!playing[]
        play_button.label[]=playing[] ? "Pause" : "Play"
        if playing[]
            @async begin
                while playing[]&&current_time[]<comparison_end_time()-DT/2
                    set_close_to!(time_slider,current_time[]+DT)
                    sleep(DT)
                end
                playing[]=false;play_button.label[]="Play"
            end
        end
    end

    last_mouse=Ref(Point2f(0,0));drag_target=Ref((:none,0))
    function nearest_target(point)
        norm(point-start_point[])<0.35&&return (:start,0)
        norm(point-goal_point[])<0.35&&return (:goal,0)
        for (i,agent) in enumerate(agent_points)
            norm(point-agent[])<0.30&&return (:start,i) # common start preserves fairness
        end
        for (i,center) in enumerate(center_points[])
            norm(point-center)<radius_values[][i]+0.18&&return (:obstacle,i)
        end
        (:none,0)
    end
    on(events(fig).mousebutton) do event
        if event.button==Mouse.left&&event.action==Mouse.press&&is_mouseinside(path_axis.scene)
            point=Point2f(mouseposition(path_axis)...);drag_target[]=nearest_target(point)
            drag_target[][1]==:obstacle&&(selected_obstacle[]=drag_target[][2];
                                          set_close_to!(radius_slider,OBSTACLE_RADII[drag_target[][2]]))
        elseif event.button==Mouse.left&&event.action==Mouse.release
            drag_target[]=(:none,0)
        end
    end
    on(events(fig).mouseposition) do _
        is_mouseinside(path_axis.scene)||(return)
        point=Point2f(mouseposition(path_axis)...);last_mouse[]=point
        kind,index=drag_target[]
        if kind==:start
            START.=point;start_point[]=point
        elseif kind==:goal
            GOAL.=point;goal_point[]=point
        elseif kind==:obstacle&&index in eachindex(OBSTACLES)
            OBSTACLES[index].=point;center_points[]=Point2f.(OBSTACLES)
        end
    end
    on(radius_slider.value) do value
        index=selected_obstacle[]
        if index in eachindex(OBSTACLE_RADII)
            OBSTACLE_RADII[index]=Float64(value);radius_values[]=copy(OBSTACLE_RADII)
        end
    end
    on(add_button.clicks) do _
        point=last_mouse[]
        push!(OBSTACLES,[Float64(point[1]),Float64(point[2])]);push!(OBSTACLE_RADII,0.50)
        center_points[]=Point2f.(OBSTACLES);radius_values[]=copy(OBSTACLE_RADII)
        selected_obstacle[]=length(OBSTACLES);set_close_to!(radius_slider,0.50)
        status_text[]="Obstacle added. Drag/resize it, then rerun."
    end
    on(delete_button.clicks) do _
        index=selected_obstacle[]
        if index in eachindex(OBSTACLES)&&length(OBSTACLES)>1
            deleteat!(OBSTACLES,index);deleteat!(OBSTACLE_RADII,index)
            selected_obstacle[]=0;center_points[]=Point2f.(OBSTACLES)
            radius_values[]=copy(OBSTACLE_RADII)
            status_text[]="Obstacle deleted. Rerun to update trajectories."
        end
    end

    function install_results!(new_results)
        results_ref[]=new_results;telemetry_ref[]=telemetry.(new_results)
        for i in eachindex(new_results)
            result=new_results[i];tel=telemetry_ref[][i]
            full_paths[i][]=Point2f.(eachrow(result.positions))
            velocity_time[i][]=copy(tel.ts);speed_data[i][]=copy(tel.speed)
            vx_data[i][]=copy(tel.vx);vy_data[i][]=copy(tel.vy)
            control_time[i][]=copy(tel.tu);long_data[i][]=copy(tel.acceleration)
            lat_data[i][]=copy(tel.lateral);safety_data[i][]=copy(result.min_h_history)
            accel_control_data[i][]=tel.acceleration./A_MAX
            steer_control_data[i][]=tel.steering./STEER_MAX
        end
        for axis in telemetry_axes;xlims!(axis,0,comparison_end_time());end
        set_close_to!(time_slider,0.0);update_time!(0.0)
    end
    on(rerun_button.clicks) do _
        playing[]=false;play_button.label[]="Play";status_text[]="Computing three controllers..."
        rerun_button.label[]="Computing..."
        new_results=run_comparison();install_results!(new_results)
        rerun_button.label[]="Rerun controllers"
        status_text[]="Updated. All plots and agents share the same time cursor."
    end
    on(export_button.clicks) do _
        playing[]=false;status_text[]="Rendering synchronized MP4..."
        export_button.label[]="Rendering..."
        @async begin
            CairoMakie.activate!();animate_comparison(results_ref[];
                output="interactive_three_controller_comparison.mp4")
            activate_interactive_backend!();export_button.label[]="Export synchronized MP4"
            status_text[]="Saved interactive_three_controller_comparison.mp4"
        end
    end

    update_time!(0.0)
    screen=Base.invokelatest(display,fig)
    # Keep a script-launched GL window alive until the user closes it. In a
    # REPL/notebook, return immediately so the session remains usable.
    if !isinteractive()&&applicable(wait,screen)
        Base.invokelatest(wait,screen)
    end
    fig
end

function main(args=ARGS)
    allowed=("--video","--static","-h","--help")
    any(a->!(a in allowed),args) &&
        error("Usage: julia new_interactive.jl [--video | --static]")
    if "-h" in args || "--help" in args
        println("""Usage: julia new_interactive.jl [--video | --static]

No option opens the interactive GLMakie scene editor and synchronized dashboard.
--video writes interactive_three_controller_comparison.png and .mp4.
--static writes only interactive_three_controller_comparison.png.""")
        return
    end
    if "--video" in args || "--static" in args
        results=run_comparison()
        plot_comparison(results;output="interactive_three_controller_comparison.png")
        "--video" in args && animate_comparison(results;
            output="interactive_three_controller_comparison.mp4")
    else
        interactive_gui()
    end
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()
