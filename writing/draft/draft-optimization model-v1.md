## Urban Functional Recovery Optimization

We formulate a data-calibrated urban functional recovery model to quantify how much post-disaster urban functional loss can be reduced through targeted management interventions. The model is designed to represent recovery at the level of urban functions rather than at the level of individual infrastructure components. It does not schedule road repairs, route repair crews, or solve a traffic assignment problem. Instead, it allocates high-level recovery capacities across urban spatial units to reduce the functional loss experienced by residents.

Let the city be partitioned into a set of spatial units \(V=\{1,\ldots,N\}\) , where \(i,j\in V\) denote origin and destination units. The recovery horizon is discretized into time steps \(t=0,1,\ldots,T\)  with duration \(\Delta t\). 

The model represents urban recovery through three functional states. The first state,

\[
b_{i,t}\in[0,1],
\]

denotes the underlying functional deficit of spatial unit \(i\) at time \(t\). A value of \(b_{i,t}=0\) indicates normal functionality, while \(b_{i,t}=1\) indicates complete loss of the relevant urban function. The second state,

\[
d_{i,t}\in[0,1],
\]

denotes the local experienced deficit after temporary services or emergency capacity have partially mitigated the underlying deficit. The third state,

\[
\ell_{i,t}\in[0,1],
\]

denotes the access-weighted functional loss experienced by residents whose origin is unit \(i\). This final state is the main outcome of the model because residents do not experience urban recovery only through the condition of their home location. They also experience it through their ability to access work, food, health care, transit, public services, and other essential destinations.

The model therefore distinguishes between three layers of recovery:

\[
b_{i,t}\rightarrow d_{i,t}\rightarrow \ell_{i,t}.
\]

The first layer describes whether the city’s underlying functions have recovered. The second layer describes whether temporary capacity has reduced the locally experienced deficit. The third layer describes whether residents can access the urban functions on which they depend.

## Functional Dependence Matrix

Urban functional dependence is represented by a time-varying matrix

\[
Q_t\in \mathbb{R}^{N\times N},
\]

where

\[
Q_{ij,t}\geq 0
\]

denotes the dependence of residents from origin \(i\) on destination \(j\) at time \(t\). Each row of \(Q_t\) is normalized:

\[
\sum_{j\in V}Q_{ij,t}=1,\qquad \forall i,t.
\]

The matrix \(Q_t\) can be calibrated from pre-disaster origin-destination flows, post-disaster travel times, service accessibility, and the functional importance of destinations. One possible empirical specification is

\[
Q_{ij,t}
=
\frac{
OD^0_{ij}\exp(-\kappa \tau_{ij,t})S_j
}{
\sum_{j'\in V}OD^0_{ij'}\exp(-\kappa \tau_{ij',t})S_{j'}
},
\]

where \(OD^0_{ij}\) is the baseline dependence or mobility flow from \(i\) to \(j\), \(\tau_{ij,t}\) is the post-disaster travel time or generalized access cost, \(S_j\) measures the service or functional importance of destination \(j\), and \(\kappa\) is a travel-time decay parameter.

Given local experienced deficits \(d_t\), the access-weighted functional loss before substitution or demand-control interventions is

\[
Q_td_t.
\]

Thus, damage in destination \(j\) affects not only residents in \(j\), but also residents in all origins that depend on \(j\). This structure allows the model to use transportation and mobility data to represent urban functional dependence without converting the recovery problem into a traffic assignment or road-repair scheduling model.

## Intervention Primitives

The model includes three high-level intervention primitives:

\[
k\in K=\{R,C,S\}.
\]

The first primitive, \(R\), represents durable restoration. These interventions reduce the underlying functional deficit \(b_{i,t}\). Examples include restoring power, reopening critical facilities, clearing essential corridors, repairing water or communication services, and restoring major service or transit nodes.

The second primitive, \(C\), represents temporary capacity. These interventions do not necessarily repair the underlying damaged system, but they reduce the local experienced deficit \(d_{i,t}\). Examples include mobile clinics, temporary food or water distribution, emergency charging stations, temporary shelters, temporary shuttle services, mobile pumps, and temporary communication facilities.

The third primitive, \(S\), represents substitution or control. These interventions reduce the access-weighted loss \(\ell_{i,t}\) experienced by residents at the origin level. Examples include route guidance, public information, demand redirection, service substitution, remote-work coordination, school or workplace closure coordination, and temporary relocation of services.

For each spatial unit \(i\), time \(t\), and intervention type \(k\), let

\[
u^k_{i,t}\geq 0
\]

denote the amount of intervention capacity allocated to unit \(i\), and let

\[
e^k_{i,t}\geq 0
\]

denote the effective recovery output generated by that allocation. In the continuous formulation, resource allocation is divisible and controlled entirely by \(u^k_{i,t}\). The effective output is linked to resource input through

\[
e^k_{i,t}\leq \phi^k_{i,t}(u^k_{i,t}).
\]

For a linear recovery specification,

\[
\phi^k_{i,t}(u)=\eta^k_{i,t}u,
\]

where \(\eta^k_{i,t}\geq0\) is the unit effectiveness of intervention \(k\) in spatial unit \(i\) at time \(t\). This linear form yields a linear programming formulation. Diminishing returns can be represented by a concave piecewise-linear function while preserving linear-programming structure.

## Continuous Linear Recovery Model

The core recovery model minimizes the population- or activity-weighted cumulative urban functional loss over the recovery horizon:

\[
\begin{aligned}
\min_{u,e,b,r^C,d,r^S,\ell} \quad 
& \sum_{t=0}^{T}\Delta t \sum_{i\in V}p_i\ell_{i,t} \\[2mm]
\text{s.t.}\quad 
& b_{t+1}=A_tb_t+h_{t+1}-M^Re^R_t, 
&& t=0,\ldots,T-1, \qquad \text{(1)} \\
& r^C_{t+1}=(1-\delta_C)r^C_t+M^Ce^C_t, 
&& t=0,\ldots,T-1, \qquad \text{(2)} \\
& d_t\geq b_t-r^C_t, 
&& t=0,\ldots,T, \qquad \text{(3)} \\
& r^S_{t+1}=(1-\delta_S)r^S_t+M^Se^S_t, 
&& t=0,\ldots,T-1, \qquad \text{(4)} \\
& \ell_t\geq Q_td_t-r^S_t, 
&& t=0,\ldots,T, \qquad \text{(5)} \\
& e^k_{i,t}\leq \eta^k_{i,t}u^k_{i,t}, 
&& i\in V,\ t=0,\ldots,T,\ k\in K, \qquad \text{(6)} \\
& \sum_{i\in V}\sum_{k\in K}c^m_{i,k,t}u^k_{i,t} \leq B^m_t, 
&& m\in\mathcal M,\ t=0,\ldots,T, \qquad \text{(7)} \\
& \sum_{t=0}^{T}\sum_{i\in V}\sum_{k\in K} c^m_{i,k,t}u^k_{i,t} \leq \bar B^m, 
&& m\in\mathcal M, \qquad \text{(8)} \\
& u^k_{i,t}=0, 
&& i\in V,\ k\in K,\ t<\Delta_k, \qquad \text{(9)} \\
& b_0=\hat b_0,\qquad r^C_0=0,\qquad r^S_0=0, 
&& \qquad \text{(10)} \\
& 0\leq b_{i,t},d_{i,t},\ell_{i,t},r^C_{i,t},r^S_{i,t}\leq1, 
&& i\in V,\ t=0,\ldots,T, \qquad \text{(11)} \\
& u^k_{i,t}\geq0,\qquad e^k_{i,t}\geq0, 
&& i\in V,\ t=0,\ldots,T,\ k\in K. \qquad \text{(12)}
\end{aligned}
\]
Here \(p_i\) is the population, activity, or exposure weight of spatial unit \(i\). The objective function aggregates the access-weighted functional loss experienced by residents over the recovery horizon. The model therefore prioritizes interventions that reduce loss for large exposed populations and for origins whose functional access is strongly affected by damaged destinations.

Constraint (1) describes the evolution of underlying functional deficit. The matrix \(A_t\) represents endogenous and routine-recovery dynamics calibrated from observed recovery trajectories. The vector \(h_{t+1}\) represents new external disturbance at time \(t+1\), such as continued flooding, secondary service failures, additional road closures, or new incident reports. The term \(M^Re^R_t\) represents the reduction in underlying deficit caused by durable restoration.

Constraint (2) describes the accumulation and decay of temporary-capacity relief. The vector \(r^C_t\) denotes the relief stock generated by temporary services, \(\delta_C\in[0,1]\) is the decay rate of that relief, and \(M^C\) maps effective temporary-capacity output into spatial relief.

Constraint (3) defines the local experienced deficit. Temporary capacity can reduce the deficit experienced locally, but it does not necessarily repair the underlying functional system. Because the objective minimizes \(\ell_t\), and \(\ell_t\) is increasing in \(d_t\), the constraint implies

\[
d_{i,t}=\max\{0,b_{i,t}-r^C_{i,t}\}
\]

at optimality, subject to the bounded state constraints.

Constraint (4) describes the accumulation and decay of substitution or control relief. The vector \(r^S_t\) captures origin-level reductions in experienced loss generated by information, redirection, service substitution, or other demand-control policies. The matrix \(M^S\) maps effective substitution-control output into origin-level relief, and \(\delta_S\in[0,1]\) is the corresponding decay rate.

Constraint (5) defines access-weighted functional loss. The term \(Q_td_t\) propagates destination-level experienced deficits to origin-level resident loss through the calibrated functional dependence matrix. Substitution or control relief then reduces this origin-level loss. Thus, \(\ell_{i,t}\) is the model’s final measure of urban functional loss experienced by residents of origin \(i\).

Constraint (6) links resource input to effective recovery output under a linear intervention-effectiveness function. Constraints (7) and (8) impose period-specific and total resource budgets for each resource type \(m\in\mathcal M\), such as labor, equipment, temporary service capacity, or funding. Constraint (9) imposes response delay: intervention type \(k\) cannot be deployed before its delay time \(\Delta_k\). Constraints (10)–(12) impose initial conditions, bounded functional states, and non-negativity of intervention variables.

## Linear Programming Structure

The continuous formulation is a linear program when \(Q_t\), \(A_t\), \(M^k\), \(c^m_{i,k,t}\), \(B^m_t\), \(\bar B^m\), and \(\eta^k_{i,t}\) are treated as fixed calibrated parameters. The objective is linear in \(\ell_{i,t}\). The state-transition equations are linear. The access-weighted loss relation

\[
\ell_t\geq Q_td_t-r^S_t
\]

is linear because \(Q_t\) is fixed within each optimization scenario. The max-type relationships implicit in \(d_t=\max\{0,b_t-r^C_t\}\) and \(\ell_t=\max\{0,Q_td_t-r^S_t\}\) are represented through linear inequalities and non-negativity constraints. The budget, response-delay, boundary, and non-negativity constraints are also linear.

Therefore, the continuous recovery model can be written in standard linear-programming form:

\[
\min_x c^\top x
\]

subject to

\[
Hx\leq g,\qquad Fx=q,\qquad x\geq0.
\]

The linear-programming structure is not a claim that all recovery processes are physically linear. It is a modeling choice that isolates the central counterfactual mechanism: limited recovery capacity reduces resident-experienced urban functional loss through empirically calibrated functional dependence.

## Piecewise-Linear Diminishing Returns

The linear effectiveness function can be replaced by a concave piecewise-linear function to represent diminishing marginal returns while preserving linear-programming tractability. Let \(\mathcal S^k_{i,t}\) denote the set of supporting line segments for intervention \(k\) in unit \(i\) at time \(t\). A concave piecewise-linear recovery function can be written as

\[
\phi^k_{i,t}(u)
=
\min_{s\in\mathcal S^k_{i,t}}
\left\{
a^k_{i,t,s}u+b^k_{i,t,s}
\right\}.
\]

The constraint

\[
e^k_{i,t}\leq \phi^k_{i,t}(u^k_{i,t})
\]

is then represented by the linear inequalities

\[
e^k_{i,t}
\leq
a^k_{i,t,s}u^k_{i,t}
+
b^k_{i,t,s},
\qquad
s\in\mathcal S^k_{i,t}.
\]

This specification allows the marginal effectiveness of resources to decline with deployment intensity without introducing integer variables. Integer variables are not required for a concave piecewise-linear hypograph representation. They arise only when the model includes discrete deployment choices, fixed activation costs, indivisible resources, mutually exclusive modes, or governance-bandwidth constraints.

## Governance-Bandwidth Extension

Urban recovery is constrained not only by material resources but also by administrative attention, coordination capacity, and implementation bandwidth. To represent this mechanism, the continuous model can be extended with binary activation variables

\[
z^k_{i,t}\in\{0,1\},
\]

where \(z^k_{i,t}=1\) indicates that intervention type \(k\) is actively deployed in unit \(i\) at time \(t\). Deployment intensity is linked to activation through

\[
0\leq u^k_{i,t}\leq \bar u^k_{i,t}z^k_{i,t},
\]

where \(\bar u^k_{i,t}\) is the maximum deployable capacity. Governance bandwidth is imposed by

\[
\sum_{i\in V}\sum_{k\in K}z^k_{i,t}\leq G_t,
\]

where \(G_t\) is the maximum number of targeted intervention actions that can be initiated at time \(t\).

This extension converts the model from a linear program to a mixed-integer linear program when the recovery-effectiveness function is linear or concave piecewise-linear. The binary variables do not arise from the functional-loss dynamics themselves. They arise from the need to represent discrete activation, limited administrative bandwidth, fixed start-up costs, minimum deployment scales, indivisible assets, or mutually exclusive intervention modes. Without these discrete deployment constraints, the recovery allocation problem remains continuous.

## Recoverable Urban Functional Resilience

Let

\[
J^*(B,\Delta)
\]

denote the optimal objective value under resource budgets \(B=\{B^m_t,\bar B^m\}\) and intervention delays \(\Delta=\{\Delta_k\}\). Let

\[
J^0
\]

denote the cumulative functional loss under the no-additional-intervention baseline, obtained by setting \(u^k_{i,t}=0\) for all \(i,t,k\). We define recoverable urban functional resilience as

\[
\mathcal R_{\mathrm{rec}}(B,\Delta)
=
1-\frac{J^*(B,\Delta)}{J^0}.
\]

This quantity measures the fraction of post-disaster urban functional loss that can be reduced by optimally allocated interventions under specified resource and response constraints. It differs from a conventional recovery-time metric because it explicitly conditions recoverability on resource availability, intervention delay, spatial dependence, and management capacity.

## Parameter Definitions and Empirical Calibration

The model combines observed quantities, empirically calibrated parameters, and counterfactual policy parameters. Table 1 summarizes the main parameters and their empirical status.

| Symbol                  | Meaning                                                      | Empirical status                                             |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| \(V\)                   | Set of spatial units                                         | Defined by the spatial resolution of the study, such as census tracts, block groups, H3 cells, or neighborhoods |
| \(T,\Delta t\)          | Recovery horizon and time-step length                        | Defined by the temporal scale of the recovery analysis       |
| \(p_i\)                 | Population, activity, or exposure weight of unit \(i\)       | Observed or calibrated from census data, mobile activity, or baseline mobility demand |
| \(\hat b_{i,0}\)        | Initial underlying functional deficit after the disaster     | Estimated from mobility activity, traffic speed or volume, POI visits, service disruption records, hazard exposure, and incident reports |
| \(A_t\)                 | Endogenous and routine-recovery operator                     | Calibrated from observed recovery trajectories across time, places, or comparable events |
| \(h_{i,t}\)             | New external disturbance affecting unit \(i\) at time \(t\)  | Observed from weather, flooding, road closures, incident records, service outages, or other event data |
| \(Q_{ij,t}\)            | Functional dependence of origin \(i\) on destination \(j\)   | Calibrated from baseline OD flows, post-disaster accessibility, destination service importance, and mobility or visit data |
| \(S_j\)                 | Functional importance of destination \(j\)                   | Estimated from employment, services, POIs, health-care capacity, transit nodes, retail activity, or other urban-function indicators |
| \(\tau_{ij,t}\)         | Travel time or generalized access cost from \(i\) to \(j\)   | Estimated from road speed, transit access, routing, or accessibility data |
| \(\kappa\)              | Travel-time decay parameter in \(Q_t\)                       | Calibrated from observed mobility or visit patterns          |
| \(M^R\)                 | Spatial effect matrix for durable restoration                | Specified from infrastructure dependence, service networks, spatial adjacency, or scenario analysis; can be set to identity for local effects |
| \(M^C\)                 | Spatial effect matrix for temporary capacity                 | Calibrated or specified from service coverage, access radius, emergency service areas, or scenario analysis |
| \(M^S\)                 | Spatial effect matrix for substitution or control            | Calibrated or specified from policy reach, information coverage, origin-level accessibility, or substitution feasibility |
| \(\eta^k_{i,t}\)        | Linear unit effectiveness of intervention \(k\)              | Calibrated when historical intervention records exist; otherwise treated as a counterfactual intervention-capacity parameter |
| \(\phi^k_{i,t}(\cdot)\) | General intervention-effectiveness function                  | Specified as linear, concave piecewise-linear, or concave nonlinear; subject to sensitivity or ensemble analysis |
| \(\delta_C\)            | Decay rate of temporary-capacity relief                      | Calibrated from duration of temporary services or specified through sensitivity analysis |
| \(\delta_S\)            | Decay rate of substitution-control relief                    | Calibrated from duration of information/control effects or specified through sensitivity analysis |
| \(c^m_{i,k,t}\)         | Unit cost of resource type \(m\) for intervention \(k\) in unit \(i\) at time \(t\) | Estimated from deployment cost, accessibility, isolation, logistics difficulty, or administrative records |
| \(B^m_t\)               | Period-specific budget of resource type \(m\)                | Policy scenario or observed emergency resource availability  |
| \(\bar B^m\)            | Total budget of resource type \(m\) over the recovery horizon | Policy scenario or observed resource constraint              |
| \(\Delta_k\)            | Response delay for intervention type \(k\)                   | Estimated from administrative response records, operational constraints, or scenario analysis |
| \(\bar u^k_{i,t}\)      | Maximum deployable capacity                                  | Determined by physical capacity, inventory, accessibility, staffing, or deployment feasibility |
| \(G_t\)                 | Governance bandwidth at time \(t\)                           | Policy or administrative-capacity parameter used in the mixed-integer extension |

The most strongly data-calibrated components are \(p_i\), \(\hat b_{i,0}\), \(h_t\), and \(Q_t\), which can be estimated from population, mobility, traffic, service, hazard, and incident data. The recovery-dynamics operator \(A_t\), deployment costs \(c^m_{i,k,t}\), and decay rates \(\delta_C,\delta_S\) are empirically informed but typically subject to uncertainty. The intervention-effectiveness functions \(\phi^k_{i,t}\), together with their linear or piecewise-linear parameters, define standardized counterfactual recovery capacity. These parameters should therefore be evaluated through sensitivity analysis or model ensembles rather than interpreted as directly observed causal effects of real-world interventions.
