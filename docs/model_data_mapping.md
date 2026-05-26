# Model-Data Mapping

| Model element | Empirical role | Candidate data source | Current status |
| --- | --- | --- | --- |
| `p_i` | Population, activity, or exposure weight | demand volumes, OD totals, optional census/mobile data | Partially supported by demand data |
| `b_{i,t}` | underlying functional deficit | speed deficit, volume reduction, service disruption proxy | Supported as mobility-functional proxy |
| `d_{i,t}` | locally experienced deficit after temporary capacity | requires intervention/relief records or counterfactual assumption | Requires model assumptions |
| `ell_{i,t}` | access-weighted experienced loss | demand-weighted speed/access loss | Partially supported |
| `Q_t` | origin-destination functional dependence | OD demand, route assignment, accessibility cost | Partially supported |
| `A_t` | endogenous recovery dynamics | observed recovery trajectories after rainfall/disruption | Partially supported |
| `h_t` | external disturbance | rainfall and possible incident/hazard records | Supported for rainfall shocks |
| `eta^k` | intervention effectiveness | emergency response/intervention records | Not directly observed |
| resource budgets | counterfactual constraints | scenario design or emergency resource records | Requires scenarios |
| response delay | counterfactual/policy constraint | administrative records or scenarios | Requires scenarios |

The data-mining stage can assess the observability of disruption, recovery, and dependence. It cannot identify intervention recoverability without the optimization/counterfactual layer.
