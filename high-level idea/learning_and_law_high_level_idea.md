# Learning Recoverability Laws from Resource-Constrained Urban Recovery Optimization

## 1. Purpose of this document

This document summarizes the high-level scientific idea, learning strategy, and law-discovery pipeline for a paper on **recoverable urban resilience** under rainfall disruptions.

The project studies multiple large U.S. cities using empirical rainfall events, observed origin-destination (OD) demand, and traffic speed data. For each city-event, the rainfall start time is aligned to time zero, and the following 12 hours are analyzed as the recovery window. An optimization model has already been constructed to estimate how limited management resources could be allocated during this window to reduce urban functional deficit.

The next step is not merely to train a neural network to reproduce the optimizer. The goal is to use learning-to-optimize as a scientific tool to discover **general laws of managed recovery**: when a rainfall disruption is recoverable, where intervention has the highest value, and why some events are more decision-critical than others.

The central methodological philosophy is:

> **Optimization is the counterfactual probe. Neural learning is the structure extractor. Symbolic law is the final scientific product.**

The paper should therefore avoid being framed as an algorithmic paper about better optimization. Instead, it should be framed as a scientific study of **recoverable loss**: the portion of observed urban functional deficit that can be reduced by targeted, limited, and timely management action.

---

## 2. Scientific motivation

Urban resilience is often measured by observing how a city degrades and recovers after a disruption. In this project, the observed disruption is rainfall-induced degradation in urban mobility and accessibility. The empirical data reveal when and where the transportation system slows down, how OD demand is affected, and how the system returns toward normal conditions over a 12-hour window.

However, observed recovery alone does not answer the most decision-relevant question:

> **How much of the observed functional loss was actually recoverable through intelligent intervention?**

Two rainfall events may produce the same total deficit, but they can differ fundamentally in recoverability. In one event, the deficit may be spread diffusely across many low-leverage roads, making targeted intervention only weakly useful. In another event, a moderate deficit may be concentrated on a small set of demand-exposed bottlenecks, so that a small amount of well-placed intervention could sharply reduce system-wide loss.

Thus, the project distinguishes three concepts:

1. **Observed loss**: the empirically measured degradation over the 12-hour rainfall recovery window.
2. **Recoverable loss**: the portion of observed loss that can be reduced by feasible intervention under resource, time, and deployment constraints.
3. **Decision leverage**: the additional improvement obtained by intelligent allocation relative to naive policies such as random allocation, damage-first allocation, volume-first allocation, or betweenness-first allocation.

The key argument is:

> **Urban disruptions are not decision-critical simply because they are large. They are decision-critical when recoverable value is concentrated in a small number of intervention-sensitive locations.**

This distinction is the foundation of the learning and law-discovery pipeline.

---

## 3. Core research question

The paper should be guided by the following question:

> **Across cities and rainfall events, what structural conditions make urban functional loss recoverable through limited management intervention?**

More concretely, the paper asks:

- Which roads, zones, or intervention actions have the highest marginal recovery value?
- Why are some high-deficit locations not worth prioritizing?
- Why are some moderate-deficit locations highly recovery-critical?
- Under what rainfall, network, OD, and resource conditions does intelligent allocation matter most?
- Can these patterns be distilled into a compact, interpretable law that generalizes across cities?

The expected contribution is not a city-specific recovery policy, but a general scientific understanding of **managed recoverability**.

---

## 4. Key conceptual shift: learn the recovery-value field, not the selected actions

A straightforward learning-to-optimize approach would train a neural network to predict the optimizer's selected resource allocation for each city-event. This is not ideal.

The selected optimal set can be unstable. Many resource allocation problems have multiple near-optimal solutions. If two road segments have almost identical value, the optimizer may select one and not the other due to small numerical differences, tie-breaking rules, or scenario-specific constraints. A neural network trained directly on binary selected/not-selected labels may learn solver artifacts rather than scientific structure.

The better target is the **marginal recovery-value field**.

For each city-event-resource scenario, define each possible intervention as an action token. An action token may represent a road, zone, time step, or intervention type, depending on the current optimization setup. The learning target is not simply whether this action was selected, but how much system-wide deficit would be reduced by allocating marginal resource to that action.

This reframing changes the learning problem from:

> “Which locations did the optimizer choose?”

into:

> “Where does recoverable value exist, how concentrated is it, and what structural conditions generate it?”

This is the central methodological decision of the project.

---

## 5. Data and optimization setting

The empirical setting is rainfall disruption across multiple large U.S. cities.

The project uses:

- **Observed OD data**, capturing how people move between origins and destinations.
- **Observed speed data**, capturing rainfall-induced degradation in traffic performance.
- **Rainfall event timing**, with each event aligned so that the event start is time zero.
- **A 12-hour analysis window**, within which deficit accumulates and management resources may be deployed.
- **Optimization outputs**, estimating how limited resources could reduce system deficit under each city-event scenario.

The optimization model should be treated as an **oracle for counterfactual recoverability**. It asks: under a standardized intervention capacity, how much deficit could have been reduced if resources were allocated optimally?

For learning and law extraction, each optimization scenario should produce several labels:

1. **Observed loss** over the 12-hour window.
2. **Optimal recovery gain** achieved by the optimizer.
3. **Naive-policy gains**, such as random, damage-first, volume-first, betweenness-first, and persistent-deficit-first policies.
4. **Recoverable resilience**, defined as the fraction of observed loss that optimal intervention can reduce.
5. **Decision leverage**, defined as the improvement of optimized allocation over naive allocation.
6. **Action-level marginal recovery values**, which identify the local sources of recoverable value.
7. **Top-tail concentration of recovery value**, which describes whether a small number of actions capture most of the recoverable value.

The action-level labels are the most important for law discovery.

---

## 6. Main risk: the model may learn city-event identity instead of general law

Because the optimization model is solved separately for each city and rainfall event, direct learning can easily produce scenario memorization. For example, a model may learn that a particular highway in one city is usually important, rather than learning a transferable principle about persistent deficit, OD exposure, bottleneck leverage, and substitutability.

This risk should be addressed at multiple levels.

### 6.1 Convert all city-events into a canonical problem family

All variables should be normalized into dimensionless or within-city comparable quantities. The learning model should avoid raw variables that encode city identity.

Examples of useful normalized features include:

- speed deficit relative to the city's normal condition;
- cumulative deficit as a fraction of total event loss;
- road or zone importance expressed as percentile within the city;
- OD exposure normalized by total event demand;
- intervention cost normalized by city-level cost scale;
- budget expressed as a fraction of total event deficit;
- response delay expressed as a fraction of the 12-hour recovery window;
- remaining deficit expressed as area under the deficit curve after the intervention time.

The model should be allowed to know the **structure** of a city, but not the **identity** of the city. It should not receive city ID, road ID, event ID, or raw coordinates that function as identifiers.

### 6.2 Use strict generalization tests

The main validation should not be random splitting across city-events. Random splits can put similar events from the same city in both training and testing sets, making the model look more general than it actually is.

The paper should include at least three generalization tests:

1. **Leave-one-city-out validation**: train on several cities and test on an unseen city.
2. **Leave-event-regime-out validation**: train on certain rainfall regimes and test on different regimes, such as heavier rain, longer-duration events, or spatially concentrated storms.
3. **Leave-time-period-out validation**: train on earlier years and test on later years to reduce temporal leakage.

A law should only be called cross-city if it survives leave-city-out testing.

### 6.3 Use scenario augmentation to force structural learning

Each empirical city-event should be expanded into multiple related but distinct optimization scenarios. This makes it harder for the neural network to memorize a single solution and easier for it to learn how structure changes under different conditions.

Recommended augmentations include:

- **Budget augmentation**: solve the optimization under multiple resource levels.
- **Delay augmentation**: vary the response delay, such as immediate response, 1-hour delay, 2-hour delay, 4-hour delay, and 6-hour delay.
- **Intervention-effectiveness augmentation**: vary assumptions about how strongly a unit of resource reduces deficit.
- **Demand-context augmentation**: use different OD demand profiles such as morning peak, midday, evening peak, and weekend demand, when available.
- **Rainfall-deficit augmentation**: generate plausible alternative deficit footprints from empirically calibrated rainfall-speed relationships, while preserving spatial correlation and temporal realism.

The purpose of augmentation is not to create arbitrary synthetic data. The purpose is to expose the learning model to a distribution of plausible recovery problems, so that it learns principles rather than memorized event-specific patterns.

### 6.4 Consider city-invariant representation learning

A useful optional strategy is city-adversarial learning. The neural model learns a latent representation for predicting marginal recovery value, while an adversarial classifier tries to predict the city identity from that representation. Training discourages the latent representation from encoding city identity directly.

This should be used carefully. The goal is not to remove city structure. The goal is to prevent the model from using city identity as a shortcut. The model should still learn general structural features such as redundancy, bottleneckness, OD concentration, alternative-route scarcity, and rainfall footprint concentration.

---

## 7. Learning target: action-level marginal recovery value

The primary learning unit should be an **action token**.

An action token represents a candidate intervention in a given city-event scenario. Depending on the current optimization design, an action token may be:

- a road segment to receive resource;
- a spatial zone to receive resource;
- a road-time or zone-time pair;
- an intervention type at a location and time;
- a marginal unit of resource allocated to a candidate location.

The target is the **normalized marginal recovery value** of the action.

Several label-generation strategies are possible:

### 7.1 Single-action marginal value

Apply a small standardized intervention to one candidate action while keeping everything else unchanged. Measure how much total deficit is reduced.

This is simple and interpretable, but it may miss interactions between actions.

### 7.2 Greedy trajectory marginal value

Construct a resource allocation sequence. At each step, evaluate the residual marginal value of candidate actions after the previously selected actions have already been applied.

This captures diminishing returns and action interactions more realistically.

### 7.3 Perturbed-optimum selection frequency

Slightly perturb costs, intervention effectiveness, or deficit estimates and solve the optimization multiple times. Record how frequently each action appears in near-optimal solutions.

This addresses instability and non-uniqueness in the optimizer's selected set.

### 7.4 Recommended labeling strategy

The best practical strategy is to combine **greedy residual marginal value** and **perturbed-optimum selection frequency**.

The greedy residual marginal value tells us the action's incremental contribution. The perturbed selection frequency tells us whether the action is robustly important across near-optimal solutions. Together, they produce smoother and more scientifically meaningful labels than binary selected/not-selected outputs.

---

## 8. Neural model: a recoverability decision surrogate

The neural model should be described as a **recoverability decision surrogate**. Its purpose is to learn the optimizer's recovery-value field across many city-event-resource scenarios.

The model should not be presented as merely accelerating optimization. Its scientific role is to compress thousands or millions of optimized recovery decisions into a learnable representation from which general laws can be distilled.

### 8.1 Input representation

Each action token should be represented by five groups of features.

#### A. Deficit features

These describe how the rainfall event degrades the candidate location:

- instantaneous speed deficit;
- peak deficit;
- cumulative deficit;
- remaining deficit after the candidate intervention time;
- deficit duration;
- passive recovery slope;
- whether the local condition is still worsening or already recovering;
- rainfall intensity and persistence around the location.

The most important feature is expected to be **remaining deficit area**, not peak deficit. A severe deficit that disappears quickly may have less recovery value than a moderate deficit that persists for many hours.

#### B. Demand exposure features

These describe how much human mobility depends on the candidate location:

- OD demand using or affected by the location;
- number of OD pairs exposed;
- time-of-day demand intensity;
- origin population exposure;
- destination functional importance;
- essential-trip exposure, if available.

Demand exposure is critical because a local degradation only becomes systemically important when it affects many people or important trips.

#### C. Structural leverage features

These describe whether the location is a network bottleneck or leverage point:

- betweenness centrality;
- OD-weighted betweenness;
- road hierarchy;
- bridge, cut, or corridor indicators;
- detour ratio;
- alternative-route capacity;
- local redundancy;
- accessibility contribution;
- path substitutability.

The key hypothesis is not that high-betweenness locations are always important. Rather, bottleneck leverage is **activated** only when it overlaps with persistent deficit and exposed demand.

#### D. Intervention feasibility features

These describe whether intervention can still help:

- resource cost;
- deployment difficulty;
- intervention effectiveness;
- response delay;
- remaining time in the 12-hour window;
- local accessibility for deployment;
- whether intervention effects are temporary or persistent.

An action with high structural importance may have low recovery value if it is too costly, too late, or naturally recovering before intervention can take effect.

#### E. Event-level context features

These describe the whole rainfall event and city context:

- total event loss;
- rainfall footprint size;
- rainfall footprint concentration;
- spatial autocorrelation of deficit;
- deficit entropy or Gini concentration;
- OD concentration;
- city-level redundancy;
- resource intensity;
- response delay ratio;
- fraction of loss located on bottlenecks;
- fraction of loss located in low-substitution areas.

Event-level context helps the model understand whether the scenario is globally decision-critical.

### 8.2 Architecture

A suitable architecture is a **spatio-temporal graph action-value scorer**.

The model can include:

1. **Spatial graph encoder**: captures road or zone adjacency and structural relationships.
2. **Temporal encoder**: captures the 12-hour evolution of deficit.
3. **OD-dependency encoder**: captures which origins and destinations depend on each candidate location.
4. **Global context encoder**: captures event-level rainfall, budget, delay, and city-level conditions.
5. **Action-value head**: predicts the marginal recovery value of each candidate action.
6. **Ranking head**: learns within-event action ordering.
7. **Episode-level head**: predicts recoverable resilience, decision leverage, and top-tail concentration.

The architecture should be designed for interpretability. A fully unconstrained graph transformer may achieve high accuracy but make law extraction difficult. A better design is a **factorized model** that separates local action score, event-level modulation, and interaction terms.

For example, the model can be conceptually organized as:

- local recoverability score from action features;
- event context modifier from city-event features;
- interaction correction for deficit-demand-structure coupling.

This structure helps the later symbolic law extraction step.

---

## 9. Training objectives

The model should be trained with multiple complementary objectives.

### 9.1 Marginal value regression

The model predicts normalized marginal recovery value for each action token.

Normalization is important. The raw recovery value should be divided by event-level observed loss, maximum possible recovery, or another comparable scale. This prevents large cities or high-loss events from dominating the learning process.

### 9.2 Within-event ranking loss

The model should learn which actions are more valuable than others within the same event. Pairwise or listwise ranking losses are appropriate.

Ranking is often more robust than absolute-value regression because different events have different loss scales.

### 9.3 Top-K policy regret loss

The model should be evaluated and possibly trained based on how much recovery is lost when selecting the top-K actions predicted by the model instead of the optimizer's top-K actions.

This aligns the neural objective with the scientific and policy objective: selecting high-value actions under limited resources.

### 9.4 Episode-level loss

The model should also predict event-level quantities:

- recoverable resilience;
- decision leverage;
- top-tail concentration;
- optimized-over-random gain.

This auxiliary task encourages the model to understand the global structure of recoverability, not only local scores.

### 9.5 Optional city-adversarial loss

A city-adversarial loss can discourage the latent representation from encoding city identity directly. This can improve cross-city generalization and reduce the risk of memorization.

---

## 10. Evaluation metrics

The evaluation should be decision-centered, not only prediction-centered.

Useful metrics include:

- **NDCG@K** for ranking high-value actions;
- **Precision@K** for identifying optimizer-important actions;
- **Spearman rank correlation** within each event;
- **Top-K regret** relative to the optimizer;
- **Recovery gain achieved by the neural surrogate**;
- **Recoverable resilience prediction error**;
- **Decision leverage prediction error**;
- **Leave-one-city-out performance**;
- **Leave-regime-out performance**.

The paper should avoid relying only on R² or mean squared error. A model can predict average values accurately while failing to identify the small top-tail of actions that matters most for intervention.

---

## 11. Structure decoupler: identifying the variables that matter

After training the recoverability decision surrogate, the next step is to identify the low-dimensional structure behind its predictions.

This step can be called the **Structure Decoupler**.

The goal is not merely to explain the neural network. The goal is to determine which physical, demand, temporal, and intervention variables determine the optimizer-derived recovery-value field.

### 11.1 Feature-group ablation

Remove one feature group at a time and measure how much decision performance declines.

Feature groups include:

- deficit features;
- demand exposure features;
- structural leverage features;
- substitutability features;
- passive recovery features;
- intervention feasibility features;
- event-level rainfall features;
- budget and delay features.

The key metric should be top-K regret and recovery gain, not only prediction error.

### 11.2 Interaction ablation

The expected law is not additive. It depends on interactions.

The paper should compare models with increasing interaction complexity:

1. deficit only;
2. deficit plus OD exposure;
3. deficit, OD exposure, and bottleneck leverage;
4. deficit, OD exposure, bottleneck leverage, and substitution scarcity;
5. full model with time window and intervention feasibility.

This will test whether recoverability emerges from coupled deficit-demand-structure-time mechanisms.

### 11.3 Graph structure ablation

Test how much graph information is actually necessary.

Compare:

- full road graph or zone graph;
- shuffled graph;
- local features only;
- structural features only;
- OD-dependency graph only;
- road adjacency plus OD-dependency graph.

If low-dimensional structural features capture most of the performance, that supports a simple law. If explicit graph interactions are necessary, the law may need a higher-order structure.

### 11.4 Stability analysis

Repeat the decoupling analysis across:

- different cities;
- different rainfall intensities;
- different times of day;
- different resource budgets;
- different response delays;
- different intervention-effectiveness assumptions;
- different optimization parameter ensembles.

Only variables that are stable across these settings should enter the final symbolic law.

---

## 12. Formula extractor: distilling interpretable laws

The final step is to translate the neural surrogate and decoupled variables into interpretable laws.

This step can be called the **Formula Extractor**.

The formula extractor should not use every possible feature. Too few features will underfit. Too many features will create formulas that are accurate but uninterpretable. The right approach is to search for the best trade-off between predictive power and symbolic simplicity.

Recommended tools include:

- sparse regression;
- generalized additive models;
- monotonic neural additive models;
- symbolic regression;
- equation search with complexity penalties;
- Pareto-front selection between accuracy and simplicity.

The final law should be selected using four criteria:

1. It explains a large fraction of optimizer-derived recovery value.
2. It has low symbolic complexity.
3. It generalizes across unseen cities and event regimes.
4. It has a clear physical interpretation.

The law should then be validated by using it directly as a policy heuristic: rank candidate actions by the symbolic score, select top-K actions, and compare the achieved recovery gain against the optimizer and baselines.

---

## 13. Expected Law A: the activated-bottleneck law

The first expected law is a **local action-level law**.

A candidate location becomes recovery-critical when several conditions align:

- the location has persistent remaining deficit;
- substantial OD demand depends on it;
- it has high structural leverage, such as bottleneck or corridor importance;
- substitutes are scarce, so travelers cannot easily avoid the disrupted location;
- intervention can occur before the deficit naturally disappears;
- the intervention cost is not prohibitive.

In compact verbal form:

> **Recovery value arises when persistent deficit, exposed demand, bottleneck leverage, substitution scarcity, and intervention feasibility coincide.**

A possible conceptual score is:

> **Recovery value ≈ persistent remaining deficit × OD exposure × bottleneck leverage × substitution scarcity × intervention window / intervention cost.**

This law is deliberately stronger than simple heuristics.

It implies that the most valuable intervention target is not necessarily:

- the location with the largest speed drop;
- the location with the highest traffic volume;
- the location with the highest betweenness;
- the location with the largest rainfall exposure.

Instead, the most valuable target is the location that converts a unit of intervention into the largest reduction in system-wide experienced deficit.

The expected non-obvious finding is:

> **The most disrupted locations are often not the most recoverable locations.**

Another expected finding is:

> **Bottleneck centrality is not itself recovery value; it is a latent leverage that becomes activated only when rainfall-induced persistent deficit and exposed OD demand overlap with low substitutability.**

This is the local law of managed recoverability.

---

## 14. Expected Law B: the top-tail decision-criticality law

The second expected law is an **event-level law**.

For each rainfall event, the marginal recovery values of all candidate actions form a distribution. Some events may have a heavy top tail: a small number of actions account for most recoverable value. Other events may have a flat distribution: recoverable value is spread diffusely across many actions.

The central hypothesis is:

> **A rainfall event is decision-critical when recoverable value is concentrated in a small top-tail of candidate interventions.**

This creates a distinction between two event-level quantities:

1. **Addressable loss**: how much of the observed loss can be reduced by intervention at all.
2. **Top-tail concentration**: how much of that recoverable value can be captured by the best few actions.

Together, these determine decision leverage.

In compact verbal form:

> **Recoverable resilience depends on the amount of addressable loss; decision leverage depends on the inequality of the marginal recovery-value distribution.**

This law is important because it separates disruption magnitude from intervention value.

A severe rainfall event may produce large total deficit but low decision leverage if the deficit is widespread and diffuse. A moderate rainfall event may produce high decision leverage if it concentrates persistent deficit on a small number of demand-exposed, low-substitution bottlenecks.

The expected non-obvious finding is:

> **Large disruptions are not necessarily decision-critical; moderate disruptions can be more decision-critical when they activate a small set of high-value recovery targets.**

This should become one of the main scientific messages of the paper.

---

## 15. Expected non-obvious empirical findings

The learning and law-discovery pipeline should be designed to test the following findings.

### 15.1 Highest deficit is not highest priority

A location with severe speed degradation may not have high recovery value if few OD trips depend on it, if substitutes are available, or if the deficit recovers naturally within a short time.

### 15.2 High volume is not enough

A high-volume road may not be the best intervention target if the rainfall-induced deficit is small or short-lived.

### 15.3 High betweenness is not enough

A topological bottleneck matters for recoverability only when the rainfall event actually degrades it and affected OD demand lacks substitutes.

### 15.4 Persistent deficit dominates peak deficit

For a 12-hour recovery window, the value of intervention should depend more on the remaining area under the deficit curve than on the instantaneous peak deficit.

### 15.5 Decision leverage is budget-nonmonotonic

The advantage of intelligent allocation over naive allocation may be highest at intermediate resource levels. When resources are extremely scarce, no policy can help much. When resources are abundant, many locations can be treated, so the choice of allocation matters less. Under intermediate scarcity, choosing the right targets matters most.

### 15.6 Severe events may have low decision leverage

Widespread severe rainfall can cause high total loss but low optimized-over-naive gain if recoverable value is diffuse. This creates a counterintuitive distinction between event severity and decision-criticality.

### 15.7 Moderate events may have high decision leverage

A moderate event that hits a few critical, demand-exposed, low-substitution corridors may be more recoverable and more decision-critical than a larger diffuse event.

---

## 16. Hindsight versus online recoverability

The current analysis aligns events at rainfall start and considers the next 12 hours. If the optimization uses the full observed 12-hour trajectory, the study should be framed as **hindsight counterfactual recoverability**.

This means the paper asks:

> Given what actually happened during the 12-hour window, how much of the loss was theoretically recoverable under standardized intervention capacity?

This is a valid scientific question. It does not require claiming that the city had perfect real-time knowledge during the event.

A separate supplementary analysis can test early predictability:

- using only the first 30 minutes;
- using only the first 1 hour;
- using only the first 2 hours;
- comparing early-window predictions with full-window recoverability.

This would show whether decision-critical events can be identified early enough for operational relevance. But the main paper should first establish the hindsight law robustly before claiming online decision support.

---

## 17. Full pipeline

The recommended workflow has six stages.

### Stage 1: Generate optimization oracle outputs

For each city-rainfall event, run the optimization under multiple resource and delay scenarios. Store not only the optimal allocation but also recovery gains, naive baseline gains, action-level marginal values, and top-tail concentration statistics.

### Stage 2: Build the action-token dataset

Create one training sample for each action token in each scenario. Each sample contains local deficit features, OD exposure features, structural leverage features, substitution features, intervention feasibility features, and event-level context.

Do not include city ID, road ID, event ID, or raw coordinates that allow direct memorization.

### Stage 3: Train the recoverability decision surrogate

Train a spatio-temporal graph action-value model to predict action-level marginal recovery value, within-event ranking, and episode-level recoverability quantities.

Use leave-city-out and leave-regime-out validation as primary tests.

### Stage 4: Apply the structure decoupler

Use feature ablation, interaction ablation, graph ablation, attribution methods, and stability analysis to identify the low-dimensional variables that determine recovery value.

The output should be a compact set of candidate variables for law extraction.

### Stage 5: Extract symbolic laws

Use sparse interpretable models and symbolic regression to distill local and event-level laws. Select laws based on accuracy, simplicity, generalization, and physical interpretability.

### Stage 6: Validate the laws as policies

Use the symbolic law to rank candidate interventions. Compare the recovery gain of law-guided interventions against:

- the optimizer;
- the neural surrogate;
- damage-first allocation;
- volume-first allocation;
- betweenness-first allocation;
- persistent-deficit-first allocation;
- random allocation.

The law should be considered successful if it captures a large fraction of optimizer recovery gain while using only a small number of interpretable variables.

---

## 18. Recommended figure structure

### Figure 1: From observed rainfall disruption to recoverable loss

Show empirical rainfall events, speed deficits, OD demand, 12-hour recovery curves, and the distinction between observed loss, optimized recoverable loss, and decision leverage.

Main message:

> Observed disruption, recoverable loss, and decision leverage are different quantities.

### Figure 2: Learning the recovery-value field

Show the action-token dataset, neural surrogate architecture, and leave-city-out performance. Compare surrogate-selected top-K actions with optimizer-selected top-K actions.

Main message:

> The neural model learns cross-city recovery-value structure rather than city-specific solutions.

### Figure 3: Structure decoupler

Show feature-group importance, interaction ablation, and top-K regret after removing key variable groups.

Main message:

> Recoverability depends on the interaction of persistent deficit, OD exposure, bottleneck leverage, substitutability, and intervention timing.

### Figure 4: Local activated-bottleneck law

Show symbolic local recovery-value law, scatter of predicted versus optimizer-derived marginal values, and examples where highest-deficit, highest-volume, or highest-betweenness heuristics fail.

Main message:

> The most recoverable location is not necessarily the most disrupted location.

### Figure 5: Event-level top-tail law

Show marginal recovery-value distributions across events, top-tail concentration, and a phase diagram of observed loss versus decision leverage.

Main message:

> Events are decision-critical when recoverable value is concentrated in a small top tail.

### Figure 6: Law-guided intervention validation

Compare law-guided allocation with optimizer and baselines across cities, rainfall intensities, budgets, and delays.

Main message:

> A compact law can capture most of the optimizer's recoverability benefit and generalize across cities.

---

## 19. Main pitfalls and final design choices

### Pitfall 1: Learning binary optimizer decisions

Binary selected/not-selected labels are unstable and may reflect arbitrary solver choices.

Final choice:

> Learn marginal recovery values and within-event rankings instead of binary decisions.

### Pitfall 2: Learning only event-level recoverability

Event-level prediction does not explain where recoverable value comes from.

Final choice:

> Learn action-level value fields and aggregate them into event-level laws.

### Pitfall 3: Memorizing city identity

The model may learn that certain city-specific roads are always important.

Final choice:

> Use normalized variables, avoid identity features, perform leave-city-out validation, use scenario augmentation, and optionally use city-adversarial learning.

### Pitfall 4: Producing a black-box neural policy

A black-box policy is not the scientific contribution and will be hard to defend.

Final choice:

> Use the neural model as an intermediate structure extractor, followed by decoupling and symbolic law extraction.

### Pitfall 5: Extracting an overcomplicated formula

A formula with too many variables may be accurate but not scientifically useful.

Final choice:

> Use a Pareto frontier between accuracy and complexity; select the simplest law that remains predictive and stable across cities.

### Pitfall 6: Optimization-model artifact

The discovered law may depend on arbitrary assumptions in the optimization model.

Final choice:

> Test stability across budgets, delays, intervention efficiencies, cost assumptions, and alternative optimization parameterizations. Only robust patterns should be called laws.

### Pitfall 7: Confusing hindsight analysis with online decision-making

Using the full 12-hour trajectory does not represent real-time information.

Final choice:

> Frame the main analysis as hindsight counterfactual recoverability, then optionally add early-window prediction for policy relevance.

---

## 20. Relationship to abductive AI law discovery

The project follows an abductive AI logic: use a high-capacity neural model to learn complex high-dimensional mappings, then use interpretability and symbolic distillation to extract simple laws.

The reference transportation-resilience paper follows a three-part pipeline:

1. a dynamics surrogate based on graph neural networks;
2. a structure decoupler based on explainable AI;
3. a formula extractor based on neuro-symbolic regression.

This project adapts that logic but changes the scientific target.

The reference paper studies **inherent transportation resilience**: how a network degrades under simulated disruptions. This project studies **recoverable urban resilience**: how much empirically observed rainfall-induced loss can be reduced by limited intervention.

The key difference is:

> The reference problem learns laws of degradation; this project learns laws of managed recovery.

Therefore, the neural surrogate should not predict only system resilience. It should learn the optimizer's marginal recovery-value field. The symbolic law should not merely identify static critical bottlenecks. It should identify **event-activated recovery-critical locations** and **decision-critical rainfall events**.

---

## 21. Proposed paper-level thesis

The paper can be organized around the following thesis:

> **Urban recoverability is not determined by disruption magnitude alone. Across rainfall events and cities, recoverable value emerges when persistent mobility deficit overlaps with exposed OD demand, structural bottleneck leverage, limited substitutability, and feasible intervention timing. At the event level, intelligent management matters most when this recoverable value forms a heavy top tail across candidate interventions.**

A more concise version is:

> **Rainfall disruptions are decision-critical not when they are largest, but when persistent functional loss is concentrated on demand-exposed, low-substitution bottlenecks whose recovery value forms a heavy top tail.**

This is the high-level law the learning pipeline should be designed to test and refine.

---

## 22. Suggested terminology

Recommended terms:

- **Recoverable urban resilience**: fraction of observed loss reducible by feasible targeted intervention.
- **Recoverable loss**: loss that can be reduced by intervention.
- **Decision leverage**: optimized-over-naive gain.
- **Marginal recovery-value field**: spatial-temporal distribution of marginal intervention value.
- **Activated bottleneck**: a structurally important location whose recovery value is activated by persistent deficit, OD exposure, and low substitutability.
- **Top-tail concentration**: fraction of recoverable value captured by top-ranked actions.
- **Decision-critical event**: an event in which optimized intervention substantially outperforms naive intervention because recoverable value is highly concentrated.

Terms to avoid as the central framing:

- road repair optimization;
- traffic assignment optimization;
- emergency vehicle routing;
- resource dispatch algorithm;
- neural optimizer acceleration.

These terms risk making the paper look like a transportation engineering or operations research paper rather than a study of recoverable urban resilience.

---

## 23. Minimal implementation checklist

The immediate next steps are:

1. For every city-event-resource scenario, compute observed loss, optimized gain, and baseline-policy gains.
2. Generate action-level marginal recovery labels, preferably using residual marginal value and perturbed-optimum stability.
3. Normalize all action and event features into city-comparable or dimensionless quantities.
4. Train simple baselines first: damage-only, volume-only, betweenness-only, persistent-deficit-only, and manually constructed multiplicative scores.
5. Train the recoverability decision surrogate with action-value regression and ranking losses.
6. Evaluate under leave-city-out and leave-regime-out splits.
7. Run structure decoupling to identify robust feature groups and interactions.
8. Distill local and event-level symbolic laws.
9. Validate law-guided intervention against optimizer and baselines.
10. Build the final narrative around activated bottlenecks and top-tail decision-criticality.

---

## 24. Final one-paragraph summary

This project uses optimization not as the final contribution, but as a counterfactual engine for measuring recoverable urban loss under rainfall disruptions. For each city-event, the optimizer reveals where limited intervention could reduce 12-hour mobility deficit. Instead of learning the optimizer's discrete selected actions, the proposed learning-to-optimize pipeline learns the marginal recovery-value field across all candidate interventions. A spatio-temporal graph surrogate then captures how persistent deficit, OD exposure, structural leverage, substitutability, and intervention timing interact to generate recovery value. Through structure decoupling and symbolic distillation, the paper aims to extract two laws: a local activated-bottleneck law explaining why certain locations become recovery-critical, and an event-level top-tail law explaining when rainfall events are truly decision-critical. The expected scientific insight is that urban recoverability is not governed by disruption magnitude alone, but by the concentration of addressable functional loss on a small set of demand-exposed, low-substitution bottlenecks.
