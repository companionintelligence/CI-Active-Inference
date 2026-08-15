# Friston papers (active sampling)

Notes on Karl Friston's papers that underwrite an epistemic sampler, not a pixel-scanning eye. Sources are on [his UCL page](https://www.fil.ion.ucl.ac.uk/~karl/). Runnable MATLAB lives in [SPM](https://www.fil.ion.ucl.ac.uk/spm/) (`DEM` GUI: Visual search and Visual foraging / `DEM_demo_MDP_search.m`; solver `spm_MDP_VB_X.m`). PDFs stay local and are not in git.

These six papers are one argument written across four years. Read them in this order.

## 1. Perceptions as Hypotheses: Saccades as Experiments (2012)

`2012-perceptions-as-hypotheses-saccades-as-experiments.pdf`

The eye paper. Gregory said perception is hypothesis testing. Friston says a saccade is the experiment. The agent does not stare at a whole image. It holds a few internal templates (upright face, inverted face, rotated face) and only sees a foveal patch through center-surround receptive fields, with a Hamming falloff so the periphery is imprecise.

After about 196 ms it builds a salience map: for every possible next fixation, how much would uncertainty shrink if it looked there. It jumps to the peak, then inhibits that spot (inhibition of return). In the demo the scene is Nefertiti or a face. The map lights up ear, eye, mouth, the same places Yarbus recorded. Salience is not in the pixels. It is in the hypothesis. If the picture is not one of the templates, the agent never settles and wanders.

Dark-room problem, inverted: prior beliefs say salient features will disclose themselves. Agents must sample what would confirm or kill a hypothesis, or they never know anything.

## 2. Embodied Inference and Spatial Cognition (2012)

`2012-embodied-inference-and-spatial-cognition.pdf`

Same simulation, shorter paper. The extra point is embodiment. Action is a reflex that kills proprioceptive prediction error. You predict where the eye will be, the brainstem fulfills it, so looking is the body completing a prediction. The only hypothesis that survives eight saccades is the one that correctly forecast the next patch. Perception verifies itself by moving.

Also shows a handwriting arm doing the same trick with a Lotka-Volterra attractor. Recap of paper 1, not new math. Skip once you have 1.

## 3. Active Inference and Epistemic Value (2015)

`2015-active-inference-and-epistemic-value.pdf`

The equation to implement. Expected free energy of a policy splits in two:

- **Epistemic value**: information gain. Will this observation change my beliefs.
- **Extrinsic value**: preference. Do I like the outcome.

You explore while uncertain, then exploit when you are sure. Curiosity without a separate reward hack. For an eye, looking at an informative patch is epistemic. For companion memory, opening the file that would collapse a belief is the same term.

This is why HBPE and the sampler are different jobs. HBPE forecasts what you will do. This math picks what to sample so you know.

## 4. Active Inference: A Process Theory (2016)

`2016-active-inference-a-process-theory.pdf`

How a brain, or a program, would run that math. Perception, policy selection, and learning all **descend the same free energy**: they walk downhill on one scalar. High free energy means the current beliefs fail to explain what was just seen. You cannot jump to the perfect posterior, so you nudge knobs in the direction that makes the score smaller.

- Perception nudges beliefs about hidden states.
- Policy selection nudges beliefs about what to do next.
- Learning nudges the A and B matrices.

Hidden states are kept over time, past and future, so you get prediction and postdiction (Bayesian smoothing). Updates are softmax of linear mixtures, which is why later code (`pymdp`, `spm_MDP_VB_X`) looks the way it does. The demo is a radial maze, not a face. If we build from his math, this is the update loop.

## 5. Active Inference and Learning (2016)

`2016-active-inference-and-learning.pdf`

Companion to process theory. Once you can choose, you can learn A and B by watching yourself (Dirichlet / associative updates). Habits are what you get when a goal-directed policy gets copied into a state-action map. Ambiguity-resolving exploration has to happen before reward-seeking, or you never learn the map. If there is no ambiguity, the scheme collapses to Bellman.

This is how the sampler gets better: after each read, increment the likelihood that this slice came from that hypothesis. After the day closes, postdiction asks which unread slice would have made the outcome cheap to explain, then nudges those pairings.

## 6. Scene Construction, Visual Foraging, and Active Inference (2016)

`2016-scene-construction-visual-foraging.pdf`

The discrete eye, and the one to port first. You cannot see the whole scene. Four corners, start in the center. Objects are bird, seed, cat. Category is feed, flee, or wait, from spatial relations, not absolute position.

Hidden states **factor** into context, where you are looking, and flips of the scene (mean field: you do not store every object in every location). You look, update, look again. When you are sure, you saccade to a choice spot and get right or wrong. That preference is the only C vector that matters.

Code: `DEM_demo_MDP_search.m` in SPM. Same solver as process theory.

## What we are taking (and not taking)

Not a pixel-scanning fovea. The 2012 continuous eye is the pretty picture. The 2016 discrete forager plus the expected-free-energy split from 2015 is the stack: sample the next informative slice of a vast dataset, keep a factorial state space, score branching futures, measure surprise, postdict, improve the matrices.

Runnable MATLAB is not in this folder. Papers 1 and 6 point at the DEM toolbox inside SPM.
