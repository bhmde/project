# Mechanistic Interpretability of Learning Dynamics in Game-Playing Neural Networks

Elliot Meldrum, Bradley Arias, Max Fierro, Dmytro Krukovskyi, Humberto Gutierrez

---

## Abstract

We investigate the relationship between interpretability and learning dynamics within neural networks that optimize for two computationally equivalent game-theoretic tasks; decision-making and utility evaluation. In the decision-making network, we use a selection of probes to determine the recoverability of interpretable features over the course of training. In the evaluation network, we analyze the preservation of algebraic properties of the native state representations across training epochs through dimensionality reduction and mechanistic analysis of intermediate activations. We leverage our access to a strong solution of the game underlying these tasks to eschew the need for human selection of interpretable features and to provide ground-truth statistics which help with the interpretation of our measurements.

## Hypotheses

We hypothesize that in the case of the policy (decision-making) network, there will be an increase in the recoverability of features that are highly correlated with optimal moves under optimal-play dynamics over the course of training, in a way that is potentially independent of traditional performance metrics. In particular:

* The recoverability of interpretable features via probing may continue to increase throughout training despite there being no improvements to model accuracy.
* Sudden and dramatic increases in validation accuracy after no improvements to model loss (so-called “grokking”) may be accompanied by considerable increases to probe accuracy.
* The recoverability of interpretable features that are uncorrelated with optimal actions will be invariant or reduced when compared to correlated features.

In the case of the evaluation network, we hypothesize that nonlinear dimensionality reduction techniques applied to intermediate activations of game states that are symmetric to each other will reveal lower-rank structures that reflect this algebraic property. This is a similar expectation to the findings of Power et al. (2022), where dimensionality reduction of intermediate activations of a network tasked over an algebraic domain (calculating remainders of integer division) resulted in visualizations which reflect the domain’s algebra (of cyclic subgroups).
