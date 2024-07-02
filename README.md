# Importance Sampling for Spatial data

In machine learning models, the estimation of errors is often
complex due to distribution bias, particularly in spatial data such as
those found in environmental studies. We introduce an approach based
on the ideas of importance sampling to obtain an unbiased estimate of
the target error. By taking into account difference between desirable error
and available data, our method reweights errors at each sample point and
neutralizes the shift. Importance sampling technique and kernel density
estimation were used for reweighteing. In this work we are trying to implement new techiques to get better accuracy of IS methods. 
As for example, Mandolince framework, Clip, Regularisation and KL-divergence methods.
Finally, we will check our estimations on real data and write theoretical part of our experiments.