# Modeling Contour Measurements of Elliptical Extended Objects via Gaussian Spatial Distributions

Code for the paper "Modeling Contour Measurements of Elliptical Extended Objects via Gaussian Spatial Distributions" by
Simon Steuernagel, Kolja Thormann, and Marcus Baum, published at the 2024 SDF Workshop.

**Abstract**:
Many elliptical extended object tracking methods model the spatial distribution of the measurement sources on the object
with a Gaussian distribution. With the help of moment matching, alternative spatial distributions of the measurement
sources can be incorporated. For example, a uniform distribution on the surface leads to a constant scaling of the
matching Gaussian distribution’s covariance matrix. This work is concerned with measurement sources from the contour of
an elliptical object. It is shown that for a circle with uniformly distributed measurement sources on the contour, a
constant scalar factor is obtained. This scaling factor still holds for an ellipse when the contour points are stretched
accordingly. If, however, the semi-axes are not equal and the distribution is uniform on the contour, individual scaling
factors for each axis are required. These depend on the (unknown) ratio of semi-axis lengths, but it is shown that they
can also be estimated online in a recursive filtering framework. The applicability of the results to extended object
tracking are evaluated based on simulations, showing that by employing the correct scaling factors, the Random Matrix (
RM) filter can accurately track an extended object given contour measurements.

Furthermore, please consider having a look at the following closely related work:\
[Random Matrix-based Tracking of Rectangular Extended Objects with Contour Measurements](https://ieeexplore.ieee.org/abstract/document/10706288)\
[GitHub repository](https://github.com/Fusion-Goettingen/FUSION_2024_Steuernagel_LidarRM)

where rectangular objects are considered.
