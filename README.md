# DynamicalModeling
Functions and code for DESJ2112 dynamical + lens modeling algorithm


"""
This repository contains all the functions for the ASTRO3D Galaxy Evolution With Lenses (AGEL; see Barone et al 2026 AJ 171 57) survey's ongoing analysis of DESJ2112, or "Spiral III." The repository includes the full pipeline for the integrated velocity dispersion, spatially-resolved kinematics, and (TBD) the joint dynamical + lens model (using jampy and PyAutoLens). The kinematics/dynamics part of this pipeline was assembled with help from the TDCOSMO collaboration, particularly William Sheu, Shawn Knabel, and Anowar Shajib.


ACKNOWLEDGMENTS:
Integrated velocity dispersion: Much of the methodology for this pipeline follows Knabel et al. (2025).
Spatially-resolved kinematics: Our code uses Squirrel, which was developed as part of TDCOSMO XXIV (Shajib et al. 2026). Our code was assembled with help from Shawn Knabel's Squirrel code.
Dynamical Modeling: the methodology for this pipeline largely follows TDCOSMO XII (Shajib et al. 2023), with help from the code used in TDCOSMO XXV (Sheu et al. 2026). Additional help was obtained using Knabel et al. (2026).

All data are described in AGEL DR2 (Barone et al. 2025). Thanks to Kaustubh Gupta for his excellent reduction of our red-side data.
"""
