# superbit_neff
Veome's Senior Thesis repo '21. Uses existing HST results to model the expected SIDM cross-section constraints achievable by SuperBIT.

## Usage:
1. Run `generate_catalog.ipynb`. This will give all the COSMOS sources in terms of Shape Band AB Mags. WARNING: This may take several minutes.
2. Run `neff_superbit.ipynb`. This notebook estimates the Background Galaxy Density (Neff) vs AB Magnitude observed by SuperBIT, factoring in blending as well as band effects.
3. Run `sidm_uncertainty.ipynb`. This notebook uses a given Neff to estimate the constraints on SIDM cross-section. This computation is a back-of-the-envelope implementation of the complete HST simulations done in [Harvey 13]( https://doi.org/10.1093/mnras/stt819) and [Harvey 14](https://doi.org/10.1093/mnras/stu337).

