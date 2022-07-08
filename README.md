## dft_gan: the combined use of DFT, microkinetics, and GAN
### flow

1) make_surf.py: make surface --> surf.json is generated

2) calc_reaction_energy.py: calculate reaction energy --> reaction_energy.json is generated
* ase.Atoms is loaded from "surf.json"
* reaction energy is calculated according to the calculator, and then calculated reaction energy is added to "reaction_energy.json"

3) nn_reac.py: GAN
* loads "surf.json" and "reaction_energy.json" 
* proposes new systems with GAN
* new systems are written to "surf.json"
