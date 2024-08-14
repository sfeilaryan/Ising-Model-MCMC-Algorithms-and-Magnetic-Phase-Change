# Magnetic Phase Transitions in Lattice Configurations Modelled by the 2D Ising Model and MCMC Methods.

## Synopsis

This repository contains the report, presentation, and supporting scripts for the project presented by Ryan Sfeila, Paul Mathiot, and Andrey Pillay for the PHY204 - Classical Electrodynamics Bachelor Program course directed by Professor Arnaud Couairon at École Polyetchnique. The report and presentation can be found in the PHY204 Project Files -> Report and Presentation directory, and all supporting documents can be found in their respective sub-directories in PHY204 Project Files.

Throughout this project, we investigate the properties and behaviors of simple
magnetic systems using a 2D Ising model and the metropolis move method, an example of Monte-Carlo Markov Chain algorithms. By simulating lattice configurations and incorporating notions from statistical physics and thermodynamics, we explore key concepts such as Curie’s law of paramagnetism, spontaneous magnetization, and the lattice’s response to external fields and temperature variations. We are
particularly interested in recovering evidence of phase transition analogous to the predicted
shift from paramagnetic to ferromagnetic character when nearing the Curie temperature of
certain materials.
 
All relevant information is outlined in the report, which we strongly encourage the reader to refer to.

## Citation
If you use this report or the provided scripts in your research, please cite them using the following metadata.

``` bibtex
@misc{sfeila_pillay_mathiot_2024,
  author = {Ryan Sfeila and Andrey Pillay and Paul Mathiot},
  title = {Magnetic Phase Transitions in Lattice Magnet Configurations Modelled by the 2D Ising Model and Monte-Carlo Markov Chain Algorithms},
  year = {2024},
  month = {May},
  institution = {École Polytechnique},
  supervisor = {Arnaud Couairon},
  url = {https://github.com/sfeilaryan/Ising-Model-MCMC-Algorithms-and-Magnetic-Phase-Change},
  license = {BSD-3-Clause}
}
```

## Contributions

All authors were involved in the development of the basic simulation algorithms and Metropolis-Hasting protocols. Pillay and Mathiot were in charge of generating
the essential data analysis used to demonstrate the results of this project as well as the simulation algorithms. Pillay was heavily involved in ensuring the integrity of all the data collection algorithms and
further developed the code for the continuous version of the model. Mathiot was particularly involved
in the description of the collected results and bringing them into the theoretical context. Sfeila was in charge of establishing all the necessary theory and supporting knowledge regarding both the 2D Ising
model as a mathematical problem as well as its specific meaning in the project’s context, namely the
Helmholtz free energy and its central role in dictating spontaneous process and the discontinuities of
which correspond to the sought phase transitions. Sfeila was also in charge of the additional material
regarding the simulation improvements by considering system autocorrelation and the convergence of the
Markov chain’s probability distribution to a stationary distribution.

Ryan Sfeila - Author - Email: [ryan.sfeila@polytechnique.edu](ryan.sfeila@polytechnique.edu)

Paul Mathiot - Author - Email: [paul.mathiot@polytechnique.edu](paul.mathiot@polytechnique.edu)

Andrey Pillay - Author - Email: [andrey.pillay@polytechnique.edu](andrey.pillay@polytechnique.edu)

Arnaud Couairon - Professor & Project Supervisor - Email: [arnaud.couairon@polytechnique.edu](arnaud.couairon@polytechnique.edu)