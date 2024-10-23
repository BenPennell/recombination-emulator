# Emulating Recombination with Neural Networks using Universal Differential Equations

*We are in the era of precision cosmology, where extremely constraining datasets provide insights
into a cosmological model dominated by unknown contents. Cosmic Microwave Background (CMB)
data in particular provide a clean glimpse into the interaction of dark matter, baryons, and radi-
ation in the early Universe, but interpretation of this data requires the evolution of the ionization
history of the Universe. Testing new physics with improved CMB data will require fast and flexible
calculation of the ionization history. Existing methods to compute the ionisation history, such as
RECFAST, are already emulators for more complicated atomic physics, but they are highly tuned
to the standard model of cosmology. We develop a differentiable machine learning model for recom-
bination physics using a neural network ordinary differential equation model (Universal Differential
Equations, UDEs).*

This repository contains the code used for training the emulator described in our paper, it is left here for reproducibility and for the interest of anyone looking into using it for future science.

The code was written using the [SciML](https://github.com/SciML/) machine learning ecosystem.

## Setup
`Manifest.toml` and `Project.toml` contain everything needed to create a julia environment with the required packages. I made it in Julia version 1.9.2. Here are the steps to get the code working:

- [download julia](https://julialang.org/) and clone this repository.
- Navigate to the directory one above the repository directory and activate the environment. Here's what it should look like:
```
$ julia
$ ]
(@v1.9) pkg> activate recombination-emulator
  Activating new project at `c:/git/recombination-emulator`

(recombination-emulator) pkg> instantiate
  No Changes to `c:/git/recombination-emulator/Project.toml`
  No Changes to `c:/git/recombination-emulator/Manifest.toml`
```
- Run `\code\UseScript.jl` by copying the text from `\code\UseExample.pbs` and see if it makes a plot without any errors

## Training
`\code\TrainingScript.jl` trains the neural network and takes inputs for a variety of network training options and architecture.

`\code\TrainingExample.jl` contains an exmaple for running the training script from the command line, the exact command used for the training of the network used in the paper. Barring differences in random number generation, this should train the same network as the one used in the results of the paper.

`\code\network_parameters\parameters` contains the parameters to the neural network used in the results of the paper.
 
## Using the Network
`\code\UseScript.jl` takes in the values of the variable parameters and plots the corresponding network output from the network with a given set of network parameters

`\code\UseExample.jl` contains an exmaple for running the use script from the command line, which plots the result of putting the Planck 2018 best fits through the exact network used for the results in the paper

## Data
`\code\data\Data.json` contains the training data, created from hyrec, that was used in the training of this network. It is a dictionary that contains `training_set` of 48 sample ionization histories, and a `test_set` with 17 additional ionization histories