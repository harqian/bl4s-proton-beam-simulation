# bl4s_sim

simple proton-shield simulation for a water/bismuth shielding concept.

## experiment idea

the project idea is to compare several water/bismuth shield trials in front of a proton beam and measure how much energy gets through the shield. the mixed shields are intended to test whether water plus bismuth can act as a better practical barrier than a pure high-z material alone.

the motivating idea is:

- water is a useful low-z material for moderating charged-particle energy deposition
- bismuth is a dense high-z material that can increase attenuation without the toxicity of lead
- comparing different bismuth loadings gives a way to test how composition changes transmitted beam energy and heating inside the shield

## what it does

`sim.py` models:

- proton energy loss through a 1d shield
- heat deposition and diffusion in the shield over time
- comparison plots for the trial compositions defined in the script

## connection to the simulation

the simulation is a simplified version of the experiment, not a full beam-transport model.

- it represents the shield as a 1d stack of water and bismuth cells
- it uses a bethe-bloch stopping model for the primary proton energy loss
- it converts deposited energy into a heat source and evolves temperature with a 1d diffusion solver
- it generates comparison plots for the trial compositions listed in the script

the simulation is most useful for comparing relative trends between trial compositions:

- how quickly the primary proton loses energy with depth
- where heat is deposited in the shield
- how the final temperature profile changes across trials

it does not currently model a full secondary-radiation cascade, detector response, or a detailed 3d suspension geometry.

## requirements

- python 3.13+
- `uv`

dependencies are listed in [pyproject.toml](/Users/hq/code/bl4s_sim/pyproject.toml).

## run

```bash
uv run python sim.py
```

## outputs

running the script writes:

- `simulation_results.png`
- `energy_remaining_by_trial.png`
- `temperature_rise_by_trial.png`

the script also prints the grid size, timestep, final average temperature, and final exit proton energy.
