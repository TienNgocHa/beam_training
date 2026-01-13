# beam_training

Quick start
 
1. Install dependencies from `pyproject.toml` (using `pip` or `uv`).
   - With pip:

	pip install .

   - Or, if you have `uv` installed, sync dependencies with:
```bash
	uv sync
```
2. Initialize and update the `gsim` submodule:
```bash
	cd gsim
	git submodule init
	git submodule update
	cd ..
```
3. Run the GSIM installer script (this will create `gsim_conf.py`):
```bash
	bash gsim/install.sh
```
4. Edit `gsim_conf.py` and set the module name to the beamforming experiments module:

	module_name = "experiments.beamforming_experiments"

More information about GSIM and its configuration can be found at:
https://github.com/fachu000/GSim-Python

Running an experiment

To run an experiment, call `run_experiment.py` with the experiment number. Example:
```bash
    python run_experiment.py 1001
```
Citation

If this repository or the results are helpful, please cite:

T. N. Ha, D. Romero and R. López-Valcarce, "Radio Maps for Beam Alignment in mmWave Communications with Location Uncertainty," 2024 IEEE 99th Vehicular Technology Conference (VTC2024-Spring), Singapore, Singapore, 2024, pp. 1-7, doi: 10.1109/VTC2024-Spring62846.2024.10683362.
