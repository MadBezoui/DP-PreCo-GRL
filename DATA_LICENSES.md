# Licences and provenance of the data

The code of this repository is released under the MIT licence (see `LICENSE`). The data in `data/`
come from third parties and keep their own terms.

| Path | Content | Source and terms |
|---|---|---|
| `data/energy/` | Raw responses of the Energy-Charts API for the DE-LU bidding zone on 28 September 2026 (15-min day-ahead prices, solar generation, grid CO2-equivalent intensity), retrieved on the operating day at 22:00 | Fraunhofer Institute for Solar Energy Systems ISE, Energy-Charts (https://api.energy-charts.info). Prices: Bundesnetzagentur / SMARD.de, licence CC BY 4.0. Please attribute the providers and check the terms at the source. |
| `data/fjsp/` | Flexible job-shop benchmark instances Mk01 to Mk10 (Brandimarte) and La16, La21 (Hurink edata), in the usual text format with 0-based machine ids | Compiled by the maintainers of SchedulingLab/fjsp-instances (https://github.com/SchedulingLab/fjsp-instances, MIT licence, copyright 2022 SchedulingLab), which in turn cite the original papers: Brandimarte, Annals of Operations Research 41 (1993), doi:10.1007/BF02023073, and Hurink, Jurisch and Thole, OR Spektrum 15 (1994), doi:10.1007/BF01719451. The original publications state no redistribution licence and we are not aware of one, so the MIT licence of the compilation is the only explicit term. Please cite the original papers. `data/fjsp/SHA256SUMS` lists the checksums of the files as redistributed here. If redistribution of the raw files is a concern, delete `data/fjsp/`, download the instances from the compilation, the checksums documenting the exact files used here. |
| `results/` | Outputs computed by the code of this repository (trained checkpoints, raw objective vectors, summaries, logs) | MIT licence, as the code |

Machine powers, fuzzy-duration spreads, capacities and time scales of the benchmark instances are
synthetic annotations generated deterministically by `morl_fjsp/instance.py`. They are not part of the
original benchmarks.
