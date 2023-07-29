# kdpeak

kdpeak is a Python package designed to identifying genomic peaks from genomic reads in bed format using kernel density estimation (KDE).

## Getting Started

### Installation

Clone the repository, navigate to the directory, and install the package using pip:

```bash
git clone https://github.com/settylab/kdpeak.git
cd kdpeak
pip install .
```

Alternatively, you can install the package directly from GitHub:

```bash
pip install git+https://github.com/settylab/kdpeak.git
```

## Usage

kdpeak processes a bed file, applies KDE, identifies peaks based on the KDE, and writes the result to an output file. It provides the flexibility to customize your analysis by adjusting various parameters, including KDE bandwidth, logging level, sequence blacklist, and others. It is specifically tailored to provide a specified fraction-in-peaks (FRiP) that defaults to 0.3.

Basic usage:

```bash
kdpeak reads.bed --frip 0.3 --out peaks.py
```

To explore the complete list of parameters, use the `-h` or `--help` flag:

```bash
kdpeak --help
```

Here's an example with all available options:

```bash
kdpeak reads.bed --out output_directory --log DEBUG --logfile debug.log --blacklisted-seqs chr1 chr2 --kde-bw 500 --min-peak-size 50 --frip 0.5 --span 5
```

## Disclaimer

Please note that kdpeak is currently in its Alpha stage. We recommend using it with caution and welcome users to report any issues encountered during use.