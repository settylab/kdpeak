# kdpeak

[![DOI](https://zenodo.org/badge/672132987.svg)](https://zenodo.org/badge/latestdoi/672132987)

kdpeak is a Python package designed to identifying genomic peaks from genomic reads in bed format using kernel density estimation (KDE).

## Installation

Install via PyPI with:

```bash
pip install kdpeak
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/settylab/kdpeak.git
```

## Using kdpeak

kdpeak allows for processing of a bed file. It applies KDE, pinpoints peaks based on the fragemnt end density, and records the results in an output bed file.
The package enables customization of parameters like KDE bandwidth, sequence blacklist, minimum peak size, and others. Designed to deliver a specific fraction-in-peaks (FRiP), it defaults to 0.3.

Elementary usage:

```bash
kdpeak reads.bed --frip 0.3 --out peaks.bed
```

## Parameters

```bash
usage: kdpeak [-h] [--out OUTPUT_FILE] [-l LEVEL] [--logfile LOGFILE] [--blacklisted-seqs chrN [chrN ...]] [--kde-bw FLOAT] [--min-peak-size INT] [--fraction-in-peaks FLOAT] [--span INT] READS.BED
```

**Positional Argument:**

- `reads.bed` - Path to the bed file containing the genomic reads.

**Options:**

- `-h, --help` - Show this help message and exit.
- `--out OUTPUT_FILE` - Path to the output file to save the results. Default is ./peaks.bed.
- `-l LEVEL, --log LEVEL` - Set the logging level. Options include: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.
- `--logfile LOGFILE` - Path to the file to write a detailed log.
- `--blacklisted-seqs chrN [chrN ...]` - List of sequences (e.g., chromosomes) to exclude from peak calling. Input as space-separated values.
- `--kde-bw FLOAT` - Bandwidth (standard deviation, sigma in base pairs) for the KDE. Increase for larger features to reduce noise. Default is 200.
- `--min-peak-size INT` - Minimal size (in base pairs) for a peak to be considered valid. Default is 100.
- `--fraction-in-peaks FLOAT, --frip FLOAT` - Expected fraction of total reads to be located in peaks. Default is 0.3.
- `--span INT` - Resolution of the analysis in base pairs, determining the granularity of the KDE and peak calling. Default is 10.

## Utilizing All Available Options:

```bash
kdpeak reads.bed --out peaks.bed --log DEBUG --logfile debug.log --blacklisted-seqs chr1 chr2 --kde-bw 500 --min-peak-size 50 --frip 0.5 --span 5
```

## Disclaimer

kdpeak, being in its Alpha stage, encourages usage with care. We warmly welcome users to report any issues experienced during utilization. Together, we can enhance kdpeak for a better genomic analysis experience.
