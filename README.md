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
usage: kdpeak [-h] [--out OUTPUT_FILE] [--summits-out SUMMITS_FILE] [--density-out DENSITY_FILE] [--chrom-sizes CHROM_SIZES] [-l LEVEL] [--logfile LOGFILE] [--blacklisted-seqs chrN [chrN ...]] [--kde-bw FLOAT] [--min-peak-size INT] [--fraction-in-peaks FLOAT] [--span INT] READS.BED
```

**Positional Argument:**

- `reads.bed` - Path to the bed file containing the genomic reads.

**Options:**

- `-h, --help` - Show this help message and exit.
- `--out output_file.bed` - Path to the output file where the results will be saved.
  Peaks are saved in bed format with the columns:
  start, end, peak name, AUC (area under the cut density curve
  where cut-density is in cuts per 100 base pairs). Defaults to peaks.bed.
- `--summits-out summits_file.bed` - Path to the output file where the peak summits will be saved.
  The file will have columns for start, end (start+1),
  peak name, and summit height (in cuts per 100 base pairs).
  If nothing is specified the summits will not be saved.
- `--density-out density_file.bw` - Path to the output file where the event density will be saved.
  The event density is the internally computed signal on which the peaks
  are called based on a cutoff. It will be saved in the bigwig format for visualization
  in genome browsers like IGV or JBrowse.
- `--chrom-sizes chrom-sizes-file` - Chromosome sizes file with the two columns: seqname and size.
  This file is only needed if --density-out is specified.
  You can use a script like https://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/fetchChromSizes
  to fetch the file.
- `-l LEVEL, --log LEVEL` - Set the logging level. Options include: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.
- `--logfile LOGFILE` - Path to the file to write a detailed log.
- `--blacklisted-seqs chrN [chrN ...]` - List of sequences (e.g., chromosomes) to exclude from peak calling. Input as space-separated values.
- `--kde-bw FLOAT` - Bandwidth (standard deviation, sigma in base pairs) for the KDE. Increase for larger features to reduce noise. Default is 200.
- `--min-peak-size INT` - Minimal size (in base pairs) for a peak to be considered valid. Default is 100.
- `--fraction-in-peaks FLOAT, --frip FLOAT` - Expected fraction of total reads to be located in peaks. Default is 0.3.
- `--span INT` - Resolution of the analysis in base pairs, determining the granularity of the KDE and peak calling. Default is 10.

## Utilizing All Available Options:

```bash
kdpeak reads.bed --out peaks.bed --summits-out summits.bed --density-out density.bw --chrom-sizes hg38.chrom.sizes --log DEBUG --logfile debug.log --blacklisted-seqs chr1 chr2 --kde-bw 500 --min-peak-size 50 --frip 0.5 --span 5
```

## BigWig Output for Visualization

kdpeak can export the computed kernel density estimation as BigWig files for visualization in genome browsers:

```bash
# Basic usage with BigWig output
kdpeak reads.bed --out peaks.bed --density-out density.bw --chrom-sizes hg38.chrom.sizes

# Get chromosome sizes file (example for hg38)
fetchChromSizes hg38 > hg38.chrom.sizes
```

The BigWig file contains the event density values (in cuts per 100 base pairs) that are used internally for peak calling. These files can be loaded into genome browsers like IGV or JBrowse to visualize the underlying signal alongside the called peaks.

## BigWig Operations Tool (bwops)

The `bwops` utility provides mathematical operations and regression analysis on BigWig files with comprehensive chromosome filtering capabilities:

### Basic Operations

```bash
# Add multiple BigWig files
bwops add file1.bw file2.bw file3.bw --out sum.bw --chrom-sizes hg38.chrom.sizes

# Add files excluding contigs and mitochondrial DNA
bwops add file1.bw file2.bw --out sum.bw \
      --exclude-contigs --blacklisted-seqs chrM \
      --chrom-sizes hg38.chrom.sizes

# Multiply BigWig files (convolution) for human autosomes only
bwops multiply signal1.bw signal2.bw --out product.bw \
      --chromosome-pattern "chr[0-9]+$" \
      --chrom-sizes hg38.chrom.sizes

# Output to different formats
bwops add file1.bw file2.bw --out results.csv --format csv
```

### Regression Analysis

The `bwops regress` command provides flexible regression analysis with support for complex file paths and easy-to-read formulas:

```bash
# Simple regression with auto-generated formula (target ~ a + b)
bwops regress --target "signal.bw" \
              --predictors "control.bw" "treatment.bw" \
              --out-prediction predictions.bw \
              --chrom-sizes hg38.chrom.sizes

# Named variables with custom formula for clarity
bwops regress --target "response=ChIP_signal.bw" \
              --predictors "input=input_control.bw" "dnase=DNase_signal.bw" \
              --formula "response ~ input + dnase + input*dnase" \
              --out-prediction predictions.bw \
              --out-residuals residuals.bw \
              --out-stats stats.json \
              --chrom-sizes hg38.chrom.sizes

# Complex file paths with spaces (no parsing issues)
bwops regress --target "signal=/path/with spaces/signal file.bw" \
              --predictors "ctrl=/data/control samples/ctrl.bw" \
              --formula "signal ~ ctrl" \
              --format csv --out results.csv

# Regression excluding problematic sequences
bwops regress --target "target.bw" --predictors "predictor.bw" \
              --exclude-contigs --blacklisted-seqs chrM chrY \
              --chromosome-pattern "chr[0-9XY]+$" \
              --out-prediction predictions.bw \
              --chrom-sizes hg38.chrom.sizes

# Logistic regression with mixed naming
bwops regress --target "binary_target.bw" \
              --predictors "pred1=predictor1.bw" "predictor2.bw" \
              --formula "target ~ pred1 + a"  # a = predictor2.bw \
              --type logistic \
              --out-prediction logistic_pred.bw \
              --chrom-sizes hg38.chrom.sizes
```

#### Variable Naming System

- **Explicit naming**: `"varname=file.bw"` creates variable `varname` for `file.bw`
- **Default naming**: Just `"file.bw"` uses `target` for target, `a`, `b`, `c`... for predictors
- **Auto-generated formulas**: If no `--formula` provided, generates `target ~ a + b + c`
- **Clean formulas**: Use variable names instead of file paths for readability

#### Required Arguments

- `--target`: Target variable file with optional name (`target=file.bw` or `file.bw`)
- `--predictors`: One or more predictor files with optional names (`pred1=file1.bw pred2=file2.bw`)

#### Optional Arguments

- `--formula`: R-style formula using variable names (auto-generated if omitted)
- `--type`: Regression type (`linear` or `logistic`, default: `linear`)
- `--out-prediction`: Output file for predictions
- `--out-residuals`: Output file for residuals  
- `--out-stats`: Output file for detailed statistics (JSON format)

### Filtering Options

All bwops operations support comprehensive chromosome filtering (consistent with kdpeak):

- `--blacklisted-seqs`: Exclude specific chromosomes (e.g., `chrM chrY`)
- `--exclude-contigs`: Automatically exclude contigs/scaffolds  
- `--chromosome-pattern`: Include only chromosomes matching regex pattern
- `--chromosomes`: Analyze only specified chromosomes
- `--region`: Limit analysis to genomic region (chr:start-end)

### Additional Options

- `--span INT`: Resolution in base pairs (default: auto-detect from BigWig files)
- `--format`: Output format (bigwig, csv, tsv, bed, json)

The regression analysis operates across all chromosomes simultaneously and prints summary statistics including R², p-values, and fitted coefficients to stdout. It can output predictions, residuals, and detailed statistics to separate files.

### Correlation Analysis

The `bwops correlate` command computes pairwise correlation matrices between multiple BigWig files with support for different correlation methods and analysis scopes:

```bash
# Basic pairwise correlation analysis
bwops correlate file1.bw file2.bw file3.bw --out correlations.csv --chrom-sizes hg38.chrom.sizes

# Spearman correlation with per-chromosome analysis
bwops correlate *.bw --out spearman_corr.csv \
      --method spearman --scope per-chromosome \
      --chrom-sizes hg38.chrom.sizes

# Correlation with visualization and detailed statistics
bwops correlate ChIP1.bw ChIP2.bw Control1.bw Control2.bw \
      --out correlation_matrix.csv \
      --method pearson \
      --include-stats \
      --heatmap correlation_heatmap.png \
      --scatter-plots scatter_plots/ \
      --chrom-sizes hg38.chrom.sizes

# Correlation analysis excluding problematic regions
bwops correlate signal1.bw signal2.bw signal3.bw \
      --out filtered_correlations.tsv --format tsv \
      --exclude-contigs --blacklisted-seqs chrM chrY \
      --chromosome-pattern "chr[0-9XY]+$" \
      --min-overlap 5000 \
      --chrom-sizes hg38.chrom.sizes

# Regional correlation analysis
bwops correlate enhancer_marks.bw promoter_marks.bw \
      --out region_correlation.json --format json \
      --region chr1:1000000-2000000 \
      --method kendall \
      --chrom-sizes hg38.chrom.sizes
```

#### Correlation Methods

- **Pearson**: Linear correlation coefficient (default) - measures linear relationships
- **Spearman**: Rank-based correlation - robust to outliers and non-linear monotonic relationships  
- **Kendall**: Tau correlation coefficient - alternative rank-based method, more robust for small samples

#### Analysis Scope

- **Global** (default): Compute single correlation matrix using all genomic positions across chromosomes
- **Per-chromosome**: Compute separate correlation matrices for each chromosome

#### Required Arguments

- **Input files**: Two or more BigWig files for correlation analysis
- `--out`: Output file for correlation matrix
- `--chrom-sizes`: Chromosome sizes file for BigWig processing

#### Optional Arguments

- `--method`: Correlation method (`pearson`, `spearman`, `kendall`)
- `--scope`: Analysis scope (`global`, `per-chromosome`) 
- `--format`: Output format (`csv`, `tsv`, `json`)
- `--min-overlap`: Minimum number of overlapping non-zero values required (default: 1000)
- `--include-stats`: Include additional statistics (mean, std, coverage) in output
- `--heatmap`: Generate correlation heatmap plot (requires matplotlib)
- `--scatter-plots`: Directory for pairwise scatter plots

#### Output Formats

The correlation analysis outputs a symmetric matrix showing correlations between all file pairs:

**CSV/TSV format:**
```csv
File1,File2,File3
File1,1.000,0.847,0.623
File2,0.847,1.000,0.751
File3,0.623,0.751,1.000
```

**JSON format** (with `--include-stats`):
```json
{
  "correlations": {
    "File1_vs_File2": 0.847,
    "File1_vs_File3": 0.623,
    "File2_vs_File3": 0.751
  },
  "statistics": {
    "File1": {"mean": 2.34, "std": 1.45, "coverage": 0.23},
    "File2": {"mean": 1.89, "std": 1.12, "coverage": 0.31}
  }
}
```

**Per-chromosome analysis** includes separate correlation matrices for each chromosome, helping identify chromosome-specific correlation patterns.

## Disclaimer

kdpeak, being in its Alpha stage, encourages usage with care. We warmly welcome users to report any issues experienced during utilization. Together, we can enhance kdpeak for a better genomic analysis experience.
