# NenuPlot V4 README

## Overview

NenuPlot is a tool designed to assist in the visualization and analysis of PSRFITS folded files. It provides a quick-look generation in PDF and PNG formats and offers numerous options for data handling, including cleaning, rebinding, RM and DM fitting, and data extraction within specified frequency and time ranges.

## Features

- Generation of quick-look files in PDF/PNG format
- Fits for Rotation Measure (RM) and Dispersion Measure (DM)
- Automatic rebinning for optimal data visualization
- Metadata storage in a tiny database
- Data extraction by time and frequency
- Data cleaning and interference mitigation
- Data scrunching in time and frequency domains
- Mail sending capabilities with custom titles
- Command-line and GUI modes
- Data dedispersion and defaraday options

## Features in progress an TODO

- Uploading capabilities for metadata, masks, archives, and quick-look files 
- Random bug with time scrunch after cleaning
- build a setup.py + requirement.txt

## Installation

To install NenuPlot, clone the repository or download the source code to your local machine. Make sure you have Python and the necessary dependencies installed.

```
git clone https://github.com/louisbondonneau/NenuPlot.git
cd NenuPlot
```

Follow any additional installation instructions provided by the repository.

## Usage

NenuPlot can be used with a series of command-line arguments to customize the processing and output of the PSRFITS data files. Below is a general usage pattern:

```
NenuPlot [options] INPUT_ARCHIVE
```

`INPUT_ARCHIVE` should be the path to the file or files you wish to process.

### Command-line Arguments

#### Required Arguments:
- `INPUT_ARCHIVE`: Path to the Archives to be processed.

#### Optional Arguments:
- `-h, --help`: Show the help message and exit.
- `-u PATH`: Specify the output path (default: current directory).
- `-o NAME`: Set the output name (default: first archive name).
- `-v`: Enable Verbose mode.
- `-version`: Show the program's version number and exit.
- `-gui`: Open the matplotlib graphical user interface.
- Cleaning options (`-clean`, `-noclean`).
- DM fitting options (`-fit_DM`, `-nofit_DM`).
- RM fitting options (`-fit_RM`, `-nofit_RM`).
- Rebinning options (`-autorebin`, `-noautorebin`).
- Database options (`-database`, `-nodatabase`).
- Mail options (`-mail SENDMAIL`, `-mailtitle MAILTITLE`).
- Time and frequency extraction options (`-mintime`, `-maxtime`, `-minsub`, `-maxsub`, `-minfreq`, `-maxfreq`).
- Mask and RM input options (`-mask_input MASK_INPUT`, `-RM_input RM_INPUT`).
- Scrunching options before and after CoastGuard cleaning (`-b`, `-t`, `-f`, `-ba`, `-ta`, `-fa`).
- Output options for archives, metadata, RM files, masks, PDFs, and PNGs.
- Visual options (`-Coherence`, `-timepolar`, `-timenorme`, `-threshold`).
- Cleaning parameters (`-force_niter`, `-fast`, `-chanthresh`, `-subintthresh`, `-first_chanthresh`, `-first_subintthresh`, `-bad_subint`, `-bad_chan`).
- Dedispersion and defaraday options (`-dm`, `-rm`, `-defaraday`, `-nodefaraday`).
- Uploading options for various outputs.

## Examples

Here is an example command that processes an input archive, fits for a new DM, and outputs a PDF quick-look:

```
NenuPlot -fit_DM -PDF_out -u /path/to/output/ /path/to/input/archive
```

To get a full list of options with explanations and default values, run:

```
NenuPlot -h
```

## Output

Depending on the options chosen, NenuPlot will generate the specified output files in the designated directory. This can include cleaned archives, metadata, RM files, masks, and visual representations of the data in PDF or PNG format.

## Contributing

Contributions to NenuPlot are welcome. Please follow the repository's guidelines for contributing, which may include coding standards, commit message formatting, and other best practices.