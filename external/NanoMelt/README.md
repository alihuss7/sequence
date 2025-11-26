# NanoMelt: semi-supervised ensemble model for nanobody thermostability prediction

</div>

## License

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (see License file).
This software is not to be used for commerical purposes.

## Reference

> Original publication (mAbs 2025): https://doi.org/10.1080/19420862.2024.2442750


## Presentation

NanoMelt is a semi-supervised ensemble model trained to predict the apparent
melting temperatures of nanobodies. 

The model involves a Ridge stacking of multiple Gaussian Process Regressors (GPRs)
and Suppot Vector Regressors (SVRs) combined with diverse pre-trained nanobody embeddings 
(i.e., one-hot, VHSE, ESM-1b, and ESM-2).

* NanoMelt can be used via:
    - `nanomelt predict`: to directly predict the apparent melting temperatures of given nanobodies, or
    - `nanomelt train`: to retrain a custom model with your own dataset. 

<strong>A webserver for nanobody thermostability prediction is available at https://www-cohsoftware.ch.cam.ac.uk/</strong>

You can find the parsed datasets of nanobody thermostabily in the `/datasets` directory. 
All Jupyter notebooks used to create the figures in the original publication are available in the `/notebooks` directory. 

## Setup NanoMelt 

> **Compatible with python 3.12 (>=3.8)**

We recommend running NanoMelt on a `GPU` for optimal performance, as it relies on ESM models to generate input embeddings.

**Step 1. Ensure all conda dependencies are installed**

```bash
conda install -c conda-forge openmm pdbfixer biopython
```

**Step 2. Install ANARCI**

```bash
# For Linux (x86_64)
conda install -c bioconda anarci

# For Apple Silicon (arm64), due to limited support, the following steps are recommended
conda install -c biocore hmmer
git clone https://github.com/oxpig/ANARCI.git
cd ANARCI
python setup.py install
conda install nomkl #Optional, for compatibility with numpy
```

**Step 3. Install NanoMelt from PyPI**
```
# Install from PyPI
pip install nanomelt
```

## NanoMelt command-line interface

### 1 - Prediction of nanobody apparent melting temperatures

To predict the apparent melting temperature of nanobody sequences, use the `nanobody predict` command line. Parallelisation can be applied by specifying the number of cpus `-ncpu`. NanoMelt will automatically search for CUDA or Apple MX GPU to accelerate the embedding step. 

<details>
    <summary>See <strong>nanomelt predict</strong> command line description</summary>

```
nanomelt predict [-h] [-i INPUT_FILEPATH_OR_SEQ] [-o OUTPUT_SAVEFP] [-align] [-ncpu NCPU]

Predict nanobody apparent melting temperatures with NanoMelt

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILEPATH_OR_SEQ, --input_filepath_or_seq INPUT_FILEPATH_OR_SEQ
                        Filepath to the fasta file .fa to score or directly a single string sequence (default: to_score.fa)
  -o OUTPUT_SAVEFP, --output_savefp OUTPUT_SAVEFP
                        Filename of the .csv file to save the predictions in (default: nanomelt_run.csv)
  -align, --do_align    Do the alignment and the cleaning of the given sequences before training. This step can takes a lot of time if the number of
                        sequences is huge. (default: False)
  -ncpu NCPU, --ncpu NCPU
                        If ncpu>1 will parallelise the alignment process (default: 1)
```
</details>


\
Testing files are presented in `/test`, with examples of output files.

Examples of `nanomelt predict` usage:

```bash
# For one single sequence
nanomelt predict -i QVQLVESGGGLVQAGGSLRLSCAASGYIFGRNAMGWYRQAPGKERELVAGITRRGSITYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAADPASPAYGDYWGQGTQVTVSS -align

# Predict the apparent melting temepatures for a set of sequences in a fasta file
nanomelt predict -i test/application6.fa  -o test/NanoMeltrun_application6.csv -align
```

Additionally, NanoMelt can be used directly via its in-built function `NanoMeltPredPipe()`. It takes as inputs a list of SeqRecords (seq_records, see BioPython). 
See description of function for further details. For instance:

```bash
# Import
from nanomelt.model.nanomelt import NanoMeltPredPipe

# Run the model
data = NanoMeltPredPipe(seq_records, do_align=True, ncpus=4, verbose=True)
```

### 2 - Re-training NanoMelt

To re-train NanoMelt on a custom input dataset of nanobody sequences with their respective melting temperature, use the `nanomelt train` command line.
Be careful with the format of your input file (see command line description).

<details>
    <summary>See <strong>nanomelt train</strong> command line description</summary>

```
nanomelt train [-h] [-i INPUT_CSV] [-od OUTPUT_DIR] [-align] [-ncpu NCPU]

Train NanoMelt with a new input dataset.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_CSV, --input_csv INPUT_CSV
                        Filepath to the .csv file of the nanobody dataset. It needs to have at least columns with Id, Sequence, Experimental method
                        and Measured apparent melting temperature (C). (default: new_nano_thermostability_dataset.csv)
  -od OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Filepath to the output dir where to save the saved models (default: nanomelt_retrain)
  -align, --do_align    Do the alignment and the cleaning of the given sequences before training. This step can takes a lot of time if the number of
                        sequences is huge. (default: False)
  -ncpu NCPU, --ncpu NCPU
                        If ncpu>1 will parallelise the algnment process (default: 1)
```

</details>

Example of usage of `nanomelt train`:

```bash
# Train.
nanomelt train -i datasets/NanoMelt_640_nanobody_apparent_melting_temperatures.csv -od test/nanomelt_retrain -align -ncpu 4
```

## Issues

- The installation of OpenMM might create troubles with your device. If you have an `import error` with `lib glibxx_3.4.30`, you could solve it with `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` (issue due to AbNatiV installation).

If you experience any issues please add an issue to the [Gitlab] or contact directly ar2033@cam.ac.uk.

## Contact

Please contact ar2033@cam.ac.uk to report issues of for any questions.


