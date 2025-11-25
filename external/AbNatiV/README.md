# AbNatiV: VQ-VAE-based assessment of antibody and nanobody nativeness for hit selection, humanisation, and engineering

</div>

## License

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (see License file).
This software is not to be used for commerical purposes.

## References

> <strong>AbNatiV</strong> (original publication): 
https://www.nature.com/articles/s42256-023-00778-3

> <strong>AbNatiV2</strong> (pre-print): 
https://www.biorxiv.org/content/10.1101/2025.10.31.685806v1 


## Datasets
The datasets used for training and testing are available at https://zenodo.org/records/17466150 

## Presentation

> <strong>UPDATES (Oct 25)</strong>:\
   &emsp;&emsp; 1. AbNatiV2 unpaired and paired models are now available (scoring and humanisation),\
   &emsp;&emsp; 2. Paired humanisation with p-AbNativ2,\
   &emsp;&emsp; 3. Automatically computes CDR-displacement upon humanisation and generate a ChimeraX session,\
   &emsp;&emsp; 4. It is now compatible with python 3.12,

AbNatiV is a deep-learning tool for assessing the nativeness of antibodies and nanobodies,
i.e., their likelihood of belonging to the distribution
of immune-system derived human antibodies or camelid nanobodies,
AbNatiV is a deep-learning tool for assessing the nativeness of antibodies and nanobodies,
i.e., their likelihood of belonging to the distribution
of immune-system derived human antibodies or camelid nanobodies,
which can be exploited to guide antibody engineering and humanisation.

The model is a vector-quantized variational auto-encoder (VQ-VAE) that generates
an interpretable nativeness score
and a residue-level nativeness profile for a given input sequence.
The model is a vector-quantized variational auto-encoder (VQ-VAE) that generates
an interpretable nativeness score
and a residue-level nativeness profile for a given input sequence.

AbNatiV2 incorporates architectural updates and was trained on more than 20M sequences. 
It also comprises a paired model p-AbNatiV2, trained on ~4M paired sequences via contrastive learning to 
assess the pairing likelihood of a given VH/VL Fv pair.

* AbNatiV provides a <strong><ins>nativeness score</ins></strong> for each of its 4 default training datasets:\
   &emsp;&emsp; 1. `VH`: human immune-system derived heavy chains,\
   &emsp;&emsp; 2. `VKappa`: human immune-system derived kappa light chains,\
   &emsp;&emsp; 3. `VLambda`: human immune-system derived lambda light chains,\
   &emsp;&emsp; 4. `VHH`: camelid immune-system derived single-domain antibody sequences.

* To use AbNatiV2 (<strong><ins>unpaired</ins></strong>), employ: `VH2`, `VL2`, and `VHH2`.

* To use p-AbNatiV2 (<strong><ins>paired</ins></strong>), employ the new in-built function `abnativ_scoring_paired` or <paired_score> command line.

* AbNatiV (v1 and v2) can additionally be used to <strong><ins>humanise</ins></strong> Fv sequences (nanobodies and paired VH/VL):\
  &emsp;&emsp; 1. <strong>nanobodies</strong>: it employs a dual-control strategy aiming to increase the humanness of the sequence without decreasing its initial VHH-nativenees,\
  &emsp;&emsp; 2. <strong>VH/VL Fvs</strong>:\
  &emsp;&emsp;&emsp;&emsp; 2.1 <strong> using the unpaired models</strong>: it directly increases the VH-humanness and VL-humanneess of both sequences.\
  &emsp;&emsp;&emsp;&emsp; 2.2 <strong>using the paired model</strong>: it employs a dual-control strategy aiming to increase the paired humanness and the pairing likelihood of the Fv.

<strong>A web server for scoring is available at https://www-cohsoftware.ch.cam.ac.uk/index.php/abnativ</strong>

## Setup AbNatiV

> **Compatible with python 3.12 (>=3.8)**

We recommend running AbNatiV on a `GPU` for optimal performance.

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
cd ..
```

**Step 3. Install AbNatiV**
```bash
# Install from pypi
pip install abnativ --upgrade

# Download model weights
abnativ init 
```


## AbNatiV command-line interface

### 1.1 - Antibody nativeness scoring (unpaired v1 and v2)

To score input antibody sequences, use the `abnativ score` command line. You can plot nativeness profiles using the `-plot` option.

AbNatiV provides an interpretable overall nativeness score, which approaches 1 for highly native sequences and where 0.8 represents the threshold that best separates native from non-native sequences. This score is computed for the whole Fv sequence, but can also be computed for individual CDRs or framework region (closest to 1, highest nativeness).

NB: Input antibody sequences need to be aligned to be processed by AbNatiV (AHo scheme). AbNatiV can directly align them with the option `-align`. If working with nanobodies, precise `-isVHH`, it considers the VHH seed for the alignment. `-align` and `-plot` will slow down the scoring.

<details>
    <summary>See <strong>abnativ score</strong> command line description</summary>

```
abnativ score [-h] [-nat NATIVENESS_TYPE] [-mean] [-i INPUT_FILEPATH_OR_SEQ] [-odir OUTPUT_DIRECTORY] [-oid OUTPUT_ID] [-align] [-ncpu NCPU] [-isVHH] [-plot]

Use a trained AbNatiV model (default or custom) to score a set of input antibody sequences

optional arguments:
  -h, --help            show this help message and exit
  -nat NATIVENESS_TYPE, --nativeness_type NATIVENESS_TYPE
                        To load the AbNatiV default trained models type VH, VKappa, VLambda, or VHH, otherwise add directly the path to your own AbNatiV trained
                        checkpoint .ckpt (default: VH)
  -mean, --mean_score_only
                        Generate only a file with a score per sequence. If not, generate a second file with a nativeness score per positin with a probability
                        score for each aa at each position. (default: False)
  -i INPUT_FILEPATH_OR_SEQ, --input_filepath_or_seq INPUT_FILEPATH_OR_SEQ
                        Filepath to the fasta file .fa to score or directly a single string sequence (default: to_score.fa)
  -odir OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Filepath of the folder where all files are saved (default: abnativ_scoring)
  -oid OUTPUT_ID, --output_id OUTPUT_ID
                        Prefix of all the saved filenames (e.g., name sequence) (default: antibody_vh)
  -align, --do_align    Do the alignment and the cleaning of the given sequences before scoring. This step can takes a lot of time if the number of sequences is
                        huge. (default: False)
  -ncpu NCPU, --ncpu NCPU
                        If ncpu>1 will parallelise the algnment process (default: 1)
  -isVHH, --is_VHH      Considers the VHH seed for the alignment. It is more suitable when aligning nanobody sequences (default: False)
  -plot, --is_plotting_profiles
                        Plot profile for every input sequence and save them in {output_directory}/{output_id}_profiles. (default: False)
```
</details>


\
Testing files are presented in `/test`, with examples of output files.

Examples of `abnativ score` usage:

```bash
# Align and Compute the AbNatiV VH-humanness scores (sequence and residue levels) for a set of sequences in a fasta file
# In directory test/test_scoring are saved test_vh_abnativ_seq_scores.csv and test_vh_abnativ_res_scores.csv
# Profile figures are saved in test/test_vh_profiles for each sequence
# To use AbNatiV2, employ: `VH2`, `VL2`, or `VHH2`.
abnativ score -nat VH2 -i test/4_heavy_sequences.fa -odir test/test_results2 -oid test_vh -align -ncpu 4

# For one single sequence
abnativ score -nat VH2 -i EIQLVQSGPELKQPGETVRISCKASGYTFTNYGMNWVKQAPGKGLKWMGWINTYTGEPTYAADFKRRFTFSLETSASTAYLQISNLKNDDTATYFCAKYPHYYGSSHWYFDVWGAGTTVTVSS -odir test/test_results2 -oid test_single_vh -align -plot
```

If you want to use your own trained model for scoring (see bellow `abnativ train`), precise the filepath to the .ckpt checkpoint file with the argument -m instead of the default parameters: VH, VKappa, VLambda or VHH. In that case, the scores won't be linearly rescaled as proposed in the default AbNatiV (see Methods paper). For instance:


```bash
# Align and nativeness scoring from a custom retrained AbNatiV model
abnativ score -nat my_trained_model.ckpt -i test/4_heavy_sequences.fa -odir test -oid test_vh -align -ncpu 4
```

Additionally, AbNatiV nativeness scoring can be used directly via its in-built function. It takes as inputs a list of SeqRecords (seq_records, see BioPython). For instance:

```bash
from abnativ.model.scoring_functions import abnativ_scoring

abnativ_scores_df, abnativ_profiles_df  = abnativ_scoring(model_type='VH',seq_records=seq_records, batch_size=128,
                                    mean_score_only=False, do_align=True, is_VHH=False, output_dir='test', 
                                    output_id='test_vh', run_parall_al=4)
```


### 1.2 - Antibody nativeness scoring (paired v2 only)

To score input antibody sequences, use the `abnativ paired_score` command line. You can plot nativeness profiles using the `-plot` option.

In addition to a hummanness paired score, the model gives a paired likelihood score (probability that the heavy and light chains form a compatible pair).

NB: Input antibody sequences need to be aligned to be processed by AbNatiV (AHo scheme). AbNatiV can directly align them with the option `-align`. `-align` and `-plot` will slow down the scoring.

<details>
    <summary>See <strong>abnativ paired_score</strong> command line description</summary>

```
abnativ paired_score [-h] [-i INPUT_FILEPATH_CSV] [-vh VH] [-vl VL]
                            [-cid COL_ID] [-cvh COL_VH] [-cvl COL_VL] [-mean] [-odir OUTPUT_DIRECTORY]
                            [-oid OUTPUT_ID] [-align] [-ncpu NCPU] [-plot]

Use the paired AbNatiV2 model to score a set of input antibody sequences

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILEPATH_CSV, --input_filepath_csv INPUT_FILEPATH_CSV
                        Filepath to the csv file .csv to score sequences. Leave empty if running directly
                        a sequence without a csv file. (default: to_score.fa)
  -vh VH, --vh VH       VH sequence to score directly. Leave empty if running directly a csv file.
                        (default: no_seq.)
  -vl VL, --vl VL       VL sequence to score directly. Leave empty if running directly a csv file.
                        (default: no_seq.)
  -cid COL_ID, --col_id COL_ID
                        Name of the ID column of the csv file. Leave empty if running directly a sequence
                        without a csv file. (default: ID)
  -cvh COL_VH, --col_vh COL_VH
                        Name of the vh column of the csv file. Leave empty if running directly a sequence
                        without a csv file. (default: vh_seq)
  -cvl COL_VL, --col_vl COL_VL
                        Name of the vl column of the csv file. Leave empty if running directly a sequence
                        without a csv file. (default: vl_seq)
  -mean, --mean_score_only
                        Generate only a file with a score per sequence. If not, generate a second file
                        with a nativeness score per positin with a probability score for each aa at each
                        position. (default: False)
  -odir OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Filepath of the folder where all files are saved (default: abnativ_scoring)
  -oid OUTPUT_ID, --output_id OUTPUT_ID
                        Prefix of all the saved filenames (e.g., name sequence) (default: antibody_vh)
  -align, --do_align    Do the alignment and the cleaning of the given sequences before scoring. This step
                        can takes a lot of time if the number of sequences is huge. (default: False)
  -ncpu NCPU, --ncpu NCPU
                        If ncpu>1 will parallelise the algnment process (default: 1)
  -plot, --is_plotting_profiles
                        Plot profile for every input sequence and save them in
                        {output_directory}/{output_id}_profiles. (default: False)
```
</details>


\
Testing files are presented in `/test`, with examples of output files.

Examples of `abnativ paired_score` usage:

```bash
# Align and Compute the AbNatiV humanness scores (sequence and residue levels) for a set of sequences in a csv file. 
# With at least three columns for the ID, the heavy sequence and the light sequence (arg.parse the column names).
abnativ paired_score -i test/4_paired_sequences.csv -cid 'ID' -cvh 'vh_seq' -cvl 'vl_seq' -odir test/test_paired_scoring2 -oid test_paired -align -ncpu 4

# For a VH/VL sequence
abnativ paired_score -vh QVQLVQSGAEVKKPGASVKVSCKVSGYTLSDLSIHWVRQAPGKGLEWMGGFDPQDGETIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCATGSSSSWFDPWGQGTLVTVSS -vl DIQMTQSPSSVSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIYGASNLESGVPSRFSGSGSGTDFTLTISSLQPEDFANYYCQQANSFPWTFGQGTKVEIK -odir test/test_paired_scoring2 -oid test_paired_Abrilumab -align -plot
```

Additionally, AbNatiV nativeness scoring can be used directly via its in-built function. It takes as inputs a list of SeqRecords (seq_records, see BioPython). For instance:

```bash
from abnativ.model.scoring_functions import abnativ_scoring_paired

abnativ_p_scores_df, abnativ_p_profiles_df  = abnativ_paired_scoring(df_pairs=pd.DataFrame, col_id='ID', col_vh='vh_seq', col_vl='vl_seq', batch_size=128,
                                    mean_score_only=False, do_align=True, output_dir='test', 
                                    output_id='test_vh', run_parall_al=4)
```

### 2 - Humanisation of Fv sequences (nanobodies and paired VH/VL Fv sequences)

#### 2.1 - Humanisation of nanobodies

To humanise a nanobody sequence with the dual-control strategy of AbNatiV, use the `abnativ hum_vhh` command line.

The dual-control strategy aims to increase the AbNatiV VH-hummanness of a sequence while retaining its VHH-nativeness. All sampling parameters are fully adjustable via the command line (see description bellow).

Two sampling methods are available:\
&emsp; 1. <strong>Enhanced sampling</strong> (default): iteratively explores the mutational space aiming for rapid convergence to generate a single humanised sequence,\
&emsp; 2. <strong>Exhaustive sampling</strong> (if `-isExhaustive`): assesses all mutation combinations within the available mutational space (PSSM-allowed mutations) and selects the best sequences (Pareto Front). It returns a variant with the highest VH-humanness for each number of mutations that are beneficial to the VH-humanness (i.e., when increasing the number of mutations only increases the VH-humanness).

A `-rasa` of 0 will consider every framework residue for mutation. A `-rasa` of 0.15 will considered only solvent-exposed framework residues (as defined in our paper).

NB: a crystal structure (pdb format) can be included (via the filepath `-pdb`, and the chain ID `-ch`) to better assess the solvent-exposed surface of the protein. If `None`, NanoBuilder2 will predict the structure to work on. Only cleaned pdb files will be tolerated. If there is an error to process your pdb file, it is recommended to use the NanoBuilder2 option.

<details>
    <summary>See <strong>abnativ hum_vhh</strong> command line description</summary>

```
abnativ hum_vhh [-h] [-i INPUT_FILEPATH_OR_SEQ] [-odir OUTPUT_DIRECTORY] [-oid OUTPUT_ID] [-VHscore THRESHOLD_ABNATIV_SCORE] [-rasa THRESHOLD_RASA_SCORE]
                       [-isExhaustive] [-VHHdecrease PERC_ALLOWED_DECREASE_VHH] [-a A] [-b B] [-pdb PDB_FILE] [-ch CH_ID]

Use AbNatiV to humanise nanobody sequences by combining AbNatiV VH and VHH assessments (dual-control stategy).

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILEPATH_OR_SEQ, --input_filepath_or_seq INPUT_FILEPATH_OR_SEQ
                        Filepath to the fasta file .fa to score or directly a single string sequence (default: to_score.fa)
  -odir OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Filepath of the folder where all files are saved (default: abnativ_humanisation_vhh)
  -oid OUTPUT_ID, --output_id OUTPUT_ID
                        Prefix of all the saved filenames (e.g., name sequence) (default: nanobody_vhh)
  -VHscore THRESHOLD_ABNATIV_SCORE, --threshold_abnativ_score THRESHOLD_ABNATIV_SCORE
                        Bellow the AbNatiV VH threshold score, a position is considered as a liability (default: 0.98)
  -rasa THRESHOLD_RASA_SCORE, --threshold_rasa_score THRESHOLD_RASA_SCORE
                        Above this threshold, the residue is considered solvent exposed and is considered for mutation (default: 0.15)
  -isExhaustive, --is_Exhaustive
                        If True, runs the Exhaustive sampling strategy. If False, runs the enhanced sampling method (default: False)
  -fmut [FORBIDDEN_MUT [FORBIDDEN_MUT ...]], --forbidden_mut [FORBIDDEN_MUT [FORBIDDEN_MUT ...]]
                        List of string residues to ban for mutation, i.e. C M (default: ['C', 'M'])
  -VHHdecrease PERC_ALLOWED_DECREASE_VHH, --perc_allowed_decrease_vhh PERC_ALLOWED_DECREASE_VHH
                        Maximun ΔVHH score decrease allowed for a mutation (default: 0.015)
  -a A, --a A           Used for enhanced sampling method in multi-objective selection function: aΔVH+bΔVHH (default: 0.8)
  -b B, --b B           Used for enhanced sampling method in multi-objective selection function: aΔVH+bΔVHH (default: 0.2)
  -pdb PDB_FILE, --pdb_file PDB_FILE
                        Filepath to a pdb crystal structure of the nanobody of interest used to compute the solvent exposure. If the PDB is not very cleaned that
                        might lead to some false results (which should be flagged by the program). If None, will predict the structure using NanoBuilder2 (default:
                        None)
  -ch CH_ID, --ch_id CH_ID
                        PDB chain id of the nanobody of interest. If -pdb is None, it does not matter (default: H)
```
</details>

\
Examples of `abnativ hum_vhh` usage:

```bash
# Humanise with the dual-control strategy the mNb6 WT nanobody using the Enhanced sampling (default) on solvent-exposed framework residues (default).
# In directory test/test_humanisation is saved the folder /mNb6_enhanced with the profile, structures, and scored sequences involved in the sampling.
abnativ hum_vhh -i QVQLVESGGGLVQAGGSLRLSCAASGYIFGRNAMGWYRQAPGKERELVAGITRRGSITYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAADPASPAYGDYWGQGTQVTVSS -odir mNb6_enhanced -oid mNb6

# Humanise with the same nanobody with the Exhaustive sampling (-isExhaustive) on solvent-exposed framework residues (default).
# In directory test/test_humanisation is saved the folder /mNb6_exhaustive with the profiles, structures, and selected sequences (Pareto front) involved in the sampling.
abnativ hum_vhh -i QVQLVESGGGLVQAGGSLRLSCAASGYIFGRNAMGWYRQAPGKERELVAGITRRGSITYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAADPASPAYGDYWGQGTQVTVSS -odir mNb6_exhaustive -oid mNb6 -isExhaustive

# You can even directly humanise a fasta file of sequence by giving its filepath as input -i argument.
```

#### 2.2 - Humanisation of VH/VL Fv sequences (using the unpaired models)

To humanise a set of VH/VL Fv sequences with AbNatiV with the unpaired models (humanising VH and VL in parallel), use the `abnativ hum_vh_vl` command line.

A single-control strategy only is applied. It aims to increase the AbNatiV VH- and VL- hummanness of each sequence separately.

Two sampling methods are available:\
&emsp; 1. <strong>Enhanced sampling</strong> (default): iteratively explores the mutational space aiming for rapid convergence to generate a single humanised sequence,\
&emsp; 2. <strong>Exhaustive sampling</strong> (if `-isExhaustive`): assesses all mutation combinations within the available mutational space (PSSM-allowed mutations) and selects the best sequences (Pareto Front). It returns a variant with the highest VH-humanness for each number of mutations that are beneficial to the VH-humanness (i.e., when increasing the number of mutations only increases the humanness).

A `-rasa` of 0 will consider every framework residue for mutation. A `-rasa` of 0.15 will considered only solvent-exposed framework residues (as defined in our paper).
NB: a crystal structure (pdb format) can be included (via the filepath `-pdb`, and the chain IDs `-ch_vh` and `-ch_vl`) to better assess the solvent-exposed surface of the paired chains. If `None`, ABodyBuilder2 will predict the structure to work on. Only cleaned pdb files will be tolerated. If there is an error to process your pdb file, it is recommended to use the ABodyBuilder2 option.

<details>
    <summary>See <strong>abnativ hum_vh_vl</strong> command line description</summary>

```
abnativ hum_vh_vl [-h] [-i_vh INPUT_SEQ_VH] [-i_vl INPUT_SEQ_VL] [-odir OUTPUT_DIRECTORY] [-oid OUTPUT_ID] [-VHscore THRESHOLD_ABNATIV_SCORE]
                         [-rasa THRESHOLD_RASA_SCORE] [-isExhaustive] [-pdb PDB_FILE] [-ch_vh CH_ID_VH] [-ch_vl CH_ID_VL]

Use AbNatiV to humanise a pair of VH/VL Fv sequences by increasing AbNatiV VH- and VL- humanness.

optional arguments:
  -h, --help            show this help message and exit
  -i_vh INPUT_SEQ_VH, --input_seq_vh INPUT_SEQ_VH
                        A single VH string sequence (default: None)
  -i_vl INPUT_SEQ_VL, --input_seq_vl INPUT_SEQ_VL
                        A single VL string sequence (default: None)
  -odir OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Filepath of the folder where all files are saved (default: abnativ_humanisation_vh_vl)
  -oid OUTPUT_ID, --output_id OUTPUT_ID
                        Prefix of all the saved filenames (e.g., name sequence) (default: antibody_vh_vl)
  -VHscore THRESHOLD_ABNATIV_SCORE, --threshold_abnativ_score THRESHOLD_ABNATIV_SCORE
                        Bellow the AbNatiV VH threshold score, a position is considered as a liability (default: 0.98)
  -rasa THRESHOLD_RASA_SCORE, --threshold_rasa_score THRESHOLD_RASA_SCORE
                        Above this threshold, the residue is considered solvent exposed and is considered for mutation (default: 0.15)
  -isExhaustive, --is_Exhaustive
                        If True, runs the Exhaustive sampling strategy. If False, runs the enhanced sampling method (default: False)
  -fmut [FORBIDDEN_MUT [FORBIDDEN_MUT ...]], --forbidden_mut [FORBIDDEN_MUT [FORBIDDEN_MUT ...]]
                        List of string residues to ban for mutation, i.e. C M (default: ['C', 'M'])
  -pdb PDB_FILE, --pdb_file PDB_FILE
                        Filepath to a pdb crystal structure of the nanobody of interest used to compute the solvent exposure. If the PDB is not very cleaned that
                        might lead to some false results (which should be flagged by the program). If None, will predict the paired structure using ABodyBuilder2
                        (default: None)
  -ch_vh CH_ID_VH, --ch_id_vh CH_ID_VH
                        PDB chain id of the heavy chain of interest. If -pdb is None, it does not matter (default: H)
  -ch_vl CH_ID_VL, --ch_id_vl CH_ID_VL
                        PDB chain id of the light chain of interest. If -pdb is None, it does not matter (default: L)
```
</details>

\
Examples of `abnativ hum_vh_vl` usage:

```bash
# Humanise conjointly the VH and VL cahins using the Enhanced sampling (default) on solvent-exposed framework residues (default).
# In directory test/test_humanisation is saved the folder /test_vh_vl_enhanced with the profile, structures, and scored sequences involved in the sampling.
abnativ hum_vh_vl -i_vh QVQLVQSGPELVKPGASLKLSCTASGFNIKDTYIHWVKQAPGQGLEWIGRIYPTNGYTRYDQKFQDRATITVDTSINTAYLHVTRLTSDDTAVYYCSRWGGDGFYAMDYWGQGALVTVSS -i_vl DIQMTQSPSSLSTSVGDRVTITCRASQDVNTAVAWYQQKPGKSPKLLIYSASFLQTGVPSRFTGSRSGTDFTFTISSVQAEDVAVYYCQQHYTTPPTFGGGTKVEIK -odir test_vh_vl_enhanced -oid test_vh_vl

# Humanise with the same VH/VL paired with the Exhaustive sampling (-isExhaustive) on solvent-exposed framework residues (default).
# In directory test/test_humanisation is saved the folder /test_vh_vl_exhaustive with the profiles, structures, and selected sequences (Pareto front) involved in the sampling.
abnativ hum_vh_vl -i_vh QVQLVQSGPELVKPGASLKLSCTASGFNIKDTYIHWVKQAPGQGLEWIGRIYPTNGYTRYDQKFQDRATITVDTSINTAYLHVTRLTSDDTAVYYCSRWGGDGFYAMDYWGQGALVTVSS -i_vl DIQMTQSPSSLSTSVGDRVTITCRASQDVNTAVAWYQQKPGKSPKLLIYSASFLQTGVPSRFTGSRSGTDFTFTISSVQAEDVAVYYCQQHYTTPPTFGGGTKVEIK -odir test_vh_vl_exhaustive -oid test_vh_vl -isExhaustive
```

#### 2.3 - Humanisation of VH/VL Fv sequences (using the paired model)

To humanise a set of VH/VL Fv sequences with the paired model p-AbNatiV with the unpaired models, use the `abnativ hum_vh_vl_paired` command line.

The dual-control strategy aims to increase the AbNatiV hummanness of the VH and VL sequences while improving its pairing likelihood. All sampling parameters are fully adjustable via the command line (see description bellow). 

Only the enhanced sampling strategy is available (the exhaustive one will be too slow on trying all the VH and VL mutation combinations together):\
&emsp; 1. <strong>Enhanced sampling</strong> (default): iteratively explores the mutational space aiming for rapid convergence to generate a single humanised sequence,\

A `-rasa` of 0 will consider every framework residue for mutation. A `-rasa` of 0.15 will considered only solvent-exposed framework residues (as defined in our paper).
NB: a crystal structure (pdb format) can be included (via the filepath `-pdb`, and the chain IDs `-ch_vh` and `-ch_vl`) to better assess the solvent-exposed surface of the paired chains. If `None`, ABodyBuilder2 will predict the structure to work on. Only cleaned pdb files will be tolerated. If there is an error to process your pdb file, it is recommended to use the ABodyBuilder2 option.

<details>
    <summary>See <strong>abnativ hum_vh_vl_paired</strong> command line description</summary>

```
abnativ hum_vh_vl_paired [-h] [-i_vh INPUT_SEQ_VH] [-i_vl INPUT_SEQ_VL] [-odir OUTPUT_DIRECTORY] [-oid OUTPUT_ID]
                                [-score THRESHOLD_ABNATIV_SCORE] [-rasa THRESHOLD_RASA_SCORE] [-fmut [FORBIDDEN_MUT ...]]
                                [-Pairingdecrease PERC_ALLOWED_DECREASE_PAIRING] [-a A] [-b B] [-pdb PDB_FILE] [-ch_vh CH_ID_VH]
                                [-ch_vl CH_ID_VL]

Use AbNatiV to humanise a pair of VH/VL Fv sequences by increasing AbNatiV VH- and VL- humanness jointy with the paired model, while improving
the pairing likelihood.

options:
  -h, --help            show this help message and exit
  -i_vh INPUT_SEQ_VH, --input_seq_vh INPUT_SEQ_VH
                        A single VH string sequence (default: None)
  -i_vl INPUT_SEQ_VL, --input_seq_vl INPUT_SEQ_VL
                        A single VL string sequence (default: None)
  -odir OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Filepath of the folder where all files are saved (default: abnativ_humanisation_vh_vl)
  -oid OUTPUT_ID, --output_id OUTPUT_ID
                        Prefix of all the saved filenames (e.g., name sequence) (default: antibody_vh_vl)
  -score THRESHOLD_ABNATIV_SCORE, --threshold_abnativ_score THRESHOLD_ABNATIV_SCORE
                        Bellow the AbNatiV threshold score, a position is considered as a liability (default: 0.98)
  -rasa THRESHOLD_RASA_SCORE, --threshold_rasa_score THRESHOLD_RASA_SCORE
                        Above this threshold, the residue is considered solvent exposed and is considered for mutation (default: 0.15)
  -fmut [FORBIDDEN_MUT ...], --forbidden_mut [FORBIDDEN_MUT ...]
                        List of string residues to ban for mutation, i.e. C M (default: ['C', 'M'])
  -Pairingdecrease PERC_ALLOWED_DECREASE_PAIRING, --perc_allowed_decrease_pairing PERC_ALLOWED_DECREASE_PAIRING
                        Maximun ΔPairing score decrease allowed for a mutation (default: 0.1)
  -a A, --a A           Used for enhanced sampling method in multi-objective selection function: aΔVH+bΔPairing (default: 10)
  -b B, --b B           Used for enhanced sampling method in multi-objective selection function: aΔVH+bΔPairing (default: 1)
  -pdb PDB_FILE, --pdb_file PDB_FILE
                        Filepath to a pdb crystal structure of the nanobody of interest used to compute the solvent exposure. If the PDB is not
                        very cleaned that might lead to some false results (which should be flagged by the program). If None, will predict the
                        paired structure using ABodyBuilder2 (default: None)
  -ch_vh CH_ID_VH, --ch_id_vh CH_ID_VH
                        PDB chain id of the heavy chain of interest. If -pdb is None, it does not matter (default: H)
  -ch_vl CH_ID_VL, --ch_id_vl CH_ID_VL
                        PDB chain id of the light chain of interest. If -pdb is None, it does not matter (default: L)
```
</details>

\
Examples of `abnativ hum_vh_vl_paired` usage:

```bash
# Humanise conjointly the VH and VL cahins using the Enhanced sampling (default) on solvent-exposed framework residues (default).
# In directory test/test_humanisation is saved the folder /test_vh_vl_enhanced with the profile, structures, and scored sequences involved in the sampling.
abnativ hum_vh_vl_paired -i_vh QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYDMSWVRQAPGKGLEWVAKVSSGGGSTYYLDTVQGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARHLHGSFASWGQGTTVTVSS -i_vl EIVLTQSPATLSLSPGERATLSCQASQSISNFLHWYQQRPGQAPRLLIRYRSQSISGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQSGSWPLTFGGGTKVEIK -odir test/test_humanisation -oid etaracizumab_paired
```

### 3 - Training AbNatiV

To train AbNativ on a custom input dataset of antibody sequences, use the `abnativ train` command line.

<details>
    <summary>See <strong>abnativ train</strong> command line description</summary>

```
abnativ train [-h] [-tr TRAIN_FILEPATH] [-va VAL_FILEPATH] [-hp HPARAMS] [-mn MODEL_NAME] [-rn RUN_NAME] [-align]
                     [-isVHH] [-ncpu NCPU]

Train AbNatiV on a new input dataset of antibody sequences

optional arguments:
  -h, --help            show this help message and exit
  -tr TRAIN_FILEPATH, --train_filepath TRAIN_FILEPATH
                        Filepath to fasta file .fa with sequences for training (default: train_2M.fa)
  -va VAL_FILEPATH, --val_filepath VAL_FILEPATH
                        Filepath to fasta file .fa with sequences for validation (default: val_50k.fa)
  -hp HPARAMS, --hparams HPARAMS
                        Filepath to the hyperparameter dictionary .yml (default: hparams.yml)
  -mn MODEL_NAME, --model_name MODEL_NAME
                        Name of the model weight and biases will load the data in (default: abnativ_v2)
  -align, --do_align    Do the alignment and the cleaning of the given sequences before training. This step can takes a lot of
                        time if the number of sequences is huge. (default: False)
  -ncpu NCPU, --ncpu NCPU
                        If ncpu>1 will parallelise the algnment process (default: 1)
  -isVHH, --is_VHH      Considers the VHH seed for the alignment/ It is more suitable when aligning nanobody sequences
                        (default: False)
```

</details>

Example of usage of `abnativ train`:

```bash
# Train.
abnativ train -tr train_sequences.fa -va val_sequences.fa -hp hparams.yml -mn model_name -align -ncpu 4
```

The hyperparameters need to be provided under a YAML file (see `test/hparams.yml`), such as:

```bash
embedding_dim_code_book: 64
kernel: 8
learning_rate: 4.0e-05
```

Every epoch of the training will be saved in `./checkpoints/<run_name>` (as specified in hparams.yml) and the logs in `./mlruns`.
The Lightning Pytorch logging is monitored with Weights and Biases (wandb) under the <model_name> (see WandB documentation: https://wandb.ai/site).

## Issues

- The installation of OpenMM might create troubles with your device. If you have an `import error` with `lib glibxx_3.4.30`, you could solve it with `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.

If you experience any issues please add an issue to the [Gitlab](https://gitlab.developers.cam.ac.uk/ch/sormanni/abnativ).

## Contact

Please contact ar2033@cam.ac.uk to report issues of for any questions.

## Acknowledgements

Part of the training of AbNativV is based on open-source antibody repertoires from the Observed Antibody Space:
> Kovaltsuk, A., Leem, J., Kelm, S., Snowden, J., Deane, C. M., & Krawczyk, K. (2018). Observed Antibody Space: A Resource for Data Mining Next-Generation Sequencing of Antibody Repertoires. The Journal of Immunology, 201(8), 2502–2509. https://doi.org/10.4049/jimmunol.1800708

and PairedAbNGS paired dataset: 
> Dudzic, P., Chomicz, D., Bielska, W. et al. Conserved heavy/light contacts and germline preferences revealed by a large-scale analysis of natively paired human antibody sequences and structural data. Commun Biol 8, 1110 (2025). https://doi.org/10.1038/s42003-025-08388-y 