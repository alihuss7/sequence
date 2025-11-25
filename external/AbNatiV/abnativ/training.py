# (c) 2023 Sormannilab and Aubin Ramon
#
# Training of the AbNatiV model with the Pytorch Lightning module.
#
# ============================================================================

from .model.abnativ import AbNatiV_Model
from .model.onehotencoder import data_loader_masking_bert_onehot_fasta
from .model.alignment.mybio import anarci_alignments_of_Fv_sequences

import random
import os
import argparse
import yaml
from Bio import SeqIO

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def run(args: argparse.Namespace):
    # fix seed for reproductibility
    pl.seed_everything(11)
    random.seed(11)

    ## BUILDING MODULES ##
    with open(args.hparams, "r") as f:
        hparams = yaml.safe_load(f)

    model = AbNatiV_Model(hparams)

    ## ALIGNMENT ##
    if args.do_align:
        # Train
        print(f"\n### ANARCI alignment of {args.train_filepath}###\n")
        VH, VK, VL, failed, mischtype = anarci_alignments_of_Fv_sequences(
            args.train_filepath,
            isVHH=args.is_VHH,
            del_cyst_misalign=True,
            run_parallel=args.ncpu,
        )
        VH.add(VK)
        VH.add(VL)
        fp_al_train = "temp_al_train_seqs.fa"
        VH.Print(fp_al_train)

        # Val
        print(f"\n### ANARCI alignment of {args.val_filepath}###\n")
        VH, VK, VL, failed, mischtype = anarci_alignments_of_Fv_sequences(
            args.val_filepath,
            isVHH=args.is_VHH,
            del_cyst_misalign=True,
            run_parallel=args.ncpu,
        )
        VH.add(VK)
        VH.add(VL)
        fp_al_val = "temp_al_val_seqs.fa"
        VH.Print(fp_al_val)

    else:
        fp_al_train = args.train_filepath
        fp_al_val = args.val_filepath

    ## DATA LOADING ##
    train_loader = data_loader_masking_bert_onehot_fasta(
        fp_al_train,
        hparams["batch_size"],
        hparams["perc_masked_residues"],
        is_masking=True,
    )

    val_loader = data_loader_masking_bert_onehot_fasta(
        fp_al_val, hparams["batch_size"], perc_masked_residues=0, is_masking=False
    )

    ## TRAINING ##
    # Logging
    logger = WandbLogger(
        project=args.model_name, name=hparams["run_name"], log_model=False
    )

    # Checkpointing
    ckpt_root_dir = os.path.join("checkpoints", hparams["run_name"])
    ckpt_callback = ModelCheckpoint(ckpt_root_dir, save_top_k=-1)  # to save every epoch

    trainer = pl.Trainer(
        limit_train_batches=1,
        limit_val_batches=hparams["limit_val_batches"],
        max_epochs=hparams["max_epochs"],
        deterministic=True,
        accelerator="auto",
        logger=logger,
        callbacks=[ckpt_callback],
    )

    # Training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Remove temporary files
    if args.do_align:
        os.remove(fp_al_val)
        os.remove(fp_al_val)
