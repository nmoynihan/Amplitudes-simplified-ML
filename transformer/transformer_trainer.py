'''Script for training a transformer model on amplitude simplification'''
import os

os.environ["OMP_NUM_THREADS"] = "1"           # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"      # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"           # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"    # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"       # NumExpr

import argparse
import multiprocessing

import torch
import torch.nn as nn

from data_import import load_and_prepare_data
from transformer_functions import create_model, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the amplitude simplification transformer.")
    parser.add_argument("--run_name", "--run-name", dest="run_name", default="default_run")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=["gi_4pt_os_tok.csv"],
        help="Tokenized CSV file(s). Relative paths are resolved under ./data by data_import.py.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", "--embedding_dim", dest="embedding_dim", type=int, default=512)
    parser.add_argument("--n-heads", "--n_heads", dest="n_heads", type=int, default=4)
    parser.add_argument("--n-enc-layers", "--n_enc_layers", dest="n_enc_layers", type=int, default=5)
    parser.add_argument("--n-dec-layers", "--n_dec_layers", dest="n_dec_layers", type=int, default=5)
    parser.add_argument("--head-ff-dim", "--head_ff_dim", dest="head_ff_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.025)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm for clipping. Set to 0 to disable.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches per optimizer step.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--dynamic-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bucketing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bucket-size-multiplier", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume-from",
        default="",
        help="Checkpoint path to resume from. --epochs is interpreted as additional epochs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    if device.type == "cpu":
        n_threads = multiprocessing.cpu_count()
        torch.set_num_threads(n_threads)
        print(f"Using {n_threads} CPU threads for PyTorch.")

    training_hyperparams = {
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_split": args.train_split,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "max_length": args.max_length,
        "data_files": args.data_files,
        "dynamic_padding": args.dynamic_padding,
        "bucketing": args.bucketing,
        "bucket_size_multiplier": args.bucket_size_multiplier,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype,
        "resume_from": args.resume_from,
        "grad_clip": args.grad_clip if args.grad_clip > 0 else None,
        "gradient_accumulation_steps": max(
            1, args.gradient_accumulation_steps
        ),
        "label_smoothing": args.label_smoothing,
    }
    model_hyperparams = {
        "embedding_dim": args.embedding_dim,
        "n_heads": args.n_heads,
        "n_enc_layers": args.n_enc_layers,
        "n_dec_layers": args.n_dec_layers,
        "dropout": args.dropout,
        "sinusoidal_embeddings": True,
        "head_ff_dim": args.head_ff_dim,
        "device": device,
    }

    # Hardcode the vocab size to match the tokenizer configuration used by the data scripts.
    vocab_size = 58

    train_loader, val_loader = load_and_prepare_data(
        args.data_files,
        batch_size=training_hyperparams["batch_size"],
        max_length=training_hyperparams["max_length"],
        train_split=training_hyperparams["train_split"],
        dynamic_padding=training_hyperparams["dynamic_padding"],
        bucketing=training_hyperparams["bucketing"],
        bucket_size_multiplier=training_hyperparams["bucket_size_multiplier"],
        num_workers=training_hyperparams["num_workers"],
        pin_memory=training_hyperparams["pin_memory"] and device.type == "cuda",
    )

    model = create_model(vocab_size, **model_hyperparams)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=training_hyperparams["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_hyperparams["learning_rate"],
        weight_decay=training_hyperparams["weight_decay"],
    )
    warmup_epochs = max(1, args.epochs // 20)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=training_hyperparams["learning_rate"] * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )
    start_epoch = 0
    initial_best_val_loss = None
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        initial_best_val_loss = checkpoint.get("val_loss")
        print(
            f"Resumed from {args.resume_from} at epoch {start_epoch}; "
            f"training for {args.epochs} additional epochs."
        )

    amp_enabled = training_hyperparams["amp"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if training_hyperparams["amp_dtype"] == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")
    print(f"Hyperparameters:\nModel:{model_hyperparams}\nTraining:{training_hyperparams}")

    train_model(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        epochs=training_hyperparams["n_epochs"],
        run_name=args.run_name,
        early_stopping_patience=training_hyperparams["early_stopping_patience"],
        early_stopping_min_delta=training_hyperparams["early_stopping_min_delta"],
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
        start_epoch=start_epoch,
        initial_best_val_loss=initial_best_val_loss,
        scheduler=scheduler,
        grad_clip=training_hyperparams["grad_clip"],
        gradient_accumulation_steps=training_hyperparams[
            "gradient_accumulation_steps"
        ],
    )


if __name__ == "__main__":
    main()
