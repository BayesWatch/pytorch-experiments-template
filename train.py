import argparse
import json
import os
import numpy as np
import tqdm
import logging

from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from models import model_zoo
from utils.arg_parsing import add_extra_option_args, process_args
from utils.gpu_selection_utils import select_devices
from utils.storage import build_experiment_folder, save_checkpoint, restore_model
import random
import glob
import tarfile
import time
from rich import print
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    RenderableColumn,
    SpinnerColumn,
)


def get_base_argument_parser():
    parser = argparse.ArgumentParser()
    # data and I/O
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--num_gpus_to_use",
        type=int,
        default=0,
        help="The number of GPUs to use, use 0 to enable CPU",
    )

    parser.add_argument(
        "--gpu_ids_to_use",
        type=str,
        default=None,
        help="The IDs of the exact GPUs to use, this bypasses num_gpus_to_use if used",
    )

    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--data_filepath", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=100)

    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=False, dest="resume", action="store_true")
    parser.add_argument("--test", dest="test", default=True, action="store_true")

    # logging
    parser.add_argument("--experiment_name", type=str, default="dev")
    parser.add_argument("--logs_path", type=str, default="log")

    parser.add_argument("--filepath_to_arguments_json_config", type=str, default=None)

    parser.add_argument("--save", dest="save", default=True, action="store_true")
    parser.add_argument("--nosave", dest="save", default=True, action="store_false")

    # model
    parser.add_argument("--model.type", type=str, default="ResNet18")
    parser.add_argument("--model.dropout_rate", type=float, default=0.3)

    parser.add_argument("--val_set_percentage", type=float, default=0.1)
    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="MultiStep",
        help="Scheduler for learning rate annealing: CosineAnnealing | MultiStep",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[60, 120, 160],
        help="Multi step scheduler annealing milestones",
    )
    parser.add_argument("--optim", type=str, default="SGD", help="Optimizer?")

    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser = add_extra_option_args(parser)

    return parser


######################################################################################################### Training


def train(
    epoch,
    data_loader,
    model,
    metric_tracker,
    progress_panel,
    progress_tracker,
    overall_progress,
    overall_task,
):
    model = model.train()
    # with tqdm.tqdm(initial=0, total=len(data_loader), smoothing=0) as pbar:
    num_batches = len(data_loader)
    epoch_start_time = time.time()
    progress_panel.reset(progress_tracker)
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        batch_time = time.time()
        data_time = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        data_time = time.time() - data_time

        logits, features = model(inputs)

        loss = criterion(input=logits, target=targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_time

        eta_time_epoch = ((time.time() - epoch_start_time) / (batch_idx + 1)) * (
            num_batches - (batch_idx + 1)
        )

        metric_tracker.push(
            epoch,
            batch_idx,
            data_time,
            batch_time,
            eta_time_epoch,
            logits,
            targets,
        )

        iter_update_dict = {
            **dict(task_id=progress_tracker, advance=1),
            **metric_tracker.get_current_iteration_metric_text_column_fields(),
        }
        progress_panel.update(**iter_update_dict)
        overall_progress.advance(overall_task, advance=1)
    metric_tracker.update_per_epoch_table()


def eval(
    epoch,
    data_loader,
    model,
    metric_tracker,
    progress_panel,
    progress_tracker,
    overall_progress,
    overall_task,
):
    # with tqdm.tqdm(initial=0, total=len(data_loader), smoothing=0) as pbar:
    num_batches = len(data_loader)
    epoch_start_time = time.time()
    model = model.eval()
    progress_panel.reset(progress_tracker)

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        batch_time = time.time()
        data_time = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        data_time = time.time() - data_time

        logits, features = model(inputs)

        batch_time = time.time() - batch_time

        eta_time_epoch = ((time.time() - epoch_start_time) / (batch_idx + 1)) * (
            num_batches - (batch_idx + 1)
        )

        metric_tracker.push(
            epoch,
            batch_idx,
            data_time,
            batch_time,
            eta_time_epoch,
            logits,
            targets,
        )

        iter_update_dict = {
            **dict(task_id=progress_tracker, advance=1),
            **metric_tracker.get_current_iteration_metric_text_column_fields(),
        }
        progress_panel.update(**iter_update_dict)
        overall_progress.advance(overall_task, advance=1)
    metric_tracker.update_per_epoch_table()


if __name__ == "__main__":
    argument_parser = get_base_argument_parser()
    args = process_args(argument_parser)

    if args.gpu_ids_to_use is None:
        select_devices(
            args.num_gpus_to_use,
            max_load=args.max_gpu_selection_load,
            max_memory=args.max_gpu_selection_memory,
            exclude_gpu_ids=args.excude_gpu_list,
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids_to_use.replace(" ", ",")

    from datasets.dataset_loading_hub import load_dataset
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision.utils import save_image
    import torch.backends.cudnn as cudnn
    from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
    from utils.metric_tracking import MetricTracker, compute_accuracy

    ######################################################################################################### Admin
    saved_models_filepath, logs_filepath, images_filepath = build_experiment_folder(
        experiment_name=args.experiment_name, log_path=args.logs_path
    )

    ######################################################################################################### Data

    #  if you have an environment variable set, prioritise this over the default argparse option
    environment_data_filepath = os.environ.get("PYTORCH_DATA_LOC")
    if environment_data_filepath is not None and os.path.exists(
        environment_data_filepath
    ):
        logging.warning(
            f"You have a data filepath set in your environment: {environment_data_filepath}. This will override argparse."
        )
        data_filepath = environment_data_filepath
    else:
        data_filepath = args.data_filepath
    (
        train_set_loader,
        val_set_loader,
        test_set_loader,
        train_set,
        val_set,
        test_set,
        data_shape,
        num_classes,
    ) = load_dataset(
        dataset=args.dataset_name,
        data_filepath=args.data_filepath
        if environment_data_filepath is None
        else environment_data_filepath,
        batch_size=args.batch_size,
        test_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        download=True,
        val_set_percentage=args.val_set_percentage,
    )

    ######################################################################################################### Determinism
    # Seeding can be annoying in pytorch at the moment. Based on my experience, the below means of seeding
    # allows for deterministic experimentation.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # set seed
    random.seed(args.seed)
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available() and args.num_gpus_to_use > 0
        else "cpu"
    )
    args.device = device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # Always save a snapshot of the current state of the code. I've found this helps immensely if you find that one of your
    # many experiments was actually quite good but you forgot what you did

    snapshot_filename = f"{saved_models_filepath}/snapshot.tar.gz"
    filetypes_to_include = [".py"]
    all_files = []
    for filetype in filetypes_to_include:
        all_files += glob.glob("**/*.py", recursive=True)
    with tarfile.open(snapshot_filename, "w:gz") as tar:
        for file in all_files:
            tar.add(file)

    ######################################################################################################### Model

    model = model_zoo[args.model.type](
        num_classes=num_classes, in_channels=data_shape[0]
    ).to(device)

    # alternatively one can define a model directly as follows
    # ```
    # model = ResNet18(num_classes=num_classes, variant=args.dataset_name).to(device)
    # ```

    if args.num_gpus_to_use > 1:
        model = nn.parallel.DistributedDataParallel(
            model
        )  # more efficient version of DataParallel

    model = model.to(device)

    ######################################################################################################### Optimisation

    params = model.parameters()
    criterion = nn.CrossEntropyLoss()

    if args.optim.lower() == "sgd":
        optimizer = optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )

    if args.scheduler == "CosineAnnealing":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=args.max_epochs, eta_min=0
        )
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

    ######################################################################################################### Restoring

    restore_fields = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    start_epoch = 0
    if args.resume:
        resume_epoch = restore_model(restore_fields, path=saved_models_filepath)
        if resume_epoch == -1:
            raise IOError(
                f"Failed to load from {saved_models_filepath}/ckpt.pth.tar, which probably means that the "
                f"latest checkpoint is missing, please remove the --resume flag to try training from scratch"
            )
        else:
            start_epoch = resume_epoch + 1

    ######################################################################################################### Metric

    metrics_to_track = {
        "cross_entropy": lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(),
        "accuracy": compute_accuracy,
    }
    metric_tracker_train, metric_tracker_val, metric_tracker_test = (
        MetricTracker(
            metrics_to_track=metrics_to_track,
            load=True if start_epoch > 0 else False,
            path=f"{logs_filepath}/metrics_{tracker_name}.pt",
            tracker_name=tracker_name,
        )
        for tracker_name in ["training", "validation", "testing"]
    )

    epoch_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        metric_tracker_train.get_metric_text_column(),
    )

    train_progress_dict = {
        **dict(description="[green]Training", total=len(train_set_loader)),
        **metric_tracker_train.get_current_iteration_metric_text_column_fields(),
    }

    val_progress_dict = {
        **dict(description="[yellow]Validation", total=len(val_set_loader)),
        **metric_tracker_val.get_current_iteration_metric_text_column_fields(),
    }

    train_epoch_progress = epoch_progress.add_task(**train_progress_dict)
    val_epoch_progress = epoch_progress.add_task(**val_progress_dict)

    for task in epoch_progress.tasks:
        print(task)

    overall_progress = Progress()

    total_iters = (args.max_epochs - 1 - start_epoch) * (
        len(train_set_loader) + len(val_set_loader)
    ) + len(test_set_loader) * args.test

    overall_task = overall_progress.add_task(
        "Experiment Progress",
        completed=(start_epoch + 1) * (len(train_set_loader) + len(val_set_loader)),
        total=total_iters,
    )

    progress_table = Table.grid()
    progress_table.add_row(
        Panel.fit(
            overall_progress,
            title="Experiment Progress",
            border_style="green",
            padding=(2, 2),
        ),
    )
    progress_table.add_row(
        Panel.fit(epoch_progress, title="[b]Jobs", border_style="red", padding=(1, 2))
    )
    progress_table.add_row(metric_tracker_train.per_epoch_table)
    progress_table.add_row(metric_tracker_val.per_epoch_table)

    train_iterations = 0

    with Live(progress_table, refresh_per_second=10) as interface_panel:
        for epoch in range(start_epoch, args.max_epochs):

            train(
                epoch,
                data_loader=train_set_loader,
                model=model,
                metric_tracker=metric_tracker_train,
                progress_panel=epoch_progress,
                progress_tracker=train_epoch_progress,
                overall_progress=overall_progress,
                overall_task=overall_task,
            )
            with torch.no_grad():
                eval(
                    epoch,
                    data_loader=val_set_loader,
                    model=model,
                    metric_tracker=metric_tracker_val,
                    progress_panel=epoch_progress,
                    progress_tracker=val_epoch_progress,
                    overall_progress=overall_progress,
                    overall_task=overall_task,
                )

            scheduler.step()

            metric_tracker_train.plot(path=f"{images_filepath}/train/metrics.png")
            metric_tracker_val.plot(path=f"{images_filepath}/val/metrics.png")
            metric_tracker_train.save()
            metric_tracker_val.save()

            ################################################################################ Saving models
            if args.save:
                state = {
                    "args": args,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }

                current_epoch_filename = f"{epoch}_ckpt.pth.tar"
                latest_epoch_filename = "latest_ckpt.pth.tar"

                save_checkpoint(
                    state=state,
                    directory=saved_models_filepath,
                    filename=current_epoch_filename,
                    is_best=False,
                )

                save_checkpoint(
                    state=state,
                    directory=saved_models_filepath,
                    filename=latest_epoch_filename,
                    is_best=False,
                )
            ############################################################################################################

        if args.test:
            test_progress_dict = {
                **dict(description="[green]Training", total=len(test_set_loader)),
                **metric_tracker_test.get_current_iteration_metric_text_column_fields(),
            }

            test_epoch_progress = epoch_progress.add_task(**test_progress_dict)

            if args.val_set_percentage >= 0.0:
                best_epoch_val_model = metric_tracker_val.get_best_epoch_for_metric(
                    evaluation_metric=np.argmax, metric_name="accuracy_mean"
                )
                resume_epoch = restore_model(
                    restore_fields,
                    path=saved_models_filepath,
                    epoch=best_epoch_val_model,
                )

            eval(
                epoch,
                model=model,
                data_loader=test_set_loader,
                metric_tracker=metric_tracker_test,
                progress_panel=epoch_progress,
                progress_tracker=test_epoch_progress,
            )

            metric_tracker_test.save()
