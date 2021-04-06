import argparse
import json
import rich
import time
import tqdm
import random
import logging
import argparse
import datetime

import numpy as np
import tqdm


from models import get_model
from utils.arg_parsing import add_extra_option_args, process_args
from utils.gpu_selection_utils import select_devices
from utils.storage import build_experiment_folder, save_checkpoint, restore_model
import random
import glob
import tarfile


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
    parser.add_argument("--data_filepath", type=str, default="../data/cifar10")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)

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
    parser.add_argument("--model.type", type=str, default="resnet18")
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


def train(epoch, data_loader, model, metric_tracker):

    n_batches = len(data_loader)

    with Live(metric_tracker.table, refresh_per_second=1):

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            batch_time = time.time()
            data_time = time.time()

            inputs, targets = inputs.to(device), targets.to(device)
            data_time = time.time() - data_time

            model = model.train()

            logits, features = model(inputs)

            loss = criterion(input=logits, target=targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_time

            metric_tracker.push(
                epoch,
                batch_idx,
                data_time,
                batch_time,
                str(datetime.timedelta(seconds=batch_time * (n_batches - batch_idx))),
                logits,
                targets,
            )

    return log_string


def eval(epoch, data_loader, model, metric_tracker):
    n_batches = len(data_loader)
    with Live(metric_tracker.table, refresh_per_second=1):

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            batch_time = time.time()
            data_time = time.time()

            inputs, targets = inputs.to(device), targets.to(device)
            data_time = time.time() - data_time

            model = model.eval()

            logits, features = model(inputs)

            batch_time = time.time() - batch_time

            metric_tracker.push(
                epoch,
                batch_idx,
                data_time,
                batch_time,
                str(datetime.timedelta(seconds=batch_time * (n_batches - batch_idx))),
                logits,
                targets,
            )

    return log_string


def run_epoch(epoch, model, training, metric_tracker, data_loader):
    training_iterations = epoch * len(data_loader)

    with tqdm.tqdm(initial=0, total=len(data_loader), smoothing=0) as pbar:

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if training:
                log_string = train_iter(
                    model=model,
                    x=inputs,
                    y=targets,
                    iteration=training_iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,
                    metric_tracker=metric_tracker,
                )
                training_iterations += 1
            else:
                log_string = eval_iter(
                    model=model,
                    x=inputs,
                    y=targets,
                    iteration=training_iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,
                    metric_tracker=metric_tracker,
                )

            pbar.set_description(log_string)
            pbar.update()


if __name__ == "__main__":
    argument_parser = get_base_argument_parser()
    args = process_args(argument_parser)

    os.environ["CUDA_VISIBLE_DEVICES"] = select_devices(
        args.gpu_ids_to_use, args.num_gpus_to_use
    )

    print(
        f"[red bold] WARNING: It looks like you're running an experiment. Make sure you commit :smiley:"
    )

    repo = git.Repo()
    sha = repo.head.object.hexsha
    model_checkpoint_file_name = f"{args.experiment_name}_{args.model.type}_{args.dataset_name}_{args.seed}_{sha[:7]}"

    ######################################################################################################### Data

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
        args.dataset_name,
        args.data_filepath,
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

    snapshot_filename = "{}/snapshot.tar.gz".format(saved_models_filepath)
    filetypes_to_include = [".py"]
    all_files = []
    for filetype in filetypes_to_include:
        all_files += glob.glob("**/*.py", recursive=True)
    with tarfile.open(snapshot_filename, "w:gz") as tar:
        for file in all_files:
            tar.add(file)

    ######################################################################################################### Model

    model = model_zoo[args.model.type](num_classes=num_classes).to(device)

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
            raise IOError(f"Failed to load from {saved_models_filepath}/ckpt.pth.tar, which probably means that the "
                          f"latest checkpoint is missing, please remove the --resume flag to try training from scratch")
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

    train_iterations = 0

    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        for epoch in range(start_epoch, args.max_epochs):

        train(
            epoch,
            data_loader=train_set_loader,
            model=model,
            metric_tracker=metric_tracker_train,
        )
        eval(
            epoch,
            data_loader=val_set_loader,
            model=model,
            metric_tracker=metric_tracker_val,
        )
        scheduler.step()

            metric_tracker_train.plot(
                path=f"{images_filepath}/train/metrics.png"
            )
            metric_tracker_val.plot(path=f"{images_filepath}/val/metrics.png")
            metric_tracker_train.save()
            metric_tracker_val.save()

        state = {
            "args": args,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_metrics": metric_tracker_train.metrics,
            "val_metrics": metric_tracker_val.metrics,
        }

        torch.save(
            state,
            f"{args.experiment_folder}/{model_checkpoint_file_name}.checkpoint",
        )

        metric_tracker_train.save()
        metric_tracker_val.save()

    if args.test:
        if args.val_set_percentage >= 0.0:
            best_epoch_val_model = metric_tracker_val.get_best_epoch_for_metric(
                evaluation_metric=np.argmax, metric_name="accuracy_mean"
            )
            resume_epoch = restore_model(
                restore_fields,
                path=args.experiment_folder,
                epoch=best_epoch_val_model,
            )

        eval(
            epoch,
            model=model,
            data_loader=test_set_loader,
            metric_tracker=metric_tracker_test,
        )

            metric_tracker_test.save()
