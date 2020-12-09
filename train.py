import argparse
import os
import pathlib
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torchvision.transforms
from deep_classifier.config.defaults import get_cfg_defaults
from deep_classifier.datasets import get_dataloader
from deep_classifier.logger.logger import CreateLogger
from deep_classifier.logger.logger_utils import (make_dir, save_configs,
                                                  save_model, save_params,
                                                  savelogs)
from deep_classifier.models import load_model
from deep_classifier.optim import get_optimizer, get_scheduler
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)


# Parse console arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "config", type=str, help="yaml type config file to be used for training"
)
parser.add_argument(
    "--resume-train",
    action="store_true",
    help="if true, resume training from the checkpoint",
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Use mixed precision in training"
)
parser.add_argument("--checkpoint", type=str, help="a path for the checkpoint")
parser.add_argument(
    "--opts",
    help="Modify config options using the command-line",
    default=[],
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

# Get configs from a config file
CFG = get_cfg_defaults()
CFG.merge_from_file(args.config)
CFG.merge_from_list(args.opts)

device_ids = ",".join(str(d) for d in CFG.SYSTEM.DEVICE_IDS)
os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

# Create system logger
CFG.SOLVER.CKPT_DIR = os.path.join("checkpoint", CFG.SOLVER.SAVEDIR)
pathlib.Path(CFG.SOLVER.CKPT_DIR).mkdir(parents=True, exist_ok=True)
MY_LOGGER = CreateLogger("deep-classifier", CFG.SOLVER.CKPT_DIR + "/train_result.log")
MY_LOGGER.info(CFG)

def save_configfile(save_dir, config_filename):
    savefile = os.path.join(save_dir, os.path.basename(config_filename))
    with open(savefile, "w") as f:
        f.write(CFG.dump())

def test_model(model, testloader, epoch, best_acc, num_classes, writer):
    model.eval()        
    running_corrects = 0
    num_data = 0
    MY_LOGGER.info(
        "==================================Validation phase=================================="
    )
    with torch.no_grad():
        for data in tqdm(testloader, total=len(testloader)):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            num_data += inputs.size(0)
            
        test_acc = running_corrects / num_data * 100.0
        MY_LOGGER.info(
            "Validation Accuracy for the {:d} test images: {:.2f}%".format(
                num_data, test_acc
            )
        )

        # tensorboard logging for every epoch
        writer.add_scalar("Epoch Val accuracy", test_acc, epoch + 1)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = model.state_dict()
            save_params(best_model_wts, CFG.SOLVER.CKPT_DIR, "best_weights.pth")

        return test_acc, best_acc


# train model function
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    num_classes,
    start_epoch,
    writer,
    use_fp16=False
):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Set iterations and epochs
    global_iterations = 0
    num_train = len(dataloaders["train"].dataset)
    num_val = len(dataloaders["val"].dataset)
    iter_per_epoch = len(dataloaders["train"])
    num_epochs = CFG.SOLVER.NUM_EPOCHS

    if use_fp16:
        MY_LOGGER.info("Use automatic mixed precision for training.")
        scaler = GradScaler()

    for epoch in range(num_epochs):
        epoch += start_epoch

        MY_LOGGER.info("Epoch {}/{}".format(epoch + 1, num_epochs + start_epoch))
        MY_LOGGER.info("-" * 10)

        # Iterate over data
        if epoch % CFG.SOLVER.VALID_INTERVAL == 0:
            phases = ["train", "val"]
        else:
            phases = ["train"]

        for phase in phases:
            if phase == "train":
                model.train(True)
                if scheduler:
                    scheduler.step()
            elif phase == "val":
                val_acc, best_acc = test_model(
                    model, dataloaders[phase], epoch, best_acc, num_classes, writer
                )
                break

            epoch_lr = optimizer.param_groups[0]["lr"]
            running_loss = 0.0
            running_num_examples = 0
            running_corrects = 0
            i_start_time = time.time()
            
            for i, data in enumerate(dataloaders[phase]):
                global_iterations += 1

                # log start time for ith iteration
                if i % CFG.SOLVER.PRINT_INTERVAL == 0:
                    i_start_time = time.time()

                # get the inputs and labels
                inputs, labels = data

                inputs = inputs.cuda()
                labels = labels.cuda()
                model = model.cuda()

                # zero the parameter gradients every iteration
                optimizer.zero_grad()

                # forward pass
                with autocast(enabled=use_fp16):
                    if CFG.MODEL.BACKBONE == "inception-v3":
                        if phase == "train":
                            logit, aux_logit = model.forward(inputs)
                        else:
                            logit = model.forward(inputs)

                    else:
                        logit = model.forward(inputs)

                    _, preds = torch.max(logit.data, 1)

                    # compute loss
                    loss = criterion(logit, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        if use_fp16:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                # compute statistics
                batch_loss = loss.item()
                batch_corrects = torch.sum(preds == labels.data).item()
                batch_acc = batch_corrects / inputs.size(0) * 100
                running_loss += batch_loss
                running_corrects += batch_corrects
                running_num_examples += inputs.size(0)

                # MY_LOGGER.info key statistics
                if i % CFG.SOLVER.PRINT_INTERVAL == 0:
                    time_elapsed = ( time.time() - i_start_time ) / CFG.SOLVER.PRINT_INTERVAL
                    MY_LOGGER.info(
                        "Epoch: {0:}/{1:}, Iterations: {2:}/{3:}, {4:} loss: {5:6.4f}, accuracy: {6:.2f}%, lr: {7:.6f} time elapsed: {8:6.4f}".format(
                            epoch + 1,
                            num_epochs + start_epoch,
                            i + 1,
                            iter_per_epoch,
                            phase,
                            batch_loss,
                            batch_acc,
                            epoch_lr,
                            time_elapsed % 60,
                        )
                    )
                    i_start_time = time.time()
                    
                if i % 100 == 0:
                    # ============ TensorBoard logging ============#
                    # Save log file every 1000 iteration
                    # (1) Log the scalar values
                    avg_loss = running_loss / (i + 1)
                    avg_acc = running_corrects / running_num_examples * 100
                    savelogs(writer, phase, epoch, avg_loss, avg_acc, global_iterations)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / running_num_examples * 100
            MY_LOGGER.info(
                "Epoch {0:}/{1:} {2:} Loss: {3:.4f}, Accuracy: {4:.2f}% [{5:}/{6:}]".format(
                    epoch + 1,
                    num_epochs + start_epoch,
                    phase,
                    epoch_loss,
                    epoch_acc,
                    running_corrects,
                    running_num_examples,
                )
            )

            # tensorboard logging for every epoch
            writer.add_scalar("Epoch " + phase + " loss", epoch_loss, epoch + 1)
            writer.add_scalar("Epoch " + phase + " accuracy", epoch_acc, epoch + 1)
            writer.add_scalar("learning rate", epoch_lr, epoch + 1)

        # deep copy the model
        save_model(
            model, optimizer, epoch, CFG.SOLVER.CKPT_DIR, "last_checkpoint.pth.tar"
        )

    time_elapsed = time.time() - since
    MY_LOGGER.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )


def main():
    # save config file
    save_configfile(CFG.SOLVER.CKPT_DIR, args.config)

    # load dataloader
    dataloaders, num_classes = get_dataloader(CFG)
    MY_LOGGER.info("Number of classes for the dataset is %d" % num_classes)
    MY_LOGGER.info(dataloaders["train"].dataset._labeldict)

    # load model
    model = load_model(CFG.MODEL.BACKBONE, num_classes, CFG.MODEL.PRETRAINED,
                      batch_norm=CFG.MODEL.BATCH_NORM)
    model = model.cuda()
    MY_LOGGER.info(model)
    
    # loss functions and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model.parameters(), CFG)

    if CFG.SYSTEM.MULTI_GPU:
        MY_LOGGER.info("Using DataParallel")
        model = torch.nn.DataParallel(model)
        
    scheduler = get_scheduler(optimizer, CFG)

    if CFG.SOLVER.LR_WARMUP:
        from deep_classifier.optim.lr_scheduler import GradualWarmupScheduler

        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=CFG.SOLVER.LR_MULTIPLIER,
            total_epoch=CFG.SOLVER.WARMUP_EPOCHS,
            after_scheduler=scheduler,
        )
        MY_LOGGER.info(
            "Warmup lr for %d epochs.\n Initial LR: %f, LR after warmup %f"
            % (
                CFG.SOLVER.WARMUP_EPOCHS,
                CFG.SOLVER.BASE_LR,
                CFG.SOLVER.BASE_LR * CFG.SOLVER.LR_MULTIPLIER,
            )
        )

    start_epoch = 0

    # Load model state if resuming the train
    if args.resume_train:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt["state_dict"])
        optimizer = ckpt["optimizer"]
        start_epoch = ckpt["epoch"]
        MY_LOGGER.info(
            "Resuming the training. Start epoch is {}".format(start_epoch + 1)
        )

    # set tensorboard logger
    logdir = "logs/" + CFG.SOLVER.SAVEDIR
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir)

    # Train the model
    train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        num_classes,
        start_epoch,
        writer,
        use_fp16=args.fp16
    )


if __name__ == "__main__":
    main()
