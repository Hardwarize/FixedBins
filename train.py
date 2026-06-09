import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import time
import datetime
import os

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.loss import (
    SILogLoss,
    RMSELoss,
    ChamferDistanceLoss,
    RelativeSSILoss,
    RelativeGradientLoss
)
from depth_estimation.utils.visualization import get_tensorboard_grids

from data.flsea.dataset import get_flsea_dataset
from data.example_dataset.dataset import get_example_dataset


##############################################################
########################## CONFIG ############################
##############################################################

# training parameters
BATCH_SIZE = 12
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.90
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
LOSS_FUNCTIONS = {
    "SILog_Loss": SILogLoss(correction=0.85, scaling=10.0),
    "Chamfer_Loss": ChamferDistanceLoss(),
    "L2_Loss": RMSELoss(),
    "L1_Loss": torch.nn.L1Loss(),
}
LOSS_WEIGHTS = {"w_SILog_Loss": 0.6, "w_Chamfer_Loss": 0.1, "w_L2_Loss": 0.3}

TRAINING_LOSS_NAMES = [
    "training_loss",
    "training_loss/SILog Loss",
    "training_loss/Bins Chamfer Loss",
    "training_loss/L2 Loss (RMSE)",
    "training_loss/L1 Loss (MAE)",
    "training_loss/L2 Log Loss (RMSE log)",
    "training_loss/L2 Loss [d<5m] (RMSE)",
]
VALIDATION_LOSS_NAMES = [
    "validation_loss",
    "validation_loss/SILog Loss",
    "validation_loss/Bins Chamfer Loss",
    "validation_loss/L2 Loss (RMSE)",
    "validation_loss/L1 Loss (MAE)",
    "validation_loss/L2 Log Loss (RMSE log)",
    "validation_loss/L2 Loss [d<5m] (RMSE)",
]
"""

LOSS_FUNCTIONS = {
    "SSI_ZScore_Loss": RelativeSSILoss(),
    "Gradient_ZScore_Loss": RelativeGradientLoss()
}
LOSS_WEIGHTS = {"w_SSI_ZScore_Loss": 0.7, "w_Gradient_ZScore_Loss": 0.3}


TRAINING_LOSS_NAMES = [
    "training_loss",
    "training_loss/SSI_ZScore_Loss",
    "training_loss/Gradient_ZScore_Loss"
]
VALIDATION_LOSS_NAMES = [
    "validation_loss",
    "validation_loss/SSI_ZScore_Loss",
    "validation_loss/Gradient_ZScore_Loss"
]

# datasets
TRAIN_DATASET = get_flsea_dataset(
     split="dataset_with_matched_features",
     train=True,
     shuffle=True,
     device=DEVICE,
)

VALIDATION_DATASET = get_flsea_dataset(
     split="dataset_with_matched_features",
     train=False,
     shuffle=True,
     device=DEVICE,
)

# tensorboard output frequencies
WRITE_TRAIN_IMG_EVERY_N_BATCHES = 100
WRITE_VALIDATION_IMG_EVERY_N_BATCHES = 300

############################################################
############################################################
############################################################


def train_UDFNet():
    """Train loop to train a UDFNet model."""

    # print run infos
    run_name = f"udfnet_lr{LEARNING_RATE}_bs{BATCH_SIZE}_lrd{LEARNING_RATE_DECAY}"
    print(
        f"Training run {run_name} with parameters:\n"
        + f"    learning rate: {LEARNING_RATE}\n"
        + f"    learning rate decay: {LEARNING_RATE_DECAY}\n"
        + f"    batch size: {BATCH_SIZE}\n"
        + f"    device: {DEVICE}"
    )

    # tensorboard summary writer
    global summary_writer
    summary_writer = SummaryWriter(run_name)

    # initialize model
    model = UDFNet(n_bins=80).to(DEVICE)

    # dataloaders
    train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(VALIDATION_DATASET, batch_size=BATCH_SIZE)

    # train epochs
    for epoch in range(EPOCHS):

        # decayed learning rate
        lr = LEARNING_RATE * (LEARNING_RATE_DECAY**epoch)

        # epoch info
        print("------------------------")
        print(f"Epoch {epoch}/{EPOCHS} (lr: {lr}, batch_size: {BATCH_SIZE})")
        print("------------------------")

        # train epoch
        start_time = time.time()
        training_losses = train_epoch(
            dataloader=train_dataloader,
            model=model,
            learning_rate=lr,
            epoch=epoch,
        )
        print(
            f"Epoch time: {str(datetime.timedelta(seconds=(time.time() - start_time)))}"
        )

        # validate epoch
        validation_losses = validate(
            dataloader=validation_dataloader,
            model=model,
            epoch=epoch,
        )

        # tensorboard summary for training and validation
        for loss, loss_name in zip(training_losses, TRAINING_LOSS_NAMES):
            summary_writer.add_scalar(f"{loss_name}", loss, epoch)
        for loss, loss_name in zip(validation_losses, VALIDATION_LOSS_NAMES):
            summary_writer.add_scalar(f"{loss_name}", loss, epoch)

        # save model after every epoch
        save_model(model, epoch, run_name)


def train_epoch(
    dataloader,
    model,
    learning_rate,
    epoch=0,
):
    """Train a model for one epoch.
    - dataloader: the dataloader to use
    - model: The model to train
    - learning_rate: the learning rate for the optimizer
    - epoch: epoch id"""

    # set training mode
    model.train()

    # optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    n_batches = len(dataloader)

    training_losses = np.zeros(len(TRAINING_LOSS_NAMES))

    epoch_start_time = time.time()
    batch_loading_time = None

    for batch_id, data in enumerate(dataloader):



        batch_start_time = time.time()

        if not batch_loading_time:
            print("Batch loading time:", batch_start_time - epoch_start_time)
        else:
            print("Batch loading time:", batch_start_time - batch_loading_time)

        # move to device
        X = data[0].to(DEVICE)  # RGB image
        y = data[1].to(DEVICE)  # depth image
        mask = data[2].to(DEVICE)  # mask for valid values
        prior = data[3].to(DEVICE)  # precomputed features and depth values

        # nullprior, for training without any priors
        prior[:, :, :, :] = 0.0


        model_inference_start_time = time.time()

        # prediction
        pred, bin_edges = model(X, prior)
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        model_inference_end_time = time.time()
        print("Model inference time:", model_inference_end_time - model_inference_start_time)

        # individual losses
        #batch_loss_silog = LOSS_FUNCTIONS["SILog_Loss"](pred, y, mask)
        #batch_loss_chamfer = LOSS_FUNCTIONS["Chamfer_Loss"](y, bin_centers, mask)
        #batch_loss_l2 = LOSS_FUNCTIONS["L2_Loss"](pred, y, mask)
        #batch_loss_l1 = LOSS_FUNCTIONS["L1_Loss"](pred[mask], y[mask])  # , mask)
        #batch_loss_l2_log = LOSS_FUNCTIONS["L2_Loss"](torch.log(pred), torch.log(y), mask)
        #close_range = y[mask] < 5.0  # close range mask (less than 5m)
        #batch_loss_l2_close = LOSS_FUNCTIONS["L2_Loss"](pred[mask][close_range], y[mask][close_range])
        batch_loss_ssi_relative = LOSS_FUNCTIONS["SSI_ZScore_Loss"](pred, y, mask)
        batch_loss_gradient_relative = LOSS_FUNCTIONS["Gradient_ZScore_Loss"](pred, y, mask)

        # guidance signal for points outside of mask (usually points at infinity)
        #batch_loss_silog = batch_loss_silog + 0.02 * LOSS_FUNCTIONS["SILog_Loss"](pred, y, ~mask)
        #batch_loss_l2 = batch_loss_l2 + 0.02 * LOSS_FUNCTIONS["L2_Loss"](pred, y, ~mask)

        # learning objective loss
        #batch_loss = (batch_loss_silog * LOSS_WEIGHTS["w_SILog_Loss"]+ batch_loss_chamfer * LOSS_WEIGHTS["w_Chamfer_Loss"]+ batch_loss_l2 * LOSS_WEIGHTS["w_L2_Loss"])
        batch_loss = (batch_loss_ssi_relative * LOSS_WEIGHTS["w_SSI_ZScore_Loss"]+ batch_loss_gradient_relative * LOSS_WEIGHTS["w_Gradient_ZScore_Loss"])

        model_optim_start_time = time.time()

        # backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        model_optim_end_time = time.time()
        print("Model optimization time:", model_optim_end_time - model_optim_start_time)

        # statistics for tensorboard visualization graphs
        batch_losses = np.array(
            [
                batch_loss.item(),
                batch_loss_ssi_relative.item(),
                batch_loss_gradient_relative.item()
                #batch_loss_silog.item(),
                #batch_loss_chamfer.item(),
                #batch_loss_l2.item(),
                #batch_loss_l1.item(),
                #batch_loss_l2_log.item(),
                #batch_loss_l2_close.item(),
            ]
        )
        training_losses += batch_losses

        # tensorboard summary grids for visual inspection
        if (batch_id % WRITE_TRAIN_IMG_EVERY_N_BATCHES == 0) and (
            X.size(0) == BATCH_SIZE
        ):

            with torch.no_grad():  # no gradients for visualization

                # get tensorboard grids
                grids = get_tensorboard_grids(
                    X, y, prior, pred, mask, bin_edges, device=DEVICE
                )

                # write to tensorboard
                summary_writer.add_image(
                    f"train_rgb_target_pred_error", grids[0], batch_id#epoch
                )
                summary_writer.add_image(
                    f"train_target_parametrization", grids[1], batch_id#epoch
                )

        if batch_id % 50 == 0:
            print(f"batch {batch_id}/{n_batches}, batch training loss: {batch_losses}")
        
        batch_end_time = time.time()
        print("Batch time:", batch_end_time - batch_start_time)

        batch_loading_time = time.time()

    avg_batch_losses = training_losses / n_batches
    print(f"Average batch training loss: {avg_batch_losses}")

    epoch_end_time = time.time()
    print("Epoch time:", epoch_end_time - epoch_start_time)
    return avg_batch_losses


@torch.no_grad()  # no gradients needed during validation
def validate(
    dataloader,
    model,
    epoch=0,
):
    """Validate a model, typically done after each training epoch."""

    # set evaluation mode
    model.eval()

    n_batches = len(dataloader)

    validation_losses = np.zeros(len(VALIDATION_LOSS_NAMES))
    for batch_id, data in enumerate(dataloader):

        # move to device
        X = data[0].to(DEVICE)  # RGB image
        y = data[1].to(DEVICE)  # depth image
        mask = data[2].to(DEVICE)  # mask for valid values
        prior = data[3].to(DEVICE)  # precomputed features and depth values

        # nullprior
        prior[:, :, :, :] = 0.0

        # prediction
        pred, bin_edges = model(X, prior)
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        # individual losses
        batch_loss_ssi_relative = LOSS_FUNCTIONS["SSI_ZScore_Loss"](pred, y, mask)
        batch_loss_gradient_relative = LOSS_FUNCTIONS["Gradient_ZScore_Loss"](pred, y, mask)

 

        # objective (for reference)
        batch_loss = (batch_loss_ssi_relative * LOSS_WEIGHTS["w_SSI_ZScore_Loss"]+ batch_loss_gradient_relative * LOSS_WEIGHTS["w_Gradient_ZScore_Loss"])

        # statistics for tensorboard visualization graphs
        batch_losses = np.array(
            [
                batch_loss.item(),
                batch_loss_ssi_relative.item(),
                batch_loss_gradient_relative.item()
                #batch_loss_silog.item(),
                #batch_loss_chamfer.item(),
                #batch_loss_l2.item(),
                #batch_loss_l1.item(),
                #batch_loss_l2_log.item(),
                #batch_loss_l2_close.item(),
            ]
        )
        validation_losses += batch_losses

        # tensorboard summary grids for visual inspection
        if (batch_id % WRITE_VALIDATION_IMG_EVERY_N_BATCHES == 0) and (
            X.size(0) == BATCH_SIZE
        ):

            # get grids
            grids = get_tensorboard_grids(
                X, y, prior, pred, mask, bin_edges, device=DEVICE
            )

            # write to tensorboard
            summary_writer.add_image(
                f"rgb_target_pred_error/{batch_id}", grids[0], epoch
            )
            summary_writer.add_image(
                f"target_parametrization/{batch_id}", grids[1], epoch
            )

        if batch_id % 100 == 0:
            print(
                f"batch {batch_id}/{n_batches}, batch validation losses: {batch_losses}"
            )

    avg_batch_losses = validation_losses / n_batches
    print(f"Average batch validation losses: {avg_batch_losses}")
    return avg_batch_losses


def save_model(model, epoch, run_name):

    print(f"Saving model after epoch {epoch} ...")

    # check if folder exists
    folder_name = "saved_models"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    # save model
    model_filename = f"{folder_name}/model_e{epoch}_{run_name}.pth"
    torch.save(model.state_dict(), model_filename)


if __name__ == "__main__":

    train_UDFNet()


"""
Time on CPU:

Batch size: 6
Steps per batch: 3416
Model inference time: 5.807194232940674
Model optimization time: 9.680723190307617
Batch time: 16.66849660873413





[Time] 1. Get filenames: 0.000003s
[Time] 2. Read & resize images: 0.055666s
[Time] 3. Input & Target transforms: 0.040360s
[Time] 4. Mask validity check: 0.000115s
[Time] 5. Read sparse depth priors: 0.001928s
[Time] 6. Feature validity check: 0.000001s
[Time] 1. Empty tensor allocation: 0.000034s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000191s
  [Time] 3. get_distance_maps: 2.183512s
  [Time] 4. torch.min: 0.025471s
  [Time] 5. Prior/Dist assignment: 0.000582s
[Time] 6. get_probability_maps: 0.000853s
[Time] 7. torch.cat: 0.000073s
[Time] ---> TOTAL get_depth_prior time: 2.211412s

[Time] 7. Dense parametrization: 2.218594s
[Time] 8. Target + prior transform: 0.000186s
[Time] 9. Mutual transforms: 0.000477s
[Time] ---> TOTAL __getitem__ time: 2.318859s

[Time] 1. Get filenames: 0.000002s
[Time] 2. Read & resize images: 0.065254s
[Time] 3. Input & Target transforms: 0.044100s
[Time] 4. Mask validity check: 0.000107s
[Time] 5. Read sparse depth priors: 0.002606s
[Time] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000023s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000188s
  [Time] 3. get_distance_maps: 2.111200s
  [Time] 4. torch.min: 0.027402s
  [Time] 5. Prior/Dist assignment: 0.000387s
[Time] 6. get_probability_maps: 0.000819s
[Time] 7. torch.cat: 0.000070s
[Time] ---> TOTAL get_depth_prior time: 2.140246s

[Time] 7. Dense parametrization: 2.146862s
[Time] 8. Target + prior transform: 0.000283s
[Time] 9. Mutual transforms: 0.000553s
[Time] ---> TOTAL __getitem__ time: 2.259913s

######################################################################33

Time on GPU:

Batch size: 6
Steps per batch: 3416

Batch loading time: 0.857043981552124
Model inference time: 0.02470874786376953
Model optimization time: 0.35797548294067383
Batch time: 0.7897467613220215

Batch loading time: 0.9013848304748535
Model inference time: 0.023865938186645508
Model optimization time: 0.41518688201904297
Batch time: 0.8368041515350342

Batch loading time: 0.8147425651550293
Model inference time: 0.019650697708129883
Model optimization time: 0.3532900810241699
Batch time: 0.7512681484222412

Batch loading time: 0.8395915031433105
Model inference time: 0.019138813018798828
Model optimization time: 0.3462402820587158
Batch time: 0.736764669418335

Batch loading time: 0.8275854587554932
Model inference time: 0.020373106002807617
Model optimization time: 0.3528451919555664
Batch time: 0.7438409328460693

Batch loading time: 0.9422821998596191
Model inference time: 0.025702953338623047
Model optimization time: 0.3306915760040283
Batch time: 0.7213571071624756

Batch loading time: 0.8692047595977783
Model inference time: 0.01810002326965332
Model optimization time: 0.5109179019927979
Batch time: 0.9680628776550293


------------------------
Epoch 0/25 (lr: 0.0001, batch_size: 6)
------------------------
[Time] [Item: 3196] 1. Get filenames: 0.000002s
[Time] [Item: 3196] 2. Read & resize images: 0.342371s
[Time] [Item: 3196] 3. Input & Target transforms: 1.250630s
[Time] [Item: 3196] 4. Mask validity check: 0.000152s
[Time] [Item: 3196] 5. Read sparse depth priors: 0.094019s
[Time] [Item: 3196] 6. Feature validity check: 0.000001s
[Time] 1. Empty tensor allocation: 0.000578s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.006319s
  [Time] 3. get_distance_maps: 0.167245s
  [Time] 4. torch.min: 0.001091s
  [Time] 5. Prior/Dist assignment: 0.000615s
[Time] 6. get_probability_maps: 0.037577s
[Time] 7. torch.cat: 0.000175s
[Time] ---> TOTAL get_depth_prior time: 0.214170s

[Time] [Item: 3196] 7. Dense parametrization: 0.216051s
[Time] [Item: 3196] 8. Target + prior transform: 0.000659s
[Time] [Item: 3196] 9. Mutual transforms: 0.000078s
[Time] [Item: 3196] ---> TOTAL __getitem__ time: 1.904238s

[Time] [Item: 11145] 1. Get filenames: 0.000003s
[Time] [Item: 11145] 2. Read & resize images: 0.062379s
[Time] [Item: 11145] 3. Input & Target transforms: 0.006418s
[Time] [Item: 11145] 4. Mask validity check: 0.000110s
[Time] [Item: 11145] 5. Read sparse depth priors: 0.070368s
[Time] [Item: 11145] 6. Feature validity check: 0.000001s
[Time] 1. Empty tensor allocation: 0.000482s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000962s
  [Time] 3. get_distance_maps: 0.088814s
  [Time] 4. torch.min: 0.000425s
  [Time] 5. Prior/Dist assignment: 0.000330s
[Time] 6. get_probability_maps: 0.001486s
[Time] 7. torch.cat: 0.000092s
[Time] ---> TOTAL get_depth_prior time: 0.093082s

[Time] [Item: 11145] 7. Dense parametrization: 0.093316s
[Time] [Item: 11145] 8. Target + prior transform: 0.000296s
[Time] [Item: 11145] 9. Mutual transforms: 0.001484s
[Time] [Item: 11145] ---> TOTAL __getitem__ time: 0.234706s

[Time] [Item: 9624] 1. Get filenames: 0.000002s
[Time] [Item: 9624] 2. Read & resize images: 0.316904s
[Time] [Item: 9624] 3. Input & Target transforms: 0.006953s
[Time] [Item: 9624] 4. Mask validity check: 0.000098s
[Time] [Item: 9624] 5. Read sparse depth priors: 0.052484s
[Time] [Item: 9624] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000472s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000977s
  [Time] 3. get_distance_maps: 0.086717s
  [Time] 4. torch.min: 0.000435s
  [Time] 5. Prior/Dist assignment: 0.000221s
[Time] 6. get_probability_maps: 0.001243s
[Time] 7. torch.cat: 0.000083s
[Time] ---> TOTAL get_depth_prior time: 0.090603s

[Time] [Item: 9624] 7. Dense parametrization: 0.090804s
[Time] [Item: 9624] 8. Target + prior transform: 0.000169s
[Time] [Item: 9624] 9. Mutual transforms: 0.000246s
[Time] [Item: 9624] ---> TOTAL __getitem__ time: 0.467931s

[Time] [Item: 11692] 1. Get filenames: 0.000002s
[Time] [Item: 11692] 2. Read & resize images: 0.261705s
[Time] [Item: 11692] 3. Input & Target transforms: 0.010672s
[Time] [Item: 11692] 4. Mask validity check: 0.000070s
[Time] [Item: 11692] 5. Read sparse depth priors: 0.096460s
[Time] [Item: 11692] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000427s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000953s
  [Time] 3. get_distance_maps: 0.089456s
  [Time] 4. torch.min: 0.000443s
  [Time] 5. Prior/Dist assignment: 0.000240s
[Time] 6. get_probability_maps: 0.001242s
[Time] 7. torch.cat: 0.000062s
[Time] ---> TOTAL get_depth_prior time: 0.093232s

[Time] [Item: 11692] 7. Dense parametrization: 0.093448s
[Time] [Item: 11692] 8. Target + prior transform: 0.000171s
[Time] [Item: 11692] 9. Mutual transforms: 0.000044s
[Time] [Item: 11692] ---> TOTAL __getitem__ time: 0.462807s

[Time] [Item: 2042] 1. Get filenames: 0.000002s
[Time] [Item: 2042] 2. Read & resize images: 0.298902s
[Time] [Item: 2042] 3. Input & Target transforms: 0.006208s
[Time] [Item: 2042] 4. Mask validity check: 0.000100s
[Time] [Item: 2042] 5. Read sparse depth priors: 0.076427s
[Time] [Item: 2042] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000435s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000749s
  [Time] 3. get_distance_maps: 0.086882s
  [Time] 4. torch.min: 0.000424s
  [Time] 5. Prior/Dist assignment: 0.000260s
[Time] 6. get_probability_maps: 0.001086s
[Time] 7. torch.cat: 0.000046s
[Time] ---> TOTAL get_depth_prior time: 0.090266s

[Time] [Item: 2042] 7. Dense parametrization: 0.090477s
[Time] [Item: 2042] 8. Target + prior transform: 0.000229s
[Time] [Item: 2042] 9. Mutual transforms: 0.000075s
[Time] [Item: 2042] ---> TOTAL __getitem__ time: 0.472625s

[Time] [Item: 14073] 1. Get filenames: 0.000002s
[Time] [Item: 14073] 2. Read & resize images: 0.073879s
[Time] [Item: 14073] 3. Input & Target transforms: 0.007135s
[Time] [Item: 14073] 4. Mask validity check: 0.000086s
[Time] [Item: 14073] 5. Read sparse depth priors: 0.060047s
[Time] [Item: 14073] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000463s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.001011s
  [Time] 3. get_distance_maps: 0.091839s
  [Time] 4. torch.min: 0.000425s
  [Time] 5. Prior/Dist assignment: 0.000267s
[Time] 6. get_probability_maps: 0.001131s
[Time] 7. torch.cat: 0.000084s
[Time] ---> TOTAL get_depth_prior time: 0.095758s

[Time] [Item: 14073] 7. Dense parametrization: 0.095965s
[Time] [Item: 14073] 8. Target + prior transform: 0.000215s
[Time] [Item: 14073] 9. Mutual transforms: 0.000073s
[Time] [Item: 14073] ---> TOTAL __getitem__ time: 0.237609s

Batch loading time: 3.790299415588379
Model inference time: 0.9818115234375
Model optimization time: 1.5435552597045898
batch 0/3416, batch training loss: [ 7.17612696 10.15011978  0.98993385  3.29020452  2.10637498  1.79992902
  1.77920198]
Batch time: 4.889613389968872
[Time] [Item: 17561] 1. Get filenames: 0.000003s
[Time] [Item: 17561] 2. Read & resize images: 0.173219s
[Time] [Item: 17561] 3. Input & Target transforms: 0.200682s
[Time] [Item: 17561] 4. Mask validity check: 0.000092s
[Time] [Item: 17561] 5. Read sparse depth priors: 0.087392s
[Time] [Item: 17561] 6. Feature validity check: 0.000001s
[Time] 1. Empty tensor allocation: 0.000443s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000897s
  [Time] 3. get_distance_maps: 0.087817s
  [Time] 4. torch.min: 0.000412s
  [Time] 5. Prior/Dist assignment: 0.000265s
[Time] 6. get_probability_maps: 0.001174s
[Time] 7. torch.cat: 0.000073s
[Time] ---> TOTAL get_depth_prior time: 0.091535s

[Time] [Item: 17561] 7. Dense parametrization: 0.091746s
[Time] [Item: 17561] 8. Target + prior transform: 0.000215s
[Time] [Item: 17561] 9. Mutual transforms: 0.000070s
[Time] [Item: 17561] ---> TOTAL __getitem__ time: 0.574159s

[Time] [Item: 4901] 1. Get filenames: 0.000002s
[Time] [Item: 4901] 2. Read & resize images: 0.064922s
[Time] [Item: 4901] 3. Input & Target transforms: 0.007245s
[Time] [Item: 4901] 4. Mask validity check: 0.000090s
[Time] [Item: 4901] 5. Read sparse depth priors: 0.061381s
[Time] [Item: 4901] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000458s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000973s
  [Time] 3. get_distance_maps: 0.088360s
  [Time] 4. torch.min: 0.000453s
  [Time] 5. Prior/Dist assignment: 0.000239s
[Time] 6. get_probability_maps: 0.001336s
[Time] 7. torch.cat: 0.000067s
[Time] ---> TOTAL get_depth_prior time: 0.092346s

[Time] [Item: 4901] 7. Dense parametrization: 0.092554s
[Time] [Item: 4901] 8. Target + prior transform: 0.000201s
[Time] [Item: 4901] 9. Mutual transforms: 0.000063s
[Time] [Item: 4901] ---> TOTAL __getitem__ time: 0.226741s

[Time] [Item: 6009] 1. Get filenames: 0.000002s
[Time] [Item: 6009] 2. Read & resize images: 0.049040s
[Time] [Item: 6009] 3. Input & Target transforms: 0.006760s
[Time] [Item: 6009] 4. Mask validity check: 0.000075s
[Time] [Item: 6009] 5. Read sparse depth priors: 0.064105s
[Time] [Item: 6009] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000427s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000737s
  [Time] 3. get_distance_maps: 0.088928s
  [Time] 4. torch.min: 0.000406s
  [Time] 5. Prior/Dist assignment: 0.000230s
[Time] 6. get_probability_maps: 0.001225s
[Time] 7. torch.cat: 0.000099s
[Time] ---> TOTAL get_depth_prior time: 0.092537s

[Time] [Item: 6009] 7. Dense parametrization: 0.092736s
[Time] [Item: 6009] 8. Target + prior transform: 0.000209s
[Time] [Item: 6009] 9. Mutual transforms: 0.000068s
[Time] [Item: 6009] ---> TOTAL __getitem__ time: 0.213193s

[Time] [Item: 10785] 1. Get filenames: 0.000002s
[Time] [Item: 10785] 2. Read & resize images: 0.326987s
[Time] [Item: 10785] 3. Input & Target transforms: 0.007271s
[Time] [Item: 10785] 4. Mask validity check: 0.000077s
[Time] [Item: 10785] 5. Read sparse depth priors: 0.059133s
[Time] [Item: 10785] 6. Feature validity check: 0.000001s
[Time] 1. Empty tensor allocation: 0.000455s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000971s
  [Time] 3. get_distance_maps: 0.087801s
  [Time] 4. torch.min: 0.000425s
  [Time] 5. Prior/Dist assignment: 0.000240s
[Time] 6. get_probability_maps: 0.001151s
[Time] 7. torch.cat: 0.000046s
[Time] ---> TOTAL get_depth_prior time: 0.091582s

[Time] [Item: 10785] 7. Dense parametrization: 0.091755s
[Time] [Item: 10785] 8. Target + prior transform: 0.000212s
[Time] [Item: 10785] 9. Mutual transforms: 0.000072s
[Time] [Item: 10785] ---> TOTAL __getitem__ time: 0.485746s

[Time] [Item: 20058] 1. Get filenames: 0.000015s
[Time] [Item: 20058] 2. Read & resize images: 0.083002s
[Time] [Item: 20058] 3. Input & Target transforms: 0.007147s
[Time] [Item: 20058] 4. Mask validity check: 0.000086s
[Time] [Item: 20058] 5. Read sparse depth priors: 0.092022s
[Time] [Item: 20058] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000517s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.001134s
  [Time] 3. get_distance_maps: 0.092492s
  [Time] 4. torch.min: 0.000442s
  [Time] 5. Prior/Dist assignment: 0.000238s
[Time] 6. get_probability_maps: 0.001248s
[Time] 7. torch.cat: 0.000093s
[Time] ---> TOTAL get_depth_prior time: 0.096626s

[Time] [Item: 20058] 7. Dense parametrization: 0.096804s
[Time] [Item: 20058] 8. Target + prior transform: 0.000186s
[Time] [Item: 20058] 9. Mutual transforms: 0.000073s
[Time] [Item: 20058] ---> TOTAL __getitem__ time: 0.279561s

[Time] [Item: 379] 1. Get filenames: 0.000002s
[Time] [Item: 379] 2. Read & resize images: 0.054456s
[Time] [Item: 379] 3. Input & Target transforms: 0.007037s
[Time] [Item: 379] 4. Mask validity check: 0.000121s
[Time] [Item: 379] 5. Read sparse depth priors: 0.059457s
[Time] [Item: 379] 6. Feature validity check: 0.000001s
[Time] 1. Empty tensor allocation: 0.000461s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000932s
  [Time] 3. get_distance_maps: 0.093394s
  [Time] 4. torch.min: 0.000422s
  [Time] 5. Prior/Dist assignment: 0.000248s
[Time] 6. get_probability_maps: 0.001326s
[Time] 7. torch.cat: 0.000108s
[Time] ---> TOTAL get_depth_prior time: 0.097304s

[Time] [Item: 379] 7. Dense parametrization: 0.097498s
[Time] [Item: 379] 8. Target + prior transform: 0.000198s
[Time] [Item: 379] 9. Mutual transforms: 0.000271s
[Time] [Item: 379] ---> TOTAL __getitem__ time: 0.219275s

Batch loading time: 1.9996991157531738
Model inference time: 0.03250432014465332
Model optimization time: 0.5244817733764648
Batch time: 1.267104148864746
[Time] [Item: 971] 1. Get filenames: 0.000002s
[Time] [Item: 971] 2. Read & resize images: 0.287174s
[Time] [Item: 971] 3. Input & Target transforms: 0.005758s
[Time] [Item: 971] 4. Mask validity check: 0.000077s
[Time] [Item: 971] 5. Read sparse depth priors: 0.042272s
[Time] [Item: 971] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000438s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000856s
  [Time] 3. get_distance_maps: 0.084566s
  [Time] 4. torch.min: 0.000381s
  [Time] 5. Prior/Dist assignment: 0.000249s
[Time] 6. get_probability_maps: 0.001311s
[Time] 7. torch.cat: 0.000096s
[Time] ---> TOTAL get_depth_prior time: 0.088321s

[Time] [Item: 971] 7. Dense parametrization: 0.088516s
[Time] [Item: 971] 8. Target + prior transform: 0.000211s
[Time] [Item: 971] 9. Mutual transforms: 0.000285s
[Time] [Item: 971] ---> TOTAL __getitem__ time: 0.424568s

[Time] [Item: 18284] 1. Get filenames: 0.000002s
[Time] [Item: 18284] 2. Read & resize images: 0.345002s
[Time] [Item: 18284] 3. Input & Target transforms: 0.006345s
[Time] [Item: 18284] 4. Mask validity check: 0.000084s
[Time] [Item: 18284] 5. Read sparse depth priors: 0.072124s
[Time] [Item: 18284] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000408s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000706s
  [Time] 3. get_distance_maps: 0.086357s
  [Time] 4. torch.min: 0.000392s
  [Time] 5. Prior/Dist assignment: 0.000280s
[Time] 6. get_probability_maps: 0.001355s
[Time] 7. torch.cat: 0.000117s
[Time] ---> TOTAL get_depth_prior time: 0.090026s

[Time] [Item: 18284] 7. Dense parametrization: 0.090250s
[Time] [Item: 18284] 8. Target + prior transform: 0.000187s
[Time] [Item: 18284] 9. Mutual transforms: 0.000284s
[Time] [Item: 18284] ---> TOTAL __getitem__ time: 0.514492s

[Time] [Item: 5967] 1. Get filenames: 0.000004s
[Time] [Item: 5967] 2. Read & resize images: 0.288952s
[Time] [Item: 5967] 3. Input & Target transforms: 0.005910s
[Time] [Item: 5967] 4. Mask validity check: 0.000093s
[Time] [Item: 5967] 5. Read sparse depth priors: 0.057623s
[Time] [Item: 5967] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000431s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000938s
  [Time] 3. get_distance_maps: 0.087612s
  [Time] 4. torch.min: 0.000383s
  [Time] 5. Prior/Dist assignment: 0.000245s
[Time] 6. get_probability_maps: 0.001163s
[Time] 7. torch.cat: 0.000095s
[Time] ---> TOTAL get_depth_prior time: 0.091293s

[Time] [Item: 5967] 7. Dense parametrization: 0.091506s
[Time] [Item: 5967] 8. Target + prior transform: 0.000207s
[Time] [Item: 5967] 9. Mutual transforms: 0.000231s
[Time] [Item: 5967] ---> TOTAL __getitem__ time: 0.444774s

[Time] [Item: 6793] 1. Get filenames: 0.000003s
[Time] [Item: 6793] 2. Read & resize images: 0.318009s
[Time] [Item: 6793] 3. Input & Target transforms: 0.006335s
[Time] [Item: 6793] 4. Mask validity check: 0.000090s
[Time] [Item: 6793] 5. Read sparse depth priors: 0.061547s
[Time] [Item: 6793] 6. Feature validity check: 0.000001s
[Time] 1. Empty tensor allocation: 0.000506s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000956s
  [Time] 3. get_distance_maps: 0.090789s
  [Time] 4. torch.min: 0.000384s
  [Time] 5. Prior/Dist assignment: 0.000256s
[Time] 6. get_probability_maps: 0.001196s
[Time] 7. torch.cat: 0.000085s
[Time] ---> TOTAL get_depth_prior time: 0.094641s

[Time] [Item: 6793] 7. Dense parametrization: 0.094833s
[Time] [Item: 6793] 8. Target + prior transform: 0.000227s
[Time] [Item: 6793] 9. Mutual transforms: 0.000295s
[Time] [Item: 6793] ---> TOTAL __getitem__ time: 0.481623s

[Time] [Item: 18137] 1. Get filenames: 0.000002s
[Time] [Item: 18137] 2. Read & resize images: 0.356154s
[Time] [Item: 18137] 3. Input & Target transforms: 0.005938s
[Time] [Item: 18137] 4. Mask validity check: 0.000083s
[Time] [Item: 18137] 5. Read sparse depth priors: 0.096160s
[Time] [Item: 18137] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000446s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000965s
  [Time] 3. get_distance_maps: 0.091260s
  [Time] 4. torch.min: 0.000359s
  [Time] 5. Prior/Dist assignment: 0.000246s
[Time] 6. get_probability_maps: 0.001052s
[Time] 7. torch.cat: 0.000066s
[Time] ---> TOTAL get_depth_prior time: 0.094777s

[Time] [Item: 18137] 7. Dense parametrization: 0.094978s
[Time] [Item: 18137] 8. Target + prior transform: 0.000187s
[Time] [Item: 18137] 9. Mutual transforms: 0.000299s
[Time] [Item: 18137] ---> TOTAL __getitem__ time: 0.554072s

[Time] [Item: 12574] 1. Get filenames: 0.000002s
[Time] [Item: 12574] 2. Read & resize images: 0.067427s
[Time] [Item: 12574] 3. Input & Target transforms: 0.006402s
[Time] [Item: 12574] 4. Mask validity check: 0.000084s
[Time] [Item: 12574] 5. Read sparse depth priors: 0.047807s
[Time] [Item: 12574] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000450s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000945s
  [Time] 3. get_distance_maps: 0.085846s
  [Time] 4. torch.min: 0.000402s
  [Time] 5. Prior/Dist assignment: 0.000221s
[Time] 6. get_probability_maps: 0.000821s
[Time] 7. torch.cat: 0.000043s
[Time] ---> TOTAL get_depth_prior time: 0.089208s

[Time] [Item: 12574] 7. Dense parametrization: 0.089395s
[Time] [Item: 12574] 8. Target + prior transform: 0.000189s
[Time] [Item: 12574] 9. Mutual transforms: 0.000280s
[Time] [Item: 12574] ---> TOTAL __getitem__ time: 0.211805s

Batch loading time: 2.6324033737182617
Model inference time: 0.028865575790405273
Model optimization time: 0.361253023147583
Batch time: 0.8347072601318359
[Time] [Item: 10164] 1. Get filenames: 0.000004s
[Time] [Item: 10164] 2. Read & resize images: 0.242480s
[Time] [Item: 10164] 3. Input & Target transforms: 0.005992s
[Time] [Item: 10164] 4. Mask validity check: 0.000083s
[Time] [Item: 10164] 5. Read sparse depth priors: 0.039779s
[Time] [Item: 10164] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000443s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000540s
  [Time] 3. get_distance_maps: 0.091123s
  [Time] 4. torch.min: 0.000415s
  [Time] 5. Prior/Dist assignment: 0.000254s
[Time] 6. get_probability_maps: 0.001314s
[Time] 7. torch.cat: 0.000121s
[Time] ---> TOTAL get_depth_prior time: 0.094706s

[Time] [Item: 10164] 7. Dense parametrization: 0.094880s
[Time] [Item: 10164] 8. Target + prior transform: 0.000188s
[Time] [Item: 10164] 9. Mutual transforms: 0.000068s
[Time] [Item: 10164] ---> TOTAL __getitem__ time: 0.383767s

[Time] [Item: 15623] 1. Get filenames: 0.000004s
[Time] [Item: 15623] 2. Read & resize images: 0.291489s
[Time] [Item: 15623] 3. Input & Target transforms: 0.006425s
[Time] [Item: 15623] 4. Mask validity check: 0.000086s
[Time] [Item: 15623] 5. Read sparse depth priors: 0.119224s
[Time] [Item: 15623] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000497s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.001020s
  [Time] 3. get_distance_maps: 0.081417s
  [Time] 4. torch.min: 0.000375s
  [Time] 5. Prior/Dist assignment: 0.000259s
[Time] 6. get_probability_maps: 0.001035s
[Time] 7. torch.cat: 0.000081s
[Time] ---> TOTAL get_depth_prior time: 0.085130s

[Time] [Item: 15623] 7. Dense parametrization: 0.085361s
[Time] [Item: 15623] 8. Target + prior transform: 0.000236s
[Time] [Item: 15623] 9. Mutual transforms: 0.000271s
[Time] [Item: 15623] ---> TOTAL __getitem__ time: 0.503348s

[Time] [Item: 14332] 1. Get filenames: 0.000002s
[Time] [Item: 14332] 2. Read & resize images: 0.272715s
[Time] [Item: 14332] 3. Input & Target transforms: 0.006161s
[Time] [Item: 14332] 4. Mask validity check: 0.000095s
[Time] [Item: 14332] 5. Read sparse depth priors: 0.075283s
[Time] [Item: 14332] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000473s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000827s
  [Time] 3. get_distance_maps: 0.071681s
  [Time] 4. torch.min: 0.000356s
  [Time] 5. Prior/Dist assignment: 0.000257s
[Time] 6. get_probability_maps: 0.001098s
[Time] 7. torch.cat: 0.000112s
[Time] ---> TOTAL get_depth_prior time: 0.075308s

[Time] [Item: 14332] 7. Dense parametrization: 0.075563s
[Time] [Item: 14332] 8. Target + prior transform: 0.000217s
[Time] [Item: 14332] 9. Mutual transforms: 0.000299s
[Time] [Item: 14332] ---> TOTAL __getitem__ time: 0.430618s

[Time] [Item: 10839] 1. Get filenames: 0.000002s
[Time] [Item: 10839] 2. Read & resize images: 0.067840s
[Time] [Item: 10839] 3. Input & Target transforms: 0.006012s
[Time] [Item: 10839] 4. Mask validity check: 0.000083s
[Time] [Item: 10839] 5. Read sparse depth priors: 0.062037s
[Time] [Item: 10839] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000409s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000837s
  [Time] 3. get_distance_maps: 0.086554s
  [Time] 4. torch.min: 0.000417s
  [Time] 5. Prior/Dist assignment: 0.000227s
[Time] 6. get_probability_maps: 0.001290s
[Time] 7. torch.cat: 0.000086s
[Time] ---> TOTAL get_depth_prior time: 0.090283s

[Time] [Item: 10839] 7. Dense parametrization: 0.090488s
[Time] [Item: 10839] 8. Target + prior transform: 0.000178s
[Time] [Item: 10839] 9. Mutual transforms: 0.000070s
[Time] [Item: 10839] ---> TOTAL __getitem__ time: 0.226945s

[Time] [Item: 4320] 1. Get filenames: 0.000003s
[Time] [Item: 4320] 2. Read & resize images: 0.302892s
[Time] [Item: 4320] 3. Input & Target transforms: 0.005471s
[Time] [Item: 4320] 4. Mask validity check: 0.000056s
[Time] [Item: 4320] 5. Read sparse depth priors: 0.106765s
[Time] [Item: 4320] 6. Feature validity check: 0.000001s
[Time] 1. Empty tensor allocation: 0.000424s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000800s
  [Time] 3. get_distance_maps: 0.085851s
  [Time] 4. torch.min: 0.000402s
  [Time] 5. Prior/Dist assignment: 0.000231s
[Time] 6. get_probability_maps: 0.001255s
[Time] 7. torch.cat: 0.000113s
[Time] ---> TOTAL get_depth_prior time: 0.089608s

[Time] [Item: 4320] 7. Dense parametrization: 0.089865s
[Time] [Item: 4320] 8. Target + prior transform: 0.000210s
[Time] [Item: 4320] 9. Mutual transforms: 0.000076s
[Time] [Item: 4320] ---> TOTAL __getitem__ time: 0.505901s

[Time] [Item: 13114] 1. Get filenames: 0.000002s
[Time] [Item: 13114] 2. Read & resize images: 0.289062s
[Time] [Item: 13114] 3. Input & Target transforms: 0.006721s
[Time] [Item: 13114] 4. Mask validity check: 0.000088s
[Time] [Item: 13114] 5. Read sparse depth priors: 0.119902s
[Time] [Item: 13114] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000564s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000918s
  [Time] 3. get_distance_maps: 0.090866s
  [Time] 4. torch.min: 0.000428s
  [Time] 5. Prior/Dist assignment: 0.000269s
[Time] 6. get_probability_maps: 0.001438s
[Time] 7. torch.cat: 0.000085s
[Time] ---> TOTAL get_depth_prior time: 0.095069s

[Time] [Item: 13114] 7. Dense parametrization: 0.095317s
[Time] [Item: 13114] 8. Target + prior transform: 0.000179s
[Time] [Item: 13114] 9. Mutual transforms: 0.000283s
[Time] [Item: 13114] ---> TOTAL __getitem__ time: 0.512386s

Batch loading time: 2.5640156269073486
Model inference time: 0.03465008735656738
Model optimization time: 0.3165152072906494
Batch time: 0.7231342792510986
[Time] [Item: 6769] 1. Get filenames: 0.000002s
[Time] [Item: 6769] 2. Read & resize images: 0.075033s
[Time] [Item: 6769] 3. Input & Target transforms: 0.006005s
[Time] [Item: 6769] 4. Mask validity check: 0.000087s
[Time] [Item: 6769] 5. Read sparse depth priors: 0.053833s
[Time] [Item: 6769] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000411s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000908s
  [Time] 3. get_distance_maps: 0.091066s
  [Time] 4. torch.min: 0.000379s
  [Time] 5. Prior/Dist assignment: 0.000254s
[Time] 6. get_probability_maps: 0.001357s
[Time] 7. torch.cat: 0.000095s
[Time] ---> TOTAL get_depth_prior time: 0.094922s

[Time] [Item: 6769] 7. Dense parametrization: 0.095112s
[Time] [Item: 6769] 8. Target + prior transform: 0.000204s
[Time] [Item: 6769] 9. Mutual transforms: 0.000081s
[Time] [Item: 6769] ---> TOTAL __getitem__ time: 0.230584s

[Time] [Item: 1924] 1. Get filenames: 0.000002s
[Time] [Item: 1924] 2. Read & resize images: 0.312304s
[Time] [Item: 1924] 3. Input & Target transforms: 0.006276s
[Time] [Item: 1924] 4. Mask validity check: 0.000085s
[Time] [Item: 1924] 5. Read sparse depth priors: 0.073181s
[Time] [Item: 1924] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000452s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000940s
  [Time] 3. get_distance_maps: 0.089948s
  [Time] 4. torch.min: 0.000419s
  [Time] 5. Prior/Dist assignment: 0.000237s
[Time] 6. get_probability_maps: 0.001321s
[Time] 7. torch.cat: 0.000104s
[Time] ---> TOTAL get_depth_prior time: 0.093928s

[Time] [Item: 1924] 7. Dense parametrization: 0.094147s
[Time] [Item: 1924] 8. Target + prior transform: 0.000161s
[Time] [Item: 1924] 9. Mutual transforms: 0.000045s
[Time] [Item: 1924] ---> TOTAL __getitem__ time: 0.486437s

[Time] [Item: 265] 1. Get filenames: 0.000002s
[Time] [Item: 265] 2. Read & resize images: 0.277628s
[Time] [Item: 265] 3. Input & Target transforms: 0.006124s
[Time] [Item: 265] 4. Mask validity check: 0.000110s
[Time] [Item: 265] 5. Read sparse depth priors: 0.057230s
[Time] [Item: 265] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000470s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.001062s
  [Time] 3. get_distance_maps: 0.087001s
  [Time] 4. torch.min: 0.000380s
  [Time] 5. Prior/Dist assignment: 0.000244s
[Time] 6. get_probability_maps: 0.001227s
[Time] 7. torch.cat: 0.000078s
[Time] ---> TOTAL get_depth_prior time: 0.090925s

[Time] [Item: 265] 7. Dense parametrization: 0.091147s
[Time] [Item: 265] 8. Target + prior transform: 0.000195s
[Time] [Item: 265] 9. Mutual transforms: 0.000235s
[Time] [Item: 265] ---> TOTAL __getitem__ time: 0.432911s

[Time] [Item: 17125] 1. Get filenames: 0.000002s
[Time] [Item: 17125] 2. Read & resize images: 0.067637s
[Time] [Item: 17125] 3. Input & Target transforms: 0.005801s
[Time] [Item: 17125] 4. Mask validity check: 0.000047s
[Time] [Item: 17125] 5. Read sparse depth priors: 0.065537s
[Time] [Item: 17125] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000466s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000948s
  [Time] 3. get_distance_maps: 0.087640s
  [Time] 4. torch.min: 0.000393s
  [Time] 5. Prior/Dist assignment: 0.000266s
[Time] 6. get_probability_maps: 0.001386s
[Time] 7. torch.cat: 0.000114s
[Time] ---> TOTAL get_depth_prior time: 0.091739s

[Time] [Item: 17125] 7. Dense parametrization: 0.091972s
[Time] [Item: 17125] 8. Target + prior transform: 0.000162s
[Time] [Item: 17125] 9. Mutual transforms: 0.000057s
[Time] [Item: 17125] ---> TOTAL __getitem__ time: 0.231424s

[Time] [Item: 13680] 1. Get filenames: 0.000005s
[Time] [Item: 13680] 2. Read & resize images: 0.376393s
[Time] [Item: 13680] 3. Input & Target transforms: 0.006320s
[Time] [Item: 13680] 4. Mask validity check: 0.000080s
[Time] [Item: 13680] 5. Read sparse depth priors: 0.066263s
[Time] [Item: 13680] 6. Feature validity check: 0.000000s
[Time] 1. Empty tensor allocation: 0.000454s
  --- Batch Item 0 ---
  [Time] 2. Masking & array indexing: 0.000957s
  [Time] 3. get_distance_maps: 0.088368s
  [Time] 4. torch.min: 0.000413s
  [Time] 5. Prior/Dist assignment: 0.000296s
[Time] 6. get_probability_maps: 0.001424s
[Time] 7. torch.cat: 0.000086s
[Time] ---> TOTAL get_depth_prior time: 0.092526s


"""

# python -c "import tifffile;import time;start_time=time.time();tifffile.imread('/teamspace/studios/this_studio/red_sea/big_dice_loop/imgs/16316007748831346.tiff');end_time=time.time();print(end_time-start_time)"
# python -c "from PIL import Image;import time;start_time=time.time();Image.open('/teamspace/studios/this_studio/red_sea/big_dice_loop/imgs/16316007748831346.tiff');end_time=time.time();print(end_time-start_time)"