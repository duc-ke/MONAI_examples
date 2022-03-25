import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.metrics import Metric
from monai.data import decollate_batch
from monai.data.nifti_writer import write_nifti
from monai.engines import SupervisedEvaluator
from monai.engines.utils import IterationEvents, default_prepare_batch
from monai.inferers import Inferer
from monai.networks.utils import eval_mode
from monai.transforms import AsDiscrete, Transform
from torch.utils.data import DataLoader

from transforms import recovery_prediction

from monai.transforms import Compose, AddChannel, MapLabelValue

class DynUNetInferrer(SupervisedEvaluator):
    """
    This class inherits from SupervisedEvaluator in MONAI, and is used with DynUNet
    on Decathlon datasets.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be
            torch.DataLoader.
        network: use the network to run model forward.
        output_dir: the path to save inferred outputs.
        num_classes: the number of classes (output channels) for the task.
        epoch_length: number of iterations for one epoch, default to
            `len(val_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        tta_val: whether to do the 8 flips (8 = 2 ** 3, where 3 represents the three dimensions)
            test time augmentation, default is False.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: DataLoader,
        network: torch.nn.Module,
        output_dir: str,                                      # infer specific
        num_classes: Union[str, int],
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        tta_val: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            inferer=inferer,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
            amp=amp,
        )  

        if not isinstance(num_classes, int):
            num_classes = int(num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.output_dir = output_dir               # infer specific
        self.tta_val = tta_val
        self.num_classes = num_classes

    def _iteration(
        self, engine: Engine, batchdata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below item in a dictionary:
            - PRED: prediction result of model.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, _ = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, _, args, kwargs = batch

        def _compute_pred():
            ct = 1.0
            pred = self.inferer(inputs, self.network, *args, **kwargs).cpu()
            pred = nn.functional.softmax(pred, dim=1)
            if not self.tta_val:
                return pred
            else:
                for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                    flip_inputs = torch.flip(inputs, dims=dims)
                    flip_pred = torch.flip(
                        self.inferer(flip_inputs, self.network).cpu(), dims=dims
                    )
                    flip_pred = nn.functional.softmax(flip_pred, dim=1)
                    del flip_inputs
                    pred += flip_pred
                    del flip_pred
                    ct += 1
                return pred / ct

        # execute forward computation
        with eval_mode(self.network):
            if self.amp:
                with torch.cuda.amp.autocast():
                    predictions = _compute_pred()
            else:
                predictions = _compute_pred()

        inputs = inputs.cpu()
        predictions = self.post_pred(decollate_batch(predictions)[0])

        affine = batchdata["image_meta_dict"]["affine"].numpy()[0]            # infer specific
        resample_flag = batchdata["resample_flag"]
        anisotrophy_flag = batchdata["anisotrophy_flag"]
        crop_shape = batchdata["crop_shape"][0].tolist()
        original_shape = batchdata["original_shape"][0].tolist()

        if resample_flag:
            # convert the prediction back to the original (after cropped) shape
            predictions = recovery_prediction(
                predictions.numpy(), [self.num_classes, *crop_shape], anisotrophy_flag
            )
        else:
            predictions = predictions.numpy()

        predictions = np.argmax(predictions, axis=0)

        # pad the prediction back to the original shape
        predictions_org = np.zeros([*original_shape])
        box_start, box_end = batchdata["bbox"][0]
        h_start, w_start, d_start = box_start
        h_end, w_end, d_end = box_end
        predictions_org[h_start:h_end, w_start:w_end, d_start:d_end] = predictions
        del predictions

        filename = batchdata["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]

        print(
            "save {} with shape: {}, mean values: {}".format(
                filename, predictions_org.shape, predictions_org.mean()
            )
        )
        write_nifti(
            data=predictions_org,
            file_name=os.path.join(self.output_dir, filename),
            affine=affine,
            resample=False,
            output_dtype=np.uint8,
        )
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        return {"pred": predictions_org}

class DynUNetInferrer_PostMapping(SupervisedEvaluator):
    """
    This class inherits from SupervisedEvaluator in MONAI, and is used with DynUNet
    on Decathlon datasets.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be
            torch.DataLoader.
        network: use the network to run model forward.
        output_dir: the path to save inferred outputs.
        num_classes: the number of classes (output channels) for the task.
        epoch_length: number of iterations for one epoch, default to
            `len(val_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        tta_val: whether to do the 8 flips (8 = 2 ** 3, where 3 represents the three dimensions)
            test time augmentation, default is False.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: DataLoader,
        network: torch.nn.Module,
        output_dir: str,                                      # infer specific
        num_classes: Union[str, int],
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        tta_val: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            inferer=inferer,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
            amp=amp,
        )  

        if not isinstance(num_classes, int):
            num_classes = int(num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.output_dir = output_dir               # infer specific
        self.tta_val = tta_val
        self.num_classes = num_classes
        
        orig_label_classes, target_label_classes = (
            np.array([   0,    2,    3,    4,    5,    7,    8,   10,   11,   12,   13,
                14,   15,   16,   17,   18,   24,   26,   28,   30,   31,   41,
                42,   43,   44,   46,   47,   49,   50,   51,   52,   53,   54,
                58,   60,   62,   63,   77,   80,   85,  251,  252,  253,  254,
                255, 1000, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
            1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
            1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
            2000, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
            2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
            2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035], dtype=np.float64),
            np.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
                65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
                78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
                91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
            104, 105, 106, 107, 108], dtype=np.float64)
        )
        self.orig_label_classes = orig_label_classes
        self.target_label_classes = target_label_classes
        self.post_trans = MapLabelValue(    # 0~108로 잡혀있는 label을 original index로 변경
                                        orig_labels=self.target_label_classes, 
                                        target_labels=self.orig_label_classes
                          )

    def _iteration(
        self, engine: Engine, batchdata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below item in a dictionary:
            - PRED: prediction result of model.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, _ = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, _, args, kwargs = batch

        def _compute_pred():
            ct = 1.0
            pred = self.inferer(inputs, self.network, *args, **kwargs).cpu()
            pred = nn.functional.softmax(pred, dim=1)
            if not self.tta_val:
                return pred
            else:
                for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                    flip_inputs = torch.flip(inputs, dims=dims)
                    flip_pred = torch.flip(
                        self.inferer(flip_inputs, self.network).cpu(), dims=dims
                    )
                    flip_pred = nn.functional.softmax(flip_pred, dim=1)
                    del flip_inputs
                    pred += flip_pred
                    del flip_pred
                    ct += 1
                return pred / ct

        # execute forward computation
        with eval_mode(self.network):
            if self.amp:
                with torch.cuda.amp.autocast():
                    predictions = _compute_pred()
            else:
                predictions = _compute_pred()

        inputs = inputs.cpu()
        predictions = self.post_pred(decollate_batch(predictions)[0])

        affine = batchdata["image_meta_dict"]["affine"].numpy()[0]            # infer specific
        resample_flag = batchdata["resample_flag"]
        anisotrophy_flag = batchdata["anisotrophy_flag"]
        crop_shape = batchdata["crop_shape"][0].tolist()
        original_shape = batchdata["original_shape"][0].tolist()

        if resample_flag:
            # convert the prediction back to the original (after cropped) shape
            predictions = recovery_prediction(
                predictions.numpy(), [self.num_classes, *crop_shape], anisotrophy_flag
            )
        else:
            predictions = predictions.numpy()

        predictions = np.argmax(predictions, axis=0)

        # pad the prediction back to the original shape
        predictions_org = np.zeros([*original_shape])
        box_start, box_end = batchdata["bbox"][0]
        h_start, w_start, d_start = box_start
        h_end, w_end, d_end = box_end
        predictions_org[h_start:h_end, w_start:w_end, d_start:d_end] = predictions
        del predictions
        
        print('변형전 dtype', predictions_org.dtype)
        print(np.unique(predictions_org))
        predictions_org = self.post_trans(predictions_org)   # 원래대로 pred index 복구
        print('변형후 dtype', predictions_org.dtype)
        print(np.unique(predictions_org))

        filename = batchdata["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]

        print(
            "save {} with shape: {}, mean values: {}".format(
                filename, predictions_org.shape, predictions_org.mean()
            )
        )
        write_nifti(
            data=predictions_org,
            file_name=os.path.join(self.output_dir, filename),
            affine=affine,
            resample=False,
            output_dtype=np.uint32,
        )
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        return {"pred": predictions_org}
