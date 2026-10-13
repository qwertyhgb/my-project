"""V2 native Trainer hooks: architecture, prior contracts, auxiliary loss, phase."""
import json
import re
from pathlib import Path

from zonal_reliability_fusion.anatomy.dataset import validate_predicted_prior_dataset
from zonal_reliability_fusion.lesion.prior_channels import restrict_intensity_transforms_to_mri
from zonal_reliability_fusion.nnunet.bases import nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT
from zonal_reliability_fusion.v2.losses.coarse_loss import CoarseAuxiliaryLoss
from zonal_reliability_fusion.v2.models.anatomy_guided_c2f import AnatomyGuidedC2F, ConcatFusionUNet

PHASE_BOUNDARIES = (200, 600)


def log_epoch_progress(trainer, phase):
    """Native epoch hook adds total/rate/ETA without replacing the training loop."""
    history = trainer.logger.my_fantastic_logging
    durations = [end - start for start, end in zip(
        history["epoch_start_timestamps"], history["epoch_end_timestamps"], strict=False)
        if end > start]
    mean = sum(durations) / len(durations) if durations else None
    rate = f"{3600 / mean:.2f} epochs/hour" if mean else "pending first epoch"
    eta = f"{mean * (trainer.num_epochs - trainer.current_epoch):.1f}s" if mean else "pending"
    trainer.print_to_log_file(
        f"VERSION=V2 phase={phase} epoch={trainer.current_epoch + 1}/{trainer.num_epochs} "
        f"progress={100 * trainer.current_epoch / trainer.num_epochs:.1f}% rate={rate} ETA={eta}")


class _V2Mixin:
    research_version = "V2"
    architecture_family = "multi_encoder"
    fusion_type = "simple"
    anatomy_conditioned = False
    coarse_head = False
    refiner = False
    phase_scheduled = False
    concat_fusion = False
    phase_boundaries = PHASE_BOUNDARIES

    @classmethod
    def build_network_architecture(cls, architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        expected = 6 if cls.anatomy_conditioned else 3
        if num_input_channels != expected:
            raise ValueError(f"{cls.__name__} requires {expected} input channels")
        if cls.concat_fusion:
            return ConcatFusionUNet(
                architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                num_output_channels, enable_deep_supervision, fusion_type=cls.fusion_type)
        return AnatomyGuidedC2F(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_output_channels, enable_deep_supervision, fusion_type=cls.fusion_type,
            coarse_head=cls.coarse_head, refiner=cls.refiner)

    def _do_i_compile(self):
        return False  # Auxiliary side channel and explicit phase changes stay visible.

    @classmethod
    def phase_for_epoch(cls, epoch):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be nonnegative integer")
        if not cls.phase_scheduled:
            return "V2-P3" if cls.refiner else "V2-P2" if cls.fusion_type != "simple" else "V2-P1"
        return "V2-P1" if epoch < cls.phase_boundaries[0] else (
            "V2-P2" if epoch < cls.phase_boundaries[1] else "V2-P3")

    def _validate_v2_contract(self):
        if self.is_ddp or self.configuration_name != "3d_fullres" or self.fold != 0:
            raise ValueError("V2 first version requires single process / 3d_fullres / fold 0")
        if self.configuration_manager.previous_stage_name is not None:
            raise ValueError("V2 does not support cascade inputs")
        if self.label_manager.has_regions or self.label_manager.ignore_label is not None:
            raise ValueError("V2 requires binary lesion classes without ignore labels")
        if self.dataset_json.get("labels") != {"background": 0, "lesion": 1}:
            raise ValueError("V2 requires binary lesion labels")
        if self.anatomy_conditioned:
            contract = validate_predicted_prior_dataset(self.dataset_json, self.plans_manager.dataset_name)
            if (self.configuration_manager.preprocessor_name != "PredictedAnatomyPreprocessor"
                    or self.configuration_manager.normalization_schemes[3:] != ["NoNormalization"] * 3):
                raise ValueError("V2 predicted priors require audited preprocessing / noNorm")
        else:
            contract = None
            if (self.plans_manager.dataset_name != "Dataset605_PICAI"
                    or list(self.dataset_json.get("channel_names", {}).values()) != ["T2W", "ADC", "HBV"]):
                raise ValueError("V2 image-only arms require Dataset605 T2W/ADC/HBV")
        split = json.loads((Path(self.preprocessed_dataset_folder_base) / "splits_final.json").read_text())[0]
        train, val = split["train"], split["val"]
        if any(not isinstance(c, str) or re.fullmatch(r"[0-9]+_[0-9]+", c) is None for c in [*train, *val]):
            raise ValueError("invalid case ID in fixed split")
        if not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val):
            raise ValueError("empty or duplicate case IDs in fixed split")
        if {c.split("_")[0] for c in train} & {c.split("_")[0] for c in val}:
            raise ValueError("patient split leakage")
        if contract and (train != contract["train_cases"] or val != contract["val_cases"]):
            raise ValueError("predicted prior provenance differs from fixed lesion split")
        return train, val

    def initialize(self):
        self._validate_v2_contract()
        from zonal_reliability_fusion.nnunet.prior_preprocessor import (
            install_prior_preprocessor_resolver,
        )

        install_prior_preprocessor_resolver()
        super().initialize()
        self.network.set_phase(self.phase_for_epoch(self.current_epoch))

    def do_split(self):
        train, val = self._validate_v2_contract()
        if self.dataset_class is not None:
            ids = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            if set(ids) != set(train) | set(val):
                raise ValueError("V2 preprocessed case set differs from fixed split")
        return train, val

    @classmethod
    def get_training_transforms(cls, *args, **kwargs):
        transforms = super().get_training_transforms(*args, **kwargs)
        if cls.anatomy_conditioned:
            transforms, count = restrict_intensity_transforms_to_mri(transforms, 3)
            if count == 0:
                raise ValueError("no intensity transforms found for prior protection")
        return transforms

    def _build_loss(self):
        main = super()._build_loss()
        return CoarseAuxiliaryLoss(main, self.network) if self.coarse_head else main

    def on_epoch_start(self):
        phase = self.phase_for_epoch(self.current_epoch)
        if self.network.phase != phase:
            self.print_to_log_file(f"VERSION=V2 phase={phase} epoch={self.current_epoch}")
        self.network.set_phase(phase)
        log_epoch_progress(self, phase)
        super().on_epoch_start()

    def load_checkpoint(self, *args, **kwargs):
        # Also restore activation for native validation-only, which has no epoch hook.
        super().load_checkpoint(*args, **kwargs)
        self.network.set_phase(self.phase_for_epoch(self.current_epoch))


class nnUNetTrainerPICAI_V2_EarlyFusion_DiceCE_NoFFT(nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT):
    """V2-A: matched strong family, independent name/output, unchanged native network."""
    research_version = "V2"
    architecture_family = "early_fusion"
    fusion_type = "early"
    anatomy_conditioned = coarse_head = refiner = phase_scheduled = False

    _validate_v2_contract = _V2Mixin._validate_v2_contract

    def initialize(self):
        _V2Mixin._validate_v2_contract(self)
        super().initialize()

    def do_split(self):
        return _V2Mixin.do_split(self)

    def on_epoch_start(self):
        log_epoch_progress(self, "V2-EARLY")
        super().on_epoch_start()


class nnUNetTrainerPICAI_V2_MultiEncoder_DiceCE_NoFFT(_V2Mixin, nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT):
    """V2-B: three full native modality encoders, mean skips, shared native decoder."""


class nnUNetTrainerPICAI_V2_AdaptiveFusion_DiceCE_NoFFT(nnUNetTrainerPICAI_V2_MultiEncoder_DiceCE_NoFFT):
    """V2-C: image-adaptive fusion at selected deep scales."""
    fusion_type = "adaptive"


class nnUNetTrainerPICAI_V2_AnatomyFusion_DiceCE_NoFFT(nnUNetTrainerPICAI_V2_AdaptiveFusion_DiceCE_NoFFT):
    """V2-D: predicted zonal anatomy controls the fusion policy."""
    fusion_type = "anatomy"
    anatomy_conditioned = True


class nnUNetTrainerPICAI_V2_Coarse_DiceCE_NoFFT(nnUNetTrainerPICAI_V2_AnatomyFusion_DiceCE_NoFFT):
    """V2-E: bottleneck lesion evidence and Dice+BCE auxiliary supervision."""
    coarse_head = True


class nnUNetTrainerPICAI_V2_Refine_DiceCE_NoFFT(nnUNetTrainerPICAI_V2_Coarse_DiceCE_NoFFT):
    """V2-F: probability-localized late residual logit refiner, all-on control."""
    refiner = True


class nnUNetTrainerPICAI_V2_FullScheduled_DiceCE_NoFFT(nnUNetTrainerPICAI_V2_Refine_DiceCE_NoFFT):
    """V2-F-SCHED: same modules/budget/SGD/PolyLR, phase activation at epochs 200/600."""
    phase_scheduled = True


class nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT(nnUNetTrainerPICAI_V2_MultiEncoder_DiceCE_NoFFT):
    """B-concat: all-scale concat/3D-conv/norm/activation; no adaptive controller."""
    fusion_type = "concat"
    concat_fusion = True


class nnUNetTrainerPICAI_V2_ConcatAdaptiveFusion_DiceCE_NoFFT(nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT):
    """C-concat: identical projection plus image-only soft modality rescaling."""
    fusion_type = "concat_adaptive"


class nnUNetTrainerPICAI_V2_ConcatAnatomyFusion_DiceCE_NoFFT(nnUNetTrainerPICAI_V2_ConcatAdaptiveFusion_DiceCE_NoFFT):
    """D-concat: same image policy conditioned by predicted zonal context."""
    fusion_type = "concat_anatomy"
    anatomy_conditioned = True

