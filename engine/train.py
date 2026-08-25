"""Training loop: full multi-epoch training, checkpointing/resume,
TensorBoard logging, and the mandatory --overfit-one-batch sanity gate.

Usage:
    .venv\\Scripts\\python.exe -m engine.train --overfit-one-batch
    .venv\\Scripts\\python.exe -m engine.train
    .venv\\Scripts\\python.exe -m engine.train --resume checkpoints/last.pth
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import WiderFaceDataset
from data.transforms import build_eval_transform, build_train_transform
from losses.loss import DetectionLoss
from models.detector import FaceDetector
from utils.boxes import postprocess_batch
from utils.logging import AverageMeter, get_logger

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = get_logger()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    requested = cfg["env"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("env.device is 'cuda' but CUDA is not available - falling back to CPU. Training will be slow.")
        return torch.device("cpu")
    return torch.device(requested)


def collate_fn(batch: list[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch])
    return {
        "images": images,
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def build_dataloader(
    cfg: dict, split: str, shuffle: bool, batch_size: int | None = None, num_workers: int | None = None
) -> DataLoader:
    transform = build_train_transform(cfg) if split == "train" else build_eval_transform(cfg)
    dataset = WiderFaceDataset(
        root=REPO_ROOT / cfg["data"]["root"],
        split=split,
        transform=transform,
        min_face_size=cfg["data"]["min_face_size"],
        drop_invalid=cfg["data"]["drop_invalid"],
        drop_empty_images=cfg["data"]["drop_empty_images"],
    )
    workers = cfg["training"]["num_workers"] if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=workers > 0,
    )


def build_optimizer(model: torch.nn.Module, cfg: dict, lr_override: float | None = None) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    lr = lr_override if lr_override is not None else train_cfg["lr"]
    if train_cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=train_cfg["weight_decay"])
    if train_cfg["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=train_cfg["sgd_momentum"], weight_decay=train_cfg["weight_decay"]
        )
    raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    train_cfg = cfg["training"]
    warmup_steps = train_cfg["warmup_epochs"] * steps_per_epoch
    total_steps = train_cfg["epochs"] * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        if train_cfg["lr_schedule"] == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        if train_cfg["lr_schedule"] == "step":
            epoch = step / steps_per_epoch
            factor = 1.0
            for milestone in train_cfg["step_lr_milestones"]:
                if epoch >= milestone:
                    factor *= train_cfg["step_lr_gamma"]
            return factor
        raise ValueError(f"Unknown lr_schedule: {train_cfg['lr_schedule']!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val_loss: float, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    logger.info(f"Resumed from {path} at epoch {ckpt['epoch']}")
    return ckpt["epoch"] + 1, ckpt["best_val_loss"]


def run_epoch(
    model, criterion, loader: DataLoader, device: torch.device, cfg: dict,
    optimizer=None, scheduler=None, scaler=None, writer: SummaryWriter | None = None,
    epoch: int = 0, global_step: int = 0,
) -> tuple[dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)

    meters = {"loss": AverageMeter(), "cls_loss": AverageMeter(), "bbox_loss": AverageMeter()}
    amp_enabled = cfg["env"]["amp"] and device.type == "cuda"
    start = time.time()

    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        gt_boxes_list = [b.to(device, non_blocking=True) for b in batch["boxes"]]

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(images)
                loss_dict = criterion(out["cls_logits"], out["bbox_deltas"], model.anchors, gt_boxes_list)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_dict["loss"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
                    optimizer.step()
                scheduler.step()
                global_step += 1

        bs = images.shape[0]
        meters["loss"].update(loss_dict["loss"].item(), bs)
        meters["cls_loss"].update(loss_dict["cls_loss"].item(), bs)
        meters["bbox_loss"].update(loss_dict["bbox_loss"].item(), bs)

        if is_train and step % cfg["training"]["log_every_n_steps"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            imgs_per_sec = (step + 1) * bs / max(elapsed, 1e-6)
            logger.info(
                f"epoch {epoch} step {step}/{len(loader)} "
                f"loss={loss_dict['loss'].item():.4f} cls={loss_dict['cls_loss'].item():.4f} "
                f"bbox={loss_dict['bbox_loss'].item():.4f} lr={lr:.6f} img/s={imgs_per_sec:.1f}"
            )
            if writer is not None:
                writer.add_scalar("train/loss_step", loss_dict["loss"].item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

    return {k: m.avg for k, m in meters.items()}, global_step


def train(cfg: dict, resume: str | None = None) -> None:
    set_seed(cfg["project"]["seed"])
    torch.backends.cudnn.benchmark = cfg["env"]["cudnn_benchmark"]
    device = get_device(cfg)
    logger.info(f"device: {device}")

    model = FaceDetector(cfg).to(device)
    criterion = DetectionLoss(cfg)
    train_loader = build_dataloader(cfg, "train", shuffle=True)
    val_loader = build_dataloader(cfg, "val", shuffle=False)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["env"]["amp"] and device.type == "cuda")

    checkpoint_dir = REPO_ROOT / cfg["training"]["checkpoint_dir"]
    last_path = checkpoint_dir / "last.pth"
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if resume:
        start_epoch, best_val_loss = load_checkpoint(Path(resume), model, optimizer, scheduler, device)
    elif last_path.exists():
        raise FileExistsError(
            f"{last_path} already exists from a previous run. Pass --resume {last_path} to continue it, "
            "or move/delete it first if you intend to start fresh - training never overwrites checkpoints silently."
        )

    writer = SummaryWriter(log_dir=str(REPO_ROOT / cfg["training"]["log_dir"]))
    logger.info(f"train: {len(train_loader.dataset)} images, val: {len(val_loader.dataset)} images")
    logger.info(f"epochs={cfg['training']['epochs']}, steps/epoch={len(train_loader)}")

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        train_metrics, global_step = run_epoch(
            model, criterion, train_loader, device, cfg,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            writer=writer, epoch=epoch, global_step=global_step,
        )
        val_metrics, _ = run_epoch(model, criterion, val_loader, device, cfg, epoch=epoch)

        logger.info(
            f"epoch {epoch} done - train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}"
        )
        writer.add_scalar("train/loss_epoch", train_metrics["loss"], epoch)
        writer.add_scalar("val/loss_epoch", val_metrics["loss"], epoch)

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_loss, cfg)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, scheduler, epoch, best_val_loss, cfg)
            logger.info(f"new best val_loss={best_val_loss:.4f}, saved best.pth")

    writer.close()


def overfit_one_batch(cfg: dict) -> None:
    """Train on a single fixed batch until loss collapses and boxes visibly
    lock onto the GT faces. This is the gate that proves data -> anchors ->
    model -> loss -> decode -> draw are all wired together correctly -
    required to pass before any real (multi-hour) training run."""
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg)
    logger.info(f"device: {device}")

    model = FaceDetector(cfg).to(device)
    criterion = DetectionLoss(cfg)
    overfit_cfg = cfg["training"]["overfit"]
    train_loader = build_dataloader(cfg, "train", shuffle=True, batch_size=overfit_cfg["batch_size"], num_workers=0)

    batch = next(iter(train_loader))
    images = batch["images"].to(device)
    gt_boxes_list = [b.to(device) for b in batch["boxes"]]
    num_gt = sum(len(b) for b in gt_boxes_list)
    logger.info(f"overfitting on a single batch of {images.shape[0]} images, {num_gt} GT faces total")

    optimizer = build_optimizer(model, cfg, lr_override=overfit_cfg["lr"])

    model.train()
    losses = []
    for step in range(overfit_cfg["steps"]):
        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        loss_dict = criterion(out["cls_logits"], out["bbox_deltas"], model.anchors, gt_boxes_list)
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
        optimizer.step()
        losses.append(loss_dict["loss"].item())

        if step % 20 == 0 or step == overfit_cfg["steps"] - 1:
            logger.info(
                f"step {step}/{overfit_cfg['steps']} loss={loss_dict['loss'].item():.4f} "
                f"cls={loss_dict['cls_loss'].item():.4f} bbox={loss_dict['bbox_loss'].item():.4f} "
                f"num_pos={loss_dict['num_pos'].item():.0f}"
            )

    logger.info(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    out_dir = REPO_ROOT / "outputs" / "overfit_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_overfit_visualization(model, images, gt_boxes_list, cfg, out_dir)
    logger.info(f"wrote visualization to {out_dir}")

    return losses


def _save_overfit_visualization(model, images: torch.Tensor, gt_boxes_list: list[torch.Tensor], cfg: dict, out_dir: Path) -> None:
    model.eval()
    with torch.no_grad():
        out = model(images)

    overfit_cfg = cfg["training"]["overfit"]
    variances = tuple(cfg["anchors"]["variances"])
    size = cfg["data"]["image_size"]
    predictions = postprocess_batch(
        out["cls_logits"], out["bbox_deltas"], model.anchors, variances,
        image_width=size, image_height=size,
        conf_thresh=overfit_cfg["conf_thresh"], nms_thresh=overfit_cfg["nms_thresh"],
    )

    norm_cfg = cfg["augmentation"]["normalize"]
    mean = torch.tensor(norm_cfg["mean"], device=images.device).view(3, 1, 1)
    std = torch.tensor(norm_cfg["std"], device=images.device).view(3, 1, 1)

    for i in range(images.shape[0]):
        img = (images[i] * std + mean).clamp(0, 1)
        # Already BGR (cv2.imread never gets converted to RGB anywhere in the
        # pipeline - Normalize's mean/std just apply in that native channel
        # order, which is fine for a from-scratch model). No color conversion
        # needed before cv2.imwrite, which also expects BGR.
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

        for x1, y1, x2, y2 in gt_boxes_list[i].cpu().numpy():
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # GT in green

        pred_boxes, pred_scores = predictions[i]
        for (x1, y1, x2, y2), score in zip(pred_boxes.cpu().numpy(), pred_scores.cpu().numpy()):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)  # predictions in red
            cv2.putText(img, f"{score:.2f}", (int(x1), max(int(y1) - 4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imwrite(str(out_dir / f"{i:02d}.jpg"), img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.overfit_one_batch:
        overfit_one_batch(cfg)
    else:
        train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
