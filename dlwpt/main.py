from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks.progress import RichProgressBar


from dlwpt.utils import get_mnist_datasets


@hydra.main(version_base="1.3", config_name="config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    train, test = get_mnist_datasets(do_augment=False)
    train_loader = DataLoader(
        train,
        batch_size=cfg.hparams.batch,
        shuffle=True,
        num_workers=cfg.hparams.num_workers,
    )
    test_loader = DataLoader(
        test, batch_size=cfg.hparams.batch, num_workers=cfg.hparams.num_workers
    )

    aim_logger = hydra.utils.instantiate(cfg.logger)
    mod = hydra.utils.instantiate(cfg.experiment.model)
    trainer = L.Trainer(
        max_epochs=cfg.hparams.epochs,
        accelerator="mps",
        devices=-1,
        logger=aim_logger,
        callbacks=[RichProgressBar(refresh_rate=5, leave=True)],
    )
    trainer.fit(mod, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    main()
