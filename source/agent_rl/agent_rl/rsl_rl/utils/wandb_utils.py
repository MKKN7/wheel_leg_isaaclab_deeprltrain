try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")

from rsl_rl.utils.wandb_utils import WandbSummaryWriter


class WandbSummaryWriterExt(WandbSummaryWriter):

    def save_model(self, model_path, iter):
        super().save_model(model_path, iter)
        artifact = wandb.Artifact(
            name=model_path.split('/')[-2],
            type="model",
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=f"Iter {iter}")
