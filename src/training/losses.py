import torch
import torch.nn as nn


class SuperconductivityLoss(nn.Module):
    """Combined BCE classification and robust Tc regression loss."""

    def __init__(self, lambda_tc: float = 1.0, regression_loss: str = "smooth_l1") -> None:
        super().__init__()
        self.lambda_tc = lambda_tc
        self.cls_loss = nn.BCEWithLogitsLoss()
        if regression_loss == "mae":
            self.tc_loss = nn.L1Loss()
        elif regression_loss == "smooth_l1":
            self.tc_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported regression_loss: {regression_loss}")

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        cls = self.cls_loss(outputs["logit_supra"], batch["label_supra"])
        tc = self.tc_loss(outputs["tc"], batch["Tc"])
        total = cls + self.lambda_tc * tc
        return total, {"loss": float(total.detach().cpu()), "loss_cls": float(cls.detach().cpu()), "loss_tc": float(tc.detach().cpu())}
