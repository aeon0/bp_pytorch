from pydantic import BaseModel


class ModelCfg(BaseModel):
    lr: float


class TrainCfg(BaseModel):
    model: ModelCfg
    max_epochs: int
    accelerator: str
    devices: int
    batch_size: int
    num_workers: int


class PredictCfg(BaseModel):
    accelerator: str
    devices: int
    batch_size: int
    num_workers: int
