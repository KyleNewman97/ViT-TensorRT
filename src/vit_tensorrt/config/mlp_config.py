from pydantic import BaseModel, Field


class MLPConfig(BaseModel):
    hidden_layer_sizes: list[int] = Field(default=[3072])
