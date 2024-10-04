from pydantic import BaseModel, Field


class AttentionConfig(BaseModel):
    num_heads: int = Field(default=16)
    dropout: float = Field(default=0)
