"""Request and response schemas for the prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class AbaloneFeatures(BaseModel):
    """One abalone's measurements, named as in the training CSV."""

    sex: Literal["M", "F", "I"] = Field(..., alias="Sex")
    length: float = Field(..., alias="Length", ge=0)
    diameter: float = Field(..., alias="Diameter", ge=0)
    height: float = Field(..., alias="Height", ge=0)
    whole_weight: float = Field(..., alias="Whole weight", ge=0)
    shucked_weight: float = Field(..., alias="Shucked weight", ge=0)
    viscera_weight: float = Field(..., alias="Viscera weight", ge=0)
    shell_weight: float = Field(..., alias="Shell weight", ge=0)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "Sex": "M",
                "Length": 0.455,
                "Diameter": 0.365,
                "Height": 0.095,
                "Whole weight": 0.514,
                "Shucked weight": 0.2245,
                "Viscera weight": 0.101,
                "Shell weight": 0.15,
            }
        }


class RingsPrediction(BaseModel):
    """Predicted ring count; age in years is roughly rings + 1.5."""

    rings: float
