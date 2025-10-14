from typing import List, Optional
from pydantic import BaseModel
from pandas import DatetimeTZDtype

class KMMetric(BaseModel):
    name: str
    category: Optional[str]
    value: Optional[float]

class KMR2(KMMetric):
    name: str = r"R^2"
    category: str = "regression"

class KMMetricReport(BaseModel):
    metrics: List[KMMetric]

class KMRequest(BaseModel):
    t_start: DatetimeTZDtype
    hours: int = 48
    loc: tuple[int]
