from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass(eq=True)
class AnalysisPlanConfig:
    metrics: List[Dict[str, str]]
    variants: List[Dict[str, Union[str, bool]]]
    analysis_type: str
    variant_col: str = "experiment_group"
    alpha: float = 0.05
    dimensions: List[Dict[str, Union[str, List]]] = field(default_factory=lambda: [])
    analysis_config: Dict = field(default_factory=lambda: {})
    custom_analysis_type_mapper: Optional[Dict] = None
