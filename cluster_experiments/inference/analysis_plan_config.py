from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass(eq=True)
class AnalysisPlanMetricsConfig:
    metrics: List[Dict[str, str]]
    variants: List[Dict[str, Union[str, bool]]]
    analysis_type: str
    variant_col: str = "experiment_group"
    alpha: float = 0.05
    dimensions: List[Dict[str, Union[str, List]]] = field(default_factory=lambda: [])
    analysis_config: Dict = field(default_factory=lambda: {})
    custom_analysis_type_mapper: Optional[Dict] = None

    def __str__(self) -> str:
        return (
            f"AnalysisPlanMetricsConfig(metrics={len(self.metrics)}, "
            f"variants={len(self.variants)}, analysis_type={self.analysis_type!r}, "
            f"variant_col={self.variant_col!r}, alpha={self.alpha})"
        )


@dataclass(eq=True)
class AnalysisPlanConfig:
    tests: List[Dict[str, Union[List[Dict], Dict, str]]]
    variants: List[Dict[str, Union[str, bool]]]
    variant_col: str = "experiment_group"
    alpha: float = 0.05

    def __str__(self) -> str:
        return (
            f"AnalysisPlanConfig(tests={len(self.tests)}, "
            f"variants={len(self.variants)}, variant_col={self.variant_col!r}, "
            f"alpha={self.alpha})"
        )
