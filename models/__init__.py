"""RankLab recommendation models."""

from .base import Recommender
from .item_cf import ItemCFModel
from .lightgcn import LightGCNModel
from .popularity import PopularityModel
from .recency import RecencyModel
from .xsimgcl import XSimGCLModel

MODEL_REGISTRY: dict[str, type[Recommender]] = {
    PopularityModel.name: PopularityModel,
    RecencyModel.name: RecencyModel,
    ItemCFModel.name: ItemCFModel,
    LightGCNModel.name: LightGCNModel,
    XSimGCLModel.name: XSimGCLModel,
}

__all__ = [
    "Recommender",
    "PopularityModel",
    "RecencyModel",
    "ItemCFModel",
    "LightGCNModel",
    "XSimGCLModel",
    "MODEL_REGISTRY",
]
