from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class GraphAttentionBackbone(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv, TransformerConv
        except Exception as exc:
            raise RuntimeError("torch_geometric is unavailable") from exc
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        head_dim = hidden_dim // num_heads
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            conv_cls = GATConv if i == 0 else TransformerConv
            self.layers.append(
                conv_cls(hidden_dim, head_dim, heads=num_heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = node_x
        for conv, norm in zip(self.layers, self.norms):
            new_h = norm(conv(h, edge_index))
            h = F.relu(new_h + h)
            h = self.dropout(h)
        return h


class KnowledgeGraphRecommender(nn.Module):
    def __init__(
        self,
        num_symptoms: int = 390,
        num_herbs: int = 811,
        num_properties: int = 23,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        jpa_dim: int = 90,
        prop_dim: int = 23,
        use_jpa: bool = True,
        use_23d: bool = True,
        use_kg: bool = True,
        use_bge: bool = False,
        bge_symptom_dim: int | None = None,
        backend: str = "auto",
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.num_herbs = num_herbs
        self.num_properties = num_properties if use_23d else 0
        self.hidden_dim = hidden_dim
        self.use_jpa = use_jpa
        self.use_23d = use_23d
        self.use_kg = use_kg
        self.use_bge = use_bge
        self.backend = backend
        self.symptom_input_encoder = nn.Sequential(
            nn.Linear(num_symptoms, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.symptom_node_embed = nn.Parameter(
            torch.randn(num_symptoms, hidden_dim) * 0.02
        )
        self.herb_node_embed = nn.Parameter(torch.randn(num_herbs, hidden_dim) * 0.02)
        self.property_node_embed = (
            nn.Parameter(torch.randn(self.num_properties, hidden_dim) * 0.02)
            if self.num_properties
            else None
        )
        self.jpa_projection = (
            nn.Linear(jpa_dim, hidden_dim) if use_jpa and jpa_dim > 0 else None
        )
        self.property_projection = (
            nn.Linear(prop_dim, hidden_dim) if use_23d and prop_dim > 0 else None
        )
        self.bge_symptom_projection = (
            nn.Linear(bge_symptom_dim, hidden_dim)
            if use_bge and bge_symptom_dim
            else None
        )
        if use_kg:
            if backend not in {"auto", "pyg"}:
                raise ValueError(
                    "Knowledge graph model requires GATConv and TransformerConv"
                )
            self.graph_backbone = GraphAttentionBackbone(
                hidden_dim, num_heads, num_layers, dropout
            )
            self.graph_backend = "pyg:GATConv+TransformerConv"
        else:
            self.graph_backbone = nn.Identity()
            self.graph_backend = "disabled"
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.herb_prediction_head = nn.Linear(hidden_dim, num_herbs)
        self.herb_bias = nn.Parameter(torch.zeros(num_herbs))

    def initial_nodes(
        self,
        herb_features_jpa: torch.Tensor | None = None,
        herb_features_23d: torch.Tensor | None = None,
        bge_symptom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        symptom = self.symptom_node_embed
        if self.bge_symptom_projection is not None:
            if bge_symptom_features is None:
                raise ValueError("use_bge=True requires bge_symptom_features")
            expected = (self.num_symptoms, self.bge_symptom_projection.in_features)
            if tuple(bge_symptom_features.shape) != expected:
                raise ValueError(
                    f"Expected BGE symptom features {expected}, got {tuple(bge_symptom_features.shape)}"
                )
            symptom = symptom + self.bge_symptom_projection(bge_symptom_features)
        herb = self.herb_node_embed
        if self.jpa_projection is not None and herb_features_jpa is not None:
            herb = herb + self.jpa_projection(herb_features_jpa)
        if self.property_projection is not None and herb_features_23d is not None:
            herb = herb + self.property_projection(herb_features_23d)
        nodes = [symptom, herb]
        if self.property_node_embed is not None:
            nodes.append(self.property_node_embed)
        return torch.cat(nodes, dim=0)

    def encode_nodes(
        self,
        edge_index: torch.Tensor,
        herb_features_jpa: torch.Tensor | None = None,
        herb_features_23d: torch.Tensor | None = None,
        bge_symptom_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_x = self.initial_nodes(
            herb_features_jpa, herb_features_23d, bge_symptom_features
        )
        if self.use_kg:
            node_x = self.graph_backbone(node_x, edge_index)
        symptom_nodes = node_x[: self.num_symptoms]
        herb_nodes = node_x[self.num_symptoms : self.num_symptoms + self.num_herbs]
        return (symptom_nodes, herb_nodes)

    def forward(
        self,
        symptom_multihot: torch.Tensor,
        edge_index: torch.Tensor,
        herb_features_jpa: torch.Tensor | None = None,
        herb_features_23d: torch.Tensor | None = None,
        bge_symptom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if symptom_multihot.dim() != 2 or symptom_multihot.size(1) != self.num_symptoms:
            raise ValueError(
                f"Expected symptom input [B,{self.num_symptoms}], got {tuple(symptom_multihot.shape)}"
            )
        symptom_nodes, herb_nodes = self.encode_nodes(
            edge_index, herb_features_jpa, herb_features_23d, bge_symptom_features
        )
        denom = symptom_multihot.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_symptom_nodes = symptom_multihot @ symptom_nodes / denom
        input_repr = self.symptom_input_encoder(symptom_multihot)
        query = input_repr.unsqueeze(1)
        herb_context, _ = self.cross_attention(
            query,
            herb_nodes.unsqueeze(0).expand(symptom_multihot.size(0), -1, -1),
            herb_nodes.unsqueeze(0).expand(symptom_multihot.size(0), -1, -1),
        )
        fused = self.fusion(
            torch.cat([pooled_symptom_nodes, herb_context.squeeze(1)], dim=1)
        )
        logits = self.herb_prediction_head(fused) + self.herb_bias
        if logits.shape != (symptom_multihot.size(0), self.num_herbs):
            raise AssertionError(
                f"Candidate-generator logits shape {tuple(logits.shape)} != {(symptom_multihot.size(0), self.num_herbs)}"
            )
        if logits.size(1) == 390:
            raise AssertionError(
                "Refusing 390-dimensional symptom prediction output in HerbRec."
            )
        return logits
