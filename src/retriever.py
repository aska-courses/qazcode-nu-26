"""
Graph-augmented retriever for medical diagnosis.
ChromaDB  → vector search over symptom nodes
NetworkX  → graph scoring with supports / contradicts / co-occurs / appear-with
"""

from __future__ import annotations

import json, pickle
from pathlib import Path
from typing import Optional

import chromadb
import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# ── Hyper-params ───────────────────────────────────────────────────────────────
CONTRADICT_PENALTY = 1.9
CO_OCCUR_DISCOUNT  = 0.5
APPEAR_WITH_BOOST  = 0.3
TOP_K_VECTOR       = 5
TOP_N_FINAL        = 10
EMBED_MODEL        = "ai-forever/FRIDA"


USELESS_PROTOCOLS = ['p_7d0d2e28b2',
 'p_142a38cb50',
 'p_a63da0db69',
 'p_eceb563add',
 'p_44a0ebf2ec',
 'p_27de892b06',
 'p_862b3ea3c7',
 'p_e5bdd8e7f5',
 'p_c1bb94fa4b',
 'p_f65478df1e',
 'p_60e0ba761f']

def batched_upsert(col, docs, embs, ids, metas, batch_size=5000):
    for i in range(0, len(ids), batch_size):
        col.upsert(
            documents=docs[i:i+batch_size],
            embeddings=embs[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
        )
# ══════════════════════════════════════════════════════════════════════════════
# 1. Embedder  (ruRoberta – pool last hidden layer)
# ══════════════════════════════════════════════════════════════════════════════
class RuRobertaEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        out         = self.model(**enc, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]               # (B, T, H)
        mask        = enc["attention_mask"].unsqueeze(-1) # (B, T, 1)
        pooled      = (last_hidden * mask).sum(1) / mask.sum(1)  # (B, H)
        return pooled.cpu().numpy()

    def embed_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ══════════════════════════════════════════════════════════════════════════════
# 2. ChromaDB – symptom node vector index
# ══════════════════════════════════════════════════════════════════════════════
class VectorIndex:
    COLLECTION = "symptom_nodes"

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedder: Optional[RuRobertaEmbedder] = None,
    ):
        self.client   = chromadb.PersistentClient(path=persist_dir)
        self.embedder = embedder or RuRobertaEmbedder()
        self.col      = self.client.get_or_create_collection(
            self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    def index_graphs(self, graph_dir: str, batch_size: int = 256) -> None:
        docs, ids, metas = [], [], []
        seen_ids: set[str] = set()
    
        for node_file in Path(graph_dir).rglob("*_nodes.json"):
            protocol_id = node_file.parent.name
            if protocol_id in USELESS_PROTOCOLS:
                continue
    
            nodes = json.loads(node_file.read_text(encoding="utf-8"))
            for node in nodes:
                if node.get("type") != "symptom":
                    continue
    
                label = node.get("label", "")
                node_id = (
                    node.get("id")
                    or node.get("node_id")
                    or node.get("icd_code")
                    or label.replace(" ", "_").lower()
                )
                if not node_id:
                    print(f'no node_id{protocol_id}')
                    continue
    
                uid = f"{protocol_id}__{node_id}"
                if uid in seen_ids:
                    continue  # 👈 skip duplicate
    
                seen_ids.add(uid)
                docs.append(label)
                ids.append(uid)
                metas.append({
                    "protocol_id": protocol_id,
                    "node_id": node_id,
                    "label": label,
                })
    
        if not docs:
            return
    
        embs = []
        for i in range(0, len(docs), batch_size):
            embs.append(self.embedder.encode(docs[i:i + batch_size]))
        embs = np.vstack(embs).tolist()
        batched_upsert(self.col, docs, embs, ids, metas)
        # self.col.upsert(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
        print(f"[VectorIndex] Indexed {len(docs)} symptom nodes.")

    # ------------------------------------------------------------------
    def query(self, symptom_texts: list[str], top_n: int = TOP_N_FINAL) -> list[dict]:
        """Return de-duplicated hits for all symptom texts combined."""
        if not symptom_texts:
            return []

        embs    = self.embedder.encode(symptom_texts).tolist()
        results = self.col.query(
            query_embeddings=embs,
            n_results=TOP_K_VECTOR,
            include=["metadatas", "distances"],
        )

        seen: set[tuple] = set()
        hits: list[dict] = []
        for batch_metas, batch_dists in zip(results["metadatas"], results["distances"]):
            for meta, dist in zip(batch_metas, batch_dists):
                key = (meta["protocol_id"], meta["node_id"])
                if key not in seen:
                    seen.add(key)
                    hits.append({**meta, "similarity": float(1.0 - dist)})
        return hits


# ══════════════════════════════════════════════════════════════════════════════
# 3. NetworkX – protocol knowledge graph
# ══════════════════════════════════════════════════════════════════════════════

            
class GraphIndex:
    def __init__(self, graph_dir: str, cache_path: str = "./graph_cache/graph.pkl"):
        self.cache_path = Path(cache_path)
        self.G = nx.DiGraph()
        self._node_meta: dict[str, dict] = {}
        self._max_weights: dict[str, float] = {} # <--- ADDED
        self.graph_dir = graph_dir

        if self.cache_path.exists():
            self._load_cache()
        else:
            self._load()
            self._save_cache()

    # ── cache ────────────────────────────────────────────────────────
    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump((self.G, self._node_meta, self._max_weights), f) # <--- UPDATED

    def _load_cache(self):
        with open(self.cache_path, "rb") as f:
            # <--- UPDATED to handle existing caches without third value safely
            data = pickle.load(f)
            if len(data) == 3:
                self.G, self._node_meta, self._max_weights = data
            else:
                self.G, self._node_meta = data
                self._compute_max_weights() # Fallback

    # ── build from json ──────────────────────────────────────────────
    @staticmethod
    def _uid(protocol_id: str, node_id: str) -> str:
        return f"{protocol_id}__{node_id}"

    def _compute_max_weights(self): # <--- ADDED
        """Calculates the sum of all 'supports' weights for every diagnosis."""
        for node, data in self.G.nodes(data=True):
            if data.get("type") == "diagnosis":
                total = sum(
                    float(edata.get("weight", 0.5))
                    for _, _, edata in self.G.in_edges(node, data=True)
                    if edata.get("type") == "supports"
                )
                # We floor at 1.0 to avoid division by zero and over-inflating 
                # scores for protocols with only 1 weak symptom.
                self._max_weights[node] = max(total, 1.0)

    def _load(self) -> None:
        # ... (Existing logic for loading nodes and edges remains exactly the same) ...
        graph_dir = self.graph_dir
        loaded = 0
        for gf in Path(graph_dir).rglob("*_graph.json"):
            g = json.loads(gf.read_text(encoding="utf-8"))
            pid = g["protocol_id"]
            if pid in USELESS_PROTOCOLS: continue

            for node in g.get("nodes", []):
                label = node.get("label", "")
                node_id = node.get("id") or node.get("node_id") or node.get("icd_code") or label.replace(" ", "_").lower()
                uid = self._uid(pid, node_id)
                if uid in self.G: continue
                self.G.add_node(uid, **node, protocol_id=pid)
                self._node_meta[uid] = {**node, "protocol_id": pid}

            for edge in g.get("edges", []):
                src = self._uid(pid, edge["source"])
                tgt = self._uid(pid, edge["target"])
                if self.G.has_node(src) and self.G.has_node(tgt):
                    self.G.add_edge(src, tgt, type=edge.get("type", "supports"),
                                    weight=float(edge.get("weight", 0.5)), evidence=edge.get("evidence", ""))
            loaded += 1

        self._compute_max_weights() # <--- ADDED
        print(f"[GraphIndex] {loaded} protocols | nodes={self.G.number_of_nodes()} edges={self.G.number_of_edges()}")
        self._save_cache()

    # ── api ──────────────────────────────────────────────────────────
    def get_node(self, uid: str) -> dict:
        return self._node_meta.get(uid, {})
    
    def get_max_weight(self, uid: str) -> float: # <--- ADDED
        return self._max_weights.get(uid, 1.0)

    def out_edges(self, uid: str):
        for _, tgt, data in self.G.out_edges(uid, data=True):
            yield tgt, data


# ══════════════════════════════════════════════════════════════════════════════
# 4. Graph scorer (Updated with Normalization)
# ══════════════════════════════════════════════════════════════════════════════
class GraphScorer:
    def __init__(self, graph_index: GraphIndex):
        self.gi = graph_index

    def score(self, matched_symptom_uids: list[str]) -> list[dict]:
            scores:   dict[str, float]      = {}
            evidence: dict[str, list[str]]  = {}

            def _add(d_uid: str, delta: float, ev: str = "") -> None:
                scores[d_uid]  = scores.get(d_uid, 0.0) + delta
                evidence.setdefault(d_uid, [])
                if ev: evidence[d_uid].append(ev)

            indirect_symptom_uids: set[str] = set()

            # ── 1. Direct scoring ──────────────────────────────────────────
            for s_uid in matched_symptom_uids:
                for nbr, edata in self.gi.out_edges(s_uid):
                    nbr_meta = self.gi.get_node(nbr)
                    etype    = edata.get("type", "")
                    w        = float(edata.get("weight", 0.5))
                    ev       = edata.get("evidence", "")

                    if nbr_meta.get("type") == "diagnosis":
                        if etype == "supports":
                            _add(nbr, +w, ev)
                        elif etype == "contradicts":
                            _add(nbr, -w * CONTRADICT_PENALTY, ev)
                    elif nbr_meta.get("type") == "symptom" and etype == "co-occurs":
                        indirect_symptom_uids.add(nbr)

            # ── 2. Indirect scoring ────────────────────────────────────────
            for s_uid in indirect_symptom_uids - set(matched_symptom_uids):
                for nbr, edata in self.gi.out_edges(s_uid):
                    nbr_meta = self.gi.get_node(nbr)
                    if nbr_meta.get("type") == "diagnosis" and edata.get("type") == "supports":
                        w  = float(edata.get("weight", 0.5))
                        ev = edata.get("evidence", "")
                        _add(nbr, +w * CO_OCCUR_DISCOUNT, ev)

            # ── 3. Normalization ──────────────────────────────────────────
            normalized_scores = {}
            for d_uid, raw_sc in scores.items():
                max_w = self.gi.get_max_weight(d_uid)
                normalized_scores[d_uid] = raw_sc / max_w

            # Sort all UIDs by their normalized score
            ranked_uids = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

            # ── 4. Diversity Filter (The Fix) ──────────────────────────────
            output = []
            seen_diagnoses = set() # Track unique labels + ICD codes
            
            for d_uid, sc in ranked_uids:
                if len(output) >= TOP_N_FINAL:
                    break
                    
                meta = self.gi.get_node(d_uid)
                label = meta.get("label", d_uid)
                icd = meta.get("icd_code", "")
                
                # Create a unique key for this specific medical condition
                # We use both label and ICD to be safe
                diag_key = f"{label.lower()}_{icd}"
                
                if diag_key in seen_diagnoses:
                    continue # Skip if we already added this disease from another protocol
                
                seen_diagnoses.add(diag_key)

                output.append({
                    "rank":        len(output) + 1,
                    "uid":         d_uid,
                    "diagnosis":   label,
                    "icd10_code":  icd,
                    "protocol_id": meta.get("protocol_id", ""),
                    "score":       round(sc, 4),
                    "evidence":    list(set(evidence.get(d_uid, []))), # unique evidence strings
                })
                
            return output


# ══════════════════════════════════════════════════════════════════════════════
# 5. Public facade
# ══════════════════════════════════════════════════════════════════════════════
class MedicalRetriever:
    def __init__(
        self,
        graph_dir:     str  = "./data/graphs_new",
        chroma_dir:    str  = "./chroma_db",
        embed_model:   str  = EMBED_MODEL,
        device:        str  = "cpu",
        rebuild_index: bool = False,
    ):
        # SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embedder     = SentenceTransformer(embed_model)
        # RuRobertaEmbedder(embed_model, device)
        self.vector_index = VectorIndex(chroma_dir, self.embedder)
        self.graph_index = GraphIndex(
        graph_dir=graph_dir,
        cache_path="./graph_cache/graph.pkl"
    )
        self.scorer       = GraphScorer(self.graph_index)

        if rebuild_index or self.vector_index.col.count() == 0:
            self.vector_index.index_graphs(graph_dir)

    def dedup_symptoms(self, candidates: list[dict], min_sim: float = 0.6) -> list[dict]:
        best = {}
    
        for c in candidates:
            key = c["node_id"]          # or use c["label"] if preferred
            sim = c["similarity"]
    
            if sim < min_sim:
                continue
    
            if key not in best or sim > best[key]["similarity"]:
                best[key] = c
    
        # sort by similarity desc
        return sorted(best.values(), key=lambda x: x["similarity"], reverse=True)

    # ------------------------------------------------------------------
    def retrieve(
            self,
            extracted_symptoms: list[str],
            top_n: int = TOP_N_FINAL,
        ) -> list[dict]:
            # 1. Get hits from Vector DB
            hits = self.vector_index.query(extracted_symptoms)
            
            # 2. Get unique labels found by the vector search
            matched_labels = {h['label'].lower().strip() for h in hits}
            
            # 3. Find ALL UIDs in the graph that have these labels
            # This is the fix: if "жидкий стул" was found, find EVERY node with that label
            expanded_uids = []
            for uid, node_data in self.graph_index.G.nodes(data=True):
                if node_data.get('type') == 'symptom':
                    lbl = node_data.get('label', '').lower().strip()
                    if lbl in matched_labels:
                        expanded_uids.append(uid)
            
            # 4. Score based on expanded list
             # 4. Score based on expanded list
            output = self.scorer.score(expanded_uids)
            return output[:top_n]
            # from collections import defaultdict
    
            # protocol_scores = defaultdict(float)
            
            # for item in output:
            #     protocol_scores[item["protocol_id"]] += item["score"]
            
            # protocol_scores_sorted = dict(
            #     sorted(protocol_scores.items(), key=lambda x: x[1], reverse=True)
            # )
            # first_protocol_id, first_score = next(iter(protocol_scores_sorted.items()))
            # return first_protocol_id
