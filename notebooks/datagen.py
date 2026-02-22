import json
import sys
from pathlib import Path
from utils.generator import LLMGenerator

import json
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()


API_KEY="EMPTY"
HUB_URL=os.getenv("VLLM_120B_URL") 
MODEL=os.getenv("MODEL")

client = OpenAI(base_url=HUB_URL, api_key=API_KEY)

generator = LLMGenerator(client=client, model_name=MODEL)

load_dotenv()

data = []
with open("../data/corpus/protocols_corpus.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

subset = data[args.start:args.end]
SYSTEM = """
You are a clinical knowledge graph extraction engine.
Given a Kazakh clinical protocol text, extract structured medical entities.
Respond ONLY in valid JSON. No explanation, no markdown.
"""

USER = """
Extract all medical entities from this protocol:

PROTOCOL_ID: {protocol_id}
TEXT: {text}

Return JSON with this exact schema:
{{
  "nodes": [
    {{
      "id": "unique_snake_case_id",
      "type": "symptom | diagnosis",
      "label": "short human-readable name in Russian",
      "icd_code": "J18.9",          // only for diagnosis nodes, extract STRICTLY from  'Код(ы) по МКБ-10' section in the text, no NULL!
    }}
  ]
}}

Node type rules:
- "symptom": clinical signs the patient presents (fever, pain, etc.)
- "diagnosis": the actual disease/condition
"""


SYSTEM2 = """
You are a clinical knowledge graph relationship extractor.
Given protocol text and already-extracted nodes, find relationships between them.
Respond ONLY in valid JSON. No explanation, no markdown.
"""

USER2 = """
Given these already-extracted nodes:
{extracted_nodes}

And the original protocol text:
{text}

Extract relationships between nodes. Return JSON:
{{
  "edges": [
    {{
      "source": "node_id",
      "target": "node_id",
      "type": "supports | contradicts | co-occurs",
      "weight": 0.0-1.0,
      "evidence": "exact short quote from text justifying this edge",
      "properties": {{}}
    }}
  ]
}}

Edge type rules:
- "supports":    symptom → diagnosis (symptom increases probability)
- "contradicts": symptom → diagnosis (symptom lowers probability)
- "co-occurs":   symptom ↔ symptom (commonly appear together)
- "appear-with":   diagnosis ↔ diagnosis (commonly appear together)

For weight:
- Use signal words: "всегда/always" → 0.95, "часто/often" → 0.75,
  "иногда/sometimes" → 0.5, "редко/rarely" → 0.25
"""

def parse_json_safe(raw: str) -> dict:
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


def extract_graph(protocol: dict, out_dir="../data/graphs_new", client=client, max_retries=3) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    savedir  = f"{out_dir}/{protocol['protocol_id']}"
    os.makedirs(savedir, exist_ok=True)

    graph_path = f"{savedir}/{protocol['protocol_id']}_graph.json"
    if os.path.exists(graph_path):
        # with open(graph_path, "r", encoding="utf-8") as f:
        #     graph = json.load(f)
        # return graph
        print(f"Graph for protocol {protocol['protocol_id']} already exists. Skipping extraction.")
        return {'exists': True}

    text = protocol["text"]
    # text = ''.join(
    #                     ch for ch in unicodedata.normalize('NFKC', text)
    #                     if unicodedata.category(ch) != 'Cf')

    # Pass 1 — Nodes
    node_path = f"{savedir}/{protocol['protocol_id']}_nodes.json"
    nodes = None
    
    if os.path.exists(node_path):
        with open(node_path, "r", encoding="utf-8") as f:
            nodes = json.load(f)
    else:
        for attempt in range(max_retries):
            try:
                messages = [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER.format(
                            protocol_id=protocol["protocol_id"],
                            icd_codes=protocol["icd_codes"],
                            source_file=protocol["source_file"],
                            text=text
                        )}
                    ]
                node_message = generator.generate(messages=messages, reasoning_effort="medium", temperature=0.2, max_new_tokens=50000)
                nodes = parse_json_safe(node_message)["nodes"]
                break # If successful, break out of the retry loop
            except Exception as e:
                print(f"Error parsing nodes JSON for protocol {protocol['protocol_id']} (Attempt {attempt + 1}/{max_retries}): {e}")
        
        # If nodes is still None after all retries, skip this protocol
        if nodes is None:
            print(f"Failed to extract nodes for protocol {protocol['protocol_id']} after {max_retries} retries. Skipping protocol.")
            return None

        # Save nodes
        with open(node_path, "w", encoding="utf-8") as f:
            json.dump(nodes, f, ensure_ascii=False, indent=2)

    # Pass 2 — Edges
    edge_path = f"{savedir}/{protocol['protocol_id']}_edges.json"
    edges = None
    
    if os.path.exists(edge_path):
        with open(edge_path, "r", encoding="utf-8") as f:
            edges = json.load(f)
    else:
        for attempt in range(max_retries):
            try:
                edge_message = generator.generate([
                        {"role": "system", "content": SYSTEM2},
                        {"role": "user", "content": USER2.format(
                            extracted_nodes=json.dumps(nodes, ensure_ascii=False),
                            text=text
                        )}], reasoning_effort="medium", temperature=0.2)
                edges = parse_json_safe(edge_message)["edges"]
                break # If successful, break out of the retry loop
            except Exception as e:
                print(f"Error parsing edges JSON for protocol {protocol['protocol_id']} (Attempt {attempt + 1}/{max_retries}): {e}")
                
        # If edges is still None after all retries, skip this protocol
        if edges is None:
            print(f"Failed to extract edges for protocol {protocol['protocol_id']} after {max_retries} retries. Skipping protocol.")
            return None

        # Save edges
        with open(edge_path, "w", encoding="utf-8") as f:
            json.dump(edges, f, ensure_ascii=False, indent=2)


    # Phase 3 — Combine into graph JSON
    graph = {
        "protocol_id": protocol["protocol_id"],
        "nodes": nodes,
        "edges": edges
    }

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    return graph


# python script.py --start 0 --end 200
from tqdm import tqdm
for protocol in tqdm(subset):
    graph = extract_graph(protocol, client=client)
    # The returned graph will be None if it failed after max retries
    if graph is None:
        continue