import json
import sys
import re
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# --- CONFIGURATION ---
ORIGINAL_DIR = "../data/graphs_copy"
VERIFY_DIR = "../data/graphs_to_verify"

sys.path.append(str(Path("/home/dsrc_iskakova/Untitled Folder/datasaur_2026").resolve()))
from utils.generator import LLMGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

API_KEY = "EMPTY"
HUB_URL = os.getenv("VLLM_120B_URL") 
MODEL = "gpt-oss-120b"

client = OpenAI(base_url=HUB_URL, api_key=API_KEY)
generator = LLMGenerator(client=client, model_name=MODEL)
load_dotenv()

# --- PROMPTS (Unchanged as requested) ---
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
    """Enhanced parser to handle LLM conversational filler."""
    try:
        # 1. Try cleaning markdown blocks
        clean = re.sub(r"```json|```", "", raw).strip()
        # 2. Try to find the first '{' and last '}' to isolate JSON
        start_idx = clean.find('{')
        end_idx = clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            clean = clean[start_idx:end_idx+1]
        return json.loads(clean)
    except Exception as e:
        # Debugging: Print a snippet of what failed
        print(f"      [Parse Error] Start of raw response: {raw[:150]}...")
        raise e

def is_json_file_empty(filepath):
    if not os.path.exists(filepath):
        return True
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Check if it's an empty list or a dict with empty lists
            if isinstance(data, list) and len(data) == 0: return True
            if isinstance(data, dict):
                if not data: return True
                # Specific to your nodes/edges structure
                if "nodes" in data and len(data["nodes"]) == 0: return True
                if "edges" in data and len(data["edges"]) == 0: return True
            return False
    except:
        return True

def extract_graph(protocol: dict, max_retries=3) -> dict:
    pid = protocol['protocol_id']
    orig_save_dir = f"{ORIGINAL_DIR}/{pid}"
    ver_save_dir = f"{VERIFY_DIR}/{pid}"
    
    orig_node_path = f"{orig_save_dir}/{pid}_nodes.json"
    orig_edge_path = f"{orig_save_dir}/{pid}_edges.json"
    
    ver_node_path = f"{ver_save_dir}/{pid}_nodes.json"
    ver_edge_path = f"{ver_save_dir}/{pid}_edges.json"
    ver_graph_path = f"{ver_save_dir}/{pid}_graph.json"

    nodes_empty = is_json_file_empty(orig_node_path)
    edges_empty = is_json_file_empty(orig_edge_path)

    if not nodes_empty and not edges_empty:
        return {'status': 'skipped'}

    os.makedirs(ver_save_dir, exist_ok=True)
    text = protocol["text"]
    
    # --- Pass 1: Nodes ---
    nodes = []
    if not nodes_empty:
        with open(orig_node_path, "r", encoding="utf-8") as f:
            nodes = json.load(f)
    else:
        for attempt in range(max_retries):
            try:
                print(f"  [{pid}] Nodes Attempt {attempt+1}...")
                node_message = generator.generate(
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER.format(protocol_id=pid, text=text)}
                    ], 
                    reasoning_effort="medium", temperature=0.2, max_new_tokens=50000
                )
                nodes = parse_json_safe(node_message).get("nodes", [])
                if nodes and len(nodes) > 0:
                    with open(ver_node_path, "w", encoding="utf-8") as f:
                        json.dump(nodes, f, ensure_ascii=False, indent=2)
                    break
            except Exception:
                continue

    if not nodes:
        print(f"  !! Failed Nodes for {pid} after {max_retries} retries.")
        return None

    # --- Pass 2: Edges ---
    edges = []
    # If edges were empty in original, or we just created new nodes, we must run edges
    for attempt in range(max_retries):
        try:
            print(f"  [{pid}] Edges Attempt {attempt+1}...")
            edge_message = generator.generate([
                {"role": "system", "content": SYSTEM2},
                {"role": "user", "content": USER2.format(
                    extracted_nodes=json.dumps(nodes, ensure_ascii=False),
                    text=text
                )}], reasoning_effort="medium", temperature=0.2)
            edges = parse_json_safe(edge_message).get("edges", [])
            if edges and len(edges) > 0:
                with open(ver_edge_path, "w", encoding="utf-8") as f:
                    json.dump(edges, f, ensure_ascii=False, indent=2)
                break
        except Exception:
            continue

    # --- Pass 3: Combine ---
    graph = {
        "protocol_id": pid,
        "nodes": nodes,
        "edges": edges if edges else []
    }
    with open(ver_graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    return graph

# --- EXECUTION ---
data = []
with open("../data/corpus/protocols_corpus.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

subset = data[args.start:args.end]

print(f"Checking {len(subset)} protocols...")
for protocol in tqdm(subset):
    extract_graph(protocol)