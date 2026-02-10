


import os
import re
import zipfile
import tempfile
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List

from tableauhyperapi import HyperProcess, Telemetry, Connection

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tableau-metadata")

# ============================================================
# UTILS
# ============================================================

def strip_ns(root: ET.Element):
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def clean(val: str) -> str:
    if not val:
        return ""
    return re.sub(r'[\[\]"]', "", val).strip()


def normalize_table_name(name: str) -> str:
    name = clean(name)
    if ".csv_" in name:
        return name.split(".csv_", 1)[0]
    return name

# ============================================================
# STEP 1: HYPER — TABLES & COLUMN MAP
# ============================================================

def extract_hyper_metadata(hyper_path: str):
    tables: Dict[str, List[str]] = {}
    column_table_map: Dict[str, List[str]] = {}

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(hyper.endpoint, hyper_path) as conn:
            for schema in conn.catalog.get_schema_names():
                for table in conn.catalog.get_table_names(schema):
                    table_name = normalize_table_name(str(table.name))
                    cols = []

                    table_def = conn.catalog.get_table_definition(table)
                    for c in table_def.columns:
                        col = clean(str(c.name))
                        cols.append(col)
                        column_table_map.setdefault(col, []).append(table_name)

                    tables[table_name] = cols

    return tables, column_table_map

# ============================================================
# STEP 2: RELATIONSHIPS (FINAL FORMAT)
# ============================================================

def extract_relationships(root, column_table_map, tables):
    relationships = []
    seen = set()

    def add(from_t, from_c, to_t, to_c):
        key = (from_t, from_c, to_t, to_c)
        if key in seen:
            return
        seen.add(key)
        relationships.append({
            "fromTable": from_t.lower(),
            "fromColumn": from_c,
            "toTable": to_t.lower(),
            "toColumn": to_c,
            "relationshipType": "Many-to-One"
        })

    # -------- XML relationships --------
    for rel in root.findall(".//object-graph/relationships/relationship"):
        expr = rel.find("expression")
        if expr is None:
            continue

        cols = [clean(e.get("op")) for e in expr.findall("expression")]
        if len(cols) != 2:
            continue

        left, right = cols
        lt = column_table_map.get(left, [])
        rt = column_table_map.get(right, [])

        if lt and rt:
            add(lt[0], left, rt[0], right)

    # -------- Fallback: common column heuristic --------
    if not relationships:
        table_items = list(tables.items())
        for i, (t1, cols1) in enumerate(table_items):
            for t2, cols2 in table_items[i + 1:]:
                common = set(cols1) & set(cols2)
                for col in common:
                    add(t1, col, t2, col)

    return relationships

# ============================================================
# CORE FUNCTION (USED BY main.py)
# ============================================================

def extract_metadata_from_twbx(twbx_path: str):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(twbx_path, "r") as z:
            z.extractall(tmp)

        twb = hyper = None
        for root_dir, _, files in os.walk(tmp):
            for f in files:
                if f.endswith(".twb"):
                    twb = os.path.join(root_dir, f)
                elif f.endswith(".hyper"):
                    hyper = os.path.join(root_dir, f)

        if not twb or not hyper:
            raise ValueError("Invalid TWBX file")

        tree = ET.parse(twb)
        root = tree.getroot()
        strip_ns(root)

        tables, col_map = extract_hyper_metadata(hyper)
        relationships = extract_relationships(root, col_map, tables)

        return {
            "relationships": relationships
        }


