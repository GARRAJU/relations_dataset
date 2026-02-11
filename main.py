


# import os
# import re
# import time
# import logging
# import tempfile
# import requests
# import pandas as pd

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from azure.storage.blob import BlobServiceClient
# from dotenv import load_dotenv

# # ✅ Import extractor
# from extractor import extract_metadata_from_twbx

# # ============================================================
# # ENV + CONFIG
# # ============================================================

# load_dotenv()

# POWERBI_API = "https://api.powerbi.com/v1.0/myorg"

# TENANT_ID = os.getenv("TENANT_ID")
# CLIENT_ID = os.getenv("CLIENT_ID")
# CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# TEMPLATE_WORKSPACE_ID = os.getenv("TEMPLATE_WORKSPACE_ID")
# TEMPLATE_REPORT_ID = os.getenv("TEMPLATE_REPORT_ID")

# AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# # Containers
# TWBX_CONTAINER = os.getenv("TWBX_CONTAINER")   # contains .twbx
# CSV_CONTAINER = os.getenv("CSV_CONTAINER")          # contains csvs

# REPORT_NAME = "Final_Sales_Report"

# # ============================================================
# # LOGGING
# # ============================================================

# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger("tableau-pbi-migrator")

# # ============================================================
# # FASTAPI APP
# # ============================================================

# app = FastAPI(title="Tableau → Power BI Migration")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================================================
# # HELPERS
# # ============================================================

# def extract_second_word_table_name(filename: str) -> str:
#     """
#     Extract_customers.csv_HASH -> customers
#     """
#     base = filename.split(".csv")[0]
#     parts = base.split("_")
#     table_name = parts[1] if len(parts) >= 2 else parts[0]
#     return re.sub(r"[^a-zA-Z]", "", table_name).lower()


# def get_auth_token() -> str:
#     url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"

#     resp = requests.post(
#         url,
#         data={
#             "grant_type": "client_credentials",
#             "client_id": CLIENT_ID,
#             "client_secret": CLIENT_SECRET,
#             "scope": "https://analysis.windows.net/powerbi/api/.default",
#         },
#     )

#     resp.raise_for_status()
#     return resp.json()["access_token"]


# def download_twbx_from_blob(folder_name: str) -> str:
#     """
#     Downloads <folder_name>.twbx directly from TWBX container
#     """
#     blob_service = BlobServiceClient.from_connection_string(
#         AZURE_STORAGE_CONNECTION_STRING
#     )
#     container = blob_service.get_container_client(TWBX_CONTAINER)

#     twbx_blob_name = f"{folder_name}.twbx"

#     try:
#         data = container.download_blob(twbx_blob_name).readall()

#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".twbx")
#         tmp.write(data)
#         tmp.close()

#         log.info(f"Downloaded TWBX: {twbx_blob_name}")
#         return tmp.name

#     except Exception:
#         raise Exception(f"TWBX file not found: {twbx_blob_name}")

# # ============================================================
# # MIGRATION API
# # ============================================================

# @app.post("/migrate-static")
# def migrate_static(folder_name: str, target_workspace_id: str):
#     try:
#         # ----------------------------------------------------
#         # 1. AUTH
#         # ----------------------------------------------------
#         token = get_auth_token()

#         # ----------------------------------------------------
#         # 2. DOWNLOAD TWBX & EXTRACT METADATA
#         # ----------------------------------------------------
#         twbx_path = download_twbx_from_blob(folder_name)
#         metadata = extract_metadata_from_twbx(twbx_path)
#         os.remove(twbx_path)

#         relationships_metadata = metadata["relationships"]
#         log.info("Extracted Tableau relationships")

#         # ----------------------------------------------------
#         # 3. READ CSVs FROM AZURE BLOB (METADATA DRIVEN)
#         # ----------------------------------------------------
#         blob_service = BlobServiceClient.from_connection_string(
#             AZURE_STORAGE_CONNECTION_STRING
#         )
#         container = blob_service.get_container_client(CSV_CONTAINER)

#         blob_tables = {}
#         prefix = f"{folder_name.rstrip('/')}/"

#         valid_tables = set()
#         for r in relationships_metadata:
#             valid_tables.add(r["fromTable"])
#             valid_tables.add(r["toTable"])

#         for blob in container.list_blobs(name_starts_with=prefix):
#             filename = os.path.basename(blob.name)

#             if not filename.lower().endswith(".csv"):
#                 continue

#             table_name = extract_second_word_table_name(filename)

#             if table_name not in valid_tables:
#                 continue

#             data = container.download_blob(blob.name).readall()
#             blob_tables[table_name] = pd.read_csv(
#                 pd.io.common.BytesIO(data)
#             )

#             log.info(f"Loaded table: {table_name}")

#         # ----------------------------------------------------
#         # 4. BUILD POWER BI RELATIONSHIPS
#         # ----------------------------------------------------
#         pbi_relationships = []
#         for r in relationships_metadata:
#             pbi_relationships.append({
#                 "name": f"{r['fromTable']}_{r['toTable']}",
#                 "fromTable": r["fromTable"],
#                 "fromColumn": r["fromColumn"],
#                 "toTable": r["toTable"],
#                 "toColumn": r["toColumn"],
#                 "crossFilteringBehavior": "BothDirections",
#             })

#         # ----------------------------------------------------
#         # 5. DEFINE DATASET
#         # ----------------------------------------------------
#         dataset_payload = {
#             "name": f"{REPORT_NAME}_DS",
#             # "defaultMode": "Push",
#             "tables": [],
#             "relationships": pbi_relationships,
#         }

#         for table_name, df in blob_tables.items():
#             columns = []

#             for col in df.columns:
#                 if "id" in col.lower():
#                     dtype, summarize = "Int64", "none"
#                 elif df[col].dtype == "float64":
#                     dtype, summarize = "Double", "sum"
#                 elif df[col].dtype == "int64":
#                     dtype, summarize = "Int64", "sum"
#                 else:
#                     dtype, summarize = "String", "none"

#                 columns.append({
#                     "name": col,
#                     "dataType": dtype,
#                     # "summarizeBy": summarize,
#                 })

#             dataset_payload["tables"].append({
#                 "name": table_name,
#                 "columns": columns,
#             })

#         # ----------------------------------------------------
#         # 6. CREATE DATASET
#         # ----------------------------------------------------
#         ds_resp = requests.post(
#             f"{POWERBI_API}/groups/{target_workspace_id}/datasets",
#             headers={"Authorization": f"Bearer {token}"},
#             json=dataset_payload,
#         )
#         ds_resp.raise_for_status()

#         dataset_id = ds_resp.json()["id"]
#         log.info(f"Dataset created: {dataset_id}")

#         # ----------------------------------------------------
#         # 7. PUSH DATA
#         # ----------------------------------------------------
#         time.sleep(5)

#         for table_name, df in blob_tables.items():
#             df_clean = df.where(pd.notnull(df), None)
#             rows = df_clean.to_dict(orient="records")

#             for i in range(0, len(rows), 2500):
#                 requests.post(
#                     f"{POWERBI_API}/groups/{target_workspace_id}/datasets/{dataset_id}/tables/{table_name}/rows",
#                     headers={"Authorization": f"Bearer {token}"},
#                     json={"rows": rows[i:i + 2500]},
#                 ).raise_for_status()

#             log.info(f"Pushed {len(rows)} rows into {table_name}")

#         # ----------------------------------------------------
#         # 8. CLONE REPORT
#         # ----------------------------------------------------
#         clone_resp = requests.post(
#             f"{POWERBI_API}/groups/{TEMPLATE_WORKSPACE_ID}/reports/{TEMPLATE_REPORT_ID}/Clone",
#             headers={"Authorization": f"Bearer {token}"},
#             json={
#                 "name": REPORT_NAME,
#                 "targetWorkspaceId": target_workspace_id,
#                 "targetModelId": dataset_id,
#             },
#         )
#         clone_resp.raise_for_status()

#         return {
#             "status": "SUCCESS",
#             "dataset_id": dataset_id,
#             "report_id": clone_resp.json()["id"],
#             "message": "TWBX metadata + data migrated successfully",
#         }

#     except Exception as e:
#         log.exception("Migration failed")
#         raise HTTPException(status_code=500, detail=str(e))


# import os
# import re
# import time
# import logging
# import tempfile
# import requests
# import pandas as pd
# import json

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from azure.storage.blob import BlobServiceClient
# from dotenv import load_dotenv

# # ✅ Import extractor
# from extractor import extract_metadata_from_twbx

# # ============================================================
# # ENV + CONFIG
# # ============================================================

# load_dotenv()

# POWERBI_API = "https://api.powerbi.com/v1.0/myorg"

# TENANT_ID = os.getenv("TENANT_ID")
# CLIENT_ID = os.getenv("CLIENT_ID")
# CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# TEMPLATE_WORKSPACE_ID = os.getenv("TEMPLATE_WORKSPACE_ID")
# TEMPLATE_REPORT_ID = os.getenv("TEMPLATE_REPORT_ID")

# AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# # Containers
# TWBX_CONTAINER = os.getenv("TWBX_CONTAINER")
# CSV_CONTAINER = os.getenv("CSV_CONTAINER")

# REPORT_NAME = "Final_Sales_Report"

# # ============================================================
# # LOGGING
# # ============================================================

# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger("tableau-pbi-migrator")

# # ============================================================
# # FASTAPI APP
# # ============================================================

# app = FastAPI(title="Tableau → Power BI Migration")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================================================
# # STATIC RELATIONSHIPS (fallback)
# # ============================================================

# STATIC_RELATIONSHIPS = [
#     {
#         "fromTable": "products",
#         "fromColumn": "ProductID",
#         "toTable": "sales",
#         "toColumn": "ProductID",
#         "relationshipType": "Many-to-One"
#     },
#     {
#         "fromTable": "sales",
#         "fromColumn": "CustomerID",
#         "toTable": "customers",
#         "toColumn": "CustomerID",
#         "relationshipType": "Many-to-One"
#     }
# ]

# # ============================================================
# # HELPERS
# # ============================================================

# def extract_second_word_table_name(filename: str) -> str:
#     """
#     Extract_customers.csv_HASH -> customers
#     """
#     base = filename.split(".csv")[0]
#     parts = base.split("_")
#     table_name = parts[1] if len(parts) >= 2 else parts[0]
#     return re.sub(r"[^a-zA-Z]", "", table_name).lower()


# def get_auth_token() -> str:
#     url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"

#     resp = requests.post(
#         url,
#         data={
#             "grant_type": "client_credentials",
#             "client_id": CLIENT_ID,
#             "client_secret": CLIENT_SECRET,
#             "scope": "https://analysis.windows.net/powerbi/api/.default",
#         },
#     )

#     resp.raise_for_status()
#     return resp.json()["access_token"]


# def download_twbx_from_blob(folder_name: str) -> str:
#     """
#     Downloads <folder_name>.twbx directly from TWBX container
#     """
#     blob_service = BlobServiceClient.from_connection_string(
#         AZURE_STORAGE_CONNECTION_STRING
#     )
#     container = blob_service.get_container_client(TWBX_CONTAINER)

#     twbx_blob_name = f"{folder_name}.twbx"

#     try:
#         data = container.download_blob(twbx_blob_name).readall()

#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".twbx")
#         tmp.write(data)
#         tmp.close()

#         log.info(f"Downloaded TWBX: {twbx_blob_name}")
#         return tmp.name

#     except Exception as e:
#         log.error(f"Failed to download TWBX: {str(e)}")
#         raise Exception(f"TWBX file not found: {twbx_blob_name}")


# # ============================================================
# # MIGRATION API - UPDATED WITH EXTRACTOR RELATIONSHIPS
# # ============================================================

# @app.post("/migrate-static")
# def migrate_static(folder_name: str, target_workspace_id: str):
#     try:
#         # ----------------------------------------------------
#         # 1. AUTH
#         # ----------------------------------------------------
#         token = get_auth_token()
#         log.info("✅ Authentication successful")

#         # ----------------------------------------------------
#         # 2. DOWNLOAD TWBX & EXTRACT METADATA
#         # ----------------------------------------------------
#         twbx_path = download_twbx_from_blob(folder_name)
#         metadata = extract_metadata_from_twbx(twbx_path)
#         os.remove(twbx_path)

#         # Use extracted relationships if available, else fallback to STATIC_RELATIONSHIPS
#         # relationships_metadata = metadata.get("relationships") or STATIC_RELATIONSHIPS
#         # log.info(f"Using {len(relationships_metadata)} relationships")
#         relationships_metadata = metadata.get("relationships")
#         if relationships_metadata:
#             log.info(f"✅ Extracted {len(relationships_metadata)} relationships from TWBX:")
#             for r in relationships_metadata:
#                 log.info(f"  {r['fromTable']}.{r['fromColumn']} -> {r['toTable']}.{r['toColumn']}")
#         else:
#             log.warning("⚠️ No relationships found in TWBX. Using fallback static relationships.")
#             relationships_metadata = STATIC_RELATIONSHIPS

#         # ----------------------------------------------------
#         # 3. READ CSVs FROM AZURE BLOB
#         # ----------------------------------------------------
#         blob_service = BlobServiceClient.from_connection_string(
#             AZURE_STORAGE_CONNECTION_STRING
#         )
#         container = blob_service.get_container_client(CSV_CONTAINER)

#         blob_tables = {}
#         prefix = f"{folder_name.rstrip('/')}/"

#         valid_tables = set()
#         for r in relationships_metadata:
#             valid_tables.add(r["fromTable"])
#             valid_tables.add(r["toTable"])
#         log.info(f"Valid tables from relationships: {valid_tables}")

#         all_blobs = list(container.list_blobs(name_starts_with=prefix))
#         log.info(f"Found {len(all_blobs)} blobs with prefix '{prefix}'")

#         for blob in all_blobs:
#             filename = os.path.basename(blob.name)
#             if not filename.lower().endswith(".csv"):
#                 continue

#             table_name = extract_second_word_table_name(filename)
#             if table_name not in valid_tables:
#                 continue

#             try:
#                 data = container.download_blob(blob.name).readall()
#                 blob_tables[table_name] = pd.read_csv(pd.io.common.BytesIO(data))
#                 log.info(f"Loaded table: {table_name}")
#             except Exception as e:
#                 log.error(f"Failed to load {table_name}: {str(e)}")

#         if not blob_tables:
#             raise Exception(f"No CSV tables loaded for folder: {prefix}")

#         # ----------------------------------------------------
#         # 4. BUILD POWER BI RELATIONSHIPS
#         # ----------------------------------------------------
#         pbi_relationships = []
#         for r in relationships_metadata:
#             if r["fromTable"] in blob_tables and r["toTable"] in blob_tables:
#                 pbi_relationships.append({
#                     "name": f"{r['fromTable']}_{r['toTable']}",
#                     "fromTable": r["fromTable"],
#                     "fromColumn": r["fromColumn"],
#                     "toTable": r["toTable"],
#                     "toColumn": r["toColumn"],
#                     "crossFilteringBehavior": "BothDirections",
#                 })

#         # ----------------------------------------------------
#         # 5. CREATE DATASET
#         # ----------------------------------------------------
#         dataset_payload = {
#             "name": f"{REPORT_NAME}_DS",
#             "tables": [],
#             "relationships": pbi_relationships if pbi_relationships else None,
#             "defaultMode": "Push",
#         }

#         for table_name, df in blob_tables.items():
#             columns = []
#             for col in df.columns:
#                 if "id" in col.lower():
#                     dtype, summarize = "Int64", "none"
#                 elif df[col].dtype == "float64":
#                     dtype, summarize = "Double", "sum"
#                 elif df[col].dtype == "int64":
#                     dtype, summarize = "Int64", "sum"
#                 else:
#                     dtype, summarize = "String", "none"

#                 columns.append({
#                     "name": col,
#                     "dataType": dtype,
#                     "summarizeBy": summarize,
#                 })

#             dataset_payload["tables"].append({
#                 "name": table_name,
#                 "columns": columns,
#             })

#         ds_resp = requests.post(
#             f"{POWERBI_API}/groups/{target_workspace_id}/datasets",
#             headers={
#                 "Authorization": f"Bearer {token}",
#                 "Content-Type": "application/json"
#             },
#             json=dataset_payload,
#         )
#         ds_resp.raise_for_status()
#         dataset_id = ds_resp.json()["id"]
#         log.info(f"✅ Dataset created: {dataset_id}")

#         # ----------------------------------------------------
#         # 6. PUSH DATA
#         # ----------------------------------------------------
#         time.sleep(5)
#         for table_name, df in blob_tables.items():
#             rows = df.where(pd.notnull(df), None).to_dict(orient="records")
#             for i in range(0, len(rows), 2500):
#                 requests.post(
#                     f"{POWERBI_API}/groups/{target_workspace_id}/datasets/{dataset_id}/tables/{table_name}/rows",
#                     headers={
#                         "Authorization": f"Bearer {token}",
#                         "Content-Type": "application/json"
#                     },
#                     json={"rows": rows[i:i + 2500]},
#                 ).raise_for_status()
#             log.info(f"Pushed {len(rows)} rows into {table_name}")

#         # ----------------------------------------------------
#         # 7. CLONE REPORT
#         # ----------------------------------------------------
#         clone_resp = requests.post(
#             f"{POWERBI_API}/groups/{TEMPLATE_WORKSPACE_ID}/reports/{TEMPLATE_REPORT_ID}/Clone",
#             headers={
#                 "Authorization": f"Bearer {token}",
#                 "Content-Type": "application/json"
#             },
#             json={
#                 "name": REPORT_NAME,
#                 "targetWorkspaceId": target_workspace_id,
#                 "targetModelId": dataset_id,
#             },
#         )
#         clone_resp.raise_for_status()

#         return {
#             "status": "SUCCESS",
#             "dataset_id": dataset_id,
#             "report_id": clone_resp.json()["id"],
#             "tables_migrated": list(blob_tables.keys()),
#             "relationships_created": len(pbi_relationships) > 0,
#             "relationships": relationships_metadata,
#             "message": "TWBX metadata + data migrated successfully with extracted relationships"
#         }

#     except Exception as e:
#         log.exception("Migration failed")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/health")
# def health():
#     return {"status": "healthy"}


import os
import re
import time
import logging
import tempfile
import zipfile
import requests
import pandas as pd
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tableauhyperapi import HyperProcess, Telemetry, Connection, TableName

# ============================================================
# ENV + CONFIG
# ============================================================

load_dotenv()

POWERBI_API = "https://api.powerbi.com/v1.0/myorg"

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

TEMPLATE_WORKSPACE_ID = os.getenv("TEMPLATE_WORKSPACE_ID")
TEMPLATE_REPORT_ID = os.getenv("TEMPLATE_REPORT_ID")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

TWBX_CONTAINER = os.getenv("TWBX_CONTAINER")
CSV_CONTAINER = os.getenv("CSV_CONTAINER")

REPORT_NAME = "Final_Sales_Report"

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tableau-pbi-migrator")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Tableau → Power BI Migration")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# EXTRACTOR LOGIC (WORKING VERSION FROM LOCAL)
# ============================================================

def strip_ns(root: ET.Element):
    """Remove XML namespaces"""
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def clean(val: str) -> str:
    """Clean column/table names"""
    if not val:
        return ""
    return re.sub(r'[\[\]"]', "", val).strip()


def normalize_table_name(name: str) -> str:
    """Normalize table names"""
    name = clean(name)
    if ".csv_" in name:
        return name.split(".csv_", 1)[0]
    return name


def extract_hyper_metadata(hyper_path: str):
    """Extract table and column metadata from Hyper file"""
    tables: Dict[str, List[str]] = {}
    column_table_map: Dict[str, List[str]] = {}

    log.info(f"Opening Hyper file: {hyper_path}")
    
    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(hyper.endpoint, hyper_path) as conn:
            schemas = conn.catalog.get_schema_names()
            log.info(f"Found {len(schemas)} schema(s): {schemas}")
            
            for schema in schemas:
                schema_tables = conn.catalog.get_table_names(schema)
                log.info(f"Schema '{schema}' contains {len(schema_tables)} table(s)")
                
                for table in schema_tables:
                    table_name = normalize_table_name(str(table.name))
                    cols = []

                    table_def = conn.catalog.get_table_definition(table)
                    for c in table_def.columns:
                        col = clean(str(c.name))
                        cols.append(col)
                        column_table_map.setdefault(col, []).append(table_name)

                    tables[table_name] = cols
                    log.info(f"  Table '{table_name}' has columns: {cols}")

    log.info(f"HYPER EXTRACTION COMPLETE:")
    log.info(f"  Total tables: {len(tables)}")
    log.info(f"  Tables: {list(tables.keys())}")
    log.info(f"  Column-to-table map: {dict(column_table_map)}")
    
    return tables, column_table_map


def extract_relationships(root, column_table_map, tables):
    """Extract relationships from XML (working version)"""
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
        log.info(f"  Added relationship: {from_t}.{from_c} -> {to_t}.{to_c}")

    # -------- XML relationships (object-graph) --------
    log.info("Searching for XML relationships in object-graph...")
    object_graph_rels = root.findall(".//object-graph/relationships/relationship")
    log.info(f"Found {len(object_graph_rels)} relationship elements in object-graph")
    
    for rel in object_graph_rels:
        expr = rel.find("expression")
        if expr is None:
            log.debug("  Skipping relationship: no expression element")
            continue

        cols = [clean(e.get("op")) for e in expr.findall("expression") if e.get("op")]
        log.info(f"  Found expression columns: {cols}")
        
        if len(cols) != 2:
            log.debug(f"  Skipping: expected 2 columns, got {len(cols)}")
            continue

        left, right = cols
        lt = column_table_map.get(left, [])
        rt = column_table_map.get(right, [])

        log.info(f"  Column '{left}' found in tables: {lt}")
        log.info(f"  Column '{right}' found in tables: {rt}")

        if lt and rt:
            add(lt[0], left, rt[0], right)
        else:
            log.warning(f"  Could not map columns to tables: {left} -> {right}")

    # -------- Fallback: common column heuristic --------
    if not relationships:
        log.info("No XML relationships found. Using common column heuristic...")
        table_items = list(tables.items())
        for i, (t1, cols1) in enumerate(table_items):
            for t2, cols2 in table_items[i + 1:]:
                common = set(cols1) & set(cols2)
                if common:
                    log.info(f"  Tables '{t1}' and '{t2}' share columns: {common}")
                    for col in common:
                        add(t1, col, t2, col)

    log.info(f"RELATIONSHIP EXTRACTION COMPLETE: {len(relationships)} relationship(s) found")
    return relationships


def extract_metadata_from_twbx(twbx_path: str):
    """Extract metadata from TWBX file (working version)"""
    log.info("=" * 80)
    log.info("STARTING TWBX METADATA EXTRACTION")
    log.info("=" * 80)
    log.info(f"TWBX file: {twbx_path}")
    
    with tempfile.TemporaryDirectory() as tmp:
        log.info(f"Extracting TWBX to temporary directory: {tmp}")
        
        with zipfile.ZipFile(twbx_path, "r") as z:
            z.extractall(tmp)

        # List extracted files
        all_files = []
        for root_dir, _, files in os.walk(tmp):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root_dir, f), tmp)
                all_files.append(rel_path)
        log.info(f"Extracted files: {all_files}")

        twb = hyper = None
        for root_dir, _, files in os.walk(tmp):
            for f in files:
                if f.endswith(".twb"):
                    twb = os.path.join(root_dir, f)
                    log.info(f"Found TWB file: {twb}")
                elif f.endswith(".hyper"):
                    hyper = os.path.join(root_dir, f)
                    log.info(f"Found Hyper file: {hyper}")

        if not twb or not hyper:
            raise ValueError("Invalid TWBX file: missing .twb or .hyper")

        log.info("Parsing TWB XML...")
        tree = ET.parse(twb)
        root = tree.getroot()
        
        log.info(f"XML root element: <{root.tag}> with {len(root)} children")
        
        log.info("Stripping XML namespaces...")
        strip_ns(root)

        log.info("Extracting Hyper metadata...")
        tables, col_map = extract_hyper_metadata(hyper)

        log.info("Extracting relationships...")
        relationships = extract_relationships(root, col_map, tables)

        log.info("=" * 80)
        log.info("EXTRACTION SUMMARY")
        log.info("=" * 80)
        log.info(f"Tables found: {len(tables)}")
        log.info(f"Relationships found: {len(relationships)}")
        for r in relationships:
            log.info(f"  - {r['fromTable']}.{r['fromColumn']} -> {r['toTable']}.{r['toColumn']}")
        log.info("=" * 80)

        return {
            "relationships": relationships,
            "tables": tables
        }

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_second_word_table_name(filename: str) -> str:
    """Extract table name from CSV filename"""
    base = filename.split(".csv")[0]
    
    # Handle Extract_tablename format
    if base.startswith("Extract_"):
        base = base.replace("Extract_", "")
        # Remove hash suffix if present
        parts = base.split("_")
        if len(parts) > 1 and len(parts[-1]) > 20:  # Likely a hash
            base = "_".join(parts[:-1])
        return re.sub(r"[^a-zA-Z]", "", base).lower()
    
    # Handle other formats
    parts = base.split("_")
    table_name = parts[1] if len(parts) >= 2 else parts[0]
    return re.sub(r"[^a-zA-Z]", "", table_name).lower()

def get_auth_token() -> str:
    """Get Power BI authentication token"""
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"

    resp = requests.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope": "https://analysis.windows.net/powerbi/api/.default",
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def download_twbx_from_blob(folder_name: str) -> str:
    """Download TWBX file from Azure Blob Storage"""
    blob_service = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
    container = blob_service.get_container_client(TWBX_CONTAINER)

    twbx_blob_name = f"{folder_name}.twbx"

    try:
        data = container.download_blob(twbx_blob_name).readall()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".twbx")
        tmp.write(data)
        tmp.close()
        log.info(f"Downloaded TWBX: {twbx_blob_name}")
        return tmp.name
    except Exception as e:
        log.error(f"Failed to download TWBX: {str(e)}")
        raise Exception(f"TWBX file not found: {twbx_blob_name}")

# ============================================================
# MIGRATION ENDPOINT
# ============================================================

@app.post("/migrate")
def migrate(folder_name: str, target_workspace_id: str):
    try:
        # 1️⃣ AUTH
        token = get_auth_token()
        log.info("✅ Authentication successful")

        # 2️⃣ DOWNLOAD TWBX & EXTRACT METADATA
        twbx_path = download_twbx_from_blob(folder_name)
        metadata = extract_metadata_from_twbx(twbx_path)
        os.remove(twbx_path)

        # Get relationships
        relationships_metadata = metadata.get("relationships", [])
        if relationships_metadata:
            log.info(f"✅ Extracted {len(relationships_metadata)} relationships from TWBX:")
            for r in relationships_metadata:
                log.info(f"  {r['fromTable']}.{r['fromColumn']} -> {r['toTable']}.{r['toColumn']}")
        else:
            log.warning("⚠️ No relationships found in TWBX. Migration will continue without relationships.")

        # 3️⃣ READ CSVs FROM AZURE BLOB
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container = blob_service.get_container_client(CSV_CONTAINER)

        blob_tables = {}
        prefix = f"{folder_name.rstrip('/')}/"

        # Get valid tables from relationships
        valid_tables = set()
        for r in relationships_metadata:
            valid_tables.add(r["fromTable"])
            valid_tables.add(r["toTable"])
        
        # Also include all tables from metadata
        for table_name in metadata.get("tables", {}).keys():
            valid_tables.add(table_name)
        
        log.info(f"Valid tables to load: {valid_tables}")

        all_blobs = list(container.list_blobs(name_starts_with=prefix))
        log.info(f"Found {len(all_blobs)} blobs with prefix '{prefix}'")

        for blob in all_blobs:
            filename = os.path.basename(blob.name)
            if not filename.lower().endswith(".csv"):
                continue

            table_name = extract_second_word_table_name(filename)
            
            # Load table if it's in valid_tables or if no relationships exist (load all)
            if not valid_tables or table_name in valid_tables:
                try:
                    data = container.download_blob(blob.name).readall()
                    blob_tables[table_name] = pd.read_csv(pd.io.common.BytesIO(data))
                    log.info(f"✅ Loaded table: {table_name} ({len(blob_tables[table_name])} rows)")
                except Exception as e:
                    log.error(f"Failed to load {table_name}: {str(e)}")

        if not blob_tables:
            raise Exception(f"No CSV tables loaded for folder: {prefix}")

        # 4️⃣ BUILD POWER BI RELATIONSHIPS
        pbi_relationships = []
        for r in relationships_metadata:
            from_table = r["fromTable"]
            to_table = r["toTable"]
            from_col = r["fromColumn"]
            to_col = r["toColumn"]
            
            # Validate both tables exist
            if from_table not in blob_tables or to_table not in blob_tables:
                log.warning(f"⚠️ Skipping relationship {from_table}.{from_col} -> {to_table}.{to_col}: table not found")
                continue
            
            # Validate columns exist in the actual data
            from_df = blob_tables[from_table]
            to_df = blob_tables[to_table]
            
            if from_col not in from_df.columns:
                log.warning(f"⚠️ Skipping relationship: column '{from_col}' not found in table '{from_table}'")
                log.info(f"   Available columns in {from_table}: {list(from_df.columns)}")
                continue
                
            if to_col not in to_df.columns:
                log.warning(f"⚠️ Skipping relationship: column '{to_col}' not found in table '{to_table}'")
                log.info(f"   Available columns in {to_table}: {list(to_df.columns)}")
                continue
            
            # All validations passed - add the relationship
            pbi_relationships.append({
                "name": f"{from_table}_{to_table}",
                "fromTable": from_table,
                "fromColumn": from_col,
                "toTable": to_table,
                "toColumn": to_col,
                "crossFilteringBehavior": "BothDirections",
            })
            log.info(f"✅ Validated relationship: {from_table}.{from_col} -> {to_table}.{to_col}")

        # 5️⃣ CREATE DATASET
        dataset_payload = {
            "name": f"{REPORT_NAME}_DS",
            "tables": [],
            "defaultMode": "Push",
        }
        
        # Only add relationships if we have valid ones
        if pbi_relationships:
            dataset_payload["relationships"] = pbi_relationships
            log.info(f"Adding {len(pbi_relationships)} relationships to dataset")
        else:
            log.info("No relationships to add to dataset")

        for table_name, df in blob_tables.items():
            columns = []
            for col in df.columns:
                col_lower = col.lower()
                
                # Determine data type and summarization
                if "id" in col_lower:
                    dtype, summarize = "Int64", "none"
                elif df[col].dtype == "float64":
                    dtype, summarize = "Double", "sum"
                elif df[col].dtype == "int64":
                    dtype, summarize = "Int64", "sum"
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    dtype, summarize = "DateTime", "none"
                else:
                    dtype, summarize = "String", "none"

                columns.append({
                    "name": col,
                    "dataType": dtype,
                    "summarizeBy": summarize,
                })

            dataset_payload["tables"].append({
                "name": table_name,
                "columns": columns,
            })

        log.info(f"Dataset payload: {json.dumps(dataset_payload, indent=2)}")

        # Create dataset
        try:
            ds_resp = requests.post(
                f"{POWERBI_API}/groups/{target_workspace_id}/datasets",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json=dataset_payload,
            )
            
            if not ds_resp.ok:
                log.error(f"❌ Dataset creation failed: {ds_resp.status_code}")
                log.error(f"Response body: {ds_resp.text}")
                log.error(f"Request payload was: {json.dumps(dataset_payload, indent=2)}")
            
            ds_resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log.error(f"Power BI API Error Details:")
            log.error(f"  Status Code: {ds_resp.status_code}")
            log.error(f"  Response: {ds_resp.text}")
            log.error(f"  Payload sent: {json.dumps(dataset_payload, indent=2)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Power BI dataset creation failed: {ds_resp.text}"
            )
        dataset_id = ds_resp.json()["id"]
        log.info(f"✅ Dataset created: {dataset_id}")

        # 6️⃣ PUSH DATA
        time.sleep(5)
        for table_name, df in blob_tables.items():
            # Convert DataFrame to records
            rows = df.where(pd.notnull(df), None).to_dict(orient="records")
            
            # Push in batches
            batch_size = 2500
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                push_resp = requests.post(
                    f"{POWERBI_API}/groups/{target_workspace_id}/datasets/{dataset_id}/tables/{table_name}/rows",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    json={"rows": batch},
                )
                
                if not push_resp.ok:
                    log.error(f"Failed to push batch to {table_name}: {push_resp.text}")
                
                push_resp.raise_for_status()
            
            log.info(f"✅ Pushed {len(rows)} rows into {table_name}")

        # 7️⃣ CLONE REPORT
        clone_resp = requests.post(
            f"{POWERBI_API}/groups/{TEMPLATE_WORKSPACE_ID}/reports/{TEMPLATE_REPORT_ID}/Clone",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={
                "name": REPORT_NAME,
                "targetWorkspaceId": target_workspace_id,
                "targetModelId": dataset_id,
            },
        )
        clone_resp.raise_for_status()

        return {
            "status": "SUCCESS",
            "dataset_id": dataset_id,
            "report_id": clone_resp.json()["id"],
            "tables_migrated": list(blob_tables.keys()),
            "relationships_created": len(pbi_relationships),
            "relationships": relationships_metadata,
            "message": f"TWBX migrated successfully with {len(pbi_relationships)} relationships"
        }

    except Exception as e:
        log.exception("Migration failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/diagnose-twbx")
def diagnose_twbx_endpoint(folder_name: str):
    """
    Diagnostic endpoint to examine TWBX structure and find why relationships aren't extracted
    """
    try:
        import io
        from contextlib import redirect_stdout
        
        # Download TWBX
        twbx_path = download_twbx_from_blob(folder_name)
        log.info(f"Downloaded TWBX for diagnostics: {twbx_path}")
        
        # Capture diagnostic output
        output = io.StringIO()
        
        with tempfile.TemporaryDirectory() as tmp:
            # Extract TWBX
            with zipfile.ZipFile(twbx_path, "r") as z:
                z.extractall(tmp)
            
            # Find files
            twb_path = hyper_path = None
            all_files = []
            for root_dir, dirs, files in os.walk(tmp):
                for f in files:
                    full_path = os.path.join(root_dir, f)
                    rel_path = os.path.relpath(full_path, tmp)
                    all_files.append(rel_path)
                    if f.endswith(".twb"):
                        twb_path = full_path
                    elif f.endswith(".hyper"):
                        hyper_path = full_path
            
            diagnostic_info = {
                "files_in_twbx": all_files,
                "twb_file": os.path.basename(twb_path) if twb_path else None,
                "hyper_file": os.path.basename(hyper_path) if hyper_path else None,
                "xml_elements": {},
                "hyper_tables": {},
                "relationship_attempts": []
            }
            
            if not twb_path or not hyper_path:
                return {
                    "status": "ERROR",
                    "message": "Missing TWB or Hyper file",
                    "diagnostic": diagnostic_info
                }
            
            # Parse XML
            tree = ET.parse(twb_path)
            root = tree.getroot()
            
            # Count elements before namespace stripping
            diagnostic_info["xml_elements"]["before_strip"] = {
                "relationship": len(root.findall(".//relationship")),
                "relationships": len(root.findall(".//relationships")),
                "relation": len(root.findall(".//relation")),
                "datasource": len(root.findall(".//datasource")),
                "connection": len(root.findall(".//connection")),
            }
            
            # Strip namespaces
            strip_ns(root)
            
            # Count elements after namespace stripping
            diagnostic_info["xml_elements"]["after_strip"] = {
                "relationship": len(root.findall(".//relationship")),
                "relationships": len(root.findall(".//relationships")),
                "relation": len(root.findall(".//relation")),
                "datasource": len(root.findall(".//datasource")),
                "connection": len(root.findall(".//connection")),
            }
            
            # Analyze relations (joins)
            relations_info = []
            for rel in root.findall(".//relation"):
                rel_info = {
                    "type": rel.get("type"),
                    "name": rel.get("name"),
                    "join": rel.get("join"),
                    "clauses": []
                }
                
                for clause in rel.findall(".//clause"):
                    clause_info = {
                        "type": clause.get("type"),
                        "expressions": []
                    }
                    for expr in clause.findall(".//expression"):
                        clause_info["expressions"].append({
                            "op": expr.get("op"),
                            "text": expr.text
                        })
                    if clause_info["expressions"]:
                        rel_info["clauses"].append(clause_info)
                
                if rel_info["join"] or rel_info["clauses"]:
                    relations_info.append(rel_info)
            
            diagnostic_info["relations_found"] = relations_info
            
            # Analyze Hyper
            tables, col_map = extract_hyper_metadata(hyper_path)
            diagnostic_info["hyper_tables"] = {
                table: cols for table, cols in tables.items()
            }
            
            # Try relationship extraction
            log.info("Attempting XML relationship extraction...")
            xml_relationships = extract_relationships_from_xml(root, col_map)
            diagnostic_info["relationship_attempts"].append({
                "method": "XML extraction",
                "count": len(xml_relationships),
                "relationships": xml_relationships
            })
            
            log.info("Attempting schema inference...")
            inferred_relationships = infer_relationships_from_schema(tables, col_map)
            diagnostic_info["relationship_attempts"].append({
                "method": "Schema inference",
                "count": len(inferred_relationships),
                "relationships": inferred_relationships
            })
            
            os.remove(twbx_path)
            
            return {
                "status": "SUCCESS",
                "diagnostic": diagnostic_info,
                "summary": {
                    "total_tables": len(tables),
                    "xml_relationships_found": len(xml_relationships),
                    "inferred_relationships": len(inferred_relationships),
                    "total_relationships": len(xml_relationships) + len(inferred_relationships)
                }
            }
            
    except Exception as e:
        log.exception("Diagnostic failed")
        return {
            "status": "ERROR",
            "error": str(e),
            "traceback": str(e.__traceback__)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
