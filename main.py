


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
# ENHANCED EXTRACTOR LOGIC
# ============================================================

def strip_ns(root: ET.Element):
    """Remove XML namespaces from all elements"""
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
        # Also strip namespace from attributes
        for attr_name in list(el.attrib.keys()):
            if "}" in attr_name:
                new_name = attr_name.split("}", 1)[1]
                el.attrib[new_name] = el.attrib.pop(attr_name)

def clean(val: str) -> str:
    """Clean column/table names"""
    if not val:
        return ""
    # Remove brackets, quotes, and extra whitespace
    cleaned = re.sub(r'[\[\]"\']', "", val).strip()
    return cleaned

def normalize_table_name(name: str) -> str:
    """Normalize table names from various formats"""
    name = clean(name)
    
    # Handle Extract_tablename format
    if name.startswith("Extract_"):
        name = name.replace("Extract_", "")
    
    # Handle .csv_ suffix with hash
    if ".csv_" in name:
        return name.split(".csv_", 1)[0]
    
    # Handle .csv extension
    if name.endswith(".csv"):
        return name[:-4]
    
    # Remove common prefixes
    for prefix in ["Extract.", "Custom SQL Query.", "federated."]:
        if name.startswith(prefix):
            name = name.replace(prefix, "")
    
    return name.lower()

def extract_hyper_metadata(hyper_path: str):
    """Extract table and column metadata from Hyper file"""
    tables: Dict[str, List[str]] = {}
    column_table_map: Dict[str, List[str]] = {}

    try:
        with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
            with Connection(hyper.endpoint, hyper_path) as conn:
                log.info("Connected to Hyper file successfully")
                
                # Get all schemas
                schemas = conn.catalog.get_schema_names()
                log.info(f"Found schemas: {schemas}")
                
                for schema in schemas:
                    schema_tables = conn.catalog.get_table_names(schema)
                    log.info(f"Schema '{schema}' has {len(schema_tables)} tables")
                    
                    for table in schema_tables:
                        # Get the raw table name
                        raw_table_name = str(table.name)
                        table_name = normalize_table_name(raw_table_name)
                        
                        log.info(f"Processing table: {raw_table_name} -> {table_name}")
                        
                        cols = []
                        table_def = conn.catalog.get_table_definition(table)
                        
                        for c in table_def.columns:
                            col = clean(str(c.name))
                            cols.append(col)
                            column_table_map.setdefault(col, []).append(table_name)
                        
                        tables[table_name] = cols
                        log.info(f"  Columns: {cols}")

        log.info(f"Total tables extracted: {list(tables.keys())}")
        log.info(f"Column to table mapping: {dict(column_table_map)}")
        
    except Exception as e:
        log.error(f"Error extracting Hyper metadata: {str(e)}")
        raise

    return tables, column_table_map

def extract_relationships_from_xml(root: ET.Element, column_table_map: Dict[str, List[str]]) -> List[Dict]:
    """Extract relationships from TWB XML using multiple strategies"""
    relationships = []
    seen = set()

    def add_relationship(from_t: str, from_c: str, to_t: str, to_c: str):
        """Add a relationship if not already seen"""
        from_t = from_t.lower()
        to_t = to_t.lower()
        
        key = tuple(sorted([(from_t, from_c), (to_t, to_c)]))
        if key in seen:
            return False
        
        seen.add(key)
        relationships.append({
            "fromTable": from_t,
            "fromColumn": from_c,
            "toTable": to_t,
            "toColumn": to_c,
            "relationshipType": "Many-to-One"
        })
        log.info(f"Found relationship: {from_t}.{from_c} -> {to_t}.{to_c}")
        return True

    # Strategy 1: Look for explicit relationship definitions
    log.info("Strategy 1: Searching for explicit relationships in XML...")
    for rel in root.findall(".//relationship"):
        try:
            # Try to get relationship attributes
            rel_type = rel.get("type", "")
            
            # Look for expression elements
            expressions = rel.findall(".//expression")
            if len(expressions) >= 2:
                cols = []
                for expr in expressions:
                    # Try multiple attribute names
                    col = expr.get("op") or expr.get("name") or expr.text
                    if col:
                        col = clean(col)
                        cols.append(col)
                
                if len(cols) >= 2:
                    left_col, right_col = cols[0], cols[1]
                    left_tables = column_table_map.get(left_col, [])
                    right_tables = column_table_map.get(right_col, [])
                    
                    if left_tables and right_tables:
                        add_relationship(left_tables[0], left_col, right_tables[0], right_col)
        except Exception as e:
            log.warning(f"Error parsing relationship: {e}")

    # Strategy 2: Look for join clauses in datasource connections
    log.info("Strategy 2: Searching for joins in datasource connections...")
    for relation in root.findall(".//relation"):
        join_type = relation.get("join", "")
        if join_type:
            # Look for join conditions
            for clause in relation.findall(".//clause"):
                clause_type = clause.get("type", "")
                if clause_type == "join":
                    expressions = clause.findall(".//expression")
                    if len(expressions) >= 2:
                        cols = []
                        for expr in expressions:
                            col = expr.get("op") or expr.text
                            if col:
                                col = clean(col)
                                cols.append(col)
                        
                        if len(cols) >= 2:
                            left_col, right_col = cols[0], cols[1]
                            left_tables = column_table_map.get(left_col, [])
                            right_tables = column_table_map.get(right_col, [])
                            
                            if left_tables and right_tables:
                                add_relationship(left_tables[0], left_col, right_tables[0], right_col)

    # Strategy 3: Look for column references in calculations
    log.info("Strategy 3: Analyzing calculation fields...")
    for calc in root.findall(".//column[@caption]"):
        formula = calc.find(".//calculation")
        if formula is not None:
            formula_text = formula.get("formula", "")
            if formula_text:
                # Look for field references like [TableName].[ColumnName]
                field_refs = re.findall(r'\[([^\]]+)\]\.\[([^\]]+)\]', formula_text)
                if len(field_refs) >= 2:
                    for i in range(len(field_refs) - 1):
                        table1, col1 = field_refs[i]
                        table2, col2 = field_refs[i + 1]
                        
                        table1 = normalize_table_name(table1)
                        table2 = normalize_table_name(table2)
                        
                        if table1 != table2:
                            add_relationship(table1, col1, table2, col2)

    # Strategy 4: Look for metadata records
    log.info("Strategy 4: Searching metadata records...")
    for metadata_record in root.findall(".//metadata-record"):
        remote_name = metadata_record.get("remote-name", "")
        parent_name = metadata_record.find(".//parent-name")
        
        if remote_name and parent_name is not None:
            parent = parent_name.text or parent_name.get("name", "")
            if parent and remote_name != parent:
                # These might be related columns
                for table1, cols1 in column_table_map.items():
                    if clean(remote_name) in [clean(c) for c in cols1]:
                        for table2, cols2 in column_table_map.items():
                            if table1 != table2 and clean(parent) in [clean(c) for c in cols2]:
                                add_relationship(table1, clean(remote_name), table2, clean(parent))

    return relationships

def infer_relationships_from_schema(tables: Dict[str, List[str]], column_table_map: Dict[str, List[str]]) -> List[Dict]:
    """Infer relationships based on common naming patterns and schema analysis"""
    relationships = []
    seen = set()

    def add_relationship(from_t: str, from_c: str, to_t: str, to_c: str):
        from_t = from_t.lower()
        to_t = to_t.lower()
        
        key = tuple(sorted([(from_t, from_c), (to_t, to_c)]))
        if key in seen:
            return False
        
        seen.add(key)
        relationships.append({
            "fromTable": from_t,
            "fromColumn": from_c,
            "toTable": to_t,
            "toColumn": to_c,
            "relationshipType": "Many-to-One"
        })
        log.info(f"Inferred relationship: {from_t}.{from_c} -> {to_t}.{to_c}")
        return True

    log.info("Inferring relationships from schema patterns...")
    
    # Strategy 1: Look for foreign key patterns (table_id in other tables)
    table_list = list(tables.items())
    for i, (table1, cols1) in enumerate(table_list):
        for table2, cols2 in table_list[i + 1:]:
            # Check if table1 has table2_id or vice versa
            for col in cols1:
                col_lower = col.lower()
                # Check if this column references table2
                if col_lower == f"{table2}_id" or col_lower == f"{table2}id":
                    # Find the primary key column in table2 (usually 'id' or 'table2_id')
                    pk_candidates = [c for c in cols2 if c.lower() in ['id', f"{table2}_id", f"{table2}id"]]
                    if pk_candidates:
                        add_relationship(table1, col, table2, pk_candidates[0])
                    else:
                        add_relationship(table1, col, table2, col)
            
            for col in cols2:
                col_lower = col.lower()
                if col_lower == f"{table1}_id" or col_lower == f"{table1}id":
                    pk_candidates = [c for c in cols1 if c.lower() in ['id', f"{table1}_id", f"{table1}id"]]
                    if pk_candidates:
                        add_relationship(table2, col, table1, pk_candidates[0])
                    else:
                        add_relationship(table2, col, table1, col)

    # Strategy 2: Common column names (same name in multiple tables)
    for col, table_names in column_table_map.items():
        if len(table_names) >= 2:
            # Connect tables that share this column
            for i in range(len(table_names)):
                for j in range(i + 1, len(table_names)):
                    add_relationship(table_names[i], col, table_names[j], col)

    return relationships

def extract_metadata_from_twbx(twbx_path: str):
    """Extract comprehensive metadata from TWBX file"""
    with tempfile.TemporaryDirectory() as tmp:
        # Extract TWBX
        with zipfile.ZipFile(twbx_path, "r") as z:
            z.extractall(tmp)
            log.info(f"Extracted TWBX to {tmp}")
            log.info(f"Contents: {os.listdir(tmp)}")

        # Find TWB and Hyper files
        twb_path = hyper_path = None
        for root_dir, dirs, files in os.walk(tmp):
            for f in files:
                full_path = os.path.join(root_dir, f)
                if f.endswith(".twb"):
                    twb_path = full_path
                    log.info(f"Found TWB: {twb_path}")
                elif f.endswith(".hyper"):
                    hyper_path = full_path
                    log.info(f"Found Hyper: {hyper_path}")

        if not twb_path or not hyper_path:
            raise ValueError("Invalid TWBX file: missing .twb or .hyper")

        # Parse TWB XML
        tree = ET.parse(twb_path)
        root = tree.getroot()
        strip_ns(root)

        # Extract metadata from Hyper
        tables, col_map = extract_hyper_metadata(hyper_path)
        
        if not tables:
            raise ValueError("No tables found in Hyper file")

        # Extract relationships using multiple strategies
        log.info("=" * 60)
        log.info("EXTRACTING RELATIONSHIPS")
        log.info("=" * 60)
        
        relationships_from_xml = extract_relationships_from_xml(root, col_map)
        log.info(f"Found {len(relationships_from_xml)} relationships from XML")
        
        # If no relationships found in XML, try to infer them
        if not relationships_from_xml:
            log.info("No explicit relationships found. Attempting to infer from schema...")
            relationships = infer_relationships_from_schema(tables, col_map)
        else:
            relationships = relationships_from_xml

        log.info("=" * 60)
        log.info(f"TOTAL RELATIONSHIPS FOUND: {len(relationships)}")
        log.info("=" * 60)
        
        for r in relationships:
            log.info(f"  {r['fromTable']}.{r['fromColumn']} -> {r['toTable']}.{r['toColumn']}")

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
            if r["fromTable"] in blob_tables and r["toTable"] in blob_tables:
                pbi_relationships.append({
                    "name": f"{r['fromTable']}_{r['toTable']}",
                    "fromTable": r["fromTable"],
                    "fromColumn": r["fromColumn"],
                    "toTable": r["toTable"],
                    "toColumn": r["toColumn"],
                    "crossFilteringBehavior": "BothDirections",
                })
                log.info(f"✅ Created PBI relationship: {r['fromTable']}.{r['fromColumn']} -> {r['toTable']}.{r['toColumn']}")

        # 5️⃣ CREATE DATASET
        dataset_payload = {
            "name": f"{REPORT_NAME}_DS",
            "tables": [],
            "defaultMode": "Push",
        }
        
        # Only add relationships if we have valid ones
        if pbi_relationships:
            dataset_payload["relationships"] = pbi_relationships

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
        ds_resp = requests.post(
            f"{POWERBI_API}/groups/{target_workspace_id}/datasets",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json=dataset_payload,
        )
        
        if not ds_resp.ok:
            log.error(f"Dataset creation failed: {ds_resp.status_code}")
            log.error(f"Response: {ds_resp.text}")
        
        ds_resp.raise_for_status()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
