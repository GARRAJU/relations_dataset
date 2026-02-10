


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


import os
import re
import time
import logging
import tempfile
import requests
import pandas as pd
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# ✅ Import extractor
from extractor import extract_metadata_from_twbx

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

# Containers
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
# STATIC RELATIONSHIPS (fallback)
# ============================================================

STATIC_RELATIONSHIPS = [
    {
        "fromTable": "products",
        "fromColumn": "ProductID",
        "toTable": "sales",
        "toColumn": "ProductID",
        "relationshipType": "Many-to-One"
    },
    {
        "fromTable": "sales",
        "fromColumn": "CustomerID",
        "toTable": "customers",
        "toColumn": "CustomerID",
        "relationshipType": "Many-to-One"
    }
]

# ============================================================
# HELPERS
# ============================================================

def extract_second_word_table_name(filename: str) -> str:
    """
    Extract_customers.csv_HASH -> customers
    """
    base = filename.split(".csv")[0]
    parts = base.split("_")
    table_name = parts[1] if len(parts) >= 2 else parts[0]
    return re.sub(r"[^a-zA-Z]", "", table_name).lower()


def get_auth_token() -> str:
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
    """
    Downloads <folder_name>.twbx directly from TWBX container
    """
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
# MIGRATION API - UPDATED WITH EXTRACTOR RELATIONSHIPS
# ============================================================

@app.post("/migrate-static")
def migrate_static(folder_name: str, target_workspace_id: str):
    try:
        # ----------------------------------------------------
        # 1. AUTH
        # ----------------------------------------------------
        token = get_auth_token()
        log.info("✅ Authentication successful")

        # ----------------------------------------------------
        # 2. DOWNLOAD TWBX & EXTRACT METADATA
        # ----------------------------------------------------
        twbx_path = download_twbx_from_blob(folder_name)
        metadata = extract_metadata_from_twbx(twbx_path)
        os.remove(twbx_path)

        # Use extracted relationships if available, else fallback to STATIC_RELATIONSHIPS
        # relationships_metadata = metadata.get("relationships") or STATIC_RELATIONSHIPS
        # log.info(f"Using {len(relationships_metadata)} relationships")
        relationships_metadata = metadata.get("relationships")
        if relationships_metadata:
            log.info(f"✅ Extracted {len(relationships_metadata)} relationships from TWBX:")
            for r in relationships_metadata:
                log.info(f"  {r['fromTable']}.{r['fromColumn']} -> {r['toTable']}.{r['toColumn']}")
        else:
            log.warning("⚠️ No relationships found in TWBX. Using fallback static relationships.")
            relationships_metadata = STATIC_RELATIONSHIPS

        # ----------------------------------------------------
        # 3. READ CSVs FROM AZURE BLOB
        # ----------------------------------------------------
        blob_service = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )
        container = blob_service.get_container_client(CSV_CONTAINER)

        blob_tables = {}
        prefix = f"{folder_name.rstrip('/')}/"

        valid_tables = set()
        for r in relationships_metadata:
            valid_tables.add(r["fromTable"])
            valid_tables.add(r["toTable"])
        log.info(f"Valid tables from relationships: {valid_tables}")

        all_blobs = list(container.list_blobs(name_starts_with=prefix))
        log.info(f"Found {len(all_blobs)} blobs with prefix '{prefix}'")

        for blob in all_blobs:
            filename = os.path.basename(blob.name)
            if not filename.lower().endswith(".csv"):
                continue

            table_name = extract_second_word_table_name(filename)
            if table_name not in valid_tables:
                continue

            try:
                data = container.download_blob(blob.name).readall()
                blob_tables[table_name] = pd.read_csv(pd.io.common.BytesIO(data))
                log.info(f"Loaded table: {table_name}")
            except Exception as e:
                log.error(f"Failed to load {table_name}: {str(e)}")

        if not blob_tables:
            raise Exception(f"No CSV tables loaded for folder: {prefix}")

        # ----------------------------------------------------
        # 4. BUILD POWER BI RELATIONSHIPS
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # 5. CREATE DATASET
        # ----------------------------------------------------
        dataset_payload = {
            "name": f"{REPORT_NAME}_DS",
            "tables": [],
            "relationships": pbi_relationships if pbi_relationships else None,
            "defaultMode": "Push",
        }

        for table_name, df in blob_tables.items():
            columns = []
            for col in df.columns:
                if "id" in col.lower():
                    dtype, summarize = "Int64", "none"
                elif df[col].dtype == "float64":
                    dtype, summarize = "Double", "sum"
                elif df[col].dtype == "int64":
                    dtype, summarize = "Int64", "sum"
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

        ds_resp = requests.post(
            f"{POWERBI_API}/groups/{target_workspace_id}/datasets",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json=dataset_payload,
        )
        ds_resp.raise_for_status()
        dataset_id = ds_resp.json()["id"]
        log.info(f"✅ Dataset created: {dataset_id}")

        # ----------------------------------------------------
        # 6. PUSH DATA
        # ----------------------------------------------------
        time.sleep(5)
        for table_name, df in blob_tables.items():
            rows = df.where(pd.notnull(df), None).to_dict(orient="records")
            for i in range(0, len(rows), 2500):
                requests.post(
                    f"{POWERBI_API}/groups/{target_workspace_id}/datasets/{dataset_id}/tables/{table_name}/rows",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    json={"rows": rows[i:i + 2500]},
                ).raise_for_status()
            log.info(f"Pushed {len(rows)} rows into {table_name}")

        # ----------------------------------------------------
        # 7. CLONE REPORT
        # ----------------------------------------------------
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
            "relationships_created": len(pbi_relationships) > 0,
            "relationships": relationships_metadata,
            "message": "TWBX metadata + data migrated successfully with extracted relationships"
        }

    except Exception as e:
        log.exception("Migration failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}
