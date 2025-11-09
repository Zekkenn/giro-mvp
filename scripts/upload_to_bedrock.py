#!/usr/bin/env python3
"""
Upload educational resources to AWS Bedrock Knowledge Base.

Uploads:
1. Neuropsychological profile (student assessment)
2. Activity worksheet (with step-by-step exercises)

Tests:
- Profile retrieval (strengths, vulnerabilities)
- Activity step retrieval (get step N based on progress)
"""

import json
import os
import tempfile
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

REGION = os.getenv("AWS_REGION", "us-east-1")

# Your Bedrock KB Configuration
KB_ROLE_ARN = "arn:aws:iam::123456789012:role/bedrock-kb-role"
OSS_COLLECTION_ARN = "arn:aws:aoss:us-east-1:123456789012:collection/your-collection-id"
EMBEDDING_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
TEXT_MODEL_ARN = "anthropic.claude-3-5-sonnet-20240620-v1:0"

KB_NAME = "kb-giro-teacher-poc"
S3_BUCKET = "your-kb-bucket-name"
DATA_SOURCE_NAME = "core-knowledge-base"
INDEX_NAME = "giro_documents"

# Document paths
PROJECT_ROOT = Path(__file__).parent.parent
ACTIVITY_PDF = PROJECT_ROOT / "resources" / "activities" / "activity.pdf"
PROFILE_PDF = PROJECT_ROOT / "resources" / "profiles" / "student1_profile.pdf"

# AWS Clients
s3 = boto3.client("s3", region_name=REGION)
agent = boto3.client("bedrock-agent", region_name=REGION)
rt = boto3.client("bedrock-agent-runtime", region_name=REGION)


# =============================================================================
# S3 Upload
# =============================================================================

def ensure_bucket():
    """Create bucket if needed."""
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        print(f"Bucket exists: {S3_BUCKET}")
    except:
        print(f"No bucket: {S3_BUCKET}")


def upload_document(file_path: Path, s3_key: str, metadata: dict):
    """
    Upload PDF with metadata JSON file for Bedrock KB.

    Bedrock KB requires a .metadata.json file alongside documents for
    metadata filtering to work. S3 object metadata is NOT indexed.
    """
    print(f"\nUploading: {file_path.name} -> s3://{S3_BUCKET}/{s3_key}")
    print(f"  Metadata: {metadata}")

    # 1. Upload the PDF
    s3.upload_file(
        str(file_path),
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "application/pdf"},
    )
    print("  PDF uploaded!")

    # 2. Upload metadata JSON file (required for Bedrock KB filtering)
    # The metadata file must be named: <document_name>.metadata.json
    metadata_key = f"{s3_key}.metadata.json"

    # Bedrock KB metadata format
    bedrock_metadata = {
        "metadataAttributes": metadata
    }

    # Create temp file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(bedrock_metadata, f)
        temp_path = f.name

    s3.upload_file(
        temp_path,
        S3_BUCKET,
        metadata_key,
        ExtraArgs={"ContentType": "application/json"},
    )
    os.unlink(temp_path)  # Clean up temp file

    print(f"  Metadata JSON uploaded: {metadata_key}")


# =============================================================================
# Knowledge Base Setup
# =============================================================================

def ensure_kb():
    """Find or create KB."""
    existing = agent.list_knowledge_bases(maxResults=100)["knowledgeBaseSummaries"]
    for kb in existing:
        if kb["name"] == KB_NAME:
            print(f"Found KB: {kb['knowledgeBaseId']}")
            return kb["knowledgeBaseId"]

    print(f"KB not found: {existing}")

    return None


def ensure_data_source(kb_id: str):
    """Find or create S3 data source."""
    ds_list = agent.list_data_sources(knowledgeBaseId=kb_id, maxResults=50)["dataSourceSummaries"]
    for ds in ds_list:
        if ds["name"] == DATA_SOURCE_NAME:
            print(f"Found data source: {ds['dataSourceId']}")
            return ds["dataSourceId"]

    print(f"Datasource not found: {ds_list}")
    return None


def sync_kb(kb_id: str, ds_id: str):
    """Run ingestion and wait."""
    print("\nStarting ingestion...")
    job = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    job_id = job["ingestionJob"]["ingestionJobId"]

    while True:
        j = agent.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )["ingestionJob"]
        status = j["status"]
        print(f"  Status: {status}")

        if status == "COMPLETE":
            stats = j.get("statistics", {})
            print(f"  Indexed: {stats.get('numberOfNewDocumentsIndexed', 0)} docs")
            break
        elif status == "FAILED":
            raise RuntimeError(f"Ingestion failed: {j.get('failureReasons')}")
        time.sleep(5)


# =============================================================================
# Retrieval Functions
# =============================================================================

def retrieve(kb_id: str, query: str, doc_type_filter: str = None, k: int = 5):
    """Retrieve passages from KB with optional document type filter."""
    print(f"\nQuery: '{query}'")
    if doc_type_filter:
        print(f"Filter: document_type={doc_type_filter}")

    config = {
        "vectorSearchConfiguration": {
            "numberOfResults": k,
            "overrideSearchType": "HYBRID",
        }
    }

    # Add metadata filter if specified
    if doc_type_filter:
        config["vectorSearchConfiguration"]["filter"] = {
            "equals": {"key": "document_type", "value": doc_type_filter}
        }

    response = rt.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": query},
        retrievalConfiguration=config,
    )

    results = []
    for i, item in enumerate(response.get("retrievalResults", []), 1):
        content = item["content"]["text"]
        score = item.get("score", 0)
        metadata = item.get("metadata", {})

        results.append({"content": content, "score": score, "metadata": metadata})

        print(f"\n[{i}] Score: {score:.3f}")
        print(f"    Type: {metadata.get('document_type', 'unknown')}")
        print(f"    Content: {content[:200]}...")

    return results


def get_next_step(kb_id: str, completed_step: int):
    """
    Get the next step in the activity after completing a step.

    Args:
        kb_id: Knowledge Base ID
        completed_step: The step number just completed (e.g., 3)

    Returns:
        Content for the next step (e.g., step 4)
    """
    next_step = completed_step + 1
    # Use exercise number since activity uses "Ejercicio X" format
    query = f"Ejercicio {next_step}"

    print(f"\n{'='*60}")
    print(f"Student completed Exercise {completed_step}, retrieving Exercise {next_step}...")
    print("=" * 60)

    # Don't filter by metadata for now - rely on semantic search
    results = retrieve(kb_id, query, doc_type_filter=None, k=3)

    if results:
        print(f"\nNext exercise content found!")
        return results[0]["content"]
    else:
        print("No more exercises found - activity may be complete!")
        return None


def list_s3_files():
    """List files in S3 bucket to verify uploads."""
    print("\nS3 Bucket Contents:")
    print("-" * 40)
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET)
        for obj in response.get("Contents", []):
            print(f"  {obj['Key']} ({obj['Size']} bytes)")
    except Exception as e:
        print(f"  Error listing bucket: {e}")


def ask_with_context(kb_id: str, question: str):
    """Use RetrieveAndGenerate for contextual Q&A."""
    print(f"\nQuestion: '{question}'")
    print("-" * 40)

    response = rt.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": TEXT_MODEL_ARN,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5,
                        "overrideSearchType": "HYBRID",
                    }
                },
            },
        },
    )

    answer = response["output"]["text"]
    print(f"\nAnswer:\n{answer}")
    return answer


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("GIRO - Bedrock KB Upload & Test")
    print("=" * 60)

    # 1. Setup S3
    print("\n1. Setting up S3...")
    ensure_bucket()

    # 2. Upload documents
    print("\n2. Uploading documents...")

    # List current S3 files first
    list_s3_files()

    upload_document(
        PROFILE_PDF,
        "neuropsych_profile/student1_profile.pdf",
        {
            "document_type": "neuropsych_profile",
            "title": "Neuropsychological Assessment - Student 1",
            "student_id": "student1",
            "tags": "neuropsych,profile,assessment,ataxia",
        },
    )

    upload_document(
        ACTIVITY_PDF,
        "activity/math_functions.pdf",
        {
            "document_type": "activity",
            "title": "Mathematical Functions - Introduction",
            "topic": "Mathematics",
            "tags": "activity,worksheet,functions,steps",
        },
    )

    # 3. Setup KB
    print("\n3. Setting up Knowledge Base...")
    kb_id = ensure_kb()
    ds_id = ensure_data_source(kb_id)

    # 4. Sync
    print("\n4. Syncing Knowledge Base...")
    sync_kb(kb_id, ds_id)

    # 5. Test retrieval
    print("\n" + "=" * 60)
    print("5. Testing Retrieval")
    print("=" * 60)

    # Test: Basic retrieval WITHOUT filter (verify docs are indexed)
    print("\n--- Test: Basic Retrieval (no filter) ---")
    results = retrieve(kb_id, "student cognitive profile assessment", doc_type_filter=None)
    if not results:
        print("WARNING: No results without filter - documents may not be indexed!")

    print("\n--- Test: Activity content (no filter) ---")
    results = retrieve(kb_id, "function mathematics exercise", doc_type_filter=None)
    if not results:
        print("WARNING: No activity results - check if activity.pdf is indexed!")

    # Test: With metadata filter (requires .metadata.json files)
    print("\n--- Test: Student Strengths (with filter) ---")
    retrieve(kb_id, "cognitive strengths visual reasoning attention", doc_type_filter="neuropsych_profile")

    print("\n--- Test: Student Vulnerabilities (with filter) ---")
    retrieve(kb_id, "learning vulnerabilities difficulties challenges", doc_type_filter="neuropsych_profile")

    # Test: Activity step progression
    print("\n--- Test: Activity Step Progression ---")

    # Simulate: Student completed step 1, get step 2
    get_next_step(kb_id, completed_step=1)

    # Simulate: Student completed step 3, get step 4
    get_next_step(kb_id, completed_step=3)

    # Test: Combined Q&A
    print("\n--- Test: Profile-Aware Teaching ---")
    ask_with_context(
        kb_id,
        "Given the student's neuropsychological profile, how should I explain the concept of a function? What strengths can I leverage?"
    )

    # Print env vars
    print("\n" + "=" * 60)
    print("Setup Complete! Add to .env:")
    print("=" * 60)
    print(f"BEDROCK_KB_ID={kb_id}")
    print(f"BEDROCK_DATA_SOURCE_ID={ds_id}")
    print(f"BEDROCK_S3_BUCKET={S3_BUCKET}")


if __name__ == "__main__":
    main()