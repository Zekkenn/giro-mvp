"""
AWS Bedrock Knowledge Base Manager.

Handles document upload, processing, and retrieval using AWS Bedrock.
Supports both neuropsychological profiles and activity PDFs.

Single KB architecture with metadata filtering for document types:
- neuropsych_profile: Student neuropsychological assessments
- activity: Teacher-created worksheet/activity PDFs
- reference_material: Additional educational resources
"""

import os
import json
import time
import boto3
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import PyPDF2
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Singleton Instance
# ============================================================================

_manager_instance: Optional["BedrockKnowledgeManager"] = None


def get_bedrock_knowledge_manager(**kwargs) -> "BedrockKnowledgeManager":
    """
    Get or create the singleton BedrockKnowledgeManager instance.

    Args:
        **kwargs: Passed to BedrockKnowledgeManager constructor on first call

    Returns:
        BedrockKnowledgeManager singleton instance
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = BedrockKnowledgeManager(**kwargs)
    return _manager_instance


def reset_manager():
    """Reset the singleton instance (useful for testing)."""
    global _manager_instance
    _manager_instance = None


class DocumentType(str, Enum):
    """Types of documents in the knowledge base."""
    NEUROPSYCH_PROFILE = "neuropsych_profile"
    ACTIVITY = "activity"
    REFERENCE_MATERIAL = "reference_material"


@dataclass
class DocumentMetadata:
    """Metadata for uploaded documents."""
    document_id: str
    document_type: DocumentType
    title: str
    filename: str
    s3_uri: str
    upload_date: str
    total_pages: int
    tags: List[str]
    description: Optional[str] = None
    student_id: Optional[str] = None  # For neuropsych profiles
    topic: Optional[str] = None  # For activities


class BedrockKnowledgeManager:
    """
    Manages AWS Bedrock Knowledge Base for educational documents.

    Workflow:
    1. Upload PDF to S3 bucket
    2. Add metadata for indexing
    3. Sync with Bedrock Knowledge Base
    4. Query using retrieve/retrieve_and_generate APIs
    """

    def __init__(
        self,
        s3_bucket: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        data_source_id: Optional[str] = None,
        region: str = "us-east-1",
        local_storage_path: str = "bedrock_documents",
    ):
        """
        Initialize Bedrock Knowledge Manager.

        Args:
            s3_bucket: S3 bucket name for document storage
            knowledge_base_id: Bedrock Knowledge Base ID
            data_source_id: Data source ID within the Knowledge Base
            region: AWS region
            local_storage_path: Local path for document copies
        """
        self.s3_bucket = s3_bucket or os.getenv("BEDROCK_S3_BUCKET")
        self.knowledge_base_id = knowledge_base_id or os.getenv("BEDROCK_KB_ID")
        self.data_source_id = data_source_id or os.getenv("BEDROCK_DATA_SOURCE_ID")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")

        if not self.s3_bucket:
            raise ValueError("S3 bucket not configured. Set BEDROCK_S3_BUCKET env variable.")
        if not self.knowledge_base_id:
            raise ValueError("Knowledge Base ID not configured. Set BEDROCK_KB_ID env variable.")

        # AWS clients
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.bedrock_agent = boto3.client('bedrock-agent', region_name=self.region)
        self.bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=self.region)

        # Local storage
        self.storage_path = Path(local_storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Metadata tracking
        self.metadata_file = self.storage_path / "documents_metadata.json"
        self.metadata: Dict[str, DocumentMetadata] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, DocumentMetadata]:
        """Load document metadata from local JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {
                    doc_id: DocumentMetadata(**doc_data)
                    for doc_id, doc_data in data.items()
                }
        return {}

    def _save_metadata(self):
        """Save document metadata to local JSON file."""
        data = {
            doc_id: asdict(metadata)
            for doc_id, metadata in self.metadata.items()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _extract_pdf_page_count(self, pdf_path: Path) -> int:
        """Extract number of pages from PDF."""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                return len(pdf_reader.pages)
        except Exception as e:
            print(f"Warning: Could not count pages in {pdf_path}: {e}")
            return 0

    def upload_document(
        self,
        file_path: str,
        document_type: DocumentType,
        title: str,
        tags: List[str] = None,
        description: Optional[str] = None,
        student_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> str:
        """
        Upload a document to S3 and add to Bedrock Knowledge Base.

        Args:
            file_path: Path to the PDF file
            document_type: Type of document (neuropsych_profile, activity, etc.)
            title: Human-readable title
            tags: List of tags for categorization
            description: Optional description
            student_id: Student ID (for neuropsych profiles)
            topic: Topic (for activities)

        Returns:
            Document ID (S3 key)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate document ID and S3 key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = title.lower().replace(" ", "_").replace("/", "_")
        document_id = f"{document_type.value}/{timestamp}_{safe_title}.pdf"

        # Upload to S3 with metadata
        s3_metadata = {
            'document_type': document_type.value,
            'title': title,
            'upload_date': datetime.now().isoformat(),
        }

        if tags:
            s3_metadata['tags'] = ",".join(tags)
        if student_id:
            s3_metadata['student_id'] = student_id
        if topic:
            s3_metadata['topic'] = topic

        print(f"📤 Uploading {file_path.name} to S3: s3://{self.s3_bucket}/{document_id}")

        try:
            self.s3_client.upload_file(
                Filename=str(file_path),
                Bucket=self.s3_bucket,
                Key=document_id,
                ExtraArgs={
                    'Metadata': s3_metadata,
                    'ContentType': 'application/pdf',
                }
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upload to S3: {e}")

        # Save metadata locally
        page_count = self._extract_pdf_page_count(file_path)
        metadata = DocumentMetadata(
            document_id=document_id,
            document_type=document_type,
            title=title,
            filename=file_path.name,
            s3_uri=f"s3://{self.s3_bucket}/{document_id}",
            upload_date=datetime.now().isoformat(),
            total_pages=page_count,
            tags=tags or [],
            description=description,
            student_id=student_id,
            topic=topic,
        )

        self.metadata[document_id] = metadata
        self._save_metadata()

        # Copy to local storage for reference
        local_copy = self.storage_path / file_path.name
        if not local_copy.exists():
            import shutil
            shutil.copy(file_path, local_copy)

        print(f"✅ Document uploaded: {document_id}")
        print(f"   Pages: {page_count}")
        print(f"   Tags: {', '.join(tags or [])}")

        return document_id

    def sync_knowledge_base(self):
        """
        Trigger Bedrock Knowledge Base data source sync.

        This ingests new documents from S3 into the vector store.
        """
        if not self.data_source_id:
            print("⚠️  No data source ID configured. Skipping sync.")
            print("   Documents are uploaded to S3, but need manual Bedrock KB sync.")
            return

        print(f"🔄 Starting Bedrock Knowledge Base sync...")

        try:
            response = self.bedrock_agent.start_ingestion_job(
                knowledgeBaseId=self.knowledge_base_id,
                dataSourceId=self.data_source_id,
            )

            ingestion_job_id = response['ingestionJob']['ingestionJobId']
            print(f"✅ Sync started. Job ID: {ingestion_job_id}")
            print(f"   Check status in AWS Console or use get_ingestion_job()")

            return ingestion_job_id

        except Exception as e:
            print(f"❌ Failed to start sync: {e}")
            raise

    def get_sync_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of an ingestion job."""
        response = self.bedrock_agent.get_ingestion_job(
            knowledgeBaseId=self.knowledge_base_id,
            dataSourceId=self.data_source_id,
            ingestionJobId=job_id,
        )

        job = response['ingestionJob']
        status = job['status']

        print(f"Ingestion Job: {job_id}")
        print(f"Status: {status}")

        if 'statistics' in job:
            stats = job['statistics']
            print(f"Documents: {stats.get('numberOfDocumentsScanned', 0)} scanned, "
                  f"{stats.get('numberOfDocumentsIndexed', 0)} indexed")

        return job

    def retrieve(
        self,
        query: str,
        num_results: int = 3,
        document_type_filter: Optional[DocumentType] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from Knowledge Base.

        Args:
            query: Search query
            num_results: Number of results to return
            document_type_filter: Optional filter by document type

        Returns:
            List of retrieval results with content and metadata
        """
        print(f"🔍 Searching: '{query}'")

        retrieval_config = {
            'vectorSearchConfiguration': {
                'numberOfResults': num_results,
                'overrideSearchType': 'HYBRID',  # Hybrid search (vector + keyword)
            }
        }

        # Add metadata filter if specified
        if document_type_filter:
            retrieval_config['vectorSearchConfiguration']['filter'] = {
                'equals': {
                    'key': 'document_type',
                    'value': document_type_filter.value,
                }
            }

        try:
            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query},
                retrievalConfiguration=retrieval_config,
            )

            results = []
            for i, item in enumerate(response.get('retrievalResults', []), 1):
                content = item['content']['text']
                metadata = item.get('metadata', {})
                score = item.get('score', 0)

                result = {
                    'rank': i,
                    'content': content,
                    'score': score,
                    'metadata': metadata,
                    'location': item.get('location', {}),
                }
                results.append(result)

                print(f"\n--- Result {i} (score: {score:.3f}) ---")
                print(f"Content: {content[:200]}...")
                if metadata:
                    print(f"Metadata: {metadata}")

            return results

        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            raise

    def retrieve_and_generate(
        self,
        query: str,
        model_arn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Use Bedrock's RetrieveAndGenerate for direct Q&A.

        This combines retrieval + generation in one API call.

        Args:
            query: Question to answer
            model_arn: Bedrock model ARN (defaults to Claude 3 Sonnet)

        Returns:
            Generated answer with citations
        """
        if not model_arn:
            model_arn = (
                f"arn:aws:bedrock:{self.region}::foundation-model/"
                "anthropic.claude-3-sonnet-20240229-v1:0"
            )

        print(f"💬 Question: '{query}'")

        try:
            response = self.bedrock_agent_runtime.retrieve_and_generate(
                input={'text': query},
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': self.knowledge_base_id,
                        'modelArn': model_arn,
                        'retrievalConfiguration': {
                            'vectorSearchConfiguration': {
                                'numberOfResults': 3,
                            }
                        }
                    }
                }
            )

            answer = response['output']['text']
            citations = []

            for citation in response.get('citations', []):
                for reference in citation.get('retrievedReferences', []):
                    citations.append({
                        'content': reference['content']['text'][:200],
                        'metadata': reference.get('metadata', {}),
                        'location': reference.get('location', {}),
                    })

            print(f"\n✨ Answer:\n{answer}")
            if citations:
                print(f"\n📚 Sources ({len(citations)}):")
                for i, cite in enumerate(citations, 1):
                    print(f"  {i}. {cite['metadata'].get('title', 'Unknown')}")

            return {
                'answer': answer,
                'citations': citations,
                'session_id': response.get('sessionId'),
            }

        except Exception as e:
            print(f"❌ RetrieveAndGenerate failed: {e}")
            raise

    def list_documents(self, document_type: Optional[DocumentType] = None) -> List[DocumentMetadata]:
        """List all documents in the knowledge base."""
        docs = list(self.metadata.values())

        if document_type:
            docs = [d for d in docs if d.document_type == document_type]

        print(f"\n📚 Documents in Knowledge Base ({len(docs)} total)")
        print("=" * 80)

        for doc in docs:
            print(f"\n{doc.title}")
            print(f"  Type: {doc.document_type.value}")
            print(f"  File: {doc.filename}")
            print(f"  Pages: {doc.total_pages}")
            print(f"  Uploaded: {doc.upload_date}")
            if doc.tags:
                print(f"  Tags: {', '.join(doc.tags)}")
            if doc.description:
                print(f"  Description: {doc.description}")

        return docs

    # ========================================================================
    # S3 Data Source Management
    # ========================================================================

    def create_s3_data_source(
        self,
        name: str = "giro-s3-documents",
        s3_prefix: Optional[str] = None,
        inclusion_patterns: Optional[List[str]] = None,
        chunking_strategy: str = "FIXED_SIZE",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> str:
        """
        Create an S3 data source for the Bedrock Knowledge Base.

        Args:
            name: Name for the data source
            s3_prefix: Optional S3 prefix to scope documents
            inclusion_patterns: Glob patterns for files to include
            chunking_strategy: FIXED_SIZE, NONE, or HIERARCHICAL
            chunk_size: Max tokens per chunk (for FIXED_SIZE)
            chunk_overlap: Overlap between chunks (for FIXED_SIZE)

        Returns:
            Data source ID
        """
        # Check if data source already exists
        existing = self.bedrock_agent.list_data_sources(
            knowledgeBaseId=self.knowledge_base_id,
            maxResults=50,
        ).get("dataSourceSummaries", [])

        for ds in existing:
            if ds["name"] == name:
                print(f"✅ Data source '{name}' already exists: {ds['dataSourceId']}")
                self.data_source_id = ds["dataSourceId"]
                return ds["dataSourceId"]

        # Build S3 configuration
        s3_config = {
            "bucketArn": f"arn:aws:s3:::{self.s3_bucket}",
        }

        if s3_prefix:
            s3_config["inclusionPrefixes"] = [s3_prefix]

        # Build chunking configuration
        chunking_config = {"chunkingStrategy": chunking_strategy}

        if chunking_strategy == "FIXED_SIZE":
            chunking_config["fixedSizeChunkingConfiguration"] = {
                "maxTokens": chunk_size,
                "overlapPercentage": int((chunk_overlap / chunk_size) * 100),
            }

        print(f"🔧 Creating S3 data source: {name}")

        try:
            response = self.bedrock_agent.create_data_source(
                knowledgeBaseId=self.knowledge_base_id,
                name=name,
                dataSourceConfiguration={
                    "type": "S3",
                    "s3Configuration": s3_config,
                },
                vectorIngestionConfiguration={
                    "chunkingConfiguration": chunking_config,
                },
            )

            data_source_id = response["dataSource"]["dataSourceId"]
            self.data_source_id = data_source_id

            print(f"✅ Data source created: {data_source_id}")
            return data_source_id

        except Exception as e:
            print(f"❌ Failed to create data source: {e}")
            raise

    def sync_and_wait(self, timeout_seconds: int = 300) -> Dict[str, Any]:
        """
        Trigger KB sync and wait for completion.

        Args:
            timeout_seconds: Max time to wait for sync

        Returns:
            Final job status
        """
        job_id = self.sync_knowledge_base()
        if not job_id:
            return {"status": "SKIPPED", "message": "No data source configured"}

        start_time = time.time()
        while True:
            job = self.get_sync_status(job_id)
            status = job["status"]

            if status in ("COMPLETE", "FAILED"):
                return job

            if time.time() - start_time > timeout_seconds:
                return {"status": "TIMEOUT", "job_id": job_id}

            print("⏳ Waiting for sync...")
            time.sleep(5)

    # ========================================================================
    # Enhanced Retrieval with Multiple Filters
    # ========================================================================

    def retrieve_with_filters(
        self,
        query: str,
        num_results: int = 5,
        document_type: Optional[DocumentType] = None,
        student_id: Optional[str] = None,
        topic: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with multiple metadata filters.

        Args:
            query: Search query
            num_results: Max results to return
            document_type: Filter by document type
            student_id: Filter by student ID (for neuropsych profiles)
            topic: Filter by topic (for activities)
            tags: Filter by tags (any match)
            min_score: Minimum relevance score threshold

        Returns:
            Filtered retrieval results
        """
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": num_results,
                "overrideSearchType": "HYBRID",
            }
        }

        # Build filter conditions
        filters = []

        if document_type:
            filters.append({
                "equals": {"key": "document_type", "value": document_type.value}
            })

        if student_id:
            filters.append({
                "equals": {"key": "student_id", "value": student_id}
            })

        if topic:
            filters.append({
                "equals": {"key": "topic", "value": topic}
            })

        # Combine filters with AND
        if len(filters) == 1:
            retrieval_config["vectorSearchConfiguration"]["filter"] = filters[0]
        elif len(filters) > 1:
            retrieval_config["vectorSearchConfiguration"]["filter"] = {
                "andAll": filters
            }

        try:
            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=retrieval_config,
            )

            results = []
            for item in response.get("retrievalResults", []):
                score = item.get("score", 0)

                # Apply score threshold
                if score < min_score:
                    continue

                result = {
                    "content": item["content"]["text"],
                    "score": score,
                    "metadata": item.get("metadata", {}),
                    "location": item.get("location", {}),
                }
                results.append(result)

            return results

        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            return []

    # ========================================================================
    # Convenience Methods for Neuropsych Module
    # ========================================================================

    def retrieve_student_profile(
        self,
        query: str,
        student_id: Optional[str] = None,
        num_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from neuropsychological profile documents.

        Args:
            query: What aspect of the profile to retrieve
            student_id: Optional student ID filter
            num_results: Max results

        Returns:
            Relevant profile information
        """
        return self.retrieve_with_filters(
            query=query,
            num_results=num_results,
            document_type=DocumentType.NEUROPSYCH_PROFILE,
            student_id=student_id,
        )

    def retrieve_activity_content(
        self,
        query: str,
        topic: Optional[str] = None,
        num_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from activity/worksheet documents.

        Args:
            query: Concept or question to search
            topic: Optional topic filter
            num_results: Max results

        Returns:
            Relevant activity content
        """
        return self.retrieve_with_filters(
            query=query,
            num_results=num_results,
            document_type=DocumentType.ACTIVITY,
            topic=topic,
        )

    def upload_neuropsych_profile(
        self,
        file_path: str,
        student_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Upload a neuropsychological profile document.

        Args:
            file_path: Path to the PDF
            student_id: Student identifier
            title: Optional title (defaults to filename)
            description: Optional description

        Returns:
            Document ID
        """
        return self.upload_document(
            file_path=file_path,
            document_type=DocumentType.NEUROPSYCH_PROFILE,
            title=title or f"Neuropsych Profile - {student_id}",
            tags=["neuropsych", "profile", "assessment"],
            description=description or f"Neuropsychological evaluation for student {student_id}",
            student_id=student_id,
        )

    def upload_activity(
        self,
        file_path: str,
        topic: str,
        title: str,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Upload an activity/worksheet document.

        Args:
            file_path: Path to the PDF
            topic: Topic (e.g., 'Mathematics', 'Functions')
            title: Activity title
            tags: Additional tags
            description: Optional description

        Returns:
            Document ID
        """
        all_tags = ["activity", "worksheet", topic.lower()]
        if tags:
            all_tags.extend(tags)

        return self.upload_document(
            file_path=file_path,
            document_type=DocumentType.ACTIVITY,
            title=title,
            tags=all_tags,
            description=description,
            topic=topic,
        )