def setup_async_compatibility():
    """Enhanced async compatibility setup for all deployment environments"""
    import os
    import sys

    try:
        import nest_asyncio
        import asyncio

        # Always apply nest_asyncio in deployment environments
        deployment_indicators = [
            "gunicorn" in sys.modules,
            "uvicorn" in sys.modules,
            "DYNO" in os.environ,  # Heroku
            "RAILWAY_ENVIRONMENT" in os.environ,  # Railway
            "VERCEL" in os.environ,  # Vercel
            "AWS_LAMBDA_FUNCTION_NAME" in os.environ,  # AWS Lambda
            "GOOGLE_CLOUD_PROJECT" in os.environ,  # Google Cloud
            "RENDER" in os.environ,  # Render
            any(
                indicator in str(sys.argv)
                for indicator in ["gunicorn", "uvicorn", "waitress"]
            ),
        ]

        is_deployment = any(deployment_indicators)

        # Check for existing event loop
        try:
            current_loop = asyncio.get_running_loop()
            has_running_loop = True
            loop_type = str(type(current_loop))
            print(f"⚠️ Detected running event loop: {loop_type}")
        except RuntimeError:
            has_running_loop = False
            print("✅ No running event loop detected")

        # Apply nest_asyncio if needed
        if has_running_loop or is_deployment:
            try:
                nest_asyncio.apply()
                print("✅ nest_asyncio applied successfully")
                os.environ["ASYNC_COMPATIBILITY_MODE"] = "nest_asyncio"
                return True
            except Exception as e:
                print(f"⚠️ nest_asyncio application failed: {e}")
                os.environ["ASYNC_COMPATIBILITY_MODE"] = "isolated_only"
                return True
        else:
            print("✅ Standard async mode (no patches needed)")
            os.environ["ASYNC_COMPATIBILITY_MODE"] = "standard"
            return True

    except ImportError:
        print("❌ nest_asyncio not available, install with: pip install nest_asyncio")
        os.environ["ASYNC_COMPATIBILITY_MODE"] = "isolated_only"
        return False
    except Exception as e:
        print(f"⚠️ Async compatibility setup error: {e}")
        os.environ["ASYNC_COMPATIBILITY_MODE"] = "isolated_only"
        return True


# Apply this BEFORE any other imports
setup_async_compatibility()
# importations
import sys
import tempfile
import uuid
import os
from bs4 import BeautifulSoup
import json
from typing import IO, Union
import io
import time
import hashlib
import subprocess
import re
import math
import logging
from pathlib import Path
from llama_index.core import Document as LlamaDocument
from typing import Optional, Dict, Any, List, Tuple
from base64 import urlsafe_b64encode
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from llama_cloud_services import LlamaParse
import requests
from dotenv import load_dotenv

# LlamaIndex / Ollama / Qdrant imports
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from llama_index.core.node_parser import SentenceSplitter as LlamaSentenceSplitter
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SentenceSplitter as LlamaSentenceSplitter,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.readers.structured_data.base import StructuredDataReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# File readers (multi-format)
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PDFReader,
    EpubReader,
    FlatReader,
    HTMLTagReader,
    ImageReader,
    IPYNBReader,
    MarkdownReader,
    MboxReader,
    PptxReader,
    PandasCSVReader,
    PyMuPDFReader,
    XMLReader,
    PagedCSVReader,
    CSVReader,
)
from llama_index.core import PromptTemplate
from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Setup logging (retrieval debug)
logger = logging.getLogger("rag_debug")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

# Load env
load_dotenv()
# Configuration - Enhanced with better defaults
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2048"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "./data")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
MS_TENANT_ID = os.getenv("MS_TENANT_ID")
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
ONEDRIVE_SHARED_URL = os.getenv("ONEDRIVE_SHARED_URL")
SITE_SEARCH_DOMAIN = os.getenv("SITE_SEARCH_DOMAIN", "")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "1"))
RAG_MAX_SOURCE_CHARS = int(os.getenv("RAG_MAX_SOURCE_CHARS", "1000"))
WEB_REQ_TIMEOUT_SECS = 20
LLM_COMPLETE_TIMEOUT = 200
LLM_FALLBACK_TIMEOUT = 100
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.1"))
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
PARSED_CACHE_DIR = os.getenv("PARSED_CACHE_DIR", "data/parsed")
DATA_FINGERPRINT_PATH = str(
    Path(os.getenv("DATA_FINGERPRINT_PATH", "data/fingerprint.json")).resolve()
)

RAG_PROMPT_TEMPLATE = """
You are a knowledgeable assistant helping users find information. Provide clear, friendly, and conversational answers based on the available data.

**USER QUESTION**: {query_str}
**AVAILABLE DATA**: {context_str}

**CRITICAL RULES**:
1. If user asks for "Extra Data SIM Code" → Look exactly for "Extra SIM Code" (like EXTRAF25)
2. Never mix up different types of codes or values
3. Quote the EXACT value from the data - do not substitute
4. Give excat the same value ,name , code , price etc .

**HOW TO ANSWER**:
**STEP-BY-STEP PROCESS**:
    1. Identify what specific information the user wants
    2. Scan the data for that EXACT field type
    3. Extract only that specific value
    4. Double-check you have the right type of information

1. **For "What's the best..." questions**:
   - Start with "The best [item] is the [specific name]"
   - List key features and benefits in bullet points
   - Include pricing and technical details naturally

2. **For specific value questions** (price, code, etc.):
   - Give the direct answer first: "The [requested item] is [value]"
   - Add context if helpful

3. **For general questions**:
   - Provide a comprehensive but easy-to-read answer
   - Use bullet points for multiple items or features
   - Structure information logically

**FORMATTING GUIDELINES**:
- Use natural, conversational language
- Don't copy raw table formats - rewrite them naturally
- Use bullet points for features and details
- Bold important information like product names and prices
- Keep all numbers, codes, and prices exactly as provided in the data
- End with a helpful closing if appropriate

**EXAMPLES OF GOOD ANSWERS**:

Example 1 - Best product query:
"The best gaming router available is the **XR1000 Nighthawk WiFi 6 Gaming Router**. Here's what makes it great:

• **Advanced WiFi 6 technology** for superior gaming performance
• **Monthly installment**: BD 0.800
• **Product code**: XR1000I
• **VAT amount**: BD 1.920

This router is specifically designed for gaming with optimized performance features."

Example 2 - Specific value query:
"The monthly rental for Fiber Extra is **BD 19.150**."

Example 3 - Comparison/features query:
"Here are the available Fiber packages:

• **Fiber Basic**: BD 12.500 monthly
• **Fiber Extra**: BD 19.150 monthly
• **Fiber Elite**: BD 25.300 monthly

Each package offers different speeds and features to match your needs."

**YOUR RESPONSE**:
"""

prompt_template = PromptTemplate(RAG_PROMPT_TEMPLATE)


# Function to check Ollama status and embedding model
def check_ollama_embedding_status():
    """Check if Ollama and the embedding model are ready"""
    try:
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
        if response.status_code != 200:
            return {"status": "error", "message": "Ollama server not responding"}

        # Check if the embedding model is available
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "test"},
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            embedding_dim = len(result.get("embedding", []))
            return {
                "status": "ready",
                "message": f"nomic-embed-text ready (dimension: {embedding_dim})",
                "dimension": embedding_dim,
            }
        elif response.status_code == 404:
            return {
                "status": "model_missing",
                "message": "nomic-embed-text not found. Run: ollama pull nomic-embed-text",
            }
        else:
            return {
                "status": "error",
                "message": f"Unexpected response: {response.status_code}",
            }

    except requests.RequestException:
        return {
            "status": "error",
            "message": "Cannot connect to Ollama. Make sure it's running: ollama serve",
        }
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}


def ensure_ollama_model():
    """Ensure the nomic-embed-text model is available"""
    status = check_ollama_embedding_status()

    if status["status"] == "model_missing":
        print("🔄 nomic-embed-text model not found. Attempting to pull...")
        try:
            result = subprocess.run(
                ["ollama", "pull", "nomic-embed-text"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )
            if result.returncode == 0:
                print("✅ Successfully pulled nomic-embed-text model")
                return check_ollama_embedding_status()
            else:
                print(f"❌ Failed to pull model: {result.stderr}")
                return {
                    "status": "error",
                    "message": f"Failed to pull model: {result.stderr}",
                }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Model pull timed out"}
        except FileNotFoundError:
            return {
                "status": "error",
                "message": "Ollama CLI not found. Make sure Ollama is installed.",
            }
        except Exception as e:
            return {"status": "error", "message": f"Error pulling model: {e}"}

    return status


# LLM & Embedding Settings
Settings.llm = Ollama(
    model="llama3:8b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    request_timeout=LLM_COMPLETE_TIMEOUT,
    additional_kwargs={"num_ctx": 2048, "num_thread": 4},
)


# Initialize Ollama Embedding
def initialize_ollama_embedding():
    """Initialize Ollama embedding with proper error handling"""

    # First ensure the model is available
    status = ensure_ollama_model()

    if status["status"] != "ready":
        logger.error(f"Embedding model not ready: {status['message']}")
        raise RuntimeError(f"Cannot initialize embedding: {status['message']}")

    try:
        # Use LlamaIndex's OllamaEmbedding
        embedding = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url=OLLAMA_BASE_URL,
            ollama_additional_kwargs={"mirostat": 0},
        )

        # Test the embedding
        test_emb = embedding.get_text_embedding("test embedding")
        logger.info(
            f"✅ Ollama embedding initialized successfully with dimension: {len(test_emb)}"
        )
        return embedding

    except Exception as e:
        logger.error(f"❌ Failed to initialize Ollama embedding: {e}")
        raise RuntimeError(f"Failed to initialize embedding model: {e}")


# Set up embedding model
try:
    Settings.embed_model = initialize_ollama_embedding()
except Exception as e:
    logger.error(f"Failed to initialize Ollama embedding: {e}")
    print("❌ Failed to initialize Ollama embedding")
    print("💡 Please ensure:")
    print("   1. Ollama is running: ollama serve")
    print("   2. Model is available: ollama pull nomic-embed-text")
    exit(1)

# Node Parser
Settings.node_parser = None
Settings.context_window = 8000
Settings.chunk_size_limit = 1024


# Utilities
def ensure_services_running():
    """Enhanced health checks with better error handling"""
    print("Checking required services...")
    services = [
        {"name": "Qdrant", "url": QDRANT_HOST, "endpoint": "/collections"},
        {"name": "Ollama", "url": OLLAMA_BASE_URL, "endpoint": "/api/version"},
    ]
    all_ok = True
    for s in services:
        try:
            r = requests.get(f"{s['url']}{s['endpoint']}", timeout=10)
            if r.status_code < 400:
                print(f"✅ {s['name']} is running")
            else:
                print(f"⚠️ {s['name']} returned {r.status_code}")
                all_ok = False
        except requests.RequestException as e:
            print(f"❌ {s['name']} is not available: {e}")
            all_ok = False

    if not all_ok:
        compose_file = Path("docker-compose.yml")
        if compose_file.exists():
            print("\nAttempting to restart services with docker-compose...")
            try:
                subprocess.run(["docker-compose", "up", "-d"], check=True)
                print("Waiting for services to initialize...")
                time.sleep(15)
            except subprocess.SubprocessError as e:
                print(f"Error starting services: {e}")
        else:
            print(
                "Please ensure services are running:\n1) Qdrant  : 6333\n2) Ollama  : 11434"
            )


# LlamaCloud : LlamaParser
class LlamaCloudParseHelper:
    def __init__(self, api_key: Optional[str], cache_dir: str = PARSED_CACHE_DIR):
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        self.available = bool(self.api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parser = None

        # FIX: Define async_mode BEFORE using it
        self.async_mode = os.getenv("ASYNC_COMPATIBILITY_MODE", "standard")
        self.use_isolated_parsing = self.async_mode == "isolated_only"
        # END FIX

        if self.available:
            try:
                # Use the same options as your LlamaCloud playground snippet
                self.parser = LlamaParse(
                    api_key=self.api_key,
                    num_workers=4,
                    verbose=False,
                    language="en",
                    # Important parse options to preserve tables & page boundaries
                    parse_mode="parse_page_with_agent",
                    model="openai-gpt-4-1-mini",
                    high_res_ocr=True,
                    adaptive_long_table=True,
                    outlined_table_extraction=True,
                    output_tables_as_HTML=True,
                    page_separator="\n\n---\n\n",
                )
                logger.info(
                    "LlamaParse (llama-cloud-services) initialized with playground-like options."
                )
            except Exception as e:
                logger.warning(f"LlamaParse init failed: {e}")
                # Check if it's an async-related error
                if any(
                    keyword in str(e).lower()
                    for keyword in [
                        "asyncio",
                        "event loop",
                        "async",
                        "await",
                        "coroutine",
                        "nested",
                    ]
                ):
                    logger.info(
                        "Detected async-related error, will use isolated parsing mode"
                    )
                    self.use_isolated_parsing = False
                else:
                    self.available = False

    def _remove_cache(self, paths: Dict[str, Path]):
        for p in paths.values():
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    def clear_cache(self, files: Optional[List[str]] = None) -> int:
        removed = 0
        try:
            if files:
                for f in files:
                    paths = self._cache_paths(f)
                    for p in paths.values():
                        if p.exists():
                            try:
                                p.unlink()
                                removed += 1
                            except Exception:
                                pass
            else:
                for p in self.cache_dir.glob("*"):
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return removed

    def _file_hash(self, file_path: str) -> str:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _cache_paths(self, file_path: str) -> Dict[str, Path]:
        fh = self._file_hash(file_path)
        base = f"{Path(file_path).stem}_{fh[:12]}"
        return {
            "md": self.cache_dir / f"{base}.md",
            "json": self.cache_dir / f"{base}.json",
            "meta": self.cache_dir / f"{base}.metadata.json",
        }

    def _is_cache_valid(self, paths: Dict[str, Path], file_path: str) -> bool:
        try:
            if (
                not paths["md"].exists()
                or not paths["meta"].exists()
                or not paths["json"].exists()
            ):
                return False
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            return (
                meta.get("hash") == self._file_hash(file_path)
                and abs(meta.get("mtime", 0) - os.path.getmtime(file_path)) < 0.001
                and meta.get("original_path") == os.path.abspath(file_path)
            )
        except Exception:
            return False

    def _write_cache(
        self,
        paths: Dict[str, Path],
        md_text: str,
        json_obj: Dict[str, Any],
        file_path: str,
        total_pages: int,
    ):
        try:
            paths["md"].write_text(md_text, encoding="utf-8")
            paths["json"].write_text(
                json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            meta = {
                "original_path": os.path.abspath(file_path),
                "mtime": os.path.getmtime(file_path),
                "size": os.path.getsize(file_path),
                "hash": self._file_hash(file_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "md_path": str(paths["md"]),
                "json_path": str(paths["json"]),
                "pages": total_pages,
            }
            paths["meta"].write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to write parse cache for {file_path}: {e}")

    def _make_docs_from_pages(
        self,
        file_path: str,
        pages: List[Dict[str, Any]],
        total_pages: int,
    ) -> List[LlamaDocument]:
        """
        Create LlamaDocument objects from page dicts returned by LlamaParse.
        Ensures consistent id attributes and normalized metadata.
        PRESERVES HTML tables for proper extraction.
        """
        docs: List[LlamaDocument] = []
        src_name = Path(file_path).name

        for p in pages:
            page_num = p.get("page")
            md = p.get("md") or p.get("text") or ""
            if not isinstance(md, str) or len(md.strip()) < 1:
                continue

            # CRITICAL FIX: Only convert to plain text if NO tables are present
            try:
                # Check if this contains HTML tables first
                has_html_tables = "<table" in md.lower() and "</table>" in md.lower()

                if not has_html_tables and "<" in md and ">" in md:
                    # Only convert to plain text if no tables present
                    plain = BeautifulSoup(md, "html.parser").get_text("\n", strip=True)
                    if plain and len(plain.strip()) >= 1:
                        md = plain
                # If has_html_tables is True, preserve the HTML structure

            except Exception:
                pass

            if len(md.strip()) < 1:
                continue

            doc_id = (
                f"{Path(src_name).stem}_p{int(page_num):04d}"
                if page_num is not None
                else f"{Path(src_name).stem}_{uuid.uuid4().hex[:8]}"
            )

            metadata = {
                "source": src_name,
                "file_name": src_name,
                "file_path": file_path,
                "page": page_num,
                "total_pages": total_pages,
                "content_type": "text",
            }

            doc = LlamaDocument(
                text=md,  # Now preserves HTML table structure when needed
                doc_id=doc_id,
                metadata=metadata,
            )

            # Defensive normalization for different LlamaIndex versions
            try:
                setattr(doc, "id_", getattr(doc, "id_", None) or doc_id)
            except Exception:
                pass
            try:
                setattr(doc, "doc_id", getattr(doc, "doc_id", None) or doc_id)
            except Exception:
                pass

            docs.append(doc)

        return docs

    def _isolated_parse(self, file_path: str):
        """Fallback isolated parsing when direct parsing fails due to async issues"""
        safe_file_path = file_path.replace('"', '\\"').replace("'", "\\'")
        safe_api_key = self.api_key.replace('"', '\\"').replace("'", "\\'")
        safe_cwd = os.getcwd().replace('"', '\\"').replace("'", "\\'")

        temp_script_content = f'''import sys
        import os
        import json
        import traceback

        sys.path.insert(0, "{safe_cwd}")

        def isolated_parse():
            try:
                from llama_cloud_services import LlamaParse

                parser = LlamaParse(
                    api_key="{safe_api_key}",
                    num_workers=4,
                    verbose=False,
                    language="en",
                    parse_mode="parse_page_with_agent",
                    model="openai-gpt-4-1-mini",
                    high_res_ocr=True,
                    adaptive_long_table=True,
                    outlined_table_extraction=True,
                    output_tables_as_HTML=True,
                    page_separator="\\n\\n---\\n\\n",
                )

                result = parser.parse("{safe_file_path}")

                # Prefer using result.pages (contains md, text, structuredData)
                pages = []
                if hasattr(result, "pages") and result.pages:
                    for i, page in enumerate(result.pages, start=1):
                        md = getattr(page, "md", None) or getattr(page, "text", "") or ""
                        pages.append({{"page": i, "md": md}})
                else:
                    md_docs = result.get_markdown_documents(split_by_page=True)
                    if md_docs:
                        for i, d in enumerate(md_docs, start=1):
                            pages.append({{"page": i, "md": getattr(d, "text", "") or ""}})
                    else:
                        txt_docs = result.get_text_documents(split_by_page=True)
                        for i, d in enumerate(txt_docs, start=1):
                            pages.append({{"page": i, "md": getattr(d, "text", "") or ""}})

                total_pages = len(pages)
                return {{"success": True, "pages": pages, "total_pages": total_pages}}

            except Exception as e:
                return {{"success": False, "error": str(e), "traceback": traceback.format_exc()}}

        if __name__ == "__main__":
            res = isolated_parse()
            print(json.dumps(res, ensure_ascii=False))
'''

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(temp_script_content)
                temp_script_path = f.name

            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = os.getcwd()

                logger.debug(f"Using isolated parsing for {Path(file_path).name}")

                result = subprocess.run(
                    [sys.executable, temp_script_path],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env,
                    cwd=os.getcwd(),
                )

                if result.returncode != 0:
                    error_msg = f"Isolated parse failed with code {result.returncode}"
                    if result.stderr:
                        error_msg += f": {result.stderr}"
                    raise RuntimeError(error_msg)

                try:
                    parse_result = json.loads(result.stdout.strip())
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse JSON from isolated process: {result.stdout[:500]}..."
                    )
                    raise RuntimeError(f"JSON decode error: {e}")

                if not parse_result.get("success"):
                    error_msg = parse_result.get("error", "Unknown error")
                    if parse_result.get("traceback"):
                        logger.debug(
                            f"Isolated parse traceback: {parse_result['traceback']}"
                        )
                    raise RuntimeError(f"Isolated parse failed: {error_msg}")

                return parse_result

            finally:
                try:
                    os.unlink(temp_script_path)
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            logger.error(f"Isolated parse timed out for {Path(file_path).name}")
            raise RuntimeError(f"Parse timeout for {Path(file_path).name}")
        except Exception as e:
            logger.error(
                f"Isolated parse execution failed for {Path(file_path).name}: {str(e)}"
            )
            raise

    def parse_file(
        self, file_path: str, force_reparse: bool = False
    ) -> List[LlamaDocument]:
        if not self.available:
            raise RuntimeError("LlamaParse not available. Set LLAMA_CLOUD_API_KEY.")

        paths = self._cache_paths(file_path)

        # Serve from cache if valid and not forcing reparse
        if not force_reparse and self._is_cache_valid(paths, file_path):
            try:
                json_obj = json.loads(paths["json"].read_text(encoding="utf-8"))
                pages = json_obj.get("pages", [])
                total_pages = json_obj.get("total_pages", len(pages))
                docs = self._make_docs_from_pages(file_path, pages, total_pages)
                logger.info(
                    f"[Cache] {Path(file_path).name}: pages={len(pages)} -> docs={len(docs)}"
                )
                return docs
            except Exception as e:
                logger.debug(
                    f"Cache read failed for {file_path}, reparsing. Reason: {e}"
                )

        # If forcing, remove old cache before parsing
        if force_reparse:
            self._remove_cache(paths)

        # Try direct parsing first, fall back to isolated if needed
        try:
            if self.use_isolated_parsing:
                # Use isolated parsing directly
                parse_result = self._isolated_parse(file_path)
                pages = parse_result.get("pages", [])
                total_pages = parse_result.get("total_pages", len(pages))
            else:
                # Try direct parsing with the original approach
                result = self.parser.parse(file_path)
                md_docs = result.get_markdown_documents(split_by_page=True)
                total_pages = len(md_docs)

                pages: List[Dict[str, Any]] = []
                if total_pages == 0:
                    # Fallback to text documents
                    txt_docs = result.get_text_documents(split_by_page=True)
                    total_pages = len(txt_docs)
                    logger.warning(
                        f"[Parse] {Path(file_path).name}: markdown empty, fell back to text; pages={total_pages}"
                    )
                    for i, d in enumerate(txt_docs, start=1):
                        page_text = getattr(d, "text", "") or ""
                        pages.append({"page": i, "md": page_text})
                else:
                    for i, d in enumerate(md_docs, start=1):
                        page_md = getattr(d, "text", "") or ""
                        pages.append({"page": i, "md": page_md})

        except Exception as e:
            # If direct parsing fails and we haven't tried isolated yet, try isolated
            if not self.use_isolated_parsing:
                logger.warning(
                    f"Direct parsing failed for {file_path}: {e}. Trying isolated parsing..."
                )
                try:
                    parse_result = self._isolated_parse(file_path)
                    pages = parse_result.get("pages", [])
                    total_pages = parse_result.get("total_pages", len(pages))
                    # Set flag for future files
                    self.use_isolated_parsing = True
                except Exception as e2:
                    logger.error(
                        f"Both direct and isolated parsing failed for {file_path}: {e2}"
                    )
                    return []
            else:
                logger.error(f"Isolated parsing failed for {file_path}: {e}")
                return []

        # Process the results (same as original)
        usable_pages = [
            p
            for p in pages
            if isinstance(p.get("md"), str) and len((p.get("md") or "").strip()) >= 5
        ]
        if total_pages == 0 or len(usable_pages) == 0:
            logger.warning(
                f"[Parse] {Path(file_path).name}: empty parse (pages={total_pages}). Not caching."
            )
            return []

        combined_md = "\n\n---\n".join(
            [f"<!-- page {p['page']}/{total_pages} -->\n{p['md']}" for p in pages]
        )
        json_obj = {"pages": pages, "total_pages": total_pages}
        self._write_cache(paths, combined_md, json_obj, file_path, total_pages)

        docs = self._make_docs_from_pages(file_path, pages, total_pages)
        short_pages = sum(1 for p in pages if len((p.get("md") or "").strip()) < 5)
        logger.info(
            f"[Parse] {Path(file_path).name}: pages={total_pages}, docs(>=5ch)={len(docs)}, short_pages(<5ch)={short_pages}"
        )
        return docs

    def parse_files(
        self, files: List[str], force_reparse: bool = False
    ) -> List[LlamaDocument]:
        all_docs: List[LlamaDocument] = []
        for f in files:
            all_docs.extend(self.parse_file(f, force_reparse=force_reparse))
        return all_docs


# Multi-format parser
class MultiFormatParser:
    def __init__(self):
        self.llamacloud = LlamaCloudParseHelper(LLAMA_CLOUD_API_KEY)
        self.supported_exts = {
            ".pdf",
            ".docx",
            ".pptx",
            ".epub",
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".tif",
            ".bmp",
            ".gif",
            ".html",
            ".htm",
            ".md",
            ".txt",
        }

    def _detect_tables_in_content(self, text_content: str) -> bool:
        """Enhanced table detection"""
        # Check for HTML tables
        if "<table" in text_content.lower() and "</table>" in text_content.lower():
            return True

        # Check for markdown tables (more robust detection)
        lines = text_content.split("\n")
        pipe_lines = [line for line in lines if "|" in line and line.count("|") >= 2]

        if len(pipe_lines) >= 3:  # At least header + separator + 1 data row
            # Check if we have a separator line (markdown table format)
            has_separator = any(
                set(line.strip()) <= {"-", "|", ":", " "} for line in pipe_lines[1:3]
            )
            if has_separator:
                return True

            # Check for consistent column count
            column_counts = [line.count("|") for line in pipe_lines[:5]]
            if len(set(column_counts)) <= 2:  # Consistent column structure
                return True

        return False

    def _extract_and_structure_tables(
        self, page_text: str, source_name: str, page_num: int
    ) -> List[LlamaDocument]:
        """
        ENHANCED: Extract tables with better context preservation and precision.
        Captures preceding context (titles, notes) and creates multiple document types.
        """
        docs = []

        # Step 1: Extract context paragraphs BEFORE tables
        # This captures critical notes like "Gaming Router with Fiber Extra will be discounted"
        preceding_context = self._extract_table_context(page_text)

        # STRATEGY 1: HTML Table Parsing (Preferred)
        if "<table" in (page_text or "").lower():
            try:
                soup = BeautifulSoup(page_text, "html.parser")
                tables = soup.find_all("table")
                logger.info(f"Found {len(tables)} HTML tables in {source_name}")

                for table_idx, table in enumerate(tables, 1):
                    # Get context for THIS specific table
                    table_context = preceding_context.get(table_idx, "")

                    # Extract headers with normalization
                    headers = self._extract_table_headers(table)

                    # Extract data rows
                    tbody = table.find("tbody") or table
                    data_rows = []
                    for row in tbody.find_all("tr"):
                        cells = [
                            c.get_text(separator=" ", strip=True)
                            for c in row.find_all(["td", "th"])
                        ]
                        if cells and cells != headers and any(c.strip() for c in cells):
                            data_rows.append(cells)

                    if not data_rows:
                        continue

                    logger.info(
                        f"Table {table_idx}: {len(headers)} headers, {len(data_rows)} data rows"
                    )

                    # APPROACH A: Complete Table Document with Context
                    if headers and len(data_rows) > 0:
                        full_table_text = self._build_complete_table_document(
                            table_idx,
                            source_name,
                            page_num,
                            headers,
                            data_rows,
                            table_context,
                        )

                        # Create document with enhanced metadata
                        doc_id = f"{Path(source_name).stem}_p{page_num:04d}_t{table_idx}_complete"
                        complete_doc = LlamaDocument(
                            text=full_table_text,
                            doc_id=doc_id,
                            metadata={
                                "source": source_name,
                                "file_name": source_name,
                                "page": page_num,
                                "content_type": "table_complete",
                                "table_number": table_idx,
                                "row_count": len(data_rows),
                                "column_count": len(headers),
                                "headers": headers,
                                "table_context": table_context,  # NEW: Store context
                                # NEW: Extract searchable terms
                                "searchable_values": self._extract_searchable_values(
                                    data_rows, headers
                                ),
                                "has_pricing": self._contains_pricing(data_rows),
                                "has_codes": self._contains_codes(data_rows),
                            },
                        )

                        try:
                            setattr(complete_doc, "id_", doc_id)
                        except Exception:
                            pass

                        docs.append(complete_doc)

                    # APPROACH B: Individual Row Documents (Enhanced)
                    if headers and len(data_rows) > 0:
                        for row_idx, row_cells in enumerate(data_rows, 1):
                            # Normalize cell count
                            if len(row_cells) < len(headers):
                                row_cells += [""] * (len(headers) - len(row_cells))
                            elif len(row_cells) > len(headers):
                                row_cells = row_cells[: len(headers)]

                            # Build enriched row text WITH CONTEXT
                            row_text = self._build_enriched_row_document(
                                table_idx,
                                row_idx,
                                source_name,
                                page_num,
                                headers,
                                row_cells,
                                table_context,
                            )

                            if len(row_text.strip()) > 50:
                                doc_id = f"{Path(source_name).stem}_p{page_num:04d}_t{table_idx}_r{row_idx}"

                                # Extract key-value pairs for better search
                                kv_pairs = {
                                    h: v
                                    for h, v in zip(headers, row_cells)
                                    if v and v.strip()
                                }

                                row_doc = LlamaDocument(
                                    text=row_text,
                                    doc_id=doc_id,
                                    metadata={
                                        "source": source_name,
                                        "file_name": source_name,
                                        "page": page_num,
                                        "content_type": "table_row",
                                        "table_number": table_idx,
                                        "row_number": row_idx,
                                        "header_cells": headers,
                                        "raw_cells": row_cells,
                                        "table_context": table_context,
                                        "key_values": kv_pairs,
                                        "device_name": self._extract_field(
                                            kv_pairs, ["device", "description", "name"]
                                        ),
                                        "price": self._extract_field(
                                            kv_pairs,
                                            ["price", "rental", "amount", "bd", "cost"],
                                        ),
                                        "code": self._extract_field(
                                            kv_pairs, ["code", "flag", "id"]
                                        ),
                                        "speed": self._extract_field(
                                            kv_pairs,
                                            ["speed", "download", "upload", "mbps"],
                                        ),
                                    },
                                )

                                try:
                                    setattr(row_doc, "id_", doc_id)
                                except Exception:
                                    pass

                                docs.append(row_doc)

                    # APPROACH C: Create "Table Summary" document for better retrieval
                    if headers and data_rows:
                        summary_doc = self._create_table_summary_document(
                            table_idx,
                            source_name,
                            page_num,
                            headers,
                            data_rows,
                            table_context,
                        )
                        if summary_doc:
                            docs.append(summary_doc)

                # Keep surrounding non-table text if substantial
                text_without_tables = soup.get_text("\n", strip=True)
                if text_without_tables and len(text_without_tables.strip()) > 100:
                    doc_id = f"{Path(source_name).stem}_p{page_num:04d}_text"
                    text_doc = LlamaDocument(
                        text=text_without_tables,
                        doc_id=doc_id,
                        metadata={
                            "source": source_name,
                            "file_name": source_name,
                            "page": page_num,
                            "content_type": "text",
                        },
                    )

                    try:
                        setattr(text_doc, "id_", doc_id)
                    except Exception:
                        pass

                    docs.append(text_doc)

                return docs

            except Exception as e:
                logger.debug(f"HTML table parsing failed: {e}")

        # STRATEGY 2: Markdown parsing
        return self._extract_markdown_tables(page_text, source_name, page_num)

    def _extract_markdown_tables(
        self, page_text: str, source_name: str, page_num: int
    ) -> List[LlamaDocument]:
        """Extract markdown-style tables"""
        docs = []
        lines = page_text.split("\n")

        # Find table blocks
        table_blocks = []
        current_table = []

        for line in lines:
            if "|" in line and line.count("|") >= 2:
                current_table.append(line.strip())
            else:
                if len(current_table) >= 3:  # At least header + separator + data
                    table_blocks.append(current_table)
                current_table = []

        # Process last table if exists
        if len(current_table) >= 3:
            table_blocks.append(current_table)

        for table_idx, table_lines in enumerate(table_blocks, 1):
            if len(table_lines) < 3:
                continue

            # Parse headers
            header_line = table_lines[0]
            headers = [cell.strip() for cell in header_line.split("|") if cell.strip()]

            # Parse data rows (skip separator line)
            data_rows = []
            for line in table_lines[2:]:  # Skip header and separator
                cells = [cell.strip() for cell in line.split("|") if cell.strip()]
                if cells:
                    data_rows.append(cells)

            if not headers or not data_rows:
                continue

            # Create documents similar to HTML table processing
            for row_idx, row_cells in enumerate(data_rows, 1):
                # Normalize cell count
                if len(row_cells) < len(headers):
                    row_cells += [""] * (len(headers) - len(row_cells))
                elif len(row_cells) > len(headers):
                    row_cells = row_cells[: len(headers)]

                row_text = self._build_enriched_row_document(
                    table_idx,
                    row_idx,
                    source_name,
                    page_num,
                    headers,
                    row_cells,
                    "",  # No context for markdown tables
                )

                if len(row_text.strip()) > 50:
                    doc_id = f"{Path(source_name).stem}_p{page_num:04d}_t{table_idx}_r{row_idx}_md"
                    kv_pairs = {
                        h: v for h, v in zip(headers, row_cells) if v and v.strip()
                    }

                    row_doc = LlamaDocument(
                        text=row_text,
                        doc_id=doc_id,
                        metadata={
                            "source": source_name,
                            "file_name": source_name,
                            "page": page_num,
                            "content_type": "table_row",
                            "table_number": table_idx,
                            "row_number": row_idx,
                            "header_cells": headers,
                            "raw_cells": row_cells,
                            "key_values": kv_pairs,
                            "device_name": self._extract_field(
                                kv_pairs, ["device", "description", "name", "package"]
                            ),
                            "price": self._extract_field(
                                kv_pairs,
                                ["price", "rental", "amount", "bd", "cost", "monthly"],
                            ),
                            "code": self._extract_field(
                                kv_pairs, ["code", "flag", "id"]
                            ),
                        },
                    )

                    try:
                        setattr(row_doc, "id_", doc_id)
                    except Exception:
                        pass

                    docs.append(row_doc)

        return docs

    def _extract_table_context(self, page_text: str) -> Dict[int, str]:
        """
        NEW METHOD: Extract context paragraphs that appear before tables.
        Returns dict mapping table_index -> context_text
        """
        contexts = {}

        try:
            soup = BeautifulSoup(page_text, "html.parser")
            tables = soup.find_all("table")

            for idx, table in enumerate(tables, 1):
                context_parts = []

                # Look at previous siblings (paragraphs, text nodes, etc.)
                for sibling in table.find_all_previous_siblings(limit=5):
                    if sibling.name in ["p", "div", "span"]:
                        text = sibling.get_text(strip=True)
                        if text and len(text) > 10:
                            context_parts.insert(0, text)
                    elif isinstance(sibling, str):
                        text = sibling.strip()
                        if text and len(text) > 10:
                            context_parts.insert(0, text)

                # Also check for bold/italic text right before table
                prev_element = table.find_previous()
                if prev_element and prev_element.name in ["strong", "b", "em", "i"]:
                    context_parts.append(prev_element.get_text(strip=True))

                if context_parts:
                    contexts[idx] = " | ".join(
                        context_parts[:3]
                    )  # Limit to 3 most relevant

        except Exception as e:
            logger.debug(f"Context extraction failed: {e}")

        return contexts

    def _extract_table_headers(self, table) -> List[str]:
        """
        NEW METHOD: Enhanced header extraction with normalization.
        """
        headers = []

        # Try thead first
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                headers = [
                    th.get_text(strip=True) for th in header_row.find_all(["th", "td"])
                ]

        # If no thead, try first row with th elements
        if not headers:
            first_row = table.find("tr")
            if first_row and first_row.find_all("th"):
                headers = [
                    th.get_text(strip=True) for th in first_row.find_all(["th", "td"])
                ]

        # Normalize headers (remove special chars, lowercase for matching)
        headers = [h.strip().replace("\n", " ").replace("  ", " ") for h in headers]

        return headers

    def _build_complete_table_document(
        self,
        table_idx: int,
        source_name: str,
        page_num: int,
        headers: List[str],
        data_rows: List[List[str]],
        context: str,
    ) -> str:
        """
        NEW METHOD: Build a complete table document with rich context.
        """
        doc_parts = []

        # Add context if available
        if context:
            doc_parts.append(f"# Context: {context}\n")

        doc_parts.append(f"# Table {table_idx} from {source_name} (Page {page_num})\n")
        doc_parts.append(f"**Column Headers:** {' | '.join(headers)}\n")

        # Add structured data representation
        doc_parts.append("\n**Complete Table Data:**\n")

        for row_idx, row_cells in enumerate(data_rows, 1):
            if len(row_cells) < len(headers):
                row_cells += [""] * (len(headers) - len(row_cells))
            elif len(row_cells) > len(headers):
                row_cells = row_cells[: len(headers)]

            doc_parts.append(f"\n**Entry {row_idx}:**")
            for header, value in zip(headers, row_cells):
                if value and value.strip():
                    # Highlight important fields
                    if any(
                        key in header.lower()
                        for key in ["price", "amount", "bd", "cost", "rental"]
                    ):
                        doc_parts.append(f"  • **{header}**: **{value}** (PRICE)")
                    elif any(key in header.lower() for key in ["code", "flag", "id"]):
                        doc_parts.append(f"  • **{header}**: **{value}** (CODE)")
                    else:
                        doc_parts.append(f"  • {header}: {value}")

        return "\n".join(doc_parts)

    def _build_enriched_row_document(
        self,
        table_idx: int,
        row_idx: int,
        source_name: str,
        page_num: int,
        headers: List[str],
        row_cells: List[str],
        context: str,
    ) -> str:
        """
        NEW METHOD: Build enriched row document with context.
        """
        parts = []

        # Add context first
        if context:
            parts.append(f"Context: {context}\n")

        parts.append(
            f"Table {table_idx}, Entry {row_idx} from {source_name} (Page {page_num}):\n"
        )

        # Add key-value pairs with emphasis on important fields
        for header, value in zip(headers, row_cells):
            if value and value.strip():
                # Identify field type for better matching
                if any(
                    key in header.lower()
                    for key in ["price", "amount", "bd", "cost", "rental", "fee"]
                ):
                    parts.append(f"**PRICE - {header}**: {value}")
                elif any(
                    key in header.lower() for key in ["code", "flag", "id", "ref"]
                ):
                    parts.append(f"**CODE - {header}**: {value}")
                elif any(
                    key in header.lower()
                    for key in ["device", "description", "name", "package", "plan"]
                ):
                    parts.append(f"**NAME - {header}**: {value}")
                elif any(
                    key in header.lower()
                    for key in [
                        "speed",
                        "download",
                        "upload",
                        "mbps",
                        "allowance",
                        "data",
                    ]
                ):
                    parts.append(f"**SPEC - {header}**: {value}")
                else:
                    parts.append(f"{header}: {value}")

        return "\n".join(parts)

    def _extract_searchable_values(
        self, data_rows: List[List[str]], headers: List[str]
    ) -> List[str]:
        """
        NEW METHOD: Extract all unique searchable values from table.
        """
        values = set()

        for row in data_rows:
            for cell in row:
                if cell and cell.strip():
                    # Extract individual words and full phrases
                    values.add(cell.strip().lower())
                    # Also add individual words
                    values.update(cell.strip().lower().split())

        return list(values)

    def _contains_pricing(self, data_rows: List[List[str]]) -> bool:
        """NEW METHOD: Check if table contains pricing information."""
        for row in data_rows:
            for cell in row:
                if cell and (
                    "bd" in cell.lower()
                    or "price" in cell.lower()
                    or "cost" in cell.lower()
                ):
                    return True
        return False

    def _contains_codes(self, data_rows: List[List[str]]) -> bool:
        """NEW METHOD: Check if table contains code/ID information."""
        for row in data_rows:
            for cell in row:
                if cell and any(
                    indicator in cell.lower()
                    for indicator in ["flg", "code", "inst", "id"]
                ):
                    return True
        return False

    def _extract_field(self, kv_pairs: Dict[str, str], keywords: List[str]) -> str:
        """
        NEW METHOD: Extract specific field value by keywords.
        """
        for key, value in kv_pairs.items():
            if any(kw in key.lower() for kw in keywords):
                return value
        return ""

    def _create_table_summary_document(
        self,
        table_idx: int,
        source_name: str,
        page_num: int,
        headers: List[str],
        data_rows: List[List[str]],
        context: str,
    ) -> Optional[LlamaDocument]:
        """
        NEW METHOD: Create a summary document optimized for search.
        This helps when users ask general questions about what's in the table.
        """
        try:
            summary_parts = []

            if context:
                summary_parts.append(f"Table Context: {context}")

            summary_parts.append(f"Table {table_idx} Summary from {source_name}")
            summary_parts.append(
                f"Contains {len(data_rows)} entries with columns: {', '.join(headers)}"
            )

            # Extract key information types
            devices = []
            prices = []
            codes = []

            for row in data_rows:
                for i, cell in enumerate(row):
                    if i < len(headers) and cell and cell.strip():
                        header = headers[i].lower()
                        if any(
                            kw in header
                            for kw in [
                                "device",
                                "description",
                                "name",
                                "package",
                                "plan",
                            ]
                        ):
                            devices.append(cell.strip())
                        elif any(
                            kw in header for kw in ["price", "amount", "bd", "rental"]
                        ):
                            prices.append(cell.strip())
                        elif any(kw in header for kw in ["code", "flag", "id"]):
                            codes.append(cell.strip())

            if devices:
                summary_parts.append(f"Devices/Items: {', '.join(set(devices[:5]))}")
            if prices:
                summary_parts.append(f"Prices: {', '.join(set(prices[:5]))}")
            if codes:
                summary_parts.append(f"Codes: {', '.join(set(codes[:5]))}")

            summary_text = "\n".join(summary_parts)

            doc_id = f"{Path(source_name).stem}_p{page_num:04d}_t{table_idx}_summary"

            return LlamaDocument(
                text=summary_text,
                doc_id=doc_id,
                metadata={
                    "source": source_name,
                    "file_name": source_name,
                    "page": page_num,
                    "content_type": "table_summary",
                    "table_number": table_idx,
                    "row_count": len(data_rows),
                    "column_count": len(headers),
                    "headers": headers,
                },
            )

        except Exception as e:
            logger.debug(f"Summary creation failed: {e}")
            return None

    def clear_parsed_cache(self, files: Optional[List[str]] = None) -> int:
        if not (self.llamacloud and self.llamacloud.available):
            return 0
        return self.llamacloud.clear_cache(files)

    def _try_structured_data_variants(self, file_path: str, ext: str) -> List[str]:
        """
        Convert structured files to readable text blocks for indexing (pandas-based).
        Returns a list of text blocks (one per sheet/chunk). If pandas isn't available or parsing fails
        this returns an empty list.
        """
        try:
            import pandas as pd  # local import to avoid mandatory dependency
        except Exception:
            logger.warning(
                "pandas not installed; cannot parse structured data (CSV/XLSX). "
                "Run: pip install pandas openpyxl tabulate"
            )
            return []

        def df_to_text(
            df, title: str = "", max_rows: int = 1000, max_cols: int = 40
        ) -> Optional[str]:
            if df is None or df.empty:
                return None
            try:
                if df.shape[1] > max_cols:
                    df = df.iloc[:, :max_cols]
                if df.shape[0] > max_rows:
                    df = df.head(max_rows)
            except Exception:
                # defensive: some objects might not behave exactly like DataFrame
                pass

            try:
                md = df.to_markdown(index=False)
                return f"# {title}\n{md}" if title else md
            except Exception:
                try:
                    txt = df.to_csv(index=False)
                    return f"# {title}\n{txt}" if title else txt
                except Exception:
                    return None

        try:
            ext = (ext or Path(file_path).suffix).lower()

            if ext == ".csv":
                tried = []
                for encoding in ("utf-8", "latin1", "cp1252"):
                    for engine in ("python", "c"):
                        try:
                            df = pd.read_csv(
                                file_path,
                                on_bad_lines="skip",
                                low_memory=False,
                                encoding=encoding,
                                engine=engine,
                            )
                            t = df_to_text(df, title=Path(file_path).name)
                            if t:
                                return [t]
                        except Exception as e:
                            tried.append((encoding, engine, str(e)))
                            continue
                logger.debug(f"CSV parse attempts failed for {file_path}: {tried[:3]}")
                return []

            if ext in (".xlsx", ".xls"):
                texts: List[str] = []
                try:
                    sheets = pd.read_excel(file_path, sheet_name=None)
                except Exception:
                    # fallback: let pandas choose engine
                    try:
                        sheets = pd.read_excel(file_path, sheet_name=None, engine=None)
                    except Exception as e2:
                        logger.error(f"Excel read failed for {file_path}: {e2}")
                        return []

                for sheet_name, df in sheets.items():
                    t = df_to_text(
                        df, title=f"{Path(file_path).name} — Sheet: {sheet_name}"
                    )
                    if t:
                        texts.append(t)
                return texts

            if ext == ".json":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        data = data[:200]
                    txt = json.dumps(data, ensure_ascii=False, indent=2)
                    return [txt[:200_000]]
                except Exception as e:
                    logger.error(f"JSON parse failed for {file_path}: {e}")
                    return []

            if ext == ".jsonl":
                lines = []
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i >= 500:
                                break
                            s = line.strip()
                            if s:
                                lines.append(s)
                    return ["\n".join(lines)] if lines else []
                except Exception as e:
                    logger.error(f"JSONL parse failed for {file_path}: {e}")
                    return []

            return []
        except Exception as e:
            logger.error(f"Structured parsing failed for {file_path}: {e}")
            return []

    def _use_llamaparse_for_ext(self, ext: str) -> bool:
        ext = ext.lower()
        structured = {".csv", ".xlsx", ".xls", ".json", ".jsonl"}
        return ext in self.supported_exts or ext in structured

    def should_skip_file(self, file_path: str) -> bool:
        file_name = Path(file_path).name.lower()
        skip_patterns = [
            "fingerprint.json",
            ".ds_store",
            "thumbs.db",
            "desktop.ini",
            "__pycache__",
            ".git",
            ".svn",
            ".tmp",
            "~$",
            ".lock",
            ".cache",
            ".temp",
        ]
        if file_name.startswith("."):
            return True
        try:
            cache_dir = Path(PARSED_CACHE_DIR).resolve()
            if cache_dir in Path(file_path).resolve().parents:
                return True
        except Exception:
            pass
        return any(pattern in file_name for pattern in skip_patterns)

    def get_file_stats(self, directory_path: str) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for root, _, files in os.walk(directory_path):
            for file in files:
                p = os.path.join(root, file)
                if self.should_skip_file(p):
                    continue
                ext = Path(p).suffix.lower()
                if ext:
                    stats[ext] = stats.get(ext, 0) + 1
        return stats

    def parse_files_with_llamaparse(
        self, files: List[str], force_reparse: bool = False
    ) -> List[LlamaDocument]:
        if not (self.llamacloud and self.llamacloud.available):
            raise RuntimeError(
                "LlamaParse is required but LLAMA_CLOUD_API_KEY is missing or invalid."
            )

        use_files = [
            f
            for f in files
            if not self.should_skip_file(f)
            and self._use_llamaparse_for_ext(Path(f).suffix)
        ]
        if not use_files:
            return []

        docs = self.llamacloud.parse_files(use_files, force_reparse=force_reparse)
        if len(docs) == 0:
            logger.warning(
                "LlamaParse returned no documents. Check API key/quota and file types. "
                f"Examples: {', '.join(Path(f).name for f in use_files[:3])}..."
            )
        logger.info(
            f"LlamaParse produced {len(docs)} page-level documents from {len(use_files)} files"
        )
        return docs

    def load_cached_markdown_docs(self, files: List[str]) -> List[LlamaDocument]:
        docs: List[LlamaDocument] = []
        for f in files:
            try:
                paths = self.llamacloud._cache_paths(f)
                md_path = paths["md"]
                if not md_path.exists():
                    continue
                md_text = md_path.read_text(encoding="utf-8")
                parts = re.split(r"\n-{3,}\n", md_text)
                total_pages = len(parts) if parts else 0
                for i, part in enumerate(parts, start=1):
                    page_md = part.strip()
                    if len(page_md) < 5:
                        continue

                    # strip HTML to plain text if present
                    try:
                        plain = BeautifulSoup(page_md, "html.parser").get_text(
                            "\n", strip=True
                        )
                        if plain and len(plain.strip()) > 0:
                            page_md = plain
                    except Exception:
                        pass

                    doc_id = f"{Path(f).stem}_p{i:04d}"
                    src_name = Path(f).name
                    doc = LlamaDocument(
                        text=page_md,
                        doc_id=doc_id,
                        metadata={
                            "source": src_name,
                            "file_name": src_name,
                            "file_path": f,
                            "page": i,
                            "total_pages": total_pages,
                            "content_type": "text",
                        },
                    )
                    # normalize version attributes
                    try:
                        setattr(doc, "id_", getattr(doc, "id_", None) or doc_id)
                    except Exception:
                        pass
                    try:
                        setattr(doc, "doc_id", getattr(doc, "doc_id", None) or doc_id)
                    except Exception:
                        pass

                    docs.append(doc)

            except Exception as e:
                logger.warning(f"Failed loading cached md for {f}: {e}")
                continue
        return docs


# Onedrive data
class OneDriveSharedFolderClient:
    def __init__(self):
        self.base_url = "https://graph.microsoft.com/v1.0"
        self.access_token = self._get_access_token()
        self.headers = {
            "Authorization": f"Bearer {self.access_token}" if self.access_token else "",
            "Content-Type": "application/json",
        }

    def _get_access_token(self):
        if not all([MS_TENANT_ID, MS_CLIENT_ID, MS_CLIENT_SECRET]):
            print("WARNING: Missing Microsoft Graph API credentials")
            return None

        url = f"https://login.microsoftonline.com/{MS_TENANT_ID}/oauth2/v2.0/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": MS_CLIENT_ID,
            "client_secret": MS_CLIENT_SECRET,
            "scope": "https://graph.microsoft.com/.default",
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            response = requests.post(url, data=data, headers=headers, timeout=30)
            response.raise_for_status()
            print("✅ Microsoft Graph Authentication successful")
            return response.json().get("access_token")
        except Exception as e:
            print(f"⚠️ Microsoft Graph Auth Error: {e}")
            return None

    def _encode_sharing_url(self, url: str) -> str:
        if url.startswith("u!"):
            return url
        encoded = urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").rstrip("=")
        return f"u!{encoded}"

    def download_shared_folder(
        self, local_path="data", shared_folder_url=None
    ) -> List[str]:
        if not self.headers.get("Authorization") or not shared_folder_url:
            return []
        os.makedirs(local_path, exist_ok=True)
        encoded_url = self._encode_sharing_url(shared_folder_url)
        drive_item_url = f"{self.base_url}/shares/{encoded_url}/driveItem/children"
        try:
            response = requests.get(drive_item_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            items = response.json().get("value", [])
            downloaded_files = []

            for item in items:
                if "file" in item:
                    file_path = os.path.join(local_path, item["name"])
                    download_url = item.get("@microsoft.graph.downloadUrl")
                    if download_url:
                        with requests.get(download_url, stream=True, timeout=30) as r:
                            r.raise_for_status()
                            with open(file_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                        downloaded_files.append(file_path)
                elif "folder" in item:
                    subfolder_path = os.path.join(local_path, item["name"])
                    os.makedirs(subfolder_path, exist_ok=True)
            return downloaded_files
        except Exception as e:
            print(f"OneDrive download error: {e}")
            return []

    def check_onedrive_connection(self, shared_folder_url=None):
        if not self.headers:
            return {
                "success": False,
                "message": "Not authenticated with Microsoft Graph API",
            }
        if not shared_folder_url:
            return {"success": False, "message": "No shared folder URL provided"}
        try:
            encoded_url = self._encode_sharing_url(shared_folder_url)
            url = f"{self.base_url}/shares/{encoded_url}/driveItem"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            item_data = response.json()
            item_name = item_data.get("name", "Unknown item")
            permissions_url = (
                f"{self.base_url}/shares/{encoded_url}/driveItem/permissions"
            )
            perm_response = requests.get(
                permissions_url, headers=self.headers, timeout=10
            )
            if perm_response.status_code == 200:
                permissions = perm_response.json().get("value", [])
                readable = any(
                    p.get("roles", [])
                    and ("read" in p.get("roles", []) or "write" in p.get("roles", []))
                    for p in permissions
                )
                if readable:
                    return {
                        "success": True,
                        "message": f"Successfully connected to '{item_name}'",
                        "item_type": "folder" if item_data.get("folder") else "file",
                        "permissions": "read/write access confirmed",
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Connected to '{item_name}' but missing read permissions",
                    }
            return {
                "success": True,
                "message": f"Connected to '{item_name}', but couldn't verify permissions",
            }
        except requests.HTTPError as e:
            error_msg = f"Connection failed (HTTP {e.response.status_code})"
            if e.response.status_code == 404:
                error_msg = "The shared link could not be found - verify the URL is correct and shared properly"
            elif e.response.status_code == 401:
                error_msg = "Authentication failed - check credentials"
            elif e.response.status_code == 403:
                error_msg = "Permission denied - ensure proper sharing settings"
            return {"success": False, "message": error_msg}
        except Exception as e:
            return {"success": False, "message": f"Connection error: {str(e)}"}


# Document Manager
class DocumentManager:
    def __init__(self):
        self.onedrive_client = OneDriveSharedFolderClient()
        self.index = None
        self.query_engine = None
        self.vector_store = None
        self.parser = MultiFormatParser()
        self.qdrant = None
        self._raw_docs_text: List[Dict[str, Any]] = []
        self.fingerprint_path = str(Path(DATA_FINGERPRINT_PATH).resolve())
        self._setup_vector_store()

    def _deduplicate_table_rows(
        self, documents: List[LlamaDocument]
    ) -> List[LlamaDocument]:
        """Deduplicate identical table rows across different pages"""

        unique_docs = []
        seen_table_content = {}  # content_hash -> first_occurrence

        for doc in documents:
            try:
                meta = getattr(doc, "metadata", {}) or {}
                content_type = meta.get("content_type", "")

                # Only deduplicate table rows
                if content_type == "table_row":
                    # Create content signature from the actual data
                    key_values = meta.get("key_values", {})
                    header_cells = meta.get("header_cells", [])
                    raw_cells = meta.get("raw_cells", [])

                    # Create a content hash from the actual data (not page info)
                    content_signature = {
                        "headers": tuple(sorted(header_cells)) if header_cells else (),
                        "cells": tuple(sorted(raw_cells)) if raw_cells else (),
                        "key_values": tuple(sorted(key_values.items()))
                        if key_values
                        else (),
                    }

                    content_hash = hashlib.md5(
                        json.dumps(content_signature, sort_keys=True).encode()
                    ).hexdigest()

                    if content_hash in seen_table_content:
                        # This is a duplicate - skip it
                        first_occurrence = seen_table_content[content_hash]
                        logger.debug(
                            f"Skipping duplicate table row: {meta.get('source')} page {meta.get('page')} "
                            f"(duplicate of page {first_occurrence.get('page')})"
                        )
                        continue
                    else:
                        # First occurrence - keep it
                        seen_table_content[content_hash] = meta
                        unique_docs.append(doc)
                else:
                    # Not a table row - keep all non-table content
                    unique_docs.append(doc)

            except Exception as e:
                logger.warning(f"Error in table deduplication: {e}")
                # If error, keep the document to be safe
                unique_docs.append(doc)

        original_count = len(documents)
        final_count = len(unique_docs)
        duplicates_removed = original_count - final_count

        if duplicates_removed > 0:
            logger.info(
                f"Removed {duplicates_removed} duplicate table rows ({original_count} -> {final_count})"
            )

        return unique_docs

    def _collect_source_files(self, root_dir: str) -> List[str]:
        """Recursively collect supported source files under root_dir, skipping system files."""
        if not os.path.exists(root_dir):
            return []
        supported = set(self.parser.supported_exts) | {
            ".xlsx",
            ".xls",
            ".csv",
            ".json",
            ".jsonl",
        }
        files: List[str] = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    if self.parser.should_skip_file(fp):
                        continue
                    ext = Path(fp).suffix.lower()
                    if ext in supported and os.path.isfile(fp):
                        files.append(fp)
                except Exception:
                    continue
        return files

    def _setup_vector_store(self):
        """Enhanced vector store setup with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.qdrant = QdrantClient(url=QDRANT_HOST, timeout=30)
                # Test connection
                existing = [c.name for c in self.qdrant.get_collections().collections]
                logger.info("Connected to Qdrant successfully")
                break
            except Exception as e:
                logger.warning(f"Qdrant connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to connect to Qdrant after {max_retries} attempts"
                    )
                time.sleep(5)

        try:
            emb_vec = Settings.embed_model.get_text_embedding("probe")
            emb_size = len(emb_vec)
        except Exception as e:
            logger.warning(
                f"Could not probe embedding model: {e}. Defaulting emb_size to 384"
            )
            emb_size = 384

        existing = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in existing:
            logger.info(
                f"[Qdrant] Creating collection '{COLLECTION_NAME}' with size {emb_size}"
            )
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=emb_size, distance=qmodels.Distance.COSINE
                ),
                hnsw_config=qmodels.HnswConfigDiff(
                    m=16, ef_construct=200, max_indexing_threads=0
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    default_segment_number=2
                ),
            )
        else:
            logger.info(f"[Qdrant] Using existing collection: {COLLECTION_NAME}")

        self.vector_store = QdrantVectorStore(
            client=self.qdrant, collection_name=COLLECTION_NAME
        )

    def _collection_exists_and_has_data(self) -> bool:
        """Check if Qdrant collection exists and has meaningful data"""
        try:
            existing = [c.name for c in self.qdrant.get_collections().collections]
            if COLLECTION_NAME not in existing:
                return False

            collection_info = self.qdrant.get_collection(COLLECTION_NAME)
            point_count = collection_info.points_count

            # Collection should have at least some points to be considered valid
            has_data = point_count > 0
            logger.info(
                f"Collection {COLLECTION_NAME} exists with {point_count} points"
            )
            return has_data

        except Exception as e:
            logger.warning(f"Error checking collection status: {e}")
            return False

    def _has_cached_parsed_files(self, files: List[str]) -> bool:
        """Check if we have cached parsed files for all the source files"""
        if not files:
            return False

        cached_count = 0
        for f in files:
            try:
                paths = self.parser.llamacloud._cache_paths(f)
                if (
                    paths["md"].exists()
                    and paths["json"].exists()
                    and paths["meta"].exists()
                    and self.parser.llamacloud._is_cache_valid(paths, f)
                ):
                    cached_count += 1
            except Exception:
                continue

        coverage = cached_count / len(files) if files else 0
        logger.info(
            f"Parsed cache coverage: {cached_count}/{len(files)} files ({coverage:.1%})"
        )

        # Consider cache sufficient if we have 80% or more files cached
        return coverage >= 0.8

    def _load_from_parsed_cache(self, files: List[str]) -> List[LlamaDocument]:
        """Load documents from cached parsed files with table extraction"""
        logger.info("Loading documents from parsed cache with table processing...")

        all_docs = []
        for f in files:
            try:
                paths = self.parser.llamacloud._cache_paths(f)
                md_path = paths["md"]
                if not md_path.exists():
                    continue
                md_text = md_path.read_text(encoding="utf-8")
                parts = re.split(r"\n-{3,}\n", md_text)
                total_pages = len(parts) if parts else 0

                for i, part in enumerate(parts, start=1):
                    page_md = part.strip()
                    if len(page_md) < 5:
                        continue

                    src_name = Path(f).name

                    # Check if this page contains tables and extract them
                    has_tables = "|" in page_md and page_md.count("|") > 4

                    if has_tables:
                        # Extract tables as separate structured documents
                        table_docs = self.parser._extract_and_structure_tables(
                            page_md, src_name, i
                        )
                        if table_docs:
                            all_docs.extend(table_docs)
                            continue  # Skip creating regular doc for this page

                    # Create regular document for non-table content
                    doc_id = f"{Path(f).stem}_p{i:04d}"
                    all_docs.append(
                        LlamaDocument(
                            text=page_md,
                            doc_id=doc_id,
                            metadata={
                                "source": src_name,
                                "file_name": src_name,
                                "file_path": f,
                                "page": i,
                                "total_pages": total_pages,
                                "content_type": "text",
                            },
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed loading cached md for {f}: {e}")
                continue

        if all_docs:
            logger.info(
                f"Loaded {len(all_docs)} documents from parsed cache (including table extraction)"
            )

            # Count table rows
            table_count = len(
                [d for d in all_docs if d.metadata.get("content_type") == "table_row"]
            )
            logger.info(f"Found {table_count} table rows in cached documents")

            # Store raw text for keyword fallback
            self._raw_docs_text = []
            for doc in all_docs:
                try:
                    source_name = (
                        doc.metadata.get("file_name")
                        or doc.metadata.get("source")
                        or "unknown"
                    )
                    self._raw_docs_text.append(
                        {
                            "text": getattr(doc, "text", ""),
                            "source": source_name,
                            "metadata": doc.metadata,
                        }
                    )
                except Exception as e:
                    logger.debug(f"Could not store raw text: {e}")

        return all_docs

    def _index_documents_to_qdrant(self, documents: List[LlamaDocument]) -> int:
        if not documents:
            logger.warning("No documents to index")
            return 0

        try:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            # Flexible validation and id normalization
            validated_documents = []
            for doc in documents:
                # Resolve text
                text = getattr(doc, "text", None) or ""
                # Try alternative getters if present
                if not text and hasattr(doc, "get_text"):
                    try:
                        text = doc.get_text()
                    except Exception:
                        text = ""

                # Resolve doc id from several possible attributes
                doc_id = (
                    getattr(doc, "id_", None)
                    or getattr(doc, "doc_id", None)
                    or getattr(doc, "docId", None)
                )
                if not doc_id:
                    # build stable id from content hash
                    try:
                        content_hash = hashlib.md5(
                            (text or "").encode("utf-8")
                        ).hexdigest()[:12]
                        doc_id = f"doc_{content_hash}"
                        setattr(doc, "id_", doc_id)
                    except Exception:
                        # fallback to uuid
                        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
                        try:
                            setattr(doc, "id_", doc_id)
                        except Exception:
                            pass

                # Only keep docs with some meaningful text (allow very short table_row)
                min_len = 20
                try:
                    meta = getattr(doc, "metadata", {}) or {}
                    if meta.get("content_type") == "table_row":
                        min_len = 5
                except Exception:
                    pass

                if text and len(text.strip()) > min_len:
                    validated_documents.append(doc)
                else:
                    logger.warning(
                        f"Skipping tiny or empty document: id={doc_id} len={len((text or '').strip())}"
                    )

            if not validated_documents:
                logger.error("No valid documents to index after validation")
                return 0

            logger.info(f"Indexing {len(validated_documents)} documents to Qdrant...")

            # Use configurable splitter (CHUNK_SIZE, CHUNK_OVERLAP)
            try:
                splitter = LlamaSentenceSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator=" "
                )
            except Exception:
                # Fallback
                splitter = LlamaSentenceSplitter()

            # Process in batches to avoid memory spikes
            batch_size = 10
            total_nodes = 0

            for i in range(0, len(validated_documents), batch_size):
                batch = validated_documents[i : i + batch_size]
                logger.info(
                    f"Indexing batch {i // batch_size + 1}/{(len(validated_documents) + batch_size - 1) // batch_size}"
                )

                try:
                    nodes = splitter.get_nodes_from_documents(batch)
                except Exception as e:
                    logger.error(
                        f"Sentence splitting failed for batch {i // batch_size + 1}: {e}"
                    )
                    # Try naive single-node conversion
                    nodes = []
                    for bdoc in batch:
                        try:
                            node_text = getattr(bdoc, "text", "") or ""
                            if node_text:
                                nodes.append(
                                    LlamaDocument(
                                        text=node_text,
                                        doc_id=getattr(bdoc, "id_", None)
                                        or getattr(bdoc, "doc_id", None),
                                    )
                                )
                        except Exception:
                            continue

                total_nodes += len(nodes)

                try:
                    if i == 0:
                        self.index = VectorStoreIndex(
                            nodes,
                            storage_context=storage_context,
                            embed_model=Settings.embed_model,
                            show_progress=True,
                        )
                    else:
                        self.index.insert_nodes(nodes)
                except Exception as e:
                    logger.error(f"Failed to process batch {i // batch_size + 1}: {e}")
                    continue

            logger.info(
                f"Successfully indexed {len(validated_documents)} documents as {total_nodes} nodes"
            )
            return len(validated_documents)

        except Exception as e:
            logger.error(f"Error indexing documents to Qdrant: {e}")
            return 0

    def _process_documents(self, files: List[str], force_reparse: bool = False) -> int:
        """Enhanced document processing with smart caching"""
        if not files:
            return 0

        logger.info(f"Processing {len(files)} files (force_reparse={force_reparse})")

        # If not forcing reparse, try to load from cache first
        if not force_reparse:
            cached_docs = self._load_from_parsed_cache(files)
            if cached_docs:
                logger.info("Using cached parsed documents")
                return self._index_documents_to_qdrant(cached_docs)

        # If we reach here, we need to parse (either forced or no cache)
        logger.info("Parsing documents (cache miss or forced reparse)")

        parser = self.parser
        all_documents: List[LlamaDocument] = []
        self._raw_docs_text = []
        document_counter = 0

        try:
            directory_path = os.path.dirname(files[0]) if files else LOCAL_DATA_DIR
            file_stats = parser.get_file_stats(directory_path)
            logger.info(f"File statistics: {file_stats}")
        except Exception as e:
            logger.debug(f"Could not compute file stats: {e}")

        valid_files = [
            f for f in files if not parser.should_skip_file(f) and os.path.exists(f)
        ]
        logger.info(f"Processing {len(valid_files)} valid files")

        # Group files by extension for efficient processing
        files_by_ext: Dict[str, List[str]] = {}
        for file_path in valid_files:
            ext = Path(file_path).suffix.lower()
            files_by_ext.setdefault(ext, []).append(file_path)

        total_processed_files = 0

        # Process each extension group
        for ext, file_list in files_by_ext.items():
            logger.info(f"Processing {len(file_list)} {ext} files...")

            # Structured files: force LlamaParse path
            if ext in [".xlsx", ".xls", ".csv", ".json", ".jsonl"]:
                try:
                    # Try cached first unless forcing reparse
                    if not force_reparse:
                        ext_docs = parser.load_cached_markdown_docs(file_list)
                    else:
                        ext_docs = []

                    # If no cache or forced, parse fresh
                    if not ext_docs or force_reparse:
                        logger.info(
                            f"Parsing {len(file_list)} {ext} files with LlamaParse"
                        )
                        ext_docs = parser.parse_files_with_llamaparse(
                            file_list, force_reparse=force_reparse
                        )

                    # Process documents
                    filtered_docs = self._process_document_batch(
                        ext_docs, file_list, document_counter
                    )
                    all_documents.extend(filtered_docs)
                    document_counter += len(filtered_docs)
                    total_processed_files += len(file_list)

                except Exception as e:
                    logger.error(f"Failed to process {ext} files: {e}")
                    continue

            # Non-structured files
            else:
                try:
                    if force_reparse:
                        ext_docs = parser.parse_files_with_llamaparse(
                            file_list, force_reparse=True
                        )
                    else:
                        # Try cache first
                        ext_docs = parser.load_cached_markdown_docs(file_list)
                        if not ext_docs:
                            ext_docs = parser.parse_files_with_llamaparse(
                                file_list, force_reparse=False
                            )

                    # Process documents
                    filtered_docs = self._process_document_batch(
                        ext_docs, file_list, document_counter
                    )
                    all_documents.extend(filtered_docs)
                    document_counter += len(filtered_docs)
                    total_processed_files += len(file_list)

                except Exception as e:
                    logger.error(f"Failed to process {ext} files: {e}")
                    continue

        if not all_documents:
            logger.warning("No documents were processed")
            return 0

        # Deduplicate and normalize
        unique_documents = self._deduplicate_documents(all_documents)
        logger.info(f"After deduplication: {len(unique_documents)} unique documents")

        # ADD DEBUG SECTION
        logger.info("=== DOCUMENT PROCESSING DEBUG ===")
        table_docs_count = 0
        text_docs_count = 0
        table_types = {}

        for doc in unique_documents:
            content_type = doc.metadata.get("content_type", "unknown")
            if content_type.startswith("table"):
                table_docs_count += 1
                table_types[content_type] = table_types.get(content_type, 0) + 1
            else:
                text_docs_count += 1

        logger.info(f"Final document breakdown:")
        logger.info(f"  - Table documents: {table_docs_count}")
        for ttype, count in table_types.items():
            logger.info(f"    - {ttype}: {count}")
        logger.info(f"  - Text documents: {text_docs_count}")
        logger.info(f"  - Total: {len(unique_documents)}")

        # Store raw text for keyword fallback
        for doc in unique_documents:
            try:
                source_name = (
                    doc.metadata.get("file_name")
                    or doc.metadata.get("source")
                    or "unknown"
                )
                self._raw_docs_text.append(
                    {
                        "text": getattr(doc, "text", ""),
                        "source": source_name,
                        "metadata": doc.metadata,
                    }
                )
            except Exception as e:
                logger.debug(f"Could not store raw text: {e}")

        # Index to Qdrant
        unique_documents = self._remove_table_duplicates_simple(unique_documents)
        return self._index_documents_to_qdrant(unique_documents)

    def _remove_table_duplicates_simple(
        self, documents: List[LlamaDocument]
    ) -> List[LlamaDocument]:
        """Simple duplicate removal for identical table rows"""
        seen_content = set()
        filtered = []

        for doc in documents:
            meta = getattr(doc, "metadata", {}) or {}
            if meta.get("content_type") == "table_row":
                # Create simple signature
                key_vals = meta.get("key_values", {})
                signature = (
                    str(sorted(key_vals.items()))
                    if key_vals
                    else getattr(doc, "text", "")
                )

                if signature in seen_content:
                    continue  # Skip duplicate
                seen_content.add(signature)

            filtered.append(doc)

        logger.info(
            f"Simple deduplication: {len(documents)} -> {len(filtered)} documents"
        )
        return filtered

    def _process_document_batch(
        self, docs: List[LlamaDocument], file_list: List[str], counter_start: int
    ) -> List[LlamaDocument]:
        """Process documents with table-aware handling and deduplication"""
        all_docs = []
        counter = counter_start

        for d in docs:
            try:
                text_content = getattr(d, "text", "") or ""
                if not text_content or len(text_content.strip()) <= 5:
                    continue

                meta = getattr(d, "metadata", {}) or {}
                src_name = meta.get("file_name") or meta.get("source") or "unknown"
                page_num = meta.get("page", 1)

                # Check for tables
                has_tables = self.parser._detect_tables_in_content(text_content)

                if has_tables:
                    logger.info(f"Extracting tables from {src_name} page {page_num}")
                    table_docs = self.parser._extract_and_structure_tables(
                        text_content, src_name, page_num
                    )
                    if table_docs:
                        logger.info(
                            f"Created {len(table_docs)} table documents from page {page_num}"
                        )
                        all_docs.extend(table_docs)
                        counter += len(table_docs)
                        continue

                # Regular document processing...
                counter += 1
                doc_id = f"{Path(src_name).stem}_chunk_{counter}"
                doc = LlamaDocument(
                    text=text_content,
                    doc_id=doc_id,
                    metadata={
                        "source": src_name,
                        "file_name": src_name,
                        "chunk_id": counter,
                        "content_type": "text",
                    },
                )

                try:
                    setattr(doc, "id_", doc_id)
                except Exception:
                    pass

                all_docs.append(doc)

            except Exception as e:
                logger.warning(f"Error processing document: {e}")
                continue

        # CRITICAL: Apply deduplication before returning
        unique_docs = self._deduplicate_table_rows(all_docs)
        return unique_docs

    def _deduplicate_documents(
        self, documents: List[LlamaDocument]
    ) -> List[LlamaDocument]:
        """Enhanced deduplication with table-specific logic"""

        # First, deduplicate table rows by content
        documents = self._deduplicate_table_rows(documents)

        # Then apply general deduplication
        unique_documents = []
        seen_hashes = set()
        seen_ids = set()

        for doc in documents:
            try:
                text_content = getattr(doc, "text", "") or ""
                meta = getattr(doc, "metadata", {}) or {}

                # Different minimum lengths for different content types
                min_len = 20
                if meta.get("content_type") == "table_row":
                    min_len = 5
                elif meta.get("content_type") in ["table_complete", "table_summary"]:
                    min_len = 50

                if text_content and len(text_content.strip()) > min_len:
                    # Create hash for general content deduplication
                    content_hash = hashlib.md5(text_content.encode("utf-8")).hexdigest()

                    if content_hash in seen_hashes:
                        logger.debug(
                            f"Skipping duplicate content: {meta.get('source')} page {meta.get('page')}"
                        )
                        continue

                    seen_hashes.add(content_hash)

                    # Ensure unique document ID
                    doc_id = (
                        getattr(doc, "id_", None)
                        or getattr(doc, "doc_id", None)
                        or f"doc_{content_hash[:12]}_{len(unique_documents)}"
                    )

                    # Set normalized IDs
                    try:
                        setattr(doc, "id_", doc_id)
                        setattr(doc, "doc_id", doc_id)
                    except Exception:
                        pass

                    # Normalize metadata
                    if not hasattr(doc, "metadata") or not doc.metadata:
                        doc.metadata = {}

                    source_name = (
                        doc.metadata.get("file_name")
                        or doc.metadata.get("source")
                        or "unknown"
                    )

                    if source_name != "unknown":
                        source_name = Path(source_name).name

                    doc.metadata["source"] = source_name
                    doc.metadata["file_name"] = source_name

                    # Handle ID conflicts
                    if doc_id in seen_ids:
                        new_id = f"{doc_id}_{len(unique_documents)}"
                        try:
                            setattr(doc, "id_", new_id)
                            setattr(doc, "doc_id", new_id)
                        except Exception:
                            pass
                        doc_id = new_id

                    seen_ids.add(doc_id)
                    unique_documents.append(doc)

            except Exception as e:
                logger.error(f"Error processing document during deduplication: {e}")
                continue

        return unique_documents

    def _detect_tables_in_content(self, text_content: str) -> bool:
        """Enhanced table detection - delegate to parser"""
        return self.parser._detect_tables_in_content(text_content)

    def _create_query_engine(
        self, similarity_top_k=RAG_TOP_K, fetch_k=30, use_mmr=True
    ):
        if self.index is None:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
            vector_store_query_mode=("mmr" if use_mmr else "default"),
            alpha=0.5,
            embed_model=Settings.embed_model,
        )

        similarity_postprocessor = SimilarityPostprocessor(
            similarity_cutoff=SIMILARITY_THRESHOLD
        )

        return RetrieverQueryEngine(
            retriever=retriever, node_postprocessors=[similarity_postprocessor]
        )

    def get_query_engine(self):
        if not self.query_engine:
            if self.vector_store is None:
                raise ValueError("Vector store is not initialized")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            self.query_engine = self._create_query_engine()
        return self.query_engine

    def load_onedrive_documents(
        self, force_rebuild: bool = False, shared_folder_url: Optional[str] = None
    ):
        """
        ENHANCED SMART LOADING LOGIC:
        1. If changes detected: Reparse → Reindex → Save to Qdrant
        2. If no changes + collection exists: Use existing Qdrant collection
        3. If no changes + no collection: Use parsed cache → Reindex → Save to Qdrant
        """
        local_dir = "data/onedrive"
        os.makedirs(local_dir, exist_ok=True)
        files: List[str] = []
        newly_downloaded = False

        # Step 1: Handle OneDrive downloads
        if shared_folder_url:
            print(f"🔄 Checking OneDrive for updates...")
            try:
                # Get existing files before download
                existing_files = set()
                if os.path.exists(local_dir):
                    for root, _, filenames in os.walk(local_dir):
                        for filename in filenames:
                            existing_files.add(os.path.join(root, filename))

                # Download from OneDrive
                downloaded_files = self.onedrive_client.download_shared_folder(
                    local_dir, shared_folder_url
                )

                if downloaded_files:
                    # Check for new files
                    new_files = [f for f in downloaded_files if f not in existing_files]
                    if new_files:
                        print(f"📥 Downloaded {len(new_files)} new files from OneDrive")
                        newly_downloaded = True
                        # Clear cache for new files only
                        try:
                            removed = self.parser.clear_parsed_cache(new_files)
                            if removed:
                                print(
                                    f"🗑️ Cleared {removed} cached parse artifacts for new files"
                                )
                        except Exception as e:
                            print(f"⚠️ Warning: could not clear parse cache: {e}")
                    else:
                        print(f"✅ OneDrive up to date ({len(downloaded_files)} files)")

                    files = downloaded_files
                else:
                    print("📂 No files found in OneDrive folder")

            except Exception as e:
                print(f"❌ OneDrive download error: {e}")

        # Step 2: Fallback to local files if no OneDrive files
        if not files and os.path.exists(LOCAL_DATA_DIR):
            local_dir = LOCAL_DATA_DIR
            print(f"📁 Using local documents from: {local_dir}")
            files = self._collect_source_files(local_dir)

        if not files:
            print("❌ No documents found")
            return 0

        print(f"📊 Found {len(files)} total documents to process")

        # Step 3: Check current state
        collection_exists = self._collection_exists_and_has_data()
        has_parsed_cache = self._has_cached_parsed_files(files)
        data_changed = self._data_changed(local_dir, files) or newly_downloaded

        print(f"📋 Status Check:")
        print(f"   • Collection exists in Qdrant: {collection_exists}")
        print(f"   • Parsed cache available: {has_parsed_cache}")
        print(f"   • Data changed: {data_changed}")
        print(f"   • Force rebuild: {force_rebuild}")

        # Step 4: SMART DECISION LOGIC
        if force_rebuild or data_changed:
            # Case 1: Changes detected or forced rebuild
            reason = []
            if force_rebuild:
                reason.append("forced rebuild")
            if data_changed:
                reason.append("data changed")
            if newly_downloaded:
                reason.append("new files downloaded")

            print(f"🔄 REBUILDING INDEX - Reason: {', '.join(reason)}")

            try:
                # Clear existing collection
                if collection_exists:
                    try:
                        self.qdrant.delete_collection(collection_name=COLLECTION_NAME)
                        print(f"🗑️ Deleted existing collection: {COLLECTION_NAME}")
                    except Exception as e:
                        print(f"⚠️ Could not delete collection: {e}")

                # Clear all parsed cache if doing full rebuild
                if force_rebuild or (data_changed and not newly_downloaded):
                    try:
                        removed_all = self.parser.clear_parsed_cache()
                        if removed_all:
                            print(f"🗑️ Cleared {removed_all} cached parse artifacts")
                    except Exception as e:
                        print(f"⚠️ Warning: could not clear parse cache: {e}")

                # Recreate vector store
                self._setup_vector_store()

                # Parse and index documents
                n = self._process_documents(files, force_reparse=True)

                # Save fingerprint
                self._save_fingerprint(self._calc_fingerprint_for_files(files))

                print(f"✅ Successfully rebuilt index with {n} documents")
                return n

            except Exception as e:
                print(f"❌ Error rebuilding index: {e}")
                return 0

        elif collection_exists:
            # Case 2: No changes + collection exists = Use existing Qdrant
            print("✅ NO CHANGES DETECTED - Using existing Qdrant collection")
            try:
                # Just initialize query engine from existing vector store
                self.get_query_engine()

                # Load raw docs for keyword fallback if not already loaded
                if not self._raw_docs_text and has_parsed_cache:
                    try:
                        cached_docs = self._load_from_parsed_cache(files)
                        print(
                            f"📚 Loaded {len(self._raw_docs_text)} documents for keyword fallback"
                        )
                    except Exception as e:
                        print(f"⚠️ Could not load raw docs for keyword fallback: {e}")

                return len(files)

            except Exception as e:
                print(f"❌ Error using existing collection: {e}")
                # Fall through to next case
                collection_exists = False

        if not collection_exists:
            # Case 3: No changes + no collection = Use parsed cache → Index → Save
            if has_parsed_cache:
                print("🔄 NO COLLECTION FOUND - Rebuilding from parsed cache")
                try:
                    # Recreate vector store
                    self._setup_vector_store()

                    # Load from cache and index
                    n = self._process_documents(files, force_reparse=False)

                    # Save fingerprint
                    self._save_fingerprint(self._calc_fingerprint_for_files(files))

                    print(
                        f"✅ Successfully rebuilt index from cache with {n} documents"
                    )
                    return n

                except Exception as e:
                    print(f"❌ Error rebuilding from cache: {e}")
                    # Fall through to full rebuild

            # Case 4: No collection + no cache = Full rebuild
            print("🔄 NO COLLECTION OR CACHE - Full rebuild required")
            try:
                self._setup_vector_store()
                n = self._process_documents(files, force_reparse=True)
                self._save_fingerprint(self._calc_fingerprint_for_files(files))
                print(f"✅ Successfully built index with {n} documents")
                return n

            except Exception as e:
                print(f"❌ Error in full rebuild: {e}")
                return 0

        # Fallback - should not reach here
        print("⚠️ Unexpected state - attempting basic initialization")
        try:
            self.get_query_engine()
            return len(files)
        except Exception as e:
            print(f"❌ Fallback initialization failed: {e}")
            return 0

    # Keep your existing fingerprint helper methods unchanged
    def _calc_fingerprint(self, directory_path: str) -> Dict[str, Any]:
        fp: Dict[str, Any] = {}
        supported = set(self.parser.supported_exts) | {
            ".xlsx",
            ".xls",
            ".csv",
            ".json",
            ".jsonl",
        }
        parsed_dir = Path(PARSED_CACHE_DIR).resolve()

        for root, _, files in os.walk(directory_path):
            root_path = Path(root).resolve()
            if root_path == parsed_dir or parsed_dir in root_path.parents:
                continue

            for f in files:
                path = os.path.join(root, f)
                if path.endswith("fingerprint.json") or self.parser.should_skip_file(
                    path
                ):
                    continue
                ext = Path(path).suffix.lower()
                if ext not in supported:
                    continue
                try:
                    p = Path(path).resolve()
                    with p.open("rb") as fh:
                        content = fh.read()
                    fp[str(p)] = {
                        "hash": hashlib.md5(content).hexdigest(),
                        "size": p.stat().st_size,
                        "mtime": p.stat().st_mtime,
                    }
                except Exception as e:
                    logger.warning(f"Fingerprint error {path}: {e}")
                    # continue rather than returning partial fp
                    continue
        return fp

    def _calc_fingerprint_for_files(self, files: List[str]) -> Dict[str, Any]:
        fp: Dict[str, Any] = {}
        for path in files:
            try:
                if self.parser.should_skip_file(path):
                    continue
                p = Path(path).resolve()
                with p.open("rb") as fh:
                    content = fh.read()
                fp[str(p)] = {
                    "hash": hashlib.md5(content).hexdigest(),
                    "size": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                }
            except Exception as e:
                logger.warning(f"Fingerprint error {path}: {e}")
                continue
        return fp

    def _save_fingerprint(self, fp: Dict[str, Any], path: Optional[str] = None):
        path = path or self.fingerprint_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(fp, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"Failed to save fingerprint to {path}: {e}")
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass

    def _load_fingerprint(self, path: Optional[str] = None) -> Dict[str, Any]:
        path = path or self.fingerprint_path
        try:
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load fingerprint from {path}: {e}")
            return {}

    def _data_changed(
        self, directory_path: str, files: Optional[List[str]] = None
    ) -> bool:
        old = self._load_fingerprint()
        new = (
            self._calc_fingerprint_for_files(files)
            if files is not None
            else self._calc_fingerprint(directory_path)
        )

        print(
            f"[fingerprint] path={self.fingerprint_path}, exists={os.path.exists(self.fingerprint_path)}"
        )
        print(f"[fingerprint] old entries={len(old)}, new entries={len(new)}")

        old_keys = set(old.keys())
        new_keys = set(new.keys())
        added = new_keys - old_keys
        removed = old_keys - new_keys
        changed = [
            k
            for k in new_keys & old_keys
            if old.get(k, {}).get("hash") != new.get(k, {}).get("hash")
        ]
        if added or removed or changed:
            print(
                f"[fingerprint] added={list(added)[:10]}, removed={list(removed)[:10]}, changed={changed[:10]}"
            )
        else:
            print("[fingerprint] no differences detected")

        # existing logic continues...
        if not old:
            logger.info("No existing fingerprint loaded -> treat as changed")
            return True
        if old_keys != new_keys:
            logger.info(
                f"Fingerprint keys differ: added={len(added)}, removed={len(removed)}"
            )
            return True
        for k in new_keys:
            if old.get(k, {}).get("hash") != new[k].get("hash"):
                logger.info(f"Fingerprint content changed for: {k}")
                return True
        return False

    def keyword_fallback_search(
        self, query: str, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        q_terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for doc in self._raw_docs_text:
            text_terms = re.findall(r"\w+", (doc.get("text") or "").lower())
            if not text_terms:
                continue
            overlap = len(q_terms.intersection(text_terms))
            if overlap == 0:
                continue
            score = overlap / math.sqrt(max(1, len(text_terms)))
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": s,
                "source": d["source"],
                "text_snippet": (d["text"][:1000] + "...")
                if len(d["text"]) > 1000
                else d["text"],
            }
            for s, d in scored[:top_n]
        ]


# CustomWebSearchTool
class CustomWebSearchTool:
    def __init__(
        self,
        api_key: Optional[str],
        cse_id: Optional[str],
        req_timeout: int = WEB_REQ_TIMEOUT_SECS,
        llm_timeout: int = LLM_COMPLETE_TIMEOUT,
    ):
        self.api_key = api_key
        self.cse_id = cse_id
        self.available = bool(api_key and cse_id)
        self.req_timeout = req_timeout
        self.llm_timeout = llm_timeout
        self.restricted_site = "eshop.bh.zain.com"

    def is_available(self) -> bool:
        return self.available

    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """Detect what the user is specifically looking for"""
        query_lower = query.lower()

        intent = {
            "query_type": "general",
            "specific_terms": [],
            "search_focus": [],
            "expected_info": [],
        }

        # Detect specific query types
        if any(
            word in query_lower for word in ["offer", "promotion", "deal", "discount"]
        ):
            intent["query_type"] = "offers"
            intent["search_focus"] = ["offers", "promotions", "deals", "discounts"]
            intent["expected_info"] = [
                "promotional terms",
                "discount amounts",
                "validity period",
            ]

        elif any(word in query_lower for word in ["price", "cost", "how much", "bd"]):
            intent["query_type"] = "pricing"
            intent["search_focus"] = ["price", "cost", "payment", "monthly"]
            intent["expected_info"] = ["exact prices", "payment plans", "monthly costs"]

        elif any(word in query_lower for word in ["plan", "package", "subscription"]):
            intent["query_type"] = "plans"
            intent["search_focus"] = ["plans", "packages", "subscription"]
            intent["expected_info"] = ["plan details", "features", "data allowances"]

        elif any(
            word in query_lower
            for word in ["iphone", "samsung", "honor", "phone", "device"]
        ):
            intent["query_type"] = "devices"
            intent["search_focus"] = ["specifications", "features", "availability"]
            intent["expected_info"] = [
                "device specs",
                "colors",
                "storage",
                "availability",
            ]

        elif any(
            word in query_lower for word in ["broadband", "internet", "wifi", "home"]
        ):
            intent["query_type"] = "broadband"
            intent["search_focus"] = ["broadband", "internet", "home", "speed"]
            intent["expected_info"] = ["speeds", "data limits", "installation"]

        # Extract specific terms from query
        import re

        words = re.findall(r"\b\w+\b", query_lower)
        intent["specific_terms"] = [word for word in words if len(word) > 2]

        return intent

    def search(
        self, query: str, num_results: int = 5, site: Optional[str] = None
    ) -> str:
        """Enhanced search restricted to Zain eshop only"""
        if not self.available:
            return "Web search unavailable: set GOOGLE_API_KEY and GOOGLE_CSE_ID."

        try:
            intent = self._detect_query_intent(query)
            url = "https://www.googleapis.com/customsearch/v1"

            # Build targeted search query based on intent
            base_query = f"site:{self.restricted_site}"

            if intent["query_type"] == "offers":
                # For offer-related queries, search for promotional content
                search_terms = " OR ".join(
                    [f'"{term}"' for term in intent["search_focus"]]
                )
                restricted_query = f"{base_query} ({search_terms}) {query}"

            elif intent["query_type"] == "pricing":
                # For pricing queries, include pricing-related terms
                restricted_query = f'{base_query} (price OR cost OR "BD" OR payment OR monthly) {query}'

            elif intent["query_type"] == "devices":
                # For device queries, focus on product pages
                restricted_query = f"{base_query} (specifications OR features OR available OR buy) {query}"

            elif intent["query_type"] == "broadband":
                # For broadband queries
                restricted_query = f'{base_query} (broadband OR internet OR "home internet" OR speed) {query}'

            else:
                # General query - use original terms but be more specific
                restricted_query = f"{base_query} {query}"

            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": restricted_query,
                "num": num_results,
            }

            r = requests.get(url, params=params, timeout=self.req_timeout)
            r.raise_for_status()
            data = r.json()

            if "error" in data:
                return f"Error from search API: {data['error'].get('message', 'Unknown error')}"

            items = data.get("items", [])[:num_results]
            if not items:
                return f"No results found on {self.restricted_site} for: {query}"

            blocks = []
            for i, item in enumerate(items, 1):
                title = item.get("title", "No Title")
                snippet = item.get("snippet", "No description available")
                link = item.get("link", "#")
                display_link = item.get("displayLink", "")

                block = f"""**Result {i}: {title}**
    Source: {display_link}
    Description: {snippet}
    URL: {link}
    ---"""
                blocks.append(block)

            return "\n\n".join(blocks)

        except requests.Timeout:
            return f"Error: search request timed out for {self.restricted_site}."
        except Exception as e:
            return f"Error searching {self.restricted_site}: {e}"

    def search_and_analyze(
        self, query: str, convo_context: str = "", site: Optional[str] = None
    ) -> str:
        """Enhanced search with detailed analysis, restricted to Zain eshop"""
        results = self.search(query, num_results=5)

        if results.startswith("Error") or results.startswith("No results"):
            return results

        intent = self._detect_query_intent(query)

        # Create a focused prompt based on the user's specific question
        prompt = f"""
    You are an expert analyst providing specific information from Zain Bahrain's eshop website search results.

    **CRITICAL INSTRUCTION**: The user asked specifically about "{query}". Your response must DIRECTLY answer this question with specific information found in the search results.

    **User's Exact Question:** {query}
    **Query Type Detected:** {intent["query_type"]}
    **Expected Information:** {", ".join(intent["expected_info"]) if intent["expected_info"] else "specific details"}

    **Your Task:**
    1. Find information in the search results that DIRECTLY relates to "{query}"
    2. Extract SPECIFIC details (prices, features, terms, availability, etc.)
    3. Ignore general categories or unrelated products
    4. Focus only on what the user asked about

    **Analysis Rules:**
    - If no specific information about "{query}" is found, say so clearly
    - Don't include general product categories unless they directly answer the query
    - Extract exact prices, specifications, terms, or details mentioned
    - Include only relevant links that relate to the user's question
    - Be precise and specific, not generic

    **Search Results from {self.restricted_site}:**
    {results}

    **Response Structure:**

    ## Direct Answer to: "{query}"
    [Provide specific information found that answers the user's exact question]

    ## Specific Details Found
    [List only specific details related to the query: prices, features, terms, availability, etc.]

    ## Relevant Links
    [Include only links that directly relate to the user's question]

    ## Summary
    [Brief summary of what was specifically found regarding "{query}" - if nothing specific was found, state that clearly]

    **Remember**: Be specific to the user's question "{query}" - don't provide generic information.

    **Your Response:**
    """

        try:
            analysis = Settings.llm.complete(
                prompt, timeout=self.llm_timeout + 30
            ).text.strip()
            return analysis
        except Exception as e:
            return self._create_targeted_fallback(results, query, intent, e)

    def _create_targeted_fallback(
        self, results: str, query: str, intent: Dict, error: Exception
    ) -> str:
        """Create a targeted fallback response based on user's specific query"""

        result_blocks = results.split("---")
        relevant_results = []

        # Filter results for relevance to the user's query
        query_terms = set(query.lower().split())

        for block in result_blocks:
            if "**Result" in block and "Source:" in block:
                # Check if this result is relevant to the user's query
                block_text = block.lower()
                relevance_score = len(query_terms.intersection(set(block_text.split())))

                if relevance_score > 0:  # At least some query terms match
                    lines = block.strip().split("\n")
                    title = ""
                    source = ""
                    description = ""
                    url = ""

                    for line in lines:
                        if line.startswith("**Result") and ":" in line:
                            title = line.split(":", 1)[1].strip().rstrip("**")
                        elif line.startswith("Source:"):
                            source = line.replace("Source:", "").strip()
                        elif line.startswith("Description:"):
                            description = line.replace("Description:", "").strip()
                        elif line.startswith("URL:"):
                            url = line.replace("URL:", "").strip()

                    if title and source:
                        relevant_results.append(
                            {
                                "title": title,
                                "source": source,
                                "description": description,
                                "url": url,
                                "relevance": relevance_score,
                            }
                        )

        # Sort by relevance
        relevant_results.sort(key=lambda x: x["relevance"], reverse=True)

        fallback_response = f"""## Zain Bahrain Eshop Results for: "{query}"

    **Search Domain:** {self.restricted_site}

    """

        if relevant_results:
            fallback_response += f"## Relevant Results Found\n\n"
            for i, result in enumerate(relevant_results[:3], 1):  # Top 3 most relevant
                fallback_response += f"""**{i}. {result["title"]}**
    - **Details:** {result["description"]}
    - **Link:** {result["url"]}

    """

            fallback_response += f"""## Direct Links\n"""
            for i, result in enumerate(relevant_results[:3], 1):
                fallback_response += f"{i}. {result['title']}: {result['url']}\n"

            fallback_response += f"""
    ## Summary
    Found {len(relevant_results)} results related to "{query}" on Zain Bahrain's eshop."""

        else:
            fallback_response += f"""## No Specific Results
    No specific information about "{query}" was found in the search results from {self.restricted_site}.

    You may want to:
    1. Try a different search term
    2. Visit the main eshop: https://{self.restricted_site}
    3. Contact Zain Bahrain directly for specific information about "{query}"
    """

        fallback_response += f"""
    *Note: Analysis error occurred: {str(error)[:100]}*
    """

        return fallback_response

    def _create_detailed_fallback(
        self, results: str, query: str, site_note: str, error: Exception
    ) -> str:
        # Parse the results to extract titles and sources
        result_blocks = results.split("---")
        sources = []

        for block in result_blocks:
            if "**Result" in block and "Source:" in block:
                lines = block.strip().split("\n")
                title = ""
                source = ""
                description = ""
                url = ""

                for line in lines:
                    if line.startswith("**Result") and ":" in line:
                        title = line.split(":", 1)[1].strip().rstrip("**")
                    elif line.startswith("Source:"):
                        source = line.replace("Source:", "").strip()
                    elif line.startswith("Description:"):
                        description = line.replace("Description:", "").strip()
                    elif line.startswith("URL:"):
                        url = line.replace("URL:", "").strip()

                if title and source:
                    sources.append(
                        {
                            "title": title,
                            "source": source,
                            "description": description,
                            "url": url,
                        }
                    )

        # Create structured fallback response
        fallback_response = f"""## Web Search Results {site_note}

**Query:** {query}

## Detailed Findings

Based on the search results, here are the key findings:

"""

        for i, source_info in enumerate(sources, 1):
            fallback_response += f"""
**Finding {i} - {source_info["title"]}**
- Source: {source_info["source"]}
- Details: {source_info["description"]}
- Reference: {source_info["url"]}
"""

        fallback_response += f"""

## Key Information Summary

"""

        for i, source_info in enumerate(sources, 1):
            fallback_response += f"• **{source_info['title']}** ({source_info['source']}): {source_info['description'][:100]}...\n"

        fallback_response += f"""

## References

"""

        for i, source_info in enumerate(sources, 1):
            fallback_response += f"{i}. **{source_info['title']}** - {source_info['source']} [{source_info['url']}]\n"

        fallback_response += f"""

## Executive Summary

Found {len(sources)} relevant sources about "{query}". The results provide various perspectives and information from different sources including {", ".join([s["source"] for s in sources[:3]])}{"..." if len(sources) > 3 else ""}.

*Note: Detailed analysis unavailable due to processing error: {str(error)[:100]}*
"""

        return fallback_response

    def search_and_summarize(
        self, query: str, convo_context: str = "", site: Optional[str] = None
    ) -> str:
        """Legacy method - now calls the enhanced search_and_analyze"""
        return self.search_and_analyze(query, convo_context, site)


# EnInternl Agents
class InternalAgents:
    def __init__(self, query_engine: RetrieverQueryEngine, document_manager=None):
        self.query_engine = query_engine
        self.document_manager = document_manager

        # Get the index from document manager if available
        if (
            document_manager
            and hasattr(document_manager, "index")
            and document_manager.index
        ):
            self.index = document_manager.index
            logger.info("Successfully accessed index from document manager")
        else:
            self.index = None
            logger.warning("Could not access index from document manager")

        # Rest of your existing initialization code...
        self.rag_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="rag_tool",
            description="Retrieve the most relevant internal information for staff; include dates, amounts, etc.",
        )

        # Create additional function tools if needed
        def analyze_policy(query: str) -> str:
            """Analyze internal policies and provide structured information."""
            try:
                result = self.query_engine.query(f"Policy analysis: {query}")
                return str(result)
            except Exception as e:
                return f"Policy analysis error: {e}"

        def get_financial_details(query: str) -> str:
            """Get specific financial information like amounts, limits, and tenure details."""
            try:
                result = self.query_engine.query(f"Financial details: {query}")
                return str(result)
            except Exception as e:
                return f"Financial details error: {e}"

        # Create function tools
        self.policy_tool = FunctionTool.from_defaults(
            fn=analyze_policy,
            name="policy_analyzer",
            description="Analyze policies and extract structured information",
        )

        self.finance_tool = FunctionTool.from_defaults(
            fn=get_financial_details,
            name="finance_analyzer",
            description="Extract financial details like amounts, limits, tenure from documents",
        )

        # Initialize agents with correct syntax
        tools = [self.rag_tool, self.policy_tool, self.finance_tool]

        try:
            self.reasoning_agent = ReActAgent(
                llm=Settings.llm, tools=tools, verbose=False
            )
            self.reviewer_agent = ReActAgent(
                llm=Settings.llm,
                tools=[self.rag_tool],  # Reviewer only uses RAG tool
                verbose=False,
            )
            self.summarizer_agent = ReActAgent(
                llm=Settings.llm,
                tools=[],  # Summarizer doesn't need tools
                verbose=False,
            )
            logger.info("Successfully initialized ReAct agents")
        except Exception as e:
            logger.warning(f"Could not initialize ReActAgents: {e}")
            # Fallback: set to None and handle in methods
            self.reasoning_agent = None
            self.reviewer_agent = None
            self.summarizer_agent = None

    def _direct_value_lookup(self, best_candidate: Dict, query: str) -> Optional[str]:
        """Enhanced direct value extraction with proper table intersection logic"""

        query_lower = query.lower()
        key_values = best_candidate.get("_metadata", {}).get("key_values", {})
        source = best_candidate.get("_metadata", {}).get("source", "table_lookup")

        # Extract entities from query
        row_entity = self._extract_row_entity(
            query_lower
        )  # e.g., "fiber extra", "fiber cash"
        column_entity = self._extract_column_entity(
            query_lower
        )  # e.g., "connection fees", "extra data sim code"

        logger.info(
            f"Extracted entities - Row: '{row_entity}', Column: '{column_entity}'"
        )

        # ENHANCED TABLE INTERSECTION LOGIC
        if row_entity and column_entity:
            # Strategy 1: Find exact row×column intersection
            intersection_value = self._find_table_intersection(
                key_values, row_entity, column_entity
            )
            if intersection_value:
                return f"""## Answer
    **{intersection_value}**

    ## Details
    {column_entity.title()} for {row_entity.title()}

    ## Source
    *{source}*

    ## Table Context
    Row: {row_entity.title()}
    Column: {column_entity.title()}
    Value: {intersection_value}

    ---
    """

        # Strategy 2: Enhanced specific field matching
        if "connection fee" in query_lower or "connection charge" in query_lower:
            # Look for connection-related fields
            connection_fields = [
                "connection",
                "setup",
                "installation",
                "activation",
                "one time",
                "initial",
            ]

            found_value = None
            found_key = None

            for k, v in key_values.items():
                k_lower = k.lower()
                if any(field in k_lower for field in connection_fields) and any(
                    fee_word in k_lower
                    for fee_word in ["fee", "charge", "cost", "amount"]
                ):
                    found_value = v
                    found_key = k
                    break

            if found_value:
                return f"""## Answer
    **{found_value}**

    ## Details
    Connection fee as specified in the source data

    ## Source
    *{source}*

    ## Field Details
    Field: {found_key}
    Value: {found_value}

    ---
    *Connection fee extracted from structured data*"""
            else:
                # Check if connection fees are mentioned as "0" or "free"
                for k, v in key_values.items():
                    if "0" in str(v) or "free" in str(v).lower():
                        return f"""## Answer
    **{v}** (No connection fees)

    ## Details
    Based on the data structure, connection appears to be {v}

    ## Source
    *{source}*"""

        # Strategy 3: Enhanced code/SIM code matching
        elif "sim code" in query_lower or "extra data" in query_lower:
            # For queries like "Extra Data SIM Code of Fiber Extra"

            # Look for the specific package (row entity) in the column headers or values
            target_package = row_entity or "fiber extra"

            sim_code_value = None

            # Check if any key contains both the target package and relates to SIM/code
            for k, v in key_values.items():
                k_lower = k.lower()
                v_lower = str(v).lower()

                # If this key represents the target package column and relates to SIM codes
                if target_package in k_lower and any(
                    indicator in k_lower for indicator in ["code", "sim"]
                ):
                    sim_code_value = v
                    break

                # Alternative: if the value contains a code pattern and key mentions the package
                elif target_package in k_lower and re.match(
                    r"^[A-Z]{4,}[0-9]+$", str(v)
                ):
                    sim_code_value = v
                    break

            # If still no direct match, look for pattern matching
            if not sim_code_value:
                # Look for code patterns in values where key might contain target package
                for k, v in key_values.items():
                    if target_package in k.lower() and v and str(v).strip() != "-":
                        # Check if value looks like a code (e.g., EXTRAF10)
                        if re.match(r"^[A-Z]{3,}[0-9]+[A-Z0-9]*$", str(v).strip()):
                            sim_code_value = v
                            break

            if sim_code_value:
                return f"""## Answer
    **{sim_code_value}**

    ## Details
    Extra Data SIM Code for {target_package.title()}

    ## Source
    *{source}*

    ---
    *Code extracted from structured table data*"""

        # ENHANCED: Check for TABS Flag specifically
        elif "tabs flag" in query_lower:
            # Look for TABS-related keys
            tabs_value = None

            # Check direct TABS flag
            for k, v in key_values.items():
                if "tabs" in k.lower() and "flag" in k.lower():
                    tabs_value = v
                    break

            # If no direct TABS flag, check MRS Flag (often the same thing)
            if not tabs_value:
                for k, v in key_values.items():
                    if "mrs flag" in k.lower() or "flag" in k.lower():
                        tabs_value = v
                        break

            if tabs_value:
                return f"""## Answer
    **{tabs_value}**

    ## Details
    TABS Flag for iPad 10.2-inch iPad Wi-Fi + Cellular 64GB

    ## Source
    *{source}*

    ## Complete Information Found
    • **Device**: {key_values.get("Device", "N/A")}
    • **Item Name**: {key_values.get("Item Name", "N/A")}
    • **TABS/MRS Flag**: {tabs_value}
    • **Installment Amount**: {key_values.get("Installment Amount", "N/A")}

    ---
    *Value extracted from structured table data*"""

        # Enhanced code/flag detection
        elif any(term in query_lower for term in ["flag", "code"]):
            # Look for any flag or code-related keys
            found_flags = {}

            for k, v in key_values.items():
                if any(
                    indicator in k.lower()
                    for indicator in ["flag", "code", "mrs", "tabs"]
                ):
                    found_flags[k] = v

            if found_flags:
                result_lines = ["## Answer"]
                for key, val in found_flags.items():
                    result_lines.append(f"**{key}**: {val}")

                result_lines.append(f"\n## Source\n*{source}*")
                return "\n".join(result_lines)

        # Keep existing logic for other types...
        elif "total monthly" in query_lower:
            # Check key_values first
            for k, v in key_values.items():
                if "total monthly" in f"{k} {v}".lower():
                    return f"""## Answer
    **{v}**

    ## Details
    Total Monthly for the requested item

    ## Source
    *{source}*"""

        elif "package code" in query_lower:
            for k, v in key_values.items():
                if "package code" in f"{k} {v}".lower() or "code" in k.lower():
                    return f"""## Answer
    **{v}**

    ## Source
    *{source}*"""

        elif "download speed" in query_lower:
            for k, v in key_values.items():
                if "download speed" in f"{k} {v}".lower():
                    return f"""## Answer
    **{v}**

    ## Source
    *{source}*"""

        return None

    def _extract_row_entity(self, query_lower: str) -> Optional[str]:
        """Extract the row entity (package/service name) from the query"""

        # Common package patterns
        package_patterns = [
            r"\bfiber\s+extra\b",
            r"\bfiber\s+elite\b",
            r"\bfiber\s+ultra\b",
            r"\bfiber\s+turbo\b",
            r"\bfiber\s+basic\b",
            r"\bfiber\s+lite\b",
            r"\bfiber\s+cash\b",
            r"\bal\s+zain\+?\b",
            r"\bzain\s+\+?\b",
            # Gaming router patterns
            r"\bgaming\s+router\b",
            r"\bmesh\s+router\b",
            r"\brouter\b",
            # Device patterns
            r"\bipad\s+[\w\s-]*\b",
            r"\biphone\s+[\w\s-]*\b",
            r"\bsamsung\s+[\w\s-]*\b",
            # Gift card patterns
            r"\bbd\s*\d+\s+gift\s+card\b",
            r"\bgift\s+card\b",
            # Add more as needed
        ]

        for pattern in package_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(0).strip()

        # Fallback: look for "fiber" + word
        fiber_match = re.search(r"\bfiber\s+(\w+)", query_lower)
        if fiber_match:
            return f"fiber {fiber_match.group(1)}"

        return None

    def debug_content_types(self):
        """Debug method to see what content types exist in your index"""
        try:
            # Get some sample nodes
            retriever = VectorIndexRetriever(index=self.index, similarity_top_k=50)
            nodes = retriever.retrieve("Fiber Extra")

            content_types = {}
            for node in nodes:
                try:
                    node_obj = getattr(node, "node", node)
                    meta = getattr(node_obj, "metadata", {}) or {}
                    content_type = meta.get("content_type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1

                    # Log first few examples
                    if content_types[content_type] <= 2:
                        logger.info(
                            f"Content type '{content_type}': {getattr(node_obj, 'text', '')[:100]}..."
                        )

                except Exception as e:
                    continue

            logger.info(f"Content types found: {content_types}")
            return content_types

        except Exception as e:
            logger.error(f"Debug failed: {e}")
            return {}

    def _format_specific_answer(self, match: Dict, query: str) -> str:
        """Format answer for specific queries (like 'what is the code for X')"""
        text = match["text"]
        meta = match["metadata"]
        source = meta.get("source", "Unknown")

        # Extract the specific information requested
        specific_info = self._extract_requested_info(text, query)

        if specific_info:
            response = f"## Answer\n{specific_info}\n\n"
            response += f"## Complete Information\n{text}\n\n"
            response += f"## Source\n{source}"
        else:
            response = f"## Answer\n{text}\n\n"
            response += f"## Source\n{source}"

        return response

    def _format_general_answer(self, matches: List[Dict], query: str) -> str:
        """Format answer for general queries that might need multiple results"""
        if len(matches) == 1:
            return self._format_specific_answer(matches[0], query)

        response_parts = [f"## Answer\nFound {len(matches)} relevant results:\n"]

        for i, match in enumerate(matches, 1):
            text = match["text"]
            meta = match["metadata"]
            source = meta.get("source", "Unknown")

            # Extract key info from each match
            key_info = self._extract_key_info_summary(text)

            response_parts.append(f"**{i}. {source}:**")
            if key_info:
                response_parts.append(f"{key_info}")
            else:
                response_parts.append(f"{text[:200]}...")
            response_parts.append("")

        return "\n".join(response_parts)

    def enhance_answer(self, raw_answer: str, sources: List[Dict]) -> str:
        """Minimal enhancement - just ensure we have sources if missing"""

        if not raw_answer or raw_answer.strip().lower() in {"", "empty response"}:
            return "No relevant information found in the knowledge base."

        # Add source references if missing
        if (
            sources
            and "## References" not in raw_answer
            and "## Source" not in raw_answer
        ):
            source_names = [s["source"] for s in sources if s["source"] != "unknown"]
            if source_names:
                raw_answer += f"\n\n## References\n" + "\n".join(
                    [f"• {name}" for name in source_names[:3]]
                )

        return raw_answer

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query to choose appropriate response structure"""
        query_lower = query.lower()

        # Process/How-to queries
        if any(
            phrase in query_lower
            for phrase in [
                "how to",
                "how do i",
                "steps to",
                "process for",
                "procedure",
                "guide",
                "instructions",
                "way to",
            ]
        ):
            return "process"

        # Comparison queries
        elif any(
            phrase in query_lower
            for phrase in [
                "vs",
                "versus",
                "compare",
                "comparison",
                "difference between",
                "better than",
                "which is",
                "should i choose",
            ]
        ):
            return "comparison"

        # Factual/Definition queries
        elif any(
            phrase in query_lower
            for phrase in [
                "what is",
                "what are",
                "define",
                "definition",
                "meaning of",
                "explain",
                "describe",
            ]
        ):
            return "factual"

        # Specific lookup queries (your existing table logic)
        elif any(
            phrase in query_lower
            for phrase in [
                "what is the",
                "show me the",
                "give me the",
                "code for",
                "price for",
                "cost of",
                "fee for",
            ]
        ):
            return "lookup"

        # List/Options queries
        elif any(
            phrase in query_lower
            for phrase in [
                "list",
                "options",
                "available",
                "choices",
                "types of",
                "kinds of",
            ]
        ):
            return "options"

        return "general"

    def _get_enhanced_prompt_for_query(
        self, query: str, has_structured_data: bool = False
    ) -> str:
        """Always return the user-friendly prompt template"""
        return RAG_PROMPT_TEMPLATE

    def _parse_table_row_to_dict(self, text: str) -> Dict[str, str]:
        """Enhanced table row parsing with better value extraction"""
        parsed = {}

        try:
            # Handle pipe-delimited markdown tables
            if "|" in text:
                lines = [l.strip() for l in text.split("\n") if l.strip() and "|" in l]

                # Try to find header row
                headers = None
                data_rows = []

                for line in lines:
                    cells = [c.strip() for c in line.split("|") if c.strip()]
                    if not cells:
                        continue

                    # Skip separator rows
                    if all(set(c) <= {"-", " ", ":"} for c in cells):
                        continue

                    # First substantial row is likely headers
                    if headers is None and len(cells) >= 2:
                        headers = cells
                    elif headers and len(cells) >= len(headers):
                        data_rows.append(cells)

                # If we have headers and data, pair them
                if headers and data_rows:
                    for row in data_rows:
                        for i, header in enumerate(headers):
                            if i < len(row):
                                parsed[header] = row[i]

            # Handle colon-delimited key-value pairs
            for line in text.split("\n"):
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if key and val:
                            parsed[key] = val

            # Extract BD amounts if no explicit keys found
            if not any(
                k.lower() in ["rental", "price", "cost", "amount", "fee"]
                for k in parsed.keys()
            ):
                bd_matches = re.findall(r"BD\s*(\d+[,.]?\d*)", text, re.IGNORECASE)
                if bd_matches:
                    parsed["Amount"] = f"BD {bd_matches[0]}"

        except Exception as e:
            logger.debug(f"Table row parsing error: {e}")

        return parsed

    def _detect_exact_lookup_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query intent detection with strong conditional awareness"""
        query_lower = query.lower()

        exact_indicators = [
            "what is the",
            "what's the",
            "show me the",
            "give me the",
            "tell me the",
            "find the",
            "get the",
            "display the",
            "code for",
            "price for",
            "value of",
            "amount for",
            "rate for",
            "fee for",
            "cost of",
            "rental for",
            "with fiber",
            "for fiber",
            "router with",
            "gaming router",
            "discounted to",
            "discount with",
            "installment amount",
            "vat amount",
            "total amount",
            "Monthly allowance",
            "Total Monthly",
            "Voice Service",
        ]

        # NEW: Detect conditional phrases with entity
        conditional_patterns = [
            r"(\w+\s+\w+)\s+with\s+(\w+\s+\w+)",  # "Gaming Router with Fiber Extra"
            r"(\w+\s+\w+)\s+for\s+(\w+\s+\w+)",  # "price for Fiber Extra"
            r"(\w+\s+\w+)\s+on\s+(\w+\s+\w+)",  # "discount on Fiber Extra"
            r"with\s+(\w+\s+\w+)",  # "with Fiber Extra"
        ]

        conditional_entity = None
        base_entity = None

        for pattern in conditional_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if match.lastindex >= 2:
                    base_entity = match.group(1).strip()
                    conditional_entity = match.group(2).strip()
                else:
                    conditional_entity = match.group(1).strip()
                break

        is_conditional = conditional_entity is not None

        value_types = {
            "price": [
                "price",
                "cost",
                "amount",
                "bd",
                "fee",
                "rate",
                "rental",
                "charge",
                "payment",
                "vat amount",
                "monthly",
                "installment",
                "total",
                "total monthly",
                "Monthly Rental",
            ],
            "code": ["code", "id", "flag", "ref", "INST", "tabs"],
            "name": [
                "name",
                "title",
                "description",
                "called",
                "device",
                "router",
                "package",
            ],
            "specification": [
                "spec",
                "feature",
                "detail",
                "requirement",
                "speed",
                "data",
                "download",
                "upload",
                "allowance",
                "minutes",
            ],
            "device": ["router", "device", "equipment", "xr1000", "nighthawk", "mesh"],
        }

        is_exact = any(ind in query_lower for ind in exact_indicators)

        detected_type = "general"
        max_matches = 0

        for vtype, keywords in value_types.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > max_matches:
                max_matches = matches
                detected_type = vtype

        # ENHANCED: Much higher confidence for conditional queries
        confidence = 0.9 if is_exact else 0.5
        if max_matches >= 2:
            confidence += 0.1
        if is_conditional:  # Strong boost for conditional queries
            confidence = min(1.0, confidence + 0.25)

        return {
            "is_exact_lookup": is_exact,
            "value_type": detected_type,
            "needs_table_search": is_exact or max_matches >= 2 or is_conditional,
            "confidence": min(1.0, confidence),
            "is_conditional": is_conditional,
            "conditional_entity": conditional_entity,
            "base_entity": base_entity,
        }

    def _extract_column_entity(self, query_lower: str) -> Optional[str]:
        """Extract the column entity (field/attribute name) from the query"""

        column_patterns = [
            # Connection related
            (r"\bconnection\s+fee[s]?\b", "connection fee"),
            (r"\bconnection\s+charge[s]?\b", "connection charge"),
            (r"\bsetup\s+fee[s]?\b", "setup fee"),
            (r"\binstallation\s+fee[s]?\b", "installation fee"),
            (r"\bone\s+time\s+fee[s]?\b", "one time fee"),
            (r"\binitial\s+fee[s]?\b", "initial fee"),
            # Code related
            (r"\bextra\s+data\s+sim\s+code\b", "extra data sim code"),
            (r"\bsim\s+code\b", "sim code"),
            (r"\bpackage\s+code\b", "package code"),
            (r"\btabs\s+flag\b", "tabs flag"),
            (r"\bmrs\s+flag\b", "mrs flag"),
            (r"\bflag\b", "flag"),
            (r"\bcode\b", "code"),
            # Pricing
            (r"\bmonthly\s+rental\b", "monthly rental"),
            (r"\btotal\s+monthly\b", "total monthly"),
            (r"\bvat\s+amount\b", "vat amount"),
            (r"\binstallment\s+amount\b", "installment amount"),
            (r"\bdevice\s+price\b", "device price"),
            (r"\bup\s+front\b", "up front"),
            (r"\bupfront\b", "upfront"),
            # Technical specs
            (r"\bdownload\s+speed\b", "download speed"),
            (r"\bupload\s+speed\b", "upload speed"),
            (r"\bdata\s+allowance\b", "data allowance"),
            (r"\bmonthly\s+allowance\b", "monthly allowance"),
            (r"\bvoice\s+service\b", "voice service"),
            (r"\bcommitment\b", "commitment"),
            # Add more patterns as needed
        ]

        for pattern, entity_name in column_patterns:
            if re.search(pattern, query_lower):
                return entity_name

        return None

    def _find_table_intersection(
        self, key_values: Dict[str, str], row_entity: str, column_entity: str
    ) -> Optional[str]:
        """Find the intersection value in a table structure"""

        logger.info(
            f"Looking for intersection: row='{row_entity}' × column='{column_entity}'"
        )
        logger.info(f"Available key_values: {list(key_values.keys())}")

        # Strategy 1: Direct key matching - key contains both row and column info
        for key, value in key_values.items():
            key_lower = key.lower()

            # Check if key contains both entities
            has_row = any(word in key_lower for word in row_entity.split())
            has_column = any(word in key_lower for word in column_entity.split())

            if has_row and has_column and value and str(value).strip() != "-":
                logger.info(f"Direct intersection found: {key} = {value}")
                return str(value).strip()

        # Strategy 2: Key represents column, look for row-specific value
        column_key = None

        # Find the key that represents our target column
        column_words = column_entity.split()
        for key, value in key_values.items():
            key_lower = key.lower()

            # Check if this key represents the column we want
            if any(word in key_lower for word in column_words):
                column_key = key
                break

        if column_key:
            # Now look for row-specific information in the same structure
            row_words = row_entity.split()

            # Check if the column key contains row info, or look for separate row keys
            for key, value in key_values.items():
                key_lower = key.lower()

                # If this key contains our target row AND is related to our column
                if (
                    any(word in key_lower for word in row_words)
                    and any(word in key_lower for word in column_entity.split())
                    and value
                    and str(value).strip() != "-"
                ):
                    logger.info(f"Row-column intersection found: {key} = {value}")
                    return str(value).strip()

        # Strategy 3: Table structure analysis - Column header pattern matching
        # Sometimes tables are structured as: "Fiber Extra FTTHEXTRA" where the key contains the package name

        # Look for keys that contain the row entity (package name)
        row_keys = []
        for key, value in key_values.items():
            key_lower = key.lower()
            if (
                any(word in key_lower for word in row_entity.split())
                and value
                and str(value).strip() != "-"
            ):
                row_keys.append((key, value))

        # If we found row-related keys, check if any relate to our column
        for key, value in row_keys:
            key_lower = key.lower()
            if any(word in key_lower for word in column_entity.split()):
                logger.info(f"Package-column intersection found: {key} = {value}")
                return str(value).strip()

        # Strategy 4: Cross-table lookup - sometimes the table is transposed
        # Look for column header that matches our column entity
        header_key = None
        for key in key_values.keys():
            key_lower = key.lower()
            # More flexible matching for column headers
            if column_entity.replace(" ", "") in key_lower.replace(" ", "") or any(
                word in key_lower for word in column_entity.split()
            ):
                header_key = key
                break

        if header_key:
            # Look for keys that contain the row entity
            for key, value in key_values.items():
                key_lower = key.lower()
                if (
                    any(word in key_lower for word in row_entity.split())
                    and key != header_key
                    and value
                    and str(value).strip() != "-"
                ):
                    logger.info(f"Header-based intersection found: {key} = {value}")
                    return str(value).strip()

        # Strategy 5: Semantic matching - look for similar meanings
        # For connection fees, also look for setup, installation, etc.
        if "connection" in column_entity or "fee" in column_entity:
            connection_synonyms = [
                "setup",
                "installation",
                "activation",
                "initial",
                "onetime",
                "connection",
            ]
            fee_synonyms = ["fee", "charge", "cost", "amount", "price"]

            for key, value in key_values.items():
                key_lower = key.lower()
                has_connection_term = any(
                    syn in key_lower for syn in connection_synonyms
                )
                has_fee_term = any(syn in key_lower for syn in fee_synonyms)
                has_row_entity = any(word in key_lower for word in row_entity.split())

                if (
                    has_connection_term
                    and has_fee_term
                    and value
                    and str(value).strip() != "-"
                ):
                    logger.info(f"Semantic connection fee match: {key} = {value}")
                    return str(value).strip()

        logger.info("No intersection found")
        return None

    def _execute_table_lookup(
        self, query: str, intent: Dict[str, Any]
    ) -> Optional[str]:
        """Enhanced exact table lookup with LLM generation for friendly responses"""
        try:
            if not self.index:
                logger.warning("No index available for table lookup")
                return None

            table_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=25,
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="content_type", value="table_row")]
                ),
            )

            # Try multiple search variations
            search_queries = [query, f"table row: {query}", f"table data: {query}"]

            all_table_rows = []
            query_lower = query.lower()

            for search_query in search_queries:
                try:
                    nodes = table_retriever.retrieve(search_query)
                    logger.info(
                        f"Retrieved {len(nodes)} nodes for query: '{search_query}'"
                    )
                except Exception as e:
                    logger.debug(f"Retriever failed for '{search_query}': {e}")
                    continue

                for node in nodes:
                    try:
                        node_obj = getattr(node, "node", node)
                        meta = getattr(node_obj, "metadata", {}) or {}

                        # Only process table content
                        content_type = meta.get("content_type", "")
                        if content_type not in ["table_row", "table_md", "table_html"]:
                            continue

                        text = getattr(node_obj, "text", "") or ""
                        if not text or len(text.strip()) < 5:
                            continue

                        relevance_score = self._calculate_table_relevance(
                            text, query_lower, intent
                        )
                        node_score = getattr(node, "score", 0) or 0

                        all_table_rows.append(
                            {
                                "text": text,
                                "metadata": meta,
                                "relevance_score": relevance_score,
                                "node_score": node_score,
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Error processing node: {e}")
                        continue

            # FALLBACK: If no results with filter, try without filter
            if not all_table_rows:
                logger.info("No results with filter, trying without filter...")
                fallback_retriever = VectorIndexRetriever(
                    index=self.index, similarity_top_k=50
                )

                try:
                    nodes = fallback_retriever.retrieve(query)
                    logger.info(f"Fallback retrieved {len(nodes)} nodes")

                    # Filter for table rows manually
                    for node in nodes:
                        try:
                            node_obj = getattr(node, "node", node)
                            meta = getattr(node_obj, "metadata", {}) or {}

                            if meta.get("content_type") == "table_row":
                                text = getattr(node_obj, "text", "") or ""
                                if text:
                                    relevance_score = self._calculate_table_relevance(
                                        text, query_lower, intent
                                    )
                                    all_table_rows.append(
                                        {
                                            "text": text,
                                            "metadata": meta,
                                            "relevance_score": relevance_score,
                                            "node_score": getattr(node, "score", 0),
                                        }
                                    )

                        except Exception as e:
                            continue

                except Exception as e:
                    logger.error(f"Fallback retrieval failed: {e}")

            if not all_table_rows:
                logger.info("No table rows retrieved")
                return None

            parsed_rows = []
            for row in all_table_rows:
                parsed = self._parse_table_row_to_dict(row["text"])
                if not parsed:
                    continue

                parsed["_raw_text"] = row["text"]
                parsed["_relevance"] = row.get("relevance_score", 0)
                parsed["_node_score"] = row.get("node_score", 0)
                parsed["_metadata"] = row.get("metadata", {})
                parsed_rows.append(parsed)

            if not parsed_rows:
                logger.info("No parseable table rows found")
                return None

            parsed_rows = self._filter_tables_by_condition(parsed_rows, intent)

            # Filter candidates
            entity_patterns = [
                r"\b(mesh\s*router)\b",
                r"\b(al\s*zain\+?)\b",
                r"\b(fiber\s*\d+)\b",
                r"\b([a-z]+\s*[0-9]+[a-z]*)\b",
            ]

            query_entity = None
            for pattern in entity_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    query_entity = match.group(1).lower()
                    break

            if query_entity:
                candidates = []
                for row in parsed_rows:
                    row_text = row.get("_raw_text", "").lower()
                    if query_entity in row_text:
                        candidates.append(row)
                if not candidates:
                    candidates = parsed_rows
            else:
                candidates = parsed_rows

            # Sort by combined relevance
            candidates.sort(
                key=lambda r: (r.get("_relevance", 0) * 2 + r.get("_node_score", 0)),
                reverse=True,
            )

            if candidates:
                best = candidates[0]
                logger.info(
                    f"Selected best candidate with relevance={best.get('_relevance'):.2f}, node_score={best.get('_node_score'):.2f}"
                )

                # Get the structured data
                key_values = best.get("_metadata", {}).get("key_values", {})
                raw_text = best.get("_raw_text", "")
                source = best.get("_metadata", {}).get("source", "table_lookup")

                # Also get parsed data from the table row
                parsed_data = {
                    k: v for k, v in best.items() if not k.startswith("_") and v
                }

                # Combine key_values and parsed_data for comprehensive data
                all_data = {**parsed_data, **key_values}

                logger.info(f"TABLE DATA FOR LLM: {all_data}")

                # CREATE LLM PROMPT FOR TABLE DATA
                table_prompt = f"""
    You are a helpful assistant providing clear, user-friendly answers based on structured table data.

    **USER QUESTION**: {query}

    **STRUCTURED DATA FOUND**:
    Raw table text: {raw_text}

    Key information available:
    {chr(10).join([f"• {k}: {v}" for k, v in all_data.items() if v and str(v).strip() != "-"])}

    **INSTRUCTIONS**:
    1. Answer the user's question naturally and conversationally
    2. For "best" questions: Start with "The best [item] is the [name]" followed by key features
    3. For specific value questions: Give direct answers like "The [item] is [value]"
    4. Present information in a user-friendly way with bullet points for features
    5. Keep all exact values (prices, codes, numbers) unchanged from the data
    6. Don't use raw field names - translate them to natural language:
       - "DEVICE_DESCRIPTION" → "Device" or product name
       - "Installment Amount" → "Monthly installment"
       - "VAT amount" → "VAT amount"
       - "INST_CODE" → "Product code"
       - "TABS Flag" → "System flag"
       - "Extra SIM code" → "Extra SIM code"
    7. Structure your response clearly with ## Answer header
    8. Focus on what the user specifically asked about

    **EXAMPLES**:
    - For "What's the best gaming router?": "The best gaming router available is the **[Product Name]**. Here's what makes it great: • **Monthly installment**: [amount] • **Product code**: [code]"
    - For "What is the price?": "The monthly installment is **[amount]**."

    **INSTRUCTIONS**:
    1. Identify the EXACT field type the user wants
    2. Find that specific field in the data
    3. Return ONLY that value with clear labeling
    4. Do not confuse different code types

    **YOUR RESPONSE**:
    """

                try:
                    # Generate friendly response using LLM
                    completion = Settings.llm.complete(
                        table_prompt, timeout=LLM_COMPLETE_TIMEOUT
                    )
                    friendly_response = completion.text.strip()

                    if friendly_response and len(friendly_response) > 50:
                        # Add source information if not present
                        if (
                            "## Source" not in friendly_response
                            and "Source:" not in friendly_response
                        ):
                            friendly_response += f"\n\n## Source\n*{source}*"

                        logger.info("Successfully generated friendly table response")
                        return friendly_response
                    else:
                        logger.warning(
                            "LLM generated empty/short response for table data"
                        )

                except Exception as e:
                    logger.error(f"LLM generation failed for table lookup: {e}")

                # FALLBACK: If LLM fails, create a structured fallback
                return self._create_table_fallback_response(query, all_data, source)

            return None

        except Exception as e:
            logger.error(f"Table lookup error: {e}")
            return None

    def _create_table_fallback_response(
        self, query: str, key_values: Dict, source: str
    ) -> str:
        """Create a friendly fallback response when LLM generation fails"""

        query_lower = query.lower()

        # Check if it's a recommendation query
        is_recommendation = any(
            word in query_lower for word in ["best", "recommend", "which", "top"]
        )

        response_parts = []

        if is_recommendation:
            device_name = (
                key_values.get("DEVICE_DESCRIPTION")
                or key_values.get("Device")
                or key_values.get("device_name")
                or key_values.get("Name")
            )
            if device_name:
                response_parts.append(
                    f"## Answer\nThe best option available is the **{device_name}**.\n"
                )
                response_parts.append("**Key Details:**")
            else:
                response_parts.append("## Answer\nHere are the details I found:\n")
        else:
            response_parts.append("## Answer")

        # Convert technical field names to user-friendly labels
        field_mapping = {
            "DEVICE_DESCRIPTION": "Device",
            "Installment Amount": "Monthly installment",
            "VAT amount": "VAT amount",
            "INST_CODE": "Product code",
            "TABS Flag": "System flag",
            "Installment Code": "Installation code",
            "device_name": "Device",
            "price": "Price",
            "code": "Code",
        }

        # Add key information in a friendly format
        for key, value in key_values.items():
            if value and str(value).strip() not in ["-", "", "None"]:
                friendly_key = field_mapping.get(key, key.replace("_", " ").title())
                response_parts.append(f"• **{friendly_key}**: {value}")

        response_parts.append(f"\n## Source\n*{source}*")
        response_parts.append(
            "\nIf you need anything else or have follow-up questions, I'm here to assist you!"
        )

        return "\n".join(response_parts)

    def _create_table_fallback_response(
        self, query: str, key_values: Dict, source: str
    ) -> str:
        """Create a friendly fallback response when LLM generation fails"""

        query_lower = query.lower()

        # Check if it's a recommendation query
        is_recommendation = any(
            word in query_lower for word in ["best", "recommend", "which", "top"]
        )

        response_parts = []

        if is_recommendation:
            device_name = key_values.get("DEVICE_DESCRIPTION") or key_values.get(
                "Device"
            )
            if device_name:
                response_parts.append(
                    f"## Answer\nThe best option available is the **{device_name}**.\n"
                )
                response_parts.append("**Key Details:**")
            else:
                response_parts.append("## Answer\nHere are the details I found:\n")
        else:
            response_parts.append("## Answer")

        # Convert technical field names to user-friendly labels
        field_mapping = {
            "DEVICE_DESCRIPTION": "Device",
            "Installment Amount": "Monthly installment",
            "VAT amount": "VAT amount",
            "INST_CODE": "Product code",
            "TABS Flag": "System flag",
            "Installment Code": "Installation code",
        }

        # Add key information in a friendly format
        for key, value in key_values.items():
            if value and str(value).strip() != "-":
                friendly_key = field_mapping.get(key, key.replace("_", " ").title())
                response_parts.append(f"• **{friendly_key}**: {value}")

        response_parts.append(f"\n## Source\n*{source}*")

        return "\n".join(response_parts)

    def _calculate_table_relevance(
        self, text: str, query_lower: str, intent: Dict
    ) -> float:
        """
        Comprehensive table relevance scoring with enhanced flag/code detection and table intersection awareness.
        """
        text_lower = text.lower()
        score = 0.0

        # Parse structured key-value pairs
        data_pairs = {}
        lines = text.split("\n")
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key_clean = parts[0].strip().lower()
                    value_clean = parts[1].strip().lower()
                    data_pairs[key_clean] = value_clean

        # Extract query terms (filter stopwords)
        stopwords = {
            "the",
            "for",
            "what",
            "how",
            "much",
            "is",
            "are",
            "with",
            "and",
            "or",
            "of",
            "to",
            "in",
            "a",
            "an",
        }
        query_words = [
            w for w in query_lower.split() if len(w) > 2 and w not in stopwords
        ]

        # NEW: Add intersection bonus scoring
        row_entity = self._extract_row_entity(query_lower)
        column_entity = self._extract_column_entity(query_lower)

        if row_entity and column_entity:
            # Massive bonus if we can find the intersection
            intersection_value = self._find_table_intersection(
                data_pairs, row_entity, column_entity
            )

            if intersection_value:
                score += 100.0  # Huge bonus for perfect intersection
                logger.info(f"Perfect table intersection found, adding 100 points")

            # Bonus for having both entities present even if no intersection
            has_row = row_entity in text_lower
            has_column = column_entity in text_lower

            if has_row and has_column:
                score += 50.0  # Both entities present
            elif has_row or has_column:
                score += 25.0  # One entity present

        # TABS Flag specific handling
        if "tabs flag" in query_lower or "tabs" in query_lower:
            # Look for TABS or MRS flag information
            tabs_found = False
            for key, value in data_pairs.items():
                if any(indicator in key for indicator in ["tabs", "mrs flag", "flag"]):
                    if "flag" in key:
                        score += 150.0  # Massive boost for direct flag matches
                        tabs_found = True
                        logger.debug(f"Found TABS/MRS Flag: {key} = {value}")
                        break

            if not tabs_found:
                # Check if any value looks like a flag code
                for key, value in data_pairs.items():
                    if "flg" in value or any(
                        pattern in value for pattern in ["flag", "FLG"]
                    ):
                        score += 100.0  # High boost for flag-like values
                        logger.debug(f"Found flag-like value: {key} = {value}")
                        break

        # General flag/code queries
        elif any(term in query_lower for term in ["flag", "code"]):
            code_indicators = ["code", "flag", "mrs", "tabs", "id", "ref", "flg"]
            code_found = False

            for key, value in data_pairs.items():
                if any(indicator in key for indicator in code_indicators):
                    # Calculate word matches for relevance
                    key_value_text = f"{key} {value}"
                    word_matches = sum(
                        1 for word in query_words if word in key_value_text
                    )
                    if word_matches > 0:
                        score += 50.0 + (word_matches * 10.0)  # Base + word match bonus
                        code_found = True
                        logger.debug(f"Found code/flag match: {key} = {value}")

            # Additional boost for alphanumeric codes in values
            if not code_found:
                import re

                for key, value in data_pairs.items():
                    # Look for patterns like IPAD64FLG, POSTPLN12P, etc.
                    if re.search(r"[A-Z]{2,}[0-9]+[A-Z]*", value.upper()):
                        score += 40.0
                        logger.debug(f"Found alphanumeric code pattern: {value}")
                        break

        # Extract device from query
        device_keywords = [
            "ipad",
            "iphone",
            "samsung",
            "honor",
            "router",
            "device",
            "phone",
        ]
        query_device = None
        device_variations = []

        for device in device_keywords:
            if device in query_lower:
                query_device = device
                # Create variations for better matching
                if device == "ipad":
                    device_variations = ["ipad", "ipad", "tablet"]
                elif device == "iphone":
                    device_variations = ["iphone", "phone", "mobile"]
                elif device == "samsung":
                    device_variations = ["samsung", "galaxy"]
                else:
                    device_variations = [device]
                break

        if query_device:
            device_match_score = 0

            # Check device-specific keys
            device_keys = [
                "device",
                "item name",
                "offer description",
                "description",
                "name",
            ]
            for key, value in data_pairs.items():
                if any(device_key in key for device_key in device_keys):
                    # Check for any device variation in the value
                    for variation in device_variations:
                        if variation in value:
                            device_match_score += 60.0  # High device match bonus
                            logger.debug(f"Device match found: {variation} in {key}")
                            break

                    # Additional scoring for exact device model matches
                    device_words = set(value.split())
                    query_device_matches = len(
                        device_words.intersection(set(query_words))
                    )
                    if query_device_matches > 0:
                        device_match_score += query_device_matches * 20.0

            score += device_match_score

        if intent.get("is_conditional", False):
            conditional_entity = intent.get("conditional_entity", "")
            base_entity = intent.get("base_entity", "")

            # Check context for conditional phrases
            if "context:" in text_lower:
                context_text = ""
                for line in lines:
                    if line.lower().startswith("context:"):
                        context_text = line.lower()
                        break

                # MASSIVE bonus for exact conditional matches
                has_base = base_entity in context_text if base_entity else False
                has_condition = (
                    conditional_entity in context_text if conditional_entity else False
                )

                if has_base and has_condition:
                    score += 80.0  # Perfect conditional match
                    logger.debug(
                        f"Perfect conditional match: base='{base_entity}' + condition='{conditional_entity}'"
                    )
                elif has_condition:
                    score += 50.0  # Condition match
                    logger.debug(f"Conditional match: '{conditional_entity}'")
                elif has_base:
                    score += 25.0  # Base entity match

            # Also check in the entire text
            if conditional_entity and conditional_entity in text_lower:
                score += 30.0
            if base_entity and base_entity in text_lower:
                score += 20.0

        # Exact query phrase match
        if query_lower in text_lower:
            score += 40.0
            logger.debug("Exact query phrase match found")

        # Advanced word coverage scoring
        if query_words:
            text_words = set(text_lower.split())
            word_overlap = len(set(query_words).intersection(text_words))
            word_coverage = word_overlap / len(query_words)

            # Scaled word coverage bonus
            score += word_coverage * 25.0

            if word_coverage >= 0.8:  # Very high coverage
                score += 15.0
            elif word_coverage >= 0.5:  # Good coverage
                score += 8.0

        value_type = intent.get("value_type", "general")

        if value_type == "price":
            price_indicators = [
                "bd",
                "price",
                "cost",
                "fee",
                "rental",
                "monthly",
                "amount",
                "vat",
                "charge",
            ]
            price_score = 0

            for key, value in data_pairs.items():
                key_value_text = f"{key} {value}"
                price_matches = sum(
                    1 for indicator in price_indicators if indicator in key_value_text
                )
                if price_matches > 0:
                    word_matches = sum(
                        1 for word in query_words if word in key_value_text
                    )
                    if word_matches > 0:
                        price_score += 15.0 * price_matches * word_matches

            score += price_score

        elif value_type == "device":
            device_indicators = [
                "device",
                "router",
                "equipment",
                "description",
                "item name",
            ]
            for key, value in data_pairs.items():
                if any(indicator in key for indicator in device_indicators):
                    word_matches = sum(
                        1 for word in query_words if word in f"{key} {value}"
                    )
                    if word_matches > 0:
                        score += 20.0 * word_matches

        elif value_type == "specification":
            spec_indicators = [
                "speed",
                "data",
                "allowance",
                "gb",
                "mbps",
                "minutes",
                "download",
                "upload",
            ]
            for key, value in data_pairs.items():
                key_value_text = f"{key} {value}"
                if any(indicator in key_value_text for indicator in spec_indicators):
                    word_matches = sum(
                        1 for word in query_words if word in key_value_text
                    )
                    if word_matches > 0:
                        score += 12.0 * word_matches

        import re

        # BD amounts matching
        query_bd_amounts = re.findall(r"bd\s*(\d+[,.]?\d*)", query_lower)
        text_bd_amounts = re.findall(r"bd\s*(\d+[,.]?\d*)", text_lower)

        if query_bd_amounts and text_bd_amounts:
            for q_amt in query_bd_amounts:
                q_normalized = q_amt.replace(",", "").replace(".", "")
                for t_amt in text_bd_amounts:
                    t_normalized = t_amt.replace(",", "").replace(".", "")
                    if q_normalized == t_normalized:
                        score += 25.0
                        logger.debug(f"BD amount match: {q_amt}")

        # Code pattern matching
        query_codes = re.findall(r"\b([A-Z]{3,}[A-Z0-9]+)\b", query_lower.upper())
        text_codes = re.findall(r"\b([A-Z]{3,}[A-Z0-9]+)\b", text_lower.upper())

        for q_code in query_codes:
            if q_code in text_codes:
                score += 30.0
                logger.debug(f"Code match: {q_code}")

        # Speed matching
        query_speeds = re.findall(r"(\d+)\s*mbps", query_lower)
        text_speeds = re.findall(r"(\d+)\s*mbps", text_lower)

        for q_speed in query_speeds:
            if q_speed in text_speeds:
                score += 20.0
                logger.debug(f"Speed match: {q_speed}mbps")

        # Context match bonus (non-conditional)
        if "context:" in text_lower and not intent.get("is_conditional"):
            context_text = ""
            for line in lines:
                if line.lower().startswith("context:"):
                    context_text = line.lower()
                    break

            if context_text:
                context_matches = sum(1 for word in query_words if word in context_text)
                if context_matches > 0:
                    score += context_matches * 10.0

        # Field type indicators
        field_labels = ["price", "code", "name", "spec", "device", "flag"]
        for label in field_labels:
            if f"**{label}" in text_lower or f"({label})" in text_lower:
                score += 5.0

        # Table structure quality
        if len(data_pairs) >= 3:  # Rich data
            score += 8.0
        elif len(data_pairs) >= 2:  # Decent data
            score += 4.0

        # Boost for exact lookup queries
        if intent.get("is_exact_lookup", False):
            exact_confidence = intent.get("confidence", 0)
            score += exact_confidence * 15.0

        # Boost for high-confidence intent detection
        if intent.get("confidence", 0) > 0.8:
            score += 10.0

        # Penalty for very low word coverage
        if query_words:
            text_words = set(text_lower.split())
            coverage = sum(1 for w in query_words if w in text_words) / len(query_words)
            if coverage < 0.2:  # Very poor coverage
                score *= 0.7
                logger.debug("Applied low coverage penalty")

        # Bonus for compact, focused entries
        if intent.get("is_exact_lookup", False):
            line_count = len([l for l in lines if l.strip()])
            if 5 <= line_count <= 15:
                score += 8.0  # Good focused content

        # Bonus for multiple relevant field types
        field_type_count = 0
        if any("price" in k or "bd" in v for k, v in data_pairs.items()):
            field_type_count += 1
        if any("code" in k or "flag" in k for k, v in data_pairs.items()):
            field_type_count += 1
        if any("device" in k or "description" in k for k, v in data_pairs.items()):
            field_type_count += 1

        if field_type_count >= 2:
            score += 6.0

        final_score = max(0.0, score)

        # Log high-scoring matches for debugging
        if final_score > 50.0:
            logger.debug(
                f"High relevance score: {final_score:.1f} for query: {query_lower[:50]}..."
            )

        return final_score

    def _filter_tables_by_condition(
        self, candidates: List[Dict], intent: Dict
    ) -> List[Dict]:
        """Filter and prioritize tables based on conditional requirements"""

        if not intent.get("is_conditional", False):
            return candidates  # No filtering needed

        conditional_entity = intent.get("conditional_entity", "")

        # Separate conditional from non-conditional tables
        conditional_tables = []
        general_tables = []

        for row in candidates:
            text = row.get("_raw_text", "").lower()
            context = row.get("_metadata", {}).get("table_context", "").lower()

            # Check if this table is specifically for the conditional entity
            is_conditional_table = (
                conditional_entity in context
                or conditional_entity in text
                or any(
                    phrase in context
                    for phrase in ["discount", "discounted to", "special price"]
                )
            )

            if is_conditional_table:
                conditional_tables.append(row)
            else:
                general_tables.append(row)

        # PREFER conditional tables strongly
        if conditional_tables:
            logger.info(
                f"Found {len(conditional_tables)} conditional tables, prioritizing them"
            )
            # Boost scores of conditional tables
            for row in conditional_tables:
                row["_relevance"] = row.get("_relevance", 0) + 25.0

            # Return conditional tables first, then general
            return conditional_tables + general_tables

        # No conditional tables found, return all
        logger.warning("No conditional-specific tables found")
        return candidates

    def _is_specific_query(self, query: str) -> bool:
        """Determine if this is asking for a specific piece of information"""
        specific_indicators = [
            # Direct questions
            "what is the",
            "what's the",
            "show me the",
            "give me the",
            "tell me the",
            "find the",
            "get the",
            "display the",
            # Specific value requests
            "code for",
            "price for",
            "cost of",
            "fee for",
            "rental for",
            "name of",
            "speed of",
            "allowance for",
            "date for",
            "year of",
            "service for",
            "device for",
            "plan for",
            "package for",
            "rate for",
            "amount for",
            # How much/many questions
            "how much",
            "how many",
            "what cost",
            "what price",
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in specific_indicators)

    def _extract_requested_info(self, text: str, query: str) -> str:
        """Extract the specific piece of information requested - GENERIC VERSION"""
        query_lower = query.lower()
        lines = text.split("\n")

        # Parse all key-value pairs
        data = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key_clean = key.strip().lower()
                value_clean = value.strip()
                data[key_clean] = value_clean

        # Extract query terms
        query_words = [word for word in query_lower.split() if len(word) > 2]

        # Strategy 1: Look for direct keyword matches in keys
        request_type_keywords = {
            "code": ["code", "id", "ref", "flag"],
            "price": ["price", "cost", "fee", "rental", "monthly", "amount", "charge"],
            "speed": ["speed", "bandwidth", "mbps", "gbps"],
            "data": ["allowance", "data", "gb", "mb", "limit"],
            "time": ["period", "duration", "months", "years", "date", "time"],
            "service": ["service", "plan", "package", "offering"],
            "device": ["device", "router", "equipment", "hardware"],
            "name": ["name", "title", "description", "called"],
        }

        # Determine what type of information is being requested
        for request_type, keywords in request_type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Look for matching data entries
                for key, value in data.items():
                    if any(keyword in key for keyword in keywords):
                        # Also check if the context matches the query
                        key_value_text = f"{key} {value}".lower()
                        if any(word in key_value_text for word in query_words):
                            return f"The {request_type} is: {value}"

        # Strategy 2: Find the key-value pair with the most query word matches
        best_match = None
        best_score = 0

        for key, value in data.items():
            key_value_text = f"{key} {value}".lower()
            score = sum(1 for word in query_words if word in key_value_text)

            if score > best_score:
                best_score = score
                best_match = (key, value)

        if best_match and best_score >= 1:
            key, value = best_match
            return f"{key}: {value}"

        # Strategy 3: Look for any line that contains most of the query words
        for line in lines:
            if ":" in line and line.strip():
                line_lower = line.lower()
                matches = sum(1 for word in query_words if word in line_lower)
                if matches >= len(query_words) * 0.5:  # At least half the query words
                    return line.strip()

        # Fallback: return the first meaningful key-value pair
        for line in lines:
            if ":" in line and line.strip():
                return line.strip()

        return ""

    def _extract_key_info_summary(self, text: str) -> str:
        """Extract key information for summary display - GENERIC VERSION"""
        lines = text.split("\n")
        key_info = []

        # Priority order for different types of information
        priority_keywords = [
            ["package", "plan", "service", "device", "name"],  # Names/identifiers
            ["code", "id", "ref", "flag"],  # Codes
            ["price", "cost", "fee", "rental", "monthly"],  # Pricing
            ["speed", "allowance", "data", "gb"],  # Technical specs
            ["period", "duration", "months", "years"],  # Time/duration
        ]

        # Collect all key-value pairs
        all_pairs = []
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key_clean = key.strip().lower()
                value_clean = value.strip()
                all_pairs.append((key_clean, value_clean, key.strip()))

        # Select most important information based on priority
        selected_info = []

        for priority_group in priority_keywords:
            for key_lower, value, key_original in all_pairs:
                if any(keyword in key_lower for keyword in priority_group):
                    if len(selected_info) < 3:  # Limit to 3 key pieces
                        selected_info.append(f"{key_original}: {value}")
                    break
            if len(selected_info) >= 3:
                break

        # If we don't have 3 items yet, add other important-looking items
        if len(selected_info) < 3:
            for key_lower, value, key_original in all_pairs:
                if f"{key_original}: {value}" not in selected_info:
                    # Skip very long values or obviously unimportant ones
                    if len(value) < 50 and not any(
                        skip in key_lower for skip in ["source", "page", "table"]
                    ):
                        selected_info.append(f"{key_original}: {value}")
                        if len(selected_info) >= 3:
                            break

        return " | ".join(selected_info)

    def _extract_values_from_matches(
        self, matches: List[Dict], intent: Dict[str, Any]
    ) -> Dict[str, str]:
        """Extract specific values from table matches based on query intent"""
        extracted = {}

        try:
            for match in matches[:2]:  # Check top 2 matches
                text = match["text"]

                # Extract monetary values (BD amounts)
                money_patterns = [
                    r"BD\s*(\d+(?:\.\d{2})?)",
                    r"(\d+(?:\.\d{2})?)\s*BD",
                    r"Price[:\s]*BD\s*(\d+(?:\.\d{2})?)",
                    r"Cost[:\s]*BD\s*(\d+(?:\.\d{2})?)",
                    r"Fee[:\s]*BD\s*(\d+(?:\.\d{2})?)",
                ]

                for pattern in money_patterns:
                    matches_found = re.findall(pattern, text, re.IGNORECASE)
                    if matches_found and "Price" not in extracted:
                        extracted["Price"] = f"BD {matches_found[0]}"
                        break

                # Extract codes/IDs
                code_patterns = [
                    r"Code[:\s]*([A-Z0-9]+)",
                    r"ID[:\s]*([A-Z0-9]+)",
                    r"Package[:\s]*([A-Z0-9]+)",
                    r"Plan[:\s]*([A-Z0-9]+)",
                    r"([A-Z]{2,}[0-9]{2,}[A-Z0-9]*)",  # Pattern like POSTPLN12P
                ]

                for pattern in code_patterns:
                    matches_found = re.findall(pattern, text, re.IGNORECASE)
                    if matches_found and "Code" not in extracted:
                        extracted["Code"] = matches_found[0]
                        break

                # Extract plan/package names
                name_patterns = [
                    r"Plan[:\s]*([^:\n]+?)(?:[:\n]|$)",
                    r"Package[:\s]*([^:\n]+?)(?:[:\n]|$)",
                    r"Service[:\s]*([^:\n]+?)(?:[:\n]|$)",
                ]

                for pattern in name_patterns:
                    matches_found = re.findall(pattern, text, re.IGNORECASE)
                    if matches_found and "Plan/Package" not in extracted:
                        name = matches_found[0].strip()
                        if len(name) > 3 and len(name) < 50:  # Reasonable name length
                            extracted["Plan/Package"] = name
                            break

                # Extract other numeric values (data, speed, etc.)
                numeric_patterns = [
                    r"(\d+)\s*GB",  # Data in GB
                    r"(\d+)\s*MB",  # Data in MB
                    r"(\d+)\s*TB",  # Data in TB (future proof)
                    r"(\d+)\s*Mbps",  # Speed
                    r"(\d+)\s*Kbps",  # Narrowband speeds
                    r"(\d+)\s*minutes?",  # Minutes
                    r"(\d+)\s*international minutes?",  # International minutes
                    r"(\d+)\s*roaming minutes?",  # Roaming minutes
                    r"(\d+)\s*BD",  # Fees, rental, charges
                    r"BD\s*(\d+(\.\d+)?)",  # BD amounts with decimals
                    r"(\d+)\s*months?",  # Months (commitments, promos)
                    r"(\d+)\s*years?",  # Years (long commitments)
                    r"(\d+)\s*days?",  # Days validity
                    r"(\d+)\s*%",  # Percent (discounts, VAT, etc.)
                    r"(\d+)\.?\d*\s*VAT",  # VAT values
                    r"(\d+)\.?\d*",  # Decimal numbers (e.g., 10.300 BD)
                ]

                for pattern in numeric_patterns:
                    matches_found = re.findall(pattern, text, re.IGNORECASE)
                    if matches_found:
                        value = matches_found[0]
                        if "GB" in text.upper():
                            extracted["Data Allowance"] = f"{value} GB"
                        elif "Mbps" in text:
                            extracted["Speed"] = f"{value} Mbps"
                        elif "month" in text.lower():
                            extracted["Duration"] = f"{value} months"
                        break

        except Exception as e:
            logger.debug(f"Value extraction failed: {e}")

        return extracted

    def _log_retrieval_debug(self, res):
        try:
            source_nodes = getattr(res, "source_nodes", [])
            rows = []
            for n in source_nodes:
                meta = getattr(n, "metadata", {}) or {}
                txt = getattr(n, "text", "") or ""
                score = None
                if hasattr(n, "score"):
                    score = getattr(n, "score")
                if isinstance(getattr(n, "extra_info", None), dict):
                    score = n.extra_info.get("score", score)
                rows.append(
                    {
                        "source": meta.get("source", "unknown"),
                        "score": score,
                        "snippet": txt[:200],
                    }
                )
            if rows:
                logger.info("Retrieval debug (top nodes):")
                for r in rows:
                    logger.info(
                        f"  - {r['source']} | score={r['score']} | snippet={r['snippet']}"
                    )
        except Exception as e:
            logger.debug(f"Retrieval debug failed: {e}")

    def _ensure_structured_response(
        self, raw_response: str, query: str, sources: List[Dict]
    ) -> str:
        """Ensure the response is properly structured and includes helpful closing"""

        if not raw_response or len(raw_response.strip()) < 20:
            return self._create_fallback_response(query, sources, "Empty LLM response")[
                "text"
            ]

        # Clean up the response
        response = raw_response.strip()

        # Remove redundant conversational elements but keep the helpful closing
        unwanted_phrases = [
            "Dear internal knowledge reviewer",
            "Thank you for your valuable feedback",
            "Please let me know if there are any further questions",
        ]

        for phrase in unwanted_phrases:
            response = re.sub(
                rf".*{re.escape(phrase)}.*?\n",
                "",
                response,
                flags=re.IGNORECASE | re.DOTALL,
            )

        # Remove duplicate sections
        response = re.sub(
            r"## Executive Summary.*?## Executive Summary",
            "## Executive Summary",
            response,
            flags=re.DOTALL,
        )

        # Ensure proper structure
        if not "## Answer" in response:
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            if lines:
                main_answer = lines[0]
                remaining = "\n".join(lines[1:]) if len(lines) > 1 else ""

                structured = f"## Answer\n{main_answer}\n"
                if remaining and len(remaining.strip()) > 20:
                    structured += f"\n## Details\n{remaining}"

                response = structured

        # Add sources if missing
        if sources and "## References" not in response:
            source_names = [
                s["source"] for s in sources[:3] if s["source"] != "unknown"
            ]
            if source_names:
                response += f"\n\n## References\n" + "\n".join(
                    [f"• {name}" for name in source_names]
                )

        # Ensure helpful closing message
        closing_message = "If you need anything else or have follow-up questions, I'm here to assist you!"
        if closing_message not in response:
            response += f"\n\n{closing_message}"

        return response

    def _extract_main_answer(self, query: str, sources: List[Dict]) -> str:
        """Extract key information from sources to answer the specific question"""

        if not sources:
            return ""

        if any(
            word in query.lower()
            for word in ["maximum", "max", "amount", "tenure", "limit"]
        ):
            first_source_text = sources[0].get("text", "")

            amount_matches = re.findall(
                r"BD\s*\d+(?:,\d{3})*(?:\.\d{2})?", first_source_text
            )
            tenure_matches = re.findall(
                r"(\d+)\s*(?:months?|years?)", first_source_text, re.IGNORECASE
            )

            answer_parts = []
            if amount_matches:
                answer_parts.append(f"Maximum finance amount: {amount_matches[0]}")
            if tenure_matches:
                answer_parts.append(f"Maximum tenure: {tenure_matches[0]} months/years")

            if answer_parts:
                return " | ".join(answer_parts)

        first_text = sources[0].get("text", "")
        sentences = re.split(r"[.!?]\s+", first_text)
        if sentences:
            return sentences[0].strip() + "."

        return ""

    def _create_fallback_response(
        self, query: str, sources: List[Dict], error_detail: str
    ) -> Dict[str, Any]:
        """Create a structured fallback response when LLM fails"""

        if not sources:
            return {
                "text": f"I couldn't find relevant information for: {query}",
                "sources": [],
            }

        main_answer = self._extract_main_answer(query, sources)

        fallback_text = (
            f"## Answer\n{main_answer or 'Information found in documents below'}\n\n"
        )
        fallback_text += f"## Detailed Information\n"

        for i, source in enumerate(sources[:2], 1):
            fallback_text += f"\n**From {source['source']}:**\n"
            clean_text = re.sub(r"\s+", " ", source["text"][:500])
            fallback_text += f"{clean_text}...\n"

        if sources:
            source_names = [
                s["source"] for s in sources[:3] if s["source"] != "unknown"
            ]
            if source_names:
                fallback_text += f"\n## References\n" + "\n".join(
                    [f"• {name}" for name in source_names]
                )

        return {"text": fallback_text, "sources": sources}

    def rag_answer(
        self,
        query: str,
        conversation_context: str = "",
        top_k: int = RAG_TOP_K,
        max_source_chars: int = RAG_MAX_SOURCE_CHARS,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """Enhanced RAG answer with smart template selection and table/regular search decision making"""

        try:
            if not self.query_engine:
                logger.error("Query engine not initialized")
                return {
                    "text": "Knowledge base not available. Please try again later.",
                    "sources": [],
                }

            # Step 1: ALWAYS do regular RAG search first
            logger.info("Executing regular RAG query")
            res = self.query_engine.query(query)

            if not res:
                logger.error("Empty response from query engine")
                return {"text": "No response from knowledge base.", "sources": []}

        except Exception as e:
            logger.error(f"Error in rag_answer retrieval: {e}")
            return {
                "text": "I'm having trouble accessing the knowledge base right now. Please try again in a moment.",
                "sources": [],
            }

        # Step 2: Process regular RAG results
        self._log_retrieval_debug(res)
        regular_nodes = []
        try:
            regular_nodes = list(getattr(res, "source_nodes", []) or [])
            logger.debug(f"Found {len(regular_nodes)} regular RAG nodes")
        except Exception as e:
            logger.warning(f"Error getting source nodes: {e}")
            regular_nodes = []

        # Step 3: Evaluate regular RAG quality
        intent = self._detect_exact_lookup_query(query)
        regular_quality = self._evaluate_rag_quality(regular_nodes, query, intent)

        logger.info(f"Regular RAG quality score: {regular_quality['score']}")
        logger.info(
            f"Query intent - exact_lookup: {intent['is_exact_lookup']}, confidence: {intent['confidence']}"
        )

        # Step 4: Decide if we need table search
        should_try_table_search = (
            intent.get("is_exact_lookup", False)
            and intent.get("confidence", 0.0) >= 0.5
            and regular_quality.get("score", 0.0) < 0.7
        )

        table_result = None
        if should_try_table_search:
            logger.info(
                f"Regular RAG quality insufficient ({regular_quality['score']:.2f}), trying table search"
            )
            table_result = self._execute_table_lookup(query, intent)

            if table_result and len(table_result.strip()) > 50:
                table_quality = self._evaluate_table_result(table_result, query, intent)
                logger.info(f"Table search quality score: {table_quality['score']}")

                # Use table result only if it's significantly better than regular RAG
                if table_quality["score"] > regular_quality["score"] + 0.05:
                    logger.info(
                        "Table search provides better results, using table answer"
                    )
                    return {
                        "text": table_result,
                        "sources": [
                            {
                                "source": "table_lookup",
                                "text": table_result[:500],
                                "score": 1.0,
                            }
                        ],
                        "lookup_type": "table_preferred",
                    }
                else:
                    logger.info(
                        "Table search not significantly better, using regular RAG"
                    )
            else:
                logger.info("Table search returned no useful results")

        # Step 5: Process regular RAG results (since we're not using table search)
        sources, doc_blocks = self._process_rag_nodes(regular_nodes, max_source_chars)

        if not sources:
            logger.warning(f"No valid sources found for query: {query}")
            return {
                "text": "I couldn't find relevant information in the knowledge base for your question. Try rephrasing your query or use `/web` for external search.",
                "sources": [],
            }

        # Step 6: Create enhanced prompt based on query type and data
        docs_text = (
            "\n\n".join(doc_blocks) if doc_blocks else "(no matching documents found)"
        )

        # Detect if we have structured/table data
        has_structured_data = any(
            source.get("content_type") == "table_row" for source in sources
        )

        # Get appropriate template and create prompt
        template = self._get_enhanced_prompt_for_query(query, has_structured_data)
        prompt = template.format(query_str=query, context_str=docs_text)

        # Step 7: Get LLM response
        try:
            completion = Settings.llm.complete(prompt)
            final_answer = (
                completion.text.strip()
                if hasattr(completion, "text")
                else str(completion).strip()
            )

            if not final_answer:
                logger.warning("Empty response from LLM")
                return self._create_fallback_response(
                    query, sources, "Empty LLM response"
                )

            logger.info(f"LLM response length: {len(final_answer)}")
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            return self._create_fallback_response(query, sources, str(e))

        # Step 8: Post-process response
        try:
            final_answer = self._ensure_structured_response(
                final_answer, query, sources
            )
        except Exception as e:
            logger.warning(f"Error structuring response: {e}")

        return {
            "text": final_answer,
            "sources": sources,
        }

    def _evaluate_table_result(
        self, table_result: str, query: str, intent: Dict
    ) -> Dict[str, Any]:
        """Evaluate LLM-generated table responses"""

        score = 0.7  # Start with higher base score since it's LLM-generated
        reasons = ["llm_generated"]

        query_lower = query.lower()
        result_lower = table_result.lower()

        # Check for natural language patterns (good sign)
        if any(
            phrase in result_lower
            for phrase in ["the best", "available is", "here are"]
        ):
            score += 0.2
            reasons.append("natural_language")

        # Check for bullet points (good formatting)
        if "•" in table_result or "*" in table_result:
            score += 0.1
            reasons.append("good_formatting")

        # Check for preserved data
        if any(term in result_lower for term in ["bd", "code", "amount"]):
            score += 0.1
            reasons.append("preserved_data")

        return {"score": min(1.0, score), "reasons": reasons}

    def _evaluate_rag_quality(
        self, nodes: List, query: str, intent: Dict
    ) -> Dict[str, Any]:
        """Evaluate the quality of regular RAG results"""
        if not nodes:
            return {"score": 0.0, "reasons": ["no_nodes"]}

        query_lower = query.lower()
        query_words = set(word.lower() for word in query.split() if len(word) > 2)

        total_score = 0
        max_possible = 0
        quality_reasons = []

        for node in nodes[:3]:  # Check top 3 nodes
            try:
                node_obj = getattr(node, "node", node)
                text = getattr(node_obj, "text", "") or ""
                score = getattr(node, "score", 0)

                if not text:
                    continue

                text_lower = text.lower()
                text_words = set(text_lower.split())

                # Calculate various quality metrics
                node_score = 0
                max_node_score = 100

                # 1. Exact phrase match (40 points)
                if query_lower in text_lower:
                    node_score += 40
                    quality_reasons.append("exact_phrase_match")

                # 2. Word coverage (30 points)
                word_coverage = (
                    len(query_words.intersection(text_words)) / len(query_words)
                    if query_words
                    else 0
                )
                node_score += word_coverage * 30

                # 3. Retrieval score (20 points)
                if score is not None:
                    # Normalize score (assuming scores are typically between 0 and 1)
                    normalized_score = min(1.0, max(0.0, score)) if score > 0 else 0
                    node_score += normalized_score * 20

                # 4. Intent matching (10 points)
                if intent.get("is_exact_lookup"):
                    # For exact lookups, check if we have structured data
                    if any(
                        indicator in text_lower
                        for indicator in [":", "bd", "code", "price", "fee"]
                    ):
                        node_score += 10
                        quality_reasons.append("structured_data_present")

                total_score += node_score
                max_possible += max_node_score

            except Exception as e:
                logger.debug(f"Error evaluating node quality: {e}")
                continue

        # Calculate final quality score
        final_score = (total_score / max_possible) if max_possible > 0 else 0

        # Bonus for multiple good nodes
        good_nodes = len([n for n in nodes[:3] if getattr(n, "score", 0) > 0.5])
        if good_nodes >= 2:
            final_score += 0.1
            quality_reasons.append("multiple_good_nodes")

        return {
            "score": min(1.0, final_score),
            "reasons": quality_reasons,
            "nodes_evaluated": len(nodes),
        }

    def _process_rag_nodes(
        self, nodes: List, max_source_chars: int
    ) -> Tuple[List[Dict], List[str]]:
        """Process regular RAG nodes into sources and doc blocks"""
        sources = []
        doc_blocks = []

        for i, node in enumerate(nodes[:8], 1):  # Limit to top 8 nodes
            try:
                score = getattr(node, "score", None)
                node_obj = getattr(node, "node", node)
                meta = getattr(node_obj, "metadata", {}) or {}
                node_text = (
                    getattr(node_obj, "text", "") or getattr(node, "text", "") or ""
                )

                if not node_text:
                    continue

                source_name = meta.get("source") or meta.get("file_name") or "unknown"
                content_type = meta.get("content_type", "text")

                # Handle table rows differently
                if content_type == "table_row":
                    table_num = meta.get("table_number", "")
                    row_num = meta.get("row_number", "")
                    if table_num and row_num:
                        display_source = (
                            f"{source_name} (Table {table_num}, Row {row_num})"
                        )
                    else:
                        display_source = f"{source_name} (Table Data)"
                else:
                    display_source = source_name

                snippet = re.sub(r"\s+", " ", node_text).strip()
                if max_source_chars and len(snippet) > max_source_chars:
                    snippet = snippet[:max_source_chars] + "..."

                sources.append(
                    {
                        "source": display_source,
                        "score": score,
                        "text": snippet,
                        "content_type": content_type,
                        "metadata": meta,
                    }
                )

                score_text = f"Score: {round(score, 4)}" if score is not None else ""
                if content_type == "table_row":
                    doc_blocks.append(
                        f"### Table Data {i}: {display_source}\n{score_text}\n{snippet}"
                    )
                else:
                    doc_blocks.append(
                        f"### Source {i}: {display_source}\n{score_text}\n{snippet}"
                    )

            except Exception as e:
                logger.warning(f"Error processing node {i}: {e}")
                continue

        return sources, doc_blocks

    def review(self, query: str, rag_text: str, convo_context: str = "") -> str:
        """
        ENHANCED REVIEWER: Maintains strict data fidelity while improving structure.
        Does NOT change factual content, only improves formatting and completeness.
        """

        # First, check if the RAG text already has good structure and accuracy
        has_good_structure = any(
            marker in rag_text
            for marker in [
                "## Answer",
                "## Details",
                "## Source",
                "BD ",
                "Code:",
                "Price:",
            ]
        )

        # If RAG text is already well-structured with exact values, return it as-is
        if has_good_structure and len(rag_text.strip()) > 100:
            logger.info("RAG text already well-structured, minimal review needed")
            return self._minimal_review(rag_text, query)

        # Create a strict review prompt that preserves exact data
        prompt = f"""
    You are a PRECISION REVIEWER for internal staff responses. Your ONLY job is to improve formatting while preserving EXACT data accuracy.

    **CRITICAL RULES - NO EXCEPTIONS:**
    1. **PRESERVE ALL EXACT VALUES**: Keep every number, price, code, date exactly as provided
    2. **NO DATA CHANGES**: Never modify BD amounts, codes, percentages, or any specific values
    3. **NO CONVERSIONS**: Don't convert currencies, units, or formats
    4. **NO ADDITIONS**: Don't add information not present in the draft
    5. **IMPROVE STRUCTURE ONLY**: Better organize the existing information

    **CONVERSATION CONTEXT:**
    {convo_context}

    **USER QUERY:**
    {query}

    **DRAFT ANSWER (contains exact data - DO NOT CHANGE VALUES):**
    {rag_text}

    **YOUR TASK:**
    - Improve the structure and readability of the draft
    - Ensure proper markdown formatting (## headers, ** bold text)
    - Group related information logically
    - Preserve every exact value, code, price, and specification
    - Remove any redundancy but keep all unique factual content
    - Ensure the answer directly addresses the user's query

    **ENHANCED ANSWER (same facts, better structure):**
    """

        try:
            response = Settings.llm.complete(prompt, timeout=LLM_COMPLETE_TIMEOUT)
            reviewed_text = response.text.strip()

            # VALIDATION: Check if reviewer preserved key data points
            if self._validate_review_fidelity(rag_text, reviewed_text):
                logger.info("Review passed fidelity validation")
                return reviewed_text
            else:
                logger.warning("Review failed fidelity validation, returning original")
                return self._minimal_review(rag_text, query)

        except Exception as e:
            logger.warning(f"Reviewer failed: {e} — returning original RAG text")
            return self._minimal_review(rag_text, query)

    def _minimal_review(self, rag_text: str, query: str) -> str:
        """
        Minimal review that only improves structure without LLM intervention
        """
        try:
            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", rag_text.strip())

            # Ensure proper Answer section
            if not text.startswith("## Answer") and not text.startswith("# Answer"):
                lines = text.split("\n")
                first_line = lines[0] if lines else ""
                rest = "\n".join(lines[1:]) if len(lines) > 1 else ""

                if first_line.strip():
                    text = f"## Answer\n{first_line}\n\n{rest}".strip()

            # Ensure helpful closing
            closing_phrases = ["If you need anything else", "I'm here to assist"]
            has_closing = any(phrase in text for phrase in closing_phrases)

            if not has_closing:
                text += "\n\nIf you need anything else or have follow-up questions, I'm here to assist you!"

            return text

        except Exception as e:
            logger.debug(f"Minimal review failed: {e}")
            return rag_text

    def _validate_review_fidelity(self, original: str, reviewed: str) -> bool:
        """
        Validate that the review preserved all critical data points
        """
        try:
            # Extract critical data points from both texts
            original_data = self._extract_critical_data_points(original)
            reviewed_data = self._extract_critical_data_points(reviewed)

            # Check BD amounts
            if original_data["bd_amounts"] != reviewed_data["bd_amounts"]:
                logger.warning("BD amounts changed during review")
                return False

            # Check codes (allow case changes but not content changes)
            orig_codes = {code.upper() for code in original_data["codes"]}
            rev_codes = {code.upper() for code in reviewed_data["codes"]}
            if orig_codes != rev_codes:
                logger.warning("Codes changed during review")
                return False

            # Check numbers (allow minor formatting but not values)
            if len(original_data["numbers"]) != len(reviewed_data["numbers"]):
                number_diff = abs(
                    len(original_data["numbers"]) - len(reviewed_data["numbers"])
                )
                if number_diff > 1:  # Allow 1 number difference for minor formatting
                    logger.warning("Significant numbers changed during review")
                    return False

            # Check that key terms are preserved
            original_terms = set(re.findall(r"\b\w+\b", original.lower()))
            reviewed_terms = set(re.findall(r"\b\w+\b", reviewed.lower()))

            # Critical terms that must be preserved
            critical_terms = {
                "bd",
                "code",
                "price",
                "monthly",
                "total",
                "rental",
                "fiber",
                "extra",
            }
            original_critical = original_terms.intersection(critical_terms)
            reviewed_critical = reviewed_terms.intersection(critical_terms)

            if original_critical != reviewed_critical:
                logger.warning("Critical terms changed during review")
                return False

            return True

        except Exception as e:
            logger.debug(f"Validation failed: {e}")
            return False

    def _extract_critical_data_points(self, text: str) -> Dict[str, List]:
        """
        Extract critical data points for validation
        """
        data = {"bd_amounts": [], "codes": [], "numbers": [], "percentages": []}

        try:
            # Extract BD amounts
            bd_pattern = r"BD\s*(\d+(?:[,.]?\d{3})*(?:\.\d{2})?)"
            data["bd_amounts"] = re.findall(bd_pattern, text, re.IGNORECASE)

            # Extract codes (alphanumeric combinations)
            code_pattern = r"\b[A-Z]{2,}[0-9]{1,}[A-Z0-9]*\b"
            data["codes"] = re.findall(code_pattern, text.upper())

            # Extract standalone numbers
            number_pattern = r"\b\d+(?:\.\d+)?\b"
            data["numbers"] = re.findall(number_pattern, text)

            # Extract percentages
            percent_pattern = r"\d+(?:\.\d+)?%"
            data["percentages"] = re.findall(percent_pattern, text)

        except Exception as e:
            logger.debug(f"Data extraction failed: {e}")

        return data

    def summarize(self, text: str) -> str:
        prompt = (
            "Summarize the following answer into 3-5 concise bullet points. Be factual, avoid redundancy.\n\n"
            + text
        )
        try:
            return Settings.llm.complete(
                prompt, timeout=LLM_FALLBACK_TIMEOUT
            ).text.strip()
        except Exception as e:
            return f"(Summary unavailable: {e})"

    def reason(self, user_query: str, convo_context: str = "") -> str:
        if not self.reasoning_agent:
            # Fallback to direct LLM call
            prompt = f"""
            Role: Internal policy analyst.
            Use all internal data available to:
            1. Analyze the query in depth.
            2. Summarize applicable rules, eligibility conditions, and deadlines.
            3. Provide structured output.
            Conversation context:
            {convo_context}
            User Query:
            {user_query}
            Structured Output:
            1. Relevant Policy
            2. Dates & Deadlines
            3. Steps for Staff
            4. Key Details (amounts, approvals, contacts)
            """
            try:
                return Settings.llm.complete(
                    prompt, timeout=LLM_COMPLETE_TIMEOUT
                ).text.strip()
            except Exception as e:
                return f"(Reasoning error: {e})"

        prompt = f"""
        Role: Internal policy analyst.
        Use all internal data available to:
        1. Analyze the query in depth.
        2. Summarize applicable rules, eligibility conditions, and deadlines.
        3. Provide structured output.
        Conversation context:
        {convo_context}
        User Query:
        {user_query}
        Structured Output:
        1. Relevant Policy
        2. Dates & Deadlines
        3. Steps for Staff
        4. Key Details (amounts, approvals, contacts)
        """
        try:
            return Settings.llm.complete(
                prompt, timeout=LLM_COMPLETE_TIMEOUT
            ).text.strip()
        except Exception as e:
            return f"(Reasoning error: {e})"


# ConversationMemory
class ConversationMemory:
    def __init__(self, keep_last_n: int = 0):
        self.last_mode: Optional[str] = None
        self.history: List[Tuple[str, str]] = []
        self.keep_last_n = keep_last_n

    def update(self, mode: str, user_query: str, answer: str):
        self.last_mode = mode
        self.history.append((user_query, answer))
        if len(self.history) > self.keep_last_n:
            self.history = self.history[-self.keep_last_n :]

    def get_context(self, turns: int = 0) -> str:
        recent = self.history[-turns:]
        return "\n\n".join([f"User: {q}\nAssistant: {a}" for q, a in recent])


# Agents Orchestrator
class Orchestrator:
    def __init__(self, doc_manager: DocumentManager):
        self.doc_manager = doc_manager
        self.query_engine = doc_manager.get_query_engine()
        self.internal = InternalAgents(self.query_engine, doc_manager)
        self.web = CustomWebSearchTool(GOOGLE_API_KEY, GOOGLE_CSE_ID)
        self.memory = ConversationMemory(keep_last_n=6)
        self.site_domain = SITE_SEARCH_DOMAIN if SITE_SEARCH_DOMAIN else None

    def route(self, raw_query: str) -> Dict[str, Any]:
        q = raw_query.strip()
        mode, query = self._detect_mode(q, self.memory.last_mode)
        convo_ctx = self.memory.get_context(turns=1)

        if mode == "web":
            result = self._web_only(query, convo_ctx)
        else:  # "rag"
            result = self._rag_pipeline(query, convo_ctx)

        self.memory.update(result["mode"], query, result["answer"])
        return result

    def _detect_mode(self, raw: str, last_mode: Optional[str]) -> Tuple[str, str]:
        low = raw.lower()
        if low.startswith("/web "):
            return "web", raw[5:].strip()
        if low.startswith("/rag "):
            return "rag", raw[5:].strip()
        return (last_mode or "rag"), raw

    def _rag_pipeline(self, query: str, convo_ctx: str) -> Dict[str, Any]:
        t0 = time.time()
        rag = self.internal.rag_answer(query, conversation_context=convo_ctx)

        if not rag["text"] or rag["text"].strip().lower() in {"", "empty response"}:
            answer = (
                "I couldn't find enough information in the internal knowledge base.\n"
                "Tip: try `/web <your question>` for external results."
            )
            if self.site_domain and self.web.is_available():
                site_block = self.web.search_and_summarize(
                    query, convo_context=convo_ctx, site=self.site_domain
                )
                answer += "\n\n---\nSite-specific results (appendix):\n" + site_block
            dt = time.time() - t0
            return {
                "mode": "rag",
                "query": query,
                "answer": answer,
                "elapsed_sec": round(dt, 2),
                "sources": rag.get("sources", []),
            }

        enhanced_answer = self.internal.enhance_answer(
            rag["text"], rag.get("sources", [])
        )

        should_review = (
            len(query.split()) > 8
            and "compare" in query.lower()
            and len(rag.get("sources", [])) > 2
            and not any(
                indicator in enhanced_answer.lower()
                for indicator in [
                    "bd ",
                    "code:",
                    "rental:",
                    "total monthly",  # Skip review if exact values already present
                ]
            )
        )

        # ADDITIONALLY: Skip review for exact lookup queries to preserve precision
        intent = getattr(self.internal, "_detect_exact_lookup_query", lambda x: {})(
            query
        )
        if intent.get("is_exact_lookup", False) and intent.get("confidence", 0) > 0.7:
            should_review = False
            logger.info("Skipping review for exact lookup query to preserve precision")

        if should_review:
            try:
                reviewed = self.internal.review(
                    query, enhanced_answer, convo_context=convo_ctx
                )
                # Only use reviewed version if it's substantially better (not just different)
                if (
                    reviewed
                    and len(reviewed.strip()) > len(enhanced_answer.strip()) * 0.9
                ):
                    final_answer = reviewed
                else:
                    final_answer = enhanced_answer
            except Exception as e:
                logger.warning(f"Review failed: {e}")
                final_answer = enhanced_answer
        else:
            final_answer = enhanced_answer
            logger.info("Skipping review to preserve data accuracy")

        # Rest of the method remains the same...
        if len(final_answer) > 800:
            try:
                summary = self.internal.summarize(final_answer)
                if summary and len(summary.strip()) > 50:
                    # Insert summary after the main answer
                    parts = final_answer.split("\n## References")
                    main_content = parts[0]
                    refs_section = (
                        "\n## References" + parts[1] if len(parts) > 1 else ""
                    )
                    final_answer = (
                        main_content
                        + f"\n\n## Executive Summary\n{summary}"
                        + refs_section
                    )

            except Exception as e:
                logger.debug(f"Summary generation failed: {e}")

        # Ensure proper formatting
        final_answer = self._format_final_answer(final_answer, rag.get("sources", []))

        dt = time.time() - t0
        return {
            "mode": "rag",
            "query": query,
            "answer": final_answer,
            "elapsed_sec": round(dt, 2),
            "sources": rag.get("sources", []),
        }

    def _format_final_answer(self, answer: str, sources: List[Dict]) -> str:
        """Minimal formatting - the RAG should already be well-formatted"""

        # Clean up excessive whitespace
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        # Add references if missing
        if sources and not any(ref in answer for ref in ["## References", "## Source"]):
            seen = set()
            ref_lines = []
            for s in sources:
                name = s.get("source", "unknown")
                if name and name != "unknown" and name not in seen:
                    ref_lines.append(f"• {name}")
                    seen.add(name)

            if ref_lines:
                answer += "\n\n## References\n" + "\n".join(ref_lines)

        # Add helpful closing if missing
        closing_phrases = ["If you need anything else", "I'm here to assist"]
        if not any(phrase in answer for phrase in closing_phrases):
            answer += "\n\nIf you need anything else or have follow-up questions, I'm here to assist you!"

        return answer.strip()

    def _web_only(self, query: str, convo_ctx: str) -> Dict[str, Any]:
        t0 = time.time()
        if not self.web.is_available():
            text = "Web search is not configured. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID."
        else:
            text = self.web.search_and_summarize(query, convo_context=convo_ctx)
        if self.site_domain and self.web.is_available():
            site_block = self.web.search_and_summarize(
                query, convo_context=convo_ctx, site=self.site_domain
            )
            text = text + "\n\n---\nSite-specific results (appendix):\n" + site_block
        dt = time.time() - t0
        return {
            "mode": "web",
            "query": query,
            "answer": text,
            "elapsed_sec": round(dt, 2),
            "sources": [],
        }

    def _reason_only(self, query: str, convo_ctx: str) -> Dict[str, Any]:
        t0 = time.time()
        text = self.internal.reason(query, convo_context=convo_ctx)
        if self.site_domain and self.web.is_available():
            site_block = self.web.search_and_summarize(
                query, convo_context=convo_ctx, site=self.site_domain
            )
            text = text + "\n\n---\nSite-specific results (appendix):\n" + site_block
        dt = time.time() - t0
        return {
            "mode": "reason",
            "query": query,
            "answer": text,
            "elapsed_sec": round(dt, 2),
            "sources": [],
        }

    def _pipeline_with_web_append(self, query: str, convo_ctx: str) -> Dict[str, Any]:
        core = self._rag_pipeline(query, convo_ctx)
        web_block = ""
        if self.web.is_available():
            web_block = (
                "\n\n---\nExternal Web Summary (appendix):\n"
                + self.web.search_and_summarize(query, convo_context=convo_ctx)
            )
        else:
            web_block = "\n\n---\n(Web search not configured)"
        core["mode"] = "web+rag"
        core["answer"] = core["answer"] + web_block

        if self.site_domain and self.web.is_available():
            site_block = self.web.search_and_summarize(
                query, convo_context=convo_ctx, site=self.site_domain
            )
            core["answer"] = (
                core["answer"]
                + "\n\n---\nSite-specific results (appendix):\n"
                + site_block
            )

        return core


# Global instances and helpers
document_manager: Optional[DocumentManager] = None
orchestrator: Optional[Orchestrator] = None


def get_document_manager() -> DocumentManager:
    global document_manager
    if document_manager is None:
        document_manager = DocumentManager()
    return document_manager


def initialize_system():
    global orchestrator
    shared_folder_url = os.getenv("ONEDRIVE_SHARED_URL")
    print(f"Initializing system with documents from: {shared_folder_url}")
    dm = get_document_manager()
    dm.load_onedrive_documents(force_rebuild=False, shared_folder_url=shared_folder_url)
    orchestrator = Orchestrator(dm)
    return orchestrator


def refresh_document_index(
    force_rebuild: bool = False, shared_folder_url: Optional[str] = None
):
    """
    Rebuild or refresh the vector index using the lazy-initialized DocumentManager.
    This uses dm = get_document_manager() instead of a module-level pre-created instance.
    Returns a dict with {success: bool, message: str}.
    """
    try:
        dm = get_document_manager()

        # Optionally download from OneDrive first
        if shared_folder_url:
            local_dir = "data/onedrive"
            os.makedirs(local_dir, exist_ok=True)
            print(f"Downloading documents from OneDrive URL: {shared_folder_url}")
            downloaded_files = dm.onedrive_client.download_shared_folder(
                local_dir, shared_folder_url
            )
            print(f"Downloaded {len(downloaded_files)} files")

        if not os.path.exists(LOCAL_DATA_DIR):
            return {
                "success": False,
                "message": f"Directory not found: {LOCAL_DATA_DIR}",
            }

        print(f"Loading documents from {LOCAL_DATA_DIR}")
        files = dm._collect_source_files(LOCAL_DATA_DIR)
        if not files:
            return {"success": False, "message": "No documents found to index"}

        # If forcing a full rebuild: delete and recreate collection, clear caches
        if force_rebuild:
            print("Forcing rebuild of vector index")
            try:
                if dm.qdrant and COLLECTION_NAME:
                    try:
                        dm.qdrant.delete_collection(collection_name=COLLECTION_NAME)
                        print(f"Deleted collection: {COLLECTION_NAME}")
                    except Exception as e:
                        print(f"Warning: failed to delete existing collection: {e}")
                # Recreate vector store
                dm._setup_vector_store()
                print("Recreated vector store")
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error recreating vector store: {e}",
                }

            # Clear parsed cache to force reparse
            try:
                removed = dm.parser.clear_parsed_cache(files)
                if removed:
                    print(f"Cleared {removed} cached parse artifacts")
            except Exception as e:
                print(f"Warning: could not clear parse cache: {e}")

        # Process documents (force_reparse if requested)
        n = dm._process_documents(files, force_reparse=bool(force_rebuild))

        # Save fingerprint for next runs
        try:
            fp = dm._calc_fingerprint_for_files(files)
            dm._save_fingerprint(fp)
        except Exception as e:
            print(f"Warning: failed to save fingerprint: {e}")
        global orchestrator
        orchestrator = Orchestrator(dm)

        return {"success": True, "message": f"Loaded {n} documents"}

    except Exception as e:
        return {"success": False, "message": f"Document refresh failed: {e}"}


def process_user_query(raw_query: str) -> Dict[str, Any]:
    system = orchestrator or initialize_system()
    return system.route(raw_query)


def ensure_data_directory_exists():
    if not os.path.exists(LOCAL_DATA_DIR):
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        print(f"Created data directory at {LOCAL_DATA_DIR}")
    parsed_dir = PARSED_CACHE_DIR
    if not os.path.exists(parsed_dir):
        os.makedirs(parsed_dir, exist_ok=True)
        print(f"Created parsed cache directory at {parsed_dir}")


if __name__ == "__main__":
    ensure_services_running()
    ensure_data_directory_exists()
    initialize_system()
    print("\nRAG Engine Interactive Mode")
    print("Commands:")
    print("  /web <q>       -> external web search only")
    # print("  /reason <q>    -> internal reasoning only")
    # print("  /web+rag <q>   -> internal pipeline + web appendix")
    print("  /rag <q>       -> force internal pipeline")
    print("  refresh        -> rebuild index from OneDrive/local")
    print("  exit           -> quit")

    while True:
        try:
            q = input("\nEnter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() == "exit":
            break
        if q.lower() == "refresh":
            res = refresh_document_index(force_rebuild=True)
            print(res.get("message", "Refresh completed"))
            continue

        result = process_user_query(q)
        print(f"\n[Mode] {result.get('mode')} | [Time] {result.get('elapsed_sec')}s")
        print("\n=== Answer ===")
        print(result.get("answer"))
