from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import os
from database import engine
from models.Admin_model import Base as AdminBase
from models.User_models import Base as UserBase
from admin_routes import router as admin_router
from user_routes import router as user_router
from models.Admin_model import Base as AdminBase, Document
from contextlib import asynccontextmanager
import document_sync
from datetime import datetime

# Initial setup - ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("data/onedrive", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)

_periodic_task: asyncio.Task | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _periodic_task
    print("Starting application initialization...")

    try:
        # 1) Ensure DB tables exist
        print("Ensuring database tables exist...")
        AdminBase.metadata.create_all(bind=engine)
        UserBase.metadata.create_all(bind=engine)

        # 2) Sync documents to database (uses smart logic)
        print("Syncing documents to database...")
        try:
            sync_result = document_sync.sync_all_documents_to_db()
            print(f"Document sync result: {sync_result}")

            # The sync function now handles RAG initialization intelligently
            if sync_result.get("changes_made"):
                print("Changes detected during startup sync, RAG reindexing completed automatically")
            else:
                print("No changes detected, checking RAG initialization...")

                try:
                    import rag_engine_test
                    # Use smart initialization that respects existing state
                    dm = rag_engine_test.get_document_manager()
                    shared_folder_url = os.getenv("ONEDRIVE_SHARED_URL")

                    # This will use fingerprinting to determine if initialization is needed
                    indexed_count = dm.load_onedrive_documents(
                        force_rebuild=False,
                        shared_folder_url=shared_folder_url
                    )
                    print(f"RAG initialization complete (documents: {indexed_count})")

                except Exception as e:
                    print(f"RAG initialization failed: {e}")

        except Exception as e:
            print(f"Warning: document sync failed during startup: {e}")

        # 3) Start periodic sync task
        print("Setting up periodic sync task...")
        _periodic_task = asyncio.create_task(_periodic_sync_worker())
        print("Initialization complete!")

    except Exception as e:
        print(f"Error during startup: {e}")

    yield  # application runs here

    # Shutdown: cancel periodic task
    print("Shutting down application...")
    if _periodic_task:
        _periodic_task.cancel()
        try:
            await _periodic_task
        except asyncio.CancelledError:
            pass
    print("Shutdown complete.")

async def _periodic_sync_worker():
    """
    Periodic worker that:
      - runs document_sync.sync_all_documents_to_db() which uses smart logic
      - only reinitializes RAG if actual changes were detected
    Run interval: 1 hour (3600s). Adjust as needed.
    """
    interval_seconds = 3600
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            print(f"[{datetime.utcnow().isoformat()}] Running periodic document sync...")

            try:
                # This now uses smart logic and only reindexes if changes are detected
                sync_result = document_sync.sync_all_documents_to_db()

                if sync_result.get("success") and sync_result.get("changes_made"):
                    print(f"[{datetime.utcnow().isoformat()}] Changes detected during sync, RAG reindexing handled automatically")
                elif sync_result.get("success"):
                    print(f"[{datetime.utcnow().isoformat()}] No changes detected, RAG index unchanged")
                else:
                    print(f"[{datetime.utcnow().isoformat()}] Document sync failed: {sync_result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"Periodic document_sync failed: {e}")

            # No need to manually call initialize_system() here anymore
            # The sync function handles reindexing intelligently

        except asyncio.CancelledError:
            # Task was cancelled during shutdown
            break
        except Exception as e:
            print(f"Unexpected error in periodic sync worker: {e}")
            # wait a short time before continuing to avoid tight loop on failures
            await asyncio.sleep(10)

# Create FastAPI app with lifespan manager
app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include your routers (make sure imports refer to the correct modules)
from admin_routes import router as admin_router
from user_routes import router as user_router

app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
app.include_router(user_router, prefix="/api/users", tags=["users"])

# Health check endpoint
@app.get("/health", tags=["health"])
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/system-info", tags=["system"])
def system_info():
    """Get system information including document counts"""
    try:
        db = document_sync.SessionLocal()
        doc_count = db.query(Document).count()
        onedrive_count = db.query(Document).filter(Document.source_type == 'onedrive').count()
        local_count = db.query(Document).filter(Document.source_type == 'local').count()
        uploaded_count = db.query(Document).filter(Document.source_type == 'uploaded').count()
        indexed_count = db.query(Document).filter(Document.indexed == True).count()
        db.close()

        return {
            "status": "healthy",
            "documents": {
                "total": doc_count,
                "onedrive": onedrive_count,
                "local": local_count,
                "uploaded": uploaded_count,
                "indexed": indexed_count
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
