import os
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
import rag_engine_test
from database import get_db, SessionLocal
from models.Admin_model import Document, ContentSource

def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error hashing file {file_path}: {e}")
        return ""

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Get metadata for a file"""
    try:
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower().replace('.', '')

        # Get file content preview for text files
        content_preview = ""
        if file_ext in ['txt', 'md', 'csv', 'json', 'html']:
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    content_preview = f.read(500)  # First 500 chars
            except Exception as e:
                print(f"Error reading file preview: {e}")

        return {
            "filename": file_name,
            "file_type": file_ext,
            "size_bytes": file_size,
            "content_preview": content_preview,
            "file_hash": get_file_hash(file_path)
        }
    except Exception as e:
        print(f"Error getting metadata for {file_path}: {e}")
        return {
            "filename": os.path.basename(file_path),
            "file_type": "",
            "size_bytes": 0,
            "content_preview": "",
            "file_hash": ""
        }

def sync_all_documents_to_db() -> Dict[str, Any]:
    """
    Main function to sync all documents from the data directory to the database.
    This will ensure any files downloaded from OneDrive are properly registered.
    Uses the RAG engine's smart logic instead of forcing rebuilds.

    Returns:
        Dict with sync results
    """
    print("Starting document synchronization with database...")

    try:
        db = SessionLocal()

        # Track statistics
        stats = {
            "added": 0,
            "updated": 0,
            "unchanged": 0,
            "errors": 0
        }

        # Directory to scan - this includes all data including OneDrive downloads
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            print(f"Created data directory at {data_dir}")

        # Track if we made any changes that require reindexing
        changes_made = False

        # Process all files in the data directory and subdirectories
        for root, dirs, files in os.walk(data_dir):
            for file_name in files:
                # Skip system files and certain file types
                if (file_name.startswith('.') or
                    file_name == 'fingerprint.json' or
                    file_name.endswith(('.pyc', '.tmp', '.DS_Store'))):
                    continue

                file_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(file_path)

                # Skip directories
                if not os.path.isfile(file_path):
                    continue

                try:
                    # Determine source type based on path
                    source_type = 'local'
                    if 'onedrive' in rel_path.lower():
                        source_type = 'onedrive'
                    elif 'uploads' in rel_path.lower():
                        source_type = 'uploaded'

                    # Get file metadata
                    metadata = get_file_metadata(file_path)

                    # Check if document already exists in database
                    db_doc = db.query(Document).filter(Document.file_path == rel_path).first()

                    if not db_doc:
                        # Add new document
                        new_doc = Document(
                            title=file_name,
                            filename=file_name,
                            file_path=rel_path,
                            source_type=source_type,
                            file_type=metadata.get("file_type", ""),
                            size_bytes=metadata.get("size_bytes", 0),
                            content_preview=metadata.get("content_preview", ""),
                            file_hash=metadata.get("file_hash", ""),
                            indexed=False,
                            onedrive_link=None if source_type != 'onedrive' else os.getenv("ONEDRIVE_SHARED_URL", "")
                        )
                        db.add(new_doc)
                        stats["added"] += 1
                        changes_made = True
                        print(f"Added new document to database: {rel_path}")

                    else:
                        # Check if file has changed
                        current_hash = metadata.get("file_hash", "")
                        if not db_doc.file_hash or db_doc.file_hash != current_hash:
                            # Update existing document
                            db_doc.size_bytes = metadata.get("size_bytes", 0)
                            db_doc.content_preview = metadata.get("content_preview", "")
                            db_doc.file_hash = current_hash
                            db_doc.indexed = False  # Mark for reindexing
                            db_doc.updated_at = datetime.utcnow()
                            stats["updated"] += 1
                            changes_made = True
                            print(f"Updated document in database: {rel_path}")
                        else:
                            stats["unchanged"] += 1

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    stats["errors"] += 1

        # Commit all changes
        db.commit()

        # Only trigger reindexing if we actually made changes
        # Use the RAG engine's smart logic instead of forcing rebuild
        if changes_made:
            print(f"Changes detected ({stats['added']} added, {stats['updated']} updated), using RAG engine smart reindexing...")

            # Use the RAG engine's smart initialization instead of forcing rebuild
            # This will use the fingerprinting logic to determine if rebuilding is actually needed
            try:
                import rag_engine_test
                # Get the document manager and use its smart loading logic
                dm = rag_engine_test.get_document_manager()
                shared_folder_url = os.getenv("ONEDRIVE_SHARED_URL")

                # This will use the smart logic: only rebuild if fingerprint changes are detected
                indexed_count = dm.load_onedrive_documents(
                    force_rebuild=False,  # Don't force - let it decide based on fingerprint
                    shared_folder_url=shared_folder_url
                )

                if indexed_count > 0:
                    # Mark documents as indexed only if indexing actually happened
                    marked = db.query(Document).filter(Document.indexed == False).update({
                        "indexed": True,
                        "last_indexed": datetime.utcnow()
                    })
                    db.commit()
                    print(f"Marked {marked} documents as indexed")

            except Exception as e:
                print(f"Error during smart reindexing: {e}")
                # Fallback: mark as unindexed so it gets picked up later
                stats["reindex_error"] = str(e)
        else:
            print("No database changes detected, skipping reindexing")

        db.close()

        print(f"Document sync complete: {stats['added']} added, {stats['updated']} updated, {stats['unchanged']} unchanged, {stats['errors']} errors")
        return {
            "success": True,
            "stats": stats,
            "changes_made": changes_made
        }

    except Exception as e:
        print(f"Error during document sync: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def sync_and_reindex(force: bool = False) -> Dict[str, Any]:
    """
    Connect database document records with RAG indexing system.
    1. Sync files to database
    2. Check if reindexing is needed
    3. Trigger RAG reindexing if needed
    """
    print("Starting document sync and reindex process...")

    try:
        # First sync files to database
        db = SessionLocal()

        # Scan and sync files
        sync_stats = sync_all_local_files(db)

        # Check if reindexing is needed
        unindexed_count = db.query(Document).filter(Document.indexed == False).count()
        reindex_needed = unindexed_count > 0 or force

        # If reindexing is needed, use the RAG engine's method
        if reindex_needed:
            print(f"Reindexing needed (unindexed: {unindexed_count}, force: {force})")
            # Force rebuild the index using the RAG engine
            reindex_result = rag_engine_test.refresh_document_index(force_rebuild=True)

            # Mark documents as indexed
            db.query(Document).filter(Document.indexed == False).update(
                {"indexed": True, "last_indexed": datetime.utcnow()}
            )
            db.commit()

            result = {
                "success": True,
                "message": f"Synced and reindexed {unindexed_count} documents",
                "sync_stats": sync_stats,
                "reindex_result": reindex_result
            }
        else:
            # No reindexing needed
            print("No changes detected, skipping reindex")
            result = {
                "success": True,
                "message": "No changes detected, using existing index",
                "sync_stats": sync_stats,
                "reindex_result": None
            }

        db.close()
        return result

    except Exception as e:
        print(f"Error during sync and reindex: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def sync_documents(force: bool = False) -> Dict[str, Any]:
    """
    Legacy function maintained for compatibility
    Now just calls sync_all_documents_to_db
    """
    return sync_all_documents_to_db()



def sync_all_local_files(db: Session) -> Dict[str, int]:
    """Scan all files in the data directory and update the database"""
    stats = {
        "added": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 0
    }

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created data directory at {data_dir}")
        return stats

    print(f"Scanning directory: {data_dir}")

    # Get all files in data directory
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            # Skip certain files
            if (file_name.startswith('.') or
                file_name == 'fingerprint.json' or
                file_name.endswith(('.pyc', '.tmp', '.DS_Store'))):
                continue

            file_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(file_path)

            # Skip if not a file
            if not os.path.isfile(file_path):
                continue

            try:
                # Determine source type based on path
                source_type = 'local'
                if 'onedrive' in rel_path.lower():
                    source_type = 'onedrive'
                elif 'uploads' in rel_path.lower():
                    source_type = 'uploaded'

                # Get file metadata
                metadata = get_file_metadata(file_path)

                # Check if document already exists in database
                db_doc = db.query(Document).filter(Document.file_path == rel_path).first()

                if not db_doc:
                    # Add new document to database
                    new_doc = Document(
                        title=file_name,
                        filename=file_name,
                        file_path=rel_path,
                        source_type=source_type,
                        file_type=metadata.get("file_type", ""),
                        size_bytes=metadata.get("size_bytes", 0),
                        content_preview=metadata.get("content_preview", ""),
                        file_hash=metadata.get("file_hash", ""),
                        indexed=False,
                        onedrive_link=os.getenv("ONEDRIVE_SHARED_URL", "") if source_type == 'onedrive' else None
                    )
                    db.add(new_doc)
                    stats["added"] += 1
                    print(f"Added new document to database: {rel_path}")

                else:
                    # Check if file has changed
                    current_hash = metadata.get("file_hash", "")
                    if not db_doc.file_hash or db_doc.file_hash != current_hash:
                        # File has changed, update record
                        db_doc.size_bytes = metadata.get("size_bytes", 0)
                        db_doc.content_preview = metadata.get("content_preview", "")
                        db_doc.file_hash = current_hash
                        db_doc.indexed = False  # Mark for reindexing
                        db_doc.updated_at = datetime.utcnow()
                        stats["updated"] += 1
                        print(f"Updated document in database: {rel_path}")
                    else:
                        stats["unchanged"] += 1

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                stats["errors"] += 1

    # Commit changes to database
    db.commit()

    return stats

if __name__ == "__main__":
    # This allows running the sync as a standalone script
    result = sync_all_documents_to_db()
    print(result)
