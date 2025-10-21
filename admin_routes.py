from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Header
import jwt
from jwt.exceptions import PyJWTError
from database import SECRET_KEY, ALGORITHM
from sqlalchemy.orm import Session
from database import get_db, get_password_hash, create_access_token
from typing import List, Dict, Any
from fastapi import Request ,UploadFile, File, Form, APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path, Body
import os
from sqlalchemy import Boolean, Column, Integer, String, DateTime, func, Text
from models import Admin, User, ContentSource
from fastapi import BackgroundTasks
import document_sync
import shutil
import uuid
import time
import hashlib
from schemas.Admin_schemas import (
    AdminResponse, PasswordReset, PasswordUpdate, Token, AdminCreateByAdmin,
    DocumentBase, DocumentCreate, DocumentUpdate, DocumentResponse
)
from models.Admin_model import Base as AdminBase, Document
from models.User_models import Base as UserBase
from schemas.User_schemas import UserResponse, UserCreate as UserCreateByAdmin, UserUpdate
from auth import (
    authenticate_admin, get_current_admin, generate_admin_password_reset_token,
    verify_admin_reset_token, get_admin_by_username, get_admin_by_email,
    get_user_by_username, get_user_by_email,admin_oauth2_scheme
)
from typing import List, Dict, Any, Optional
from datetime import datetime
import rag_engine_test
from rag_engine_test import OneDriveSharedFolderClient
import importlib.util
import sys

router = APIRouter()
onedrive_client = OneDriveSharedFolderClient()

#create an admin account
@router.post("/create", response_model=AdminResponse)
def create_admin_account(
    admin_data: AdminCreateByAdmin,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None)  # Read the Authorization header manually
):
    """
    Allow the creation of the first admin account without authentication.
    For subsequent admin accounts, authentication is required.
    """
    # Check if any admin exists
    admin_count = db.query(Admin).count()

    if admin_count > 0:
        # If admins already exist, enforce authentication
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to create additional admin accounts."
            )

        # Extract the token from the Authorization header
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme."
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format."
            )

        # Validate the token
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials"
                )
        except PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

        # Check if the admin exists and is active
        current_admin = db.query(Admin).filter(Admin.username == username).first()
        if not current_admin or not current_admin.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive or invalid admin account"
            )

    # Create new admin
    hashed_password = get_password_hash(admin_data.password)
    db_admin = Admin(
        username=admin_data.username,
        email=admin_data.email,
        full_name=admin_data.full_name,
        hashed_password=hashed_password,
        is_active=admin_data.is_active
    )

    db.add(db_admin)
    db.commit()
    db.refresh(db_admin)
    return db_admin

#Sign in admin
@router.post("/login", response_model=Token)
def login_admin(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    admin = authenticate_admin(db, form_data.username, form_data.password)
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login time
    admin.last_login = datetime.utcnow()
    db.commit()

    access_token = create_access_token(data={"sub": admin.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/logout")
def logout_admin(current_admin: Admin = Depends(get_current_admin)):
    # In a stateless JWT auth system, the actual logout happens on the client side
    # by removing the token, but we can log it server-side if needed
    return {"message": "Successfully logged out"}

@router.post("/reset-password")
def request_password_reset(reset_data: PasswordReset, db: Session = Depends(get_db)):
    token = generate_admin_password_reset_token(db, reset_data.email)
    if token:
        # In a real application, send an email with the reset link
        return {"message": "Password reset email sent", "token": token}
    return {"message": "If the email exists, a password reset link has been sent"}

@router.post("/update-password")
def update_password(password_data: PasswordUpdate, db: Session = Depends(get_db)):
    admin = verify_admin_reset_token(db, password_data.token)
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )

    # Update password
    admin.hashed_password = get_password_hash(password_data.new_password)
    admin.reset_token = None
    admin.reset_token_expires = None
    db.commit()

    return {"message": "Password updated successfully"}

@router.get("/profile", response_model=AdminResponse)
def get_admin_profile(current_admin: Admin = Depends(get_current_admin)):
    return current_admin

#User Management
#create user account
@router.post("/create-user", response_model=UserResponse)
def create_user_account(
    user_data: UserCreateByAdmin,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Endpoint for admins to create new user accounts"""

    # Check if username already exists
    db_user = get_user_by_username(db, username=user_data.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # Check if email already exists
    db_user = get_user_by_email(db, email=user_data.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        is_active=True
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

 #List all users
@router.get("/users", response_model=List[UserResponse])
def list_users(
    skip: int = Query(0, description="Number of users to skip"),
    limit: int = Query(100, description="Maximum number of users to return"),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """List all users with pagination"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.get("/users/{user_id}", response_model=UserResponse)
def get_user_details(
    user_id: int = Path(..., description="The ID of the user to get"),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Get details for a specific user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

#Add new user
@router.put("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int = Path(..., description="The ID of the user to update"),
    user_data: UserUpdate = Depends(),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Update user information"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update only the fields that are provided
    if user_data.full_name is not None:
        user.full_name = user_data.full_name

    if user_data.email is not None and user_data.email != user.email:
        # Check if email is already used by another user
        existing_user = get_user_by_email(db, email=user_data.email)
        if existing_user and existing_user.id != user_id:
            raise HTTPException(status_code=400, detail="Email already registered")
        user.email = user_data.email

    if user_data.bio is not None:
        user.bio = user_data.bio

    if user_data.profile_picture is not None:
        user.profile_picture = user_data.profile_picture

    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)

    return user

#Delete user by ID
@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int = Path(..., description="The ID of the user to delete"),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Delete a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return None

# Content Management
# List all documents
@router.get("/documents", response_model=List[DocumentResponse])
def list_documents(
    skip: int = Query(0, description="Number of documents to skip"),
    limit: int = Query(100, description="Maximum number of documents to return"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    indexed: Optional[bool] = Query(None, description="Filter by indexed status"),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """List all documents with pagination and filtering"""
    query = db.query(Document)

    # Apply filters if provided
    if source_type:
        query = query.filter(Document.source_type == source_type)
    if file_type:
        query = query.filter(Document.file_type == file_type)
    if indexed is not None:
        query = query.filter(Document.indexed == indexed)

    # Get total count for pagination info
    total_count = query.count()

    # Apply pagination
    documents = query.order_by(Document.updated_at.desc()).offset(skip).limit(limit).all()

    return documents

#Get a specific document by ID
@router.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(
    doc_id: int = Path(..., description="The ID of the document"),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Get a specific document by ID"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

#update a specific document details in datbase
@router.put("/documents/{doc_id}", response_model=DocumentResponse)
def update_document(
    doc_id: int = Path(..., description="The ID of the document"),
    doc_data: DocumentUpdate = Body(...),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Update document metadata"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Update fields from request
    update_data = doc_data.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(doc, key, value)

    doc.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(doc)
    return doc

# Delete a document by ID
@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: int = Path(..., description="The ID of the document to delete"),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Delete a document from the database"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(doc)
    db.commit()
    return None

#Upload new doc
@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(..., description="The document file to upload."),
    title: Optional[str] = Form(None, description="An optional title for the document."),
    source_type: str = Form("uploaded", description="The source type of the document."),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):

    print("UPLOAD ENDPOINT CALLED")
    print("file:", file)
    print("title:", title)
    print("source_type:", source_type)

    """
    Upload a document file and add it to the Documents table.
    Expects a multipart/form-data request with a 'file' field.
    """
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename to avoid conflicts
        timestamp = int(time.time())
        file_extension = Path(file.filename).suffix.lower() if file.filename else ''
        safe_filename = file.filename or f"untitled_{timestamp}"
        unique_filename = f"{timestamp}_{hashlib.md5(safe_filename.encode()).hexdigest()[:10]}{file_extension}"
        file_path = uploads_dir / unique_filename

        # Save the uploaded file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get file size
        size_bytes = os.path.getsize(file_path)

        # Determine file type from extension
        file_type = file_extension.replace(".", "")

        # Use provided title or original filename
        document_title = title if title else safe_filename

        # Create a new Document record in the database
        document = Document(
            title=document_title,
            filename=safe_filename,
            file_path=str(file_path),
            source_type=source_type,
            file_type=file_type,
            size_bytes=size_bytes,
            indexed=False
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        return document

    except Exception as e:
        # Log the detailed error
        print(f"Error uploading document: {str(e)}")
        # Raise a general server error for the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while uploading the document: {str(e)}"
        )

#Smart reindex
@router.post("/documents/smart-reindex")
def smart_reindex(
    background_tasks: BackgroundTasks,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Smart reindexing that only rebuilds if changes are detected"""

    # Check for unindexed documents
    unindexed_count = db.query(Document).filter(Document.indexed == False).count()

    # Start background task
    def reindex_task():
        # First sync to make sure all files are in the database
        document_sync.sync_all_local_files(db=db)

        # Only reindex if needed
        if unindexed_count > 0:
            print(f"Found {unindexed_count} unindexed documents, rebuilding index...")
            rag_engine_test.refresh_document_index(force_rebuild=True)

            # Mark all documents as indexed
            db.query(Document).filter(Document.indexed == False).update({
                "indexed": True,
                "last_indexed": datetime.utcnow()
            })
            db.commit()
            print("Reindexing complete")
        else:
            print("No document changes detected, skipping reindexing")

    background_tasks.add_task(reindex_task)

    return {
        "message": "Smart reindex started",
        "unindexed_documents": unindexed_count,
        "reindex_needed": unindexed_count > 0
    }

# Debugging endpoint
@router.get("/documents/debug-status")
def debug_document_status(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Detailed debug information about documents and indexing"""
    # Query document stats
    total_docs = db.query(Document).count()
    indexed_docs = db.query(Document).filter(Document.indexed == True).count()
    unindexed_docs = db.query(Document).filter(Document.indexed == False).count()

    # Get docs by source type
    onedrive_docs = db.query(Document).filter(Document.source_type == 'onedrive').count()
    local_docs = db.query(Document).filter(Document.source_type == 'local').count()
    uploaded_docs = db.query(Document).filter(Document.source_type == 'uploaded').count()

    # Check data directory
    data_dir = "data"
    file_count = 0
    for root, _, files in os.walk(data_dir):
        file_count += len([f for f in files if not f.startswith('.') and not f.endswith('.pyc')])

    # Get most recent documents
    recent_docs = []
    for doc in db.query(Document).order_by(Document.updated_at.desc()).limit(5):
        recent_docs.append({
            "id": doc.id,
            "title": doc.title,
            "path": doc.file_path,
            "source_type": doc.source_type,
            "indexed": doc.indexed,
            "last_updated": doc.updated_at.isoformat() if doc.updated_at else None
        })

    # Check for common problems
    warnings = []
    if file_count > total_docs:
        warnings.append(f"Found {file_count} files but only {total_docs} in database. Run sync to add missing files.")
    if unindexed_docs > 0:
        warnings.append(f"Found {unindexed_docs} unindexed documents. Consider reindexing.")

    return {
        "document_count": {
            "total": total_docs,
            "indexed": indexed_docs,
            "unindexed": unindexed_docs
        },
        "by_source": {
            "onedrive": onedrive_docs,
            "local": local_docs,
            "uploaded": uploaded_docs
        },
        "filesystem": {
            "total_files": file_count
        },
        "recent_documents": recent_docs,
        "warnings": warnings
    }

# statistiques API
@router.get("/document-statistics", response_model=Dict[str, Any])
def get_document_statistics(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive document statistics including counts by type,
    source, indexing status, and last update time.
    """
    stats = {}

    # Total document count
    stats["total"] = db.query(Document).count()

    # Count by source type (onedrive, local, uploaded)
    source_counts = db.query(Document.source_type, func.count(Document.id)).\
                    group_by(Document.source_type).all()
    stats["by_source"] = {src: count for src, count in source_counts}

    # Count by file type (pdf, txt, docx, etc.)
    type_counts = db.query(Document.file_type, func.count(Document.id)).\
                  group_by(Document.file_type).all()
    stats["by_type"] = {file_type: count for file_type, count in type_counts}

    # Count of indexed documents
    stats["indexed"] = db.query(Document).filter(Document.indexed == True).count()

    # Count of unindexed documents
    stats["unindexed"] = db.query(Document).filter(Document.indexed == False).count()

    # Last sync time (based on most recently updated document)
    latest_doc = db.query(Document).order_by(Document.updated_at.desc()).first()
    stats["last_sync"] = latest_doc.updated_at.isoformat() if latest_doc and latest_doc.updated_at else None

    # Recent activity
    recent_updates = db.query(Document.updated_at).\
                     order_by(Document.updated_at.desc()).\
                     limit(5).all()
    stats["recent_updates"] = [update[0].isoformat() for update in recent_updates if update[0]]

    return stats
