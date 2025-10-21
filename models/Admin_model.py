from sqlalchemy import Boolean, Column, Integer, String, DateTime, func, Text
from database import Base

class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)
    reset_token = Column(String, nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)

class ContentSource(Base):
    __tablename__ = "content_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    source_type = Column(String)  # 'onedrive', 'local', 'web', etc.
    path = Column(String)  # URL, local path, or OneDrive URL
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    last_indexed = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    filename = Column(String)
    file_path = Column(String)  # Local path or OneDrive path
    source_type = Column(String)  # 'onedrive', 'local', 'uploaded'
    file_type = Column(String)  # 'pdf', 'doc', 'txt', etc.
    size_bytes = Column(Integer)
    onedrive_link = Column(String, nullable=True)
    indexed = Column(Boolean, default=False)
    last_indexed = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    content_preview = Column(Text, nullable=True)  # First few lines or summary
    file_hash = Column(String, nullable=True)  # Store file hash for change detection
