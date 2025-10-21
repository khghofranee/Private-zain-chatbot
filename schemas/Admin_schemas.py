from pydantic import BaseModel, EmailStr,validator
from typing import Optional
from datetime import datetime

class AdminBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str

class AdminCreate(AdminBase):
    password: str

class AdminResponse(AdminBase):
    id: int
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True

class AdminLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordUpdate(BaseModel):
    token: str
    new_password: str

class AdminCreateByAdmin(AdminCreate):
    is_active: bool = True

# Content source schemas
class ContentSourceBase(BaseModel):
    name: str
    source_type: str
    path: str
    description: Optional[str] = None
    is_active: Optional[bool] = True

    @validator('source_type')
    def validate_source_type(cls, v):
        allowed_types = ['onedrive', 'local', 'web']
        if v not in allowed_types:
            raise ValueError(f"Source type must be one of: {', '.join(allowed_types)}")
        return v

class ContentSourceCreate(ContentSourceBase):
    pass

class ContentSourceUpdate(BaseModel):
    name: Optional[str] = None
    path: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

class ContentSourceResponse(ContentSourceBase):
    id: int
    last_indexed: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DocumentBase(BaseModel):
    title: str
    filename: str
    file_path: str
    source_type: str
    file_type: str
    size_bytes: int
    onedrive_link: Optional[str] = None
    content_preview: Optional[str] = None
    file_hash: Optional[str] = None

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    onedrive_link: Optional[str] = None
    content_preview: Optional[str] = None

class DocumentResponse(DocumentBase):
    id: int
    indexed: bool
    last_indexed: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
