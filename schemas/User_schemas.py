from pydantic import BaseModel, EmailStr, validator
from typing import Optional ,List
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str

    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    profile_picture: Optional[str] = None
    bio: Optional[str] = None

class UserResponse(UserBase):
    id: int
    is_active: bool
    profile_picture: Optional[str] = None
    bio: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int

class TokenData(BaseModel):
    username: Optional[str] = None

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordUpdate(BaseModel):
    token: str
    new_password: str


class ChatRequest(BaseModel):
    message: str
    mode: str = "rag"

class ChatResponse(BaseModel):
    message: str
    mode: str
    processing_time: float
    sources: list
    available_modes: list
    chat_id: int

class ChatHistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    created_at: datetime

    class Config:
        from_attributes = True

class FeedbackCreate(BaseModel):
    chat_id: Optional[int] = None
    rating: int  # 1-5
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: int
    user_id: int
    chat_id: Optional[int]
    rating: int
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
