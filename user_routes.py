from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import os
import shutil
from uuid import uuid4
from datetime import datetime

from database import get_db, get_password_hash, create_access_token
from models import User
from schemas.User_schemas import ChatRequest,ChatResponse, UserCreate, UserResponse, UserUpdate, Token, PasswordReset, PasswordUpdate
from auth import (
    authenticate_user, get_current_user, get_user_by_email, get_user_by_username,
    generate_user_password_reset_token, verify_user_reset_token
)
from typing import List
from models.User_models import User, ChatHistory, Feedback
from schemas.User_schemas import ChatHistoryItem, FeedbackCreate, FeedbackResponse
from typing import Dict, Any, List
import rag_engine_test
router = APIRouter()

# User registration
@router.post("/register", response_model=UserResponse)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
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
        hashed_password=hashed_password
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# User login
@router.post("/login", response_model=Token)
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login time
    user.last_login = datetime.utcnow()
    db.commit()

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.id}

# User logout
@router.post("/logout")
def logout_user(current_user: User = Depends(get_current_user)):
    # In a stateless JWT auth system, the actual logout happens on the client side
    # by removing the token, but we can log it server-side if needed
    return {"message": "Successfully logged out"}

# Request password reset
@router.post("/forgot-password")
def forgot_password(reset_data: PasswordReset, db: Session = Depends(get_db)):
    token = generate_user_password_reset_token(db, reset_data.email)
    if token:
        # In a real application, send an email with the reset link
        # For this example, we'll just return the token
        return {"message": "Password reset email sent", "token": token}
    return {"message": "If the email exists, a password reset link has been sent"}

# Reset password
@router.post("/reset-password")
def reset_password(password_data: PasswordUpdate, db: Session = Depends(get_db)):
    user = verify_user_reset_token(db, password_data.token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )

    # Update password
    user.hashed_password = get_password_hash(password_data.new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.commit()

    return {"message": "Password updated successfully"}

# Get user profile
@router.get("/profile", response_model=UserResponse)
def get_user_profile(current_user: User = Depends(get_current_user)):
    return current_user

# Update user profile
@router.put("/profile", response_model=UserResponse)
def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Update fields that are provided
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name

    if user_update.email is not None and user_update.email != current_user.email:
        # Check if email is already used
        db_user = get_user_by_email(db, email=user_update.email)
        if db_user and db_user.id != current_user.id:
            raise HTTPException(status_code=400, detail="Email already registered")
        current_user.email = user_update.email

    if user_update.bio is not None:
        current_user.bio = user_update.bio

    if user_update.profile_picture is not None:
        current_user.profile_picture = user_update.profile_picture

    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)

    return current_user

#chat
@router.post("/chat", response_model=ChatResponse)
def chat_with_rag_bot(
    chat_req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    message = chat_req.message
    mode = chat_req.mode

    valid_modes = ["rag", "web"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {', '.join(valid_modes)}"
        )

    # Format query based on mode
    formatted_query = message
    if mode == "rag":
        formatted_query = f"/rag {message}"
    elif mode == "web":
        formatted_query = f"/web {message}"

    try:
        result = rag_engine_test.process_user_query(formatted_query)
        bot_answer = result["answer"]

        # Save chat history
        chat = ChatHistory(
            user_id=current_user.id,
            question=message,
            answer=bot_answer
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)

        return ChatResponse(
            message=bot_answer,
            mode=result["mode"],
            processing_time=result["elapsed_sec"],
            sources=result.get("sources", []),
            available_modes=valid_modes,
            chat_id=chat.id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat message: {str(e)}"
        )


# GET /api/users/chat-history
@router.get("/chat-history", response_model=List[ChatHistoryItem])
def get_chat_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    history = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == current_user.id)
        .order_by(ChatHistory.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return history

# POST /api/users/feedback
@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    feedback: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Optionally, validate chat_id belongs to user
    if feedback.chat_id:
        chat = db.query(ChatHistory).filter(
            ChatHistory.id == feedback.chat_id,
            ChatHistory.user_id == current_user.id
        ).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat history not found")

    fb = Feedback(
        user_id=current_user.id,
        chat_id=feedback.chat_id,
        rating=feedback.rating,
        comment=feedback.comment
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb
