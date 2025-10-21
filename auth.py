from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import jwt
from jwt.exceptions import PyJWTError
from datetime import datetime, timedelta
import secrets

from database import get_db, SECRET_KEY, ALGORITHM, verify_password, create_access_token
from models import Admin, User
from schemas.Admin_schemas import TokenData as AdminTokenData
from schemas.User_schemas import TokenData as UserTokenData, UserUpdate
import os
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
from datetime import datetime

import requests
from dotenv import load_dotenv
# Global Settings
load_dotenv()
GOOGLE_API_KEY        = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID         = os.getenv("GOOGLE_CSE_ID")
MS_TENANT_ID          = os.getenv("MS_TENANT_ID")
MS_CLIENT_ID          = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET      = os.getenv("MS_CLIENT_SECRET")
# OAuth2 schemes for different login endpoints
admin_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/admin/login")
user_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/users/login")

# Admin authentication functions
def get_admin_by_username(db: Session, username: str):
    return db.query(Admin).filter(Admin.username == username).first()

def get_admin_by_email(db: Session, email: str):
    return db.query(Admin).filter(Admin.email == email).first()

def authenticate_admin(db: Session, username: str, password: str):
    admin = get_admin_by_username(db, username)
    if not admin:
        return False
    if not verify_password(password, admin.hashed_password):
        return False
    return admin


def get_current_admin(token: Optional[str] = Depends(admin_oauth2_scheme), db: Session = Depends(get_db)):
    if token is None:
        return None  # No token provided, return None

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = AdminTokenData(username=username)
    except PyJWTError:
        raise credentials_exception

    admin = get_admin_by_username(db, username=token_data.username)
    if admin is None:
        raise credentials_exception
    if not admin.is_active:
        raise HTTPException(status_code=400, detail="Inactive admin account")
    return admin

def generate_admin_password_reset_token(db: Session, email: str):
    admin = get_admin_by_email(db, email)
    if not admin:
        # Don't reveal that the email doesn't exist
        return None

    # Generate a token
    reset_token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(hours=24)

    # Update admin with token
    admin.reset_token = reset_token
    admin.reset_token_expires = expires
    db.commit()

    return reset_token

def verify_admin_reset_token(db: Session, token: str):
    admin = db.query(Admin).filter(
        Admin.reset_token == token,
        Admin.reset_token_expires > datetime.utcnow()
    ).first()

    return admin

# User authentication functions
def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_user_by_id(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_current_user(token: str = Depends(user_oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = UserTokenData(username=username)
    except PyJWTError:
        raise credentials_exception

    user = get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user account")
    return user

def generate_user_password_reset_token(db: Session, email: str):
    user = get_user_by_email(db, email)
    if not user:
        # Don't reveal that the email doesn't exist
        return None

    # Generate a token
    reset_token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(hours=24)

    # Update user with token
    user.reset_token = reset_token
    user.reset_token_expires = expires
    db.commit()

    return reset_token

def verify_user_reset_token(db: Session, token: str):
    user = db.query(User).filter(
        User.reset_token == token,
        User.reset_token_expires > datetime.utcnow()
    ).first()

    return user
