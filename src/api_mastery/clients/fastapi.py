# Module 5: Building APIs with FastAPI
# Your API Mastercy Journey - Days 16-20

import asyncio
import uvicorn
import logging
import jwt
import pytest
import tempfile
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Generator
from contextlib import asynccontextmanager
from unittest.mock import patch

from fastapi import FastAPI, HTTPException, Depends, status, Request, Body, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Table, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Part 1: FastAPI Fundamentals - Building Your First API (90 minutes)

# Pydantic models for request/response validation
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=20, description="Username (3-20 characters)")
    email: EmailStr = Field(..., description="Valid email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None

class PostBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=5000)
    tags: List[str] = Field(default=[], max_items=10)

class PostCreate(PostBase):
    pass

class PostResponse(PostBase):
    id: int
    author_id: int
    created_at: datetime
    updated_at: datetime
    view_count: int = 0

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None

class CommentBase(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000)

class CommentCreate(CommentBase):
    pass

class CommentResponse(CommentBase):
    id: int
    post_id: int
    author_id: int
    is_approved: bool
    created_at: datetime

    class Config:
        from_attributes = True

class TagResponse(BaseModel):
    id: int
    name: str
    created_at: datetime

    class Config:
        from_attributes = True

class PostResponseWithDetails(PostBase):
    id: int
    author_id: int
    view_count: int
    is_published: bool
    created_at: datetime
    updated_at: datetime
    author: UserResponse
    tags: List[TagResponse] = []
    comments: List[CommentResponse] = []

    class Config:
        from_attributes = True

# Authentication models
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

# In-memory database (replace with real database in production)
fake_db = {
    'users': {},
    'posts': {},
    'next_user_id': 1,
    'next_post_id': 1
}

# Security
security = HTTPBearer()

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("🚀 Starting Blog API")
    await startup_event()
    yield
    # Application runs here
    # Shutdown
    logger.info("⏹️ Shutting down Blog API")
    await shutdown_event()

# Create FastAPI application
app = FastAPI(
    title="Professional Blog API",
    description="A production-ready blog API built with FastAPI",
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@blogapi.com",
        "url": "https://blogapi.com/support"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json"
)

async def startup_event():
    """Run on application startup"""
    logger.info("Initializing database connections...")
    logger.info("Loading configuration...")
    logger.info("Setting up monitoring...")
    
    # Create sample data
    sample_user = {
        'id': 1,
        'username': 'admin',
        'email': 'admin@example.com',
        'full_name': 'System Administrator',
        'password': 'hashed_admin_password',  # In production: use proper hashing
        'is_active': True,
        'created_at': datetime.now()
    }
    fake_db['users'][1] = sample_user
    fake_db['next_user_id'] = 2
    
    logger.info("✅ Application startup completed")

async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Closing database connections...")
    logger.info("Cleaning up resources...")
    logger.info("✅ Application shutdown completed")

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Get current authenticated user"""
    token = credentials.credentials
    # In production: validate JWT token properly
    if token == "valid_token":
        return fake_db['users'][1]  # Return admin user for demo
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

# Optional authentication (for endpoints that can work with/without auth)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[Dict]:
    """Get current user if authenticated, None otherwise"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

# Root endpoint
@app.get("/", tags=["Root"])
async def read_root():
    """API health check and information"""
    return {
        "message": "Welcome to Professional Blog API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "documentation": "/docs",
        "users_count": len(fake_db['users']),
        "posts_count": len(fake_db['posts'])
    }

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check for monitoring systems"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "database": "connected",  # In production: actual DB health check
        "memory_usage": "normal",
        "response_time_ms": 1.5
    }

# User endpoints
@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, 
          tags=["Users"], summary="Create a new user", 
          description="Create a new user account with username, email, and password")
async def create_user(user: UserCreate):
    """Create a new user"""
    # Check if username already exists
    for existing_user in fake_db['users'].values():
        if existing_user['username'] == user.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        if existing_user['email'] == user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Create new user
    user_id = fake_db['next_user_id']
    new_user = {
        'id': user_id,
        'username': user.username,
        'email': user.email,
        'full_name': user.full_name,
        'password': f"hashed_{user.password}",  # In production: use proper hashing
        'is_active': True,
        'created_at': datetime.now()
    }
    fake_db['users'][user_id] = new_user
    fake_db['next_user_id'] += 1
    
    logger.info(f"Created new user: {user.username} (ID: {user_id})")
    return UserResponse(**new_user)

@app.get("/users", response_model=List[UserResponse], tags=["Users"], 
         dependencies=[Depends(get_current_user)])
async def get_users(
    skip: int = Field(0, ge=0, description="Number of users to skip"),
    limit: int = Field(10, ge=1, le=100, description="Maximum number of users to return")
):
    """Get list of users (authenticated users only)"""
    users = list(fake_db['users'].values())
    return users[skip:skip + limit]

@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def get_user(user_id: int, current_user: Optional[Dict] = Depends(get_current_user_optional)):
    """Get user by ID"""
    if user_id not in fake_db['users']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user = fake_db['users'][user_id]
    
    # Log access (useful for analytics)
    logger.info(f"User {user_id} profile accessed by {'authenticated user' if current_user else 'anonymous'}")
    return UserResponse(**user)

@app.put("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: Dict = Depends(get_current_user)
):
    """Update user information (user must be authenticated)"""
    # Check if user exists
    if user_id not in fake_db['users']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Authorization: users can only update their own profile (or admin can update any)
    if current_user['id'] != user_id and current_user['username'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user"
        )
    
    # Update user
    user = fake_db['users'][user_id]
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        user[field] = value
    
    logger.info(f"User {user_id} updated by user {current_user['id']}")
    return UserResponse(**user)

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Users"])
async def delete_user(
    user_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """Delete user (admin only)"""
    # Authorization: only admin can delete users
    if current_user['username'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can delete users"
        )
    
    if user_id not in fake_db['users']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Don't let admin delete themselves
    if user_id == current_user['id']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    del fake_db['users'][user_id]
    logger.info(f"User {user_id} deleted by admin {current_user['id']}")

# Post endpoints
@app.post("/posts", response_model=PostResponse, status_code=status.HTTP_201_CREATED, tags=["Posts"])
async def create_post(
    post: PostCreate,
    current_user: Dict = Depends(get_current_user)
):
    """Create a new blog post (authenticated users only)"""
    post_id = fake_db['next_post_id']
    new_post = {
        'id': post_id,
        'title': post.title,
        'content': post.content,
        'tags': post.tags,
        'author_id': current_user['id'],
        'created_at': datetime.now(),
        'updated_at': datetime.now(),
        'view_count': 0
    }
    fake_db['posts'][post_id] = new_post
    fake_db['next_post_id'] += 1
    
    logger.info(f"Post {post_id} created by user {current_user['id']}")
    return PostResponse(**new_post)

@app.get("/posts", response_model=List[PostResponse], tags=["Posts"])
async def get_posts(
    skip: int = Field(0, ge=0),
    limit: int = Field(10, ge=1, le=100),
    tag: Optional[str] = Field(None, description="Filter by tag"),
    author_id: Optional[int] = Field(None, description="Filter by author ID"),
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """Get list of blog posts with optional filtering"""
    posts = list(fake_db['posts'].values())
    
    # Apply filters
    if tag:
        posts = [p for p in posts if tag in p['tags']]
    if author_id:
        posts = [p for p in posts if p['author_id'] == author_id]
    
    # Sort by creation date (newest first)
    posts.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Apply pagination
    paginated_posts = posts[skip:skip + limit]
    
    logger.info(f"Posts endpoint accessed: {len(paginated_posts)} posts returned")
    return [PostResponse(**post) for post in paginated_posts]

@app.get("/posts/{post_id}", response_model=PostResponse, tags=["Posts"])
async def get_post(
    post_id: int,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """Get a specific blog post by ID"""
    if post_id not in fake_db['posts']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    post = fake_db['posts'][post_id]
    # Increment view count
    post['view_count'] += 1
    
    logger.info(f"Post {post_id} viewed (total views: {post['view_count']})")
    return PostResponse(**post)

@app.put("/posts/{post_id}", response_model=PostResponse, tags=["Posts"])
async def update_post(
    post_id: int,
    post_update: PostUpdate,
    current_user: Dict = Depends(get_current_user)
):
    """Update a blog post (author or admin only)"""
    if post_id not in fake_db['posts']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    post = fake_db['posts'][post_id]
    
    # Authorization: only author or admin can update
    if post['author_id'] != current_user['id'] and current_user['username'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this post"
        )
    
    # Update post
    update_data = post_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        post[field] = value
    post['updated_at'] = datetime.now()
    
    logger.info(f"Post {post_id} updated by user {current_user['id']}")
    return PostResponse(**post)

@app.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Posts"])
async def delete_post(
    post_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """Delete a blog post (author or admin only)"""
    if post_id not in fake_db['posts']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    post = fake_db['posts'][post_id]
    
    # Authorization: only author or admin can delete
    if post['author_id'] != current_user['id'] and current_user['username'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this post"
        )
    
    del fake_db['posts'][post_id]
    logger.info(f"Post {post_id} deleted by user {current_user['id']}")

# Statistics endpoint
@app.get("/stats", tags=["Analytics"], dependencies=[Depends(get_current_user)])
async def get_statistics(current_user: Dict = Depends(get_current_user)):
    """Get API usage statistics (authenticated users only)"""
    posts = list(fake_db['posts'].values())
    users = list(fake_db['users'].values())
    
    # Calculate statistics
    total_views = sum(post['view_count'] for post in posts)
    active_users = sum(1 for user in users if user['is_active'])
    
    # Top authors
    author_post_counts = {}
    for post in posts:
        author_id = post['author_id']
        author_post_counts[author_id] = author_post_counts.get(author_id, 0) + 1
    
    # Most popular tags
    tag_counts = {}
    for post in posts:
        for tag in post['tags']:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'total_users': len(users),
        'active_users': active_users,
        'total_posts': len(posts),
        'total_views': total_views,
        'average_views_per_post': total_views / len(posts) if posts else 0,
        'top_tags': top_tags,
        'most_prolific_author_id': max(author_post_counts.keys(), key=author_post_counts.get) if author_post_counts else None
    }

# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    logger.error(f"ValueError in {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Part 2: Advanced Authentication & Authorization (75 minutes)

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

# JWT utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, token_type: str = "access") -> Optional[str]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type_in_payload: str = payload.get("type")
        if username is None or token_type_in_payload != token_type:
            return None
        return username
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None

# User authentication functions
def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user with username and password"""
    # Find user
    user = None
    for u in fake_db['users'].values():
        if u['username'] == username:
            user = u
            break
    
    if not user:
        return None
    
    # For demo, check if password matches "password_" + username
    # In production, use verify_password(password, user['password'])
    if not verify_password(password, user['password']):
        return None
    
    return user

# Enhanced authentication dependency
async def get_current_user_jwt(token: str = Depends(oauth2_scheme)) -> Dict:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    username = verify_token(token, "access")
    if username is None:
        raise credentials_exception
    
    # Find user
    user = None
    for u in fake_db['users'].values():
        if u['username'] == username:
            user = u
            break
    
    if user is None:
        raise credentials_exception
    
    if not user['is_active']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

# Role-based authorization
class Roles:
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

def require_role(required_role: str):
    """Dependency factory for role-based authorization"""
    async def role_checker(current_user: Dict = Depends(get_current_user_jwt)) -> Dict:
        user_role = current_user.get('role', 'user')
        
        # Admin can access everything
        if user_role == Roles.ADMIN:
            return current_user
        
        # Check specific role
        if user_role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires {required_role} role"
            )
        
        return current_user
    
    return role_checker

# Permission-based authorization
class Permissions:
    READ_POSTS = "read_posts"
    WRITE_POSTS = "write_posts"
    DELETE_POSTS = "delete_posts"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"

ROLE_PERMISSIONS = {
    Roles.ADMIN: [
        Permissions.READ_POSTS,
        Permissions.WRITE_POSTS,
        Permissions.DELETE_POSTS,
        Permissions.MANAGE_USERS,
        Permissions.VIEW_ANALYTICS
    ],
    Roles.MODERATOR: [
        Permissions.READ_POSTS,
        Permissions.WRITE_POSTS,
        Permissions.DELETE_POSTS,
        Permissions.VIEW_ANALYTICS
    ],
    Roles.USER: [
        Permissions.READ_POSTS,
        Permissions.WRITE_POSTS
    ]
}

def require_permission(required_permission: str):
    """Dependency factory for permission-based authorization"""
    async def permission_checker(current_user: Dict = Depends(get_current_user_jwt)) -> Dict:
        user_role = current_user.get('role', 'user')
        user_permissions = ROLE_PERMISSIONS.get(user_role, [])
        
        if required_permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires {required_permission} permission"
            )
        
        return current_user
    
    return permission_checker

# Create authentication router
from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

@auth_router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate):
    """Register a new user with password hashing"""
    # Check if user exists
    for existing_user in fake_db['users'].values():
        if existing_user['username'] == user.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        if existing_user['email'] == user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Create new user with hashed password
    user_id = fake_db['next_user_id']
    hashed_password = get_password_hash(user.password)
    new_user = {
        'id': user_id,
        'username': user.username,
        'email': user.email,
        'full_name': user.full_name,
        'password': hashed_password,
        'is_active': True,
        'role': 'user',  # Default role
        'created_at': datetime.now()
    }
    fake_db['users'][user_id] = new_user
    fake_db['next_user_id'] += 1
    
    logger.info(f"New user registered: {user.username} (ID: {user_id})")
    return UserResponse(**new_user)

@auth_router.post("/login", response_model=Token)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT tokens"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user['is_active']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": user['username']})
    refresh_token = create_refresh_token(data={"sub": user['username']})
    
    logger.info(f"User {user['username']} logged in successfully")
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@auth_router.post("/refresh", response_model=Token)
async def refresh_access_token(refresh_token: str = Body(..., embed=True)):
    """Refresh access token using refresh token"""
    username = verify_token(refresh_token, "refresh")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Find user
    user = None
    for u in fake_db['users'].values():
        if u['username'] == username:
            user = u
            break
    
    if not user or not user['is_active']:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    access_token = create_access_token(data={"sub": user['username']})
    new_refresh_token = create_refresh_token(data={"sub": user['username']})
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@auth_router.post("/logout", status_code=status.HTTP_200_OK)
async def logout_user(current_user: Dict = Depends(get_current_user_jwt)):
    """Logout user (in production, implement token blacklisting)"""
    logger.info(f"User {current_user['username']} logged out")
    # In production:
    # 1. Add tokens to blacklist
    # 2. Clear session/cookies
    # 3. Revoke refresh token
    return {"message": "Successfully logged out"}

@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: Dict = Depends(get_current_user_jwt)):
    """Get current user's profile"""
    return UserResponse(**current_user)

# Add auth router to main app
app.include_router(auth_router)

# Update existing endpoints to use JWT authentication
# Replace get_current_user with get_current_user_jwt in existing endpoints

# Example of using role/permission-based authorization
@app.get("/admin/users", response_model=List[UserResponse], 
         dependencies=[Depends(require_role(Roles.ADMIN))])
async def admin_get_all_users():
    """Get all users (admin only)"""
    return [UserResponse(**user) for user in fake_db['users'].values()]

@app.delete("/admin/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT, 
            dependencies=[Depends(require_permission(Permissions.DELETE_POSTS))])
async def admin_delete_post(post_id: int):
    """Delete any post (moderator or admin only)"""
    if post_id not in fake_db['posts']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    del fake_db['posts'][post_id]
    logger.info(f"Post {post_id} deleted by moderator/admin")

# API Key authentication (for external integrations)
API_KEYS = {
    "service_key_123": {"name": "External Service", "permissions": ["read_posts"]},
    "admin_key_456": {"name": "Admin Service", "permissions": ["read_posts", "write_posts", "manage_users"]}
}

async def get_api_key(api_key: str = Header(None, alias="X-API-Key")) -> Optional[Dict]:
    """Validate API key"""
    if api_key and api_key in API_KEYS:
        return API_KEYS[api_key]
    return None

def require_api_key_permission(required_permission: str):
    """Require specific API key permission"""
    async def api_key_checker(api_key_info: Dict = Depends(get_api_key)) -> Dict:
        if not api_key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Valid API key required"
            )
        
        if required_permission not in api_key_info["permissions"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key lacks {required_permission} permission"
            )
        
        return api_key_info
    
    return api_key_checker

# API endpoint for external services
@app.get("/api/posts", response_model=List[PostResponse], 
         dependencies=[Depends(require_api_key_permission("read_posts"))])
async def api_get_posts(
    limit: int = Field(10, ge=1, le=100),
    api_key_info: Dict = Depends(get_api_key)
):
    """Get posts via API key (for external integrations)"""
    posts = list(fake_db['posts'].values())
    posts.sort(key=lambda x: x['created_at'], reverse=True)
    
    logger.info(f"API posts accessed by {api_key_info['name']}")
    return [PostResponse(**post) for post in posts[:limit]]

# Part 3: Database Integration & Advanced Features (90 minutes)

# Database configuration
DATABASE_URL = "sqlite:///./blog_api.db"  # Use PostgreSQL in production
# DATABASE_URL = "postgresql://user:password@localhost/blog_db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=True  # Log SQL queries (disable in production)
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Association table for many-to-many relationship between posts and tags
post_tags = Table(
    'post_tags',
    Base.metadata,
    Column('post_id', Integer, ForeignKey('posts.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

# Database models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(String(50), default="user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="author", cascade="all, delete-orphan")

class Tag(Base):
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    posts = relationship("Post", secondary=post_tags, back_populates="tags")

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False, index=True)
    content = Column(Text, nullable=False)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    view_count = Column(Integer, default=0)
    is_published = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    author = relationship("User", back_populates="posts")
    tags = relationship("Tag", secondary=post_tags, back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

class Comment(Base):
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=False)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    is_approved = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    post = relationship("Post", back_populates="comments")
    author = relationship("User", back_populates="comments")

# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Repository pattern for data access
class UserRepository:
    """Data access layer for users"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()
    
    def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.db.query(User).offset(skip).limit(limit).all()
    
    def create_user(self, user_data: dict) -> User:
        user = User(**user_data)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def update_user(self, user: User, update_data: dict) -> User:
        for field, value in update_data.items():
            setattr(user, field, value)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def delete_user(self, user: User) -> None:
        self.db.delete(user)
        self.db.commit()

class PostRepository:
    """Data access layer for posts"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_post_by_id(self, post_id: int) -> Optional[Post]:
        return self.db.query(Post).filter(Post.id == post_id).first()
    
    def get_posts(self, skip: int = 0, limit: int = 100, author_id: Optional[int] = None, 
                  tag_name: Optional[str] = None, published_only: bool = True) -> List[Post]:
        query = self.db.query(Post)
        
        if published_only:
            query = query.filter(Post.is_published == True)
        
        if author_id:
            query = query.filter(Post.author_id == author_id)
        
        if tag_name:
            query = query.join(Post.tags).filter(Tag.name == tag_name)
        
        return query.order_by(Post.created_at.desc()).offset(skip).limit(limit).all()
    
    def create_post(self, post_data: dict, tag_names: List[str] = None) -> Post:
        # Create post
        post = Post(**post_data)
        self.db.add(post)
        self.db.flush()  # Get the ID without committing
        
        # Handle tags
        if tag_names:
            for tag_name in tag_names:
                tag = self.db.query(Tag).filter(Tag.name == tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    self.db.add(tag)
                    self.db.flush()
                post.tags.append(tag)
        
        self.db.commit()
        self.db.refresh(post)
        return post
    
    def update_post(self, post: Post, update_data: dict, tag_names: List[str] = None) -> Post:
        # Update basic fields
        for field, value in update_data.items():
            if field != 'tags':  # Handle tags separately
                setattr(post, field, value)
        
        # Update tags if provided
        if tag_names is not None:
            post.tags.clear()  # Remove existing tags
            for tag_name in tag_names:
                tag = self.db.query(Tag).filter(Tag.name == tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    self.db.add(tag)
                    self.db.flush()
                post.tags.append(tag)
        
        self.db.commit()
        self.db.refresh(post)
        return post
    
    def increment_view_count(self, post: Post) -> Post:
        post.view_count += 1
        self.db.commit()
        self.db.refresh(post)
        return post
    
    def delete_post(self, post: Post) -> None:
        self.db.delete(post)
        self.db.commit()

class CommentRepository:
    """Data access layer for comments"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_comments_for_post(self, post_id: int, approved_only: bool = True) -> List[Comment]:
        query = self.db.query(Comment).filter(Comment.post_id == post_id)
        if approved_only:
            query = query.filter(Comment.is_approved == True)
        return query.order_by(Comment.created_at.asc()).all()
    
    def create_comment(self, comment_data: dict) -> Comment:
        comment = Comment(**comment_data)
        self.db.add(comment)
        self.db.commit()
        self.db.refresh(comment)
        return comment
    
    def update_comment(self, comment: Comment, update_data: dict) -> Comment:
        for field, value in update_data.items():
            setattr(comment, field, value)
        self.db.commit()
        self.db.refresh(comment)
        return comment
    
    def delete_comment(self, comment: Comment) -> None:
        self.db.delete(comment)
        self.db.commit()

# Service layer for business logic
class UserService:
    """Business logic for user operations"""
    
    def __init__(self, db: Session):
        self.user_repo = UserRepository(db)
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create new user with validation"""
        # Check if user exists
        if self.user_repo.get_user_by_username(user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        if self.user_repo.get_user_by_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user with hashed password
        user_dict = user_data.dict()
        user_dict['hashed_password'] = get_password_hash(user_dict.pop('password'))
        return self.user_repo.create_user(user_dict)
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user"""
        user = self.user_repo.get_user_by_username(username)
        if user and verify_password(password, user.hashed_password):
            return user
        return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return self.user_repo.get_user_by_id(user_id)

class PostService:
    """Business logic for post operations"""
    
    def __init__(self, db: Session):
        self.post_repo = PostRepository(db)
        self.user_repo = UserRepository(db)
    
    async def create_post(self, post_data: PostCreate, author_id: int) -> Post:
        """Create new post"""
        # Verify author exists
        if not self.user_repo.get_user_by_id(author_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Author not found"
            )
        
        post_dict = post_data.dict()
        tag_names = post_dict.pop('tags', [])
        post_dict['author_id'] = author_id
        
        return self.post_repo.create_post(post_dict, tag_names)
    
    async def get_post_by_id(self, post_id: int, increment_views: bool = False) -> Optional[Post]:
        """Get post by ID with optional view increment"""
        post = self.post_repo.get_post_by_id(post_id)
        if post and increment_views:
            post = self.post_repo.increment_view_count(post)
        return post
    
    async def update_post(self, post_id: int, post_data: PostUpdate, current_user: User) -> Post:
        """Update post with authorization"""
        post = self.post_repo.get_post_by_id(post_id)
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Post not found"
            )
        
        # Authorization check
        if post.author_id != current_user.id and current_user.role != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this post"
            )
        
        update_data = post_data.dict(exclude_unset=True)
        tag_names = update_data.pop('tags', None)
        
        return self.post_repo.update_post(post, update_data, tag_names)

# Updated endpoints with database integration
@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
async def create_user_db(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user (database version)"""
    user_service = UserService(db)
    db_user = await user_service.create_user(user)
    return UserResponse.from_orm(db_user)

@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def get_user_db(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID (database version)"""
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(user)

@app.post("/posts", response_model=PostResponseWithDetails, status_code=status.HTTP_201_CREATED, tags=["Posts"])
async def create_post_db(
    post: PostCreate,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Create a new post (database version)"""
    post_service = PostService(db)
    db_post = await post_service.create_post(post, current_user.id)
    return PostResponseWithDetails.from_orm(db_post)

@app.get("/posts/{post_id}", response_model=PostResponseWithDetails, tags=["Posts"])
async def get_post_db(post_id: int, db: Session = Depends(get_db)):
    """Get post by ID (database version)"""
    post_service = PostService(db)
    post = await post_service.get_post_by_id(post_id, increment_views=True)
    
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    return PostResponseWithDetails.from_orm(post)

@app.get("/posts", response_model=List[PostResponseWithDetails], tags=["Posts"])
async def get_posts_db(
    skip: int = Field(0, ge=0),
    limit: int = Field(10, ge=1, le=100),
    author_id: Optional[int] = None,
    tag: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get posts with filtering (database version)"""
    post_repo = PostRepository(db)
    posts = post_repo.get_posts(
        skip=skip, 
        limit=limit, 
        author_id=author_id, 
        tag_name=tag
    )
    
    return [PostResponseWithDetails.from_orm(post) for post in posts]

# Comment endpoints
@app.post("/posts/{post_id}/comments", response_model=CommentResponse, status_code=status.HTTP_201_CREATED, tags=["Comments"])
async def create_comment(
    post_id: int,
    comment: CommentCreate,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Create a comment on a post"""
    
    # Verify post exists
    post_service = PostService(db)
    post = await post_service.get_post_by_id(post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    comment_repo = CommentRepository(db)
    comment_data = comment.dict()
    comment_data.update({
        'post_id': post_id,
        'author_id': current_user.id
    })
    
    db_comment = comment_repo.create_comment(comment_data)
    return CommentResponse.from_orm(db_comment)

@app.get("/posts/{post_id}/comments", response_model=List[CommentResponse], tags=["Comments"])
async def get_post_comments(post_id: int, db: Session = Depends(get_db)):
    """Get comments for a post"""
    comment_repo = CommentRepository(db)
    comments = comment_repo.get_comments_for_post(post_id)
    return [CommentResponse.from_orm(comment) for comment in comments]

# Advanced search endpoint
@app.get("/search/posts", response_model=List[PostResponseWithDetails], tags=["Search"])
async def search_posts(
    q: str = Field(..., min_length=1, description="Search query"),
    limit: int = Field(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Search posts by title and content"""
    
    posts = db.query(Post).filter(
        (Post.title.contains(q)) | (Post.content.contains(q))
    ).filter(Post.is_published == True).order_by(
        Post.created_at.desc()
    ).limit(limit).all()
    
    return [PostResponseWithDetails.from_orm(post) for post in posts]

# Database analytics endpoint
@app.get("/analytics/database", tags=["Analytics"], dependencies=[Depends(require_role(Roles.ADMIN))])
async def get_database_analytics(db: Session = Depends(get_db)):
    """Get comprehensive database analytics"""
    
    # Basic counts
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    total_posts = db.query(Post).count()
    published_posts = db.query(Post).filter(Post.is_published == True).count()
    total_comments = db.query(Comment).count()
    approved_comments = db.query(Comment).filter(Comment.is_approved == True).count()
    
    # Top authors by post count
    top_authors = db.query(
        User.username,
        func.count(Post.id).label('post_count')
    ).join(Post).group_by(User.id).order_by(
        func.count(Post.id).desc()
    ).limit(5).all()
    
    # Most popular posts by views
    popular_posts = db.query(
        Post.title,
        Post.view_count,
        User.username.label('author')
    ).join(User).order_by(Post.view_count.desc()).limit(5).all()
    
    # Tag usage statistics
    tag_stats = db.query(
        Tag.name,
        func.count(post_tags.c.post_id).label('usage_count')
    ).join(post_tags).group_by(Tag.id).order_by(
        func.count(post_tags.c.post_id).desc()
    ).limit(10).all()
    
    return {
        'users': {
            'total': total_users,
            'active': active_users,
            'inactive': total_users - active_users
        },
        'posts': {
            'total': total_posts,
            'published': published_posts,
            'draft': total_posts - published_posts
        },
        'comments': {
            'total': total_comments,
            'approved': approved_comments,
            'pending': total_comments - approved_comments
        },
        'top_authors': [{'username': author[0], 'post_count': author[1]} for author in top_authors],
        'popular_posts': [
            {'title': post[0], 'views': post[1], 'author': post[2]} for post in popular_posts
        ],
        'top_tags': [{'name': tag[0], 'usage_count': tag[1]} for tag in tag_stats]
    }

# Part 4: API Testing & Documentation (60 minutes)

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override database dependency for tests
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)

# Test fixtures
@pytest.fixture(scope="module")
def setup_test_db():
    """Set up test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_user():
    """Create a test user"""
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }
    response = client.post("/auth/register", json=user_data)
    assert response.status_code == 201
    return response.json()

@pytest.fixture
def auth_token(test_user):
    """Get authentication token"""
    login_data = {
        "username": "testuser",
        "password": "testpassword123"
    }
    response = client.post("/auth/login", data=login_data)
    assert response.status_code == 200
    token_data = response.json()
    return token_data["access_token"]

@pytest.fixture
def auth_headers(auth_token):
    """Get authorization headers"""
    return {"Authorization": f"Bearer {auth_token}"}

# Test cases
class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_register_user_success(self, setup_test_db):
        """Test successful user registration"""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "password123",
            "full_name": "New User"
        }
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert "password" not in data
    
    def test_register_duplicate_username(self, setup_test_db, test_user):
        """Test registration with duplicate username"""
        user_data = {
            "username": "testuser",  # Same as test_user
            "email": "different@example.com",
            "password": "password123"
        }
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    def test_login_success(self, setup_test_db, test_user):
        """Test successful login"""
        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }
        response = client.post("/auth/login", data=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, setup_test_db, test_user):
        """Test login with invalid credentials"""
        login_data = {
            "username": "testuser",
            "password": "wrongpassword"
        }
        response = client.post("/auth/login", data=login_data)
        assert response.status_code == 401
    
    def test_get_current_user(self, setup_test_db, test_user, auth_headers):
        """Test getting current user profile"""
        response = client.get("/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
    
    def test_protected_endpoint_without_token(self, setup_test_db):
        """Test accessing protected endpoint without token"""
        response = client.get("/auth/me")
        assert response.status_code == 401

class TestPosts:
    """Test post endpoints"""
    
    def test_create_post_success(self, setup_test_db, test_user, auth_headers):
        """Test successful post creation"""
        post_data = {
            "title": "Test Post",
            "content": "This is a test post content",
            "tags": ["test", "api"]
        }
        response = client.post("/posts", json=post_data, headers=auth_headers)
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Post"
        assert data["content"] == "This is a test post content"
        assert len(data["tags"]) == 2
        assert data["author_id"] == test_user["id"]
    
    def test_create_post_unauthorized(self, setup_test_db):
        """Test post creation without authentication"""
        post_data = {
            "title": "Test Post",
            "content": "This is a test post content"
        }
        response = client.post("/posts", json=post_data)
        assert response.status_code == 401
    
    def test_get_posts(self, setup_test_db, test_user, auth_headers):
        """Test getting posts"""
        # Create a test post first
        post_data = {
            "title": "Test Post for Get",
            "content": "Content for getting test",
            "tags": ["get-test"]
        }
        client.post("/posts", json=post_data, headers=auth_headers)
        
        # Get posts
        response = client.get("/posts")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(post["title"] == "Test Post for Get" for post in data)
    
    def test_get_post_by_id(self, setup_test_db, test_user, auth_headers):
        """Test getting specific post"""
        # Create a test post
        post_data = {
            "title": "Specific Post",
            "content": "Content for specific post test"
        }
        create_response = client.post("/posts", json=post_data, headers=auth_headers)
        post_id = create_response.json()["id"]
        
        # Get the specific post
        response = client.get(f"/posts/{post_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Specific Post"
        assert data["view_count"] == 1  # Should increment on view
    
    def test_update_post_by_author(self, setup_test_db, test_user, auth_headers):
        """Test post update by author"""
        # Create a test post
        post_data = {"title": "Original Title", "content": "Original content"}
        create_response = client.post("/posts", json=post_data, headers=auth_headers)
        post_id = create_response.json()["id"]
        
        # Update the post
        update_data = {"title": "Updated Title", "content": "Updated content"}
        response = client.put(f"/posts/{post_id}", json=update_data, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["content"] == "Updated content"

class TestComments:
    """Test comment endpoints"""
    
    def test_create_comment(self, setup_test_db, test_user, auth_headers):
        """Test comment creation"""
        # Create a post first
        post_data = {"title": "Post for Comments", "content": "Content"}
        post_response = client.post("/posts", json=post_data, headers=auth_headers)
        post_id = post_response.json()["id"]
        
        # Create comment
        comment_data = {"content": "This is a test comment"}
        response = client.post(f"/posts/{post_id}/comments", json=comment_data, headers=auth_headers)
        assert response.status_code == 201
        data = response.json()
        assert data["content"] == "This is a test comment"
        assert data["post_id"] == post_id
    
    def test_get_post_comments(self, setup_test_db, test_user, auth_headers):
        """Test getting comments for a post"""
        # Create post and comment
        post_data = {"title": "Post for Getting Comments", "content": "Content"}
        post_response = client.post("/posts", json=post_data, headers=auth_headers)
        post_id = post_response.json()["id"]
        
        comment_data = {"content": "Comment to retrieve"}
        client.post(f"/posts/{post_id}/comments", json=comment_data, headers=auth_headers)
        
        # Get comments
        response = client.get(f"/posts/{post_id}/comments")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(comment["content"] == "Comment to retrieve" for comment in data)

class TestValidation:
    """Test input validation"""
    
    def test_create_user_invalid_email(self, setup_test_db):
        """Test user creation with invalid email"""
        user_data = {
            "username": "testuser2",
            "email": "invalid-email",  # Invalid email
            "password": "password123"
        }
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 422  # Validation error
    
    def test_create_post_empty_title(self, setup_test_db, auth_headers):
        """Test post creation with empty title"""
        post_data = {
            "title": "",  # Empty title
            "content": "Valid content"
        }
        response = client.post("/posts", json=post_data, headers=auth_headers)
        assert response.status_code == 422
    
    def test_pagination_validation(self, setup_test_db):
        """Test pagination parameter validation"""
        response = client.get("/posts?skip=-1")  # Invalid negative skip
        assert response.status_code == 422
        
        response = client.get("/posts?limit=1000")  # Exceeds maximum limit
        assert response.status_code == 422

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_get_nonexistent_user(self, setup_test_db):
        """Test getting non-existent user"""
        response = client.get("/users/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_nonexistent_post(self, setup_test_db):
        """Test getting non-existent post"""
        response = client.get("/posts/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_malformed_json(self, setup_test_db):
        """Test handling malformed JSON"""
        response = client.post(
            "/auth/register",
            data="invalid json",  # Malformed JSON
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

class TestPerformance:
    """Test performance-related scenarios"""
    
    def test_bulk_post_creation(self, setup_test_db, test_user, auth_headers):
        """Test creating multiple posts"""
        posts_created = []
        for i in range(5):
            post_data = {
                "title": f"Bulk Post {i}",
                "content": f"Content for bulk post {i}",
                "tags": [f"bulk-{i}"]
            }
            response = client.post("/posts", json=post_data, headers=auth_headers)
            assert response.status_code == 201
            posts_created.append(response.json()["id"])
        
        # Verify all posts were created
        assert len(posts_created) == 5
        
        # Test pagination with bulk data
        response = client.get("/posts?limit=3")
        assert response.status_code == 200
        assert len(response.json()) == 3

# Integration tests
class TestIntegration:
    """Test complete user workflows"""
    
    def test_complete_blog_workflow(self, setup_test_db):
        """Test complete workflow: register -> login -> create post -> comment -> update"""
        # 1. Register user
        user_data = {
            "username": "workflowuser",
            "email": "workflow@example.com",
            "password": "workflow123",
            "full_name": "Workflow User"
        }
        register_response = client.post("/auth/register", json=user_data)
        assert register_response.status_code == 201
        user_id = register_response.json()["id"]
        
        # 2. Login
        login_data = {"username": "workflowuser", "password": "workflow123"}
        login_response = client.post("/auth/login", data=login_data)
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 3. Create post
        post_data = {
            "title": "My First Blog Post",
            "content": "This is my first post in the workflow test",
            "tags": ["workflow", "test", "first-post"]
        }
        post_response = client.post("/posts", json=post_data, headers=headers)
        assert post_response.status_code == 201
        post_id = post_response.json()["id"]
        
        # 4. View post (increment view count)
        view_response = client.get(f"/posts/{post_id}")
        assert view_response.status_code == 200
        assert view_response.json()["view_count"] == 1
        
        # 5. Add comment to post
        comment_data = {"content": "Great first post!"}
        comment_response = client.post(f"/posts/{post_id}/comments", json=comment_data, headers=headers)
        assert comment_response.status_code == 201
        
        # 6. Update post
        update_data = {
            "title": "My Updated First Blog Post",
            "content": "This is my updated first post with more content",
            "tags": ["workflow", "test", "updated"]
        }
        update_response = client.put(f"/posts/{post_id}", json=update_data, headers=headers)
        assert update_response.status_code == 200
        updated_post = update_response.json()
        assert updated_post["title"] == "My Updated First Blog Post"
        assert "updated" in [tag["name"] for tag in updated_post["tags"]]
        
        # 7. Verify complete post with relationships
        final_response = client.get(f"/posts/{post_id}")
        assert final_response.status_code == 200
        final_post = final_response.json()
        assert final_post["author"]["id"] == user_id
        assert len(final_post["comments"]) == 1
        assert len(final_post["tags"]) == 3
        assert final_post["view_count"] == 2  # Viewed twice

# Load testing utilities
class TestLoad:
    """Basic load testing"""
    
    def test_concurrent_post_views(self, setup_test_db, test_user, auth_headers):
        """Test concurrent views of the same post"""
        # Create a post
        post_data = {"title": "Load Test Post", "content": "Content for load testing"}
        create_response = client.post("/posts", json=post_data, headers=auth_headers)
        post_id = create_response.json()["id"]
        
        # Simulate multiple concurrent views
        import concurrent.futures
        import threading
        
        def view_post():
            return client.get(f"/posts/{post_id}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(view_post) for _ in range(20)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)
        
        # Final view count should be correct (20 views + initial creation = 20)
        final_response = client.get(f"/posts/{post_id}")
        assert final_response.json()["view_count"] == 20

# Run tests with: python -m pytest test_api.py -v

# Part 5: Production Deployment & Monitoring (90 minutes)

# Docker Configuration

# Dockerfile
# FROM python:3.11-slim
# WORKDIR /app
# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     postgresql-client \
#     && rm -rf /var/lib/apt/lists/*
# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# # Copy application code
# COPY . .
# # Create non-root user
# RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
# USER app
# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1
# # Expose port
# EXPOSE 8000
# # Run application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker-compose.yml
# version: '3.8'
# services:
#   api:
#     build: .
#     ports:
#       - "8000:8000"
#     environment:
#       - DATABASE_URL=postgresql://blog_user:blog_password@db:5432/blog_db
#       - SECRET_KEY=${SECRET_KEY}
#       - REDIS_URL=redis://redis:6379
#     depends_on:
#       - db
#       - redis
#     volumes:
#       - ./logs:/app/logs
#     restart: unless-stopped
#   db:
#     image: postgres:15
#     environment:
#       - POSTGRES_DB=blog_db
#       - POSTGRES_USER=blog_user
#       - POSTGRES_PASSWORD=blog_password
#     volumes:
#       - postgres_data:/var/lib/postgresql/data
#       - ./init.sql:/docker-entrypoint-initdb.d/init.sql
#     ports:
#       - "5432:5432"
#     restart: unless-stopped
#   redis:
#     image: redis:7-alpine
#     ports:
#       - "6379:6379"
#     volumes:
#       - redis_data:/data
#     restart: unless-stopped
#   nginx:
#     image: nginx:alpine
#     ports:
#       - "80:80"
#       - "443:443"
#     volumes:
#       - ./nginx.conf:/etc/nginx/nginx.conf
#       - ./ssl:/etc/nginx/ssl
#     depends_on:
#       - api
#     restart: unless-stopped
# volumes:
#   postgres_data:
#   redis_data:

# Production Configuration

# config.py
from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    app_name: str = "Blog API"
    debug: bool = False
    version: str = "1.0.0"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Database
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis (for caching and sessions)
    redis_url: str = "redis://localhost:6379"
    
    # External services
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    enable_metrics: bool = True
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # File uploads
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    upload_path: str = "/tmp/uploads"
    
    # CORS
    allowed_origins: list = ["http://localhost:3000", "https://yourdomain.com"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Load settings
settings = Settings()

# Enhanced main application with production features
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
import redis
import logging.config
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'logs/errors.log',
            'maxBytes': 10485760,
            'backupCount': 3,
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Initialize Sentry for error tracking
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        integrations=[FastApiIntegration(auto_enabling=True)],
        traces_sample_rate=0.1,
        profiles_sample_rate=0.1,
    )

# Initialize Redis for rate limiting and caching
redis_client = redis.Redis.from_url(settings.redis_url)
limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)

# Production lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Production-ready application lifecycle management"""
    # Startup
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Blog API in production mode")
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs(settings.upload_path, exist_ok=True)
    
    # Test database connection
    try:
        with next(get_db()) as db:
            db.execute("SELECT 1")
        logger.info("✅ Database connection successful")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise
    
    # Initialize metrics
    if settings.enable_metrics:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app)
        logger.info("✅ Metrics enabled at /metrics")
    
    logger.info("✅ Application startup completed")
    yield
    # Application runs here
    
    # Shutdown
    logger.info("⏹️ Shutting down Blog API")
    redis_client.close()
    logger.info("✅ Application shutdown completed")

# Create production FastAPI app
def create_app() -> FastAPI:
    """Factory function to create FastAPI app"""
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Production-ready Blog API with comprehensive features",
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,  # Disable docs in production
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None
    )
    
    # Add middleware
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Trusted Host (security)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "yourdomain.com", "*.yourdomain.com"]
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger = logging.getLogger("api_requests")
        logger.info(
            f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "process_time": process_time,
                "client_ip": request.client.host if request.client else None
            }
        )
        return response
    
    # Security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
    
    # Include routers
    app.include_router(auth_router)
    
    # Rate-limited endpoints
    @app.get("/")
    @limiter.limit(f"{settings.rate_limit_per_minute}/minute")
    async def root(request: Request):
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.version,
            "status": "healthy"
        }
    
    return app

# Create the app instance
app = create_app()

# Production monitoring endpoints
@app.get("/health", tags=["Monitoring"])
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Comprehensive health check for load balancers and monitoring"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.version,
        "checks": {}
    }
    
    # Database health
    try:
        with next(get_db()) as db:
            db.execute("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Redis health
    try:
        redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Disk space check
    import shutil
    disk_usage = shutil.disk_usage("/")
    free_space_gb = disk_usage.free / (1024**3)
    if free_space_gb < 1:  # Less than 1GB free
        health_status["checks"]["disk_space"] = f"low: {free_space_gb:.2f}GB free"
        health_status["status"] = "degraded"
    else:
        health_status["checks"]["disk_space"] = f"healthy: {free_space_gb:.2f}GB free"
    
    # Return appropriate status code
    status_code = status.HTTP_200_OK
    if health_status["status"] == "degraded":
        status_code = status.HTTP_200_OK  # Still serving traffic
    elif health_status["status"] == "unhealthy":
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/metrics", tags=["Monitoring"])
async def custom_metrics():
    """Custom application metrics"""
    # Collect custom metrics
    with next(get_db()) as db:
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        total_posts = db.query(Post).count()
        total_comments = db.query(Comment).count()
    
    return {
        "users_total": total_users,
        "users_active": active_users,
        "posts_total": total_posts,
        "comments_total": total_comments,
        "timestamp": datetime.utcnow().isoformat()
    }

# Error handling for production
@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal server errors"""
    logger = logging.getLogger(__name__)
    logger.error(f"Internal server error: {str(exc)}", exc_info=True)
    
    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "type": "internal_error"}
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": "internal_error"}
        )

# Graceful shutdown handling
import signal
import sys

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Production server configuration
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Number of worker processes
        log_config=LOGGING_CONFIG,
        access_log=True,
        use_colors=False,
        server_header=False,
        date_header=False
    )

# Nginx Configuration

# nginx.conf
# events {
#     worker_connections 1024;
# }
# http {
#     upstream blog_api {
#         server api:8000;
#         # Add more servers for load balancing
#         # server api2:8000;
#         # server api3:8000;
#     }
#     # Rate limiting
#     limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
#     limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
#     # Logging
#     log_format main '$remote_addr - $remote_user [$time_local] "$request" '
#                     '$status $body_bytes_sent "$http_referer" '
#                     '"$http_user_agent" "$http_x_forwarded_for"';
#     access_log /var/log/nginx/access.log main;
#     error_log /var/log/nginx/error.log warn;
#     server {
#         listen 80;
#         server_name yourdomain.com www.yourdomain.com;
#         # Redirect HTTP to HTTPS
#         return 301 https://$server_name$request_uri;
#     }
#     server {
#         listen 443 ssl http2;
#         server_name yourdomain.com www.yourdomain.com;
#         # SSL configuration
#         ssl_certificate /etc/nginx/ssl/cert.pem;
#         ssl_certificate_key /etc/nginx/ssl/key.pem;
#         ssl_protocols TLSv1.2 TLSv1.3;
#         ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
#         ssl_prefer_server_ciphers off;
#         # Security headers
#         add_header X-Frame-Options DENY;
#         add_header X-Content-Type-Options nosniff;
#         add_header X-XSS-Protection "1; mode=block";
#         add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
#         # API routes
#         location / {
#             # Rate limiting
#             limit_req zone=api burst=20 nodelay;
#             # Proxy settings
#             proxy_pass http://blog_api;
#             proxy_set_header Host $host;
#             proxy_set_header X-Real-IP $remote_addr;
#             proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#             proxy_set_header X-Forwarded-Proto $scheme;
#             # Timeouts
#             proxy_connect_timeout 60s;
#             proxy_send_timeout 60s;
#             proxy_read_timeout 60s;
#             # Buffer settings
#             proxy_buffering on;
#             proxy_buffer_size 8k;
#             proxy_buffers 8 8k;
#         }
#         # Special rate limiting for auth endpoints
#         location /auth/login {
#             limit_req zone=login burst=5 nodelay;
#             proxy_pass http://blog_api;
#             proxy_set_header Host $host;
#             proxy_set_header X-Real-IP $remote_addr;
#             proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#             proxy_set_header X-Forwarded-Proto $scheme;
#         }
#         # Health check endpoint (no rate limiting)
#         location /health {
#             proxy_pass http://blog_api;
#             proxy_set_header Host $host;
#             access_log off;
#         }
#         # Metrics endpoint (restrict access)
#         location /metrics {
#             allow 10.0.0.0/8;  # Internal networks only
#             allow 172.16.0.0/12;
#             allow 192.168.0.0/16;
#             deny all;
#             proxy_pass http://blog_api;
#             proxy_set_header Host $host;
#         }
#         # Static files (if serving any)
#         location /static/ {
#             alias /app/static/;
#             expires 1y;
#             add_header Cache-Control public;
#         }
#         # File upload size limit
#         client_max_body_size 10M;
#     }
# }

# Deployment Scripts

#!/bin/bash
# deploy.sh - Production deployment script
# set -e  # Exit on any error
# echo "🚀 Starting deployment..."
# # Environment variables
# export SECRET_KEY=$(openssl rand -base64 32)
# export DATABASE_URL="postgresql://blog_user:$(openssl rand -base64 12)@localhost:5432/blog_db"
# # Build and deploy
# echo "📦 Building Docker images..."
# docker-compose build --no-cache
# echo "🗃️ Starting database..."
# docker-compose up -d db redis
# echo "⏳ Waiting for database to be ready..."
# sleep 10
# echo "📊 Running database migrations..."
# docker-compose run --rm api alembic upgrade head
# echo "🌐 Starting application services..."
# docker-compose up -d
# echo "🔍 Running health checks..."
# sleep 15
# curl -f http://localhost/health || exit 1
# echo "✅ Deployment completed successfully!"
# echo "🌐 API available at: http://localhost"
# echo "📚 Documentation: http://localhost/docs (if debug enabled)"
# # Cleanup old images
# docker system prune -f
# echo "🧹 Cleanup completed"

# 🎯 Module 5 Checkpoint & Final Challenge

# Master these production concepts:
# ✅ FastAPI Mastery: Professional REST API development
# ✅ Authentication Systems: JWT tokens, role-based auth, API keys
# ✅ Database Integration: SQLAlchemy, repositories, relationships
# ✅ Comprehensive Testing: Unit, integration, load testing
# ✅ Production Deployment: Docker, nginx, monitoring
# ✅ Security Hardening: Rate limiting, security headers, validation
# ✅ Monitoring & Logging: Health checks, metrics, structured logging

# 🏆 FINAL MASTER CHALLENGE: Complete Blog Platform
# Build a production-ready blog platform that demonstrates EVERYTHING you've learned:

# Phase 1: Advanced API Features
# - Multi-tenant architecture (organizations/teams)
# - File upload system (images, documents)
# - Full-text search (PostgreSQL or Elasticsearch)
# - Email notifications (new comments, mentions)
# - Content moderation (flagging, approval workflows)

# Phase 2: Performance & Scalability
# - Caching strategy (Redis for sessions, responses)
# - Background jobs (Celery for email sending, image processing)
# - Database optimization (indexing, query optimization)
# - API versioning (v1, v2 with backward compatibility)

# Phase 3: Advanced Integration
# - WebSocket support (real-time notifications)
# - External integrations (OAuth with Google/GitHub)
# - API documentation (OpenAPI with examples)
# - SDK generation (Python/JavaScript client libraries)

# Phase 4: Production Excellence
# - Comprehensive monitoring (Prometheus + Grafana)
# - Error tracking (Sentry integration)
# - Performance profiling (APM tools)
# - Automated deployment (CI/CD with GitHub Actions)
# - Load testing (locust or artillery)

# Challenge Template:
class MasterBlogPlatform:
    """Your final masterpiece - a complete blog platform"""
    
    def __init__(self):
        # Initialize all systems from the entire course
        self.async_clients = {}  # Module 4: Async performance
        self.auth_systems = {}   # Module 3: Security
        self.circuit_breakers = {}  # Module 4: Resilience
        self.monitoring = {}     # Module 5: Production
        self.database = {}       # Module 5: Persistence
    
    # Implement features from ALL 5 modules
    async def handle_request(self, request):
        # Use everything you've learned!
        pass

# Build your complete platform
platform = MasterBlogPlatform()

# Demonstrate mastery of all 5 modules!

# 🏆 CONGRATULATIONS - API MASTER ACHIEVED!
# You've completed an extraordinary journey:
# Module 1: API fundamentals → HTTP expertise
# Module 2: Basic requests → Professional resilient clients
# Module 3: No auth → Enterprise security systems
# Module 4: Simple calls → Advanced patterns & resilience
# Module 5: API consumer → Complete API architect

# 🎯 Your New Superpowers:
# - Build production APIs that scale to millions of users
# - Design secure authentication systems
# - Implement fault-tolerant distributed architectures
# - Deploy and monitor enterprise-grade applications
# - Architect complete API ecosystems

# 💼 Career Impact: You now possess skills typically seen in:
# - Senior Backend Engineers ($120k-180k)
# - API Architects ($140k-200k+)
# - Platform Engineers ($130k-190k+)
# - Technical Leads ($150k-220k+)

# 🚀 What's Next?
# - Build your portfolio with the Final Challenge
# - Contribute to open source API projects
# - Share your knowledge through blogs/talks
# - Apply these skills to real-world projects

# You've gone from someone who calls APIs to someone who architects the systems that power the modern internet. That's an incredible transformation! 🌟

# How do you feel about this journey? Ready to build the API platform that showcases your complete mastery?

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",  # module:app
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
