from typing import Optional, List
from sqlmodel import Field, SQLModel, Relationship, JSON
from datetime import datetime

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    
    projects: List["Project"] = Relationship(back_populates="owner")

class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(default="Untitled Project")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    data: dict = Field(default={}, sa_type=JSON)  # Stores the 3D layout JSON
    
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    owner: Optional[User] = Relationship(back_populates="projects")
