"""
Version Configuration for Tantra
Manages versioning settings and metadata
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time
import hashlib
import json
import os
from datetime import datetime


@dataclass
class VersionConfig:
    """Configuration for versioning system"""
    
    # Version settings
    version_scheme: str = "semantic"  # semantic, timestamp, custom
    major_version: int = 1
    minor_version: int = 0
    patch_version: int = 0
    pre_release: Optional[str] = None  # alpha, beta, rc
    build_metadata: Optional[str] = None
    
    # Model versioning
    model_versioning_enabled: bool = True
    auto_increment_patch: bool = True
    auto_increment_minor: bool = False
    auto_increment_major: bool = False
    
    # Training versioning
    training_versioning_enabled: bool = True
    track_training_metrics: bool = True
    track_data_changes: bool = True
    track_config_changes: bool = True
    
    # Data versioning
    data_versioning_enabled: bool = True
    track_data_checksums: bool = True
    track_data_sources: bool = True
    data_retention_days: int = 30
    
    # Storage settings
    version_storage_path: str = "versions"
    model_storage_path: str = "Model/versions"
    data_storage_path: str = "data/versions"
    training_storage_path: str = "training/versions"
    
    # Metadata
    project_name: str = "tantra"
    project_description: str = "OCR-Native Conversational Speech LLM"
    maintainer: str = "Tantra Team"
    license: str = "MIT"
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize versioning system"""
        # Create storage directories
        os.makedirs(self.version_storage_path, exist_ok=True)
        os.makedirs(self.model_storage_path, exist_ok=True)
        os.makedirs(self.data_storage_path, exist_ok=True)
        os.makedirs(self.training_storage_path, exist_ok=True)
    
    def get_version_string(self) -> str:
        """Get current version string"""
        version_parts = [f"{self.major_version}.{self.minor_version}.{self.patch_version}"]
        
        if self.pre_release:
            version_parts.append(f"-{self.pre_release}")
        
        if self.build_metadata:
            version_parts.append(f"+{self.build_metadata}")
        
        return "".join(version_parts)
    
    def increment_version(self, increment_type: str = "patch") -> str:
        """Increment version and return new version string"""
        if increment_type == "major":
            self.major_version += 1
            self.minor_version = 0
            self.patch_version = 0
        elif increment_type == "minor":
            self.minor_version += 1
            self.patch_version = 0
        elif increment_type == "patch":
            self.patch_version += 1
        
        return self.get_version_string()
    
    def set_pre_release(self, pre_release: str) -> str:
        """Set pre-release identifier"""
        self.pre_release = pre_release
        return self.get_version_string()
    
    def set_build_metadata(self, build_metadata: str) -> str:
        """Set build metadata"""
        self.build_metadata = build_metadata
        return self.get_version_string()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version_scheme': self.version_scheme,
            'major_version': self.major_version,
            'minor_version': self.minor_version,
            'patch_version': self.patch_version,
            'pre_release': self.pre_release,
            'build_metadata': self.build_metadata,
            'model_versioning_enabled': self.model_versioning_enabled,
            'training_versioning_enabled': self.training_versioning_enabled,
            'data_versioning_enabled': self.data_versioning_enabled,
            'project_name': self.project_name,
            'project_description': self.project_description,
            'maintainer': self.maintainer,
            'license': self.license,
            'custom_metadata': self.custom_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def save_config(self, path: str):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'VersionConfig':
        """Load configuration from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class VersionInfo:
    """Information about a specific version"""
    
    version: str
    timestamp: float
    version_type: str  # model, training, data
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    file_path: Optional[str] = None
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize version info"""
        if self.timestamp == 0:
            self.timestamp = time.time()
    
    def get_timestamp_string(self) -> str:
        """Get formatted timestamp string"""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    def add_tag(self, tag: str):
        """Add a tag to this version"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str):
        """Remove a tag from this version"""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if version has a specific tag"""
        return tag in self.tags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'version_type': self.version_type,
            'description': self.description,
            'metadata': self.metadata,
            'checksum': self.checksum,
            'file_path': self.file_path,
            'parent_version': self.parent_version,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionInfo':
        """Create from dictionary"""
        return cls(**data)
    
    def save(self, path: str):
        """Save version info to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'VersionInfo':
        """Load version info from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def calculate_checksum(file_path: str) -> str:
    """Calculate SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def generate_version_id() -> str:
    """Generate a unique version ID"""
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:8]