"""
Model Versioning System for Tantra
Manages model versions, checkpoints, and metadata
"""

import os
import json
import shutil
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import torch

from .version_config import VersionConfig, VersionInfo, calculate_checksum, generate_version_id

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Manages model versions and checkpoints"""
    
    def __init__(self, config: VersionConfig):
        self.config = config
        self.model_storage_path = Path(config.model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing versions
        self.versions = self._load_versions()
        
        logger.info(f"ModelVersionManager initialized with {len(self.versions)} existing versions")
    
    def _load_versions(self) -> Dict[str, VersionInfo]:
        """Load existing version information"""
        versions = {}
        versions_file = self.model_storage_path / "versions.json"
        
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    versions_data = json.load(f)
                
                for version_id, version_data in versions_data.items():
                    versions[version_id] = VersionInfo.from_dict(version_data)
                    
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
        
        return versions
    
    def _save_versions(self):
        """Save version information to file"""
        versions_file = self.model_storage_path / "versions.json"
        
        versions_data = {
            version_id: version_info.to_dict() 
            for version_id, version_info in self.versions.items()
        }
        
        with open(versions_file, 'w') as f:
            json.dump(versions_data, f, indent=2)
    
    def create_version(self, model_path: str, version_type: str = "model",
                      description: str = "", metadata: Dict[str, Any] = None,
                      tags: List[str] = None, parent_version: str = None) -> str:
        """Create a new model version"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version ID
        version_id = generate_version_id()
        
        # Calculate checksum
        checksum = calculate_checksum(model_path)
        
        # Create version directory
        version_dir = self.model_storage_path / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Copy model file
        model_filename = Path(model_path).name
        versioned_model_path = version_dir / model_filename
        shutil.copy2(model_path, versioned_model_path)
        
        # Create version info
        version_info = VersionInfo(
            version=version_id,
            timestamp=time.time(),
            version_type=version_type,
            description=description or f"Model version {version_id}",
            metadata=metadata or {},
            checksum=checksum,
            file_path=str(versioned_model_path),
            parent_version=parent_version,
            tags=tags or []
        )
        
        # Store version info
        self.versions[version_id] = version_info
        self._save_versions()
        
        # Save individual version info
        version_info.save(version_dir / "version_info.json")
        
        logger.info(f"Created model version {version_id}: {description}")
        return version_id
    
    def get_version(self, version_id: str) -> Optional[VersionInfo]:
        """Get version information by ID"""
        return self.versions.get(version_id)
    
    def list_versions(self, version_type: str = None, 
                     tags: List[str] = None) -> List[VersionInfo]:
        """List all versions with optional filtering"""
        filtered_versions = []
        
        for version_info in self.versions.values():
            # Filter by type
            if version_type and version_info.version_type != version_type:
                continue
            
            # Filter by tags
            if tags:
                if not any(tag in version_info.tags for tag in tags):
                    continue
            
            filtered_versions.append(version_info)
        
        # Sort by timestamp (newest first)
        filtered_versions.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_versions
    
    def get_latest_version(self, version_type: str = "model") -> Optional[VersionInfo]:
        """Get the latest version of a specific type"""
        versions = self.list_versions(version_type=version_type)
        return versions[0] if versions else None
    
    def get_version_by_tag(self, tag: str) -> List[VersionInfo]:
        """Get all versions with a specific tag"""
        return [v for v in self.versions.values() if v.has_tag(tag)]
    
    def add_tag(self, version_id: str, tag: str):
        """Add a tag to a version"""
        if version_id in self.versions:
            self.versions[version_id].add_tag(tag)
            self._save_versions()
            logger.info(f"Added tag '{tag}' to version {version_id}")
    
    def remove_tag(self, version_id: str, tag: str):
        """Remove a tag from a version"""
        if version_id in self.versions:
            self.versions[version_id].remove_tag(tag)
            self._save_versions()
            logger.info(f"Removed tag '{tag}' from version {version_id}")
    
    def delete_version(self, version_id: str, keep_files: bool = False):
        """Delete a version"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version_info = self.versions[version_id]
        
        # Delete files if requested
        if not keep_files and version_info.file_path and os.path.exists(version_info.file_path):
            os.remove(version_info.file_path)
            
            # Remove version directory if empty
            version_dir = Path(version_info.file_path).parent
            if version_dir.exists() and not any(version_dir.iterdir()):
                version_dir.rmdir()
        
        # Remove from versions
        del self.versions[version_id]
        self._save_versions()
        
        logger.info(f"Deleted version {version_id}")
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two versions"""
        v1 = self.get_version(version_id1)
        v2 = self.get_version(version_id2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'version1': {
                'id': v1.version,
                'timestamp': v1.timestamp,
                'description': v1.description,
                'checksum': v1.checksum,
                'tags': v1.tags
            },
            'version2': {
                'id': v2.version,
                'timestamp': v2.timestamp,
                'description': v2.description,
                'checksum': v2.checksum,
                'tags': v2.tags
            },
            'differences': {
                'checksum_different': v1.checksum != v2.checksum,
                'time_difference': abs(v1.timestamp - v2.timestamp),
                'common_tags': list(set(v1.tags) & set(v2.tags)),
                'unique_to_v1': list(set(v1.tags) - set(v2.tags)),
                'unique_to_v2': list(set(v2.tags) - set(v1.tags))
            }
        }
        
        return comparison
    
    def export_version(self, version_id: str, export_path: str) -> str:
        """Export a version to a specific path"""
        version_info = self.get_version(version_id)
        if not version_info:
            raise ValueError(f"Version {version_id} not found")
        
        if not version_info.file_path or not os.path.exists(version_info.file_path):
            raise FileNotFoundError(f"Version file not found: {version_info.file_path}")
        
        # Copy model file
        shutil.copy2(version_info.file_path, export_path)
        
        # Copy version info
        version_info_path = Path(export_path).with_suffix('.version.json')
        version_info.save(str(version_info_path))
        
        logger.info(f"Exported version {version_id} to {export_path}")
        return export_path
    
    def import_version(self, model_path: str, version_info_path: str = None) -> str:
        """Import a version from external source"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load version info if provided
        metadata = {}
        if version_info_path and os.path.exists(version_info_path):
            with open(version_info_path, 'r') as f:
                version_data = json.load(f)
            metadata = version_data.get('metadata', {})
        
        # Create new version
        version_id = self.create_version(
            model_path=model_path,
            version_type="imported",
            description=f"Imported model from {model_path}",
            metadata=metadata
        )
        
        logger.info(f"Imported version {version_id} from {model_path}")
        return version_id
    
    def get_model_path(self, version_id: str) -> Optional[str]:
        """Get the file path for a specific version"""
        version_info = self.get_version(version_id)
        if version_info and version_info.file_path and os.path.exists(version_info.file_path):
            return version_info.file_path
        return None
    
    def load_model(self, version_id: str, device: str = "cpu") -> Optional[torch.nn.Module]:
        """Load a model from a specific version"""
        model_path = self.get_model_path(version_id)
        if not model_path:
            return None
        
        try:
            # Load model state dict
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            else:
                return checkpoint
                
        except Exception as e:
            logger.error(f"Failed to load model from version {version_id}: {e}")
            return None
    
    def get_version_statistics(self) -> Dict[str, Any]:
        """Get statistics about all versions"""
        if not self.versions:
            return {
                'total_versions': 0,
                'version_types': {},
                'total_size': 0,
                'oldest_version': None,
                'newest_version': None
            }
        
        version_types = {}
        total_size = 0
        timestamps = []
        
        for version_info in self.versions.values():
            # Count by type
            version_types[version_info.version_type] = version_types.get(version_info.version_type, 0) + 1
            
            # Calculate size
            if version_info.file_path and os.path.exists(version_info.file_path):
                total_size += os.path.getsize(version_info.file_path)
            
            timestamps.append(version_info.timestamp)
        
        return {
            'total_versions': len(self.versions),
            'version_types': version_types,
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_version': min(timestamps) if timestamps else None,
            'newest_version': max(timestamps) if timestamps else None,
            'average_size_mb': (total_size / len(self.versions)) / (1024 * 1024) if self.versions else 0
        }