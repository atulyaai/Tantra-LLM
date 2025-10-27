"""
GitHub Integration for Tantra Model Management
Handles model saving, versioning, and releases on GitHub
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import base64
import hashlib
import time

try:
    from github import Github, GithubException
    from github.GitRelease import GitRelease
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class GitHubModelManager:
    """Manages model saving and versioning on GitHub"""
    
    def __init__(self, github_token: str, repository: str):
        self.github_token = github_token
        self.repository = repository
        self.github = None
        self.repo = None
        
        if GITHUB_AVAILABLE:
            try:
                self.github = Github(github_token)
                self.repo = self.github.get_repo(repository)
                logger.info(f"Connected to GitHub repository: {repository}")
            except Exception as e:
                logger.error(f"Failed to connect to GitHub: {e}")
                self.github = None
                self.repo = None
        else:
            logger.warning("GitHub integration not available. Install PyGithub: pip install PyGithub")
    
    def is_available(self) -> bool:
        """Check if GitHub integration is available"""
        return self.repo is not None
    
    def save_model_file(self, local_path: str, github_path: str, 
                       commit_message: str, branch: str = "main") -> bool:
        """Save a model file to GitHub repository"""
        if not self.is_available():
            logger.warning("GitHub repository not available")
            return False
        
        try:
            # Read file content
            with open(local_path, 'rb') as f:
                file_content = f.read()
            
            # Encode content
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            
            # Get file SHA if it exists
            file_sha = None
            try:
                existing_file = self.repo.get_contents(github_path, ref=branch)
                file_sha = existing_file.sha
                logger.info(f"Updating existing file: {github_path}")
            except GithubException:
                logger.info(f"Creating new file: {github_path}")
            
            # Create or update file
            if file_sha:
                # Update existing file
                self.repo.update_file(
                    github_path,
                    commit_message,
                    encoded_content,
                    file_sha,
                    branch=branch
                )
            else:
                # Create new file
                self.repo.create_file(
                    github_path,
                    commit_message,
                    encoded_content,
                    branch=branch
                )
            
            logger.info(f"Successfully saved {github_path} to GitHub")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save file {github_path} to GitHub: {e}")
            return False
    
    def save_model_with_metadata(self, model_path: str, model_name: str,
                                metadata: Dict[str, Any], 
                                commit_message: str = None) -> bool:
        """Save model with metadata to GitHub"""
        if not self.is_available():
            return False
        
        if commit_message is None:
            commit_message = f"Update {model_name} model"
        
        # Create model directory path
        model_dir = f"models/{model_name}"
        
        # Save model file
        model_file_path = f"{model_dir}/{model_name}.pt"
        success = self.save_model_file(model_path, model_file_path, commit_message)
        
        if not success:
            return False
        
        # Save metadata
        metadata_path = f"{model_dir}/metadata.json"
        metadata_content = json.dumps(metadata, indent=2)
        
        # Create temporary metadata file
        temp_metadata_path = f"/tmp/{model_name}_metadata.json"
        with open(temp_metadata_path, 'w') as f:
            f.write(metadata_content)
        
        # Save metadata to GitHub
        metadata_success = self.save_model_file(
            temp_metadata_path, 
            metadata_path, 
            f"Update {model_name} metadata"
        )
        
        # Clean up temp file
        os.remove(temp_metadata_path)
        
        return metadata_success
    
    def create_release(self, tag: str, title: str, description: str,
                      model_files: List[str] = None,
                      prerelease: bool = False) -> Optional[GitRelease]:
        """Create a GitHub release with model files"""
        if not self.is_available():
            logger.warning("GitHub repository not available")
            return None
        
        try:
            # Get the latest commit
            latest_commit = self.repo.get_commits()[0]
            
            # Create release
            release = self.repo.create_git_release(
                tag=tag,
                name=title,
                message=description,
                target_commitish=latest_commit.sha,
                prerelease=prerelease
            )
            
            # Upload model files as release assets
            if model_files:
                for model_file in model_files:
                    if os.path.exists(model_file):
                        with open(model_file, 'rb') as f:
                            release.upload_asset(
                                path=model_file,
                                label=os.path.basename(model_file),
                                content_type="application/octet-stream"
                            )
                        logger.info(f"Uploaded {model_file} as release asset")
            
            logger.info(f"Created release {tag}: {title}")
            return release
            
        except Exception as e:
            logger.error(f"Failed to create GitHub release: {e}")
            return None
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model from GitHub"""
        if not self.is_available():
            return []
        
        try:
            model_dir = f"models/{model_name}"
            contents = self.repo.get_contents(model_dir)
            
            versions = []
            for item in contents:
                if item.name.endswith('.pt'):
                    # Get file info
                    file_info = self.repo.get_contents(item.path)
                    
                    versions.append({
                        'name': item.name,
                        'path': item.path,
                        'size': file_info.size,
                        'sha': file_info.sha,
                        'download_url': file_info.download_url
                    })
            
            return versions
            
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []
    
    def download_model(self, model_name: str, version: str = "latest",
                      local_path: str = None) -> Optional[str]:
        """Download a model from GitHub"""
        if not self.is_available():
            return None
        
        try:
            if version == "latest":
                versions = self.get_model_versions(model_name)
                if not versions:
                    logger.error(f"No versions found for model {model_name}")
                    return None
                version = versions[0]['name']
            
            model_path = f"models/{model_name}/{version}"
            file_content = self.repo.get_contents(model_path)
            
            # Decode content
            model_data = base64.b64decode(file_content.content)
            
            # Save to local path
            if local_path is None:
                local_path = f"downloaded_{model_name}_{version}"
            
            with open(local_path, 'wb') as f:
                f.write(model_data)
            
            logger.info(f"Downloaded model {model_name} version {version} to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None
    
    def list_releases(self) -> List[Dict[str, Any]]:
        """List all releases in the repository"""
        if not self.is_available():
            return []
        
        try:
            releases = self.repo.get_releases()
            
            release_list = []
            for release in releases:
                release_list.append({
                    'tag_name': release.tag_name,
                    'name': release.title,
                    'body': release.body,
                    'created_at': release.created_at,
                    'published_at': release.published_at,
                    'prerelease': release.prerelease,
                    'assets': [asset.name for asset in release.get_assets()]
                })
            
            return release_list
            
        except Exception as e:
            logger.error(f"Failed to list releases: {e}")
            return []
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """Delete a specific model version from GitHub"""
        if not self.is_available():
            return False
        
        try:
            model_path = f"models/{model_name}/{version}"
            file_content = self.repo.get_contents(model_path)
            
            # Delete file
            self.repo.delete_file(
                model_path,
                f"Delete {model_name} version {version}",
                file_content.sha
            )
            
            logger.info(f"Deleted model {model_name} version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def get_repository_info(self) -> Dict[str, Any]:
        """Get repository information"""
        if not self.is_available():
            return {}
        
        try:
            return {
                'name': self.repo.name,
                'full_name': self.repo.full_name,
                'description': self.repo.description,
                'html_url': self.repo.html_url,
                'clone_url': self.repo.clone_url,
                'default_branch': self.repo.default_branch,
                'stars': self.repo.stargazers_count,
                'forks': self.repo.forks_count,
                'open_issues': self.repo.open_issues_count
            }
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            return {}


class ModelVersionManager:
    """Manages model versioning and metadata"""
    
    def __init__(self, github_manager: GitHubModelManager):
        self.github_manager = github_manager
    
    def create_model_metadata(self, model_path: str, model_name: str,
                            training_config: Dict[str, Any],
                            performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive model metadata"""
        
        # Get file info
        file_stats = os.stat(model_path)
        file_size = file_stats.st_size
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(model_path)
        
        # Create metadata
        metadata = {
            'model_name': model_name,
            'version': self._generate_version(),
            'created_at': time.time(),
            'file_size': file_size,
            'file_hash': file_hash,
            'training_config': training_config,
            'performance_metrics': performance_metrics,
            'model_type': 'tantra_conversational_speech',
            'framework': 'pytorch',
            'architecture': 'transformer_ocr_native'
        }
        
        return metadata
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _generate_version(self) -> str:
        """Generate version string based on timestamp"""
        return f"v{int(time.time())}"
    
    def save_model_with_versioning(self, model_path: str, model_name: str,
                                 training_config: Dict[str, Any],
                                 performance_metrics: Dict[str, Any]) -> bool:
        """Save model with full versioning and metadata"""
        
        # Create metadata
        metadata = self.create_model_metadata(
            model_path, model_name, training_config, performance_metrics
        )
        
        # Save model with metadata
        success = self.github_manager.save_model_with_metadata(
            model_path, model_name, metadata
        )
        
        if success:
            logger.info(f"Saved model {model_name} with version {metadata['version']}")
        
        return success