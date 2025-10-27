#!/usr/bin/env python3
"""
System Cleanup Script for OCR-Native LLM
Cleans up temporary files, debug outputs, and optimizes system
"""

import os
import shutil
import glob
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemCleaner:
    """Comprehensive system cleanup and optimization"""
    
    def __init__(self, workspace_path: str = "/workspace"):
        self.workspace_path = Path(workspace_path)
        self.cleaned_files = 0
        self.cleaned_dirs = 0
        self.freed_space = 0
        
    def clean_debug_files(self):
        """Clean up debug and temporary files"""
        logger.info("🧹 Cleaning debug files...")
        
        # Debug output files
        debug_patterns = [
            "demo_output/*.png",
            "demo_output/*.jpg",
            "demo_output/*.jpeg",
            "*.log",
            "*.tmp",
            "*.temp",
            "debug_*.py",
            "test_*.py",
            "temp_*",
            "__pycache__",
            "*.pyc",
            "*.pyo"
        ]
        
        for pattern in debug_patterns:
            files = glob.glob(str(self.workspace_path / pattern), recursive=True)
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        self.cleaned_files += 1
                        self.freed_space += size
                        logger.info(f"  Removed file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        self.cleaned_dirs += 1
                        logger.info(f"  Removed directory: {file_path}")
                except Exception as e:
                    logger.warning(f"  Could not remove {file_path}: {e}")
    
    def clean_model_cache(self):
        """Clean up model cache and temporary weights"""
        logger.info("🗂️ Cleaning model cache...")
        
        cache_dirs = [
            "models/weights/temp_*",
            "models/cache/*",
            ".cache",
            "cache"
        ]
        
        for pattern in cache_dirs:
            files = glob.glob(str(self.workspace_path / pattern), recursive=True)
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        self.cleaned_files += 1
                        self.freed_space += size
                        logger.info(f"  Removed cache file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        self.cleaned_dirs += 1
                        logger.info(f"  Removed cache directory: {file_path}")
                except Exception as e:
                    logger.warning(f"  Could not remove {file_path}: {e}")
    
    def optimize_logs(self):
        """Clean up and optimize log files"""
        logger.info("📝 Optimizing log files...")
        
        log_files = glob.glob(str(self.workspace_path / "logs/*.log"), recursive=True)
        for log_file in log_files:
            try:
                # Keep only last 100 lines of each log file
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 100:
                    with open(log_file, 'w') as f:
                        f.writelines(lines[-100:])
                    logger.info(f"  Truncated log file: {log_file}")
                    
            except Exception as e:
                logger.warning(f"  Could not optimize {log_file}: {e}")
    
    def clean_test_outputs(self):
        """Clean up test outputs and temporary test files"""
        logger.info("🧪 Cleaning test outputs...")
        
        test_patterns = [
            "test_models/*",
            "test_outputs/*",
            "temp_test_*",
            "*.test_output"
        ]
        
        for pattern in test_patterns:
            files = glob.glob(str(self.workspace_path / pattern), recursive=True)
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        self.cleaned_files += 1
                        self.freed_space += size
                        logger.info(f"  Removed test file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        self.cleaned_dirs += 1
                        logger.info(f"  Removed test directory: {file_path}")
                except Exception as e:
                    logger.warning(f"  Could not remove {file_path}: {e}")
    
    def optimize_directories(self):
        """Create and optimize directory structure"""
        logger.info("📁 Optimizing directory structure...")
        
        # Create necessary directories
        dirs_to_create = [
            "logs",
            "cache",
            "temp",
            "outputs",
            "models/weights",
            "models/configs",
            "data/processed",
            "data/raw"
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.workspace_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Ensured directory exists: {dir_path}")
    
    def create_gitignore(self):
        """Create or update .gitignore file"""
        logger.info("📄 Updating .gitignore...")
        
        gitignore_content = """# OCR-Native LLM - Ignore patterns

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
*.pkl
*.pickle

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Cache
.cache/
cache/
*.tmp
*.temp

# Debug outputs
demo_output/*.png
demo_output/*.jpg
debug_*.py
temp_*

# Test outputs
test_models/
test_outputs/
*.test_output

# Model weights (large files)
models/weights/*.pt
models/weights/*.pth
!models/weights/.gitkeep

# Data files
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
"""
        
        gitignore_path = self.workspace_path / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        logger.info("  Updated .gitignore file")
    
    def create_keep_files(self):
        """Create .gitkeep files for empty directories"""
        logger.info("📄 Creating .gitkeep files...")
        
        keep_dirs = [
            "demo_output",
            "models/weights",
            "data/raw",
            "data/processed",
            "logs",
            "cache"
        ]
        
        for dir_path in keep_dirs:
            keep_file = self.workspace_path / dir_path / ".gitkeep"
            keep_file.touch()
            logger.info(f"  Created .gitkeep: {dir_path}/.gitkeep")
    
    def run_full_cleanup(self):
        """Run complete system cleanup"""
        logger.info("🚀 Starting comprehensive system cleanup...")
        start_time = time.time()
        
        # Run all cleanup operations
        self.clean_debug_files()
        self.clean_model_cache()
        self.optimize_logs()
        self.clean_test_outputs()
        self.optimize_directories()
        self.create_gitignore()
        self.create_keep_files()
        
        # Summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("✅ Cleanup completed successfully!")
        logger.info(f"📊 Summary:")
        logger.info(f"  Files removed: {self.cleaned_files}")
        logger.info(f"  Directories removed: {self.cleaned_dirs}")
        logger.info(f"  Space freed: {self.freed_space / (1024*1024):.2f} MB")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info("=" * 60)


def main():
    """Main cleanup function"""
    print("🧹 OCR-Native LLM System Cleanup")
    print("=" * 50)
    
    cleaner = SystemCleaner()
    cleaner.run_full_cleanup()
    
    print("\n🎉 System cleanup completed!")
    print("Your OCR-Native LLM workspace is now clean and optimized.")


if __name__ == "__main__":
    main()