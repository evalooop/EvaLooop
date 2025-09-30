"""System validation utilities for EvaLoop."""

import os
import torch
import logging
from typing import Dict, Any


class SystemValidator:
    """Validator for EvaLoop system requirements."""
    
    def __init__(self):
        """Initialize the system validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_all(
        self,
        check_gpu: bool = True,
        check_api_keys: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate all system requirements.
        
        Args:
            check_gpu: Whether to check GPU availability.
            check_api_keys: Whether to check API key configuration.
            
        Returns:
            Dictionary containing validation results.
        """
        results = {}
        
        # Check Python environment
        results["python_version"] = self._check_python_version()
        
        # Check required packages
        results["required_packages"] = self._check_required_packages()
        
        # Check GPU if requested
        if check_gpu:
            results["gpu_availability"] = self._check_gpu_availability()
            results["cuda_version"] = self._check_cuda_version()
        
        # Check API keys if requested
        if check_api_keys:
            results["openai_api_key"] = self._check_openai_api_key()
            results["anthropic_api_key"] = self._check_anthropic_api_key()
        
        # Check disk space
        results["disk_space"] = self._check_disk_space()
        
        return results
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        import sys
        
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        is_compatible = version.major >= required_major and version.minor >= required_minor
        
        return {
            "passed": is_compatible,
            "message": f"Python {version.major}.{version.minor}.{version.micro}" + 
                      ("" if is_compatible else f" (requires >= {required_major}.{required_minor})"),
            "version": f"{version.major}.{version.minor}.{version.micro}"
        }
    
    def _check_required_packages(self) -> Dict[str, Any]:
        """Check if required packages are installed."""
        required_packages = [
            "torch",
            "transformers", 
            "fire",
            "datasets",
            "evalplus",
            "pandas",
            "matplotlib",
            "seaborn"
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        all_installed = len(missing_packages) == 0
        
        message = f"Installed: {len(installed_packages)}/{len(required_packages)}"
        if missing_packages:
            message += f" (Missing: {', '.join(missing_packages)})"
        
        return {
            "passed": all_installed,
            "message": message,
            "installed": installed_packages,
            "missing": missing_packages
        }
    
    def _check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability."""
        try:
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            
            if gpu_available:
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                message = f"{gpu_count} GPU(s) available: {', '.join(gpu_names)}"
            else:
                message = "No GPU available (CPU-only mode)"
                gpu_names = []
            
            return {
                "passed": gpu_available,
                "message": message,
                "count": gpu_count,
                "names": gpu_names
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error checking GPU: {str(e)}",
                "count": 0,
                "names": []
            }
    
    def _check_cuda_version(self) -> Dict[str, Any]:
        """Check CUDA version."""
        try:
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                message = f"CUDA {cuda_version}"
                passed = True
            else:
                cuda_version = None
                message = "CUDA not available"
                passed = False
            
            return {
                "passed": passed,
                "message": message,
                "version": cuda_version
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error checking CUDA: {str(e)}",
                "version": None
            }
    
    def _check_openai_api_key(self) -> Dict[str, Any]:
        """Check OpenAI API key configuration."""
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key:
            # Basic validation - check if it looks like an API key
            is_valid_format = api_key.startswith("sk-") and len(api_key) > 20
            message = "OpenAI API key configured" if is_valid_format else "OpenAI API key format invalid"
            passed = is_valid_format
        else:
            message = "OpenAI API key not found in environment variables"
            passed = False
        
        return {
            "passed": passed,
            "message": message,
            "configured": api_key is not None
        }
    
    def _check_anthropic_api_key(self) -> Dict[str, Any]:
        """Check Anthropic API key configuration."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if api_key:
            # Basic validation
            is_valid_format = len(api_key) > 20
            message = "Anthropic API key configured" if is_valid_format else "Anthropic API key format invalid"
            passed = is_valid_format
        else:
            message = "Anthropic API key not found (optional)"
            passed = True  # This is optional
        
        return {
            "passed": passed,
            "message": message,
            "configured": api_key is not None
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            
            # Check current directory
            total, used, free = shutil.disk_usage(".")
            
            # Convert to GB
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            # Require at least 10GB free space
            required_gb = 10
            has_enough_space = free_gb >= required_gb
            
            message = f"{free_gb:.1f}GB free of {total_gb:.1f}GB total"
            if not has_enough_space:
                message += f" (requires at least {required_gb}GB)"
            
            return {
                "passed": has_enough_space,
                "message": message,
                "free_gb": free_gb,
                "total_gb": total_gb
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error checking disk space: {str(e)}",
                "free_gb": 0,
                "total_gb": 0
            }
