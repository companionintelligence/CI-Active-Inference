"""
Tests for main CLI functionality
"""

import pytest
import subprocess
import sys
import os


class TestMainCLI:
    """Test main CLI functionality."""
    
    def test_help_command(self):
        """Test that help command works."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            cwd='/home/runner/work/CI-Active-Inference-Slopbox/CI-Active-Inference-Slopbox',
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'CI Active Inference System' in result.stdout
        assert 'demo' in result.stdout
        assert 'train' in result.stdout
        assert 'init-db' in result.stdout
    
    def test_status_command(self):
        """Test status command."""
        # First ensure database is initialized
        subprocess.run(
            [sys.executable, 'main.py', 'init-db'],
            cwd='/home/runner/work/CI-Active-Inference-Slopbox/CI-Active-Inference-Slopbox',
            capture_output=True
        )
        
        subprocess.run(
            [sys.executable, 'main.py', 'sample-data'],
            cwd='/home/runner/work/CI-Active-Inference-Slopbox/CI-Active-Inference-Slopbox',
            capture_output=True
        )
        
        # Test status command
        result = subprocess.run(
            [sys.executable, 'main.py', 'status'],
            cwd='/home/runner/work/CI-Active-Inference-Slopbox/CI-Active-Inference-Slopbox',
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'CI Active Inference System Status' in result.stderr
        assert 'Current User:' in result.stderr
    
    def test_init_db_command(self):
        """Test database initialization command."""
        result = subprocess.run(
            [sys.executable, 'main.py', 'init-db'],
            cwd='/home/runner/work/CI-Active-Inference-Slopbox/CI-Active-Inference-Slopbox',
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Database initialized successfully' in result.stderr
    
    def test_sample_data_command(self):
        """Test sample data creation command."""
        # First ensure database is initialized
        subprocess.run(
            [sys.executable, 'main.py', 'init-db'],
            cwd='/home/runner/work/CI-Active-Inference-Slopbox/CI-Active-Inference-Slopbox',
            capture_output=True
        )
        
        # Test sample data creation
        result = subprocess.run(
            [sys.executable, 'main.py', 'sample-data'],
            cwd='/home/runner/work/CI-Active-Inference-Slopbox/CI-Active-Inference-Slopbox',
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Sample data created successfully' in result.stderr


class TestShellScripts:
    """Test shell scripts."""
    
    def test_shell_scripts_exist(self):
        """Test that all shell scripts exist and are executable."""
        base_dir = '/home/runner/work/CI-Active-Inference-Slopbox/CI-Active-Inference-Slopbox'
        scripts = ['setup.sh', 'run_demo.sh', 'run_api.sh', 'run_tests.sh']
        
        for script in scripts:
            script_path = os.path.join(base_dir, script)
            assert os.path.exists(script_path), f"Script {script} should exist"
            assert os.access(script_path, os.X_OK), f"Script {script} should be executable"