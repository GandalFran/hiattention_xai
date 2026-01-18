"""
Code Parser Module

Extracts function-level information from source code using AST analysis.
Supports Python, Java, and C/C++ through language-specific parsers.
"""

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path


@dataclass
class FunctionInfo:
    """Information about a single function/method."""
    func_id: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    lines: List[str]
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    calls: List[str] = field(default_factory=list)  # Functions this calls
    variables_read: Set[str] = field(default_factory=set)
    variables_written: Set[str] = field(default_factory=set)
    complexity: int = 1  # Cyclomatic complexity
    module_id: str = ""
    class_name: Optional[str] = None
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)


@dataclass 
class ModuleInfo:
    """Information about a module/file."""
    module_id: str
    file_path: str
    functions: List[str] = field(default_factory=list)  # Function IDs
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    global_variables: Set[str] = field(default_factory=set)
    loc: int = 0  # Lines of code


class PythonFunctionVisitor(ast.NodeVisitor):
    """AST visitor to extract function information from Python code."""
    
    def __init__(self, source_lines: List[str], file_path: str):
        self.source_lines = source_lines
        self.file_path = file_path
        self.functions: Dict[str, FunctionInfo] = {}
        self.current_class: Optional[str] = None
        self.imports: List[str] = []
        self.global_vars: Set[str] = set()
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ''
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)
    
    def _process_function(self, node):
        """Extract information from a function definition."""
        # Build function ID
        if self.current_class:
            func_id = f"{self.file_path}::{self.current_class}.{node.name}"
            is_method = True
        else:
            func_id = f"{self.file_path}::{node.name}"
            is_method = False
        
        # Extract lines
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start_line + 1
        lines = self.source_lines[start_line:end_line]
        
        # Extract parameters
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(decorator.attr)
        
        # Analyze function body
        calls, vars_read, vars_written, complexity = self._analyze_body(node)
        
        func_info = FunctionInfo(
            func_id=func_id,
            name=node.name,
            file_path=self.file_path,
            start_line=start_line,
            end_line=end_line,
            lines=lines,
            parameters=params,
            docstring=docstring,
            calls=calls,
            variables_read=vars_read,
            variables_written=vars_written,
            complexity=complexity,
            module_id=self.file_path,
            class_name=self.current_class,
            is_method=is_method,
            decorators=decorators
        )
        
        self.functions[func_id] = func_info
        self.generic_visit(node)
    
    def _analyze_body(self, node) -> Tuple[List[str], Set[str], Set[str], int]:
        """Analyze function body for calls, variables, and complexity."""
        calls = []
        vars_read = set()
        vars_written = set()
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Track function calls
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
            
            # Track variable reads
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Load):
                    vars_read.add(child.id)
                elif isinstance(child.ctx, ast.Store):
                    vars_written.add(child.id)
            
            # Increment complexity for branches
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                 ast.With, ast.Assert, ast.comprehension)):
                complexity += 1
            
            # Boolean operators add complexity
            if isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return calls, vars_read, vars_written, complexity


class CodeParser:
    """
    Main code parser that extracts function-level information from repositories.
    Supports multiple programming languages.
    """
    
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.js': 'javascript',
        '.ts': 'typescript'
    }
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.functions: Dict[str, FunctionInfo] = {}
        self.modules: Dict[str, ModuleInfo] = {}
    
    def parse_repository(self) -> Dict[str, FunctionInfo]:
        """
        Parse all supported files in repository.
        
        Returns:
            Dictionary mapping function_id to FunctionInfo
        """
        for file_path in self._find_source_files():
            self._parse_file(file_path)
        
        return self.functions
    
    def _find_source_files(self) -> List[Path]:
        """Find all source files in repository."""
        source_files = []
        
        for ext in self.SUPPORTED_EXTENSIONS:
            source_files.extend(self.repo_path.rglob(f"*{ext}"))
        
        # Filter out common non-source directories
        excluded_dirs = {'node_modules', 'venv', '.git', '__pycache__', 'build', 'dist'}
        source_files = [
            f for f in source_files 
            if not any(excluded in f.parts for excluded in excluded_dirs)
        ]
        
        return source_files
    
    def _parse_file(self, file_path: Path):
        """Parse a single source file."""
        ext = file_path.suffix.lower()
        language = self.SUPPORTED_EXTENSIONS.get(ext)
        
        if language == 'python':
            self._parse_python_file(file_path)
        elif language == 'java':
            self._parse_java_file(file_path)
        elif language in ('c', 'cpp'):
            self._parse_c_file(file_path)
        # Add more languages as needed
    
    def _parse_python_file(self, file_path: Path):
        """Parse a Python file using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
                source_lines = source.split('\n')
            
            tree = ast.parse(source)
            
            rel_path = str(file_path.relative_to(self.repo_path))
            visitor = PythonFunctionVisitor(source_lines, rel_path)
            visitor.visit(tree)
            
            # Add functions to global dict
            self.functions.update(visitor.functions)
            
            # Create module info
            module_info = ModuleInfo(
                module_id=rel_path,
                file_path=str(file_path),
                functions=list(visitor.functions.keys()),
                imports=visitor.imports,
                loc=len(source_lines)
            )
            self.modules[rel_path] = module_info
            
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    def _parse_java_file(self, file_path: Path):
        """Parse a Java file using regex (simplified)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
                source_lines = source.split('\n')
            
            rel_path = str(file_path.relative_to(self.repo_path))
            
            # Regex for Java methods
            method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*\{'
            
            for match in re.finditer(method_pattern, source):
                return_type = match.group(1)
                method_name = match.group(2)
                params_str = match.group(3)
                
                start_pos = match.start()
                start_line = source[:start_pos].count('\n')
                
                # Find matching closing brace (simplified)
                brace_count = 1
                end_line = start_line
                for i, line in enumerate(source_lines[start_line + 1:], start_line + 1):
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0:
                        end_line = i
                        break
                
                func_id = f"{rel_path}::{method_name}"
                lines = source_lines[start_line:end_line + 1]
                
                # Parse parameters
                params = []
                if params_str.strip():
                    for param in params_str.split(','):
                        parts = param.strip().split()
                        if len(parts) >= 2:
                            params.append(parts[-1])
                
                func_info = FunctionInfo(
                    func_id=func_id,
                    name=method_name,
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    lines=lines,
                    parameters=params,
                    return_type=return_type,
                    module_id=rel_path
                )
                
                self.functions[func_id] = func_info
            
            # Create module info
            self.modules[rel_path] = ModuleInfo(
                module_id=rel_path,
                file_path=str(file_path),
                functions=[fid for fid in self.functions if fid.startswith(rel_path)],
                loc=len(source_lines)
            )
            
        except Exception as e:
            print(f"Error parsing Java file {file_path}: {e}")
    
    def _parse_c_file(self, file_path: Path):
        """Parse a C/C++ file using regex (simplified)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
                source_lines = source.split('\n')
            
            rel_path = str(file_path.relative_to(self.repo_path))
            
            # Regex for C functions (simplified)
            func_pattern = r'^\s*(?:static\s+)?(?:inline\s+)?(?:extern\s+)?(\w+(?:\s*\*)*)\s+(\w+)\s*\(([^)]*)\)\s*\{'
            
            for match in re.finditer(func_pattern, source, re.MULTILINE):
                return_type = match.group(1)
                func_name = match.group(2)
                params_str = match.group(3)
                
                # Skip common false positives
                if func_name in ('if', 'while', 'for', 'switch'):
                    continue
                
                start_pos = match.start()
                start_line = source[:start_pos].count('\n')
                
                # Find matching closing brace
                brace_count = 1
                end_line = start_line
                for i, line in enumerate(source_lines[start_line + 1:], start_line + 1):
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0:
                        end_line = i
                        break
                
                func_id = f"{rel_path}::{func_name}"
                lines = source_lines[start_line:end_line + 1]
                
                func_info = FunctionInfo(
                    func_id=func_id,
                    name=func_name,
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    lines=lines,
                    return_type=return_type,
                    module_id=rel_path
                )
                
                self.functions[func_id] = func_info
            
            self.modules[rel_path] = ModuleInfo(
                module_id=rel_path,
                file_path=str(file_path),
                functions=[fid for fid in self.functions if fid.startswith(rel_path)],
                loc=len(source_lines)
            )
            
        except Exception as e:
            print(f"Error parsing C file {file_path}: {e}")
    
    def get_function(self, func_id: str) -> Optional[FunctionInfo]:
        """Get function info by ID."""
        return self.functions.get(func_id)
    
    def get_module(self, module_id: str) -> Optional[ModuleInfo]:
        """Get module info by ID."""
        return self.modules.get(module_id)
    
    def get_all_functions(self) -> List[FunctionInfo]:
        """Get list of all parsed functions."""
        return list(self.functions.values())
    
    def get_functions_by_module(self, module_id: str) -> List[FunctionInfo]:
        """Get all functions in a module."""
        return [f for f in self.functions.values() if f.module_id == module_id]
    
    def get_statistics(self) -> Dict:
        """Get parsing statistics."""
        total_loc = sum(m.loc for m in self.modules.values())
        return {
            'num_modules': len(self.modules),
            'num_functions': len(self.functions),
            'total_loc': total_loc,
            'avg_function_length': sum(len(f.lines) for f in self.functions.values()) / max(1, len(self.functions)),
            'avg_complexity': sum(f.complexity for f in self.functions.values()) / max(1, len(self.functions))
        }
