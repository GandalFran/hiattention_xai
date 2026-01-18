"""
Defect Seeder Module

Injects known defect patterns into clean code for synthetic data generation.
Enables controlled experiments and balanced datasets.
"""

import re
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InjectedDefect:
    """Information about an injected defect."""
    pattern_name: str
    cwe_id: str
    line_number: int
    original_code: str
    defective_code: str
    fix_suggestion: str
    severity: str  # low, medium, high, critical


class DefectSeeder:
    """
    Injects known defect patterns into clean code.
    
    Supports common vulnerability types from CWE Top 25.
    """
    
    DEFECT_PATTERNS = {
        'buffer_overflow': {
            'cwe': 'CWE-120',
            'severity': 'critical',
            'languages': ['c', 'cpp'],
            'pattern': r'(\w+)\s*\[\s*(\w+)\s*\]',
            'description': 'Array access without bounds checking'
        },
        'null_dereference': {
            'cwe': 'CWE-476',
            'severity': 'high',
            'languages': ['c', 'cpp', 'java'],
            'pattern': r'(\w+)->(\w+)',
            'description': 'Pointer dereference without null check'
        },
        'sql_injection': {
            'cwe': 'CWE-89',
            'severity': 'critical',
            'languages': ['python', 'java', 'php'],
            'pattern': r'execute\s*\(\s*["\'].*\+',
            'description': 'SQL query with string concatenation'
        },
        'xss': {
            'cwe': 'CWE-79',
            'severity': 'high',
            'languages': ['javascript', 'python', 'php'],
            'pattern': r'innerHTML\s*=',
            'description': 'Direct HTML injection without sanitization'
        },
        'use_after_free': {
            'cwe': 'CWE-416',
            'severity': 'critical',
            'languages': ['c', 'cpp'],
            'pattern': r'free\s*\(\s*(\w+)\s*\)',
            'description': 'Memory use after free'
        },
        'integer_overflow': {
            'cwe': 'CWE-190',
            'severity': 'high',
            'languages': ['c', 'cpp', 'java'],
            'pattern': r'(\w+)\s*\+\s*(\w+)',
            'description': 'Arithmetic without overflow check'
        },
        'hardcoded_credentials': {
            'cwe': 'CWE-798',
            'severity': 'critical',
            'languages': ['python', 'java', 'javascript'],
            'pattern': r'password\s*=\s*["\']',
            'description': 'Hardcoded password in source'
        },
        'path_traversal': {
            'cwe': 'CWE-22',
            'severity': 'high',
            'languages': ['python', 'java', 'php'],
            'pattern': r'open\s*\(\s*.*\+',
            'description': 'File path with user input'
        },
        'command_injection': {
            'cwe': 'CWE-78',
            'severity': 'critical',
            'languages': ['python', 'php', 'bash'],
            'pattern': r'os\.system\s*\(\s*.*\+',
            'description': 'Shell command with user input'
        },
        'race_condition': {
            'cwe': 'CWE-362',
            'severity': 'medium',
            'languages': ['c', 'cpp', 'java', 'python'],
            'pattern': r'if\s*\(.*\)\s*\{[^}]*sleep',
            'description': 'Time-of-check to time-of-use race'
        }
    }
    
    # Templates for injecting defects by language
    INJECTION_TEMPLATES = {
        'python': {
            'buffer_overflow': 'data[user_input]  # No bounds check',
            'null_dereference': 'result.value  # result might be None',
            'sql_injection': 'cursor.execute("SELECT * FROM users WHERE id=" + user_id)',
            'hardcoded_credentials': 'password = "admin123"  # Hardcoded credential',
            'path_traversal': 'open(base_path + user_filename)',
            'command_injection': 'os.system("ls " + user_path)',
        },
        'c': {
            'buffer_overflow': 'buffer[i] = value;  // i may exceed buffer size',
            'null_dereference': 'ptr->field = value;  // ptr not null-checked',
            'use_after_free': 'free(ptr); use(ptr);  // Use after free',
            'integer_overflow': 'size_t result = x + y;  // May overflow',
        },
        'java': {
            'null_dereference': 'object.method();  // object might be null',
            'sql_injection': 'stmt.execute("SELECT * FROM users WHERE id=" + userId);',
            'integer_overflow': 'int result = a + b;  // May overflow',
        }
    }
    
    @classmethod
    def list_patterns(cls) -> List[str]:
        """List all available defect patterns."""
        return list(cls.DEFECT_PATTERNS.keys())
    
    @classmethod
    def get_pattern_info(cls, pattern_name: str) -> Dict:
        """Get information about a defect pattern."""
        return cls.DEFECT_PATTERNS.get(pattern_name, {})
    
    @classmethod
    def inject_defect(
        cls,
        code: str,
        pattern_name: str,
        language: str = 'python',
        target_line: Optional[int] = None
    ) -> Tuple[str, Optional[InjectedDefect]]:
        """
        Inject a defect pattern into code.
        
        Args:
            code: Original clean code
            pattern_name: Name of defect pattern to inject
            language: Programming language
            target_line: Specific line to inject (random if None)
        
        Returns:
            (modified_code, InjectedDefect info or None if failed)
        """
        if pattern_name not in cls.DEFECT_PATTERNS:
            return code, None
        
        pattern_info = cls.DEFECT_PATTERNS[pattern_name]
        
        if language not in pattern_info['languages']:
            return code, None
        
        lines = code.split('\n')
        
        # Find suitable injection point
        if target_line is not None:
            injection_line = target_line
        else:
            # Find lines that could host this defect
            suitable_lines = cls._find_suitable_lines(lines, pattern_name, language)
            if not suitable_lines:
                # Insert at random non-empty line
                non_empty = [i for i, line in enumerate(lines) if line.strip()]
                if not non_empty:
                    return code, None
                injection_line = random.choice(non_empty)
            else:
                injection_line = random.choice(suitable_lines)
        
        if injection_line >= len(lines):
            return code, None
        
        original_line = lines[injection_line]
        
        # Get injection template
        templates = cls.INJECTION_TEMPLATES.get(language, {})
        defective_line = templates.get(pattern_name)
        
        if defective_line is None:
            # Generate based on original line
            defective_line = cls._generate_defective_line(original_line, pattern_name)
        
        # Preserve indentation
        indent = len(original_line) - len(original_line.lstrip())
        defective_line = ' ' * indent + defective_line.strip()
        
        # Replace line
        lines[injection_line] = defective_line
        
        # Create defect info
        defect = InjectedDefect(
            pattern_name=pattern_name,
            cwe_id=pattern_info['cwe'],
            line_number=injection_line,
            original_code=original_line,
            defective_code=defective_line,
            fix_suggestion=f"Add proper validation before {pattern_name.replace('_', ' ')}",
            severity=pattern_info['severity']
        )
        
        return '\n'.join(lines), defect
    
    @classmethod
    def _find_suitable_lines(
        cls,
        lines: List[str],
        pattern_name: str,
        language: str
    ) -> List[int]:
        """Find lines suitable for defect injection."""
        suitable = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip comments and empty lines
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
            
            # Pattern-specific heuristics
            if pattern_name == 'null_dereference':
                if '.' in stripped or '->' in stripped:
                    suitable.append(i)
            
            elif pattern_name == 'buffer_overflow':
                if '[' in stripped and ']' in stripped:
                    suitable.append(i)
            
            elif pattern_name == 'sql_injection':
                if 'execute' in stripped.lower() or 'query' in stripped.lower():
                    suitable.append(i)
            
            elif pattern_name in ('path_traversal', 'command_injection'):
                if 'open' in stripped or 'system' in stripped or 'exec' in stripped:
                    suitable.append(i)
            
            else:
                # Generic: lines with assignments or function calls
                if '=' in stripped or '(' in stripped:
                    suitable.append(i)
        
        return suitable
    
    @classmethod
    def _generate_defective_line(
        cls,
        original_line: str,
        pattern_name: str
    ) -> str:
        """Generate a defective version of a line."""
        # Simple transformations
        if pattern_name == 'null_dereference':
            # Remove null checks
            return re.sub(r'if\s*\([^)]*!=\s*None[^)]*\)\s*:', '', original_line)
        
        elif pattern_name == 'integer_overflow':
            # Add arithmetic without check
            return original_line + ' + 999999999'
        
        elif pattern_name == 'hardcoded_credentials':
            return 'password = "secretpassword123"'
        
        return original_line
    
    @classmethod
    def generate_synthetic_dataset(
        cls,
        clean_code_samples: List[str],
        patterns: Optional[List[str]] = None,
        language: str = 'python',
        defect_ratio: float = 0.5
    ) -> List[Tuple[str, int, Optional[InjectedDefect]]]:
        """
        Generate a synthetic dataset with controlled defect injection.
        
        Args:
            clean_code_samples: List of clean code samples
            patterns: Defect patterns to inject (all if None)
            language: Programming language
            defect_ratio: Ratio of samples to inject defects into
        
        Returns:
            List of (code, label, defect_info)
        """
        if patterns is None:
            patterns = [p for p, info in cls.DEFECT_PATTERNS.items()
                       if language in info['languages']]
        
        dataset = []
        
        for code in clean_code_samples:
            if random.random() < defect_ratio:
                # Inject defect
                pattern = random.choice(patterns)
                modified_code, defect = cls.inject_defect(code, pattern, language)
                
                if defect:
                    dataset.append((modified_code, 1, defect))
                else:
                    # Injection failed, keep as clean
                    dataset.append((code, 0, None))
            else:
                # Keep clean
                dataset.append((code, 0, None))
        
        return dataset
