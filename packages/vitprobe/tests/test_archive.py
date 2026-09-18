"""Self-contained artifact integrity checks; source checkout is not required."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_exact_hashes_and_small_files():
    manifest = json.loads((ROOT / 'evidence_manifest.json').read_text())
    for entry in manifest['files']:
        path = ROOT / entry['copy']
        data = path.read_bytes()
        assert path.suffix in {'.md', '.csv', '.json'}
        assert len(data) == entry['bytes'] <= 1_000_000
        assert hashlib.sha256(data).hexdigest() == entry['sha256']


def test_runtime_imports_are_independent():
    allowed = {'torch', 'numpy', 'math', 'itertools', 'random', 'contextlib', 'vitprobe', 'dataclasses'}
    for path in (ROOT / 'src/vitprobe').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for item in node.names:
                    if item.name == 'pandas':
                        assert path.name == 'sweep.py'
                        method = next(n for n in ast.walk(ast.parse(path.read_text()))
                                      if isinstance(n, ast.FunctionDef) and n.name == 'results')
                        assert method.lineno < node.lineno <= method.end_lineno
                    else:
                        assert item.name.split('.')[0] in allowed
            if isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module.split('.')[0] in allowed


def test_domain_name_exception_is_only_approved_factory():
    import re
    for path in (ROOT / 'src/vitprobe').glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Name, ast.arg, ast.Attribute)):
                name = getattr(node, 'name', getattr(node, 'id', getattr(node, 'arg', getattr(node, 'attr', ''))))
                if re.search('voc|coco|mctformer|cam', name, re.IGNORECASE):
                    assert path.name == 'layout.py' and name == 'mctformer_plus'


def test_template_json_and_internal_references():
    schema = json.loads((ROOT / 'templates/manifest.schema.json').read_text())
    assert {'precision', 'autocast', 'probe_set_sha256'} <= set(schema['required'])
    assert set(schema['required']) <= schema['properties'].keys()
    def visit(node):
        if isinstance(node, dict):
            if '$ref' in node:
                assert node['$ref'].startswith('#/$defs/')
                assert node['$ref'].split('/')[-1] in schema['$defs']
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(schema)


def test_isolated_import_without_repository_or_dataframe():
    script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'analysis', 'models', 'tools', 'pandas'}:
            raise ImportError('deliberately blocked dependency')
sys.meta_path.insert(0, Block())
from vitprobe import TokenLayout, stats, relations, artifacts, spectral, state
from vitprobe.graph import aggregate_p2p, propagate_weights
from vitprobe.bootstrap import paired_image_bootstrap, paired_confusion_bootstrap
from vitprobe.sweep import ConfigSweep
assert TokenLayout.dinov3().n_patch == 784
sweep = ConfigSweep([{}], 1, 2)
import numpy as np
target = np.asarray([0, 1])
sweep.observe(0, target, target)
assert sweep.results()[0]['mean_iou'] == 1
try:
    sweep.results(dataframe=True)
except ImportError as error:
    assert 'vitprobe[dataframe]' in str(error)
else:
    raise AssertionError('optional import guard did not work')
'''
    subprocess.run([sys.executable, '-I', '-c', script], check=True, capture_output=True, text=True)
