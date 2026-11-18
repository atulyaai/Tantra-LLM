"""Regression checks for public pretraining orchestration dispatch."""
import ast
from pathlib import Path


def test_pretraining_jobs_dispatch_to_shared_pilot_runner():
    source = Path(__file__).resolve().parents[1] / "Tantra" / "pretrain.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    expected = {"continue_run": ("train", "pilot_run"), "job_run": ("train_base", "pilot_run")}
    for function_name, (alias, target) in expected.items():
        function = functions[function_name]
        assignments = [node for node in function.body if isinstance(node, ast.Assign)]
        assert any(
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == alias
            and isinstance(node.value, ast.Name)
            and node.value.id == target
            for node in assignments
        ), f"{function_name} must dispatch {alias} to {target}, not to itself"
