"""Shared module for dynamically discovering and loading experiment types."""

import importlib
import re
from pathlib import Path


EXPERIMENT_TYPES_DIR = Path(__file__).resolve().parent.parent / "experiment_types"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config_files"
EXPERIMENT_NAME_OVERRIDES = {
    "TextureExperimentFB": "Texture FB",
    "TextureExperimentFBVGG": "Texture FB-VGG",
    "TextureExperimentFBVGGMultiTime": "Texture FB-VGGMultiTime",
}


def class_name_to_display_name(class_name):
    """Convert CamelCase class name to display name with spaces."""
    base_name = class_name[:-10] if class_name.endswith("Experiment") else class_name
    return re.sub(r"(?<!^)(?=[A-Z])", " ", base_name).strip()


def discover_experiment_types():
    """Dynamically discover and load experiment types from the experiment_types package.
    
    Returns:
        dict: Mapping of display names to experiment classes.
    """
    discovered = {}
    for module_path in sorted(EXPERIMENT_TYPES_DIR.glob("*.py")):
        if module_path.name == "__init__.py":
            continue

        module_name = module_path.stem
        try:
            module = importlib.import_module(f"experiment_types.{module_name}")
        except Exception as e:
            print(f"Skipping experiment module '{module_name}': {e}")
            continue

        exp_class = getattr(module, module_name, None)
        if exp_class is None:
            continue

        display_name = EXPERIMENT_NAME_OVERRIDES.get(module_name, class_name_to_display_name(module_name))
        discovered[display_name] = exp_class

    return discovered


def get_experiment_list():
    """Get sorted list of available experiment names (display names).
    
    Returns:
        list: Sorted list of experiment display names.
    """
    exp_types = discover_experiment_types()
    return sorted(exp_types.keys())


def resolve_experiment_config_file(exp_name, exp_type=None):
    """Resolve the config file for an experiment display name and/or class."""
    candidates = []

    if exp_type is None:
        exp_type = discover_experiment_types().get(exp_name)

    display_stem = exp_name.replace(' ', '_')
    candidates.append(CONFIG_DIR / f"{display_stem}_config.yaml")
    candidates.append(CONFIG_DIR / f"{display_stem.lower()}_config.yaml")
    candidates.append(CONFIG_DIR / f"{display_stem}.yaml")

    if exp_type is not None:
        module_stem = exp_type.__module__.split('.')[-1]
        class_stem = exp_type.__name__
        candidates.append(CONFIG_DIR / f"{module_stem}_config.yaml")
        candidates.append(CONFIG_DIR / f"{class_stem}_config.yaml")
        candidates.append(CONFIG_DIR / f"{module_stem}.yaml")
        candidates.append(CONFIG_DIR / f"{class_stem}.yaml")

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.is_file():
            return candidate

    return candidates[0]
