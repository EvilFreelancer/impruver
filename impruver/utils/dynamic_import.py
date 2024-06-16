import importlib


def dynamic_import(module_class_string: str, class_name: str = None, default_class: str = None):
    try:
        module_name, extracted_class_name = module_class_string.rsplit(".", 1)
        class_name = class_name or extracted_class_name
        module = importlib.import_module(module_name)

        if hasattr(module, class_name):
            cls = getattr(module, class_name)
        elif default_class and hasattr(module, default_class):
            cls = getattr(module, default_class)
        else:
            raise ImportError(f"Class {class_name} or default class {default_class} not found in module {module_name}")

        return cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {module_class_string}: {e}")
