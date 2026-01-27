import os
import tarfile
import shutil
import sys


def generate_readme(base_path):
    content = """# Barebones Proxy Deployment

This package contains the essential components to run the LLM API Key Proxy on a minimal VPS.

## Prerequisites
- Python 3.9+
- A `.env` file with your API keys. Use `simple-env-template.env` as a reference.

## Installation
1. Extract the package: `tar -xzf barebones_proxy.tar.gz`
2. Install dependencies: `pip install -r requirements.txt`

## Running the Proxy
For minimal resource usage, run uvicorn directly:

```bash
python -m uvicorn src.proxy_app.main:app --host 0.0.0.0 --port 8000
```

Or using the launcher if you want the TUI:
```bash
python src/proxy_app/launcher_tui.py
```

## Configuration
Essential configs are in `config/router_config.yaml` and `config/model_rankings.yaml`.
"""
    readme_path = os.path.join(base_path, "README_VPS.md")
    with open(readme_path, "w") as f:
        f.write(content)
    return "README_VPS.md"


def create_barebones_package():
    # Identify project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    os.chdir(project_root)

    package_name = "barebones_proxy.tar.gz"
    essential_items = [
        "src/proxy_app/",
        "src/rotator_library/",
        "config/router_config.yaml",
        "config/model_rankings.yaml",
        "requirements.txt",
        "simple-env-template.env",
    ]

    readme = generate_readme(project_root)
    essential_items.append(readme)

    print(f"Creating {package_name} in {project_root}...")

    with tarfile.open(package_name, "w:gz") as tar:
        for item in essential_items:
            if os.path.exists(item):
                # Filter out __pycache__ and .git
                def filter_files(tarinfo):
                    if (
                        "__pycache__" in tarinfo.name
                        or ".git" in tarinfo.name
                        or ".csv" in tarinfo.name
                    ):
                        return None
                    return tarinfo

                print(f"Adding {item}...")
                tar.add(item, filter=filter_files)
            else:
                print(f"Warning: {item} not found!")

    size_bytes = os.path.getsize(package_name)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Package created successfully: {package_name} ({size_mb:.2f} MB)")
    return package_name, size_mb


if __name__ == "__main__":
    create_barebones_package()
