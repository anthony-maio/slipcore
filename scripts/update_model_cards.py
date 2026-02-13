"""Update Eve-2 model cards on HuggingFace with logo and fixes."""
import os
import sys
import warnings
import tempfile

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
warnings.filterwarnings("ignore")

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])

LOGO_LINE = '<div align="center">\n<img src="eve-2-swarm.jpg" alt="Eve-2 Swarm" width="600">\n</div>\n'


def upload_readme(repo_id, content):
    tmp = os.path.join(tempfile.gettempdir(), "readme_tmp.md")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    api.upload_file(
        path_or_fileobj=tmp,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card with Eve-2 logo",
    )
    print(f"  Uploaded: {repo_id}")


def fix_it_readme():
    """Fix IT-272M: remove stray code block, fix image display."""
    repo = "anthonym21/Eve-2-MoE-IT-272M"
    print(f"Fixing {repo}...")
    path = api.hf_hub_download(repo_id=repo, filename="README.md", repo_type="model")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove the stray ``` that opens a code block before the div
    # The pattern is: ---\n\n```\n\n<div align="center">
    content = content.replace(
        '---\n\n```\n\n<div align="center">\n![Eve-2 Swarm](eve-2-swarm.jpg)  <h1>Eve-2-MoE-IT-272M</h1>\n<h3>The Foundation for Nano-Scale Swarm Intelligence</h3>\n</div>',
        '---\n\n<div align="center">\n<img src="eve-2-swarm.jpg" alt="Eve-2 Swarm" width="600">\n<h1>Eve-2-MoE-IT-272M</h1>\n<h3>The Foundation for Nano-Scale Swarm Intelligence</h3>\n</div>',
    )

    upload_readme(repo, content)


def add_logo_to_nano(repo_id):
    """Add logo to a Nano specialist model card."""
    print(f"Updating {repo_id}...")
    path = api.hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Already has the logo?
    if "eve-2-swarm" in content:
        print(f"  Skipped (already has logo)")
        return

    # Insert logo after the YAML frontmatter (second ---)
    parts = content.split("---", 2)
    if len(parts) >= 3:
        # parts[0] is empty (before first ---), parts[1] is YAML, parts[2] is body
        body = parts[2]
        # Get the model name from the first heading
        model_short = repo_id.split("/")[1]
        new_body = f"\n\n{LOGO_LINE}\n{body.lstrip()}"
        content = f"---{parts[1]}---{new_body}"

    upload_readme(repo_id, content)


if __name__ == "__main__":
    # Fix IT-272M (main model - has broken HTML)
    fix_it_readme()

    # Add logo to all 8 Nano specialists
    nano_repos = [
        "anthonym21/Eve-2-MoE-NanoFunction-272M",
        "anthonym21/Eve-2-MoE-NanoExtract-272M",
        "anthonym21/Eve-2-MoE-NanoPII-272M",
        "anthonym21/Eve-2-MoE-NanoRouter-272M",
        "anthonym21/Eve-2-MoE-NanoSummary-272M",
        "anthonym21/Eve-2-MoE-NanoSQL-272M",
        "anthonym21/Eve-2-MoE-NanoPrompt-272M",
        "anthonym21/Eve-2-MoE-NanoCommit-272M",
    ]
    for repo in nano_repos:
        add_logo_to_nano(repo)

    print("\nDone! All 9 model cards updated.")
