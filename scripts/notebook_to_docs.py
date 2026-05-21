"""Convert course notebooks into MkDocs Markdown pages.

The generated pages keep notebook Markdown cells and code cells in order, so the
documentation stays aligned with the runnable notebooks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RAW_IMAGE_BASE = "https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images"

NOTEBOOKS = [
    ("unit1/01_introduction_to_diffusers.ipynb", "unit1/01-introduction-to-diffusers.md"),
    ("unit1/02_diffusion_models_from_scratch.ipynb", "unit1/02-diffusion-models-from-scratch.md"),
    ("unit2/01_finetuning_and_guidance.ipynb", "unit2/01-finetuning-and-guidance.md"),
    ("unit2/02_class_conditioned_diffusion_model_example.ipynb", "unit2/02-class-conditioned-diffusion-model.md"),
    ("unit3/stable_diffusion_introduction.ipynb", "unit3/stable-diffusion-introduction.md"),
    ("unit4/01_ddim_inversion.ipynb", "unit4/01-ddim-inversion.md"),
    ("unit4/02_diffusion_for_audio.ipynb", "unit4/02-diffusion-for-audio.md"),
    ("hackathon/dreambooth.ipynb", "hackathon/dreambooth.md"),
]

README_PAGES = [
    ("unit0/README.md", "unit0/index.md"),
    ("unit1/README.md", "unit1/index.md"),
    ("unit2/README.md", "unit2/index.md"),
    ("unit3/README.md", "unit3/index.md"),
    ("unit4/README.md", "unit4/index.md"),
    ("unit5/README.md", "unit5/index.md"),
    ("hackathon/README.md", "hackathon/index.md"),
]

MARKDOWN_PAGES = [
    ("unit5/01_video_generation_prerequisites.md", "unit5/01-video-generation-prerequisites.md"),
    ("unit5/02_video_tensors_and_datasets.md", "unit5/02-video-tensors-and-datasets.md"),
    ("unit5/03_toy_video_diffusion_from_scratch.md", "unit5/03-toy-video-diffusion-from-scratch.md"),
]

LINK_REPLACEMENTS = {
    "../unit0/README.md": "../unit0/index.md",
    "../unit1/README.md": "../unit1/index.md",
    "../unit2/README.md": "../unit2/index.md",
    "../unit3/README.md": "../unit3/index.md",
    "../unit4/README.md": "../unit4/index.md",
    "../unit5/README.md": "../unit5/index.md",
    "../hackathon/README.md": "../hackathon/index.md",
    "../docs/learning-path.md": "../learning-path.md",
    "../docs/modern-diffusion-roadmap.md": "../modern-diffusion-roadmap.md",
    "../CONTRIBUTING.md": "https://github.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/blob/master/CONTRIBUTING.md",
    "docs/modern-diffusion-roadmap.md": "modern-diffusion-roadmap.md",
    "01_introduction_to_diffusers.ipynb": "01-introduction-to-diffusers.md",
    "02_diffusion_models_from_scratch.ipynb": "02-diffusion-models-from-scratch.md",
    "01_finetuning_and_guidance.ipynb": "01-finetuning-and-guidance.md",
    "02_class_conditioned_diffusion_model_example.ipynb": "02-class-conditioned-diffusion-model.md",
    "stable_diffusion_introduction.ipynb": "stable-diffusion-introduction.md",
    "01_ddim_inversion.ipynb": "01-ddim-inversion.md",
    "02_diffusion_for_audio.ipynb": "02-diffusion-for-audio.md",
    "dreambooth.ipynb": "dreambooth.md",
    "01_video_generation_prerequisites.md": "01-video-generation-prerequisites.md",
    "02_video_tensors_and_datasets.md": "02-video-tensors-and-datasets.md",
    "03_toy_video_diffusion_from_scratch.md": "03-toy-video-diffusion-from-scratch.md",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def normalise_markdown(source: str) -> str:
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    for old, new in LINK_REPLACEMENTS.items():
        source = source.replace(old, new)
    source = re.sub(
        r"(!\[[^\]]*\]\()(?:(?:\.\./)+)?images/([^)]+)\)",
        lambda m: f"{m.group(1)}{RAW_IMAGE_BASE}/{m.group(2)})",
        source,
    )
    return source.strip()


def code_language(source: str) -> str:
    stripped = source.lstrip()
    if stripped.startswith("%%bash") or stripped.startswith("!"):
        return "bash"
    if stripped.startswith("%%writefile"):
        return "text"
    return "python"


def render_code_cell(source: str) -> str:
    source = source.rstrip()
    if not source:
        return ""
    fence = "````" if "```" in source else "```"
    return f"{fence}{code_language(source)}\n{source}\n{fence}"


def render_outputs(outputs: list[dict]) -> list[str]:
    rendered: list[str] = []
    for output in outputs:
        if output.get("output_type") == "stream":
            text = "".join(output.get("text", [])).rstrip()
            if text:
                rendered.append(f"```text\n{text}\n```")
        elif output.get("output_type") in {"execute_result", "display_data"}:
            data = output.get("data", {})
            text = data.get("text/plain")
            if text:
                rendered.append(f"```text\n{''.join(text).rstrip()}\n```")
        elif output.get("output_type") == "error":
            traceback = "\n".join(output.get("traceback", [])).rstrip()
            if traceback:
                rendered.append(f"```text\n{traceback}\n```")
    return rendered


def convert_notebook(src_rel: str, dst_rel: str) -> None:
    src = ROOT / src_rel
    nb = json.loads(read_text(src))
    repo_source = f"https://github.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/blob/master/{src_rel}"
    parts = [
        "<!-- This page is generated from the matching notebook by scripts/notebook_to_docs.py. -->",
        f"> 原始 Notebook：[{src_rel}]({repo_source})",
    ]

    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            md = normalise_markdown(source)
            if md:
                parts.append(md)
        elif cell.get("cell_type") == "code":
            code = render_code_cell(source)
            if code:
                parts.append(code)
            rendered_outputs = render_outputs(cell.get("outputs", []))
            parts.extend(rendered_outputs)

    write_text(DOCS / dst_rel, "\n\n".join(parts))


def copy_readme(src_rel: str, dst_rel: str) -> None:
    text = read_text(ROOT / src_rel)
    text = normalise_markdown(text)
    write_text(DOCS / dst_rel, text)


def copy_markdown(src_rel: str, dst_rel: str) -> None:
    text = read_text(ROOT / src_rel)
    text = normalise_markdown(text)
    write_text(DOCS / dst_rel, text)


def copy_static_files() -> None:
    source = read_text(ROOT / "unit2" / "finetune_model.py")
    write_text(DOCS / "unit2" / "finetune_model.py", source)
    toy_video = read_text(ROOT / "unit5" / "toy_video_diffusion.py")
    write_text(DOCS / "unit5" / "toy_video_diffusion.py", toy_video)


def main() -> None:
    for src, dst in README_PAGES:
        copy_readme(src, dst)
    for src, dst in MARKDOWN_PAGES:
        copy_markdown(src, dst)
    for src, dst in NOTEBOOKS:
        convert_notebook(src, dst)
    copy_static_files()


if __name__ == "__main__":
    main()
