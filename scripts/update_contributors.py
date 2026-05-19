#!/usr/bin/env python3
"""Update the contributors section in README.md."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START = "<!-- contributors:start -->"
END = "<!-- contributors:end -->"
BOT_LOGINS = {"dependabot[bot]", "github-actions[bot]", "allcontributors[bot]"}


def get_repo_from_git() -> tuple[str | None, str | None]:
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", result.stdout.strip())
    if not match:
        return None, None
    return match.group(1), match.group(2).removesuffix(".git")


def fetch_github_contributors(owner: str, repo: str, token: str | None = None) -> list[dict]:
    contributors: list[dict] = []
    page = 1
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "diffusion-zero-to-hero-contributors",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=100&page={page}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.loads(response.read().decode())

        if not batch:
            break

        contributors.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    return contributors


def fetch_git_contributors() -> list[dict]:
    result = subprocess.run(
        ["git", "shortlog", "-sn", "--all", "--no-merges"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    contributors: list[dict] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        count, name = re.split(r"\s+", line.strip(), maxsplit=1)
        login = name.strip()
        contributors.append(
            {
                "login": login,
                "contributions": int(count),
                "html_url": f"https://github.com/{login}",
                "avatar_url": f"https://github.com/{login}.png?size=100",
                "type": "User",
            }
        )
    return contributors


def render_contributors(contributors: list[dict]) -> str:
    filtered = [
        contributor
        for contributor in contributors
        if contributor.get("login") not in BOT_LOGINS
        and contributor.get("type", "User") == "User"
    ]

    if not filtered:
        return "暂无贡献者记录。欢迎提交第一个 PR！\n"

    cells: list[str] = []
    for contributor in filtered:
        login = contributor["login"]
        url = contributor.get("html_url", f"https://github.com/{login}")
        avatar = contributor.get("avatar_url", f"https://github.com/{login}.png?size=100")
        commits = contributor.get("contributions", 0)
        cells.append(
            "      "
            f'<td align="center">'
            f'<a href="{url}"><img src="{avatar}" width="100px;" alt="{login}"/>'
            f"<br /><sub><b>{login}</b></sub></a>"
            f"<br /><sub>{commits} commits</sub>"
            f"</td>"
        )

    return (
        '<p align="center">\n'
        "  <table>\n"
        "    <tr>\n"
        f"{chr(10).join(cells)}\n"
        "    </tr>\n"
        "  </table>\n"
        "</p>\n"
    )


def update_readme(content: str) -> bool:
    readme = README.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(readme):
        print("Contributors markers not found in README.md", file=sys.stderr)
        raise SystemExit(1)

    replacement = f"{START}\n{content.rstrip()}\n{END}"
    new_readme = pattern.sub(replacement, readme)
    if new_readme == readme:
        print("No changes needed")
        return False

    README.write_text(new_readme, encoding="utf-8")
    print("Updated README.md")
    return True


def main() -> None:
    owner, repo = get_repo_from_git()
    token = os.environ.get("GITHUB_TOKEN")
    contributors: list[dict] = []

    if owner and repo:
        try:
            contributors = fetch_github_contributors(owner, repo, token)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"GitHub API unavailable ({exc}); falling back to git shortlog")

    if not contributors:
        contributors = fetch_git_contributors()

    changed = update_readme(render_contributors(contributors))
    raise SystemExit(0 if changed else 0)


if __name__ == "__main__":
    main()
