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


def github_request(url: str, token: str | None = None) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "diffusion-zero-to-hero-contributors",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def fetch_github_contributors(owner: str, repo: str, token: str | None = None) -> list[dict]:
    contributors: list[dict] = []
    page = 1

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=100&page={page}"
        batch = github_request(url, token)
        if not isinstance(batch, list) or not batch:
            break

        contributors.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    return contributors


def fetch_contributors_from_commits(owner: str, repo: str, token: str | None = None) -> list[dict]:
    """Collect GitHub-linked authors from commit metadata."""
    counts: dict[str, dict] = {}
    page = 1

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=100&page={page}"
        batch = github_request(url, token)
        if not isinstance(batch, list) or not batch:
            break

        for commit in batch:
            author = commit.get("author")
            if not author or author.get("type") != "User":
                continue
            login = author["login"]
            if login in BOT_LOGINS:
                continue
            if login not in counts:
                counts[login] = {
                    "login": login,
                    "html_url": author.get("html_url", f"https://github.com/{login}"),
                    "avatar_url": author.get("avatar_url", f"https://github.com/{login}.png?size=100"),
                    "contributions": 0,
                    "type": "User",
                }
            counts[login]["contributions"] += 1

        if len(batch) < 100:
            break
        page += 1

    return sorted(counts.values(), key=lambda item: item["contributions"], reverse=True)


def fetch_repo_owner(owner: str, repo: str, token: str | None = None) -> dict | None:
    data = github_request(f"https://api.github.com/repos/{owner}/{repo}", token)
    if not isinstance(data, dict):
        return None
    owner_info = data.get("owner")
    if not isinstance(owner_info, dict):
        return None
    return owner_info


def count_git_commits() -> int:
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return int(result.stdout.strip())


def build_owner_fallback(owner: str, repo: str, token: str | None = None) -> list[dict]:
    owner_info = fetch_repo_owner(owner, repo, token)
    if not owner_info:
        return []

    login = owner_info["login"]
    return [
        {
            "login": login,
            "html_url": owner_info.get("html_url", f"https://github.com/{login}"),
            "avatar_url": owner_info.get("avatar_url", f"https://github.com/{login}.png?size=100"),
            "contributions": count_git_commits(),
            "type": owner_info.get("type", "User"),
        }
    ]


def resolve_contributors(owner: str | None, repo: str | None, token: str | None = None) -> list[dict]:
    if not owner or not repo:
        return []

    try:
        contributors = fetch_github_contributors(owner, repo, token)
        if contributors:
            return contributors

        contributors = fetch_contributors_from_commits(owner, repo, token)
        if contributors:
            return contributors

        return build_owner_fallback(owner, repo, token)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"GitHub API unavailable ({exc}); falling back to repo owner")
        try:
            return build_owner_fallback(owner, repo, token)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return []


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
    contributors = resolve_contributors(owner, repo, token)
    update_readme(render_contributors(contributors))


if __name__ == "__main__":
    main()
