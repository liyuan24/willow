"""Discovery and explicit invocation helpers for Willow skills.

Willow discovers skills from two roots, in order:

* user skills: ``~/.willow``
* project skills: ``<cwd>/.willow``

The preferred layout mirrors Claude Code-style skill directories:
``.willow/skills/<skill-name>/SKILL.md``. For small skills, Willow also accepts
direct markdown files under ``.willow/skills/*.md``. Skill files may include a
simple YAML-like frontmatter block with ``name`` and ``description`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SKILL_FILE_NAME = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    """Metadata for one discovered skill."""

    name: str
    description: str
    path: Path
    location: str

    def load_content(self) -> str:
        """Load the skill body from disk."""
        return self.path.read_text(encoding="utf-8")


class SkillNotFoundError(LookupError):
    """Raised when an explicit skill invocation cannot be resolved."""


def load_available_skills(cwd: str | Path) -> list[Skill]:
    """Discover user and project skills visible from ``cwd``.

    Project skills override user skills with the same name. Results are sorted
    by name for stable prompt formatting and TUI suggestions.
    """
    project_root = Path(cwd).expanduser().resolve()
    roots = (
        ("user", Path.home() / ".willow"),
        ("project", project_root / ".willow"),
    )
    by_name: dict[str, Skill] = {}
    for location, root in roots:
        for path in _iter_skill_files(root):
            skill = _read_skill_metadata(path, location=location)
            if skill.name:
                by_name[skill.name.casefold()] = skill
    return sorted(by_name.values(), key=lambda skill: skill.name.casefold())


def format_skills_for_system_prompt(skills: list[Skill]) -> str:
    """Format skill metadata without inlining skill bodies."""
    if not skills:
        return "No Willow skills discovered."

    lines = [
        "Available Willow skills:",
        "Invoke a skill directly with /<skill_name> to load its body.",
    ]
    for skill in skills:
        description = f" - {skill.description}" if skill.description else ""
        lines.append(f"- {skill.name} ({skill.location}: {skill.path}){description}")
    return "\n".join(lines)


def resolve_skill(name: str, cwd: str | Path) -> Skill:
    """Resolve a skill by exact case-insensitive name."""
    wanted = name.strip().casefold()
    for skill in load_available_skills(cwd):
        if skill.name.casefold() == wanted:
            return skill
    raise SkillNotFoundError(f"Unknown skill: {name}")


def expand_skill_invocation(text: str, cwd: str | Path) -> str | None:
    """Expand ``/<skill_name> [task]`` into model-visible skill context.

    Returns ``None`` for non-skill input, including unknown slash commands, so
    callers can let normal command handling continue.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    command, _separator, rest = stripped.partition(" ")
    skill_name = command.removeprefix("/")
    if not skill_name:
        return None

    try:
        skill = resolve_skill(skill_name, cwd)
    except SkillNotFoundError:
        return None

    content = skill.load_content().strip()
    task_text = rest.strip() or "(no task text provided)"
    return (
        f"Use the Willow skill {skill.name!r} from {skill.path}.\n\n"
        f"<skill>\n{content}\n</skill>\n\n"
        f"User task:\n{task_text}"
    )


def render_skill_suggestions(prefix: str, cwd: str | Path, *, limit: int = 8) -> str:
    """Render matching skill names for TUI suggestion rows."""
    stripped = prefix.strip()
    if not stripped.startswith("/"):
        return ""
    command_prefix = stripped.split(maxsplit=1)[0].removeprefix("/")
    query = command_prefix.casefold()
    rows: list[str] = []
    for skill in load_available_skills(cwd):
        if query and not skill.name.casefold().startswith(query):
            continue
        description = f"  {skill.description}" if skill.description else ""
        rows.append(f"/{skill.name}{description}")
        if len(rows) >= limit:
            break
    return "\n".join(rows)


def _iter_skill_files(root: Path) -> list[Path]:
    skills_root = root / "skills"
    if not skills_root.is_dir():
        return []

    paths: list[Path] = []
    for child in sorted(skills_root.iterdir(), key=lambda path: path.name.casefold()):
        if child.is_dir():
            skill_file = child / SKILL_FILE_NAME
            if skill_file.is_file():
                paths.append(skill_file)
        elif child.is_file() and child.suffix.casefold() == ".md":
            paths.append(child)
    return paths


def _read_skill_metadata(path: Path, *, location: str) -> Skill:
    content = path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(content)
    default_name = path.parent.name if path.name == SKILL_FILE_NAME else path.stem
    name = metadata.get("name", default_name).strip()
    description = metadata.get("description", "").strip()
    if not description:
        description = _first_heading_or_paragraph(body)
    return Skill(name=name, description=description, path=path, location=location)


def _split_frontmatter(content: str) -> tuple[dict[str, str], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    metadata: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return metadata, "\n".join(lines[index + 1 :])
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"name", "description"}:
            metadata[key.strip()] = value.strip().strip("\"'")
    return {}, content


def _first_heading_or_paragraph(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        return stripped
    return ""
