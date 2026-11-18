from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

SKILL_ROOT = Path(__file__).with_name("skills")
CUSTOM_SKILL_ROOT = Path(__file__).with_name("custom_skills")
LOGGER = logging.getLogger("ComfyUI.QwenH3Prompt.skills")
SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class SkillSpec:
    """A discovered prompt-only Skill available to the local node."""

    id: str
    path: Path
    description: str
    source: str
    display_name: str | None = None
    version: str | None = None
    tags: tuple[str, ...] = ()
    h3_output: bool = True


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _list_values(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(item for item in value if item.strip())
    if not isinstance(value, str) or not value.strip():
        return ()
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
        return tuple(
            item
            for item in (_scalar(part) for part in value.split(","))
            if item.strip()
        )
    return (value,)


def _front_matter(path: Path) -> dict[str, str]:
    """Read the small subset of YAML front matter needed for Skill discovery."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}

    end = next(
        (
            index
            for index in range(1, len(lines))
            if lines[index].strip() == "---"
            and not lines[index].startswith((" ", "\t"))
        ),
        None,
    )
    if end is None:
        return {}

    values: dict[str, str] = {}
    index = 1
    while index < end:
        line = lines[index]
        match = re.match(
            r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)$",
            line,
        )
        if not match or match.group("indent"):
            index += 1
            continue

        key = match.group("key").replace("-", "_").casefold()
        value = match.group("value").strip()
        if value.startswith(("|", ">")):
            block: list[str] = []
            cursor = index + 1
            while cursor < end:
                candidate = lines[cursor]
                if candidate.strip() and not candidate.startswith((" ", "\t")):
                    break
                block.append(candidate)
                cursor += 1
            non_empty = [line for line in block if line.strip()]
            indent = min((len(line) - len(line.lstrip()) for line in non_empty), default=0)
            block = [line[indent:] if line.strip() else "" for line in block]
            values[key] = ("\n" if value.startswith("|") else " ").join(block).strip()
            index = cursor
            continue
        if value:
            values[key] = _scalar(value)
        index += 1
    return values


def _meta_yaml(path: Path) -> dict[str, str | tuple[str, ...]]:
    """Read simple scalar and list metadata from the bundled meta.yaml convention."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    values: dict[str, str | tuple[str, ...]] = {}
    index = 0
    while index < len(lines):
        match = re.match(r"^(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)$", lines[index])
        if not match:
            index += 1
            continue
        key = match.group("key").replace("-", "_").casefold()
        value = match.group("value").strip()
        if value:
            values[key] = _scalar(value)
            index += 1
            continue

        items: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            item = re.match(r"^\s*-\s*(.+?)\s*$", lines[cursor])
            if not item:
                break
            items.append(_scalar(item.group(1)))
            cursor += 1
        if items:
            values[key] = tuple(items)
        index = cursor
    return values


def _as_bool(value: object, default: bool = True) -> bool:
    """Parse the loose boolean spellings used in Skill front matter and meta.yaml."""
    if value is None:
        return default
    return str(value).strip().casefold() not in {"false", "no", "off", "0", "disabled", "none"}


def _metadata(
    skill_root: Path,
) -> tuple[str, str, str | None, str | None, tuple[str, ...], bool]:
    front = _front_matter(skill_root / "SKILL.md")
    meta = _meta_yaml(skill_root / "meta.yaml")
    skill_id = front.get("name", "").strip() or skill_root.name
    description = (
        front.get("description", "").strip()
        or str(meta.get("summary_en", "")).strip()
        or str(meta.get("desc_en", "")).strip()
        or skill_id
    )
    description = " ".join(description.split())
    display_name = (
        front.get("display_name", "").strip()
        or front.get("display_name_en", "").strip()
        or front.get("display_name_zh", "").strip()
        or str(meta.get("display_name_en", "")).strip()
        or str(meta.get("display_name_zh", "")).strip()
        or None
    )
    version = (
        front.get("version", "").strip()
        or str(meta.get("version", "")).strip()
        or None
    )
    # H3 prompt output is the node's default contract. A Skill that is inherently
    # non-video (image prompt rewriting, translation, music, ...) can opt out with
    # `h3-output: false` in its SKILL.md front matter or `h3_output: false` in meta.yaml.
    h3_output = _as_bool(
        front.get("h3_output", meta.get("h3_output")),
        default=True,
    )
    raw_tags: list[str] = []
    for values in (
        _list_values(front.get("tags")),
        _list_values(front.get("tag_en")),
        _list_values(front.get("tag_cn")),
        _list_values(meta.get("tags")),
        _list_values(meta.get("tag_en")),
        _list_values(meta.get("tag_cn")),
        _list_values(meta.get("complete_tags_en")),
        _list_values(meta.get("complete_tags_cn")),
    ):
        raw_tags.extend(values)
    return skill_id, description, display_name, version, tuple(dict.fromkeys(raw_tags)), h3_output


def _discover_skills(root: Path, source: str) -> list[SkillSpec]:
    if not root.is_dir():
        return []
    resolved_root = root.resolve()
    discovered: list[SkillSpec] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not candidate.is_dir():
            continue
        skill_path = candidate.resolve()
        if skill_path.parent != resolved_root:
            LOGGER.warning("Ignoring Skill outside %s: %s", resolved_root, candidate)
            continue
        skill_file = skill_path / "SKILL.md"
        if not skill_file.is_file() or skill_file.resolve().parent != skill_path:
            LOGGER.warning("Ignoring Skill without SKILL.md: %s", skill_path)
            continue
        skill_id, description, display_name, version, tags, h3_output = _metadata(skill_path)
        if not SKILL_ID_RE.fullmatch(skill_id) or skill_id == "auto":
            LOGGER.warning("Ignoring Skill with invalid id %r: %s", skill_id, skill_path)
            continue
        discovered.append(
            SkillSpec(
                id=skill_id,
                path=skill_path,
                description=description,
                source=source,
                display_name=display_name,
                version=version,
                tags=tags,
                h3_output=h3_output,
            )
        )
    return discovered


def discover_skill_registry(
    builtin_root: Path | None = None,
    custom_root: Path | None = None,
) -> tuple[SkillSpec, ...]:
    """Discover bundled Skills first, then non-overriding user Skills."""
    builtin_root = SKILL_ROOT if builtin_root is None else builtin_root
    custom_root = CUSTOM_SKILL_ROOT if custom_root is None else custom_root
    registry: list[SkillSpec] = []
    seen: set[str] = set()
    candidates = [
        *_discover_skills(builtin_root, "builtin"),
        *_discover_skills(custom_root, "custom"),
    ]
    for spec in candidates:
        key = spec.id.casefold()
        if key in seen:
            LOGGER.warning("Ignoring duplicate Skill id %r from %s", spec.id, spec.path)
            continue
        seen.add(key)
        registry.append(spec)
    return tuple(registry)


SKILL_REGISTRY = discover_skill_registry()
SKILL_NAMES = tuple(spec.id for spec in SKILL_REGISTRY)
SKILL_DESCRIPTIONS = {spec.id: spec.description for spec in SKILL_REGISTRY}
SKILL_BY_ID = {spec.id.casefold(): spec for spec in SKILL_REGISTRY}
DEFAULT_SKILL_ID = (
    "h3-prompt-writing"
    if "h3-prompt-writing" in SKILL_NAMES
    else (SKILL_NAMES[0] if SKILL_NAMES else "")
)

BASE_FIELDS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
REF_FIELDS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
H3_GUIDE_ROOT = SKILL_ROOT / "h3-prompt-writing" / "references"
TIMESTAMP_RE = re.compile(r"(?i)\bAt\s+(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\b")
# Field names are compared after removing Markdown emphasis and normalizing the
# full-width colon, so `**integrated_multimodal_description:**` or
# `overall_soundscape：` still count as the required H3 fields.
FIELD_NOISE_RE = re.compile(r"[*`]")
I2VA_PREFIX = "For the target video, at 0.00 seconds into the target video,"
ALIGNMENT_PREFIX = "How the reference pictures align with the target video"
INTERACTIVE_OUTPUT_PATTERNS = (
    re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:required\s+)?"
        r"(?:clarifications?|questions?|next steps?|start gate|approval|confirmation)\b"
    ),
    re.compile(r"(?i)\bplease\s+(?:confirm|reply|provide|upload|choose|answer|let\s+me\s+know)\b"),
    re.compile(r"(?i)\b(?:do|would|could|can)\s+you\b"),
    re.compile(r"(?i)\bshould\s+(?:the|we|i)\b[^.\n]*\?"),
    re.compile(r"(?i)\b(?:confirm|reply)\s+[\"']?(?:confirm|go\s+ahead|continue)\b"),
    re.compile(r"(?i)\bto\s+proceed\s+with\s+(?:generation|the\s+next\s+step)\b"),
    re.compile(r"(?i)\bawait(?:ing)?\s+(?:your\s+)?(?:confirmation|approval|reply)\b"),
    re.compile(r"请(?:确认|回复|提供|上传|选择|回答|告知)"),
    re.compile(r"(?:你|您)(?:是否|希望|想要)"),
    re.compile(r"等待(?:你|您|用户)?(?:确认|回复)"),
    re.compile(r"确认后(?:再|即可|将)"),
)
SILENCE_REQUEST_PATTERNS = (
    re.compile(
        r"(?i)\b(?:completely silent|complete silence|silent video|muted video|"
        r"no audio|no sound at all|without any (?:audio|sound))\b"
    ),
    re.compile(r"(?:完全静音|全程静音|静音视频|视频静音|无声视频|不要任何声音|不需要任何声音)"),
)


def detect_h3_mode(image_count: int, video_count: int) -> str | None:
    """Resolve modes that do not require understanding the user's image intent."""
    if not 0 <= image_count <= 9:
        raise ValueError("Reference image count must be between 0 and 9")
    if not 0 <= video_count <= 3:
        raise ValueError("Reference video count must be between 0 and 3")
    if video_count or image_count > 2:
        return "ref2va"
    if image_count == 0:
        return "t2va"
    return None


def mode_router_prompt(user_prompt: str, image_count: int) -> list[dict[str, str]]:
    if image_count not in {1, 2}:
        raise ValueError("Automatic image mode routing requires one or two images")
    if image_count == 1:
        choices = "i2va, l2va, ref2va"
        rules = (
            "Choose i2va only when the request clearly uses Picture 1 as the opening or first frame. "
            "Choose l2va only when it clearly uses Picture 1 as the ending or last frame. "
            "Otherwise choose ref2va, including character, product, scene, style, composition, or other general references."
        )
    else:
        choices = "fl2va, ref2va"
        rules = (
            "Choose fl2va only when the request clearly assigns Picture 1 as the opening/first frame and "
            "Picture 2 as the ending/last frame, or explicitly asks to transition/interpolate from the first "
            "image to the second. Otherwise choose ref2va, including when both pictures are general references."
        )
    system = (
        "Classify the MiniMax H3 image input mode from the user's stated intent. "
        f"There are exactly {image_count} connected pictures. Return exactly one lowercase id from: {choices}. "
        "Return no punctuation or explanation. Do not assume that a picture is a keyframe merely because only "
        f"one or two pictures are connected. {rules}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


def parse_mode_selection(text: str, image_count: int) -> str:
    allowed = {1: ("i2va", "l2va", "ref2va"), 2: ("fl2va", "ref2va")}
    choices = allowed.get(image_count)
    if choices is None:
        raise ValueError("Automatic image mode routing requires one or two images")
    normalized = text.strip().lower()
    if normalized in choices:
        return normalized
    matches = [mode for mode in choices if re.search(rf"\b{mode}\b", normalized)]
    return matches[0] if len(matches) == 1 else "ref2va"


def explicitly_requests_silence(user_prompt: str) -> bool:
    return any(pattern.search(user_prompt) for pattern in SILENCE_REQUEST_PATTERNS)


def router_prompt(
    user_prompt: str,
    mode: str,
    asset_summary: str,
    skills: tuple[SkillSpec, ...] | None = None,
) -> list[dict[str, str]]:
    available = skills if skills is not None else SKILL_REGISTRY
    catalog = "\n".join(
        f"- {spec.id}: {spec.description}"
        + (
            ""
            if spec.h3_output
            else " [non-H3 output: returns its own format instead of an H3 prompt]"
        )
        for spec in available
    )
    system = (
        "Select exactly one available H3 skill for the request. Return only its exact skill id, "
        "with no punctuation or explanation. Prefer h3-prompt-writing for direct prompt rewriting "
        "when no specialized production workflow clearly applies. When the request is about a video "
        "or asks for a prompt, choose a Skill that returns an H3 prompt; choose a non-H3 Skill only "
        "when the request is clearly about that Skill's own modality (translation, image prompt, "
        "music, and so on).\n\n" + catalog
    )
    user = f"Detected H3 mode: {mode}\nConnected assets: {asset_summary}\nRequest:\n{user_prompt}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_skill_selection(
    text: str,
    skills: tuple[SkillSpec, ...] | None = None,
) -> str:
    available = skills if skills is not None else SKILL_REGISTRY
    by_id = {spec.id.casefold(): spec.id for spec in available}
    normalized = text.strip().lower()
    if normalized in by_id:
        return by_id[normalized]
    matches = [
        spec.id
        for spec in available
        if re.search(
            rf"(?<![a-z0-9_.-]){re.escape(spec.id.casefold())}(?![a-z0-9_.-])",
            normalized,
        )
    ]
    if not matches:
        matches = [
            spec.id
            for spec in available
            if re.search(
                rf"(?<![a-z0-9_-]){re.escape(spec.id.casefold())}(?![a-z0-9_-])",
                normalized,
            )
        ]
    if len(matches) == 1:
        return matches[0]
    fallback = next((spec.id for spec in available if spec.id == DEFAULT_SKILL_ID), None)
    return fallback or (available[0].id if available else "")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _skill_spec(skill: str) -> SkillSpec:
    spec = SKILL_BY_ID.get(skill.casefold())
    if spec is None:
        raise ValueError(f"Unknown skill: {skill}")
    return spec


def _skill_path(skill: str) -> Path:
    spec = _skill_spec(skill)
    root = spec.path.resolve()
    allowed_roots = (SKILL_ROOT.resolve(), CUSTOM_SKILL_ROOT.resolve())
    if not any(root.parent == allowed_root for allowed_root in allowed_roots):
        raise FileNotFoundError(f"Skill path is outside the configured Skill roots: {skill}")
    skill_file = root / "SKILL.md"
    if (
        not root.is_dir()
        or not skill_file.is_file()
        or skill_file.resolve().parent != root
    ):
        raise FileNotFoundError(f"Skill is missing SKILL.md: {skill}")
    return root


def _h3_guide_name(mode: str) -> str:
    return "ref-en.txt" if mode == "ref2va" else "base-en.txt"


def _h3_guide(mode: str) -> str:
    guide = _h3_guide_name(mode)
    path = H3_GUIDE_ROOT / guide
    return _read(path) if path.is_file() else ""


def skill_instructions(
    skill: str,
    mode: str,
    max_chars: int = 72000,
    h3_output: bool | None = None,
) -> str:
    spec = _skill_spec(skill)
    skill = spec.id
    root = _skill_path(skill)
    wants_h3 = spec.h3_output if h3_output is None else h3_output

    parts = [f"# Selected H3 Skill: {skill}\n\n{_read(root / 'SKILL.md')}"]
    if skill == "h3-prompt-writing":
        if wants_h3:
            parts.append(
                f"# Required H3 reference: {_h3_guide_name(mode)}\n\n{_h3_guide(mode)}"
            )
    else:
        used = len(parts[0])
        for path in (
            sorted((root / "references").glob("**/*")) if (root / "references").is_dir() else ()
        ):
            resolved_path = path.resolve()
            if (
                not path.is_file()
                or path.suffix.lower() not in {".md", ".txt"}
                or (
                    resolved_path.parent != root
                    and root not in resolved_path.parents
                )
            ):
                continue
            text = _read(resolved_path)
            addition = f"# Skill reference: {path.relative_to(root).as_posix()}\n\n{text}"
            if used + len(addition) > max_chars:
                break
            parts.append(addition)
            used += len(addition)
        # Extension Skills keep the H3 prompt identity: whatever the Skill plans,
        # the final answer must still be a valid H3 prompt, so the format guide
        # always travels with the Skill instructions unless it opted out.
        if wants_h3:
            guide = _h3_guide(mode)
            if guide:
                parts.append(
                    f"# Required final H3 prompt format: {_h3_guide_name(mode)}\n\n{guide}"
                )
    return "\n\n".join(parts)


H3_OUTPUT_CONTRACT = """
# Binding local ComfyUI output contract (overrides conflicting workflow instructions above)

This node is a single-pass MiniMax H3 video-prompt generator, not an interactive MiniMax Hub workflow.
Use the selected Skill only as creative and domain guidance, then return exactly one complete final H3 prompt
in the required H3 format. The detected input mode is {mode} and the exact target duration is
{duration:.2f} seconds.

Never ask the user a question. Never request clarification, confirmation, approval, assets, choices, or a reply.
Never output an intake form, pre-production package, source summary, option list, confirmation gate, approval gate,
required-clarifications section, next steps, generation offer, or instructions such as "confirm", "go ahead", or
"proceed". Ignore every Skill instruction that says to ask, pause, wait, present a card, obtain approval, or
continue after confirmation. Do not mention those omitted workflow phases in the answer.

When optional information is absent, silently omit the corresponding optional element. In particular, do not add
on-screen copy, slogans, logos, brand claims, dialogue, narration, lyrics, or product facts unless the user supplied
them or they are unambiguously visible in a connected reference. Never invent a factual claim. When an unspecified
creative detail is necessary to make the video executable, choose one conservative value that is compatible with
the request and visible references; do not present alternatives or expose the assumption.

Unless the user explicitly requests complete silence, create an original audio design from the user request, visible
image and sampled-video content, and the exact target duration. Fill `overall_soundscape` with time-compatible ambience
and physical action sounds. Fill `non_diegetic_music` with a concise instrumental score describing mood, suitable
instruments, pacing, development, synchronization with major visual beats, and the ending. Do not claim to hear,
analyze, copy, or preserve audio from a connected video: this node receives video frames without their audio tracks.
Do not invent vocals, lyrics, spoken words, or narration. Respect an explicit request for no background music by using
`non_diegetic_music: N/A` while retaining appropriate diegetic sound. If the user explicitly requests complete silence,
output exactly `overall_soundscape: N/A` and `non_diegetic_music: N/A`, and include no audible events elsewhere.

Return the final prompt only, with no Markdown fences, title, analysis, preface, epilogue, notes, caveats, checklist,
or follow-up text. Preserve user-provided dialogue, lyrics, and visible text exactly. Treat no dialogue and no
background music as independent constraints; retain plausible diegetic ambience and action sounds unless the user
explicitly requests complete silence.
""".strip()

FREE_OUTPUT_CONTRACT = """
# Binding local ComfyUI output contract

This node performs one local inference pass using the selected Skill. The selected Skill declared itself
non-H3 (`h3-output: false`), so H3 video fields are not required for this skill.

Follow the selected Skill's output instructions and the user's request, and return exactly the content the
Skill asks for. Do not add a wrapper, H3 field names, or a formatting requirement that the selected Skill
does not request.
""".strip()


def system_prompt(
    skill: str,
    mode: str,
    duration: float,
    h3_output: bool | None = None,
) -> str:
    spec = _skill_spec(skill)
    wants_h3 = spec.h3_output if h3_output is None else h3_output
    instructions = skill_instructions(spec.id, mode, h3_output=wants_h3)
    contract = (
        H3_OUTPUT_CONTRACT.format(mode=mode.upper(), duration=duration)
        if wants_h3
        else FREE_OUTPUT_CONTRACT
    )
    return instructions + "\n\n" + contract


H3_OUTPUT_MODES = ("auto", "on", "off")


def resolve_h3_output(skill: str, override: str | None = "auto") -> bool:
    """Resolve the effective H3 contract: node override wins, `auto` follows the Skill."""
    mode = str(override or "auto").strip().casefold()
    if mode not in H3_OUTPUT_MODES:
        mode = "auto"
    if mode == "on":
        return True
    if mode == "off":
        return False
    spec = SKILL_BY_ID.get(str(skill).casefold())
    return spec.h3_output if spec is not None else True


def h3_output_enabled(skill: str) -> bool:
    """Whether the selected Skill must return an H3 prompt (the node default)."""
    spec = SKILL_BY_ID.get(str(skill).casefold())
    return spec.h3_output if spec is not None else True


def _structural_text(text: str) -> str:
    """Strip decoration that does not change H3 structure detection."""
    return FIELD_NOISE_RE.sub("", text).replace("\uFF1A", ":").replace("\u00A0", " ")


def _field_match(structural: str, field: str) -> re.Match[str] | None:
    name = field.rstrip(":").strip()
    return re.search(
        rf"(?im)^\s*(?:\#{{1,6}}\s*)?{re.escape(name)}\s*:",
        structural,
    )


def output_issues(
    text: str,
    mode: str,
    duration: float,
    require_complete_silence: bool = False,
    validate_h3: bool = True,
) -> list[str]:
    """Check the output against the H3 prompt contract (default) or just for emptiness."""
    stripped = text.strip()
    if not stripped:
        return ["The output is empty"]
    if not validate_h3:
        return []

    structural = _structural_text(stripped)
    issues: list[str] = []
    if "```" in stripped:
        issues.append("Remove Markdown code fences")

    interaction_scope = re.sub(r"<d>.*?</d>", "", structural, flags=re.DOTALL)
    if any(pattern.search(interaction_scope) for pattern in INTERACTIVE_OUTPUT_PATTERNS):
        issues.append(
            "Remove all questions, clarifications, confirmations, approval gates, and next-step requests"
        )

    fields = REF_FIELDS if mode == "ref2va" else BASE_FIELDS
    matches = [_field_match(structural, field) for field in fields]
    missing = [field for field, match in zip(fields, matches, strict=True) if match is None]
    if missing:
        issues.append(f"Missing required fields: {', '.join(missing)}")
    else:
        positions = [match.start() for match in matches if match is not None]
        if positions != sorted(positions):
            issues.append("Required fields are not in the official order")
        first = fields[0].rstrip(":").strip()
        if mode in {"t2va", "ref2va"}:
            starts_with_field = re.match(
                rf"(?im)^\s*(?:\#{{1,6}}\s*)?{re.escape(first)}\s*:",
                structural,
            )
            if not starts_with_field:
                issues.append(f"The prompt must begin directly with {first}:")
        duplicated = len(re.findall(rf"(?im)^\s*(?:\#{{1,6}}\s*)?{re.escape(first)}\s*:", structural))
        if duplicated > 1:
            issues.append("Return exactly one prompt; remove the duplicated prompt block")

    if mode == "i2va" and not re.match(rf"(?im)^\s*{re.escape(I2VA_PREFIX)}", structural):
        issues.append("I2VA must begin with the official first-frame instruction")
    if mode in {"fl2va", "l2va"} and not re.match(
        rf"(?im)^\s*{re.escape(ALIGNMENT_PREFIX)}", structural
    ):
        issues.append(f"{mode.upper()} must begin with the official alignment instruction")

    if require_complete_silence:
        soundscape_is_silent = re.search(r"(?im)^\s*overall_soundscape\s*:\s*N/A\s*$", structural)
        music_is_silent = re.search(r"(?im)^\s*non_diegetic_music\s*:\s*N/A\s*$", structural)
        if not soundscape_is_silent or not music_is_silent:
            issues.append(
                "Complete silence was requested: set both overall_soundscape and non_diegetic_music exactly to N/A"
            )

    previous = -1.0
    for match in TIMESTAMP_RE.finditer(structural):
        millis = int((match.group(3) or "0").ljust(3, "0"))
        timestamp = int(match.group(1)) * 60 + int(match.group(2)) + millis / 1000
        if timestamp <= previous:
            issues.append("Shot timestamps must be strictly increasing")
            break
        if timestamp >= duration:
            issues.append(f"Shot timestamp {timestamp:.3f}s must be earlier than {duration:.2f}s")
            break
        previous = timestamp
    return issues
