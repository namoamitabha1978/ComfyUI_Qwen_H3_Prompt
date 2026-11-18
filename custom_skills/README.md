# Custom H3 Skills

Place one prompt-only Skill in each subdirectory. The directory is discovered when
ComfyUI loads this custom node, so restart ComfyUI after adding or changing a Skill.

Each Skill must contain a `SKILL.md`. The optional YAML front matter may provide
`name`, `description`, `display-name`, and `version`; `meta.yaml` may provide the
same metadata plus tags. The Skill ID must contain only lowercase letters, numbers,
periods, underscores, and hyphens, and must not be `auto`.

Example:

```text
custom_skills/
  cinematic-food/
    SKILL.md
    references/
      shot-guidelines.md
    meta.yaml
```

The node reads Markdown and text references as model guidance. It does not execute
scripts, tools, network calls, Canvas workflows, or approval gates declared by a
third-party Skill.

H3 prompt output is the node default. Every Skill that does not opt out has the H3
format guide (`skills/h3-prompt-writing/references/base-en.txt` or `ref-en.txt`)
and the binding H3 output contract appended after its own instructions, and its
answer is validated against that contract (required fields and order, mode
prefix, shot timestamps, audio sections).

A Skill that is inherently non-video — image prompt rewriting, translation, music,
and so on — opts out with `h3-output: false` in its `SKILL.md` front matter, or
`h3_output: false` in `meta.yaml`. An opted-out Skill is returned in its own shape,
with no H3 fields, no H3 guide, and no H3 validation.

The node's advanced `h3_output` option (`auto` / `on` / `off`, default `auto`) overrides
this per run without editing any Skill file: `auto` follows the declaration above,
`on` forces the H3 contract, `off` forces the free-form contract.

The bundled `qwen-llm-translate`, `qwen-image-2.1-text2image`, `qwen-image-2.1-edit`,
and `yue2-music` are already declared as opted-out. Copy their front matter when you
add a similar non-video Skill.

The included `example_custom_skill/` is a minimal discovery test. Select
`example_custom_skill` in the node to confirm that the custom directory is visible
and its instructions are loaded. Its requested fixed output is
`This is a example of custom_skill.`
