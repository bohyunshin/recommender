---
name: manage-skills
description: Analyzes session changes to detect missing verification skills. Dynamically explores existing skills, creates new skills or updates existing ones, and manages CLAUDE.md.
disable-model-invocation: true
argument-hint: "[optional: specific skill name or area to focus on]"
---

# Session-Based Skill Maintenance

## Purpose

Analyzes changes made in the current session to detect and fix verification skill drift:

1. **Missing coverage** — Changed files not referenced by any verify skill
2. **Invalid references** — Skills referencing deleted or moved files
3. **Missing checks** — New patterns/rules not covered by existing checks
4. **Stale values** — Configuration values or detection commands that no longer match

## When to Run

- After implementing a feature that introduces new patterns or rules
- When modifying existing verify skills and wanting to check consistency
- Before a PR to ensure verify skills cover changed areas
- When a verification run missed expected issues
- Periodically to align skills with codebase changes

## Registered Verification Skills

List of verification skills registered in the current project. Update this list when creating/deleting skills.

| Skill | Description | Covered File Patterns |
|-------|-------------|----------------------|
| `verify-test-coverage` | Verifies new models/pipelines have corresponding unit and integration tests | `tests/**/*.py`, `recommender/libs/constant/model/name.py` |
| `verify-code-convention` | Validates PEP 8 naming, type hints, ruff compliance, and import ordering | `recommender/**/*.py`, `tests/**/*.py` |
| `verify-model-registration` | Ensures new models are registered in constants, have module paths, train.py routing, and implement abstract methods | `recommender/libs/constant/model/*.py`, `recommender/train.py`, `recommender/train_csr.py`, `recommender/model/**/*.py` |
| `verify-code-simplifier` | Checks if code could be simplified — redundant booleans, manual loops, verbose patterns, dict lookup replacements | `recommender/**/*.py`, `tests/**/*.py`, `scripts/**/*.py` |

## Workflow

### Step 1: Analyze Session Changes

Collect all files changed in the current session:

```bash
# Uncommitted changes
git diff HEAD --name-only

# Commits on the current branch (if branched from main)
git log --oneline main..HEAD 2>/dev/null

# All changes since branching from main
git diff main...HEAD --name-only 2>/dev/null
```

Merge into a deduplicated list. If an optional argument specifies a skill name or area, filter to relevant files only.

**Display:** Group files by top-level directory (first 1-2 path segments):

```markdown
## Session Changes Detected

**N files changed in this session:**

| Directory | Files |
|-----------|-------|
| recommender/model | `svd.py`, `gmf.py` |
| recommender/libs | `name.py`, `module_path.py` |
| tests | `test_train.py` |
| (root) | `pyproject.toml` |
```

### Step 2: Map Changed Files to Registered Skills

Reference the skills listed in the **Registered Verification Skills** section above to build a file-to-skill mapping.

#### Sub-step 2a: Check Registered Skills

Read each skill's name and covered file patterns from the **Registered Verification Skills** table.

If there are 0 registered skills, skip directly to Step 4 (CREATE vs UPDATE decision). All changed files are treated as "UNCOVERED".

If there is 1 or more registered skills, read each skill's `.claude/skills/verify-<name>/SKILL.md` and extract additional file path patterns from:

1. **Related Files** section — Parse the table to extract file paths and glob patterns
2. **Workflow** section — Extract file paths from grep/glob/read commands

#### Sub-step 2b: Match Changed Files to Skills

For each changed file collected in Step 1, compare against registered skill patterns. A file matches a skill if:

- It matches the skill's covered file patterns
- It is located within a directory referenced by the skill
- It matches a regex/string pattern used in the skill's detection commands

#### Sub-step 2c: Display Mapping

```markdown
### File → Skill Mapping

| Skill | Trigger Files (Changed Files) | Action |
|-------|-------------------------------|--------|
| verify-model-registration | `name.py`, `module_path.py` | CHECK |
| verify-test-coverage | `test_train.py` | CHECK |
| (no skill) | `pyproject.toml` | UNCOVERED |
```

### Step 3: Coverage Gap Analysis for Affected Skills

For each AFFECTED skill (skills with matching changed files), read the full SKILL.md and check:

1. **Missing file references** — Are changed files related to this skill's domain not listed in the Related Files section?
2. **Stale detection commands** — Do the skill's grep/glob patterns still match the current file structure? Run sample commands to test.
3. **Uncovered new patterns** — Read the changed files and identify new rules, configurations, or patterns that the skill doesn't check. Look for:
   - New type definitions, enum variants, or exported symbols
   - New registrations or configurations
   - New file naming or directory conventions
4. **Residual references to deleted files** — Are there files listed in the skill's Related Files that no longer exist in the codebase?
5. **Changed values** — Have specific values (identifiers, config keys, type names) checked by the skill been modified in the changed files?

Record each gap found:

```markdown
| Skill | Gap Type | Details |
|-------|----------|---------|
| verify-model-registration | Missing file | `recommender/model/new_model.py` not in Related Files |
| verify-test-coverage | New pattern | New model uses unchecked test patterns |
```

### Step 4: CREATE vs UPDATE Decision

Apply the following decision tree:

```
For each uncovered file group:
    IF the file is related to an existing skill's domain:
        → Decision: UPDATE existing skill (expand coverage)
    ELSE IF 3+ related files share common rules/patterns:
        → Decision: CREATE new verify skill
    ELSE:
        → Mark as "exempt" (no skill needed)
```

Present results to the user:

```markdown
### Proposed Actions

**Decision: UPDATE existing skill** (N)
- `verify-model-registration` — Add 2 missing file references, update detection patterns
- `verify-test-coverage` — Update detection commands for new patterns

**Decision: CREATE new skill** (M)
- New skill needed — Covers <pattern description> (X uncovered files)

**No action needed:**
- `pyproject.toml` — Config file, exempt
- `README.md` — Documentation, exempt
```

Use `AskUserQuestion` to confirm:
- Which existing skills to update
- Whether to create proposed new skills
- Option to skip entirely

### Step 5: Update Existing Skills

For each skill the user approved for update, read the current SKILL.md and apply targeted edits:

**Rules:**
- **Add/modify only** — Never remove existing checks that still work
- Add new file paths to the **Related Files** table
- Add new detection commands for patterns found in changed files
- Add new workflow steps or sub-steps for uncovered rules
- Remove references to files confirmed deleted from the codebase
- Update specific values (identifiers, config keys, type names) that have changed

### Step 6: Create New Skills

**Important:** When creating a new skill, you must confirm the skill name with the user.

For each new skill to create:

1. **Explore** — Read the related changed files to deeply understand the patterns

2. **Confirm skill name with the user** — Use `AskUserQuestion`:

   **Naming rules:**
   - The name must start with `verify-` (e.g., `verify-auth`, `verify-api`)
   - Use kebab-case (e.g., `verify-error-handling`, not `verify_error_handling`)

3. **Create** — Create `.claude/skills/verify-<name>/SKILL.md` following this template:

```yaml
---
name: verify-<name>
description: <one-line description>. Use after <trigger condition>.
---
```

Required sections:
- **Purpose** — 2-5 numbered verification categories
- **When to Run** — 3-5 trigger conditions
- **Related Files** — Table of actual file paths in the codebase (verified with `ls`, no placeholders)
- **Workflow** — Check steps, each specifying:
  - Tool to use (Grep, Glob, Read, Bash)
  - Exact file paths or patterns
  - PASS/FAIL criteria
  - How to fix on failure
- **Output Format** — Markdown table for results
- **Exceptions** — At least 2-3 realistic "not a violation" cases

4. **Update related skill files** — After creating a new skill, you must update these 3 files:

   **4a. Update this file itself (`manage-skills/SKILL.md`):**
   - Add a new skill row to the table in the **Registered Verification Skills** section

   **4b. Update `verify-implementation/SKILL.md`:**
   - Add a new skill row to the table in the **Skills to Execute** section

   **4c. Update `CLAUDE.md`:**
   - Add a new skill row to the `## Skills` table

### Step 7: Verification

After all edits:

1. Re-read all modified SKILL.md files
2. Verify markdown formatting is correct (unclosed code blocks, consistent table columns)
3. Verify no broken file references — For each path in Related Files, check file existence:

```bash
ls <file-path> 2>/dev/null || echo "MISSING: <file-path>"
```

4. Dry-run one detection command from each updated skill to validate syntax
5. Verify the **Registered Verification Skills** table and **Skills to Execute** table are in sync

### Step 8: Summary Report

Display the final report:

```markdown
## Session Skill Maintenance Report

### Files Analyzed: N

### Skills Updated: X
- `verify-<name>`: N new checks added, Related Files updated

### Skills Created: Y
- `verify-<name>`: Covers <pattern>

### Related Files Updated:
- `manage-skills/SKILL.md`: Registered Verification Skills table updated
- `verify-implementation/SKILL.md`: Skills to Execute table updated
- `CLAUDE.md`: Skills table updated

### Unaffected Skills: Z
- (No related changes)

### Uncovered Changes (no applicable skill):
- `path/to/file` — Exempt (reason)
```

---

## Quality Criteria for Created/Updated Skills

All created or updated skills must have:

- **Actual file paths from the codebase** (verified with `ls`), not placeholders
- **Working detection commands** — Using real grep/glob patterns that match current files
- **PASS/FAIL criteria** — Clear conditions for pass and fail for each check
- **At least 2-3 realistic exceptions** — Descriptions of what is not a violation
- **Consistent formatting** — Same as existing skills (frontmatter, section headers, table structure)

---

## Related Files

| File | Purpose |
|------|---------|
| `.claude/skills/verify-implementation/SKILL.md` | Integrated verification skill (this skill manages the execution target list) |
| `.claude/skills/manage-skills/SKILL.md` | This file itself (manages the registered verification skills list) |
| `CLAUDE.md` | Project guidelines (this skill manages the Skills section) |

## Exceptions

The following are **not issues**:

1. **Lock files and generated files** — `uv.lock`, `poetry.lock`, auto-generated migration files, and build outputs do not need skill coverage
2. **One-off config changes** — Version bumps in `pyproject.toml`, minor linter/formatter config changes do not need new skills
3. **Documentation files** — `README.md`, `CHANGELOG.md`, `LICENSE`, etc. are not code patterns requiring verification
4. **Test fixture files** — Files in directories used as test fixtures are not production code
5. **Unaffected skills** — Skills marked as UNAFFECTED do not need review
6. **CLAUDE.md itself** — Changes to CLAUDE.md are documentation updates, not code patterns requiring verification
7. **CI/CD configuration** — `.github/`, `Dockerfile`, etc. are infrastructure, not application patterns requiring verification skills
