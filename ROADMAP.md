# Hermes HUD — Roadmap

A record of everything built, day by day.

---

## Foundation (Pre-Day 1)

The initial release established the core architecture: a FastAPI backend reading `~/.hermes/` and a React + Vite + Tailwind frontend. No framework dependencies on hermes-agent internals — the HUD vendors its own collectors.

**What shipped:**
- Dashboard with identity, memory bars, service health, skills, projects, cron jobs, tool usage, daily sparkline
- 11 read-only tabs — Memory, Skills, Sessions, Cron, Projects, Health, Agents, Profiles, Costs, Corrections, Patterns
- WebSocket real-time updates — `watchfiles` watches `~/.hermes/` and broadcasts `data_changed` to all clients
- Smart mtime-based caching — avoids re-reading files that haven't changed
- Four themes — Neural Awakening, Blade Runner, fsociety, Anime
- CRT scanline overlay
- Command palette (`Ctrl+K`)
- Boot screen animation
- Keyboard shortcuts (`1`–`0` tabs, `t` theme cycle)

---

## Day 1 — Chat + Corrections + Patterns

Wired up the chat engine and surfaced two previously hidden tabs.

**What shipped:**
- Live chat with Hermes agent via `hermes chat -q <msg> -Q --source tool` subprocess
- Multiple independent sessions, each with isolated message history
- SSE streaming — responses appear in real time
- Corrections tab — corrections grouped by severity (critical / major / minor)
- Patterns tab — task clusters, hourly activity heatmap, repeated prompts
- CLI decoration filtering — box-drawing characters and system headers stripped from chat output
- HUD-generated sessions tagged `--source tool` so they stay out of the Sessions tab

---

## Day 2 — Chat Polish

Made chat output readable and interactive.

**What shipped:**
- Markdown rendering — headers, lists, tables, code blocks
- Syntax-highlighted code with hover copy button
- Stop button — cancels a response mid-stream by killing the subprocess
- Multi-line warning block filtering — context compression notices and other `⚠` warnings no longer bleed into responses
- Fixed chat session isolation — switching sessions no longer shows the wrong history

---

## Day 3 — Session Transcript Viewer + Search

Made the Sessions tab useful beyond a list.

**What shipped:**
- Session transcript modal — click any session to read the full conversation with markdown rendering and per-message token counts
- Session search — searches titles and full message content via FTS, results show match type and a content snippet
- HUD-generated sessions filtered from search results

---

## Day 4 — Tool Calls + Reasoning in Chat

Surfaced what the agent is actually doing while it thinks.

**What shipped:**
- Tool call cards — after a response finishes, cards show each tool used (web_search, terminal, etc.) with arguments and status
- Reasoning blocks — agent thinking/extended reasoning appears as a collapsible "Thinking" section
- Tool call data read from `state.db` post-completion using the hermes session ID captured from stdout

---

## Day 5 — Memory Editing

Added the first write capability — the HUD was previously read-only.

- Inline edit and delete entries in the Memory tab (Agent Memory and User Profile)
- Add new entries via expandable form
- Two-click delete confirmation
- File locking (`fcntl.flock`) + atomic writes (`tempfile.mkstemp` → `os.replace`) matching hermes-agent's own locking pattern
- No direct JSON writes — entries use hermes's `\n§\n` delimiter format

---

## Day 6 — Chat History Persistence + Bug Fixes + Speed

### Chat History Persistence
- Messages and sessions survive page refresh via localStorage
- In-memory Map as L1 cache, localStorage as L2 — debounced writes (1s) to avoid thrashing during streaming
- On server restart: ChatPanel detects missing backend sessions, re-creates them, migrates message keys from old to new IDs automatically

### Streaming Bug Fixes
- Fixed stuck streaming — warning block filter was swallowing the first response line (hermes outputs content immediately after the warning with no blank line separator)
- Fixed stale closure in SSE `onerror` handler — `isStreaming` captured at creation time meant cleanup never ran
- Heartbeat SSE comment every 15s keeps connections alive through proxies

### Speed Optimization
- Switched token emission from char-by-char to line-level — reduces SSE events from ~500 to ~20 per response, noticeably faster rendering

---

## Day 7 — Cron Job Management

Made the Cron tab interactive — jobs can now be controlled without touching the CLI or editing files by hand.

- **Pause** — suspends a scheduled job
- **Resume** — re-activates a paused job and recomputes the next run time
- **Run Now** — triggers immediate execution on the next scheduler tick
- **Delete** — removes the job permanently (two-click confirmation)
- Buttons adapt to job state: Pause/Resume toggle, Run hidden for completed/paused jobs
- Delegated to `hermes cron pause|resume|remove|run <job_id>` CLI — avoids racing with the scheduler's own file writes (hermes's `save_jobs()` has no file locking)
- End-to-end tested via Playwright against live jobs

---

## What's Next (Ideas)

These are possibilities, not commitments.

- **Cron job creation** — a form to schedule new jobs without touching `jobs.json` by hand
- **Memory search** — filter/search across memory entries in the HUD
- **Live token counter** — show context usage in real time during streaming
- **Profile editing** — edit profile fields (soul, model, tools) from the Profiles tab
- **Export** — download session transcripts or memory snapshots as markdown
- **Mobile layout** — responsive polish for small screens
- **Dark/light mode** — fifth theme option using system preference
- **Multi-agent** — support for switching between multiple hermes profiles in the chat tab
