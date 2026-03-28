# TODOS

## P1 — High Priority

### Price Alert Notifications
When a screened stock hits the planned entry zone, notify via email or Telegram. Background monitoring with configurable thresholds.

**Why:** #1 dream feature. Don't have to watch screens all day. Enforces discipline (alerts based on YOUR plan, not impulse).
**Effort:** L (human) -> M with CC (~1 hour)
**Depends on:** SQLite data layer (entry zones saved in trades table)
**Added:** 2026-03-28 via /plan-ceo-review

## P2 — Medium Priority

### AI Trade Thesis Generation
Use Claude API to auto-generate a 2-3 sentence thesis when analyzing a trade: why the setup makes sense, what could invalidate it, what catalyst to watch.

**Why:** Removes friction of manual thesis writing while maintaining journaling discipline. Compare AI thesis vs your own reasoning over time.
**Effort:** M (human) -> S with CC (~20 min)
**Depends on:** Thesis text field (shipping in current plan). Write 30+ manual theses first.
**Added:** 2026-03-28 via /plan-ceo-review

### Schema Migration Framework
Add a schema_version table and a simple migration runner for automatic column additions.

**Why:** The roadmap implies 3-4 more features needing new columns. Without migration tooling, hand-writing ALTER TABLE against production DB with real trade data.
**Effort:** S (human) -> S with CC (~10 min)
**Depends on:** SQLite migration (shipping in current plan)
**Added:** 2026-03-28 via /plan-ceo-review

## P3 — Low Priority

### Extract app.py into Tab Modules
Split app.py (currently 1265 lines, growing to ~1700) into tabs/value.py, tabs/growth.py, tabs/trade.py, tabs/my_edge.py, etc.

**Why:** Single-file monolith is harder to navigate, debug, and test. Each tab should be independently readable.
**Effort:** M (human) -> S with CC (~20 min)
**Depends on:** None. Do after features ship.
**Added:** 2026-03-28 via /plan-ceo-review

### Create DESIGN.md via /design-consultation
Run /design-consultation to generate a formal design system (colors, typography, spacing, component patterns).

**Why:** The app has an implicit design system (dark theme, green/yellow/red, Plotly charts) but it's not codified. Consistency will drift as the app grows.
**Effort:** S (human) -> S with CC (~15 min)
**Depends on:** Nothing
**Added:** 2026-03-28 via /plan-design-review
