# Web App

Next.js frontend for the portfolio dashboard. Canonical repo-wide guidance now lives at `AGENTS.md`, and the code navigation map lives at `REPO_MAP.md`.

## Run
```sh
npm install
# PowerShell:
$env:NEXT_PUBLIC_API_BASE="http://localhost:8000"
npm run dev
```

## Verify
```sh
npx tsc --noEmit
npm run build
```

## Notes
- Data fetching/caching uses React Query.
- Charts use ECharts; table views use AG Grid.
- `npm run lint` currently prompts for ESLint setup interactively (no committed ESLint config yet).
