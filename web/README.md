# Web Skeleton

Next.js scaffold for the new front end. Agents should implement routing, selection state, API clients, and ECharts/AG Grid views per `REBUILD_PLAN.md`.

## Dev
```sh
npm install
npm run dev
```

## Notes
- Uses React Query for data fetching/caching (see plan).
- Charts: ECharts; Tables: AG Grid.
- Point API base to the FastAPI service via env (e.g., `NEXT_PUBLIC_API_BASE=http://localhost:8000`).
