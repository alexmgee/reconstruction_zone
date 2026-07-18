# Frontend single-file proof

This dispatch proves that the pinned Vite 8/Rolldown toolchain collapses a real
`React.lazy(() => import(...))` boundary into one self-contained, unminified
`dist/index.html`.

Use Node and npm versions pinned by `.node-version` and `package.json`, then run:

```shell
npm ci --ignore-scripts
npm run build
```

The build is accepted only when `scripts/verify-single-file.mjs` passes.
