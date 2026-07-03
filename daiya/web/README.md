# daiya/web — v0 testbed frontend

Vite + React + TypeScript + Tailwind. Hand-rolled components, Phosphor icons.

```sh
npm install
npm run dev      # http://<lan-ip>:5173 — proxies /ws and /api to the backend
npm run build    # typecheck + production build into dist/ (served by FastAPI)
```

## Backend proxy

Dev proxies `/ws/*` (WebSocket) and `/api/*` to `DAIYA_SERVER`
(default `http://127.0.0.1:8000`). In production the FastAPI server serves
`dist/` itself, so everything is same-origin. To point the built app at a
different server, set `VITE_DAIYA_SERVER` at build time. All endpoint paths
live in `src/api.ts`.

## HTTPS for LAN phone testing

`getUserMedia` requires a secure context off-localhost. Drop a cert here and
the dev server picks it up automatically (no cert → plain HTTP):

```
certs/dev.crt
certs/dev.key
```

e.g. with [mkcert](https://github.com/FiloSottile/mkcert):

```sh
mkcert -install
mkcert -cert-file certs/dev.crt -key-file certs/dev.key <lan-ip> localhost
```

Or point `DAIYA_TLS_CERT` / `DAIYA_TLS_KEY` at existing files. `certs/` is
gitignored; never commit keys.
