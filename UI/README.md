# Vekryx UI

The local frontend uses React, TypeScript, Vite, and shadcn/ui components. It
connects to the Actix server for collections, document ingestion, and ranked
text search.

Start the [Actix server](../server/README.md) and Ollama, then run these commands
from the repository root.

```sh
npm --prefix UI ci
npm --prefix UI run dev
```

Open the URL printed by Vite. The interface defaults to your system theme and
remembers your chosen theme and collection on this browser.

- Create a collection with the name of an embedding model installed in Ollama.
- Add a document by entering its ID and pasting its text. Posting the same ID
  replaces all of its chunks.
- Search with plain text and pick cosine or Euclidean for each query. Hybrid
  results show fusion scores. MMR lets you tune the relevance–diversity
  balance and candidate pool. Its results show query distance in MMR
  selection order.
- Expand **Filters** to set the result limit or supply exact-match metadata
  such as `{"project":"vekryx"}`.

## Build and serve

```sh
npm --prefix UI run build
npm --prefix UI run preview
```

Both local commands proxy `/v1` to `http://127.0.0.1:8080`. To override the target,
set `VEKRYX_API_URL` in the shell or in `UI/.env.local`. When serving `UI/dist`
through another web server, route `/v1` to Actix on the same origin.
