import { useEffect, useRef, useState, type FormEvent } from "react"
import { Database, FileText, Layers3, Moon, Plus, Search, SlidersHorizontal, Sun } from "lucide-react"

import { AddDocumentDialog, CreateCollectionDialog } from "@/components/editor-dialogs"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { NativeSelect } from "@/components/ui/native-select"
import { Textarea } from "@/components/ui/textarea"
import { api, errorMessage, parseMetadata, type Collection, type Metric, type SearchResults } from "@/lib/api"

type Theme = "light" | "dark"
const themeKey = "vekryx-theme"
const collectionKey = "vekryx-collection"

function readPreference(key: string) {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function writePreference(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // Preferences still apply for the current session.
  }
}

function ThemeToggle({ theme, onChange }: { theme: Theme; onChange: (theme: Theme) => void }) {
  return (
    <div className="theme-toggle" role="group" aria-label="Color theme">
      <Button size="icon" variant="ghost" aria-label="Light theme" aria-pressed={theme === "light"} onClick={() => onChange("light")}>
        <Sun />
      </Button>
      <Button size="icon" variant="ghost" aria-label="Dark theme" aria-pressed={theme === "dark"} onClick={() => onChange("dark")}>
        <Moon />
      </Button>
    </div>
  )
}

export function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = readPreference(themeKey)
    if (saved === "light" || saved === "dark") return saved
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"
  })
  const [collections, setCollections] = useState<Collection[]>([])
  const [selected, setSelected] = useState(() => readPreference(collectionKey) ?? "")
  const [reload, setReload] = useState(0)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState("")
  const active = collections.find(({ name }) => name === selected) ?? collections[0]

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    writePreference(themeKey, theme)
  }, [theme])

  useEffect(() => {
    const controller = new AbortController()
    api.collections(controller.signal).then((items) => {
      if (!controller.signal.aborted) setCollections(items)
    }).catch((error) => {
      if (!controller.signal.aborted) setLoadError(errorMessage(error))
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [reload])

  function choose(name: string) {
    setSelected(name)
    writePreference(collectionKey, name)
  }

  function created(collection: Collection) {
    setCollections((items) => [...items.filter(({ name }) => name !== collection.name), collection]
      .sort((left, right) => left.name.localeCompare(right.name)))
    choose(collection.name)
  }

  async function refreshCollection(name: string) {
    try {
      const updated = await api.collection(name)
      setCollections((items) => items.map((item) => item.name === name ? updated : item))
    } catch {
      // A successful document write remains successful if the count refresh fails.
    }
  }

  return (
    <div className="workspace corner-frame">
      <aside className="sidebar">
        <div className="wordmark"><Database aria-hidden="true" /> Vekryx</div>
        <nav aria-label="Collections" className="collection-nav">
          <h2><Layers3 aria-hidden="true" /> Collections</h2>
          <div className="collection-list">
            {collections.map((collection) => (
              <Button key={collection.name} variant="ghost" className="collection-link"
                aria-current={active?.name === collection.name ? "true" : undefined}
                onClick={() => choose(collection.name)}>
                <Database aria-hidden="true" /><span>{collection.name}</span>
              </Button>
            ))}
          </div>
          <CreateCollectionDialog
            trigger={<Button variant="ghost" className="new-collection" disabled={loading || !!loadError}><Plus /> New collection</Button>}
            onCreated={created}
          />
        </nav>
      </aside>

      <main className="main-pane">
        <header className="workspace-header">
          <div>
            <h1>Search</h1>
            {active && (
              <p className="collection-info">
                <span>{active.model}</span><span>{active.dimensions} dimensions</span>
                <span>{active.chunk_count} {active.chunk_count === 1 ? "chunk" : "chunks"}</span>
              </p>
            )}
          </div>
          <div className="header-actions">
            <ThemeToggle theme={theme} onChange={setTheme} />
            {active && <IngestionControl key={active.name} collection={active.name} onAdded={refreshCollection} />}
          </div>
        </header>

        {loading && <p className="empty-state" role="status">Loading collections…</p>}
        {loadError && (
          <div className="empty-state">
            <p className="error" role="alert">{loadError}</p>
            <Button onClick={() => {
              setLoadError("")
              setLoading(true)
              setReload((value) => value + 1)
            }}>Retry</Button>
          </div>
        )}
        {!loading && !loadError && (active
          ? <SearchWorkspace key={active.name} collection={active} />
          : <p className="empty-state">Create a collection to add and search documents.</p>)}
      </main>
    </div>
  )
}

function IngestionControl({ collection, onAdded }: {
  collection: string
  onAdded: (name: string) => Promise<void>
}) {
  const [notice, setNotice] = useState("")

  return (
    <div className="ingestion-control">
      <AddDocumentDialog
        collection={collection}
        trigger={<Button><Plus /> Add document</Button>}
        onAdded={(count) => {
          setNotice(`Document saved · ${count} ${count === 1 ? "chunk" : "chunks"}`)
          void onAdded(collection)
        }}
      />
      <span className="ingestion-notice" role="status">{notice}</span>
    </div>
  )
}

function SearchWorkspace({ collection }: { collection: Collection }) {
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [mode, setMode] = useState<"hybrid" | "mmr">("hybrid")
  const [lambda, setLambda] = useState(0.5)
  const [results, setResults] = useState<SearchResults | null>(null)
  const hits = results?.hits
  const [pending, setPending] = useState(false)
  const [error, setError] = useState("")
  const request = useRef<AbortController | null>(null)

  useEffect(() => () => request.current?.abort(), [])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (pending) return
    const data = new FormData(event.currentTarget)
    const query = String(data.get("query") ?? "").trim()
    setError("")
    try {
      if (!query) throw new Error("Enter a search query.")
      const k = Number(data.get("k"))
      if (!Number.isInteger(k) || k < 1 || k > 100) throw new Error("Choose between 1 and 100 results.")
      const filter = parseMetadata(String(data.get("filter") ?? ""))
      const metric = data.get("metric") as Metric
      const candidateInput = data.get("candidates")
      const candidates = candidateInput ? Number(candidateInput) : undefined
      if (mode === "mmr" && candidates !== undefined && (!Number.isInteger(candidates) || candidates < k || candidates > 1000)) {
        throw new Error("The MMR candidate pool must be at least as large as the result limit and at most 1000.")
      }
      const mmr = mode === "mmr" ? { candidates, lambda } : undefined
      const controller = new AbortController()
      request.current = controller
      setResults(null)
      setPending(true)
      setResults(await api.search(collection.name, { query, k, filter, metric, mmr }, controller.signal))
    } catch (error) {
      if (!request.current?.signal.aborted) setError(errorMessage(error))
    } finally {
      request.current = null
      setPending(false)
    }
  }

  return (
    <>
      <form className="search-form corner-frame" onSubmit={submit} role="search" aria-label="Search collection">
        <div className="query-row">
          <div className="query-input">
            <Search aria-hidden="true" />
            <Input aria-label="Search query" name="query" placeholder="Ask a question about your documents…" required maxLength={8000} />
          </div>
          <Button type="submit" variant="primary" disabled={pending}>{pending ? "Searching…" : "Search"}</Button>
          <Button aria-expanded={filtersOpen} aria-controls="search-filters" onClick={() => setFiltersOpen(!filtersOpen)}>
            <SlidersHorizontal /> Filters
          </Button>
        </div>
        <div className="search-options">
          <label className="field">
            Retrieval
            <NativeSelect name="mode" value={mode} onChange={(event) => setMode(event.target.value === "mmr" ? "mmr" : "hybrid")}>
              <option value="hybrid">Hybrid</option>
              <option value="mmr">MMR · Diverse results</option>
            </NativeSelect>
          </label>
          <label className="field">
            Distance metric
            <NativeSelect name="metric" defaultValue={collection.metric ?? "cosine"}>
              <option value="cosine">Cosine</option>
              <option value="euclidean">Euclidean</option>
            </NativeSelect>
          </label>
          {mode === "mmr" && (
            <>
              <label className="field">
                <span>Relevance weight <output>{lambda.toFixed(2)}</output></span>
                <input type="range" name="lambda" min={0} max={1} step={0.05} value={lambda}
                  aria-label="Relevance weight" aria-describedby="lambda-help"
                  onChange={(event) => setLambda(event.target.valueAsNumber)} />
                <span className="range-help muted" id="lambda-help"><span>More diverse</span><span>More relevant</span></span>
              </label>
              <label className="field">
                Candidate pool
                <Input name="candidates" type="number" min={1} max={1000} placeholder="Automatic" />
              </label>
            </>
          )}
        </div>
        <div id="search-filters" className="filter-fields" hidden={!filtersOpen}>
          <label className="field">
            <span>Metadata filter <span className="muted">(JSON, exact matches)</span></span>
            <Textarea name="filter" placeholder={'{"project":"vekryx"}'} rows={2} spellCheck={false} className="json-input" />
          </label>
          <label className="field">
            Max results
            <Input name="k" type="number" min={1} max={100} defaultValue={10} />
          </label>
        </div>
      </form>
      {error && <p className="search-error error" role="alert">{error}</p>}

      <section className="results" aria-label="Search results" aria-busy={pending}>
        {pending && <p className="empty-state" role="status">Searching for matching chunks…</p>}
        {!pending && !error && results === null && (
          <p className="empty-state">{collection.chunk_count === 0
            ? "Add a document to start searching this collection."
            : "Enter a question or keywords to find relevant chunks."}</p>
        )}
        {results && hits && (
          <>
            <div className="results-header">
              <h2 role="status">{hits.length} {hits.length === 1 ? "result" : "results"}</h2>
              {hits.length > 0 && <span>{results.mode === "mmr" ? "Query distance" : "Score"}</span>}
            </div>
            {hits.length === 0 && <p className="empty-state">No matching chunks found.</p>}
            <ol className="result-list">
              {hits.map((hit) => {
                const distance = "distance" in hit
                const value = distance ? hit.distance : hit.score
                const label = distance ? "Query distance" : "Score"
                return (
                  <li key={hit.chunk_id} className="result">
                    <div className="result-heading">
                      <h3><FileText aria-hidden="true" />{hit.document_id}</h3>
                      <data value={value} aria-label={`${label} ${value.toFixed(4)}`}>{value.toFixed(4)}</data>
                    </div>
                    <p>{hit.text}</p>
                    <details>
                      <summary>Chunk details</summary>
                      <code>{hit.chunk_id}</code>
                      {Object.keys(hit.metadata).length > 0 && <pre>{JSON.stringify(hit.metadata, null, 2)}</pre>}
                    </details>
                  </li>
                )
              })}
            </ol>
          </>
        )}
      </section>
    </>
  )
}
