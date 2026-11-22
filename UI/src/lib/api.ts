export type Metadata = Record<string, unknown>
export type Metric = "cosine" | "euclidean"

export interface Collection {
  name: string
  model: string
  metric: Metric
  dimensions: number
  chunk_count: number
}

interface Chunk {
  chunk_id: string
  document_id: string
  text: string
  metadata: Metadata
}

export type SearchResults =
  | { mode: "hybrid"; metric: Metric; hits: (Chunk & { score: number })[] }
  | { mode: "mmr"; metric: Metric; hits: (Chunk & { distance: number })[] }

interface SearchQuery {
  query: string
  k: number
  metric: Metric
  filter?: Metadata
  mmr?: { candidates?: number; lambda: number }
}

interface CollectionPage {
  collections: Collection[]
  next_cursor: string | null
}

const collectionsPath = "/v1/collections"
const collectionPath = (name: string) => `${collectionsPath}/${encodeURIComponent(name)}`

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch (error) {
    if (init?.signal?.aborted) throw error
    throw new Error("Could not reach the Vekryx server.")
  }

  const data = await response.json().catch(() => null)
  if (!response.ok || data === null) {
    throw new Error(data?.error?.message ?? `Vekryx returned an unexpected response (HTTP ${response.status}).`)
  }
  return data as T
}

function post<T>(path: string, body: object, signal?: AbortSignal) {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  })
}

export const api = {
  async collections(signal?: AbortSignal) {
    const collections: Collection[] = []
    let cursor: string | null = null
    do {
      const params = new URLSearchParams({ limit: "100" })
      if (cursor !== null) params.set("cursor", cursor)
      const page: CollectionPage = await request(`${collectionsPath}?${params}`, { signal })
      collections.push(...page.collections)
      cursor = page.next_cursor
    } while (cursor !== null)
    return collections
  },
  collection: (name: string) => request<Collection>(collectionPath(name)),
  create: (body: { name: string; model: string }) =>
    post<Collection>(collectionsPath, body),
  ingest: (name: string, body: { id: string; text: string; metadata?: Metadata }) =>
    post<{ document_id: string; chunk_count: number }>(`${collectionPath(name)}/documents`, body),
  search: (name: string, body: SearchQuery, signal?: AbortSignal) =>
    post<SearchResults>(`${collectionPath(name)}/search`, body, signal),
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Something went wrong."
}

export function parseMetadata(input: string): Metadata | undefined {
  if (!input.trim()) return undefined
  let value: unknown
  try {
    value = JSON.parse(input)
  } catch {
    throw new Error("Enter a valid JSON object.")
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Enter a JSON object with key–value pairs.")
  }
  return value as Metadata
}
