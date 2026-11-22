import { useState, type FormEvent, type ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { api, errorMessage, parseMetadata, type Collection } from "@/lib/api"

function field(data: FormData, name: string) {
  return String(data.get(name) ?? "")
}

function EditorDialog({
  trigger, title, description, submitLabel = title, pendingLabel, fields, save,
}: {
  trigger: ReactNode
  title: string
  description: string
  submitLabel?: string
  pendingLabel: string
  fields: ReactNode
  save: (data: FormData) => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState("")

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (pending) return
    const data = new FormData(event.currentTarget)
    setError("")
    setPending(true)
    try {
      await save(data)
      setOpen(false)
    } catch (error) {
      setError(errorMessage(error))
    } finally {
      setPending(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(next) => {
      if (!pending) {
        setOpen(next)
        setError("")
      }
    }}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent>
        <DialogTitle className="dialog-title">{title}</DialogTitle>
        <DialogDescription className="dialog-description">{description}</DialogDescription>
        <form className="editor-form" onSubmit={submit} aria-busy={pending}>
          <fieldset disabled={pending}>{fields}</fieldset>
          {error && <p className="error" role="alert">{error}</p>}
          <div className="dialog-actions">
            <DialogClose asChild><Button disabled={pending}>Cancel</Button></DialogClose>
            <Button type="submit" variant="primary" disabled={pending}>
              {pending ? pendingLabel : submitLabel}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

const idPattern = "[A-Za-z0-9_\\-]+"
const idHelp = "Use letters, numbers, hyphens, or underscores."

export function CreateCollectionDialog({ trigger, onCreated }: {
  trigger: ReactNode
  onCreated: (collection: Collection) => void
}) {
  return (
    <EditorDialog
      trigger={trigger}
      title="New collection"
      submitLabel="Create collection"
      description="Choose an Ollama embedding model for this collection."
      pendingLabel="Creating…"
      fields={<>
        <label className="field">
          Name
          <Input name="name" placeholder="research-notes" pattern={idPattern} maxLength={128} title={idHelp} required />
        </label>
        <label className="field">
          Embedding model
          <Input name="model" defaultValue="embeddinggemma" maxLength={256} required />
        </label>
      </>}
      save={async (data) => {
        const collection = await api.create({
          name: field(data, "name"),
          model: field(data, "model").trim(),
        })
        onCreated(collection)
      }}
    />
  )
}

export function AddDocumentDialog({ trigger, collection, onAdded }: {
  trigger: ReactNode
  collection: string
  onAdded: (count: number) => void
}) {
  return (
    <EditorDialog
      trigger={trigger}
      title="Add document"
      description="An existing document with the same ID will be replaced."
      pendingLabel="Embedding…"
      fields={<>
        <label className="field">
          Document ID
          <Input name="id" placeholder="storage-notes" pattern={idPattern} maxLength={128} title={idHelp} required />
        </label>
        <label className="field">
          Text
          <Textarea name="text" placeholder="Paste document text…" rows={6} required />
        </label>
        <label className="field">
          <span>Metadata <span className="muted">(optional JSON)</span></span>
          <Textarea name="metadata" placeholder={'{"project":"vekryx"}'} rows={2} spellCheck={false} className="json-input" />
        </label>
      </>}
      save={async (data) => {
        const text = field(data, "text")
        if (!text.trim()) throw new Error("Enter document text.")
        const metadata = parseMetadata(field(data, "metadata"))
        if (metadata && Object.hasOwn(metadata, "document_id")) {
          throw new Error("The metadata key document_id is reserved.")
        }
        const result = await api.ingest(collection, { id: field(data, "id"), text, metadata })
        onAdded(result.chunk_count)
      }}
    />
  )
}
