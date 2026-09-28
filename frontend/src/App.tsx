import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'

type DocumentInfo = {
  source: string
  file_type: string
  chunk_count: number
}

type DocumentListResponse = {
  documents: DocumentInfo[]
  total_documents: number
  total_chunks: number
}

type UploadResponse = {
  success: boolean
  filename: string
  file_type: string
  chunks_indexed: number
  message: string
}

type ChatSource = {
  source: string
  chunk_index: number
  page: number | null
}

type ChatResponse = {
  answer: string
  sources: ChatSource[]
  retrieved_chunks: number
  question: string
  success: boolean
}

type ChatTurn = ChatResponse & {
  id: number
}

type ApiErrorBody = {
  detail?: string
  message?: string
}

const MAX_FILE_SIZE = 10 * 1024 * 1024

function getFileSizeLabel(size: number) {
  if (size < 1024 * 1024) {
    return `${Math.max(1, Math.round(size / 1024))} KB`
  }

  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

async function getErrorMessage(response: Response) {
  try {
    const body = (await response.json()) as ApiErrorBody
    return body.detail ?? body.message ?? `Request failed with status ${response.status}.`
  } catch {
    return `Request failed with status ${response.status}.`
  }
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [documentTotal, setDocumentTotal] = useState(0)
  const [chunkTotal, setChunkTotal] = useState(0)
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(true)
  const [documentsError, setDocumentsError] = useState('')
  const [deletingDocument, setDeletingDocument] = useState('')

  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadError, setUploadError] = useState('')
  const [uploadMessage, setUploadMessage] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [question, setQuestion] = useState('')
  const [chatHistory, setChatHistory] = useState<ChatTurn[]>([])
  const [chatError, setChatError] = useState('')
  const [isAsking, setIsAsking] = useState(false)

  async function loadDocuments(showLoading = true) {
    if (showLoading) {
      setIsLoadingDocuments(true)
    }
    setDocumentsError('')

    try {
      const response = await fetch('/api/documents')
      if (!response.ok) {
        throw new Error(await getErrorMessage(response))
      }

      const data = (await response.json()) as DocumentListResponse
      setDocuments(data.documents)
      setDocumentTotal(data.total_documents)
      setChunkTotal(data.total_chunks)
    } catch (error) {
      setDocumentsError(
        error instanceof Error ? error.message : 'Unable to load indexed documents.',
      )
    } finally {
      if (showLoading) {
        setIsLoadingDocuments(false)
      }
    }
  }

  useEffect(() => {
    void loadDocuments()
  }, [])

  function handleFileChange(event: FormEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0] ?? null
    setSelectedFile(file)
    setUploadError('')
    setUploadMessage('')
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setUploadError('')
    setUploadMessage('')

    if (!selectedFile) {
      setUploadError('Choose a PDF, DOCX, or TXT file to upload.')
      return
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setUploadError('This file is larger than the 10 MB upload limit.')
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)
    setIsUploading(true)

    try {
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        throw new Error(await getErrorMessage(response))
      }

      const data = (await response.json()) as UploadResponse
      setUploadMessage(data.message)
      setSelectedFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
      await loadDocuments(false)
    } catch (error) {
      setUploadError(
        error instanceof Error ? error.message : 'Unable to upload this document.',
      )
    } finally {
      setIsUploading(false)
    }
  }

  async function handleDelete(documentName: string) {
    setDocumentsError('')
    setDeletingDocument(documentName)

    try {
      const response = await fetch(`/api/documents/${encodeURIComponent(documentName)}`, {
        method: 'DELETE',
      })
      if (!response.ok) {
        throw new Error(await getErrorMessage(response))
      }

      await loadDocuments(false)
    } catch (error) {
      setDocumentsError(
        error instanceof Error ? error.message : 'Unable to remove this document.',
      )
    } finally {
      setDeletingDocument('')
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion) {
      return
    }

    setChatError('')
    setIsAsking(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedQuestion }),
      })
      if (!response.ok) {
        throw new Error(await getErrorMessage(response))
      }

      const data = (await response.json()) as ChatResponse
      if (!data.success) {
        throw new Error(data.answer || 'The question could not be answered.')
      }

      setChatHistory((history) => [...history, { ...data, id: Date.now() }])
      setQuestion('')
    } catch (error) {
      setChatError(
        error instanceof Error ? error.message : 'Unable to answer your question.',
      )
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">R</span>
          <div>
            <p className="eyebrow">Local knowledge workspace</p>
            <h1>RAG Document Q&amp;A</h1>
          </div>
        </div>
        <p className="header-description">
          Upload documents, then ask grounded questions with source citations.
        </p>
      </header>

      <div className="dashboard-grid">
        <section className="workspace-column" aria-label="Document workspace">
          <section className="panel upload-panel" aria-labelledby="upload-heading">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Step 1</p>
                <h2 id="upload-heading">Add a document</h2>
              </div>
              <span className="format-badge">PDF · DOCX · TXT</span>
            </div>
            <p className="panel-description">
              Files are processed locally and added to your searchable knowledge base.
            </p>

            <form className="upload-form" onSubmit={handleUpload}>
              <label className="file-picker" htmlFor="document-file">
                <span className="file-picker-icon" aria-hidden="true">↑</span>
                <span className="file-picker-copy">
                  <strong>{selectedFile ? selectedFile.name : 'Choose a document'}</strong>
                  <small>
                    {selectedFile
                      ? `${selectedFile.type || 'Document'} · ${getFileSizeLabel(selectedFile.size)}`
                      : 'Maximum file size: 10 MB'}
                  </small>
                </span>
                <span className="choose-file">Browse</span>
              </label>
              <input
                ref={fileInputRef}
                id="document-file"
                className="visually-hidden"
                type="file"
                accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
                onChange={handleFileChange}
              />

              {uploadError && <p className="status-message error-message" role="alert">{uploadError}</p>}
              {uploadMessage && <p className="status-message success-message" role="status">{uploadMessage}</p>}

              <button className="primary-button" type="submit" disabled={!selectedFile || isUploading}>
                {isUploading ? 'Indexing document…' : 'Upload and index'}
              </button>
            </form>
          </section>

          <section className="panel documents-panel" aria-labelledby="documents-heading">
            <div className="panel-heading documents-heading">
              <div>
                <p className="eyebrow">Knowledge base</p>
                <h2 id="documents-heading">Indexed documents</h2>
              </div>
              <div className="document-summary" aria-label="Document totals">
                <span>{documentTotal} {documentTotal === 1 ? 'document' : 'documents'}</span>
                <span>{chunkTotal} {chunkTotal === 1 ? 'chunk' : 'chunks'}</span>
              </div>
            </div>

            {documentsError && (
              <div className="inline-error" role="alert">
                <span>{documentsError}</span>
                <button type="button" onClick={() => void loadDocuments()}>
                  Try again
                </button>
              </div>
            )}

            {isLoadingDocuments ? (
              <div className="empty-state loading-state" aria-live="polite">
                <span className="spinner" aria-hidden="true" />
                <p>Loading your document library…</p>
              </div>
            ) : documents.length === 0 ? (
              <div className="empty-state">
                <span className="empty-state-icon" aria-hidden="true">□</span>
                <h3>No documents indexed yet</h3>
                <p>Upload a document above to create a knowledge base for your questions.</p>
              </div>
            ) : (
              <ul className="document-list">
                {documents.map((document) => (
                  <li className="document-item" key={document.source}>
                    <div className="document-icon" aria-hidden="true">{document.file_type.slice(0, 1).toUpperCase()}</div>
                    <div className="document-details">
                      <strong title={document.source}>{document.source}</strong>
                      <span>{document.file_type.toUpperCase()} · {document.chunk_count} {document.chunk_count === 1 ? 'chunk' : 'chunks'}</span>
                    </div>
                    <button
                      className="text-button danger-button"
                      type="button"
                      onClick={() => void handleDelete(document.source)}
                      disabled={deletingDocument === document.source}
                    >
                      {deletingDocument === document.source ? 'Removing…' : 'Remove'}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </section>

        <section className="panel chat-panel" aria-labelledby="chat-heading">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Step 2</p>
              <h2 id="chat-heading">Ask your documents</h2>
            </div>
            <span className="chat-status"><span /> Ready</span>
          </div>
          <p className="panel-description">
            Answers are generated from the most relevant indexed document content.
          </p>

          <div className="chat-history" aria-live="polite">
            {chatHistory.length === 0 && !isAsking ? (
              <div className="empty-state chat-empty-state">
                <span className="empty-state-icon" aria-hidden="true">?</span>
                <h3>Start with a question</h3>
                <p>Try “What are the main ideas in my documents?” once you have uploaded a file.</p>
              </div>
            ) : (
              chatHistory.map((turn) => (
                <article className="chat-turn" key={turn.id}>
                  <div className="question-bubble">
                    <span>You</span>
                    <p>{turn.question}</p>
                  </div>
                  <div className="answer-card">
                    <div className="answer-label"><span aria-hidden="true">R</span> RAG assistant</div>
                    <p className="answer-text">{turn.answer}</p>
                    {turn.sources.length > 0 && (
                      <div className="sources">
                        <p>Sources</p>
                        <ul>
                          {turn.sources.map((source, index) => (
                            <li key={`${turn.id}-${source.source}-${source.chunk_index}-${index}`}>
                              <strong>{source.source}</strong>
                              <span>
                                {source.page ? `Page ${source.page}` : 'Page unavailable'} · Chunk {source.chunk_index}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </article>
              ))
            )}
            {isAsking && (
              <div className="answer-card pending-answer">
                <span className="spinner" aria-hidden="true" />
                <p>Searching your documents and preparing an answer…</p>
              </div>
            )}
          </div>

          {chatError && <p className="status-message error-message" role="alert">{chatError}</p>}

          <form className="question-form" onSubmit={handleAsk}>
            <label className="visually-hidden" htmlFor="question">Ask a question about your documents</label>
            <textarea
              id="question"
              rows={3}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about your documents…"
              disabled={isAsking}
            />
            <div className="question-form-footer">
              <span>Answers include source citations when available.</span>
              <button className="primary-button" type="submit" disabled={!question.trim() || isAsking}>
                {isAsking ? 'Thinking…' : 'Ask question'}
              </button>
            </div>
          </form>
        </section>
      </div>
    </main>
  )
}
