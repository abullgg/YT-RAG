'use client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useState, useRef, useCallback, useEffect } from 'react';
import {
  checkHealth,
  uploadDocument,
  pollStatus,
  askQuestion,
  type UploadResponse,
  type AskResponse,
  type HealthResponse,
} from '@/lib/api';

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
interface Message {
  id: string;
  role: 'user' | 'ai';
  content: string;
  confidence?: number;
  sources?: string[];
}

// ─────────────────────────────────────────────────────────────
// Tiny SVG icons
// ─────────────────────────────────────────────────────────────
const IconFile = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
  </svg>
);

const IconUpload = () => (
  <svg className="drop-zone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
);

const IconX = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"/>
    <line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);

const IconSend = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);

const IconChevron = ({ open }: { open: boolean }) => (
  <svg className={`src-chevron${open ? ' open' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9"/>
  </svg>
);

const IconDoc = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
    <line x1="8" y1="13" x2="16" y2="13"/>
    <line x1="8" y1="17" x2="14" y2="17"/>
  </svg>
);

// ─────────────────────────────────────────────────────────────
// Source chip (collapsible)
// ─────────────────────────────────────────────────────────────
function SourceChip({ index, text }: { index: number; text: string }) {
  const [open, setOpen] = useState(false);
  const preview = text.length > 80 ? text.slice(0, 80) + '…' : text;
  return (
    <div className="source-chip">
      <div
        className="source-chip-header"
        onClick={() => setOpen(v => !v)}
        role="button"
        tabIndex={0}
        onKeyDown={e => e.key === 'Enter' && setOpen(v => !v)}
      >
        <span className="src-num">{index}</span>
        <span className="src-preview">{preview}</span>
        <IconChevron open={open} />
      </div>
      {open && <div className="source-chip-body">{text}</div>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// AI message bubble
// ─────────────────────────────────────────────────────────────
function AiBubble({ msg }: { msg: Message }) {
  const pct = msg.confidence !== undefined ? Math.round(msg.confidence * 100) : null;
  return (
    <div className="msg-row ai">
      <div className="msg-avatar ai">AI</div>
      <div className="msg-bubble ai">
        <div className="prose">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
        </div>
        {pct !== null && (
          <span className={`conf-badge${pct >= 70 ? ' high' : ''}`}>
            {pct}% confidence
          </span>
        )}
        {msg.sources && msg.sources.length > 0 && (
          <div className="sources-wrap">
            <div className="sources-header">Sources ({msg.sources.length})</div>
            {msg.sources.map((s, i) => (
              <SourceChip key={i} index={i + 1} text={s} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// User message bubble
// ─────────────────────────────────────────────────────────────
function UserBubble({ msg }: { msg: Message }) {
  return (
    <div className="msg-row user">
      <div className="msg-avatar user-av">U</div>
      <div className="msg-bubble user">{msg.content}</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────────
export default function Home() {
  // ── State ──
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadState, setUploadState] =
    useState<'idle' | 'uploading' | 'polling' | 'ready' | 'error'>('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatusText, setUploadStatusText] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isAsking, setIsAsking] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [messages, isAsking]);

  // Auto-resize textarea
  const autoResize = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
  };

  // ── Health poll ──
  useEffect(() => {
    const run = async () => {
      try { setHealth(await checkHealth()); setHealthError(false); }
      catch { setHealthError(true); }
    };
    run();
    const t = setInterval(run, 10_000);
    return () => clearInterval(t);
  }, []);

  // ── File handling ──
  const pickFile = useCallback((f: File | null) => {
    if (!f) return;
    setFile(f);
    setUploadState('idle');
    setUploadError('');
    setUploadResult(null);
    setMessages([]);
    setInput('');
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  }, [pickFile]);

  // ── Upload ──
  const handleUpload = useCallback(async () => {
    if (!file) return;
    setUploadState('uploading');
    setUploadProgress(10);
    setUploadStatusText('Uploading…');
    setUploadError('');

    try {
      const result = await uploadDocument(file);
      setUploadResult(result);

      if (result.status === 'completed') {
        setUploadProgress(100);
        setUploadStatusText('Ready');
        setUploadState('ready');
        const h = await checkHealth().catch(() => health);
        if (h) setHealth(h);
        return;
      }

      // Poll for large files
      setUploadState('polling');
      setUploadProgress(20);
      setUploadStatusText('Processing…');

      let retries = 0;
      const poll = async () => {
        if (retries++ >= 60) { setUploadState('error'); setUploadError('Processing timed out.'); return; }
        try {
          const s = await pollStatus(result.job_id);
          setUploadProgress(Math.max(20, Math.min(s.progress ?? 0, 95)));
          setUploadStatusText(s.status === 'processing' ? 'Processing…' : `${s.status}…`);

          if (s.status === 'completed') {
            setUploadProgress(100);
            setUploadStatusText('Ready');
            setUploadState('ready');
            const h = await checkHealth().catch(() => health);
            if (h) setHealth(h);
          } else if (s.status === 'failed') {
            setUploadState('error');
            setUploadError(s.error ?? 'Processing failed.');
          } else {
            setTimeout(poll, 2000);
          }
        } catch (e) { setUploadState('error'); setUploadError(String(e)); }
      };
      setTimeout(poll, 2000);
    } catch (e) { setUploadState('uploading'); setUploadError(String(e)); setUploadState('error'); }
  }, [file, health]);

  // ── Ask ──
  const handleAsk = useCallback(async () => {
    const q = input.trim();
    if (!q || isAsking || uploadState !== 'ready') return;

    const uid = crypto.randomUUID();
    setMessages(prev => [...prev, { id: uid, role: 'user', content: q }]);
    setInput('');
    if (textareaRef.current) { textareaRef.current.style.height = 'auto'; }
    setIsAsking(true);

    try {
      const res: AskResponse = await askQuestion({
        question: q,
        top_k: 3,
        ...(uploadResult?.document_id ? { document_id: uploadResult.document_id } : {}),
      });
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'ai',
        content: res.answer,
        confidence: res.confidence,
        sources: res.sources,
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'ai',
        content: `Something went wrong: ${String(e)}`,
      }]);
    } finally {
      setIsAsking(false);
    }
  }, [input, isAsking, uploadState, uploadResult]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk(); }
  };

  const isReady = uploadState === 'ready';
  const isProcessing = uploadState === 'uploading' || uploadState === 'polling';
  const backendOnline = health !== null && !healthError;

  // ─────────────────────────────────────────────────────────
  return (
    <div className="app-shell">

      {/* ════════ SIDEBAR ════════ */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            {/* logo mark */}
            <div className="logo-mark">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
              </svg>
            </div>
            <span className="logo-text">RAG Assistant</span>
          </div>
        </div>

        <div className="sidebar-content">
          {/* ── Document section ── */}
          <div className="module-card">
            <div className="section-label">Document</div>

            {!file ? (
              <div
                className={`drop-zone${dragOver ? ' drag-over' : ''}`}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt"
                  style={{ display: 'none' }}
                  onChange={e => pickFile(e.target.files?.[0] ?? null)}
                />
                <IconUpload />
                <div className="drop-zone-title">Upload PDF or TXT</div>
                <div className="drop-zone-hint">Click or drag &amp; drop</div>
              </div>
            ) : (
              <>
                <div className="file-card">
                  <div className="file-card-icon"><IconFile /></div>
                  <div className="file-card-info">
                    <div className="file-card-name" title={file.name}>{file.name}</div>
                    <div className="file-card-meta">{(file.size / 1024).toFixed(1)} KB · {file.type || 'text/plain'}</div>
                  </div>
                  {!isProcessing && !isReady && (
                    <button className="file-card-remove" onClick={() => { setFile(null); setUploadState('idle'); }} title="Remove">
                      <IconX />
                    </button>
                  )}
                </div>

                {isProcessing && (
                  <div style={{ marginTop: 9 }}>
                    <div className="progress-bar-wrap">
                      <div className="progress-bar-fill" style={{ width: `${uploadProgress}%` }} />
                    </div>
                    <div className="progress-status">{uploadStatusText}</div>
                  </div>
                )}

                {uploadState === 'error' && (
                  <div className="alert alert-error" style={{ marginTop: 9 }}>{uploadError}</div>
                )}

                {isReady && (
                  <div className="alert alert-success" style={{ marginTop: 9 }}>
                    Indexed — ready to query
                  </div>
                )}

                {(uploadState === 'idle' || uploadState === 'error') && (
                  <button className="btn btn-primary btn-full" style={{ marginTop: 9 }} onClick={handleUpload}>
                    Process Document
                  </button>
                )}

                {isReady && (
                  <button
                    className="btn btn-secondary btn-full"
                    style={{ marginTop: 9 }}
                    onClick={() => { setFile(null); setUploadState('idle'); setMessages([]); setInput(''); }}
                  >
                    Upload another
                  </button>
                )}
              </>
            )}
          </div>

          {/* ── Session ── */}
          <div className="module-card">
            <div className="section-label">Session</div>
            <div className="stats-grid">
              <div className="stat-chip">
                <div className="stat-value">{health?.documents_indexed ?? '—'}</div>
                <div className="stat-label">Indexed</div>
              </div>
              <div className="stat-chip">
                <div className="stat-value">{messages.filter(m => m.role === 'user').length || '—'}</div>
                <div className="stat-label">Queries</div>
              </div>
            </div>
          </div>
        </div>

        <div className="sidebar-footer">
          <span className={`status-dot${backendOnline ? ' online' : ' offline'}`}>
            {backendOnline ? 'Engine online' : 'Engine offline'}
          </span>
        </div>
      </aside>

      {/* ════════ MAIN AREA ════════ */}
      <main className="main-area">
        {/* Top bar */}
        <div className="top-bar">
          <span className="top-bar-title">RAG Document Assistant</span>
          {isReady && uploadResult && (
            <div className="top-bar-doc">
              <span className="doc-badge">{uploadResult.filename}</span>
            </div>
          )}
        </div>

        {/* Offline banner */}
        {!backendOnline && !health && (
          <div style={{ padding: '10px 22px 0' }}>
            <div className="alert alert-error">
              Backend offline — run: <code style={{ fontFamily: 'monospace', fontSize: '0.82em' }}>uvicorn src.main:app --reload</code>
            </div>
          </div>
        )}

        {/* Chat wrapper (thread + input bar) */}
        <div className="chat-wrapper">
          {/* ── Thread ── */}
          <div className="chat-thread" ref={threadRef}>
            <div className="thread-inner">

              {/* Empty state */}
              {messages.length === 0 && (
                <div className="empty-state">
                  <div className="empty-icon-ring"><IconDoc /></div>
                  <h2>{isReady ? 'Ask anything' : 'No document loaded'}</h2>
                  <p>
                    {isReady
                      ? `Querying ${uploadResult?.filename}. Type a question below.`
                      : 'Upload a PDF or TXT file in the sidebar, then process it to start.'}
                  </p>
                </div>
              )}

              {/* Messages */}
              {messages.map(msg =>
                msg.role === 'user'
                  ? <UserBubble key={msg.id} msg={msg} />
                  : <AiBubble key={msg.id} msg={msg} />
              )}

              {/* Typing indicator */}
              {isAsking && (
                <div className="msg-row ai">
                  <div className="msg-avatar ai">AI</div>
                  <div className="typing-indicator">
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                  </div>
                </div>
              )}

            </div>
          </div>

          {/* ── Input bar (floating capsule module) ── */}
          <div className="input-bar">
            <div className="input-module">
              <div className="input-row">
                <textarea
                  ref={textareaRef}
                  className="chat-textarea"
                  placeholder={isReady ? 'Ask a question about your document…' : 'Process a document to start…'}
                  value={input}
                  rows={1}
                  onChange={e => { setInput(e.target.value); autoResize(); }}
                  onKeyDown={handleKeyDown}
                  disabled={!isReady || isAsking}
                />
              </div>
              <div className="input-actions-row">
                <span className="input-hint-text">Enter to send · Shift+Enter for newline</span>
                <button
                  className="send-btn"
                  onClick={handleAsk}
                  disabled={!isReady || !input.trim() || isAsking}
                  title="Send (Enter)"
                >
                  <IconSend />
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
