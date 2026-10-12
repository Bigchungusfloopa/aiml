import { useCallback, useEffect, useRef, useState } from "react";
import { checkText } from "./api";
import FileViewer, { type ViewerTarget } from "./components/FileViewer";
import PromptInput from "./components/PromptInput";
import TurnView from "./components/Turn";
import { loadHistory, saveHistory } from "./lib/storage";
import type { Turn } from "./types";

export default function App() {
  const [history, setHistory] = useState<Turn[]>(loadHistory);
  const [busy, setBusy] = useState(false);
  const [dropped, setDropped] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [viewer, setViewer] = useState<ViewerTarget | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const nextId = useRef(history.reduce((m, t) => Math.max(m, t.id), 0) + 1);
  // Original files only live for this page session; they are not persisted.
  const files = useRef(new Map<number, File>());
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    saveHistory(history);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    end.current?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "end" });
  }, [history]);

  const push = (t: Omit<Turn, "id">) => {
    const id = nextId.current++;
    setHistory((h) => [...h, { ...t, id }]);
    return id;
  };

  const reject = (message: string, filename: string) => {
    push({ words: 0, src: filename, error: message, text: "", result: null });
    setAnnouncement(message);
  };

  const submit = async (text: string, file: File | null) => {
    const id = push({ pending: true, src: file?.name ?? null });
    if (file) files.current.set(id, file);
    setBusy(true);
    setAnnouncement("analyzing…");
    try {
      const res = await checkText(text, file);
      setHistory((h) => h.map((t) => (t.id === id ? { id, ...res } : t)));
      setAnnouncement(
        res.result
          ? `Result: ${res.result.confidence_band}, ${(res.result.confidence * 100).toFixed(0)} percent sure.`
          : res.error ?? "",
      );
    } catch (e) {
      const error = `could not reach the detector — ${e instanceof Error ? e.message : e}`;
      setHistory((h) =>
        h.map((t) =>
          t.id === id ? { id, words: 0, src: file?.name ?? null, error, text: "", result: null } : t,
        ),
      );
      setAnnouncement(error);
    } finally {
      setBusy(false);
    }
  };

  const clear = () => {
    files.current.clear();
    setHistory([]);
    setAnnouncement("history cleared");
  };

  const viewTurn = (turn: Turn) =>
    setViewer({
      name: turn.src ?? "text",
      file: files.current.get(turn.id) ?? null,
      text: turn.text || null,
      phrases: turn.result?.style.cliches_found ?? [],
    });

  const closeViewer = useCallback(() => setViewer(null), []);

  return (
    <div
      className="page"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f) setDropped(f);
      }}
    >
      <main className={`terminal${dragging ? " drop" : ""}`}>
        <div className="term-bar">
          <span className="dot r" aria-hidden="true" />
          <span className="dot y" aria-hidden="true" />
          <span className="dot g" aria-hidden="true" />
          <span className="term-title">czechtext — zsh</span>
          {history.length > 0 && (
            <button type="button" className="bar-btn" onClick={clear}>clear</button>
          )}
        </div>

        {history.length === 0 && (
          <div className="term-block">
            <div className="banner">
              note: reworded ("humanized") AI is hard to catch &mdash; read this as evidence,
              not proof.
            </div>
            <span className="cmd">
              <span className="p">&gt;</span> ready. paste text or attach file to check.
            </span>
          </div>
        )}

        {history.map((turn, i) => (
          <div key={turn.id}>
            <TurnView turn={turn} onView={viewTurn} />
            {i < history.length - 1 && (
              <div className="divider" aria-hidden="true">{"-".repeat(64)}</div>
            )}
          </div>
        ))}
      </main>
      <div ref={end} />

      <div className="sr-only" role="status" aria-live="polite">{announcement}</div>

      <PromptInput
        disabled={busy}
        onSubmit={submit}
        onReject={reject}
        droppedFile={dropped}
        onPreview={(file) => setViewer({ name: file.name, file, text: null, phrases: [] })}
      />

      {viewer && <FileViewer target={viewer} onClose={closeViewer} />}
    </div>
  );
}
