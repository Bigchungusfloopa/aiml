import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { splitHighlights } from "../lib/highlight";

// Heavy renderers load only when a file of that type is opened.
const DocxPreview = lazy(() => import("./viewers/DocxPreview"));
const MarkdownPreview = lazy(() => import("./viewers/MarkdownPreview"));
const PdfPreview = lazy(() => import("./viewers/PdfPreview"));
const TextPreview = lazy(() => import("./viewers/TextPreview"));

export interface ViewerTarget {
  name: string;
  /** Original file; null once the page has been reloaded. */
  file: File | null;
  /** Text the detector actually analysed. */
  text: string | null;
  phrases: string[];
}

type Kind = "pdf" | "docx" | "md" | "txt";

export function kindOf(name: string): Kind {
  const ext = name.includes(".") ? name.split(".").pop()!.toLowerCase() : "";
  if (ext === "pdf") return "pdf";
  if (ext === "docx") return "docx";
  if (ext === "md") return "md";
  return "txt";
}

const KIND_LABEL: Record<Kind, string> = {
  pdf: "PDF",
  docx: "Word document",
  md: "Markdown",
  txt: "plain text",
};

export default function FileViewer({ target, onClose }: { target: ViewerTarget; onClose: () => void }) {
  const kind = kindOf(target.name);
  const hasText = !!target.text;
  const [tab, setTab] = useState<"preview" | "text">(target.file ? "preview" : "text");
  const closeBtn = useRef<HTMLButtonElement>(null);

  // Escape closes; focus moves into the dialog and returns on close; page scroll locks.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeBtn.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
      opener?.focus?.();
    };
  }, [onClose]);

  const docxFailed = useCallback(() => hasText && setTab("text"), [hasText]);

  const file = target.file;
  const showPreview = tab === "preview" && file;

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={`viewing ${target.name}`}>
        <div className="term-bar">
          <span className="dot r" />
          <span className="dot y" />
          <span className="dot g" />
          <span className="term-title">{target.name} — {KIND_LABEL[kind]}</span>
          <button ref={closeBtn} type="button" className="bar-btn" onClick={onClose}>close</button>
        </div>

        {file && hasText && (
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "preview"} className={tab === "preview" ? "on" : ""}
                    onClick={() => setTab("preview")}>original</button>
            <button role="tab" aria-selected={tab === "text"} className={tab === "text" ? "on" : ""}
                    onClick={() => setTab("text")}>extracted text</button>
          </div>
        )}

        <div className="modal-body">
          {!file && (
            <p className="viewer-note">
              The original file isn't kept after a reload — showing the text that was analysed.
            </p>
          )}
          <Suspense fallback={<p className="viewer-note">loading viewer…</p>}>
            {showPreview && kind === "pdf" && <PdfPreview file={file} />}
            {showPreview && kind === "docx" && <DocxPreview file={file} onFail={docxFailed} />}
            {showPreview && kind === "md" && <MarkdownPreview file={file} />}
            {showPreview && kind === "txt" && <TextPreview file={file} />}
          </Suspense>
          {tab === "text" && hasText && (
            <pre className="viewer-pre">
              {splitHighlights(target.text!, target.phrases).map((p, i) =>
                p.hit ? <mark key={i}>{p.text}</mark> : <span key={i}>{p.text}</span>,
              )}
            </pre>
          )}
          {tab === "text" && !hasText && <p className="viewer-note">no extracted text available.</p>}
        </div>
      </div>
    </div>
  );
}
