import { useEffect, useRef, useState } from "react";

export const ACCEPTED = ["pdf", "docx", "txt", "text", "md"];
const MAX_BYTES = 10 * 1024 * 1024;

interface Props {
  disabled: boolean;
  onSubmit: (text: string, file: File | null) => void;
  onReject: (message: string, filename: string) => void;
  /** File dropped anywhere on the page. */
  droppedFile: File | null;
  /** Open the viewer for the attached file before sending. */
  onPreview: (file: File) => void;
}

export function validateFile(f: File): string | null {
  const ext = f.name.includes(".") ? f.name.split(".").pop()!.toLowerCase() : "";
  if (!ACCEPTED.includes(ext)) return `Unsupported file type: .${ext}`;
  if (f.size > MAX_BYTES) return `${f.name} is larger than 10 MB`;
  return null;
}

export default function PromptInput({ disabled, onSubmit, onReject, droppedFile, onPreview }: Props) {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  const picker = useRef<HTMLInputElement>(null);

  const attach = (f: File) => {
    const problem = validateFile(f);
    if (problem) onReject(problem, f.name);
    else setFile(f);
  };

  useEffect(() => {
    if (droppedFile) attach(droppedFile);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [droppedFile]);

  // Grow with content, up to ~8 lines.
  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  const send = () => {
    if (disabled) return;
    onSubmit(text, file);
    setText("");
    setFile(null);
  };

  return (
    <div className="prompt-dock">
      <div className="prompt">
        {file && (
          <div className="chip">
            <button type="button" className="chip-name" title="preview file"
                    onClick={() => onPreview(file)}>
              {file.name}
            </button>
            <button type="button" aria-label="remove file" onClick={() => setFile(null)}>
              ×
            </button>
          </div>
        )}
        <div className="prompt-row">
          <button type="button" className="icon-btn" aria-label="attach file"
                  onClick={() => picker.current?.click()}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
                 strokeWidth="2" strokeLinecap="round">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
          <input ref={picker} type="file" hidden accept={ACCEPTED.map((e) => `.${e}`).join(",")}
                 onChange={(e) => {
                   const f = e.target.files?.[0];
                   if (f) attach(f);
                   e.target.value = "";
                 }} />
          <textarea
            ref={area}
            rows={1}
            value={text}
            placeholder="check text…"
            autoFocus
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button type="button" className="icon-btn" aria-label="send"
                  disabled={disabled} onClick={send}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
                 strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19V5M5 12l7-7 7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
