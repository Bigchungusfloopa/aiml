import { renderAsync } from "docx-preview";
import { useEffect, useRef, useState } from "react";

/** Renders the .docx in place (pages, tables, images) using docx-preview. */
export default function DocxPreview({ file, onFail }: { file: File; onFail: () => void }) {
  const host = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    (async () => {
      try {
        const buf = await file.arrayBuffer();
        if (cancelled || !host.current) return;
        host.current.replaceChildren();
        await renderAsync(buf, host.current, undefined, { inWrapper: true, breakPages: true });
        if (!cancelled) setState("ready");
      } catch {
        if (!cancelled) {
          setState("error");
          onFail();
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [file, onFail]);

  return (
    <>
      {state === "loading" && <p className="viewer-note">rendering…</p>}
      {state === "error" && (
        <p className="viewer-note error">could not render this document — see the extracted text.</p>
      )}
      <div ref={host} className="docx-host" />
    </>
  );
}
