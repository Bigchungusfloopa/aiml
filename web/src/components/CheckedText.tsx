import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { splitHighlights } from "../lib/highlight";

interface Props {
  text: string;
  phrases: string[];
  truncated?: boolean;
}

/** The text that was checked, with flagged phrases marked; collapsed to ~3 lines. */
export default function CheckedText({ text, phrases, truncated }: Props) {
  const [open, setOpen] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const pieces = useMemo(() => splitHighlights(text, phrases), [text, phrases]);
  const hits = pieces.reduce((n, p) => n + (p.hit ? 1 : 0), 0);

  // Measure while collapsed; keep the last answer while expanded.
  useLayoutEffect(() => {
    const el = box.current;
    if (!el || open) return;
    const measure = () => setOverflows(el.scrollHeight > el.clientHeight + 1);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [text, open]);

  return (
    <div className="term-block echo">
      <div ref={box} className={`echo-text${open ? " open" : ""}`}>
        {pieces.map((p, i) =>
          p.hit ? (
            <mark key={i} title="AI-typical phrase">{p.text}</mark>
          ) : (
            <span key={i}>{p.text}</span>
          ),
        )}
        {truncated && open && <span className="muted"> … (saved copy shortened)</span>}
      </div>
      {(overflows || open || hits > 0) && (
        <div className="echo-meta">
          {(overflows || open) && (
            <button type="button" className="link" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
              {open ? "show less" : "show all"}
            </button>
          )}
          {hits > 0 && (
            <span className="muted">
              {(overflows || open) && " · "}
              {hits} flagged phrase{hits === 1 ? "" : "s"} highlighted
            </span>
          )}
        </div>
      )}
    </div>
  );
}
