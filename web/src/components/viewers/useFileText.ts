import { useEffect, useState } from "react";

/** Rendering a huge file in the DOM freezes the tab; show the head and say so. */
export const MAX_CHARS = 200_000;

export type FileText =
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "ready"; text: string; clipped: boolean };

export function useFileText(file: File): FileText {
  const [value, setValue] = useState<FileText>({ state: "loading" });

  useEffect(() => {
    let cancelled = false;
    setValue({ state: "loading" });
    file
      .text()
      .then((t) => {
        if (cancelled) return;
        setValue({ state: "ready", text: t.slice(0, MAX_CHARS), clipped: t.length > MAX_CHARS });
      })
      .catch((e: unknown) => {
        if (!cancelled) setValue({ state: "error", message: e instanceof Error ? e.message : String(e) });
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  return value;
}
