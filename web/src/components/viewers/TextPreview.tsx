import { useFileText } from "./useFileText";

export default function TextPreview({ file }: { file: File }) {
  const t = useFileText(file);
  if (t.state === "loading") return <p className="viewer-note">loading…</p>;
  if (t.state === "error") return <p className="viewer-note error">could not read file — {t.message}</p>;
  return (
    <>
      <pre className="viewer-pre">{t.text}</pre>
      {t.clipped && <p className="viewer-note">showing the first 200,000 characters</p>}
    </>
  );
}
