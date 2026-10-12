import { useEffect, useState } from "react";

/** Uses the browser's built-in PDF viewer on a local blob URL. */
export default function PdfPreview({ file }: { file: File }) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    const u = URL.createObjectURL(new Blob([file], { type: "application/pdf" }));
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  if (!url) return <p className="viewer-note">loading…</p>;
  return (
    <>
      <iframe className="pdf-frame" src={url} title={`preview of ${file.name}`} />
      <p className="viewer-note">
        Not showing? <a href={url} target="_blank" rel="noopener noreferrer">open in a new tab</a> or
        use the extracted text.
      </p>
    </>
  );
}
