import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useFileText } from "./useFileText";

/** Rendered Markdown. Raw HTML is not rendered and remote images are not fetched. */
export default function MarkdownPreview({ file }: { file: File }) {
  const t = useFileText(file);
  if (t.state === "loading") return <p className="viewer-note">loading…</p>;
  if (t.state === "error") return <p className="viewer-note error">could not read file — {t.message}</p>;
  return (
    <>
      <div className="md">
        <Markdown
          remarkPlugins={[remarkGfm]}
          components={{
            img: ({ alt }) => <span className="muted">[image{alt ? `: ${alt}` : ""}]</span>,
            a: ({ href, children }) => (
              <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
            ),
          }}
        >
          {t.text}
        </Markdown>
      </div>
      {t.clipped && <p className="viewer-note">showing the first 200,000 characters</p>}
    </>
  );
}
