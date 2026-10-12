import type { CheckResponse } from "./types";

export async function checkText(text: string, file: File | null): Promise<CheckResponse> {
  const body = new FormData();
  body.append("text", text);
  if (file) body.append("file", file);

  const res = await fetch("/api/check", { method: "POST", body });
  if (!res.ok) throw new Error(`server returned ${res.status}`);
  return (await res.json()) as CheckResponse;
}
