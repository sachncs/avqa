import { useState } from "react";
import { Copy, Check } from "lucide-react";

type Props = {
  code: string;
  language?: string;
  filename?: string;
  showLineNumbers?: boolean;
};

export function CodeBlock({
  code,
  language = "python",
  filename,
  showLineNumbers = true,
}: Props) {
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">(
    "idle",
  );

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopyStatus("copied");
    } catch {
      setCopyStatus("failed");
    }
    window.setTimeout(() => setCopyStatus("idle"), 1800);
  };

  const lines = code.split("\n");

  return (
    <div className="group relative overflow-hidden border border-white/10 bg-ink-900/60">
      <div className="flex min-h-12 items-center justify-between gap-3 border-b border-white/10 bg-ink-900/80 px-4 py-2">
        <div className="flex min-w-0 items-center gap-3">
          {filename && (
            <span className="truncate font-mono text-[11px] tracking-wide text-ink-300">
              {filename}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-widest text-ink-500">
            {language}
          </span>
          <button
            type="button"
            aria-label={`${copyStatus === "copied" ? "Copied" : copyStatus === "failed" ? "Copy failed" : "Copy"} ${filename ?? "code"}`}
            onClick={onCopy}
            className="inline-flex min-h-9 items-center gap-1.5 border border-white/10 px-2.5 py-1 text-[10px] font-medium uppercase tracking-widest text-ink-300 transition hover:border-accent-300/70 hover:text-white"
          >
            {copyStatus === "copied" ? (
              <Check aria-hidden="true" className="h-3 w-3" />
            ) : (
              <Copy aria-hidden="true" className="h-3 w-3" />
            )}
            {copyStatus === "copied"
              ? "Copied"
              : copyStatus === "failed"
                ? "Unavailable"
                : "Copy"}
          </button>
        </div>
      </div>
      <pre className="overflow-x-auto px-4 py-4 text-[13px] leading-[1.7] font-mono text-ink-100">
        <code>
          {lines.map((line, i) => (
            <div key={i} className="flex">
              {showLineNumbers && (
                <span className="mr-4 inline-block w-5 select-none text-right text-ink-600">
                  {i + 1}
                </span>
              )}
              <span dangerouslySetInnerHTML={{ __html: highlight(line) }} />
            </div>
          ))}
        </code>
      </pre>
    </div>
  );
}

function escapeHtml(s: string) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlight(line: string): string {
  let s = escapeHtml(line);
  s = s.replace(/(#[^\n]*)$/g, '<span class="text-ink-500 italic">$1</span>');
  s = s.replace(
    /("""|'''|"|')([^\n]*?)\1/g,
    (m) => `<span class="text-glow-soft">${m}</span>`,
  );
  s = s.replace(
    /\b(from|import|as|def|class|return|if|elif|else|for|while|in|not|and|or|with|None|True|False|self)\b/g,
    '<span class="text-accent-300">$1</span>',
  );
  s = s.replace(
    /\b(AVQAttention|AVQConfig|AttentionShapeConfig|CodebookConfig|RoutingConfig)\b/g,
    '<span class="text-glow-soft">$1</span>',
  );
  s = s.replace(/\b([0-9]+)\b/g, '<span class="text-accent-200">$1</span>');
  return s || "&nbsp;";
}
