import { useState } from "react";
import { Copy, Check } from "lucide-react";

type Props = {
  code: string;
  language?: string;
  filename?: string;
  showLineNumbers?: boolean;
};

export function CodeBlock({ code, language = "python", filename, showLineNumbers = true }: Props) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  const lines = code.split("\n");

  return (
    <div className="group relative overflow-hidden rounded-2xl border border-white/10 bg-ink-900/60 shadow-card">
      <div className="flex items-center justify-between border-b border-white/5 bg-ink-900/80 px-4 py-2.5 backdrop-blur">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-white/10" />
          <span className="h-2.5 w-2.5 rounded-full bg-white/10" />
          <span className="h-2.5 w-2.5 rounded-full bg-white/10" />
          {filename && (
            <span className="ml-3 font-mono text-[11px] uppercase tracking-widest text-ink-400">
              {filename}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-widest text-ink-500">{language}</span>
          <button
            onClick={onCopy}
            className="inline-flex items-center gap-1.5 rounded-full border border-white/5 bg-white/[0.03] px-2.5 py-1 text-[10px] font-medium uppercase tracking-widest text-ink-300 transition hover:border-white/15 hover:bg-white/[0.08] hover:text-white"
          >
            {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>
      <pre className="overflow-x-auto px-4 py-4 text-[13px] leading-[1.7] font-mono text-ink-100">
        <code>
          {lines.map((line, i) => (
            <div key={i} className="flex">
              {showLineNumbers && (
                <span className="mr-4 inline-block w-5 select-none text-right text-ink-600">{i + 1}</span>
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
  s = s.replace(/("""|'''|"|')([^\n]*?)\1/g, (m) => `<span class="text-emerald-300/90">${m}</span>`);
  s = s.replace(/\b(from|import|as|def|class|return|if|elif|else|for|while|in|not|and|or|with|None|True|False|self)\b/g, '<span class="text-accent-300">$1</span>');
  s = s.replace(/\b(AVQAttention|AVQConfig|AttentionShapeConfig|CodebookConfig|RoutingConfig)\b/g, '<span class="text-glow-soft">$1</span>');
  s = s.replace(/\b([0-9]+)\b/g, '<span class="text-amber-200/80">$1</span>');
  return s || "&nbsp;";
}
