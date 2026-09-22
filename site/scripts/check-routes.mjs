/* global console */

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const docsSource = await readFile(
  path.join(siteRoot, "src", "DocsApp.tsx"),
  "utf8",
);

const pageIds = [
  ...docsSource.matchAll(/\{\s*id:\s*"([^"]+)",\s*label:/gs),
].map(([, id]) => id);
const renderedIds = [...docsSource.matchAll(/page === "([^"]+)"/g)].map(
  ([, id]) => id,
);
const missingRenderers = pageIds.filter(
  (id) => id !== "overview" && !renderedIds.includes(id),
);

if (pageIds.length === 0) {
  throw new Error("docs navigation declares no pages");
}
if (missingRenderers.length > 0) {
  throw new Error(
    `docs pages missing renderers: ${missingRenderers.join(", ")}`,
  );
}
if (pageIds.length !== 10) {
  throw new Error(`expected 10 documentation pages; found ${pageIds.length}`);
}
if (!docsSource.includes("PAGES.some((page) => page.id === hash)")) {
  throw new Error(
    "docs hash routing is not constrained by the declared page list",
  );
}

console.log(
  `docs route contract passed for ${pageIds.length} navigation pages`,
);
