import { copyFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const distRoot = path.join(siteRoot, "dist");
const docsRoot = path.join(distRoot, "docs");

await mkdir(docsRoot, { recursive: true });
await copyFile(
  path.join(distRoot, "index.html"),
  path.join(docsRoot, "index.html"),
);
