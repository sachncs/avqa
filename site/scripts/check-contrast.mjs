import { error, log } from "node:console";
import process from "node:process";

const PALETTE = [
  ["primary text", "#F5F3E9"],
  ["secondary text", "#E8E6D8"],
  ["tertiary text", "#D1D3C4"],
  ["muted content", "#B3BBA9"],
  ["quiet content", "#939E8C"],
  ["small muted labels", "#83907E"],
  ["persimmon label", "#F5A17C"],
  ["persimmon link", "#E9784F"],
  ["lichen signal", "#B9D692"],
  ["lichen syntax", "#D5E7B8"],
];

const SURFACES = [
  ["carbon page", "#0D1510"],
  ["raised forest", "#121C15"],
  ["deep code surface", "#080E0A"],
];

function channel(hex, index) {
  return Number.parseInt(hex.slice(index, index + 2), 16) / 255;
}

function luminance(hex) {
  const [red, green, blue] = [1, 3, 5].map((index) => channel(hex, index));
  const linear = [red, green, blue].map((value) =>
    value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4,
  );
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

function contrast(foreground, background) {
  const values = [luminance(foreground), luminance(background)].sort(
    (left, right) => right - left,
  );
  return (values[0] + 0.05) / (values[1] + 0.05);
}

const failures = [];
for (const [textRole, foreground] of PALETTE) {
  for (const [surfaceRole, background] of SURFACES) {
    const ratio = contrast(foreground, background);
    if (ratio < 4.5) {
      failures.push(
        `${textRole} on ${surfaceRole}: ${ratio.toFixed(2)}:1 (needs 4.5:1)`,
      );
    }
  }
}

const primaryButtonRatio = contrast("#132016", "#B9D692");
if (primaryButtonRatio < 4.5) {
  failures.push(
    `button text on lichen: ${primaryButtonRatio.toFixed(2)}:1 (needs 4.5:1)`,
  );
}

const lightLogoRatio = contrast("#A5422A", "#F5F3E9");
if (lightLogoRatio < 4.5) {
  failures.push(
    `light logo accent: ${lightLogoRatio.toFixed(2)}:1 (needs 4.5:1)`,
  );
}

if (failures.length > 0) {
  error(failures.join("\n"));
  process.exitCode = 1;
} else {
  log("Core text, signal, and logo contrast pairs meet WCAG AA.");
}
