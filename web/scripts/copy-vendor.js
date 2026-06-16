const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const pkgRoot = path.join(root, "node_modules", "jgoboard");
const target = path.join(root, "static", "vendor", "jgoboard");

function copyDir(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

if (!fs.existsSync(pkgRoot)) {
  console.error("jgoboard not installed. Run npm install first.");
  process.exit(1);
}

fs.rmSync(target, { recursive: true, force: true });
fs.mkdirSync(target, { recursive: true });
copyDir(path.join(pkgRoot, "dist"), path.join(target, "dist"));
copyDir(path.join(pkgRoot, "large"), path.join(target, "large"));
console.log("Copied jgoboard assets to", target);
