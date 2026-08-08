import fs

const path = 'public/app.js';
let content = fs.readFileSync(path, 'utf8');

// Replace element accesses with optional chaining where obvious
content = content.replace(/document\.getElementById\('([^']+)'\)\./g, "document.getElementById('$1')?.");
content = content.replace(/document\.querySelector\('([^']+)'\)\./g, "document.querySelector('$1')?.");
// Specifically for canvas context
content = content.replace(/\.getContext\(/g, "?.getContext(");

fs.writeFileSync('public/app_safe.js', content);
