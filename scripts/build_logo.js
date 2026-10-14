const fs = require('fs');
const path = require('path');
const { Resvg } = require('B:/workgit/node_modules/@resvg/resvg-js');

function buildLogoSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <clipPath id="squircle-clip">
      <rect x="24" y="24" width="976" height="976" rx="220" />
    </clipPath>

    <!-- Electric Cyan Optic Gradient -->
    <linearGradient id="cyan-optic" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="60%" stop-color="#00f5ff"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>

    <!-- Amber Gold Telemetry Core -->
    <linearGradient id="amber-core" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#fef08a"/>
      <stop offset="50%" stop-color="#fbbf24"/>
      <stop offset="100%" stop-color="#d97706"/>
    </linearGradient>

    <!-- Titanium Slate Facet Gradients -->
    <linearGradient id="slate-light" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#475569"/>
      <stop offset="100%" stop-color="#334155"/>
    </linearGradient>

    <linearGradient id="slate-mid" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#334155"/>
      <stop offset="100%" stop-color="#1e293b"/>
    </linearGradient>

    <linearGradient id="slate-dark" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>

    <linearGradient id="slate-deep" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#070b14"/>
    </linearGradient>

    <!-- Ambient Shadow -->
    <filter id="subtle-shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="16" stdDeviation="20" flood-color="#0b0f19" flood-opacity="0.16" />
    </filter>
  </defs>

  <!-- 1. Luxury White Squircle Container -->
  <rect x="24" y="24" width="976" height="976" rx="220" fill="#ffffff" stroke="#e2e8f0" stroke-width="6" />

  <g clip-path="url(#squircle-clip)">
    <g transform="translate(512, 512)" filter="url(#subtle-shadow)">

      <!-- 2. Hexagonal Architectural Gateway -->
      <polygon points="
        0,-390
        338,-195
        338,195
        0,390
        -338,195
        -338,-195
      " fill="none" stroke="#0f172a" stroke-width="40" stroke-linejoin="round" />

      <!-- Inner Cyan Telemetry Ring -->
      <polygon points="
        0,-355
        307,-177
        307,177
        0,355
        -307,177
        -307,-177
      " fill="none" stroke="#00f5ff" stroke-width="4" opacity="0.45" stroke-dasharray="16, 12" />

      <!-- 3. SiliconMark Mascot: The Apex Silicon Lynx -->
      <!-- Watertight Base Silhouette Path (Eliminates all background leaks) -->
      <path d="
        M 0,-330
        L 60,-240
        L 140,-310
        L 155,-220
        L 230,-120
        L 255,-20
        L 210,120
        L 170,220
        L 90,290
        L 0,335
        L -90,290
        L -170,220
        L -210,120
        L -255,-20
        L -230,-120
        L -155,-220
        L -140,-310
        L -60,-240
        Z
      " fill="#070b14" />

      <!-- Back Neck & Shoulder Armor (Deep Shadow) -->
      <polygon points="0,335 90,290 170,220 80,180 0,220" fill="url(#slate-dark)" />
      <polygon points="0,335 -90,290 -170,220 -80,180 0,220" fill="url(#slate-deep)" />

      <!-- Outer Cheek Ruffs (Faceted Titanium) -->
      <polygon points="230,-120 255,-20 180,-10 160,-100" fill="url(#slate-mid)" />
      <polygon points="-230,-120 -255,-20 -180,-10 -160,-100" fill="url(#slate-dark)" />
      
      <polygon points="255,-20 210,120 150,80 180,-10" fill="url(#slate-light)" />
      <polygon points="-255,-20 -210,120 -150,80 -180,-10" fill="url(#slate-mid)" />

      <polygon points="210,120 170,220 100,160 150,80" fill="url(#slate-mid)" />
      <polygon points="-210,120 -170,220 -100,160 -150,80" fill="url(#slate-dark)" />

      <!-- Ears: Characteristic Upright Lynx Ear Tufts -->
      <!-- Left Ear (Facing / Right Coordinate) -->
      <polygon points="60,-240 140,-310 95,-180" fill="url(#slate-light)" />
      <polygon points="140,-310 155,-220 95,-180" fill="url(#slate-mid)" />
      <polygon points="140,-310 150,-340 135,-310" fill="#00f5ff" /> <!-- Ear Tuft Needle -->

      <!-- Right Ear (Facing / Left Coordinate) -->
      <polygon points="-60,-240 -140,-310 -95,-180" fill="url(#slate-mid)" />
      <polygon points="-140,-310 -155,-220 -95,-180" fill="url(#slate-dark)" />
      <polygon points="-140,-310 -150,-340 -135,-310" fill="#0284c7" /> <!-- Ear Tuft Needle -->

      <!-- Forehead Plate & Crown -->
      <polygon points="0,-330 60,-240 0,-180" fill="url(#slate-light)" />
      <polygon points="0,-330 -60,-240 0,-180" fill="url(#slate-mid)" />
      <polygon points="60,-240 95,-180 0,-140 0,-180" fill="url(#slate-mid)" />
      <polygon points="-60,-240 -95,-180 0,-140 0,-180" fill="url(#slate-dark)" />

      <!-- Brow Ridge (Aerodynamic & Powerful) -->
      <polygon points="0,-140 70,-115 130,-105 55,-70 0,-85" fill="url(#slate-light)" />
      <polygon points="0,-140 -70,-115 -130,-105 -55,-70 0,-85" fill="url(#slate-mid)" />

      <!-- Temporal / Temple Plates -->
      <polygon points="95,-180 160,-100 130,-105 70,-115" fill="url(#slate-mid)" />
      <polygon points="-95,-180 -160,-100 -130,-105 -70,-115" fill="url(#slate-dark)" />

      <!-- Nose Bridge & Muzzle Structure -->
      <polygon points="0,-85 35,-60 25,30 0,45" fill="url(#slate-mid)" />
      <polygon points="0,-85 -35,-60 -25,30 0,45" fill="url(#slate-dark)" />

      <!-- Almond Predator Eyes (Sharp, Alert Gaze in Electric Cyan) -->
      <!-- Left Eye Frame (Facing right) -->
      <polygon points="50,-65 115,-70 125,-40 65,-40" fill="#070b14" />
      <polygon points="60,-60 110,-65 120,-45 70,-45" fill="url(#cyan-optic)" />
      <polygon points="85,-62 92,-63 88,-47 82,-47" fill="#ffffff" /> <!-- Glint -->

      <!-- Right Eye Frame (Facing left) -->
      <polygon points="-50,-65 -115,-70 -125,-40 -65,-40" fill="#070b14" />
      <polygon points="-60,-60 -110,-65 -120,-45 -70,-45" fill="url(#cyan-optic)" />
      <polygon points="-85,-62 -92,-63 -88,-47 -82,-47" fill="#ffffff" /> <!-- Glint -->

      <!-- Sub-Orbital Malar Plates (Low-poly facets, no drawn hoses) -->
      <polygon points="65,-40 125,-40 150,-10 100,0 35,-30" fill="url(#slate-light)" />
      <polygon points="-65,-40 -125,-40 -150,-10 -100,0 -35,-30" fill="url(#slate-mid)" />

      <!-- Titanium Whisker Pads & Snout (Horizontal taper, no beaver teeth) -->
      <polygon points="25,30 85,25 65,85 0,70" fill="#94a3b8" />
      <polygon points="-25,30 -85,25 -65,85 0,70" fill="#64748b" />
      <polygon points="0,45 25,30 0,70" fill="#334155" />
      <polygon points="0,45 -25,30 0,70" fill="#1e293b" />

      <!-- Minimalist Obsidian Nose Pad -->
      <polygon points="0,32 15,22 0,16 -15,22" fill="#0f172a" stroke="#00f5ff" stroke-width="1.5" />

      <!-- Chin & Jawline -->
      <polygon points="0,70 45,95 0,120" fill="url(#slate-light)" />
      <polygon points="0,70 -45,95 0,120" fill="url(#slate-mid)" />

      <!-- Neck Armor Plates -->
      <polygon points="0,120 45,95 100,160 50,180 0,160" fill="url(#slate-mid)" />
      <polygon points="0,120 -45,95 -100,160 -50,180 0,160" fill="url(#slate-dark)" />

      <!-- Central Benchmark Telemetry Dial / Core Diamond -->
      <polygon points="0,160 50,180 0,225 -50,180" fill="#0b1120" stroke="#00f5ff" stroke-width="3" />
      <polygon points="0,172 32,185 0,215 -32,185" fill="url(#amber-core)" />
      <line x1="0" y1="185" x2="16" y2="175" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" />

      <!-- Lower Collar Plates -->
      <polygon points="0,225 50,180 80,180 50,260 0,260" fill="url(#slate-light)" />
      <polygon points="0,225 -50,180 -80,180 -50,260 0,260" fill="url(#slate-mid)" />

      <!-- Chest Keel -->
      <polygon points="0,260 50,260 0,310" fill="url(#slate-mid)" />
      <polygon points="0,260 -50,260 0,310" fill="url(#slate-dark)" />

    </g>
  </g>
</svg>`;
}

function main() {
  const outputDir = path.join(__dirname, '..', 'docs', 'images');
  fs.mkdirSync(outputDir, { recursive: true });

  const svg = buildLogoSvg();
  const svgPath = path.join(outputDir, 'logo.svg');
  const pngPath = path.join(outputDir, 'logo.png');

  fs.writeFileSync(svgPath, svg, 'utf8');

  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: 1024 }
  });
  const pngData = resvg.render().asPng();
  fs.writeFileSync(pngPath, pngData);

  console.log(`✓ Successfully rendered official logo.svg and logo.png at 1024x1024 in ${outputDir}`);
}

main();
