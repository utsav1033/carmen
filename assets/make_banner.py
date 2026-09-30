# Regenerates assets/banner.svg. Needs `pip install fonttools` and the two OFL fonts from
# https://github.com/google/fonts (ofl/spacegrotesk, ofl/jetbrainsmono) in the working directory.
import base64, io
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from fontTools import subset

TITLE = "carmen"
TAG = "let the model cook. trust nothing it can’t prove."
MONO1 = "carmy writes  →  the judge proves  →  the loop learns"
MONO2 = "apple metal · mlx · softmax · masked_softmax · open source"
CARD = ["the judge", "✓ compiles", "✓ matches float64", "✓ survives unseen inputs", "✓ faster, with a CI", "✗ everything else"]

def face(path, wght, text):
    f = TTFont(path)
    f = instancer.instantiateVariableFont(f, {"wght": wght})
    opts = subset.Options(); opts.flavor = "woff"; opts.layout_features = ["kern", "liga"]
    s = subset.Subsetter(opts); s.populate(text=text); s.subset(f)
    buf = io.BytesIO(); f.flavor = "woff"; f.save(buf)
    return base64.b64encode(buf.getvalue()).decode()

grot_b = face("SpaceGrotesk[wght].ttf", 700, TITLE)
grot_r = face("SpaceGrotesk[wght].ttf", 400, TAG)
mono = face("JetBrainsMono[wght].ttf", 500, MONO1 + MONO2 + "".join(CARD))

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 420" width="1280" height="420" role="img" aria-label="carmen: let the model cook. trust nothing it can't prove.">
<title>carmen</title>
<defs>
<style>
@font-face {{ font-family: "CG"; font-weight: 700; src: url(data:font/woff;base64,{grot_b}) format("woff"); }}
@font-face {{ font-family: "CGR"; font-weight: 400; src: url(data:font/woff;base64,{grot_r}) format("woff"); }}
@font-face {{ font-family: "CM"; font-weight: 500; src: url(data:font/woff;base64,{mono}) format("woff"); }}
.t {{ font-family: "CG", "Space Grotesk", system-ui, sans-serif; font-weight: 700; }}
.g {{ font-family: "CGR", "Space Grotesk", system-ui, sans-serif; }}
.m {{ font-family: "CM", "JetBrains Mono", ui-monospace, monospace; }}
</style>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0" stop-color="#0d0c0b"/><stop offset="1" stop-color="#17120f"/>
</linearGradient>
<radialGradient id="glow" cx="0.86" cy="0.08" r="0.62">
  <stop offset="0" stop-color="#ff7a2f" stop-opacity="0.34"/><stop offset="0.55" stop-color="#ff3d6e" stop-opacity="0.07"/><stop offset="1" stop-color="#ff3d6e" stop-opacity="0"/>
</radialGradient>
<linearGradient id="ink" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#fbf4ea"/><stop offset="1" stop-color="#ffd2ad"/>
</linearGradient>
<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
  <path d="M32 0H0V32" fill="none" stroke="#ffffff" stroke-opacity="0.035"/>
</pattern>
</defs>
<rect width="1280" height="420" rx="22" fill="url(#bg)"/>
<rect width="1280" height="420" rx="22" fill="url(#grid)"/>
<rect width="1280" height="420" rx="22" fill="url(#glow)"/>
<g transform="translate(88 0)">
  <rect x="0" y="92" width="44" height="6" rx="3" fill="#ff7a2f"/>
  <text x="-6" y="238" class="t" font-size="168" letter-spacing="-7" fill="url(#ink)">carmen</text>
  <text x="0" y="296" class="g" font-size="34" fill="#e9dfd3" fill-opacity="0.92">{TAG}</text>
  <text x="0" y="352" class="m" font-size="19" fill="#ff9a5c">{MONO1}</text>
  <text x="0" y="382" class="m" font-size="15" fill="#8d8378">{MONO2}</text>
</g>
<g transform="translate(900 92)">
  <rect width="300" height="252" rx="16" fill="#ffffff" fill-opacity="0.035" stroke="#ffffff" stroke-opacity="0.09"/>
  <circle cx="24" cy="24" r="5" fill="#ff5f57" fill-opacity="0.8"/><circle cx="42" cy="24" r="5" fill="#febc2e" fill-opacity="0.8"/><circle cx="60" cy="24" r="5" fill="#28c840" fill-opacity="0.8"/>
  <text x="24" y="66" class="m" font-size="15" fill="#8d8378">{CARD[0]}</text>
  <text x="24" y="102" class="m" font-size="17" fill="#e9dfd3"><tspan fill="#5fd38d">✓</tspan>{CARD[1][1:]}</text>
  <text x="24" y="134" class="m" font-size="17" fill="#e9dfd3"><tspan fill="#5fd38d">✓</tspan>{CARD[2][1:]}</text>
  <text x="24" y="166" class="m" font-size="17" fill="#e9dfd3"><tspan fill="#5fd38d">✓</tspan>{CARD[3][1:]}</text>
  <text x="24" y="198" class="m" font-size="17" fill="#e9dfd3"><tspan fill="#5fd38d">✓</tspan>{CARD[4][1:]}</text>
  <text x="24" y="232" class="m" font-size="15" fill="#8d8378"><tspan fill="#ff5f6e">✗</tspan>{CARD[5][1:]}</text>
</g>
</svg>'''
open(__import__("pathlib").Path(__file__).with_name("banner.svg"), "w").write(svg)
print(len(svg)//1024, "KB")
