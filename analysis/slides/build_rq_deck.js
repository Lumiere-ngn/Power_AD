// Smart-sign RQ1 / RQ2 deck. Numbers come from analysis/out/rq1.json, rq2.json (2026-11-02 run).
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/home/minh/.claude/remote/plugins/5f540301/6417af884db87022/skills/pptx/scripts/apply_theme.js");

const THEME = {
  name: "Smart-sign research",
  headFontFace: "Lora",
  bodyFontFace: "Lato",
  colors: {
    dk1: "1C2228", lt1: "FAFAFA", dk2: "5B6570", lt2: "ECEEF0",
    accent1: "B4530F", accent2: "2F5D7C", accent3: "BFBFBF", accent4: "8A939C",
    accent5: "F2E3D7", accent6: "1F2328", hlink: "2F5D7C", folHlink: "5B6570",
  },
};
const HEX = { ink: "1C2228", ink2: "5B6570", accent: "B4530F", grey: "BFBFBF", rule: "E0E0E0", tint: "ECEEF0", warm: "F2E3D7", dark: "1F2328", paper: "FAFAFA", mid: "8A939C" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Minh Nguyen";
pres.title = "Smart-sign RQ1 and RQ2 results";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "Title dark",
  background: { color: HEX.dark },
  objects: [
    { placeholder: { options: { name: "eyebrow", type: "body", x: 0.8, y: 0.8, w: 8.4, h: 0.62, fontSize: 14, color: "D9A27A", bold: true, align: "center", charSpacing: 1 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.5, w: 8.4, h: 1.85, fontSize: 32, bold: true, color: "FAFAFA", align: "center", valign: "middle", fontFace: "Lora" }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 3.45, w: 8.4, h: 0.5, fontSize: 18, color: "D5D9DD", align: "center" }, text: "" } },
    { placeholder: { options: { name: "meta", type: "body", x: 0.8, y: 4.35, w: 8.4, h: 0.4, fontSize: 14, color: "AEB5BC", align: "center" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content",
  background: { color: HEX.paper },
  objects: [
    { placeholder: { options: { name: "eyebrow", type: "body", x: 0.5, y: 0.28, w: 9.0, h: 0.32, fontSize: 14, color: C.accent1, bold: true, charSpacing: 1, valign: "middle", margin: [0, 0, 0, 0] }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.6, w: 9.0, h: 0.98, fontSize: 28, bold: true, color: C.text1, valign: "top", fontFace: "Lora", margin: [0, 0, 0, 0] }, text: "" } },
  ],
  slideNumber: { x: 9.1, y: 5.22, w: 0.5, h: 0.25, fontSize: 10, color: "8A939C", align: "right" },
});

// ---------------------------------------------------------------- helpers
const T = (o) => Object.assign({ isTextBox: true, margin: 0.1, valign: "top", color: C.text1, fontSize: 16 }, o);
function head(slide, eyebrow, title) {
  slide.addText(eyebrow.toUpperCase(), { placeholder: "eyebrow" });
  slide.addText(title, { placeholder: "title" });
}
function footnote(slide, text) {
  slide.addText(text, T({ x: 0.5, y: 5.12, w: 8.5, h: 0.3, fontSize: 10, color: C.text2, valign: "middle", margin: [0, 0, 0, 0] }));
}
function chartLabel(slide, text, x, y, w) {
  slide.addText(text, T({ x, y, w, h: 0.32, fontSize: 14, bold: true, margin: [0, 0, 0, 0], valign: "middle", objectName: "chart label" }));
}
function bullets(items, size = 15, color) {
  return items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({ text: r.text, options: Object.assign({ bullet: j === 0 ? { indent: 14 } : undefined, fontSize: size, color: color || C.text1, paraSpaceAfter: j === runs.length - 1 ? 8 : 0, breakLine: j === runs.length - 1 && i < items.length - 1 }, r.bold ? { bold: true } : {}) }));
  }).flat();
}
function box(slide, x, y, w, h, opts = {}) {
  slide.addShape(opts.rounded ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, Object.assign({ x, y, w, h, fill: { color: opts.fill || C.background2 }, line: opts.line ? opts.line : { type: "none" } }, opts.rounded ? { rectRadius: 0.08 } : {}, opts.name ? { objectName: opts.name } : {}));
}
function arrow(slide, x1, y1, x2, y2, color = C.accent1, width = 1.75) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001), line: { color, width, endArrowType: "triangle" } };
  if (x2 < x1) o.flipH = true;
  if (y2 < y1) o.flipV = true;
  slide.addShape(pres.shapes.LINE, o);
}
const barBase = (o) => Object.assign({
  barGapWidthPct: 50, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 14, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.ink,
  catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.ink, catAxisLineShow: true, catAxisLineColor: HEX.grey,
  catAxisMajorTickMark: "none", valAxisHidden: true, valAxisLineShow: false, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: false, valAxisMinVal: 0, showTitle: false,
}, o);
function table(slide, rows, opts) {
  const n = rows[0].length;
  const out = rows.map((r, i) => r.map((cell, j) => {
    const c = typeof cell === "object" ? cell : { text: String(cell) };
    const align = (opts.align || [])[j] || "left";
    const border = [{ type: "none" }, { type: "none" }, i === 0 ? { pt: 1, color: HEX.mid } : { pt: 0.75, color: HEX.rule }, { type: "none" }];
    return { text: c.text, options: Object.assign({ bold: i === 0 || j === 0, align, valign: "middle", border, fontSize: opts.fontSize || 14, color: C.text1, margin: [0.04, 0.08, 0.04, 0.08] }, (opts.fillCol != null && j === opts.fillCol && i > 0) ? { fill: { color: HEX.warm } } : {}, c.options || {}) };
  }));
  slide.addTable(out, { x: opts.x, y: opts.y, w: opts.w, colW: opts.colW, rowH: opts.rowH || 0.42, autoPage: false });
}

// ---------------------------------------------------------------- 1. title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "Title dark", sectionTitle: "Opening" });
s.addText("Propagation-aware predictive health monitoring for smart mobility IoT devices".toUpperCase(), { placeholder: "eyebrow" });
s.addText("Predicting normal behaviour and detecting energy faults in smart bus-stop signs", { placeholder: "title" });
s.addText("RQ1 and RQ2 results from seven months of fleet telemetry", { placeholder: "subtitle" });
s.addText("Minh Nguyen · November 2026", { placeholder: "meta" });
s.addNotes("Two research questions, answered with the Prometheus telemetry we already have. Linux logs are the next data source; the last slides say what they would add.");

// ---------------------------------------------------------------- 2. executive summary
s = pres.addSlide({ masterName: "Content", sectionTitle: "Opening" });
head(s, "Summary", "Executive summary");
[[0.5, "RQ1 · Normal-behaviour modelling", "Largely for power, partly for heat", [
  [{ text: "Workload context cuts input-power error from " }, { text: "1.52 W to 0.50 W", bold: true }],
  [{ text: "CPU temperature is hardest: " }, { text: "31%", bold: true }, { text: " of its variation explained, 57% with measured power" }],
  [{ text: "56 V devices err " }, { text: "3.6×", bold: true }, { text: " more than 12 V; calibrated bands cover 82–90% of hours" }],
]], [5.15, "RQ2 · Anomaly detection", "Yes, with a narrow lead over learned graphs", [
  [{ text: "The hand graph catches " }, { text: "72%", bold: true }, { text: " of injected energy faults at 1% false alarms" }],
  [{ text: "Ahead of the complete graph (69%), Lasso (66%) and the best non-graph detector (54%)" }],
  [{ text: "The lead depends on how scores are combined; cooling faults are missed" }],
]]].forEach(([x, tag, verdict, items]) => {
  box(s, x, 1.72, 4.35, 3.33, { fill: C.background2, name: "summary card" });
  s.addText(tag.toUpperCase(), T({ x: x + 0.2, y: 1.85, w: 3.95, h: 0.32, fontSize: 14, bold: true, color: C.accent1, charSpacing: 1, margin: 0 }));
  s.addText(verdict, T({ x: x + 0.2, y: 2.2, w: 3.95, h: 0.7, fontSize: 20, bold: true, fontFace: "Lora", margin: 0 }));
  s.addText(bullets(items, 14), T({ x: x + 0.2, y: 2.92, w: 3.95, h: 2.05, margin: 0 }));
});
s.addNotes("RQ1: a context-conditioned model predicts power well, heat partly, and generalises poorly to hardware it has not seen. RQ2: the hand-specified graph detects the most injected energy faults, but only narrowly ahead of a dense learned graph, and only when node scores are rescaled before they are combined.");

// ---------------------------------------------------------------- 3. problem
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Problem" });
head(s, "Problem", "109 signs stream 42 health metrics, but none of their faults is labelled");
[["109", "signs across 12 transit agencies"], ["274M", "raw samples, one every 2 minutes"], ["7", "months of history, March to October 2026"], ["0", "faults labelled in the data"]].forEach(([num, lab], i) => {
  const x = 0.5 + i * 2.3;
  box(s, x, 1.75, 2.05, 1.45, { fill: C.background2, name: "stat tile" });
  s.addText(num, T({ x: x + 0.15, y: 1.85, w: 1.75, h: 0.75, fontSize: 36, bold: true, color: C.accent1, fontFace: "Lora", margin: 0, valign: "middle" }));
  s.addText(lab, T({ x: x + 0.15, y: 2.6, w: 1.75, h: 0.55, fontSize: 14, color: C.text2, margin: 0 }));
});
[[0.5, "The fleet is mixed", "12 V and 56 V supplies, solar charge controllers on 36 signs, DeepStream video analytics on some"], [5.15, "What operators need", "To know what normal power and heat look like for each sign, and to catch energy faults before a sign stops reporting"]].forEach(([x, h, t]) => {
  s.addText([{ text: h, options: { bold: true, fontSize: 18, breakLine: true, paraSpaceAfter: 6 } }, { text: t, options: { fontSize: 16, color: C.text1 } }], T({ x, y: 3.5, w: 4.35, h: 1.45, margin: 0 }));
});
s.addNotes("Prometheus telemetry from the edge exporters on each Jetson Orin sign. Power comes from INA238 and INA3221 rail monitors, heat from SoC and board sensors, context from the heartbeat, environment and DeepStream services. Nothing marks a fault, which shapes the RQ2 evaluation.");

// ---------------------------------------------------------------- 4. research questions
s = pres.addSlide({ masterName: "Content", sectionTitle: "Problem" });
head(s, "Problem", "Research questions");
[[1.7, "RQ1", "To what extent can a model conditioned on workload and environmental context predict the expected power and thermal behaviour of smart bus stop devices across differing hardware configurations and operating conditions?", "Tested with 6 power and thermal targets, 4 context groups and 3 generalisation tests"],
 [3.42, "RQ2", "Does a hand-specified dependency graph detect energy anomalies more accurately than non-graph and learned-graph detectors?", "Tested with 10 detectors, 6 injected fault types and 66 devices"]].forEach(([y, tag, q, t]) => {
  box(s, 0.5, y, 9.0, 1.58, { fill: C.background2, name: "question band" });
  s.addText(tag, T({ x: 0.7, y: y + 0.15, w: 1.0, h: 0.6, fontSize: 28, bold: true, color: C.accent1, fontFace: "Lora", margin: 0 }));
  s.addText([{ text: q, options: { fontSize: 16, breakLine: true, paraSpaceAfter: 6 } }, { text: t, options: { fontSize: 14, color: C.text2 } }], T({ x: 1.8, y: y + 0.1, w: 7.5, h: 1.42, margin: 0 }));
});
s.addNotes("RQ1 is about the normal envelope: what power and heat a device should show in its current situation. RQ2 uses that envelope: does writing down the physical dependencies between measurements help catch energy faults?");

// ---------------------------------------------------------------- 5. RQ1 method
pres.addSection({ title: "Methodology" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Methodology" });
head(s, "Methodology · RQ1", "RQ1 conditions one fleet-wide model on four groups of context");
const ctx = [["Time", "hour of day, weekday"], ["Hardware", "supply, memory, solar, agency"], ["Environment", "temperature, humidity, light"], ["Workload", "RAM used, DeepStream, network"]];
ctx.forEach(([h, t], i) => {
  const y = 1.75 + i * 0.78;
  box(s, 0.5, y, 3.2, 0.66, { fill: C.background2, name: "context box" });
  s.addText([{ text: h, options: { bold: true, fontSize: 15, breakLine: true } }, { text: t, options: { fontSize: 14, color: C.text2 } }], T({ x: 0.62, y: y + 0.03, w: 3.0, h: 0.6, margin: 0, valign: "middle" }));
  arrow(s, 3.7, y + 0.33, 4.05, 2.95 + (i - 1.5) * 0.18, C.text2, 1.25);
});
box(s, 4.05, 2.4, 1.75, 1.1, { fill: C.accent1, name: "model box" });
s.addText([{ text: "LightGBM", options: { bold: true, fontSize: 16, color: "FFFFFF", breakLine: true } }, { text: "one model for the fleet", options: { fontSize: 14, color: "FFFFFF" } }], T({ x: 4.1, y: 2.45, w: 1.65, h: 1.0, margin: 0, align: "center", valign: "middle" }));
[["Power", "input, Orin, CPU+GPU"], ["Heat", "CPU, GPU, board"]].forEach(([h, t], i) => {
  const y = 1.95 + i * 0.85;
  box(s, 6.2, y, 3.3, 0.66, { fill: C.background2, name: "target box" });
  s.addText([{ text: h, options: { bold: true, fontSize: 15, breakLine: true } }, { text: t, options: { fontSize: 14, color: C.text2 } }], T({ x: 6.32, y: y + 0.03, w: 3.1, h: 0.6, margin: 0, valign: "middle" }));
  arrow(s, 5.8, 2.95, 6.2, y + 0.33, C.text2, 1.25);
});
s.addText([{ text: "Tested three ways", options: { bold: true, fontSize: 15, breakLine: true, paraSpaceAfter: 4 } }].concat(bullets(["Temporal: fit before 1 Aug, test to 6 Oct", "Leave-device-out: 5 folds", "Leave-configuration-out: supply, solar or agency held out"], 14)), T({ x: 6.2, y: 3.6, w: 3.3, h: 1.48, margin: 0 }));
footnote(s, "The target's own past is never an input, so the model gives expected behaviour rather than tracking the last reading. Hourly device means.");
s.addNotes("Feature ladder: time, then hardware, external environment (agency mean of other devices), local enclosure sensors, workload, and finally a per-device offset. Ridge and per-supply specialists as references. Quantile models at 5 and 95 percent give the band; split-conformal calibration on July adjusts it.");

// ---------------------------------------------------------------- 6. RQ2 graph
s = pres.addSlide({ masterName: "Content", sectionTitle: "Methodology" });
head(s, "Methodology · RQ2", "RQ2 encodes ten power and heat readings as a physical dependency graph");
const BW = 1.4, BH = 0.62, colX = [0.5, 2.167, 3.833, 5.5], rowY = [1.75, 2.95, 4.2];
const N = { cg: [0, 0, "CPU+GPU\npower", true], orin: [1, 0, "Orin power"], inp: [2, 0, "Input power"], en: [3, 0, "Energy\ncounter"], hb: [1, 1, "Heartbeat\npower"], cur: [3, 1, "Input\ncurrent"], cpu: [0, 2, "CPU temp"], gpu: [1, 2, "GPU temp"], brd: [2, 2, "Board temp"], vol: [3, 2, "Input\nvoltage", true] };
const cx = k => colX[N[k][0]] + BW / 2, top = k => rowY[N[k][1]], bot = k => rowY[N[k][1]] + BH, lft = k => colX[N[k][0]], rgt = k => colX[N[k][0]] + BW, cy = k => rowY[N[k][1]] + BH / 2;
arrow(s, rgt("cg"), cy("cg"), lft("orin"), cy("orin"));
arrow(s, rgt("orin"), cy("orin"), lft("inp"), cy("inp"));
arrow(s, rgt("inp"), cy("inp"), lft("en"), cy("en"));
arrow(s, cx("orin"), bot("orin"), cx("orin"), top("hb"));
arrow(s, rgt("inp") - 0.2, bot("inp"), lft("cur") + 0.25, top("cur"));
arrow(s, cx("vol"), top("vol"), cx("cur"), bot("cur"));
arrow(s, cx("inp"), bot("inp"), cx("brd"), top("brd"));
arrow(s, cx("cg"), bot("cg"), cx("cpu"), top("cpu"));
arrow(s, cx("cg") + 0.15, bot("cg"), lft("gpu") + 0.3, top("gpu"));
arrow(s, rgt("cpu"), cy("cpu"), lft("gpu"), cy("gpu"));
Object.entries(N).forEach(([k, [c, r, label, root]]) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: colX[c], y: rowY[r], w: BW, h: BH, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { color: HEX.mid, width: 1, dashType: root ? "dash" : "solid" }, objectName: `node ${k}` });
  s.addText(label, T({ x: colX[c], y: rowY[r], w: BW, h: BH, fontSize: 14, align: "center", valign: "middle", margin: 0.03 }));
});
s.addText(bullets([
  "I = P / V ties current to power and voltage",
  "The energy counter integrates input power",
  "Heartbeat power logs the Orin rail",
  "Compute power heats the shared CPU and GPU die",
  "Dashed: no measured parent, predicted from context only",
], 14), T({ x: 7.2, y: 1.75, w: 2.3, h: 3.3, margin: 0 }));
s.addNotes("Every node is also conditioned on the RQ1 context, so the graph only decides which other measurements a node may lean on. CPU+GPU power and input voltage have no measured parent: they are predicted from time, hardware, environment and workload alone.");

// ---------------------------------------------------------------- 7. detectors
s = pres.addSlide({ masterName: "Content", sectionTitle: "Methodology" });
head(s, "Methodology · RQ2", "Ten detectors share one model class and differ only in their graph");
[[0.5, "No graph", ["Seasonal z-score", "Context model, no edges", "Isolation Forest", "PCA reconstruction"], C.background2],
 [3.55, "Learned graph", ["Top-k correlation (GDN-style)", "Lasso neighbourhoods", "Granger causality", "NOTEARS DAG", "Complete graph"], C.background2],
 [6.6, "Hand-specified", ["Physics DAG from the previous slide"], HEX.warm]].forEach(([x, h, items, fill]) => {
  box(s, x, 1.72, 2.9, 2.45, { fill, name: "detector family" });
  s.addText([{ text: h, options: { bold: true, fontSize: 18, fontFace: "Lora", breakLine: true, paraSpaceAfter: 6 } }].concat(bullets(items, 15)), T({ x: x + 0.15, y: 1.82, w: 2.6, h: 2.3, margin: 0 }));
});
s.addText([
  { text: "Each node: ", options: { bold: true, fontSize: 14 } }, { text: "per-device ridge on its parents + LightGBM on context and parents", options: { fontSize: 14, breakLine: true } },
  { text: "Scoring: ", options: { bold: true, fontSize: 14 } }, { text: "per-device threshold at 1% hourly false alarms; episodes detected, root node named", options: { fontSize: 14 } },
], T({ x: 0.5, y: 4.3, w: 9.0, h: 0.78, margin: 0 }));
s.addNotes("The ridge part lets exact relations such as I = P / V extrapolate under a fault; trees alone cannot. Learned graphs come from the training hours after removing each device's daily cycle. Fit before 1 July, calibrate on July, test from 1 August.");

// ---------------------------------------------------------------- 8. faults
s = pres.addSlide({ masterName: "Content", sectionTitle: "Methodology" });
head(s, "Methodology · RQ2", "No faults are labelled, so six fault types were injected for testing");
table(s, [
  ["Fault", "What changes", "Root node", "Severities"],
  ["F1 Parasitic load", "Extra draw: power, current, energy, board heat rise", "Input power", "+5, 10, 20%"],
  ["F2 Compute runaway", "CPU+GPU power rises with no workload change", "CPU+GPU power", "+25, 50, 100%"],
  ["F3 Energy-counter drift", "Energy integral ramps away from power", "Energy counter", "5, 10, 20%"],
  ["F4 Stuck power sensor", "Power freezes; current and voltage move on", "Input power", "6 to 48 h"],
  ["F5 Supply sag", "Voltage drops; current rises to hold power", "Input voltage", "−3, 6, 12%"],
  ["F6 Cooling fault", "CPU and GPU heat up, power unchanged (control)", "CPU / GPU temp", "+3, 6, 12 °C"],
], { x: 0.5, y: 1.72, w: 9.0, colW: [2.35, 3.75, 1.6, 1.3], rowH: 0.46 });
footnote(s, "6 episodes per device, fault and severity, at least 48 h apart, 66 devices: about 390 episodes per scenario. Propagation slopes are within-device regressions on training hours.");
s.addNotes("F1 to F5 are energy faults and enter the headline numbers; F6 is a control case with no change in power. F3 and F4 corrupt one sensor; F1, F2 and F5 are real changes that spread to other readings.");

// ---------------------------------------------------------------- 9. RQ1 ladder
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
head(s, "Results · RQ1", "Adding workload context cuts input-power error by two-thirds, to 0.50 W");
chartLabel(s, "Input-power MAE by context added, test period", 0.5, 1.68, 6.0);
s.addChart(pres.charts.BAR, [{ name: "MAE", labels: ["Time", "+ Hardware", "+ External env", "+ Local env", "+ Workload", "+ Device offset"], values: [2.77, 1.55, 1.59, 1.52, 0.50, 0.40] }],
  barBase({ x: 0.5, y: 2.0, w: 6.0, h: 3.05, barDir: "col", chartColors: [HEX.grey, HEX.grey, HEX.grey, HEX.grey, HEX.accent, HEX.grey], dataLabelFormatCode: '0.00" W"', valAxisMaxVal: 3.2 }));
s.addText(bullets([
  [{ text: "RAM in use and DeepStream state", bold: true }, { text: " carry 73% of the model's split gain" }],
  [{ text: "Environment barely helps power: 1.55 W to 1.52 W" }],
  [{ text: "A per-device offset reaches 0.40 W, against 1.58 W for each device's hour-of-day profile" }],
], 15), T({ x: 6.8, y: 2.0, w: 2.7, h: 3.05, margin: 0 }));
footnote(s, "Hourly device means; fit before 1 Aug 2026, test 1 Aug to 6 Oct 2026, 75 devices. Each bar adds one group to the bar on its left.");
s.addNotes("Persistence, which uses the previous hour's own value, reaches 0.35 W, but a normal-behaviour model must not see the target or it would follow an anomaly. Orin and CPU+GPU power show the same pattern.");

// ---------------------------------------------------------------- 10. RQ1 targets
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
head(s, "Results · RQ1", "Power follows workload closely; CPU temperature is the hardest target");
chartLabel(s, "Within-device variation explained, test period", 0.5, 1.68, 6.0);
s.addChart(pres.charts.BAR, [{ name: "R²", labels: ["Orin power", "Board temp", "CPU+GPU power", "Input power", "GPU temp", "CPU temp"], values: [0.82, 0.81, 0.73, 0.68, 0.37, 0.31] }],
  barBase({ x: 0.5, y: 2.0, w: 6.0, h: 3.05, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [HEX.grey, HEX.grey, HEX.grey, HEX.grey, HEX.accent, HEX.accent], dataLabelFormatCode: "0%", valAxisMaxVal: 1.0 }));
s.addText(bullets([
  [{ text: "Measured CPU+GPU power", bold: true }, { text: " lifts CPU temperature to 57% and GPU to 60%" }],
  [{ text: "Board temperature follows the enclosure sensor, 71% of split gain" }],
  [{ text: "Hours with DeepStream running err 1.4× more on input power (0.44 vs 0.31 W)" }],
], 15), T({ x: 6.8, y: 2.0, w: 2.7, h: 3.05, margin: 0 }));
footnote(s, "Within-device R² of the full-context model with a device offset: 1 minus its squared error over each device's own test-period variance.");
s.addNotes("CPU temperature is driven by load and clocks that the telemetry does not report, and several devices sit near 99 °C. This is the gap the Linux logs should close.");

// ---------------------------------------------------------------- 11. RQ1 transfer
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
head(s, "Results · RQ1", "Unseen configurations raise input-power error, most for solar (2.4×)");
chartLabel(s, "Input-power MAE on unseen devices", 0.5, 1.68, 6.2);
s.addChart(pres.charts.BAR, [
  { name: "Group in training", labels: ["Solar", "Mains", "TTC", "Barcelona", "EXO", "56 V"], values: [0.59, 0.66, 0.19, 0.72, 0.48, 2.18] },
  { name: "Group held out", labels: ["Solar", "Mains", "TTC", "Barcelona", "EXO", "56 V"], values: [1.40, 1.59, 0.49, 1.28, 0.62, 2.39] },
], barBase({ x: 0.5, y: 2.0, w: 6.2, h: 3.05, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, chartColors: [HEX.grey, HEX.accent], dataLabelFormatCode: '0.0" W"', dataLabelFontSize: 12, valAxisMaxVal: 2.8, showLegend: true, legendPos: "t", legendFontSize: 14, legendFontFace: "+mn-lt", legendColor: HEX.ink }));
s.addText(bullets([
  [{ text: "Median penalty over 14 held-out groups: " }, { text: "1.3×", bold: true }, { text: " (0.7 to 5.1×)" }],
  [{ text: "Seen 56 V devices still err 3.6× more than 12 V" }],
  [{ text: "Dropping hardware flags cuts held-out solar error to 0.90 W" }],
], 15), T({ x: 7.0, y: 2.0, w: 2.5, h: 3.05, margin: 0 }));
footnote(s, "Group in training: leave-device-out, 5 folds. Group held out: every device of that supply type, solar status or agency removed from training.");
s.addNotes("New hardware types need some history of their own before the pooled model is trustworthy. The 56 V group is small and heterogeneous, which is why even leave-device-out error is high there.");

// ---------------------------------------------------------------- 12. RQ1 coverage
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
head(s, "Results · RQ1", "Calibrated 90% bands cover 82–90% of hours, but half on 56 V CPU and GPU");
chartLabel(s, "Test hours inside the 90% band", 0.5, 1.68, 6.2);
s.addChart(pres.charts.BAR, [
  { name: "Raw quantiles", labels: ["Input power", "Orin power", "CPU+GPU power", "CPU temp", "GPU temp", "Board temp"], values: [0.75, 0.74, 0.73, 0.70, 0.71, 0.68] },
  { name: "Conformal, calibrated on July", labels: ["Input power", "Orin power", "CPU+GPU power", "CPU temp", "GPU temp", "Board temp"], values: [0.83, 0.83, 0.82, 0.87, 0.88, 0.90] },
], barBase({ x: 0.5, y: 2.0, w: 6.2, h: 3.05, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, chartColors: [HEX.grey, HEX.accent], dataLabelFormatCode: "0%", dataLabelFontSize: 12, valAxisMaxVal: 1.05, showLegend: true, legendPos: "t", legendFontSize: 14, legendFontFace: "+mn-lt", legendColor: HEX.ink }));
s.addText(bullets([
  [{ text: "Raw bands cover 68–75%: August to October behaves differently from the training months" }],
  [{ text: "One fleet margin beats per-device margins (77–83%)" }],
  [{ text: "56 V devices: ", bold: true }, { text: "50% coverage for CPU and GPU temperature" }],
], 15), T({ x: 7.0, y: 2.0, w: 2.5, h: 3.05, margin: 0 }));
footnote(s, "Target coverage is 90%. Quantile LightGBM at 5% and 95% with a device offset; split-conformal margin from July residuals.");
s.addNotes("An alarm envelope that misses 10 to 18 percent of normal hours will raise false alarms unless it is recalibrated online. Coverage does not fall steadily month by month; it is short in every month.");

// ---------------------------------------------------------------- 13. RQ2 recall
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
head(s, "Results · RQ2", "The hand graph catches 72% of energy faults, more than any other detector");
chartLabel(s, "Energy-fault episodes detected, F1–F5", 0.5, 1.68, 6.2);
const rec = [["Hand-specified DAG", 0.72], ["Complete graph", 0.686], ["Lasso graph", 0.662], ["Top-k correlation", 0.613], ["Granger graph", 0.583], ["NOTEARS DAG", 0.582], ["Context model", 0.536], ["PCA", 0.493], ["Seasonal z-score", 0.48], ["Isolation Forest", 0.361]];
s.addChart(pres.charts.BAR, [{ name: "Recall", labels: rec.map(r => r[0]), values: rec.map(r => r[1]) }],
  barBase({ x: 0.5, y: 2.0, w: 6.2, h: 3.05, barDir: "bar", catAxisOrientation: "maxMin", barGapWidthPct: 40, chartColors: rec.map((r, i) => i === 0 ? HEX.accent : HEX.grey), dataLabelFormatCode: "0%", valAxisMaxVal: 0.85 }));
s.addText(bullets([
  [{ text: "+3.4 pts", bold: true }, { text: " over the complete graph (95% CI +1.1 to +6.0)" }],
  [{ text: "+5.7 pts over Lasso, +18 pts over the best detector without a graph" }],
  [{ text: "Best AUPRC too: 0.53 vs 0.51 for the complete graph" }],
], 15), T({ x: 7.0, y: 2.0, w: 2.5, h: 3.05, margin: 0 }));
footnote(s, "At 1% hourly false alarms; 6 h mean of each node's tail probability; 5,874 episodes, 66 devices; CIs from resampling devices.");
s.addNotes("The ranking within learned graphs follows density: the complete graph and Lasso, which give each node redundant measured parents, come closest. Sparse Granger and NOTEARS graphs trail by about 14 points.");

// ---------------------------------------------------------------- 14. RQ2 scoring
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
head(s, "Results · RQ2", "Without per-node rescaling the complete graph leads, 50% to 44%");
chartLabel(s, "Energy-fault episodes detected, by score rule", 0.5, 1.68, 6.2);
const sc = ["Hand-specified DAG", "Complete graph", "Lasso graph", "Context model", "Seasonal z-score"];
s.addChart(pres.charts.BAR, [
  { name: "Largest raw |z|", labels: sc, values: [0.443, 0.497, 0.441, 0.348, 0.285] },
  { name: "Largest node tail probability", labels: sc, values: [0.72, 0.686, 0.662, 0.536, 0.48] },
], barBase({ x: 0.5, y: 2.0, w: 6.2, h: 3.05, barDir: "bar", barGrouping: "clustered", catAxisOrientation: "maxMin", barGapWidthPct: 50, chartColors: [HEX.grey, HEX.accent], dataLabelFormatCode: "0%", dataLabelFontSize: 12, valAxisMaxVal: 0.85, showLegend: true, legendPos: "t", legendFontSize: 14, legendFontFace: "+mn-lt", legendColor: HEX.ink }));
s.addText(bullets([
  [{ text: "With raw |z|, the noisiest node sets each device's alarm level" }],
  [{ text: "Its context-only roots, CPU+GPU power and voltage, are the noisiest" }],
  [{ text: "Rescaling each node to its own tail adds 18–28 pts to all but Isolation Forest" }],
], 15), T({ x: 7.0, y: 2.0, w: 2.5, h: 3.05, margin: 0 }));
footnote(s, "Tail probability: each node's score ranked against that device's own clean test-period scores before taking the largest. Both use 6 h means.");
s.addNotes("This is the main caveat on RQ2: the answer to 'more accurately than learned graphs' depends on the scoring rule. Against detectors without a graph, the hand graph wins under both rules.");

// ---------------------------------------------------------------- 15. RQ2 by fault
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
head(s, "Results · RQ2", "The hand graph leads on physics-linked faults but misses cooling faults");
table(s, [
  ["Fault", "Hand DAG", "Complete", "Lasso", "Context"],
  ["F1 Parasitic load", "74%", "64%", "62%", "58%"],
  ["F2 Compute runaway", "76%", "67%", "65%", "77%"],
  ["F3 Energy-counter drift", "51%", "55%", "49%", "45%"],
  ["F4 Stuck power sensor", "83%", "82%", "82%", "23%"],
  ["F5 Supply sag", "76%", "74%", "73%", "65%"],
  ["F6 Cooling fault", "35%", "84%", "87%", "58%"],
], { x: 0.5, y: 1.72, w: 6.2, colW: [2.4, 0.95, 0.95, 0.95, 0.95], align: ["left", "right", "right", "right", "right"], fillCol: 1, rowH: 0.45 });
s.addText(bullets([
  [{ text: "Strong where a fault breaks I = P / V, the energy integral or the power-rail chain (F1, F4, F5)" }],
  [{ text: "Cooling fault: ", bold: true }, { text: "it predicts CPU temperature from power and ambient only; learned graphs compare CPU with GPU and board temperature" }],
], 15), T({ x: 7.0, y: 1.72, w: 2.5, h: 3.3, margin: 0 }));
footnote(s, "Share of episodes detected, all three severities, 6 h tail scores. Context: the context-only model with no edges, the best detector without a graph.");
s.addNotes("A board-to-SoC temperature edge is the next refinement to test. F6 is outside the energy faults and does not enter the headline recall.");

// ---------------------------------------------------------------- 16. RQ2 localisation
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
head(s, "Results · RQ2", "Hand-graph localisation beats dense graphs but trails univariate scores");
chartLabel(s, "Episodes where the reported node is the root", 0.5, 1.68, 6.2);
const loc = [["Seasonal z-score", 0.547], ["Context model", 0.511], ["Granger graph", 0.465], ["Hand-specified DAG", 0.459], ["NOTEARS DAG", 0.436], ["Top-k correlation", 0.428], ["Lasso graph", 0.34], ["Complete graph", 0.334], ["PCA", 0.255]];
s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: loc.map(r => r[0]), values: loc.map(r => r[1]) }],
  barBase({ x: 0.5, y: 2.0, w: 6.2, h: 3.05, barDir: "bar", catAxisOrientation: "maxMin", barGapWidthPct: 40, chartColors: loc.map(r => r[0] === "Hand-specified DAG" ? HEX.accent : HEX.grey), dataLabelFormatCode: "0%", valAxisMaxVal: 0.68 }));
s.addText(bullets([
  [{ text: "+12 pts over the complete graph and Lasso, level with sparse graphs" }],
  [{ text: "Stuck sensor: ", bold: true }, { text: "blames the readings that inherit the bad value (10% correct)" }],
  [{ text: "Supply sag: blames current instead of voltage (24%)" }],
], 15), T({ x: 7.0, y: 2.0, w: 2.5, h: 3.05, margin: 0 }));
footnote(s, "Reported node: largest mean node score over the episode. Isolation Forest gives one score per hour and is left out.");
s.addNotes("Univariate scores localise well because only the faulty readings move in them. A graph-aware rule, blaming a sensor when two or more of its children disagree with it, should fix the stuck-sensor case.");

// ---------------------------------------------------------------- 17. real anomalies
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
head(s, "Results · RQ2", "The clean test period already contains real energy anomalies");
[[0.5, "16 h", "on 10 devices where input power differs from voltage × current by more than 2%, against 0.3% normally"], [5.15, "2 signs", "whose supply voltage left its July behaviour for weeks, with median |z| of 130 and 9"]].forEach(([x, num, lab]) => {
  box(s, x, 1.75, 4.35, 2.25, { fill: C.background2, name: "stat card" });
  s.addText(num, T({ x: x + 0.25, y: 1.9, w: 3.9, h: 0.95, fontSize: 48, bold: true, color: C.accent1, fontFace: "Lora", margin: 0, valign: "middle" }));
  s.addText(lab, T({ x: x + 0.25, y: 2.85, w: 3.9, h: 1.08, fontSize: 16, margin: 0 }));
});
s.addText([{ text: "Hourly detectors flag at most 2 of the 16 mismatch hours, ", options: { bold: true, fontSize: 16 } }, { text: "because short events average out within the hour. The voltage shifts forced per-device thresholds.", options: { fontSize: 16 } }], T({ x: 0.5, y: 4.2, w: 9.0, h: 0.85, margin: 0 }));
s.addNotes("These are the first candidate real labels. Kernel logs for undervoltage and rail-monitor errors on those devices would confirm or rule them out.");

// ---------------------------------------------------------------- 18. threats
pres.addSection({ title: "Threats to validity" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Threats to validity" });
head(s, "Threats to validity", "Threats to validity");
[[0.5, 1.72, "Internal", ["Faults are synthetic and spread along relations the hand graph encodes", "Thresholds come from clean test hours: recall is optimistic, comparisons stay fair"]],
 [5.15, 1.72, "Construct", ["Hourly means hide minute-scale events (2 of 16 real mismatch hours flagged)", "“Energy anomaly” is defined by six injected fault types"]],
 [0.5, 3.42, "External", ["One vendor, 66 to 97 devices, March to October 2026, no winter", "Only 5 test devices on 56 V"]],
 [5.15, 3.42, "Conclusion", ["The lead over the complete graph is 3.4 pts and reverses under raw |z|", "One temporal split; no neural learned-graph baseline such as GDN"]]].forEach(([x, y, h, items]) => {
  box(s, x, y, 4.35, 1.55, { fill: C.background2, name: "threat card" });
  s.addText([{ text: h, options: { bold: true, fontSize: 17, fontFace: "Lora", breakLine: true, paraSpaceAfter: 4 } }].concat(bullets(items, 14)), T({ x: x + 0.15, y: y + 0.1, w: 4.05, h: 1.4, margin: 0 }));
});
s.addNotes("The learned graphs recover most of the physical edges, which limits how much the synthetic propagation favours the hand graph, but it does not remove the bias. Real labels from the logs are the fix.");

// ---------------------------------------------------------------- 19. Linux logs
pres.addSection({ title: "Future work" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Future work" });
head(s, "Future work", "The Linux logs would fill the gaps this telemetry leaves");
table(s, [
  ["Gap found", "Log signal", "Helps"],
  ["CPU temperature only 31% explained", "CPU and GPU load, DVFS clocks (tegrastats)", "RQ1, RQ2 roots"],
  ["Cooling faults look like load", "Throttle events, fan PWM, power mode", "RQ2 cooling"],
  ["No real fault labels", "Undervoltage, INA238 and INA3221 errors, charger events", "RQ2 ground truth"],
  ["Level shifts treated as drift", "Reboots, service restarts, software updates", "RQ1 calibration"],
], { x: 0.5, y: 1.75, w: 9.0, colW: [3.1, 4.0, 1.9], rowH: 0.55 });
footnote(s, "Load and clocks would also give CPU+GPU power real parents in the RQ2 graph instead of context alone.");
s.addNotes("Priority order: load and clocks first, because they address the weakest RQ1 target and the noisiest RQ2 root node at once.");

// ---------------------------------------------------------------- 20. future work
s = pres.addSlide({ masterName: "Content", sectionTitle: "Future work" });
head(s, "Future work", "Future work");
[["Refine the hand graph", "add a board-to-SoC temperature edge for cooling faults"],
 ["Graph-aware root cause", "blame a sensor when two or more of its children disagree with it"],
 ["Online recalibration", "adaptive conformal bands that restart after known changes"],
 ["Neural learned-graph baseline", "run GDN on the same faults and thresholds"],
 ["Validate on real faults", "labels from the Linux logs, then a live alarm trial"]].forEach(([h, t], i) => {
  const y = 1.75 + i * 0.66;
  s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.05, w: 0.46, h: 0.46, fill: { color: C.accent1 }, line: { type: "none" }, objectName: "step marker" });
  s.addText(String(i + 1), T({ x: 0.5, y: y + 0.05, w: 0.46, h: 0.46, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 }));
  s.addText([{ text: h + ": ", options: { bold: true, fontSize: 16 } }, { text: t, options: { fontSize: 16 } }], T({ x: 1.15, y, w: 8.35, h: 0.56, margin: 0, valign: "middle" }));
});
footnote(s, "Ordered by expected effect on the two research questions.");
s.addNotes("Items 1 and 2 are small changes to the existing RQ2 code. Item 3 addresses the RQ1 coverage gap. Items 4 and 5 strengthen the claims against the threats on the previous slides.");

(async () => {
  const out = "/tmp/claude-1017/-home-minh-buspas/66939c2b-b3c6-408a-9d37-6f279e7471ef/scratchpad/deck/smart_sign_rq.pptx";
  await pres.writeFile({ fileName: out, compression: true });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
