const pptxgen = require("pptxgenjs");
const fs = require("node:fs");
const JSZip = require("jszip");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "ASTR GPU Porting Project";
pptx.company = "ASTR";
pptx.subject = "CUDA Fortran GPU porting progress between 47737cf and 63fe7a8b";
pptx.title = "ASTR GPU 移植阶段进展";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Noto Sans CJK SC",
  bodyFontFace: "Noto Sans CJK SC",
  lang: "zh-CN",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

const ROOT = "/home/dell/workspace/astr_gpu";
const ASSET = `${ROOT}/documents/presentations/assets/astr_gpu_progress`;
const OUTPUT = `${ROOT}/documents/presentations/ASTR_GPU_PROGRESS_47737cf_to_63fe7a8_20260903.pptx`;
const NOTES_OUTPUT = `${ROOT}/documents/presentations/ASTR_GPU_PROGRESS_47737cf_to_63fe7a8_20260903_SPEAKER_NOTES.md`;
const speakerNotes = [];

const C = {
  bg: "F7F8F6",
  paper: "FFFFFF",
  ink: "1D2A30",
  muted: "66747C",
  line: "CCD3D6",
  light: "E8ECEE",
  blue: "2F6FA3",
  blueLight: "DDEAF3",
  red: "C84747",
  redLight: "F5E1E1",
  teal: "2A8C82",
  tealLight: "DDEDE9",
  amber: "B7832F",
  amberLight: "F4EBD9",
  dark: "17242A",
};
const FONT = "Noto Sans CJK SC";
const MONO = "DejaVu Sans Mono";
const W = 13.333;
const H = 7.5;

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: opts.fontFace || FONT,
    fontSize: opts.fontSize || 18,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    margin: opts.margin === undefined ? 0 : opts.margin,
    valign: opts.valign || "mid",
    align: opts.align || "left",
    breakLine: false,
    fit: "shrink",
    ...opts,
  });
}

function addTitle(slide, title, kicker, number) {
  slide.background = { color: C.bg };
  addText(slide, kicker.toUpperCase(), 0.62, 0.28, 3.8, 0.22, {
    fontSize: 9.5, color: C.red, bold: true, charSpacing: 1.5,
  });
  addText(slide, title, 0.62, 0.52, 12.0, 0.54, {
    fontSize: 28, color: C.ink, bold: true,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.0, y: 0.0, w: 0.16, h: H,
    fill: { color: number % 2 === 0 ? C.blue : C.red },
    line: { color: number % 2 === 0 ? C.blue : C.red },
  });
  addText(slide, String(number).padStart(2, "0"), 12.25, 0.32, 0.45, 0.25, {
    fontSize: 10, color: C.muted, bold: true, align: "right",
  });
}

function addFooter(slide, source) {
  slide.addShape(pptx.ShapeType.line, {
    x: 0.62, y: 7.08, w: 12.05, h: 0,
    line: { color: C.line, width: 0.65 },
  });
  addText(slide, source, 0.62, 7.13, 12.05, 0.18, {
    fontSize: 7.5, color: C.muted, valign: "top",
  });
}

function addNotes(slide, notes) {
  speakerNotes.push(notes);
}

async function fixPresentationElementOrder(fileName) {
  const archive = await JSZip.loadAsync(fs.readFileSync(fileName));
  const entry = archive.file("ppt/presentation.xml");
  let xml = await entry.async("string");
  const notesMaster = xml.match(/<p:notesMasterIdLst>[\s\S]*?<\/p:notesMasterIdLst>/);

  if (notesMaster) {
    xml = xml.replace(notesMaster[0], "");
    xml = xml.replace("<p:sldIdLst>", `${notesMaster[0]}<p:sldIdLst>`);
    archive.file("ppt/presentation.xml", xml);
    const repaired = await archive.generateAsync({
      type: "nodebuffer",
      compression: "DEFLATE",
    });
    fs.writeFileSync(fileName, repaired);
  }
}

function addStat(slide, x, y, w, value, label, color = C.blue, detail = "") {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h: 1.18,
    fill: { color: C.paper },
    line: { color: C.line, width: 0.7 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w: 0.08, h: 1.18,
    fill: { color }, line: { color },
  });
  addText(slide, value, x + 0.22, y + 0.12, w - 0.34, 0.48, {
    fontSize: 28, bold: true, color,
  });
  addText(slide, label, x + 0.22, y + 0.61, w - 0.34, 0.27, {
    fontSize: 12.5, bold: true, color: C.ink,
  });
  if (detail) {
    addText(slide, detail, x + 0.22, y + 0.89, w - 0.34, 0.18, {
      fontSize: 8.5, color: C.muted,
    });
  }
}

function addLabel(slide, text, x, y, w, color = C.blue, fill = C.blueLight) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h: 0.36,
    rectRadius: 0.06,
    fill: { color: fill },
    line: { color: fill },
  });
  addText(slide, text, x + 0.08, y + 0.03, w - 0.16, 0.29, {
    fontSize: 10.5, color, bold: true, align: "center",
  });
}

function addBand(slide, x, y, w, h, heading, body, color = C.blue, fill = C.paper) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: fill },
    line: { color: C.line, width: 0.7 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w: 0.08, h,
    fill: { color }, line: { color },
  });
  addText(slide, heading, x + 0.22, y + 0.10, w - 0.34, 0.30, {
    fontSize: 15, bold: true, color,
  });
  addText(slide, body, x + 0.22, y + 0.44, w - 0.34, h - 0.54, {
    fontSize: 11.5, color: C.ink, valign: "top", breakLine: false,
  });
}

function addArrow(slide, x, y, w, h, color = C.muted) {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h,
    line: { color, width: 1.5, endArrowType: "triangle" },
  });
}

function addCheck(slide, x, y, text, color = C.teal, fontSize = 13) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y: y + 0.03, w: 0.22, h: 0.22,
    fill: { color }, line: { color },
  });
  addText(slide, "✓", x + 0.01, y + 0.015, 0.20, 0.22, {
    fontSize: 11, bold: true, color: C.paper, align: "center",
  });
  addText(slide, text, x + 0.32, y, 5.5, 0.30, {
    fontSize, color: C.ink,
  });
}

// 1. Title
{
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  slide.addImage({
    path: `${ASSET}/curvilinear_mesh.jpeg`,
    x: 7.05, y: 0, w: 6.283, h: 7.5,
    transparency: 8,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: 7.55, h: 7.5,
    fill: { color: C.dark, transparency: 0 },
    line: { color: C.dark },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.85, y: 0, w: 1.1, h: 7.5,
    fill: { color: C.dark, transparency: 22 },
    line: { color: C.dark, transparency: 100 },
  });
  addText(slide, "ASTR GPU 移植", 0.82, 1.42, 5.9, 0.85, {
    fontSize: 34, color: C.paper, bold: true,
  });
  addText(slide, "从纯 CPU Fortran/MPI\n到曲线网格选择性 Roe", 0.82, 2.28, 5.9, 0.82, {
    fontSize: 20, color: "D7E3E8", bold: true, valign: "top",
  });
  addText(slide, "commit/47737cf9b6a25b0  →  commit/63fe7a8b4a45b17", 0.82, 3.20, 5.9, 0.40, {
    fontSize: 12, color: "AFC2CB", fontFace: MONO,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.82, y: 4.08, w: 4.75, h: 0.62,
    fill: { color: C.red }, line: { color: C.red },
  });
  addText(slide, "组会进展汇报  |  2026-09-03", 1.05, 4.17, 4.30, 0.40, {
    fontSize: 15, color: C.paper, bold: true,
  });
  addText(slide, "CUDA Fortran · MPI · 显式高阶格式 · 多算例验证", 0.82, 5.27, 5.9, 0.35, {
    fontSize: 14, color: "AFC2CB",
  });
  addText(slide, "结论范围：GPU 数值等价性与工程可行性，不替代物理收敛验证", 0.82, 6.15, 5.95, 0.45, {
    fontSize: 11.5, color: "E7C7C7",
  });
  addNotes(slide, "本次汇报以 47737cf 的纯 CPU Fortran/MPI 程序为起点，以 63fe7a8 的曲线网格选择性 Roe GPU 路径为端点。主线包括 CUDA 后端建立、变量常驻、多方向 MPI、多类边界、激波捕捉、高超声速边界层和静态曲线网格。性能结论同时使用当前重复测试与有明确限制的历史数据。");
}

// 2. Executive summary
{
  const slide = pptx.addSlide();
  addTitle(slide, "从纯 CPU 基线到 GPU 求解主线", "Executive summary", 2);
  addStat(slide, 0.72, 1.33, 2.85, "11", "新增提交", C.red, "2026-06-20 CPU 基线至 09-03");
  addStat(slide, 3.78, 1.33, 2.85, "+31,934", "新增行", C.blue, "219 个变更文件；−2,720 行");
  addStat(slide, 6.84, 1.33, 2.85, "137", "验证矩阵 pass 记录", C.teal, "当前 GPU_VALIDATION_MATRIX");
  addStat(slide, 9.90, 1.33, 2.75, "37.84×", "单 GPU 整体加速", C.red, "128³ channel, 100 steps");

  addBand(slide, 0.72, 2.92, 3.82, 2.95, "架构", "GPU 主循环内变量常驻\nCPU/GPU 代码边界清晰\nHaloTransport 语义与后端分层\n文件输出仍由 CPU 管理", C.blue);
  addBand(slide, 4.76, 2.92, 3.82, 2.95, "能力", "x/y/z 与组合 MPI 拓扑\n周期、外推、对称、壁面、远场\nChannel/LDC/RTI 与高超声速边界层\nDucros + selective Roe MP7", C.red);
  addBand(slide, 8.80, 2.92, 3.82, 2.95, "证据", "逐场同相位 CPU/GPU 比较\n统计量、壁面不变量和精确解\nCompute Sanitizer 0 error\nNsight 证明 RK 区间无整场往返", C.teal);

  addText(slide, "核心变化：项目已从“单算例移植”转为“受能力门控、可回归、可扩展的 GPU 求解框架”。", 0.92, 6.25, 11.8, 0.46, {
    fontSize: 18, color: C.ink, bold: true, align: "center",
  });
  addFooter(slide, "Evidence: git diff 47737cf..63fe7a8b; documents/GPU_VALIDATION_MATRIX.md; fresh channel benchmark 2026-09-03");
  addNotes(slide, "137 是验证矩阵中 status=pass 的记录数，不等同于 137 个独立物理算例。37.84 倍采用 CPU NP=1 单次基线与 GPU NP=1 三次中位数，后面单独说明测量限制。");
}

// 3. Baseline vs endpoint
{
  const slide = pptx.addSlide();
  addTitle(slide, "起点与当前端点", "Baseline comparison", 3);

  addLabel(slide, "47737cf · 2026-06-20", 0.78, 1.25, 2.35, C.red, C.redLight);
  addText(slide, "纯 CPU Fortran/MPI 基线", 0.78, 1.72, 5.45, 0.48, {
    fontSize: 23, bold: true, color: C.red,
  });
  addCheck(slide, 0.84, 2.39, "src/ 单一求解后端；无 src_gpu/", C.red);
  addCheck(slide, 0.84, 2.86, "CPU RK 主循环与高阶差分/滤波", C.red);
  addCheck(slide, 0.84, 3.33, "CPU MPI halo、统计与 HDF5 I/O", C.red);
  addCheck(slide, 0.84, 3.80, "完整原始数值与物理模块作为 CPU oracle", C.red);
  addText(slide, "GPU 状态", 0.84, 4.48, 1.20, 0.30, { fontSize: 13, bold: true, color: C.muted });
  addText(slide, "没有 CUDA kernel、设备字段生命周期、GPU halo 后端或 CPU/GPU 能力门控。", 0.84, 4.84, 5.15, 0.72, {
    fontSize: 14, color: C.ink, valign: "top",
  });

  addArrow(slide, 6.08, 3.45, 1.05, 0, C.muted);

  addLabel(slide, "63fe7a8b · 2026-09-03", 7.30, 1.25, 2.35, C.blue, C.blueLight);
  addText(slide, "可扩展显式全 GPU 主线", 7.30, 1.72, 5.25, 0.48, {
    fontSize: 23, bold: true, color: C.blue,
  });
  addCheck(slide, 7.36, 2.39, "GPU-resident RK 主循环与能力门控", C.teal);
  addCheck(slide, 7.36, 2.86, "x/y/z、二维平面与 2×2×2 拓扑", C.teal);
  addCheck(slide, 7.36, 3.33, "多类边界、源项、统计与初始化", C.teal);
  addCheck(slide, 7.36, 3.80, "WENO7 / MP7 / Ducros / selective Roe", C.teal);
  addCheck(slide, 7.36, 4.27, "静态曲线网格 CURVE-C0 至 C9", C.teal);
  addText(slide, "边界", 7.36, 4.92, 0.85, 0.30, { fontSize: 13, bold: true, color: C.muted });
  addText(slide, "仍限定单组分、无 RANS/LES/化学、无 compact、无 IBM、无 GPU HDF5。", 7.36, 5.28, 5.05, 0.72, {
    fontSize: 14, color: C.ink, valign: "top",
  });

  addFooter(slide, "Source: git trees at 47737cf and 63fe7a8b; documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md");
  addNotes(slide, "左侧 47737cf 的仓库顶层没有 src_gpu 目录，是纯 CPU Fortran/MPI 程序。右侧端点不是完整 ASTR 全物理移植，而是受控的单组分显式 GPU 路线；未验证组合由能力门控拒绝。CPU 路径继续承担数值 oracle 和未移植物理的生产入口。");
}

// 4. Commit timeline
{
  const slide = pptx.addSlide();
  addTitle(slide, "11 次提交形成五个能力台阶", "Commit timeline", 4);
  slide.addShape(pptx.ShapeType.line, {
    x: 1.05, y: 3.20, w: 11.15, h: 0,
    line: { color: C.line, width: 3 },
  });
  const items = [
    { x: 1.25, date: "07-03", commits: "df1961b · 5597e69 · 7a5fdce", title: "GPU 主线起步", body: "TGV CUDA 主循环\nx/y/z halo + 工程基线", color: C.blue },
    { x: 3.62, date: "07-03", commits: "fd250ab · b5ed8c3", title: "非 TGV 扩展", body: "2dvort · HIT\nzero-extrap 边界", color: C.teal },
    { x: 6.19, date: "07-04", commits: "451fe0e · a684edb", title: "壁面与多算例", body: "Channel · wall family\nLDC · RTI · source", color: C.red },
    { x: 8.76, date: "07-11 / 07-12", commits: "270be49 · 16ded59", title: "激波与高超声速", body: "Sod/Shu-Osher\nHBL · NSCBC · Roe", color: C.amber },
    { x: 11.33, date: "07-22 / 09-03", commits: "aecb18f · 63fe7a8", title: "曲线网格收口", body: "CURVE-C0…C9\nwall/filter/sensor", color: C.blue },
  ];
  items.forEach((item, idx) => {
    slide.addShape(pptx.ShapeType.ellipse, {
      x: item.x - 0.12, y: 3.08, w: 0.24, h: 0.24,
      fill: { color: item.color }, line: { color: item.color },
    });
    const top = idx % 2 === 0;
    const boxY = top ? 1.35 : 3.65;
    const lineY = top ? 2.76 : 3.33;
    slide.addShape(pptx.ShapeType.line, {
      x: item.x, y: lineY, w: 0, h: top ? 0.32 : 0.32,
      line: { color: item.color, width: 1.3 },
    });
    addText(slide, item.date, item.x - 0.82, boxY, 1.64, 0.27, {
      fontSize: 11, color: item.color, bold: true, align: "center",
    });
    addText(slide, item.commits, item.x - 0.94, boxY + 0.31, 1.88, 0.30, {
      fontSize: 7.5, color: C.muted, fontFace: MONO, align: "center",
    });
    addText(slide, item.title, item.x - 1.02, boxY + 0.68, 2.04, 0.36, {
      fontSize: 14, color: C.ink, bold: true, align: "center",
    });
    addText(slide, item.body, item.x - 1.02, boxY + 1.08, 2.04, 0.72, {
      fontSize: 11, color: C.ink, align: "center", valign: "top",
    });
  });
  addText(slide, "演进策略：每次只扩大一个可验证维度，并保留旧算例与旧拓扑回归。", 1.0, 6.36, 11.3, 0.40, {
    fontSize: 18, bold: true, align: "center", color: C.ink,
  });
  addFooter(slide, "Source: git log --reverse 47737cf..63fe7a8b and per-commit shortstat");
  addNotes(slide, "时间线从第一个 CUDA 提交 df1961bc 开始，按能力合并而不是逐条读 commit。每个台阶都经历 CPU/GPU 同输入比较、MPI 拓扑扩展和回归。最后一段跨到 9 月，是曲线网格与选择性 Roe 的收口。");
}

// 5. Architecture
{
  const slide = pptx.addSlide();
  addTitle(slide, "从 CPU 单后端重构为 CPU/GPU 后端边界", "Architecture", 5);

  addBand(slide, 0.75, 1.42, 2.25, 4.72, "CPU 前端", "输入解析\n网格/物性初始化\nMPI 拓扑\n文件输出与 checkpoint\nCPU reference solver", C.red, C.redLight);
  addArrow(slide, 3.12, 3.38, 0.72, 0, C.muted);
  addBand(slide, 3.98, 1.42, 2.25, 4.72, "GPU facade", "gpu_runtime\n能力门控\n设备字段生命周期\n输出边界桥接\n错误检查", C.blue, C.blueLight);
  addArrow(slide, 6.35, 3.38, 0.72, 0, C.muted);
  addBand(slide, 7.21, 1.42, 2.25, 4.72, "CUDA 后端", "mainloop_gpu\nsolver / gradcal\nfilter / boundary\nstatistics / source\nshock sensor", C.teal, C.tealLight);
  addArrow(slide, 9.58, 3.38, 0.72, 0, C.muted);
  addBand(slide, 10.44, 1.42, 2.15, 4.72, "传输后端", "HaloTransport 语义\nL0 host-staged\n未来 pinned/nonblocking\nCUDA-aware MPI\nHIP/DCU 后端", C.amber, C.amberLight);

  addText(slide, "src/", 1.50, 5.56, 0.74, 0.25, { fontSize: 12, fontFace: MONO, color: C.red, bold: true, align: "center" });
  addText(slide, "src_gpu/", 7.89, 5.56, 0.92, 0.25, { fontSize: 12, fontFace: MONO, color: C.teal, bold: true, align: "center" });
  addText(slide, "求解器 kernel 不直接依赖 MPI 后端；CUDA 目录不向 CPU 数值模块反向渗透。", 0.92, 6.46, 11.8, 0.36, {
    fontSize: 17, bold: true, align: "center",
  });
  addFooter(slide, "Source: docs/adr/0013-0020; documents/ASTR_GPU_HALOTRANSPORT_SKETCH.md; src/CMakeLists.txt");
  addNotes(slide, "当前真正实现的是 CUDA Fortran 后端和 L0 host-staged 传输。HIP/DCU 是接口设计目标，不是已完成能力。文件输出继续作为 CPU-owned boundary，防止现阶段把 HDF5 迁移混入求解核心。");
}

// 6. Resident loop
{
  const slide = pptx.addSlide();
  addTitle(slide, "RK 计算循环内由 GPU 持有权威状态", "Device residency", 6);

  const stages = [
    ["边界 / halo", C.red], ["显式滤波", C.blue], ["gradcal", C.teal],
    ["convection", C.blue], ["diffusion / source", C.red], ["RK update", C.teal],
  ];
  stages.forEach((entry, idx) => {
    const x = 0.78 + idx * 2.02;
    slide.addShape(pptx.ShapeType.roundRect, {
      x, y: 2.25, w: 1.58, h: 0.78,
      rectRadius: 0.06,
      fill: { color: idx % 2 === 0 ? C.paper : C.light },
      line: { color: entry[1], width: 1.4 },
    });
    addText(slide, entry[0], x + 0.10, 2.39, 1.38, 0.48, {
      fontSize: 12.5, color: entry[1], bold: true, align: "center",
    });
    if (idx < stages.length - 1) addArrow(slide, x + 1.62, 2.64, 0.38, 0, C.muted);
  });
  slide.addShape(pptx.ShapeType.arc, {
    x: 1.45, y: 1.72, w: 10.15, h: 2.35,
    adjustPoint: 0.3,
    rotate: 180,
    fill: { color: C.bg, transparency: 100 },
    line: { color: C.muted, width: 1.3, endArrowType: "triangle" },
  });
  addText(slide, "q / primitive / metrics / work arrays 常驻 device", 3.95, 1.42, 5.55, 0.40, {
    fontSize: 18, bold: true, color: C.ink, align: "center",
  });

  addBand(slide, 0.78, 4.18, 3.55, 1.55, "进入循环前", "网格、度量、边界面法向、初始场一次性上传", C.blue);
  addBand(slide, 4.55, 4.18, 3.55, 1.55, "循环内部", "每个 kernel 后显式同步；只交换 halo 与小型统计规约", C.red);
  addBand(slide, 8.32, 4.18, 4.20, 1.55, "输出边界", "需要统计或写场时才回到 host；HDF5 仍非 GPU 迁移目标", C.teal);

  addStat(slide, 2.08, 5.80, 2.85, "176 B", "C9 最大 H2D", C.blue, "首个特征 RHS 之后");
  addStat(slide, 5.24, 5.80, 2.85, "6144 B", "C9 最大 D2H", C.red, "统计规约");
  addStat(slide, 8.40, 5.80, 2.85, "0", "≥64 KiB 传输", C.teal, "无整场往返");
  addFooter(slide, "Source: src_gpu/mainloop_gpu.cuf; documents/ASTR_GPU_DEVICE_FIELD_OWNERSHIP.md; CURVE-C9 Nsight report");
  addNotes(slide, "常驻的含义不是完全零传输。当前仍有 halo 打包/解包和标量或 reduction 数据移动，但没有每个 kernel 后整场 q 往返。显式同步是项目已确认的第一阶段约束，后续优化必须单独决策。");
}

// 7. MPI halo
{
  const slide = pptx.addSlide();
  addTitle(slide, "MPI 正确性已覆盖三方向，通信优化尚未完成", "HaloTransport", 7);

  const cols = [0.80, 4.02, 7.24, 10.46];
  const heads = ["语义操作", "宽度", "字段", "当前状态"];
  heads.forEach((h, i) => addText(slide, h, cols[i], 1.35, i === 0 ? 2.8 : 2.1, 0.35, {
    fontSize: 13, bold: true, color: C.paper, align: i === 0 ? "left" : "center",
    fill: { color: C.dark }, margin: 0.08,
  }));
  const rows = [
    ["solution qswap", "hm+1", "q(1:5)", "interface averaging"],
    ["filter", "hm", "q / qwork", "dataswap semantics"],
    ["diffusion", "hm", "sigma / qflux", "generic field halo"],
    ["shock sensor", "hm", "raw ssf", "mask local expansion"],
  ];
  rows.forEach((row, r) => {
    const y = 1.76 + r * 0.62;
    row.forEach((value, i) => addText(slide, value, cols[i], y, i === 0 ? 2.8 : 2.1, 0.47, {
      fontSize: 11.5, color: i === 1 ? C.red : C.ink,
      align: i === 0 ? "left" : "center",
      fill: { color: r % 2 === 0 ? C.paper : C.light }, margin: 0.08,
      fontFace: i === 1 ? MONO : FONT,
    }));
  });

  addText(slide, "已验证拓扑", 0.82, 4.55, 1.45, 0.30, { fontSize: 14, bold: true, color: C.ink });
  const topo = ["2×1×1", "1×2×1", "1×1×2", "2×2×1", "2×1×2", "1×2×2", "2×2×2"];
  topo.forEach((t, i) => addLabel(slide, t, 0.82 + i * 1.67, 5.02, 1.34, i < 3 ? C.blue : C.teal, i < 3 ? C.blueLight : C.tealLight));

  addBand(slide, 0.82, 5.72, 3.55, 0.88, "L0 · 当前", "blocking host-staged，作为跨平台正确性基线", C.red);
  addArrow(slide, 4.50, 6.14, 0.54, 0, C.muted);
  addBand(slide, 5.17, 5.72, 3.15, 0.88, "L1/L2 · 下一步", "nonblocking + pinned staging", C.blue);
  addArrow(slide, 8.45, 6.14, 0.54, 0, C.muted);
  addBand(slide, 9.12, 5.72, 3.40, 0.88, "L3/L4 · 生产", "CUDA/HIP-aware MPI + topology aware", C.teal);

  addFooter(slide, "Source: src_gpu/halo_exchange_gpu.cuf; documents/ASTR_GPU_HALOTRANSPORT_SKETCH.md");
  addNotes(slide, "NP=4/8/27 等在两张 GPU 上主要用于正确性，不是性能扩展证据。通信宽度和 interface averaging 是数值语义，不能为了统一 API 而隐藏。当前最明确的性能瓶颈是 L0 host-staged 加每 kernel 同步。");
}

// 8. Numerical methods
{
  const slide = pptx.addSlide();
  addTitle(slide, "GPU 主线采用局部显式高阶格式", "Numerical stack", 8);

  addText(slide, "六阶中心差分", 0.82, 1.35, 3.35, 0.36, { fontSize: 17, bold: true, color: C.blue });
  addText(slide, "f′ᵢ ≈ [45(fᵢ₊₁−fᵢ₋₁) − 9(fᵢ₊₂−fᵢ₋₂) + (fᵢ₊₃−fᵢ₋₃)] / (60Δx)", 0.82, 1.83, 5.70, 0.72, {
    fontSize: 15, fontFace: "Cambria Math", color: C.ink, valign: "top",
  });
  addText(slide, "实现：derivative:diff6ec → gradcal / diffusion / central convection", 0.82, 2.62, 5.65, 0.34, {
    fontSize: 10.5, fontFace: MONO, color: C.muted,
  });

  addText(slide, "十阶显式中心滤波", 7.00, 1.35, 3.55, 0.36, { fontSize: 17, bold: true, color: C.red });
  addText(slide, "f̃ᵢ = Σₘ₌₀⁵ cₘ(fᵢ₋ₘ + fᵢ₊ₘ)", 7.00, 1.83, 4.70, 0.62, {
    fontSize: 20, fontFace: "Cambria Math", color: C.ink,
  });
  addText(slide, "实现：filterq_explicit10；全 q ping-pong；物理边界 0-6-6-6-8-10 闭合", 7.00, 2.62, 5.45, 0.45, {
    fontSize: 10.5, fontFace: MONO, color: C.muted,
  });

  const methods = [
    ["643e", "中心对流 + 梯度/扩散", "PORTED", C.blue],
    ["543e", "Steger-Warming + WENO7 / MP7", "PORTED", C.teal],
    ["Roe", "Ducros 触发的特征空间 MP7", "PORTED", C.red],
    ["RK3", "显式三阶时间推进", "PORTED", C.blue],
    ["643c / 543c", "compact + 三/五对角线性求解", "EXCLUDED", C.muted],
  ];
  methods.forEach((row, i) => {
    const y = 3.45 + i * 0.57;
    addText(slide, row[0], 0.86, y, 1.35, 0.40, { fontSize: 13, fontFace: MONO, bold: true, color: row[3] });
    addText(slide, row[1], 2.34, y, 6.55, 0.40, { fontSize: 13, color: C.ink });
    addLabel(slide, row[2], 10.45, y + 0.02, 1.55, row[3], row[2] === "EXCLUDED" ? C.light : (row[3] === C.red ? C.redLight : row[3] === C.teal ? C.tealLight : C.blueLight));
  });
  addText(slide, "原则：只迁移可局部 stencil 化、可做 halo 分解且已建立 CPU oracle 的格式。", 0.88, 6.47, 11.9, 0.34, {
    fontSize: 17, bold: true, align: "center",
  });
  addFooter(slide, "Source: documents/ASTR_CPU_NUMERICAL_SCHEMES.md; src/derivative.F90; src/filter.F90; src_gpu/solver_gpu.cuf");
  addNotes(slide, "这里的 643e 数字标签不能机械理解为所有边界点都是六阶。CPU 实际物理边界闭合更低阶。当前 GPU 目标明确不包含 compact，因为每条网格线求解线性系统在现有实现上效率较低且通信复杂。");
}

// 9. Case coverage
{
  const slide = pptx.addSlide();
  addTitle(slide, "算例扩展遵循“先隔离模块，再组合物理”", "Case coverage", 9);
  const groups = [
    { x: 0.75, y: 1.35, w: 3.00, h: 2.18, title: "周期光滑流", color: C.blue, cases: ["TGV", "2dvort (3-D extruded)", "HIT / ABC velocity"] },
    { x: 3.94, y: 1.35, w: 3.00, h: 2.18, title: "壁面与源项", color: C.teal, cases: ["Channel flow", "Lid-Driven Cavity", "Rayleigh–Taylor"] },
    { x: 7.13, y: 1.35, w: 2.60, h: 2.18, title: "激波格式", color: C.red, cases: ["Sod", "Shu–Osher", "OpenShock gates"] },
    { x: 9.92, y: 1.35, w: 2.66, h: 2.18, title: "高超声速", color: C.amber, cases: ["Mach 3/5 HBL", "oblique shock", "controlled SBLI"] },
  ];
  groups.forEach(g => {
    addBand(slide, g.x, g.y, g.w, g.h, g.title, g.cases.map(s => `✓ ${s}`).join("\n"), g.color);
  });

  addText(slide, "能力组合路径", 0.80, 3.98, 1.45, 0.35, { fontSize: 14, bold: true, color: C.ink });
  const chain = [
    ["TGV", "主循环"], ["2dvort / HIT", "去 TGV 专用"], ["Channel / LDC / RTI", "边界与源项"],
    ["Sod / Shu–Osher", "重构与传感器"], ["HBL / SBLI", "高马赫组合"], ["CURVE-C9", "曲线网格组合"],
  ];
  chain.forEach((entry, idx) => {
    const x = 0.80 + idx * 2.04;
    slide.addShape(pptx.ShapeType.roundRect, {
      x, y: 4.55, w: 1.58, h: 1.08,
      rectRadius: 0.05,
      fill: { color: idx % 2 === 0 ? C.blueLight : C.redLight },
      line: { color: idx % 2 === 0 ? C.blue : C.red, width: 1.1 },
    });
    addText(slide, entry[0], x + 0.08, 4.68, 1.42, 0.34, { fontSize: 12, bold: true, align: "center", color: C.ink });
    addText(slide, entry[1], x + 0.08, 5.08, 1.42, 0.30, { fontSize: 9.5, align: "center", color: C.muted });
    if (idx < chain.length - 1) addArrow(slide, x + 1.60, 5.08, 0.40, 0, C.muted);
  });
  addText(slide, "每个新算例都必须先通过 CPU/GPU 同输入、场/统计量和旧路径回归。", 0.92, 6.27, 11.75, 0.38, {
    fontSize: 17, bold: true, align: "center",
  });
  addFooter(slide, "Source: documents/GPU_VALIDATION_MATRIX.md; tests/gpu_validation/README.md");
  addNotes(slide, "controlled SBLI 不是生产级激波边界层相互作用。它只证明已有边界层、持续斜激波入口、传感器和 Roe MP7 在受控网格上能够数值一致组合。");
}

// 10. Boundary coverage
{
  const slide = pptx.addSlide();
  addTitle(slide, "边界条件已形成受方向约束的能力表", "Boundary capability", 10);
  const data = [
    [{ text: "bctype / 类型", options: { bold: true, color: C.paper, fill: C.dark } }, { text: "x", options: { bold: true, color: C.paper, fill: C.dark } }, { text: "y", options: { bold: true, color: C.paper, fill: C.dark } }, { text: "z", options: { bold: true, color: C.paper, fill: C.dark } }, { text: "已验证范围", options: { bold: true, color: C.paper, fill: C.dark } }],
    ["periodic", "✓", "✓", "✓", "local + MPI"],
    ["50 zero extrapolation", "✓", "✓", "✓", "NP=1/2/4"],
    ["60 symmetry", "✓", "✓", "✓", "Cartesian；曲线 x"],
    ["41 isothermal no-slip", "✓", "✓", "✓", "Cartesian；曲线 x/y slices"],
    ["42 adiabatic no-slip", "✓", "✓", "—", "CPU 仅 ndir=1…4"],
    ["411 slip-isothermal", "—", "✓", "—", "CPU 仅 y"],
    ["421 slip-adiabatic", "—", "✓", "—", "CPU 仅 y"],
    ["11/21 profile/outflow", "✓", "—", "—", "HBL / SBLI"],
    ["12/52 characteristic/farfield", "✓", "✓", "—", "受控 Cartesian"],
    ["31 fixed", "—", "✓", "—", "RTI"],
  ];
  slide.addTable(data, {
    x: 0.78, y: 1.28, w: 11.78, h: 4.98,
    colW: [3.05, 0.72, 0.72, 0.72, 6.57],
    rowH: 0.44,
    fontFace: FONT, fontSize: 11.2, color: C.ink,
    border: { pt: 0.65, color: C.line },
    fill: C.paper,
    margin: 0.06,
    valign: "mid",
    align: "center",
    autoFit: false,
  });
  addText(slide, "方向限制来自 CPU 真实实现，不以边界名称推断“应该支持”。能力门控对未验证组合直接拒绝。", 0.90, 6.44, 11.55, 0.38, {
    fontSize: 16.5, bold: true, align: "center",
  });
  addFooter(slide, "Source: src/bc.F90; src_gpu/boundary_gpu.cuf; src_gpu/case_capability_gpu.cuf; Phase H matrix");
  addNotes(slide, "表中的勾表示存在受控验证，不表示任意物理参数、任意曲线方向都可用。例如 42 的 z、411/421 的 x/z 在 CPU 原实现中就没有对应有效分支，因此 GPU 明确拒绝。");
}

// 11. Shock path
{
  const slide = pptx.addSlide();
  addTitle(slide, "激波区域才进入 Roe 特征空间", "Shock-capturing path", 11);
  addText(slide, "S = (∇·u)² / [(∇·u)² + |ω|² + ε] · max(χₓ, χᵧ, χ_z)", 0.86, 1.22, 6.55, 0.58, {
    fontSize: 19.5, fontFace: "Cambria Math", bold: true, color: C.ink,
  });
  addText(slide, "χₓ = |pᵢ₊₁ − 2pᵢ + pᵢ₋₁| / (pᵢ₊₁ + 2pᵢ + pᵢ₋₁)", 0.86, 1.84, 6.25, 0.46, {
    fontSize: 16, fontFace: "Cambria Math", color: C.muted,
  });
  addText(slide, "实现：commcal:ducrossensor  ↔  shock_sensor_gpu:compute_shock_sensor_gpu", 0.86, 2.32, 6.25, 0.34, {
    fontSize: 9.8, fontFace: MONO, color: C.muted,
  });

  const flow = [
    ["raw sensor", C.blue], ["hm halo", C.teal], ["mask expansion", C.red], ["interface decision", C.amber],
  ];
  flow.forEach((item, i) => {
    const x = 0.86 + i * 1.63;
    slide.addShape(pptx.ShapeType.roundRect, {
      x, y: 3.03, w: 1.30, h: 0.70,
      rectRadius: 0.05, fill: { color: C.paper }, line: { color: item[1], width: 1.2 },
    });
    addText(slide, item[0], x + 0.08, 3.18, 1.14, 0.38, { fontSize: 10.8, bold: true, color: item[1], align: "center" });
    if (i < flow.length - 1) addArrow(slide, x + 1.32, 3.38, 0.27, 0, C.muted);
  });

  addArrow(slide, 6.72, 3.37, 0.76, -0.72, C.red);
  addArrow(slide, 6.72, 3.37, 0.76, 0.72, C.blue);
  addBand(slide, 7.66, 2.12, 4.65, 1.25, "lshock = true", "Roe 左特征投影 → MP7 重构 → 右特征回投", C.red, C.redLight);
  addBand(slide, 7.66, 3.90, 4.65, 1.25, "lshock = false", "物理空间 Steger-Warming split flux + MP7", C.blue, C.blueLight);

  addStat(slide, 0.86, 4.58, 2.52, "0", "C9 mask mismatch", C.teal, "7659 nodes each");
  addStat(slide, 3.62, 4.58, 2.52, "1.0e−15", "C9 raw sensor L∞", C.blue, "curvilinear HBL");
  addStat(slide, 7.66, 5.45, 2.10, "1.93×", "x kernel optimization", C.red, "cached split flux");
  addStat(slide, 9.98, 5.45, 2.10, "2.52–2.74×", "y/z kernel optimization", C.teal, "Nsight Compute");

  addFooter(slide, "Source: commcal.F90; shock_sensor_gpu.cuf; solver_gpu.cuf; S0-A4…A10 and CURVE-C9 reports");
  addNotes(slide, "当前选择性路径不是先计算所有界面再覆盖的多遍实现。特征 kernel 内复用线程局部的 split flux，减少重复计算和 local spill。mask 仍会导致路径差异，但只在传感器触发区域执行特征重构。");
}

// 12. Curvilinear progression
{
  const slide = pptx.addSlide();
  addTitle(slide, "CURVE-C0 至 C9 建立静态曲线网格证据链", "Curvilinear validation", 12);
  slide.addImage({ path: `${ASSET}/curvilinear_mesh.jpeg`, x: 0.68, y: 1.30, w: 5.62, h: 4.98 });
  addText(slide, "边界保持的三维非正交映射", 1.22, 6.26, 4.55, 0.30, { fontSize: 12.5, color: C.muted, align: "center" });

  const steps = [
    ["C0–C2", "对流 / 扩散 / 滤波 + x/y/z slabs", C.blue],
    ["C3–C4", "自由流保持 + 解析 metric oracle", C.teal],
    ["C5–C6", "曲面法向、symmetry、wall41、边界滤波", C.red],
    ["C7–C8", "Mach 5 HBL + 三维统计 + 显式滤波", C.amber],
    ["C9", "Ducros + selective Roe + MPI 组合", C.blue],
  ];
  steps.forEach((row, i) => {
    const y = 1.34 + i * 0.92;
    slide.addShape(pptx.ShapeType.chevron, {
      x: 6.72, y, w: 1.28, h: 0.66,
      fill: { color: row[2] }, line: { color: row[2] },
    });
    addText(slide, row[0], 6.82, y + 0.11, 0.86, 0.36, { fontSize: 12, bold: true, color: C.paper, align: "center" });
    addText(slide, row[1], 8.16, y + 0.03, 4.20, 0.58, { fontSize: 12.2, color: C.ink, bold: i === 4 });
  });

  addStat(slide, 6.72, 5.78, 1.75, "J > 0", "全域有限", C.teal);
  addStat(slide, 8.68, 5.78, 1.75, "5.37e−16", "metric identity", C.blue);
  addStat(slide, 10.64, 5.78, 1.75, "0 errors", "memcheck", C.red);
  addFooter(slide, "Source: CURVE-C0…C9 rows in documents/GPU_VALIDATION_MATRIX.md; grid/metric/wall validation scripts");
  addNotes(slide, "C4 用解析映射检查 jacob 和九个 dxi 分量，而不是只用 CPU/GPU 互相比。C7 修复了三维壁面面积和体积统计。C9 将传感器与选择性 Roe 组合到曲线网格，但不声称真实 SBLI 网格收敛。");
}

// 13. Validation methodology
{
  const slide = pptx.addSlide();
  addTitle(slide, "验证不是“程序能跑”，而是分层证据", "Validation methodology", 13);
  const levels = [
    ["L1", "构建与运行", "CPU/GPU 顶层 CMake；输入契约；有限状态", C.muted],
    ["L2", "同相位数值等价", "完整 RK snapshot；ρ/u/v/w/p/T 与 q1:q5", C.blue],
    ["L3", "物理/代数不变量", "壁面速度与温度；EOS；正 Jacobian；精确解", C.teal],
    ["L4", "MPI 语义", "x/y/z slabs；plane；2×2×2；halo interface", C.red],
    ["L5", "运行时可信度", "Compute Sanitizer；Nsight residency；无整场往返", C.amber],
    ["L6", "回归与范围门控", "旧算例矩阵；unsupported 组合预期拒绝", C.blue],
  ];
  levels.forEach((row, i) => {
    const y = 1.25 + i * 0.87;
    slide.addShape(pptx.ShapeType.ellipse, {
      x: 0.85, y: y + 0.02, w: 0.58, h: 0.58,
      fill: { color: row[3] }, line: { color: row[3] },
    });
    addText(slide, row[0], 0.91, y + 0.10, 0.45, 0.32, { fontSize: 12, bold: true, color: C.paper, align: "center" });
    addText(slide, row[1], 1.68, y, 2.42, 0.30, { fontSize: 15, bold: true, color: row[3] });
    addText(slide, row[2], 4.25, y, 7.65, 0.46, { fontSize: 13, color: C.ink });
  });
  addStat(slide, 9.60, 5.88, 2.30, "137 pass", "矩阵证据记录", C.teal, "另有 pending/partial");
  addText(slide, "关键规则：发现确定性 CPU bug 时暂停，以人工决策后的修正 CPU 作为新 oracle。", 0.86, 6.50, 8.45, 0.36, {
    fontSize: 16.5, bold: true, color: C.red,
  });
  addFooter(slide, "Source: docs/adr/0004, 0005, 0018; tests/gpu_validation/*; documents/GPU_VALIDATION_MATRIX.md");
  addNotes(slide, "同相位比较解决了 CPU boucon/filter 与 GPU complete-RK 输出阶段不同的问题。对 Sod 还加入精确解误差、厚度和波位置检查，说明 CPU/GPU 一致不等于数值方法本身准确。");
}

// 14. Correctness results
{
  const slide = pptx.addSlide();
  addTitle(slide, "代表性 CPU/GPU 误差均在各自验收阈值内", "Numerical evidence", 14);
  slide.addImage({ path: `${ASSET}/numerical_equivalence_errors.jpeg`, x: 0.72, y: 1.18, w: 11.90, h: 3.86 });
  addBand(slide, 0.78, 5.35, 3.70, 1.12, "光滑与壁面流", "TGV、Channel、HBL 通常处于 1e−15 至 1e−12", C.blue);
  addBand(slide, 4.80, 5.35, 3.70, 1.12, "间断格式", "Sod/Shu–Osher 的逐场差异保持 roundoff 量级", C.teal);
  addBand(slide, 8.82, 5.35, 3.70, 1.12, "2dvort filtered", "约 5.2e−9，来自接口附近输出重构差异，单独设阈值", C.red);
  addFooter(slide, "Measured maxima selected from documents/GPU_VALIDATION_MATRIX.md; diamonds are case-specific acceptance limits");
  addNotes(slide, "不要把不同量的误差绝对值直接做物理比较。这张图只展示每个验证门中最不利的已记录绝对差值与其阈值。2dvort filtered 的阈值更宽，是已知的输出重构/近零分量差异，不应隐藏。");
}

// 15. Performance vocabulary
{
  const slide = pptx.addSlide();
  addTitle(slide, "先区分整体加速、双卡收益与并行效率", "How to read performance", 15);
  addBand(slide, 0.78, 1.24, 3.70, 1.42, "Wall time · 实际耗时", "程序从启动到结束经历的时间；越短越好。本汇报不把单个 kernel 时间代替整体时间。", C.blue);
  addBand(slide, 4.82, 1.24, 3.70, 1.42, "整体加速比 · S", "S = T_CPU,NP1 / T_GPU；回答 GPU 相对单 CPU MPI rank 快多少。", C.red);
  addBand(slide, 8.86, 1.24, 3.70, 1.42, "强缩放收益 · G", "G = T_GPU1 / T_GPU2；同一网格增加 GPU 后是否更快。理想双卡为 2。", C.teal);

  addText(slide, "并行效率", 0.88, 3.03, 1.45, 0.32, { fontSize: 15, bold: true, color: C.ink });
  addText(slide, "E = G / 2", 2.38, 2.96, 1.55, 0.45, { fontSize: 22, fontFace: "Cambria Math", bold: true, color: C.teal });
  addText(slide, "例如 G=1.43 时，双卡效率约为 71%。", 4.12, 2.98, 4.30, 0.40, { fontSize: 14, color: C.ink });

  addText(slide, "MPI rank 是进程，不等于 GPU 数量", 0.88, 3.72, 5.00, 0.40, { fontSize: 18, bold: true, color: C.red });
  addText(slide, "NP=2 且两张 GPU：每个 rank 独占一张 GPU，可用于双卡性能分析。", 0.88, 4.26, 5.46, 0.62, { fontSize: 14, color: C.ink });
  addText(slide, "NP=4/8 但仍只有两张 GPU：多个 rank 共用设备，属于 oversubscription；可检验通信逻辑，不能当作四卡或八卡扩展。", 0.88, 5.12, 5.46, 0.92, { fontSize: 14, color: C.ink });

  slide.addShape(pptx.ShapeType.rect, { x: 7.06, y: 3.72, w: 2.30, h: 2.25, fill: { color: C.blueLight }, line: { color: C.blue, width: 1.2 } });
  slide.addShape(pptx.ShapeType.rect, { x: 9.86, y: 3.72, w: 2.30, h: 2.25, fill: { color: C.tealLight }, line: { color: C.teal, width: 1.2 } });
  addText(slide, "GPU 0", 7.58, 3.92, 1.28, 0.34, { fontSize: 16, bold: true, color: C.blue, align: "center" });
  addText(slide, "GPU 1", 10.38, 3.92, 1.28, 0.34, { fontSize: 16, bold: true, color: C.teal, align: "center" });
  addLabel(slide, "rank 0", 7.58, 4.60, 1.28, C.blue, C.paper);
  addLabel(slide, "rank 1", 10.38, 4.60, 1.28, C.teal, C.paper);
  addText(slide, "NP=2：一进程绑定一张物理 GPU", 7.10, 5.39, 5.02, 0.36, { fontSize: 13.5, bold: true, color: C.ink, align: "center" });
  addText(slide, "后续页面中，“相对 CPU 的加速”与“增加 GPU 的收益”始终分开报告。", 1.28, 6.45, 10.75, 0.38, { fontSize: 17, bold: true, align: "center" });
  addFooter(slide, "Definitions used consistently in slides 16–19");
  addNotes(slide, "面向非 HPC 听众，先解释三个问题：实际等了多久、GPU 相对 CPU 快多少、从一张 GPU 增加到两张后又快多少。MPI rank 只是进程；只有 rank 与物理 GPU 一一绑定时，NP 才等于 GPU 数量。");
}

// 16. Current 128 benchmark
{
  const slide = pptx.addSlide();
  addTitle(slide, "当前版本 128³：单 GPU 整体加速约 37.84×", "Current controlled benchmark", 16);
  slide.addImage({ path: `${ASSET}/channel_128_benchmark.jpeg`, x: 0.60, y: 1.18, w: 12.12, h: 4.14 });
  addText(slide, "S = T_CPU,NP1 / median(T_GPU)", 0.88, 5.52, 3.55, 0.36, { fontSize: 18, fontFace: "Cambria Math", bold: true, color: C.ink });
  addText(slide, "128³ channel · 100 steps · Δt=7.5×10⁻⁴ · 643e · diffusion on · filter off · no checkpoint", 4.18, 5.52, 8.18, 0.36, {
    fontSize: 12.2, color: C.muted, align: "right",
  });
  addBand(slide, 0.82, 6.02, 3.30, 0.70, "CPU NP=1", "526.677 s · 单次基线", C.red);
  addBand(slide, 4.34, 6.02, 3.30, 0.70, "GPU NP=1", "13.917 s · 3 次中位数", C.blue);
  addBand(slide, 7.86, 6.02, 4.42, 0.70, "GPU NP=2", "16.19–19.43 s · 三种分解", C.teal);
  addFooter(slide, "2× RTX 4000 Ada 19 GB; fresh benchmark on 2026-09-03; CPU/GPU full-program wall time");
  addNotes(slide, "CPU 只测了一次，因为单次约 8.8 分钟；GPU 每个配置测三次，图中误差棒是 min 到 max。单 GPU 相对 CPU 的整体加速明确，但这个 128 立方工作负载上，增加第二张 GPU 没有缩短时间。");
}

// 17. Historical 256 benchmark
{
  const slide = pptx.addSlide();
  addTitle(slide, "历史 256³：单卡约 49×，双卡最佳约 70×", "Historical larger-grid benchmark", 17);
  slide.addImage({ path: `${ASSET}/channel_256_historical_benchmark.jpeg`, x: 0.62, y: 1.20, w: 12.02, h: 4.60 });
  addBand(slide, 0.80, 6.00, 3.46, 0.72, "CPU NP=1", "4898 s · 约 81.6 min", C.red);
  addBand(slide, 4.46, 6.00, 3.46, 0.72, "GPU NP=1", "100 s · S=48.98×", C.blue);
  addBand(slide, 8.12, 6.00, 4.20, 0.72, "GPU NP=2 y-slab", "70 s · S=69.97× · G=1.43×", C.teal);
  addFooter(slide, "Historical single-run channel data, 2026-07-04/05; 256³, 100 steps, same two-GPU workstation");
  addNotes(slide, "左图中的 CPU、单 GPU 和三种双卡分解是可解释的性能数据。右图的 NP=4/8 仍然只使用两张物理 GPU，主要显示进程划分带来的开销变化。历史计时以整秒记录且每个配置只有一次，因此只能作为工程趋势，不与当前 128 数据做严格统计比较。");
}

// 18. Rank topology and oversubscription
{
  const slide = pptx.addSlide();
  addTitle(slide, "更多 MPI rank 扩大了通信覆盖，不等于增加 GPU", "Ranks, topology and hardware", 18);
  const rows = [
    ["NP=2", "1 rank / GPU", "2×1×1 · 1×2×1 · 1×1×2", "70–79 s", C.teal],
    ["NP=4", "2 ranks / GPU", "4×1×1 · 1×4×1 · 1×1×4 · 三种平面", "70–94 s", C.amber],
    ["NP=8", "4 ranks / GPU", "8×1×1 · 1×8×1 · 1×1×8 · 2×2×2", "83–146 s", C.red],
  ];
  addText(slide, "进程数", 0.82, 1.30, 1.25, 0.34, { fontSize: 13, bold: true, color: C.paper, fill: { color: C.dark }, align: "center" });
  addText(slide, "设备复用", 2.08, 1.30, 2.10, 0.34, { fontSize: 13, bold: true, color: C.paper, fill: { color: C.dark }, align: "center" });
  addText(slide, "256³ 已跑拓扑", 4.19, 1.30, 5.70, 0.34, { fontSize: 13, bold: true, color: C.paper, fill: { color: C.dark }, align: "center" });
  addText(slide, "历史耗时", 9.90, 1.30, 2.40, 0.34, { fontSize: 13, bold: true, color: C.paper, fill: { color: C.dark }, align: "center" });
  rows.forEach((row, i) => {
    const y = 1.72 + i * 0.78;
    const fill = i % 2 === 0 ? C.paper : C.light;
    addText(slide, row[0], 0.82, y, 1.25, 0.58, { fontSize: 15, bold: true, color: row[4], fill: { color: fill }, align: "center", margin: 0.08 });
    addText(slide, row[1], 2.08, y, 2.10, 0.58, { fontSize: 13, bold: true, color: C.ink, fill: { color: fill }, align: "center", margin: 0.08 });
    addText(slide, row[2], 4.19, y, 5.70, 0.58, { fontSize: 12.5, color: C.ink, fill: { color: fill }, margin: 0.08 });
    addText(slide, row[3], 9.90, y, 2.40, 0.58, { fontSize: 15, bold: true, color: row[4], fill: { color: fill }, align: "center", margin: 0.08 });
  });

  addBand(slide, 0.82, 4.42, 3.55, 1.42, "性能可解释范围", "NP=1 与 NP=2：两张 GPU 上最多一进程一设备，可计算双卡强缩放。", C.teal);
  addBand(slide, 4.56, 4.42, 3.55, 1.42, "通信正确性压力测试", "NP=16: 4×2×2 / 2×4×2 / 2×2×4；NP=32: 4×4×2。", C.blue);
  addBand(slide, 8.30, 4.42, 4.02, 1.42, "内部 rank 覆盖", "Channel NP=27, 3×3×3 产生完全被六个 MPI 邻居包围的内部 rank。", C.red);
  addText(slide, "NP=4/8/16/27/32 的通过证明三方向 halo 与拓扑路由可工作；只有增加物理 GPU 后才能评价对应规模的并行效率。", 0.96, 6.34, 11.20, 0.45, { fontSize: 16.5, bold: true, align: "center" });
  addFooter(slide, "Sources: historical 256 matrix; GPU_VALIDATION_MATRIX MR-L2 high-rank and channel NP=27 records");
  addNotes(slide, "这页用于阻止一个常见误解：NP=8 不代表八张 GPU。当前机器只有两张卡，所以 NP=4 以上只是改变每张卡承载的进程数。高 rank 测试很重要，因为它覆盖复杂邻居关系和内部 rank，但它不是硬件扩展曲线。");
}

// 19. Scaling interpretation
{
  const slide = pptx.addSlide();
  addTitle(slide, "网格变大后，双卡才开始抵消通信成本", "Performance interpretation", 19);
  slide.addImage({ path: `${ASSET}/channel_problem_size_scaling_context.jpeg`, x: 0.72, y: 1.25, w: 5.46, h: 3.72 });
  addBand(slide, 6.56, 1.35, 2.76, 1.25, "128³ · 当前", "G=0.86×：双卡比单卡慢。计算量不足以覆盖固定通信与同步开销。", C.blue);
  addBand(slide, 9.52, 1.35, 2.76, 1.25, "256³ · 历史", "G=1.43×：双卡开始受益，对应约 71% 并行效率。", C.teal);
  addBand(slide, 6.56, 2.86, 5.72, 1.25, "为何仍达不到理想 2×", "halo 先拷到 host，再调用阻塞 MPI；每个 kernel 后显式同步，计算与通信没有重叠。", C.red);
  addBand(slide, 6.56, 4.37, 5.72, 1.25, "256³ TGV Nsight 佐证", "10 steps：NP1 H2D/D2H=3.20 GB/31.5 MB；NP2 x-slab=11.0 GB/7.70 GB，增加量主要来自 host-staged halo。", C.amber);
  addArrow(slide, 2.10, 5.60, 8.78, 0, C.muted);
  addLabel(slide, "pinned buffers", 2.92, 5.91, 1.74, C.blue, C.blueLight);
  addLabel(slide, "nonblocking MPI", 5.02, 5.91, 1.86, C.blue, C.blueLight);
  addLabel(slide, "CUDA-aware MPI", 7.24, 5.91, 1.96, C.teal, C.tealLight);
  addLabel(slide, "repeat 256³", 9.56, 5.91, 1.72, C.red, C.redLight);
  addText(slide, "结论：已有数据支持“大问题更适合双卡”的趋势，但生产级扩展结论仍需当前版本重复测试和更多物理 GPU。", 0.92, 6.52, 11.48, 0.34, { fontSize: 16.5, bold: true, align: "center" });
  addFooter(slide, "128³ current medians vs 256³ historical single runs; Nsight source: nsys_tgv_256_np1_np2_driver_current");
  addNotes(slide, "128 和 256 的结果不矛盾。网格边长增加一倍后，总单元数增加八倍，而 halo 数据按表面积增长，计算相对通信的比例提高。由于两组数据来自不同日期且历史 256 只有单次计时，这里使用 indicates 级别的工程解释，不声称完成正式强缩放研究。");
}

// 20. Correctness findings and limits
{
  const slide = pptx.addSlide();
  addTitle(slide, "移植过程同时暴露了 CPU oracle 的确定性问题", "Correctness decisions", 20);
  const fixes = [
    ["显式十阶滤波", "物理边界读取未定义 halo", "CPU/GPU 统一 0-6-6-6-8-10 闭合"],
    ["三维 HBL 统计", "下壁面误用 i-j 面积；GPU 体积 Cartesian 化", "改为 i-k 四边形与六项六面体体积"],
    ["NSCBC transverse filter", "bctype=52 前 k-halo 可能陈旧", "CPU 先全 rank 刷新 halo，再恢复 GPU 对照"],
    ["Ducros npdc=4", "y/z 双物理侧未夹断索引", "经确认后 CPU/GPU 六侧对称处理"],
  ];
  fixes.forEach((row, i) => {
    const y = 1.28 + i * 1.12;
    addText(slide, String(i + 1), 0.83, y + 0.04, 0.46, 0.46, { fontSize: 16, bold: true, color: C.paper, align: "center", fill: { color: i % 2 ? C.blue : C.red }, margin: 0 });
    addText(slide, row[0], 1.52, y, 2.10, 0.35, { fontSize: 14.5, bold: true, color: i % 2 ? C.blue : C.red });
    addText(slide, row[1], 3.80, y, 3.70, 0.58, { fontSize: 12.3, color: C.ink });
    addArrow(slide, 7.66, y + 0.28, 0.65, 0, C.muted);
    addText(slide, row[2], 8.50, y, 3.82, 0.58, { fontSize: 12.3, color: C.ink, bold: true });
  });
  addBand(slide, 0.83, 5.94, 5.70, 0.78, "决策门", "明确 CPU bug：停止 → 报告 → 人工确认 → 修 CPU → 同步 GPU", C.red, C.redLight);
  addBand(slide, 6.76, 5.94, 5.56, 0.78, "不直接处理", "case-specific NSCBC 物理选择、CPU 含糊分支、未验证方向", C.muted, C.light);
  addFooter(slide, "Source: approved fixes recorded in ASTR_FULL_GPU_ARCHITECTURE_PLAN.md and GPU_VALIDATION_MATRIX.md");
  addNotes(slide, "这里不是把 CPU 当成绝对真值。CPU 是主要数值 oracle，但发现确定性错误时必须修正后再比较。对 NSCBC 的物理取舍，维护者反馈 case-specific，因此当前保留最简单受控边界，不扩大结论。");
}

// 21. Next plan
{
  const slide = pptx.addSlide();
  addTitle(slide, "下一阶段：先把多卡做快，再提高物理可信度", "Next steps", 21);
  const priorities = [
    ["P1", "HaloTransport 性能化", "pinned host buffer → nonblocking → CUDA-aware MPI；保留 L0 fallback", C.red],
    ["P2", "生产尺度 benchmark", "当前版本重复 256³；GPU1/GPU2；强/弱缩放；kernel/communication 分解", C.blue],
    ["P3", "真实 SBLI 可信化", "网格/时间收敛；激波位置；壁面热流；入口/远场与 sponge 的 case contract", C.teal],
    ["P4", "长期可移植架构", "backend-neutral field/halo API，为 AMD/HIP/DCU 保留实现边界", C.amber],
  ];
  priorities.forEach((row, i) => {
    const y = 1.25 + i * 1.20;
    addLabel(slide, row[0], 0.84, y + 0.16, 0.78, row[3], row[3] === C.red ? C.redLight : row[3] === C.blue ? C.blueLight : row[3] === C.teal ? C.tealLight : C.amberLight);
    addText(slide, row[1], 1.88, y, 3.20, 0.40, { fontSize: 17, bold: true, color: row[3] });
    addText(slide, row[2], 5.18, y, 7.02, 0.70, { fontSize: 13, color: C.ink });
  });

  addText(slide, "本阶段结论", 0.84, 6.12, 1.65, 0.34, { fontSize: 15, bold: true, color: C.ink });
  addText(slide, "单卡 GPU 路径已具备显著收益和完整正确性框架；多卡通信效率与真实复杂物理验证是接下来的两条主线。", 2.62, 6.02, 9.60, 0.62, {
    fontSize: 18, bold: true, color: C.ink,
  });
  addFooter(slide, "Scope held: no compact, RANS/LES, chemistry/combustion, IBM, moving/multi-block grids, or GPU HDF5 in the near-term line");
  addNotes(slide, "建议组会上明确申请下一阶段优先级：先做通信后端性能化，还是先做真实 SBLI 物理可信化。两者可以并行准备，但性能宣称必须等新后端和更大网格完成后再更新。");
}

// 22. Appendix commit ledger
{
  const slide = pptx.addSlide();
  addTitle(slide, "附录：提交清单", "Appendix A", 22);
  const rows = [
    ["Commit", "Date", "Primary progress", "Δ lines"],
    ["df1961b", "07-03", "first CUDA backend + TGV x-slab halo", "+5260/−2539"],
    ["5597e69", "07-03", "multi-direction MPI halo validation", "+961/−72"],
    ["7a5fdce", "07-03", "plans, ADRs, validation records", "+4178"],
    ["fd250ab", "07-03", "2dvort GPU + multi-rank", "+1036/−78"],
    ["b5ed8c3", "07-03", "HIT + zero-extrap boundary", "+696/−22"],
    ["451fe0e", "07-04", "channel walls + multi-rank", "+3762/−232"],
    ["a684edb", "07-04", "case/boundary/source expansion", "+2628/−244"],
    ["270be49", "07-11", "shock + hypersonic BL", "+9664/−115"],
    ["16ded59", "07-12", "selective Roe + sponge gate", "+1556/−231"],
    ["aecb18f", "07-22", "NSCBC farfield + curvilinear", "+1101/−78"],
    ["63fe7a8", "09-03", "CURVE-C5…C9 closure", "+2134/−151"],
  ];
  slide.addTable(rows, {
    x: 0.80, y: 1.16, w: 11.75, h: 5.62,
    colW: [1.45, 1.05, 7.52, 1.73],
    rowH: 0.44,
    fontFace: FONT, fontSize: 10.8,
    border: { pt: 0.65, color: C.line },
    fill: C.paper, color: C.ink, margin: 0.06,
    autoFit: false,
  });
  addFooter(slide, "Source: git log and git show --shortstat for 47737cf..63fe7a8b");
  addNotes(slide, "该页用于回答具体哪次提交完成了哪项工作。行数是每次提交 shortstat，不应与功能价值简单等同。");
}

// 23. Appendix benchmark protocol
{
  const slide = pptx.addSlide();
  addTitle(slide, "附录：性能证据与可复现路径", "Appendix B", 23);
  addBand(slide, 0.78, 1.25, 3.70, 2.22, "共同工作负载", "Channel flow\nMAXSTEP=100 · DELTAT=7.5d-4\n643e · diffusion on · filter off\nfixed forcing=1.d-4\nno checkpoint", C.blue);
  addBand(slide, 4.82, 1.25, 3.70, 2.22, "两组证据", "128³ current: CPU×1, GPU×3\n256³ historical: each×1\nwall time includes program overhead\nNP>2 shares two physical GPUs", C.red);
  addBand(slide, 8.86, 1.25, 3.70, 2.22, "硬件", "2× RTX 4000 Ada 19 GB\nDriver 595.84\nXeon Gold 6248R\n2 sockets · 48 physical cores", C.teal);

  addText(slide, "128³ 入口", 0.82, 3.82, 1.25, 0.30, { fontSize: 14, bold: true, color: C.ink });
  addText(slide, "tests/gpu_validation/run_channel_128_benchmark.sh", 2.22, 3.89, 7.00, 0.34, { fontSize: 12, fontFace: MONO, color: C.blue });
  addText(slide, "128³ 数据", 0.82, 4.38, 1.25, 0.30, { fontSize: 14, bold: true, color: C.ink });
  addText(slide, "tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903/", 2.22, 4.45, 9.85, 0.45, { fontSize: 10.2, fontFace: MONO, color: C.blue });
  addText(slide, "256³ 数据", 0.82, 4.98, 1.25, 0.30, { fontSize: 14, bold: true, color: C.ink });
  addText(slide, "tests/gpu_validation/out/channel_256_{cpu,gpu,mpirank}_*", 2.22, 5.01, 7.90, 0.34, { fontSize: 11.2, fontFace: MONO, color: C.teal });
  addText(slide, "绘图脚本", 0.82, 5.56, 1.25, 0.30, { fontSize: 14, bold: true, color: C.ink });
  addText(slide, "scripts/presentations/plot_astr_gpu_progress.py", 2.22, 5.59, 7.40, 0.34, { fontSize: 12, fontFace: MONO, color: C.blue });

  addText(slide, "限制：CPU 与历史 256³ 均为单次；未固定 GPU clocks；不同日期数据只用于工程趋势，不构成发表级强缩放结论。", 0.92, 6.30, 11.50, 0.50, {
    fontSize: 15.5, bold: true, color: C.red, align: "center",
  });
  addFooter(slide, "Generated 2026-09-03 from current checkout at commit 63fe7a8b4a45b17");
  addNotes(slide, "若需要正式性能报告，下一轮应在当前版本重复 256 立方 CPU/GPU 测试，固定 GPU clocks，记录 CPU NUMA 与 rank 绑定，拆分初始化、RK 计算和 halo 通信，并在真实多 GPU 数量上做强弱缩放。");
}

fs.writeFileSync(
  NOTES_OUTPUT,
  ["# ASTR GPU 移植阶段进展讲稿", "", ...speakerNotes.flatMap((note, index) => [
    `## Slide ${index + 1}`,
    "",
    note,
    "",
  ])].join("\n"),
  "utf8",
);

pptx.writeFile({ fileName: OUTPUT, compression: true })
  .then(async () => {
    await fixPresentationElementOrder(OUTPUT);
    console.log(OUTPUT);
  })
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
