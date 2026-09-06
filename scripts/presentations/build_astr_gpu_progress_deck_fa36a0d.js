const pptxgen = require("pptxgenjs");
const fs = require("node:fs");
const JSZip = require("jszip");

const pptx = new pptxgen();
pptx.author = "ASTR GPU Porting Project";
pptx.company = "ASTR";
pptx.subject = "ASTR GPU progress from the pure CPU baseline 47737cf to fa36a0d";
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
const OUTPUT = `${ROOT}/documents/presentations/ASTR_GPU_PROGRESS_47737cf_to_fa36a0d_20260904.pptx`;
const NOTES_OUTPUT = `${ROOT}/documents/presentations/ASTR_GPU_PROGRESS_47737cf_to_fa36a0d_20260904_SPEAKER_NOTES.md`;
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
  addText(slide, kicker.toUpperCase(), 0.62, 0.28, 4.4, 0.22, {
    fontSize: 9.5, color: C.red, bold: true, charSpacing: 1.5,
  });
  addText(slide, title, 0.62, 0.52, 11.7, 0.54, {
    fontSize: 28, color: C.ink, bold: true,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: 0.16, h: H,
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

function addNotes(slide, note, sources) {
  const sourceBlock = sources.map((source) => `- ${source}`).join("\n");
  const full = `${note}\n\n[Sources]\n${sourceBlock}`;
  slide.addNotes(full);
  speakerNotes.push(full);
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
  addText(slide, value, x + 0.22, y + 0.10, w - 0.34, 0.50, {
    fontSize: 28, bold: true, color,
  });
  addText(slide, label, x + 0.22, y + 0.61, w - 0.34, 0.27, {
    fontSize: 12.5, bold: true,
  });
  if (detail) {
    addText(slide, detail, x + 0.22, y + 0.89, w - 0.34, 0.18, {
      fontSize: 8.5, color: C.muted,
    });
  }
}

function addBand(slide, x, y, w, h, heading, body, color = C.blue, fill = C.paper, bodyColor = C.ink) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: fill },
    line: { color: C.line, width: 0.7 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w: 0.08, h,
    fill: { color }, line: { color },
  });
  addText(slide, heading, x + 0.22, y + 0.10, w - 0.34, 0.32, {
    fontSize: 15, bold: true, color,
  });
  addText(slide, body, x + 0.22, y + 0.45, w - 0.34, h - 0.55, {
    fontSize: 12.5, color: bodyColor, valign: "top",
  });
}

function addLabel(slide, text, x, y, w, color = C.blue, fill = C.blueLight) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h: 0.36,
    rectRadius: 0.05,
    fill: { color: fill }, line: { color: fill },
  });
  addText(slide, text, x + 0.08, y + 0.03, w - 0.16, 0.29, {
    fontSize: 10.5, color, bold: true, align: "center",
  });
}

function addArrow(slide, x, y, w, h, color = C.muted) {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h,
    line: { color, width: 1.5, endArrowType: "triangle" },
  });
}

function addCheck(slide, x, y, text, color = C.teal, fontSize = 14, width = 5.4) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y: y + 0.03, w: 0.22, h: 0.22,
    fill: { color }, line: { color },
  });
  addText(slide, "✓", x + 0.01, y + 0.015, 0.20, 0.22, {
    fontSize: 11, bold: true, color: C.paper, align: "center",
  });
  addText(slide, text, x + 0.34, y, width, 0.32, { fontSize, color: C.ink });
}

function addRow(slide, y, left, middle, right, color = C.blue) {
  slide.addShape(pptx.ShapeType.line, {
    x: 0.82, y: y + 0.55, w: 11.70, h: 0,
    line: { color: C.line, width: 0.55 },
  });
  addText(slide, left, 0.88, y, 1.55, 0.46, { fontSize: 12.5, bold: true, color });
  addText(slide, middle, 2.58, y, 3.28, 0.46, { fontSize: 12.5, bold: true });
  addText(slide, right, 6.05, y, 6.18, 0.46, { fontSize: 11.8, color: C.muted });
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
    fs.writeFileSync(fileName, await archive.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  }
}

// 1. Opening thesis
{
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  slide.addImage({ path: `${ASSET}/curvilinear_mesh.jpeg`, x: 7.35, y: 0, w: 5.983, h: 7.5, transparency: 5 });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: 8.05, h: 7.5,
    fill: { color: C.dark }, line: { color: C.dark },
  });
  addText(slide, "ASTR GPU 移植", 0.76, 0.72, 6.35, 0.68, { fontSize: 34, color: C.paper, bold: true });
  addText(slide, "把纯 CPU 高阶 CFD 求解器变成可验证的多 GPU 曲线网格求解框架", 0.76, 1.44, 6.40, 0.76, {
    fontSize: 20, color: "D7E3E8", bold: true, valign: "top",
  });
  addBand(slide, 0.76, 2.47, 6.18, 0.82, "要解决的问题", "CPU 计算周期长，复杂三维算例的网格规模和参数扫描受限。", C.red, "25343A", "D7E3E8");
  addBand(slide, 0.76, 3.48, 6.18, 0.82, "采用的方法", "CUDA Fortran + MPI；变量常驻 GPU；显式高阶格式；逐层正确性验证。", C.blue, "25343A", "D7E3E8");
  addBand(slide, 0.76, 4.49, 6.18, 0.82, "得到的结果", "Channel 单卡相对单 CPU 37.84×；256³ 曲线 C10 双卡相对单卡 1.58×。", C.teal, "25343A", "D7E3E8");
  addText(slide, "47737cf 纯 CPU  →  fa36a0d 最新 GPU", 0.76, 5.78, 6.18, 0.36, {
    fontSize: 13, color: "AFC2CB", fontFace: MONO,
  });
  addText(slide, "组会进展汇报  |  2026-09-04", 0.76, 6.43, 4.35, 0.38, {
    fontSize: 14, color: C.paper, bold: true,
  });
  addNotes(slide,
    "开场只保留三个信息：问题是 CPU 计算周期长；方法是把计算和数据生命周期迁到 CUDA Fortran/MPI；结果是代表性 Channel 获得 37.84 倍单卡加速，最新 256 立方曲线 C10 双卡相对单卡获得 1.58 倍加速。后续页面分别回答为什么可信、能算什么和还有什么没做。",
    ["git 47737cf9b6a25b00bc5f1139cdaec36d55e5024e", "git fa36a0d657d0c22ae6d456ec442013d2f723364f", "tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903", "tests/gpu_validation/out/curvilinear_hbl_c10_256_benchmark_after_reduction_20260904"]);
}

// 2. Why the port matters
{
  const slide = pptx.addSlide();
  addTitle(slide, "GPU 移植首先解决的是等待时间", "Why it matters", 2);
  addStat(slide, 0.80, 1.36, 3.40, "526.7 s", "单 CPU：128³ Channel 100 steps", C.red, "约 8.8 分钟");
  addArrow(slide, 4.52, 1.95, 1.10, 0, C.muted);
  addStat(slide, 5.92, 1.36, 3.40, "13.92 s", "单 GPU：相同算例", C.blue, "三次 GPU 计时中位数");
  addStat(slide, 9.62, 1.36, 2.85, "37.84×", "整体加速", C.teal, "相对 NP=1 CPU");
  addText(slide, "同一算例从“等几分钟”缩短到“十几秒”", 0.92, 3.36, 11.55, 0.58, {
    fontSize: 27, bold: true, align: "center", color: C.ink,
  });
  addBand(slide, 0.92, 4.38, 3.55, 1.36, "更高分辨率", "在相近周转时间内使用更多网格，减少尺度分辨不足。", C.blue);
  addBand(slide, 4.88, 4.38, 3.55, 1.36, "更多方案比较", "同一时间预算内测试更多边界条件、参数和模型。", C.red);
  addBand(slide, 8.84, 4.38, 3.55, 1.36, "更快开发闭环", "代码修改后更快完成回归，及时发现数值与边界问题。", C.teal);
  addText(slide, "用途不是“让结果更好看”，而是把原本受计算时间限制的研究规模变得可执行。", 1.02, 6.25, 11.30, 0.45, {
    fontSize: 18, bold: true, align: "center",
  });
  addFooter(slide, "Channel 128³, MAXSTEP=100, fixed forcing; CPU one run, GPU three-run median");
  addNotes(slide,
    "这页只解释收益。37.84 倍来自同一个 128 立方 Channel 工况，CPU NP=1 用时 526.677 秒，GPU NP=1 三次数据的中位数为 13.917 秒。它是端到端壁钟时间，不是单个 kernel 的理论峰值。",
    ["tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903/benchmark_times.tsv", "tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903/gpu_repeats.tsv"]);
}

// 3. Method
{
  const slide = pptx.addSlide();
  addTitle(slide, "核心方法：计算期间让数据留在 GPU", "Method in plain language", 3);
  const stages = [
    ["CPU 准备", "读取输入、生成网格\n初始化物理量", C.red, C.redLight],
    ["上传一次", "主变量、网格度量\n进入 GPU 显存", C.blue, C.blueLight],
    ["GPU 循环", "边界、梯度、对流\n扩散、滤波、RK", C.teal, C.tealLight],
    ["只交换边界", "MPI 只传 halo\n统计只回传标量", C.amber, C.amberLight],
    ["CPU 输出", "Checkpoint/HDF5\n仍由 CPU 管理", C.red, C.redLight],
  ];
  stages.forEach((item, i) => {
    const x = 0.62 + i * 2.54;
    slide.addShape(pptx.ShapeType.rect, {
      x, y: 1.65, w: 2.05, h: 2.55,
      fill: { color: item[3] }, line: { color: item[2], width: 1.1 },
    });
    addText(slide, item[0], x + 0.15, 1.90, 1.75, 0.40, { fontSize: 17, bold: true, color: item[2], align: "center" });
    addText(slide, item[1], x + 0.15, 2.58, 1.75, 0.92, { fontSize: 14, align: "center", valign: "mid" });
    if (i < stages.length - 1) addArrow(slide, x + 2.08, 2.92, 0.43, 0, C.muted);
  });
  addText(slide, "过去：每一步都可能在 CPU 上计算", 0.90, 4.72, 4.65, 0.44, { fontSize: 17, color: C.red, bold: true });
  addText(slide, "现在：整个 Runge–Kutta 计算循环由 GPU 持有权威状态", 0.90, 5.32, 7.70, 0.44, { fontSize: 20, color: C.teal, bold: true });
  addText(slide, "这决定了 GPU 能否真正发挥吞吐能力。", 0.90, 5.96, 6.15, 0.38, { fontSize: 16, color: C.muted });
  addFooter(slide, "Current scope: GPU-resident compute loop; CPU-owned initialization and HDF5 output");
  addNotes(slide,
    "常驻不是完全没有传输。MPI halo 必须交换，输出时仍要回到 CPU。关键变化是 q、原始变量、网格度量、梯度和工作数组在 RK 循环内持续驻留，不再为每个 kernel 搬整场数据。",
    ["src_gpu/mainloop_gpu.cuf", "src_gpu/commarray_gpu.cuf", "documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md"]);
}

// 4. Capability progression
{
  const slide = pptx.addSlide();
  addTitle(slide, "能力已从基础涡流扩展到曲线网格高超声速路径", "What can run now", 4);
  const levels = [
    ["基础正确性", "TGV · 2dvort · HIT", "周期边界、中心差分、滤波", C.blue],
    ["壁面流动", "Channel · Lid-Driven Cavity", "无滑移、绝热/等温、体积力", C.red],
    ["复杂界面", "RTI · Sod · Shu–Osher", "重力源项、激波与接触间断", C.amber],
    ["高超声速", "Mach 5 boundary layer", "Sutherland 黏度、远场与入口", C.teal],
    ["曲线激波路径", "CURVE-C0…C15", "Ducros + selective Roe MP7 + diffusion", C.red],
  ];
  levels.forEach((row, i) => {
    const y = 1.25 + i * 1.03;
    addText(slide, String(i + 1), 0.84, y + 0.05, 0.45, 0.45, {
      fontSize: 16, bold: true, color: C.paper, align: "center", fill: { color: row[3] }, margin: 0,
    });
    addText(slide, row[0], 1.55, y, 2.10, 0.42, { fontSize: 17, bold: true, color: row[3] });
    addText(slide, row[1], 3.85, y, 3.35, 0.42, { fontSize: 15.5, bold: true });
    addText(slide, row[2], 7.38, y, 4.65, 0.42, { fontSize: 14, color: C.muted });
    if (i < levels.length - 1) addArrow(slide, 1.06, y + 0.62, 0, 0.28, C.line);
  });
  addText(slide, "重点：不是给每个算例单独写一套 GPU 程序，而是逐步形成可复用的求解能力。", 0.95, 6.43, 11.45, 0.42, {
    fontSize: 18, bold: true, align: "center",
  });
  addFooter(slide, "Capability summary: CONTEXT.md and documents/GPU_VALIDATION_MATRIX.md");
  addNotes(slide,
    "算例名称用于说明能力层次。TGV、2dvort 和 HIT 是基础周期流动；Channel 与 LDC 引入壁面；RTI 和 Sod 系列引入源项与间断；Mach 5 边界层与 CURVE-C15 组合曲线网格、黏性扩散、激波传感器和选择性 Roe。",
    ["CONTEXT.md", "documents/GPU_VALIDATION_MATRIX.md", "tests/gpu_validation/README.md"]);
}

// 5. Correctness
{
  const slide = pptx.addSlide();
  addTitle(slide, "代表性 CPU/GPU 差异均低于验收门槛", "Correctness result", 5);
  slide.addImage({ path: `${ASSET}/numerical_equivalence_errors.jpeg`, x: 0.72, y: 1.26, w: 8.35, h: 4.70 });
  addStat(slide, 9.42, 1.48, 2.78, "≤10⁻¹⁰", "统一主要门槛", C.red, "同相位场/统计量");
  addBand(slide, 9.42, 3.02, 2.78, 1.26, "比较内容", "逐场最大差值\n全局统计量\n边界不变量", C.blue);
  addBand(slide, 9.42, 4.56, 2.78, 1.26, "额外检查", "Jacobian 为正\n激波 mask 一致\nSanitizer 0 error", C.teal);
  addText(slide, "先证明“GPU 算的是同一个问题”，再讨论速度。", 1.08, 6.37, 11.15, 0.42, {
    fontSize: 20, bold: true, align: "center",
  });
  addFooter(slide, "Representative maxima from GPU_VALIDATION_MATRIX; markers are acceptance limits");
  addNotes(slide,
    "图中每根柱子取该验证门中最不利的已记录差值。CURVE-C12 的统计差异约 5e-11，仍低于 1e-10。该图说明 CPU/GPU 数值等价，不代表物理模型已经被实验数据验证。",
    ["documents/GPU_VALIDATION_MATRIX.md", "documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md"]);
}

// 6. Physical credibility
{
  const slide = pptx.addSlide();
  addTitle(slide, "曲线网格不只“能跑”，网格加密后结果变化在缩小", "Physical consistency", 6);
  slide.addImage({ path: `${ASSET}/curvilinear_mesh.jpeg`, x: 0.75, y: 1.30, w: 4.42, h: 4.58 });
  slide.addImage({ path: `${ASSET}/hbl_refinement_ratios.jpeg`, x: 5.48, y: 1.30, w: 6.78, h: 4.58 });
  addLabel(slide, "静态非正交曲线网格", 1.55, 5.91, 2.80, C.blue, C.blueLight);
  addLabel(slide, "全部比值 < 1", 7.48, 5.91, 2.70, C.teal, C.tealLight);
  addText(slide, "三套网格、两种时间步、冷热壁面均通过：误差趋势不是靠单一网格判断。", 0.95, 6.37, 11.45, 0.42, {
    fontSize: 18, bold: true, align: "center",
  });
  addFooter(slide, "CURVE-C14: 96×96×8, 144×144×8, 192×192×8; matched final time 2e-4");
  addNotes(slide,
    "纵轴是细网格变化量与粗网格变化量之比。小于 1 表明继续加密后结果改变量在缩小。C14 同时检查壁面摩擦、冷壁热流和速度/温度剖面。热壁热流接近零点，因此只作为信息项，不用不稳定的相对误差作验收。",
    ["tests/gpu_validation/out/curvilinear_hbl_physical_dual_after_reduction_20260904", "tests/gpu_validation/analyze_curvilinear_hbl_physics.py", "tests/gpu_validation/summarize_curvilinear_hbl_refinement.py"]);
}

// 7. Performance
{
  const slide = pptx.addSlide();
  addTitle(slide, "单卡显著缩短计算时间，双卡在大网格上继续受益", "Measured performance", 7);
  slide.addImage({ path: `${ASSET}/latest_performance_summary.jpeg`, x: 0.72, y: 1.27, w: 11.84, h: 4.66 });
  addText(slide, "37.84×", 1.55, 5.83, 2.20, 0.50, { fontSize: 27, bold: true, color: C.red, align: "center" });
  addText(slide, "Channel：单 GPU 相对单 CPU", 0.95, 6.29, 3.40, 0.34, { fontSize: 13.5, align: "center" });
  addText(slide, "1.58×", 8.80, 5.83, 2.20, 0.50, { fontSize: 27, bold: true, color: C.teal, align: "center" });
  addText(slide, "256³ C10：双 GPU 相对单 GPU", 8.12, 6.29, 3.55, 0.34, { fontSize: 13.5, align: "center" });
  addFooter(slide, "Left: Channel 128³/100 steps. Right: latest C10 256³/6 RK advances, three repeats");
  addNotes(slide,
    "左图和右图回答两个不同问题。左图比较 CPU 与 GPU 的整体加速，右图只比较一张和两张 GPU。右图最新三次中位数为 57.517 秒和 36.487 秒，对应 1.5764 倍加速和 78.82% 并行效率。",
    ["tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903", "tests/gpu_validation/out/curvilinear_hbl_c10_256_benchmark_after_reduction_20260904/benchmark_summary.md"]);
}

// 8. Residency
{
  const slide = pptx.addSlide();
  addTitle(slide, "GPU 确实在持续计算，统计量不再搬回大数组", "Residency evidence", 8);
  addBand(slide, 0.82, 1.44, 4.15, 2.02, "发现的问题", "256³ 首次 profile 发现统计归约每次回传约 3.0 MiB block partials。\n这不是整场，但违反驻留门槛。", C.red, C.redLight);
  addArrow(slide, 5.18, 2.42, 0.90, 0, C.muted);
  addBand(slide, 6.28, 1.44, 4.15, 2.02, "修复后的路径", "第二级归约留在 GPU 上完成。\n主机只收到 6 个 double，共 48 B。", C.teal, C.tealLight);
  addStat(slide, 10.72, 1.44, 1.78, "48 B", "最大 D2H", C.teal, "RK 区间");
  addStat(slide, 0.82, 4.15, 3.15, "0", "≥64 KiB 传输", C.blue, "首个 RHS 后");
  addStat(slide, 4.25, 4.15, 3.15, "100%", "峰值 GPU 利用率", C.red, "NP=1 与 NP=2");
  addStat(slide, 7.68, 4.15, 3.15, "0 errors", "Compute Sanitizer", C.teal, "归约核专项检查");
  addText(slide, "性能结果有 profiler 与 nvitop 的运行证据支撑，不是仅凭程序“跑完”推断。", 0.92, 6.30, 11.48, 0.44, {
    fontSize: 18, bold: true, align: "center",
  });
  addFooter(slide, "C15 Nsight Systems residency interval and nvitop runtime observation");
  addNotes(slide,
    "C15 是本次相对旧稿的关键新增。原来的第一层归约产生 65536 乘 6 个 double，合计 3,145,728 字节。新实现使用第二个 GPU kernel 归约到 6 个标量。最新 trace 在 221 个 kernel 的 RK 区间内只有两次 176 字节 H2D 和两次 48 字节 D2H。",
    ["src_gpu/statistic_gpu.cuf", "tests/gpu_validation/out/curvilinear_hbl_c10_256_residency_after_reduction_20260904/residency_report.txt", "tests/gpu_validation/out/curvilinear_hbl_c10_256_residency_after_reduction_20260904/nsys_summary.txt"]);
}

// 9. Usefulness
{
  const slide = pptx.addSlide();
  addTitle(slide, "当前版本已经能支撑三类研究工作", "What it enables", 9);
  addBand(slide, 0.82, 1.40, 3.72, 3.72, "1  快速基准与回归", "TGV、HIT、Channel 等算例可以快速重复运行。\n\n适合检查新格式、新边界或新 kernel 是否改变原有结果。", C.blue);
  addBand(slide, 4.80, 1.40, 3.72, 3.72, "2  多 GPU 大网格", "固定 halo 的 x/y/z 分解均已验证。\n\n256³ C10 已证明两张 GPU 能同时工作并缩短时间。", C.teal);
  addBand(slide, 8.78, 1.40, 3.72, 3.72, "3  复杂流动研发", "曲线网格、壁面、远场、激波传感器和选择性 Roe 可以组合。\n\n为真实高超声速边界层与 SBLI 研发提供基础。", C.red);
  addText(slide, "项目价值已经从“移植一个算例”提升为“建立可复用、可验证的 GPU 求解底座”。", 0.98, 5.85, 11.40, 0.58, {
    fontSize: 21, bold: true, align: "center",
  });
  addFooter(slide, "Validated scope only; production SBLI and broader physics remain future work");
  addNotes(slide,
    "这页直接回答有什么用。当前程序可用于高频回归、两卡大网格计算和复杂流动能力开发。它还不是任意工程外形求解器，也没有完成生产级 SBLI 物理对标。",
    ["documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md", "documents/GPU_VALIDATION_MATRIX.md"]);
}

// 10. Limits and next step
{
  const slide = pptx.addSlide();
  addTitle(slide, "下一阶段不是继续堆算例，而是扩大可信使用范围", "Next step", 10);
  addBand(slide, 0.82, 1.35, 5.65, 1.30, "优先 1：真实复杂工况", "用可解释的 SBLI 或高超声速边界层工况做网格/时间收敛、激波位置和壁面热流验证。", C.red);
  addBand(slide, 6.82, 1.35, 5.65, 1.30, "优先 2：多卡通信效率", "把 host-staged halo 逐步升级为 pinned、nonblocking，并为 CUDA-aware MPI 保留后端。", C.blue);
  addBand(slide, 0.82, 3.00, 5.65, 1.30, "优先 3：曲线边界泛化", "继续坚持几何投影，把已验证的六面 symmetry 方法推广到更多物理边界。", C.teal);
  addBand(slide, 6.82, 3.00, 5.65, 1.30, "优先 4：可移植后端", "保持求解器与通信后端分离，为未来 AMD/HIP/DCU 适配减少重复改造。", C.amber);
  addText(slide, "暂不进入", 0.88, 5.05, 1.10, 0.36, { fontSize: 15, bold: true, color: C.red });
  addText(slide, "compact · RANS/LES · 化学反应/燃烧 · IBM · 动网格/多块 · GPU HDF5", 2.08, 5.05, 9.80, 0.36, {
    fontSize: 15, color: C.muted,
  });
  addText(slide, "原则：每扩大一种物理能力，仍然先建立 CPU/GPU 同相位与物理诊断门槛。", 0.94, 6.13, 11.45, 0.48, {
    fontSize: 19, bold: true, align: "center",
  });
  addFooter(slide, "Near-term scope in ASTR_FULL_GPU_ARCHITECTURE_PLAN.md");
  addNotes(slide,
    "下一阶段建议围绕两条主线：真实复杂工况的物理可信化，以及通信后端性能化。多组分、化学、湍流模型、IBM、动网格和 GPU HDF5 当前都不应混入主线，否则会同时扩大数值、物理和工程风险。",
    ["documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md", "CONTEXT.md"]);
}

// 11. Plain-language close
{
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  addText(slide, "大白话总结", 0.82, 0.72, 4.30, 0.62, { fontSize: 30, color: C.paper, bold: true });
  addText(slide, "以前", 0.88, 1.74, 1.15, 0.46, { fontSize: 19, bold: true, color: C.red });
  addText(slide, "ASTR 只能依靠 CPU，算得慢，GPU 也没有完整求解路径。", 2.20, 1.68, 9.65, 0.60, { fontSize: 23, color: C.paper, bold: true });
  addText(slide, "现在", 0.88, 3.05, 1.15, 0.46, { fontSize: 19, bold: true, color: C.teal });
  addText(slide, "主要计算能留在 GPU，多张卡能一起算，曲线网格和激波路径也有验证。", 2.20, 2.98, 9.65, 0.72, { fontSize: 23, color: C.paper, bold: true });
  addText(slide, "接下来", 0.88, 4.52, 1.35, 0.46, { fontSize: 19, bold: true, color: C.blue });
  addText(slide, "要证明它不仅算得快、算得一样，还能在真实复杂工况中算得可信。", 2.40, 4.45, 9.45, 0.72, { fontSize: 23, color: C.paper, bold: true });
  slide.addShape(pptx.ShapeType.line, { x: 0.86, y: 5.77, w: 11.48, h: 0, line: { color: "53636B", width: 1 } });
  addText(slide, "结论：GPU 移植主线已经成立，下一阶段转向真实复杂物理与生产级扩展。", 0.92, 6.08, 11.42, 0.62, {
    fontSize: 22, color: "D7E3E8", bold: true, align: "center",
  });
  addNotes(slide,
    "结束时只重复三句话：以前只有 CPU；现在 GPU 主线、多卡和曲线激波能力已经建立；下一步要把工程正确性推进到真实复杂工况的物理可信度。",
    ["Summary of slides 1-10"]);
}

// 12. Appendix divider
{
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  addText(slide, "附录", 0.86, 1.45, 3.20, 0.70, { fontSize: 38, color: C.paper, bold: true });
  addText(slide, "技术细节与可追溯证据", 0.86, 2.30, 7.20, 0.58, { fontSize: 23, color: "D7E3E8", bold: true });
  addText(slide, "提交记录 · 架构 · 数值格式 · 边界与算例 · MPI · 验证 · 性能 · Profiling", 0.86, 3.32, 11.20, 0.45, {
    fontSize: 17, color: "AFC2CB",
  });
  addText(slide, "仅在问答时展开", 0.86, 5.55, 3.25, 0.44, { fontSize: 17, color: C.red, bold: true });
  addNotes(slide, "附录不在主线中逐页讲解，只在提问涉及具体提交、格式、边界、MPI 或性能协议时使用。", ["Local project evidence"]);
}

// 13. Commit ledger
{
  const slide = pptx.addSlide();
  addTitle(slide, "12 次提交把纯 CPU 程序推进到 CURVE-C15", "Appendix A · commits", 13);
  const rows = [
    ["47737cf", "纯 CPU 基线", "原始 Fortran/MPI 求解器"],
    ["df1961b", "GPU 主线起步", "TGV、设备字段与 x-slab"],
    ["5597e69 / 7a5fdce", "多向 halo 与工程规范", "x/y/z、计划、ADR、验证"],
    ["fd250ab / b5ed8c3", "非 TGV 算例", "2dvort、HIT、zero-extrap"],
    ["451fe0e / a684edb", "壁面与源项", "Channel、LDC、RTI、wall family"],
    ["270be49 / 16ded59", "激波与高超声速", "Sod、Shu–Osher、HBL、Roe、sponge"],
    ["aecb18f / 63fe7a8", "曲线网格 C0-C9", "NSCBC、滤波、传感器、选择性 Roe"],
    ["fa36a0d", "曲线网格 C10-C15", "黏性 Roe、六面几何、物理/性能收口"],
  ];
  rows.forEach((row, i) => addRow(slide, 1.18 + i * 0.67, row[0], row[1], row[2], i === 0 ? C.red : i === rows.length - 1 ? C.teal : C.blue));
  addText(slide, "累计：231 个文件变更，+33,941 / −2,730 行", 0.92, 6.56, 11.30, 0.34, { fontSize: 16, bold: true, align: "center" });
  addFooter(slide, "git log and git diff --shortstat 47737cf..fa36a0d");
  addNotes(slide,
    "行数只反映工作量，不等同于功能价值。最新 fa36a0d 单次提交包含 30 个文件、2082 行新增和 85 行删除，重点是 C10-C15。",
    ["git log 47737cf..fa36a0d", "git diff --shortstat 47737cf..fa36a0d"]);
}

// 14. Architecture
{
  const slide = pptx.addSlide();
  addTitle(slide, "CPU 保留输入输出与基准，GPU 接管计算主循环", "Appendix B · architecture", 14);
  addBand(slide, 0.80, 1.32, 3.58, 3.95, "CPU 前端与基准", "src/\n\n输入、网格、初始化\nCPU reference solver\nHDF5/checkpoint\nMPI 拓扑建立", C.red);
  addArrow(slide, 4.56, 3.05, 0.70, 0, C.muted);
  addBand(slide, 5.45, 1.32, 3.58, 3.95, "GPU facade 与能力门控", "gpu_runtime\ncase_capability_gpu\n\n决定当前工况是否进入 GPU\n拒绝未验证组合", C.blue);
  addArrow(slide, 9.21, 3.05, 0.70, 0, C.muted);
  addBand(slide, 10.10, 1.32, 2.45, 3.95, "CUDA 后端", "src_gpu/\n\n边界\n梯度\n对流/扩散\n滤波/RK\n统计/源项", C.teal);
  addText(slide, "低限度修改 src/，GPU 实现集中在 src_gpu/；未来通信与 HIP/DCU 后端可独立演进。", 0.94, 5.91, 11.45, 0.52, {
    fontSize: 18, bold: true, align: "center",
  });
  addFooter(slide, "Source boundaries: src/ and src_gpu/ at fa36a0d");
  addNotes(slide,
    "CPU 代码仍然是主要 oracle，并负责尚未 GPU 化的文件边界。GPU facade 避免 src/ 直接依赖大量 CUDA 模块。能力门控确保未验证的数值/边界组合明确拒绝，而不是静默进入错误路径。",
    ["src/astr.F90", "src/mainloop.F90", "src_gpu/gpu_runtime.cuf", "src_gpu/case_capability_gpu.cuf"]);
}

// 15. Numerical methods
{
  const slide = pptx.addSlide();
  addTitle(slide, "已移植的是显式高阶路径，不包含 compact 线性求解", "Appendix C · numerics", 15);
  addRow(slide, 1.28, "对流", "六阶显式中心", "平滑区基础格式；物理边界采用 CPU 对应闭合", C.blue);
  addRow(slide, 2.05, "激波重构", "WENO7 / MP7", "可选物理空间或 Roe 特征空间重构", C.red);
  addRow(slide, 2.82, "激波判断", "Ducros sensor", "仅传感器标记区域进入 selective Roe", C.amber);
  addRow(slide, 3.59, "黏性项", "六阶显式中心", "曲线度量 + Sutherland 黏度", C.teal);
  addRow(slide, 4.36, "滤波", "十阶显式中心", "ping-pong；物理边界 0-6-6-6-8-10 闭合", C.blue);
  addRow(slide, 5.13, "时间推进", "三阶 Runge–Kutta", "每个 GPU kernel 后按当前约束显式同步", C.red);
  addText(slide, "不移植：紧致差分、三对角/五对角求解。", 0.94, 6.30, 11.40, 0.40, { fontSize: 18, bold: true, align: "center" });
  addFooter(slide, "Numerical scope fixed by project decisions and current capability gates");
  addNotes(slide,
    "显式格式更适合当前 GPU 路径，因为不需要沿整条网格线求解线性系统。选择性 Roe 只在 shock mask 触发处进入特征空间，平滑区保留物理空间重构。",
    ["documents/ASTR_CPU_NUMERICAL_SCHEMES.md", "documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md", "src_gpu/solver_gpu.cuf"]);
}

// 16. Cases and boundaries
{
  const slide = pptx.addSlide();
  addTitle(slide, "算例与边界按“验证过的组合”开放", "Appendix D · cases and BC", 16);
  addBand(slide, 0.78, 1.32, 3.66, 4.30, "已验证算例", "TGV / 2dvort / HIT\nChannel / LDC / RTI\nSod / Shu–Osher\nMach 5 boundary layer\ncontrolled curved SBLI path", C.blue);
  addBand(slide, 4.83, 1.32, 3.66, 4.30, "已覆盖边界族", "1 periodic\n0 UDF lid\n11/12 inflow\n21 outflow\n31 fixed\n41/42 no-slip\n411/421 slip\n50 zero-extrap\n51/52 farfield\n60 symmetry", C.red);
  addBand(slide, 8.88, 1.32, 3.66, 4.30, "使用规则", "勾选一种 bctype 不代表任意方向都可用。\n\n能力门控同时检查方向、格式、物性、MPI 拓扑和 sponge。\n\n未验证组合直接拒绝。", C.teal);
  addText(slide, "142 条 pass 记录是验证证据，不等于 142 个独立物理算例。", 0.96, 6.20, 11.40, 0.42, { fontSize: 18, bold: true, align: "center" });
  addFooter(slide, "Current matrix: 142 pass rows in documents/GPU_VALIDATION_MATRIX.md");
  addNotes(slide,
    "边界方向限制来自 CPU 现有实现和已完成验证。例如 42 当前只支持 x/y，411 和 421 当前只支持 y。曲线几何优先使用局部法向投影，不能把 Cartesian 分量钳制直接推广到曲线面。",
    ["documents/GPU_VALIDATION_MATRIX.md", "src_gpu/case_capability_gpu.cuf", "src_gpu/boundary_gpu.cuf"]);
}

// 17. MPI
{
  const slide = pptx.addSlide();
  addTitle(slide, "MPI 正确性覆盖三方向，性能证据限定两张物理 GPU", "Appendix E · MPI", 17);
  addBand(slide, 0.82, 1.38, 3.40, 2.00, "方向覆盖", "x-slab：2×1×1\ny-slab：1×2×1\nz-slab：1×1×2", C.blue);
  addBand(slide, 4.48, 1.38, 3.40, 2.00, "组合覆盖", "NP=4：二维平面\nNP=8：2×2×2\nNP=27：3×3×3 内部 rank", C.red);
  addBand(slide, 8.14, 1.38, 4.10, 2.00, "当前硬件解释", "机器只有两张 GPU。\nNP>2 是共享 GPU 的正确性压力测试，不是多卡扩展曲线。", C.teal);
  addText(slide, "每个 rank 固定绑定一个设备；只交换 hm 层 halo；物理边界由 MPI_PROC_NULL 判定。", 0.92, 4.23, 11.52, 0.50, {
    fontSize: 19, bold: true, align: "center",
  });
  addArrow(slide, 1.62, 5.35, 9.90, 0, C.muted);
  addLabel(slide, "pack", 1.08, 5.70, 1.35, C.blue, C.blueLight);
  addLabel(slide, "host-staged MPI", 3.18, 5.70, 2.08, C.red, C.redLight);
  addLabel(slide, "unpack", 6.02, 5.70, 1.35, C.blue, C.blueLight);
  addLabel(slide, "future CUDA-aware", 8.18, 5.70, 2.32, C.teal, C.tealLight);
  addFooter(slide, "Correctness topology coverage is broader than physical-GPU performance coverage");
  addNotes(slide,
    "NP 表示 MPI rank 数量，不自动等于 GPU 数量。NP=8 和 NP=27 在本机只用于检查三方向邻居、内部 rank 和边界 ownership。当前正式性能证据只有 NP=1 单卡和 NP=2 双卡。",
    ["src_gpu/halo_exchange_gpu.cuf", "documents/ASTR_GPU_MULTI_RANK_PORTING_PLAN.md", "documents/GPU_VALIDATION_MATRIX.md"]);
}

// 18. Validation ladder
{
  const slide = pptx.addSlide();
  addTitle(slide, "验证不是“程序跑完”，而是六层证据", "Appendix F · validation", 18);
  const layers = [
    ["L1", "编译与 smoke", "程序可启动、无 NaN/崩溃", C.blue],
    ["L2", "CPU/GPU 同相位", "逐场与统计量差异", C.red],
    ["L3", "边界与几何", "壁面不变量、Jacobian、法向投影", C.teal],
    ["L4", "物理诊断", "精确解、Blasius、网格/时间趋势", C.amber],
    ["L5", "内存正确性", "Compute Sanitizer 0 error", C.red],
    ["L6", "性能与驻留", "重复计时、nvitop、Nsight transfers", C.blue],
  ];
  layers.forEach((row, i) => {
    const y = 1.15 + i * 0.86;
    addLabel(slide, row[0], 0.84, y + 0.08, 0.72, row[3], row[3] === C.red ? C.redLight : row[3] === C.teal ? C.tealLight : row[3] === C.amber ? C.amberLight : C.blueLight);
    addText(slide, row[1], 1.84, y, 3.12, 0.42, { fontSize: 17, bold: true, color: row[3] });
    addText(slide, row[2], 5.10, y, 6.95, 0.42, { fontSize: 15, color: C.ink });
  });
  addText(slide, "顺序不能倒置：正确性未通过时，不讨论加速比。", 0.95, 6.43, 11.42, 0.40, { fontSize: 18, bold: true, align: "center" });
  addFooter(slide, "Validation philosophy encoded in tests/gpu_validation and GPU_VALIDATION_MATRIX.md");
  addNotes(slide,
    "每一级回答不同问题。CPU/GPU 一致只说明移植一致；物理诊断说明离散结果是否呈合理趋势；Sanitizer 检查越界和非法访问；Nsight 与 nvitop 证明计算和数据驻留。",
    ["tests/gpu_validation/README.md", "documents/GPU_VALIDATION_MATRIX.md"]);
}

// 19. Benchmark detail
{
  const slide = pptx.addSlide();
  addTitle(slide, "性能数字对应两组明确的测试协议", "Appendix G · benchmark", 19);
  addBand(slide, 0.80, 1.35, 5.70, 3.58, "Channel：CPU/GPU 整体加速", "128³ · MAXSTEP=100\n固定驱动力 1e-4\n六阶显式中心 + diffusion\nCPU NP=1：526.677 s（1 次）\nGPU NP=1：13.641 / 13.917 / 14.552 s\n中位加速：37.84×", C.red);
  addBand(slide, 6.82, 1.35, 5.70, 3.58, "C10：最新双 GPU 扩展", "256³ · 实际 6 个 RK advances\n曲线网格 + Ducros + Roe MP7\n六阶 diffusion + Sutherland\nGPU NP=1：57.297 / 57.517 / 57.712 s\nGPU NP=2：36.320 / 36.487 / 36.548 s\n双卡加速：1.576×", C.teal);
  addText(slide, "两组数据回答不同问题，不能把 1.58× 当成相对 CPU 的加速。", 0.95, 5.48, 11.45, 0.46, { fontSize: 19, bold: true, align: "center" });
  addText(slide, "共同硬件：2× RTX 4000 Ada 19 GB；壁钟时间包含初始化与输出。", 0.95, 6.12, 11.45, 0.38, { fontSize: 15.5, color: C.muted, align: "center" });
  addFooter(slide, "Raw timing files retained under tests/gpu_validation/out");
  addNotes(slide,
    "Channel 的 CPU 基线只测一次，因此 37.84 倍是工程证据，不是发表级统计。C10 每个 GPU 配置有预热和三次有效计时，且报告 min/median/max。两组都包含 CPU-owned 文件边界开销。",
    ["tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903", "tests/gpu_validation/out/curvilinear_hbl_c10_256_benchmark_after_reduction_20260904/timings.tsv"]);
}

// 20. Profiling and defect decisions
{
  const slide = pptx.addSlide();
  addTitle(slide, "Profiler 用于定位瓶颈，CPU 问题必须先人工决策", "Appendix H · profiling and audit", 20);
  addBand(slide, 0.80, 1.30, 5.68, 2.25, "最新 C15 驻留证据", "221 个 kernel 后续区间\nH2D：2 × 176 B\nD2H：2 × 48 B\n≥64 KiB：0\n第二级归约 kernel：约 77.6 μs/次", C.blue);
  addBand(slide, 6.82, 1.30, 5.68, 2.25, "已按流程修复的 CPU 确定性问题", "显式滤波物理边界闭合\n三维 HBL 壁面积统计\nNSCBC transverse halo 时序\nDucros 双物理侧索引\n曲线 symmetry 上壁法向", C.red);
  addBand(slide, 0.80, 4.02, 5.68, 1.50, "决策门", "发现 CPU 明确 bug：停止 → 报告 → 人工确认 → 修 CPU → 同步 GPU。", C.red, C.redLight);
  addBand(slide, 6.82, 4.02, 5.68, 1.50, "不扩大结论", "NSCBC case-specific 选择、全域激波滤波、未验证方向均保持关闭。", C.teal, C.tealLight);
  addText(slide, "保留 CPU oracle，但不把 CPU 历史实现当成不可质疑的真值。", 0.95, 6.22, 11.44, 0.46, { fontSize: 19, bold: true, align: "center" });
  addFooter(slide, "Audit trail: architecture plan, validation matrix, source fixes, Nsight report");
  addNotes(slide,
    "旧 CPU 程序在物理边界滤波、三维壁面积、NSCBC halo 时序、Ducros 索引和 symmetry 法向上暴露过确定性问题。所有明确修复都先经过人工确认。对物理语义不唯一的 NSCBC 分支不做擅自泛化。",
    ["documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md", "documents/GPU_VALIDATION_MATRIX.md", "tests/gpu_validation/out/curvilinear_hbl_c10_256_residency_after_reduction_20260904/nsys_summary.txt"]);
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
