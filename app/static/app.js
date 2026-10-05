const UNSURE_BELOW = 0.75;
const HISTORY_LIMIT = 12;

const $ = (id) => document.getElementById(id);
const pct = (p) => `${Math.round(p * 100)}%`;

let samples = [];
let classes = {};
let requestSeq = 0;
const sampleResults = new Map(); // sample id -> correct?

async function init() {
  wireInputs();
  const [status, sampleList] = await Promise.all([
    fetch("/api/status").then((r) => r.json()),
    fetch("/api/samples").then((r) => r.json()),
  ]);
  classes = status.classes;
  renderModelChip(status);
  renderAbout(status);
  samples = sampleList;
  renderSamples("all");
  if (samples.length) classifySample(samples[0]);
}

function renderModelChip({ backend, description }) {
  const chip = $("model-chip");
  chip.dataset.backend = backend;
  if (backend === "trained") {
    $("model-label").textContent = "Trained model";
    $("model-detail").innerHTML = `<p>${escapeHtml(description)}.</p><p>Predictions come from the model trained on your dataset.</p>`;
  } else {
    $("model-label").textContent = "Demo model";
    $("model-detail").innerHTML = `
      <p>No trained model found, so results come from a stock ImageNet model that votes "organic" when it sees food or plants.</p>
      <p>Train the real classifier, then restart the app:</p>
      <p><code>python -m waste_classifier.train --data path/to/DATASET</code></p>`;
  }
}

const MODEL_BLURBS = {
  waste_mobilenet: "a MobileNetV2 network pre-trained on ImageNet, with a new classification head",
  waste_cnn: "a small convolutional neural network trained from scratch",
};

function renderAbout({ backend, model_name, metrics }) {
  const note = $("about-score");
  if (backend !== "trained") {
    $("about-model").textContent = "a stock ImageNet model for now (demo mode). The real classifier is";
    note.textContent = "No trained model is loaded, so there is no accuracy score yet.";
    return;
  }
  $("about-model").textContent = MODEL_BLURBS[model_name] ?? "a convolutional neural network";
  if (!metrics) {
    note.textContent = "Accuracy not recorded for this model. Run python -m waste_classifier.evaluation to score it.";
    return;
  }
  const n = metrics.n_test.toLocaleString("en");
  note.textContent = `Test accuracy: ${(metrics.accuracy * 100).toFixed(1)}% · balanced accuracy ${(metrics.balanced_accuracy * 100).toFixed(1)}% · ` +
    `macro F1 ${metrics.macro_f1.toFixed(3)} · ROC AUC ${metrics.roc_auc.toFixed(3)} on ${n} held-out test images.`;
}

function renderSamples(filter) {
  const grid = $("sample-grid");
  grid.replaceChildren();
  for (const s of samples) {
    if (filter !== "all" && s.expected !== filter) continue;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample";
    btn.id = `sample-${s.id}`;
    btn.dataset.id = s.id;
    btn.setAttribute("aria-label", `Classify sample: ${s.name}`);
    btn.innerHTML = `
      <img src="/samples/${s.file}" alt="" loading="lazy">
      <span class="sample-meta">
        <span class="sample-name">${escapeHtml(s.name)}</span>
        <span class="sample-bin" data-as="${s.expected}" title="${escapeHtml(classes[s.expected]?.bin ?? "")}"></span>
      </span>`;
    btn.addEventListener("click", () => classifySample(s));
    grid.append(btn);
  }
  markActiveSample();
}

let activeSampleId = null;
function markActiveSample() {
  document.querySelectorAll(".sample").forEach((el) => el.classList.toggle("is-active", el.dataset.id === activeSampleId));
}

async function classifySample(sample) {
  activeSampleId = sample.id;
  markActiveSample();
  const blob = await fetch(`/samples/${sample.file}`).then((r) => r.blob());
  const credit = `Photo: <a href="${sample.source}" target="_blank" rel="noopener">${escapeHtml(sample.author)}</a>, ${escapeHtml(sample.license)}`;
  classify(blob, { sample, credit });
}

async function classifyFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("That file isn't an image. Use a JPG, PNG or WebP photo.");
    return;
  }
  activeSampleId = null;
  markActiveSample();
  classify(await shrinkImage(file), { credit: escapeHtml(file.name || "Pasted image") });
}

// The model only looks at 224x224 pixels, and the hosted app caps uploads at ~6 MB,
// so scale big photos down in the browser before sending them.
const MAX_SIDE = 1280;
async function shrinkImage(file) {
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = MAX_SIDE / Math.max(bitmap.width, bitmap.height);
    if (scale >= 1 && file.size < 1_500_000) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * Math.min(scale, 1));
    canvas.height = Math.round(bitmap.height * Math.min(scale, 1));
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    return blob ? new File([blob], (file.name || "photo").replace(/\.\w+$/, "") + ".jpg", { type: "image/jpeg" }) : file;
  } catch {
    return file; // the browser can't decode it (e.g. HEIC); let the server try
  }
}

async function classify(blob, { sample = null, credit = "" } = {}) {
  const seq = ++requestSeq;
  const url = URL.createObjectURL(blob);
  const preview = $("preview");
  preview.src = url;
  preview.alt = sample ? sample.name : "Uploaded item";
  preview.hidden = false;
  $("dropzone").classList.add("has-image", "is-busy");
  $("credit").innerHTML = credit;
  $("error").hidden = true;

  const body = new FormData();
  body.append("file", blob, sample ? sample.file : blob.name || "upload.jpg");
  try {
    const res = await fetch("/api/classify", { method: "POST", body });
    const data = await res.json();
    if (seq !== requestSeq) return; // a newer image superseded this one
    if (!res.ok) throw new Error(data.detail || `The server returned ${res.status}.`);
    renderResult(data, sample);
    addHistory(url, data, sample);
  } catch (err) {
    if (seq !== requestSeq) return;
    showError(err.message === "Failed to fetch" ? "Can't reach the classifier. Check the server is still running." : err.message);
  } finally {
    if (seq === requestSeq) $("dropzone").classList.remove("is-busy");
  }
}

function renderResult(r, sample) {
  const bin = $("bin");
  bin.dataset.label = r.label;
  $("bin-class").textContent = r.name;
  $("bin-name").textContent = r.bin;
  $("bin-conf").textContent = `${pct(r.confidence)} confident`;
  $("unsure").hidden = r.confidence >= UNSURE_BELOW;

  const pO = r.probabilities.O ?? 0;
  const pR = r.probabilities.R ?? 0;
  $("split-o").style.width = pct(pO);
  $("split-r").style.width = pct(pR);
  $("pct-o").textContent = pct(pO);
  $("pct-r").textContent = pct(pR);
  $("advice").textContent = r.advice;

  const check = $("check");
  if (sample) {
    const correct = sample.expected === r.label;
    sampleResults.set(sample.id, correct);
    check.hidden = false;
    check.className = `check ${correct ? "is-match" : "is-miss"}`;
    check.textContent = correct
      ? `✓ Correct: this sample belongs in the ${classes[sample.expected].bin.toLowerCase()}.`
      : `✗ Wrong: this sample belongs in the ${classes[sample.expected].bin.toLowerCase()}.`;
    updateScore();
  } else {
    check.hidden = true;
  }

  const wrap = $("evidence-wrap");
  wrap.hidden = !r.evidence.length;
  $("evidence").replaceChildren(...r.evidence.map((e) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="ev-name">${escapeHtml(e.label)}</span>
      <span class="ev-bar"><span data-as="${e.counts_as}" style="width:${pct(e.score)}"></span></span>
      <span class="ev-score">${pct(e.score)}</span>
      <span class="tag" data-as="${e.counts_as}">${e.counts_as}</span>`;
    return li;
  }));
}

function updateScore() {
  const tried = sampleResults.size;
  const right = [...sampleResults.values()].filter(Boolean).length;
  $("score").textContent = `${right} of ${tried} sample${tried === 1 ? "" : "s"} correct · ${samples.length - tried} untried`;
}

function addHistory(url, r, sample) {
  $("history-section").hidden = false;
  const li = document.createElement("li");
  li.dataset.label = r.label;
  li.innerHTML = `<img src="${url}" alt="${escapeHtml(sample ? sample.name : "Uploaded item")}"><span>${r.label} · ${pct(r.confidence)}</span>`;
  const list = $("history");
  list.prepend(li);
  while (list.children.length > HISTORY_LIMIT) {
    const old = list.lastElementChild;
    const oldUrl = old.querySelector("img").src;
    if (oldUrl !== $("preview").src) URL.revokeObjectURL(oldUrl);
    old.remove();
  }
}

function showError(message) {
  const el = $("error");
  el.textContent = message;
  el.hidden = false;
  $("dropzone").classList.remove("is-busy");
}

function wireInputs() {
  $("file-input").addEventListener("change", (e) => {
    classifyFile(e.target.files[0]);
    e.target.value = "";
  });

  const dz = $("dropzone");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("is-over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("is-over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("is-over");
    classifyFile(e.dataTransfer.files[0]);
  });
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); }
  });
  document.addEventListener("paste", (e) => {
    const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
    if (item) classifyFile(item.getAsFile());
  });

  document.querySelectorAll(".chip[data-filter]").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip[data-filter]").forEach((c) => c.classList.toggle("is-on", c === chip));
      renderSamples(chip.dataset.filter);
    });
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

init().catch(() => showError("Couldn't load the app data. Check the server is running and reload."));
