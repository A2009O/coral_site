const fileInput = document.getElementById("file");
const btn = document.getElementById("btn");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const progressFill = document.getElementById("progressFill");
const progressPct = document.getElementById("progressPct");
const spinner = document.getElementById("spinner");
const errorBox = document.getElementById("error");

const originalImg = document.getElementById("originalImg");
const resultImg = document.getElementById("resultImg");
const rawJson = document.getElementById("rawJson");
const summary = document.getElementById("summary");

// controls
const minConf = document.getElementById("minConf");
const minArea = document.getElementById("minArea");
const nmsIoU = document.getElementById("nmsIoU");
const maxTotal = document.getElementById("maxTotal");
const maxPerClass = document.getElementById("maxPerClass");
const smallMode = document.getElementById("smallMode");

// labels
const minConfVal = document.getElementById("minConfVal");
const minAreaVal = document.getElementById("minAreaVal");
const nmsVal = document.getElementById("nmsVal");
const maxTotalVal = document.getElementById("maxTotalVal");
const maxPerClassVal = document.getElementById("maxPerClassVal");

function setState(state, text) {
  statusDot.className = "dot " + state;
  statusText.textContent = text;
}

function setProgress(p) {
  const pct = Math.max(0, Math.min(100, p));
  progressFill.style.width = pct + "%";
  progressPct.textContent = pct.toFixed(0) + "%";
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
  setState("error", "Error");
  spinner.classList.add("hidden");
}

function clearError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function detectionsSummary(preds, counts) {
  if (!preds || preds.length === 0) return `No detections found. (raw=${counts?.raw ?? "?"})`;
  const countsByClass = {};
  preds.forEach(p => {
    const c = p.class || "object";
    countsByClass[c] = (countsByClass[c] || 0) + 1;
  });
  const parts = Object.entries(countsByClass).map(([k, v]) => `${k}: ${v}`);
  const rawCount = counts?.raw ?? "?";
  const finalCount = counts?.final ?? preds.length;
  return `Raw: ${rawCount} → Final: ${finalCount}.  ` + parts.join(", ");
}

function refreshLabels() {
  minConfVal.textContent = Number(minConf.value).toFixed(2);
  minAreaVal.textContent = Number(minArea.value).toFixed(2) + "%";
  nmsVal.textContent = Number(nmsIoU.value).toFixed(2);
  maxTotalVal.textContent = String(maxTotal.value);
  maxPerClassVal.textContent = String(maxPerClass.value);
}

[minConf, minArea, nmsIoU, maxTotal, maxPerClass].forEach(el => {
  el.addEventListener("input", refreshLabels);
});
refreshLabels();

btn.addEventListener("click", () => {
  clearError();

  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    showError("Please choose an image first.");
    return;
  }

  // Reset UI
  setProgress(0);
  spinner.classList.remove("hidden");
  setState("running", "Uploading…");
  summary.textContent = "Working…";
  rawJson.textContent = "—";
  originalImg.removeAttribute("src");
  resultImg.removeAttribute("src");

  const form = new FormData();
  form.append("image", file);

  // send fast-fix parameters
  form.append("min_conf", minConf.value);
  form.append("min_area_frac", (Number(minArea.value) / 100).toString()); // percent -> fraction
  form.append("nms_iou", nmsIoU.value);
  form.append("max_total", maxTotal.value);
  form.append("max_per_class", maxPerClass.value);
  form.append("small_mode", smallMode.checked ? "1" : "0");

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/detect", true);

  // Upload progress
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = (e.loaded / e.total) * 100;
      setProgress(pct);
    }
  };

  xhr.onload = () => {
    spinner.classList.add("hidden");

    let data;
    try {
      data = JSON.parse(xhr.responseText);
    } catch (err) {
      showError("Server returned non-JSON response.");
      return;
    }

    if (!data.ok) {
      showError(data.error || "Unknown error");
      return;
    }

    setProgress(100);
    setState("done", "Done ✓");

    originalImg.src = data.original_url + "?t=" + Date.now();
    resultImg.src = data.result_url + "?t=" + Date.now();

    summary.textContent = detectionsSummary(data.predictions, data.counts);
    rawJson.textContent = JSON.stringify(data, null, 2);
  };

  xhr.onerror = () => {
    showError("Network error. Check internet connection and server logs.");
  };

  xhr.onreadystatechange = () => {
    if (xhr.readyState === 2 || xhr.readyState === 3) {
      if (progressPct.textContent === "100%") {
        setState("running", "Running model + filtering…");
        spinner.classList.remove("hidden");
      }
    }
  };

  xhr.send(form);
});