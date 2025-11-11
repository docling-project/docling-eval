const state = {
  manifest: null,
  entries: [],
  visibleEntryIds: [],
  currentVisibleIndex: 0,
  decisions: new Map(),
  activeVisualizationIndex: new Map(),
  storageHandle: null,
  localStorageKey: "",
  editBuffer: null,
  hasUnsavedChanges: false,
  persistenceMode: "local",
  lastImportSource: null,
};

const elements = {
  headerTitle: document.getElementById("headerTitle"),
  queueList: document.getElementById("queueList"),
  searchInput: document.getElementById("searchInput"),
  statusFilter: document.getElementById("statusFilter"),
  entryTitle: document.getElementById("entryTitle"),
  vizTabs: document.getElementById("vizTabs"),
  vizFrame: document.getElementById("vizFrame"),
  commentInput: document.getElementById("commentInput"),
  decisionButtons: document.querySelectorAll(".status-btn"),
  clearDecision: document.getElementById("clearDecision"),
  prevEntry: document.getElementById("prevEntry"),
  nextEntry: document.getElementById("nextEntry"),
  reviewCounts: document.getElementById("reviewCounts"),
  globalCounts: document.getElementById("globalCounts"),
  connectBundle: document.getElementById("connectBundle"),
  importButton: document.getElementById("importDecisions"),
  exportButton: document.getElementById("exportCsv"),
  importInput: document.getElementById("importInput"),
  storageStatus: document.getElementById("storageStatus"),
  storageTarget: document.getElementById("storageTarget"),
  subsetValue: document.getElementById("subsetValue"),
  imageValue: document.getElementById("imageValue"),
  priorityLabel: document.getElementById("priorityLabel"),
  priorityValue: document.getElementById("priorityValue"),
  saveDecision: document.getElementById("saveDecision"),
  discardChanges: document.getElementById("discardChanges"),
  unsavedNotice: document.getElementById("unsavedNotice"),
  saveStatus: document.getElementById("saveStatusMessage"),
};

function createEmptyStatuses() {
  return USERS.reduce((acc, user) => {
    acc[user.key] = null;
    return acc;
  }, {});
}

function extractSubmissionName(submissionDir) {
  if (!submissionDir) {
    return "";
  }
  const normalized = submissionDir.replace(/[/\\]+$/, "");
  const parts = normalized.split(/[/\\]/).filter(Boolean);
  return parts[parts.length - 1] ?? "";
}

function setStorageTarget(mode, description) {
  state.persistenceMode = mode;
  elements.storageTarget.textContent = `Saving to: ${description}`;
}

const STATUS_LABELS = {
  correct: "Correct",
  need_changes: "Need changes",
};

const USERS = [
  { key: "user_a", label: "User A" },
  { key: "user_b", label: "User B" },
];

async function init() {
  try {
    const manifest = await fetchJson("manifest.json");
    setupManifest(manifest);
    await hydrateDecisions();
    attachEvents();
    renderQueue();
    selectVisibleIndex(0);
  } catch (error) {
    elements.globalCounts.textContent =
      "Unable to load manifest. Serve this folder with `python -m http.server` and reload.";
    console.error("Failed to initialize review bundle", error);
  }
}

function setupManifest(manifest) {
  state.manifest = manifest;
  state.entries = manifest.entries;
  state.visibleEntryIds = state.entries.map((_, index) => index);
  state.localStorageKey = buildStorageKey(manifest);
  elements.priorityLabel.textContent = manifest.review_column;
  const submissionName = extractSubmissionName(manifest.submission_dir);
  if (submissionName) {
    elements.headerTitle.textContent = `Docling Review Bundle — ${submissionName}`;
  }
  setStorageTarget("local", "Browser storage (connect a bundle to persist review_state.json)");
  elements.globalCounts.textContent = `${manifest.total_entries} frames queued (sorted by ${manifest.review_column})`;
}

async function hydrateDecisions() {
  loadDecisionsFromLocalStorage();
  try {
    const fileState = await fetchJson("review_state.json");
    if (fileState && Array.isArray(fileState.decisions)) {
      mergeDecisions(fileState.decisions);
    }
  } catch (error) {
    console.warn("No review_state.json found yet", error);
  }
  updateCounts();
}

function buildStorageKey(manifest) {
  const slugBase = `${manifest.submission_dir}-${manifest.generated_at}-${manifest.total_entries}`;
  return `docling-review-${slugBase.replace(/[^a-zA-Z0-9]+/g, "-")}`;
}

function loadDecisionsFromLocalStorage() {
  if (!state.localStorageKey) {
    return;
  }
  const payload = localStorage.getItem(state.localStorageKey);
  if (!payload) {
    return;
  }
  try {
    const parsed = JSON.parse(payload);
    mergeDecisions(parsed.decisions ?? []);
  } catch (error) {
    console.warn("Unable to parse local storage payload", error);
  }
}

function persistLocal(decisionsArray) {
  if (!state.localStorageKey) {
    return;
  }
  const payload = { decisions: decisionsArray };
  localStorage.setItem(state.localStorageKey, JSON.stringify(payload));
}

function mergeDecisions(decisionArray) {
  decisionArray.forEach((decision) => {
    if (!decision || !decision.entry_id) {
      return;
    }
    const normalized = {
      ...decision,
      status_user_a:
        decision.status_user_a ?? decision.status ?? null,
      status_user_b:
        decision.status_user_b ?? decision.status ?? null,
    };
    state.decisions.set(decision.entry_id, normalized);
  });
}

function initializeEditBuffer(entry, options = {}) {
  const { preserveStatusMessage = false } = options;
  const saved = state.decisions.get(entry.entry_id);
  state.editBuffer = {
    entry_id: entry.entry_id,
    statuses: {
      ...createEmptyStatuses(),
      user_a: saved?.status_user_a ?? null,
      user_b: saved?.status_user_b ?? null,
    },
    comment: saved?.comment ?? "",
  };
  elements.commentInput.value = state.editBuffer.comment ?? "";
  updateDecisionButtons(state.editBuffer.statuses);
  setUnsavedChanges(false);
  if (!preserveStatusMessage) {
    setSaveStatus();
  }
}

function updateDecisionButtons(statuses) {
  elements.decisionButtons.forEach((button) => {
    const userKey = button.dataset.user;
    const userStatus = statuses?.[userKey] ?? null;
    button.classList.toggle("active", userStatus === button.dataset.status);
  });
}

function setUnsavedChanges(hasChanges) {
  state.hasUnsavedChanges = hasChanges;
  elements.saveDecision.disabled = !hasChanges;
  elements.discardChanges.disabled = !hasChanges;
  elements.unsavedNotice.hidden = !hasChanges;
  elements.searchInput.disabled = hasChanges;
  elements.statusFilter.disabled = hasChanges;
  if (!hasChanges) {
    elements.unsavedNotice.classList.remove("pulse");
  } else {
    setSaveStatus();
  }
  updateNavigationButtons();
}

function showUnsavedWarning() {
  if (!state.hasUnsavedChanges) {
    return;
  }
  elements.unsavedNotice.hidden = false;
  elements.unsavedNotice.classList.add("pulse");
  setTimeout(() => {
    elements.unsavedNotice.classList.remove("pulse");
  }, 500);
}

function setSaveStatus(message, variant = "info") {
  const el = elements.saveStatus;
  if (!message) {
    el.hidden = true;
    el.textContent = "";
    el.className = "save-status";
    return;
  }
  el.hidden = false;
  el.textContent = message;
  el.className = `save-status ${variant}`;
}

function extractDecision(decision) {
  if (!decision) {
    return {
      statuses: createEmptyStatuses(),
      comment: "",
    };
  }
  return {
    statuses: {
      ...createEmptyStatuses(),
      user_a: decision.status_user_a ?? null,
      user_b: decision.status_user_b ?? null,
    },
    comment: decision.comment ?? "",
  };
}

function summarizeDecision(decision) {
  const { statuses, comment } = extractDecision(decision);
  const statusValues = Object.values(statuses);
  const hasNeedChanges = statusValues.includes("need_changes");
  const allCorrect = statusValues.every((value) => value === "correct" && value !== null);
  const anyStatus = statusValues.some((value) => Boolean(value));
  const hasComment = Boolean(comment?.trim().length);
  let label = "Pending";
  let className = "pending";
  if (hasNeedChanges) {
    label = "Needs changes";
    className = "need_changes";
  } else if (allCorrect && anyStatus) {
    label = "All correct";
    className = "correct";
  } else if (anyStatus) {
    label = "Partial review";
  } else if (hasComment) {
    label = "Comment only";
  }
  return { statuses, comment, label, className };
}

function attachEvents() {
  elements.searchInput.addEventListener("input", () => {
    applyFilters();
  });
  elements.statusFilter.addEventListener("change", () => {
    applyFilters();
  });
  elements.prevEntry.addEventListener("click", () => navigate(-1));
  elements.nextEntry.addEventListener("click", () => navigate(1));
  elements.decisionButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const status = button.dataset.status;
      const userKey = button.dataset.user;
      toggleDecisionStatus(userKey, status);
    });
  });
  elements.clearDecision.addEventListener("click", clearDecision);
  elements.commentInput.addEventListener("input", handleCommentInput);
  elements.saveDecision.addEventListener("click", saveCurrentDecision);
  elements.discardChanges.addEventListener("click", discardChanges);
  elements.connectBundle.addEventListener("click", enableFilePersistence);
  elements.exportButton.addEventListener("click", exportCsvReport);
  elements.importButton.addEventListener("click", () => elements.importInput.click());
  elements.importInput.addEventListener("change", handleImportSelection);
}

function applyFilters() {
  if (state.hasUnsavedChanges) {
    showUnsavedWarning();
    return;
  }
  const term = elements.searchInput.value.trim().toLowerCase();
  const statusFilter = elements.statusFilter.value;
  state.visibleEntryIds = state.entries
    .map((entry, index) => ({ entry, index }))
    .filter(({ entry }) => filterEntry(entry, term, statusFilter))
    .map(({ index }) => index);
  const nextIndex = Math.min(state.currentVisibleIndex, state.visibleEntryIds.length - 1);
  state.currentVisibleIndex = Math.max(0, nextIndex);
  renderQueue();
  if (state.visibleEntryIds.length > 0) {
    selectVisibleIndex(state.currentVisibleIndex);
  }
}

function filterEntry(entry, term, statusFilter) {
  const matchesTerm = !term
    || entry.doc_name.toLowerCase().includes(term)
    || entry.image_name.toLowerCase().includes(term);
  if (!matchesTerm) {
    return false;
  }
  if (statusFilter === "all") {
    return true;
  }
  const decision = state.decisions.get(entry.entry_id);
  const summary = summarizeDecision(decision);
  const statusValues = Object.values(summary.statuses);
  if (statusFilter === "pending") {
    return statusValues.some((value) => !value);
  }
  if (statusFilter === "correct") {
    return statusValues.every((value) => value === "correct" && value !== null);
  }
  if (statusFilter === "need_changes") {
    return statusValues.includes("need_changes");
  }
  return true;
}

function renderQueue() {
  elements.queueList.innerHTML = "";
  if (!state.visibleEntryIds.length) {
    elements.queueList.innerHTML = '<p class="empty">No rows match the current filters.</p>';
    return;
  }
  state.visibleEntryIds.forEach((entryIndex, visibleIndex) => {
    const entry = state.entries[entryIndex];
    const wrapper = document.createElement("div");
    wrapper.className = "queue-item";
    if (visibleIndex === state.currentVisibleIndex) {
      wrapper.classList.add("active");
    }
    const decision = state.decisions.get(entry.entry_id);
    const summary = summarizeDecision(decision);
    const reviewMetric = formatPriority(entry.review_value);
    const userSummary = USERS.map((user) => {
      const userStatus = summary.statuses[user.key];
      const label = userStatus ? STATUS_LABELS[userStatus] : "Pending";
      const className = userStatus ? userStatus : "";
      return `<span class="user-chip ${className}">${user.label}: ${label}</span>`;
    }).join("");
    wrapper.innerHTML = `
      <p class="title">${entry.doc_name}</p>
      <p class="subtitle">${entry.image_name}</p>
      <div class="queue-meta">
        <span class="review-chip">${state.manifest.review_column}: ${reviewMetric}</span>
        <span class="badge ${summary.className}">${summary.label}</span>
      </div>
      <div class="user-summary">${userSummary}</div>
    `;
    wrapper.addEventListener("click", () => {
      const targetEntryIndex = state.visibleEntryIds[visibleIndex];
      const currentEntryIndex = state.visibleEntryIds[state.currentVisibleIndex];
      if (state.hasUnsavedChanges && targetEntryIndex !== currentEntryIndex) {
        showUnsavedWarning();
        return;
      }
      selectVisibleIndex(visibleIndex);
    });
    elements.queueList.appendChild(wrapper);
  });
}

function selectVisibleIndex(visibleIndex) {
  if (!state.visibleEntryIds.length) {
    return;
  }
  state.currentVisibleIndex = Math.max(0, Math.min(visibleIndex, state.visibleEntryIds.length - 1));
  const entryIndex = state.visibleEntryIds[state.currentVisibleIndex];
  const entry = state.entries[entryIndex];
  renderCurrentEntry(entry);
  renderQueue();
  updateNavigationButtons();
}

function updateNavigationButtons() {
  const locked = state.hasUnsavedChanges;
  elements.prevEntry.disabled = locked || state.currentVisibleIndex <= 0;
  elements.nextEntry.disabled = locked || state.currentVisibleIndex >= state.visibleEntryIds.length - 1;
}

function renderCurrentEntry(entry) {
  if (!entry) {
    elements.entryTitle.textContent = "No entries available";
    state.editBuffer = null;
    elements.commentInput.value = "";
    updateDecisionButtons(null);
    setUnsavedChanges(false);
    setSaveStatus();
    return;
  }
  elements.entryTitle.textContent = `${entry.doc_name}`;
  renderEntryInfo(entry);
  renderVisualization(entry);
  initializeEditBuffer(entry);
}

function renderEntryInfo(entry) {
  const subset = entry.metadata?.subset ?? "—";
  elements.subsetValue.textContent = subset || "—";
  elements.imageValue.textContent = entry.image_name ?? "—";
  elements.priorityValue.textContent = formatPriority(entry.review_value);
}

function formatPriority(value) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
  return value;
}

function renderVisualization(entry) {
  const visuals = entry.visualizations ?? [];
  elements.vizTabs.innerHTML = "";
  if (!visuals.length) {
    elements.vizFrame.src = "about:blank";
    elements.vizFrame.title = "No visualization available";
    elements.vizFrame.classList.add("empty-frame");
    elements.vizTabs.innerHTML = '<p class="empty">No visualization HTML files located for this doc.</p>';
    return;
  }
  const storedIndex = state.activeVisualizationIndex.get(entry.entry_id) ?? 0;
  const activeIndex = Math.min(storedIndex, visuals.length - 1);
  state.activeVisualizationIndex.set(entry.entry_id, activeIndex);
  visuals.forEach((visual, index) => {
    const button = document.createElement("button");
    button.className = "viz-tab";
    if (index === activeIndex) {
      button.classList.add("active");
    }
    button.textContent = visual.label;
    button.addEventListener("click", () => {
      state.activeVisualizationIndex.set(entry.entry_id, index);
      renderVisualization(entry);
    });
    elements.vizTabs.appendChild(button);
  });
  elements.vizFrame.classList.remove("empty-frame");
  elements.vizFrame.src = visuals[activeIndex].relative_path;
  elements.vizFrame.title = visuals[activeIndex].label;
}

function toggleDecisionStatus(userKey, status) {
  if (!state.editBuffer) {
    return;
  }
  const currentStatus = state.editBuffer.statuses[userKey];
  const nextStatus = currentStatus === status ? null : status;
  if (currentStatus === nextStatus) {
    return;
  }
  state.editBuffer.statuses[userKey] = nextStatus;
  updateDecisionButtons(state.editBuffer.statuses);
  setUnsavedChanges(true);
}

function resolveActiveVisualization(entry) {
  const visuals = entry.visualizations ?? [];
  if (!visuals.length) {
    return null;
  }
  const index = state.activeVisualizationIndex.get(entry.entry_id) ?? 0;
  return visuals[Math.min(index, visuals.length - 1)].relative_path;
}

function handleCommentInput() {
  if (!state.editBuffer) {
    return;
  }
  const value = elements.commentInput.value;
  if (state.editBuffer.comment === value) {
    return;
  }
  state.editBuffer.comment = value;
  setUnsavedChanges(true);
}

function clearDecision() {
  if (!state.editBuffer) {
    return;
  }
  state.editBuffer.statuses = createEmptyStatuses();
  state.editBuffer.comment = "";
  elements.commentInput.value = "";
  updateDecisionButtons(state.editBuffer.statuses);
  setUnsavedChanges(true);
}

function saveCurrentDecision() {
  if (!state.editBuffer || !state.hasUnsavedChanges) {
    return;
  }
  const entry = getCurrentEntry();
  if (!entry) {
    return;
  }
  const statuses = state.editBuffer.statuses;
  const commentValue = state.editBuffer.comment ?? "";
  const hasComment = commentValue.trim().length > 0;
  const hasAnyStatus = USERS.some((user) => statuses[user.key]);
  let message = "";
  let variant = "success";
  if (!hasAnyStatus && !hasComment) {
    state.decisions.delete(entry.entry_id);
    message = "Decision cleared.";
    variant = "muted";
  } else {
    const decision = {
      entry_id: entry.entry_id,
      doc_name: entry.doc_name,
      image_name: entry.image_name,
      review_value: entry.review_value,
      visualization_path: resolveActiveVisualization(entry),
      status_user_a: statuses.user_a ?? null,
      status_user_b: statuses.user_b ?? null,
      comment: commentValue,
      updated_at: new Date().toISOString(),
    };
    state.decisions.set(entry.entry_id, decision);
    if (hasAnyStatus) {
      const parts = USERS.filter((user) => statuses[user.key]).map(
        (user) => `${user.label}: ${STATUS_LABELS[statuses[user.key]]}`,
      );
      message = `Decision saved (${parts.join(" • ")}).`;
      variant = "success";
    } else {
      message = "Comment saved (statuses remain pending).";
      variant = "info";
    }
  }
  persistDecisions();
  updateCounts();
  renderQueue();
  initializeEditBuffer(entry, { preserveStatusMessage: true });
  setSaveStatus(message, variant);
}

function discardChanges() {
  const entry = getCurrentEntry();
  if (!entry) {
    return;
  }
  initializeEditBuffer(entry);
  setSaveStatus("Changes discarded.", "muted");
}

function getCurrentEntry() {
  if (!state.visibleEntryIds.length) {
    return null;
  }
  const entryIndex = state.visibleEntryIds[state.currentVisibleIndex];
  return state.entries[entryIndex];
}

function navigate(delta) {
  if (state.hasUnsavedChanges) {
    showUnsavedWarning();
    return;
  }
  const next = state.currentVisibleIndex + delta;
  if (next < 0 || next >= state.visibleEntryIds.length) {
    return;
  }
  selectVisibleIndex(next);
}

function updateCounts() {
  const totals = USERS.reduce((acc, user) => {
    acc[user.key] = { correct: 0, need_changes: 0 };
    return acc;
  }, {});
  let reviewed = 0;
  state.decisions.forEach((decision) => {
    const { statuses, comment } = extractDecision(decision);
    const hasAnyStatus = Object.values(statuses).some((value) => Boolean(value));
    const hasComment = Boolean(comment?.trim().length);
    if (hasAnyStatus || hasComment) {
      reviewed += 1;
    }
    USERS.forEach((user) => {
      const userStatus = statuses[user.key];
      if (userStatus && userStatus in totals[user.key]) {
        totals[user.key][userStatus] += 1;
      }
    });
  });
  const total = state.entries.length;
  const perUser = USERS.map((user) => {
    const userTotals = totals[user.key];
    return `${user.label} – Correct ${userTotals.correct}, Need changes ${userTotals.need_changes}`;
  }).join(" • ");
  const countsMessage = `Reviewed ${reviewed}/${total} • ${perUser}`;
  elements.reviewCounts.textContent = countsMessage;
  elements.globalCounts.textContent = `${total} frames queued • ${reviewed} reviewed`;
}

function persistDecisions() {
  const decisionsArray = Array.from(state.decisions.values());
  persistLocal(decisionsArray);
  saveToBundle(decisionsArray).catch((error) => {
    if (error) {
      console.warn("Unable to write review_state.json", error);
      elements.storageStatus.textContent =
        "Connect the bundle directory to enable automatic saving.";
    }
  });
}

async function enableFilePersistence() {
  if (!window.showDirectoryPicker) {
    elements.storageStatus.textContent =
      "Browser does not support the File System Access API. Use Chrome or export CSV manually.";
    return;
  }
  try {
    const dirHandle = await window.showDirectoryPicker({ mode: "readwrite" });
    const manifestHandle = await dirHandle.getFileHandle("manifest.json");
    const manifestFile = await manifestHandle.getFile();
    const manifestData = JSON.parse(await manifestFile.text());
    if (manifestData.generated_at !== state.manifest.generated_at) {
      throw new Error("Selected directory does not match this bundle.");
    }
    const stateHandle = await dirHandle.getFileHandle("review_state.json", { create: true });
    state.storageHandle = stateHandle;
    elements.storageStatus.textContent = "Connected. Decisions save directly to review_state.json.";
    elements.connectBundle.textContent = "Change bundle connection";
    setStorageTarget(
      "bundle",
      "review_state.json in the selected bundle directory",
    );
    persistDecisions();
  } catch (error) {
    console.error("Unable to enable persistence", error);
    elements.storageStatus.textContent =
      "Permission denied or wrong folder. Ensure you select the bundle directory.";
  }
}

async function saveToBundle(decisionsArray) {
  if (!state.storageHandle) {
    return;
  }
  const writable = await state.storageHandle.createWritable();
  await writable.write(JSON.stringify({ decisions: decisionsArray }, null, 2));
  await writable.close();
  elements.storageStatus.textContent = `Saved ${decisionsArray.length} items to review_state.json`;
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Unable to load ${path}`);
  }
  return response.json();
}

function exportCsvReport() {
  const rows = [
    [
      "entry_id",
      "doc_name",
      "image_name",
      "status_user_a",
      "status_user_b",
      "comment",
      "updated_at",
      "review_value",
      "visualization_path",
    ],
  ];
  state.decisions.forEach((decision) => {
    rows.push([
      decision.entry_id,
      decision.doc_name,
      decision.image_name,
      decision.status_user_a ?? "",
      decision.status_user_b ?? "",
      (decision.comment ?? "").replace(/\n/g, " "),
      decision.updated_at ?? "",
      decision.review_value ?? "",
      decision.visualization_path ?? "",
    ]);
  });
  const csvContent = rows.map((cols) => cols.map(escapeCsv).join(",")).join("\n");
  const blob = new Blob([csvContent], { type: "text/csv" });
  const link = document.createElement("a");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  link.href = URL.createObjectURL(blob);
  link.download = `review_log_${timestamp}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function escapeCsv(value) {
  const stringValue = value ?? "";
  if (/,|"|\n/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }
  return stringValue;
}

function handleImportSelection(event) {
  if (state.hasUnsavedChanges) {
    showUnsavedWarning();
    event.target.value = "";
    return;
  }
  const [file] = event.target.files ?? [];
  if (!file) {
    return;
  }
  file
    .text()
    .then((text) => {
      if (file.name.endsWith(".csv")) {
        importFromCsv(text, file.name);
      } else {
        importFromJson(text, file.name);
      }
    })
    .finally(() => {
      event.target.value = "";
    });
}

function importFromJson(jsonText, sourceLabel = "imported file") {
  try {
    const payload = JSON.parse(jsonText);
    if (Array.isArray(payload.decisions)) {
      mergeDecisions(payload.decisions);
      persistDecisions();
      updateCounts();
      renderQueue();
      const entry = getCurrentEntry();
      if (entry) {
        initializeEditBuffer(entry);
      }
      setSaveStatus("Decisions imported.", "info");
      elements.storageStatus.textContent = `Imported log from ${sourceLabel}. Remember to save or export.`;
    }
  } catch (error) {
    console.error("Unable to parse JSON log", error);
  }
}

function importFromCsv(csvText, sourceLabel = "CSV file") {
  const [headerLine, ...rows] = csvText.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  const hasDualStatus = headers.includes("status_user_a") && headers.includes("status_user_b");
  const hasLegacyStatus = headers.includes("status");
  if (!hasDualStatus && !hasLegacyStatus) {
    console.warn("CSV file missing status columns", headers);
    return;
  }
  const required = hasDualStatus
    ? ["entry_id", "status_user_a", "status_user_b", "comment"]
    : ["entry_id", "status", "comment"];
  if (!required.every((key) => headers.includes(key))) {
    console.warn("CSV file missing required headers", headers);
    return;
  }
  const entries = rows.map((line) => parseCsvLine(line, headers));
  const normalizedEntries = entries.map((entry) => {
    if (hasDualStatus) {
      return entry;
    }
    return {
      ...entry,
      status_user_a: entry.status ?? null,
      status_user_b: entry.status ?? null,
    };
  });
  mergeDecisions(normalizedEntries);
  persistDecisions();
  updateCounts();
  renderQueue();
  const entry = getCurrentEntry();
  if (entry) {
    initializeEditBuffer(entry);
  }
  setSaveStatus("Decisions imported.", "info");
  elements.storageStatus.textContent = `Imported log from ${sourceLabel}. Remember to save or export.`;
}

function parseCsvLine(line, headers) {
  const values = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  const record = {};
  headers.forEach((header, index) => {
    record[header] = values[index];
  });
  return record;
}

init();
