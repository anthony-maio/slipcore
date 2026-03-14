const forceTokens = new Set([
  "Observe",
  "Inform",
  "Ask",
  "Request",
  "Propose",
  "Commit",
  "Eval",
  "Meta",
  "Accept",
  "Reject",
  "Error",
  "Fallback",
]);

const tokenPattern = /^[A-Za-z0-9]+$/;
const fallbackRefPattern = /^ref[A-Za-z0-9]{1,20}$/;

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("slipstream-theme", theme);
}

function initializeTheme() {
  const stored = localStorage.getItem("slipstream-theme");
  const systemLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  setTheme(stored || (systemLight ? "light" : "dark"));
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function tokensFromPayload(rawPayload) {
  return rawPayload
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function renderError(target, message) {
  target.innerHTML = `<div class="output-error"><strong>Invalid input:</strong> ${escapeHtml(message)}</div>`;
}

function renderEncode() {
  const src = $("src-input").value.trim();
  const dst = $("dst-input").value.trim();
  const force = $("force-input").value.trim();
  const obj = $("obj-input").value.trim();
  const payload = tokensFromPayload($("payload-input").value);
  const target = $("encode-output");

  if (!tokenPattern.test(src) || src.length > 20) {
    renderError(target, "source must be 1-20 alphanumeric characters");
    return;
  }

  if (!tokenPattern.test(dst) || dst.length > 20) {
    renderError(target, "destination must be 1-20 alphanumeric characters");
    return;
  }

  if (!forceTokens.has(force)) {
    renderError(target, "force must be one of the 12 closed tokens");
    return;
  }

  if (!tokenPattern.test(obj) || obj.length > 30) {
    renderError(target, "object must be alphanumeric and at most 30 characters");
    return;
  }

  if (payload.length > 20) {
    renderError(target, "payload is limited to 20 tokens");
    return;
  }

  if (payload.some((token) => !tokenPattern.test(token) || token.length > 30)) {
    renderError(target, "payload tokens must be alphanumeric and at most 30 characters");
    return;
  }

  if (force === "Fallback") {
    if (payload.length !== 1 || !fallbackRefPattern.test(payload[0])) {
      renderError(target, "Fallback requires exactly one ref token such as ref7f3a1b2c");
      return;
    }
  }

  const wire = ["SLIP", "v3", src, dst, force, obj, ...payload].join(" ");
  target.innerHTML = `<strong>Wire:</strong><br><code>${escapeHtml(wire)}</code>`;
}

function renderDecode() {
  const wire = $("wire-input").value.trim();
  const target = $("decode-output");
  const parts = wire.split(/\s+/).filter(Boolean);

  if (parts.length < 6) {
    renderError(target, "wire must have at least 6 tokens");
    return;
  }

  const [prefix, version, src, dst, force, obj, ...payload] = parts;

  if (prefix !== "SLIP" || version !== "v3") {
    renderError(target, "wire must start with SLIP v3");
    return;
  }

  if (!tokenPattern.test(src) || src.length > 20) {
    renderError(target, "source token is invalid");
    return;
  }

  if (!tokenPattern.test(dst) || dst.length > 20) {
    renderError(target, "destination token is invalid");
    return;
  }

  if (!forceTokens.has(force)) {
    renderError(target, "force token is not part of the closed vocabulary");
    return;
  }

  if (!tokenPattern.test(obj) || obj.length > 30) {
    renderError(target, "object token is invalid");
    return;
  }

  if (payload.length > 20) {
    renderError(target, "payload exceeds the 20 token limit");
    return;
  }

  if (payload.some((token) => !tokenPattern.test(token) || token.length > 30)) {
    renderError(target, "payload contains invalid tokens");
    return;
  }

  if (force === "Fallback") {
    if (payload.length !== 1 || !fallbackRefPattern.test(payload[0])) {
      renderError(target, "Fallback wires require exactly one valid ref token");
      return;
    }
  }

  const payloadText = payload.length ? payload.join(" ") : "(none)";
  target.innerHTML = [
    `<strong>Source:</strong> ${escapeHtml(src)}`,
    `<strong>Destination:</strong> ${escapeHtml(dst)}`,
    `<strong>Intent:</strong> ${escapeHtml(force)}:${escapeHtml(obj)}`,
    `<strong>Payload:</strong> ${escapeHtml(payloadText)}`,
  ].join("<br>");
}

function renderCalculator() {
  const agents = Number($("agents-input").value);
  const messagesPerDay = Number($("messages-input").value);
  const pricePerMillion = Number($("price-input").value);
  const target = $("calc-output");

  if (!Number.isFinite(agents) || agents < 1 || !Number.isFinite(messagesPerDay) || messagesPerDay < 1 || !Number.isFinite(pricePerMillion) || pricePerMillion < 0) {
    target.innerHTML = '<div class="output-error"><strong>Invalid input:</strong> enter positive values for agents, messages, and token price.</div>';
    return;
  }

  const annualMessages = agents * messagesPerDay * 365;
  const jsonTokens = annualMessages * 41.9;
  const slipTokens = annualMessages * 7.4;
  const jsonCost = (jsonTokens / 1_000_000) * pricePerMillion;
  const slipCost = (slipTokens / 1_000_000) * pricePerMillion;
  const savings = jsonCost - slipCost;

  target.innerHTML = [
    `<div class="calc-card"><span>JSON annual cost</span><strong>${formatCurrency(jsonCost)}</strong></div>`,
    `<div class="calc-card"><span>Slipstream annual cost</span><strong>${formatCurrency(slipCost)}</strong></div>`,
    `<div class="calc-card"><span>Annual savings</span><strong>${formatCurrency(savings)}</strong></div>`,
  ].join("");
}

function initializeEvents() {
  const themeToggle = document.querySelector("[data-theme-toggle]");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      setTheme(current === "dark" ? "light" : "dark");
    });
  }

  $("encode-btn").addEventListener("click", renderEncode);
  $("decode-btn").addEventListener("click", renderDecode);
  $("calc-btn").addEventListener("click", renderCalculator);

  renderEncode();
  renderDecode();
  renderCalculator();
}

initializeTheme();
initializeEvents();
