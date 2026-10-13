const STORAGE_KEY = "what-to-eat-session-v3";
const MEAL_ICONS = ["salad", "soup", "cooking-pot"];
const HERO_PHRASES = [
  "把冰箱里的东西，变成今晚的菜单。",
  "晚餐怎么安排，才能不重样？",
  "想吃什么，不用再翻半小时菜谱。",
];

const defaults = {
  maxMinutes: "30",
  allergies: "",
};

function createId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const state = {
  conversationId: createId(),
  history: [],
  lastAnswer: "",
  lastThreadId: null,
  lastMealPlan: null,
  lastShoppingList: [],
  lastEvidence: [],
  lastToolCalls: [],
  manualTab: false,
  dirtyConstraints: new Set(),
  form: { ...defaults },
  running: false,
};

const elements = {
  chatsForm: document.querySelector("#chat-form"),
  goal: document.querySelector("#goal"),
  maxMinutes: document.querySelector("#max-minutes"),
  allergies: document.querySelector("#allergies"),
  runButton: document.querySelector("#run-button"),
  startPlanningButton: document.querySelector("#start-planning-button"),
  newConversationButton: document.querySelector("#new-conversation-button"),
  themeToggleButton: document.querySelector("#theme-toggle-button"),
  themeLabel: document.querySelector("#theme-label"),
  copyButton: document.querySelector("#copy-button"),
  currentGoal: document.querySelector("#current-goal"),
  runStatus: document.querySelector("#run-status"),
  toolCount: document.querySelector("#tool-count"),
  evidenceCount: document.querySelector("#evidence-count"),
  threadId: document.querySelector("#thread-id"),
  conversationId: document.querySelector("#conversation-id"),
  conditionsSummary: document.querySelector("#conditions-summary"),
  conversation: document.querySelector("#conversation"),
  mealPlan: document.querySelector("#meal-plan"),
  shoppingList: document.querySelector("#shopping-list"),
  evidenceList: document.querySelector("#evidence-list"),
  toolTrace: document.querySelector("#tool-trace"),
  serviceState: document.querySelector("#service-state"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function splitItems(value) {
  return value
    .split(/[，,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function readForm() {
  return {
    maxMinutes: elements.maxMinutes.value || defaults.maxMinutes,
    allergies: elements.allergies.value,
  };
}

function applyForm(form) {
  const next = { ...defaults, ...(form || {}) };
  elements.maxMinutes.value = next.maxMinutes;
  elements.allergies.value = next.allergies;
}

function buildConstraints() {
  const form = readForm();
  return {
    max_minutes: Number(form.maxMinutes),
    allergies: splitItems(form.allergies),
    explicit_constraints: [...state.dirtyConstraints],
  };
}

function persistSession() {
  state.form = readForm();
  updateConditionsSummary();
  const snapshot = {
    conversationId: state.conversationId,
    history: state.history,
    lastAnswer: state.lastAnswer,
    lastThreadId: state.lastThreadId,
    lastMealPlan: state.lastMealPlan,
    lastShoppingList: state.lastShoppingList,
    lastEvidence: state.lastEvidence,
    lastToolCalls: state.lastToolCalls,
    form: state.form,
  };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
}

function updateConditionsSummary() {
  if (!elements.conditionsSummary) return;
  const form = readForm();
  const allergies = splitItems(form.allergies);
  const parts = [`${form.maxMinutes} 分钟`];
  if (allergies.length) parts.push(`过敏：${allergies.join("、")}`);
  elements.conditionsSummary.textContent = parts.join(" · ");
}

function restoreSession() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "null");
    if (!saved || typeof saved !== "object") return;

    state.conversationId = saved.conversationId || createId();
    state.history = Array.isArray(saved.history) ? saved.history : [];
    state.lastAnswer = typeof saved.lastAnswer === "string" ? saved.lastAnswer : "";
    state.lastThreadId = saved.lastThreadId || null;
    state.lastMealPlan = saved.lastMealPlan || null;
    state.lastShoppingList = Array.isArray(saved.lastShoppingList)
      ? saved.lastShoppingList
      : [];
    state.lastEvidence = Array.isArray(saved.lastEvidence) ? saved.lastEvidence : [];
    state.lastToolCalls = Array.isArray(saved.lastToolCalls) ? saved.lastToolCalls : [];
    state.form = { ...defaults, ...(saved.form || {}) };
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
  }
}

function setStatus(text, type = "") {
  elements.runStatus.textContent = text;
  elements.runStatus.className = `run-status ${type}`.trim();
}

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function activateTab(name, { manual = false } = {}) {
  if (manual) state.manualTab = true;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".board-panel").forEach((panel) => {
    panel.classList.toggle("hidden", panel.id !== `panel-${name}`);
  });
}

// 本轮产出落在哪个面板，就自动打开哪个；用户手动选过则不打扰。
function selectTabForResult(payload) {
  if (state.manualTab) return;
  const days = (payload.meal_plan && payload.meal_plan.days) || [];
  const hasShopping = Array.isArray(payload.shopping_list) && payload.shopping_list.length > 0;
  const hasEvidence = Array.isArray(payload.evidence) && payload.evidence.length > 0;
  const hasTools = Array.isArray(payload.tool_calls) && payload.tool_calls.length > 0;
  if (days.length > 0 || hasShopping) {
    activateTab("plan");
  } else if (hasEvidence) {
    activateTab("evidence");
  } else if (hasTools) {
    activateTab("trace");
  }
}

function setProgress(stage, status) {
  const element = document.querySelector(`[data-stage="${stage}"]`);
  if (!element) return;
  element.classList.remove("active", "complete");
  if (status) element.classList.add(status);
}

function resetProgress() {
  document.querySelectorAll(".progress-step").forEach((step) => {
    step.classList.remove("active", "complete");
  });
}

function completeProgress() {
  ["understand", "retrieve", "validate", "respond"].forEach((stage) => {
    setProgress(stage, "complete");
  });
}

function renderConversation() {
  if (!state.history.length) {
    elements.conversation.innerHTML = `
      <div class="empty-state">
        <span class="empty-kicker">等待第一条消息</span>
        <strong>菜单会在这里逐步长出来</strong>
        <p>先用上方输入框描述需求。生成后，可以直接说“把第二天换成素菜”。</p>
        <div class="empty-tags">
          <span>真实菜谱</span>
          <span>硬约束校验</span>
          <span>多轮修改</span>
        </div>
      </div>
    `;
    return;
  }

  elements.conversation.innerHTML = state.history
    .map((message) => {
      const isUser = message.role === "user";
      const isError = message.role === "error";
      const avatarIcon = isUser ? "user-round" : isError ? "triangle-alert" : "chef-hat";
      // 用户这侧不显示“你”这种裸代词，头像已经说明是谁在说；只保留一条轻量说明。
      const speaker = isUser ? "" : isError ? "系统" : "今晚吃啥";
      const subtitle = isUser ? "本轮需求" : isError ? "运行提示" : "菜谱规划 Agent";
      const meta = [
        speaker ? `<strong>${speaker}</strong>` : "",
        subtitle ? `<span>${subtitle}</span>` : "",
      ].join("");
      return `
        <article class="conversation-message ${escapeHtml(message.role)}">
          <span class="message-avatar">
            <i data-lucide="${avatarIcon}"></i>
            ${isUser ? "" : '<span class="avatar-live-dot"></span>'}
          </span>
          <div class="message-content">
            <div class="message-meta">${meta}</div>
            <div>${escapeHtml(message.content)}</div>
          </div>
        </article>
      `;
    })
    .join("");
  refreshIcons();
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function appendConversation(role, content) {
  state.history.push({ role, content });
  renderConversation();
  persistSession();
}

function renderMealPlan(plan) {
  if (!plan?.days?.length) {
    elements.mealPlan.className = "empty-state compact";
    elements.mealPlan.textContent = "尚未生成菜单。";
    return;
  }

  elements.mealPlan.className = "";
  elements.mealPlan.innerHTML = plan.days
    .map(
      (day, index) => `
        <article class="meal-day">
          <span class="meal-thumb meal-thumb-${(index % MEAL_ICONS.length) + 1}">
            <i data-lucide="${MEAL_ICONS[index % MEAL_ICONS.length]}"></i>
          </span>
          <div>
            <span class="meal-day-index">第 ${day.day} 天</span>
            <strong>${day.meals.map((meal) => escapeHtml(meal.name)).join("、")}</strong>
          </div>
        </article>
      `
    )
    .join("");
  refreshIcons();
}

function renderShoppingList(items = []) {
  if (!items.length) {
    elements.shoppingList.className = "empty-state compact";
    elements.shoppingList.textContent = "尚未生成清单。";
    return;
  }

  elements.shoppingList.className = "";
  elements.shoppingList.innerHTML = `
    <ul class="check-list">
      ${items
        .map(
          (item) => `
            <li>
              <input type="checkbox" aria-label="${escapeHtml(item.name)}">
              <span>${escapeHtml(item.name)}</span>
            </li>
          `
        )
        .join("")}
    </ul>
  `;
}

function renderEvidence(items = []) {
  if (!items.length) {
    elements.evidenceList.className = "empty-state compact";
    elements.evidenceList.textContent = "尚未检索到菜谱。";
    return;
  }

  elements.evidenceList.className = "";
  elements.evidenceList.innerHTML = items
    .map(
      (item) => `
        <article class="evidence-item">
          <strong>${escapeHtml(item.name)}</strong>
          <p>${escapeHtml((item.ingredients || []).join("、") || "未提供食材信息")}</p>
          <div class="tag-row">
            <span class="tag">${escapeHtml(item.category || "未分类")}</span>
            <span class="tag">${escapeHtml(item.source || "无来源")}</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderTrace(calls = []) {
  if (!calls.length) {
    elements.toolTrace.className = "empty-state compact";
    elements.toolTrace.textContent = "尚未调用工具。";
    return;
  }

  elements.toolTrace.className = "";
  elements.toolTrace.innerHTML = calls
    .map(
      (call, index) => `
        <article class="trace-item">
          <span class="trace-step">${index + 1}</span>
          <strong>${escapeHtml(call.tool_name)}</strong>
          <p>${call.ok ? "执行成功" : `执行失败：${escapeHtml(call.error || "未知错误")}`}</p>
          <p>${escapeHtml(JSON.stringify(call.arguments || {}))}</p>
        </article>
      `
    )
    .join("");
}

function renderResult(payload) {
  state.lastAnswer = payload.final_answer || "";
  state.lastThreadId = payload.thread_id || null;
  state.lastMealPlan = payload.meal_plan || state.lastMealPlan;
  state.lastShoppingList = Array.isArray(payload.shopping_list)
    ? payload.shopping_list
    : state.lastShoppingList;
  state.lastEvidence = Array.isArray(payload.evidence) ? payload.evidence : [];
  state.lastToolCalls = Array.isArray(payload.tool_calls) ? payload.tool_calls : [];

  elements.toolCount.textContent = String(state.lastToolCalls.length);
  elements.evidenceCount.textContent = String(state.lastEvidence.length);
  elements.threadId.textContent = state.lastThreadId ? state.lastThreadId.slice(0, 8) : "未创建";
  elements.threadId.title = state.lastThreadId || "";
  renderMealPlan(state.lastMealPlan);
  renderShoppingList(state.lastShoppingList);
  renderEvidence(state.lastEvidence);
  renderTrace(state.lastToolCalls);
  selectTabForResult(payload);
  completeProgress();
  setStatus(
    payload.status === "completed" ? "已完成" : payload.status || "未知状态",
    payload.status === "completed" ? "status-success" : "status-warning"
  );
  persistSession();
}

function renderStoredResult() {
  elements.toolCount.textContent = String(state.lastToolCalls.length);
  elements.evidenceCount.textContent = String(state.lastEvidence.length);
  elements.threadId.textContent = state.lastThreadId ? state.lastThreadId.slice(0, 8) : "未创建";
  elements.threadId.title = state.lastThreadId || "";
  elements.conversationId.textContent = state.conversationId.slice(0, 8);
  elements.conversationId.title = state.conversationId;
  renderMealPlan(state.lastMealPlan);
  renderShoppingList(state.lastShoppingList);
  renderEvidence(state.lastEvidence);
  renderTrace(state.lastToolCalls);
}

function resetConversation() {
  state.conversationId = createId();
  state.history = [];
  state.lastAnswer = "";
  state.lastThreadId = null;
  state.lastMealPlan = null;
  state.lastShoppingList = [];
  state.lastEvidence = [];
  state.lastToolCalls = [];
  state.dirtyConstraints.clear();
  state.form = { ...defaults };

  applyForm(state.form);
  elements.goal.value = "";
  elements.goal.placeholder = "例如：规划晚餐，控制在 30 分钟内";
  elements.currentGoal.textContent = "新的菜谱规划";
  elements.conversationId.textContent = state.conversationId.slice(0, 8);
  elements.conversationId.title = state.conversationId;
  elements.threadId.textContent = "未创建";
  elements.threadId.title = "";
  elements.toolCount.textContent = "0";
  elements.evidenceCount.textContent = "0";
  setStatus("等待运行");
  resetProgress();
  state.manualTab = false;
  activateTab("plan");
  renderConversation();
  renderMealPlan(null);
  renderShoppingList([]);
  renderEvidence([]);
  renderTrace([]);
  persistSession();
  elements.goal.focus();
}

async function loadHealth() {
  try {
    const response = await fetch("/healthz");
    const health = await response.json();
    const modes = [];
    if (health.neo4j === "true") modes.push("Neo4j");
    if (health.hybrid === "true") modes.push("混合检索");
    modes.push(health.llm_planner === "true" ? "大模型规划" : "规则规划");
    elements.serviceState.classList.add("online");
    elements.serviceState.querySelector("span:last-child").textContent =
      `服务正常 · ${modes.join(" · ")}`;
  } catch {
    elements.serviceState.classList.remove("online");
    elements.serviceState.querySelector("span:last-child").textContent = "服务不可用";
  }
}

function startHeroRotation() {
  let phraseIndex = 0;
  window.setInterval(() => {
    if (state.history.length || elements.goal.value.trim()) return;
    phraseIndex = (phraseIndex + 1) % HERO_PHRASES.length;
    elements.currentGoal.textContent = HERO_PHRASES[phraseIndex];
  }, 5200);
}

function applyTheme(theme) {
  const normalized = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = normalized;
  window.localStorage.setItem("what-to-eat-theme", normalized);
  document.querySelector('meta[name="theme-color"]').setAttribute(
    "content",
    normalized === "dark" ? "#080b0f" : "#eef3ef"
  );
  elements.themeToggleButton.innerHTML = `
    <i data-lucide="${normalized === "dark" ? "sun" : "moon"}"></i>
    <span id="theme-label">${normalized === "dark" ? "日间" : "夜间"}</span>
  `;
  elements.themeLabel = document.querySelector("#theme-label");
  refreshIcons();
}

async function runAgent() {
  const message = elements.goal.value.trim();
  if (!message || state.running) return;

  const historyForRequest = [...state.history];
  state.manualTab = false;
  appendConversation("user", message);
  elements.goal.value = "";
  elements.goal.placeholder = "正在规划，请稍候…";
  state.running = true;
  elements.runButton.disabled = true;
  elements.currentGoal.textContent = message;
  resetProgress();
  setProgress("understand", "active");
  setStatus("运行中", "status-warning");

  try {
    const response = await fetch("/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        goal: message,
        constraints: buildConstraints(),
        user_id: "web-user",
        thread_id: null,
        conversation_id: state.conversationId,
        history: historyForRequest,
        meal_plan: state.lastMealPlan,
        shopping_list: state.lastShoppingList,
      }),
    });
    const rawBody = await response.text();
    let payload = {};
    try {
      payload = rawBody ? JSON.parse(rawBody) : {};
    } catch {
      payload = { detail: rawBody || `HTTP ${response.status}` };
    }
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }

  renderResult(payload);
  state.dirtyConstraints.clear();
  appendConversation("assistant", payload.final_answer || "Agent 未返回文本。");
    elements.goal.value = "";
    elements.goal.placeholder = "继续调整，例如：把第二天换成素菜";
  } catch (error) {
    setStatus("运行失败", "status-error");
    elements.goal.placeholder = "运行失败，请调整需求后重试";
    appendConversation("error", `运行失败：${error.message}`);
  } finally {
    state.running = false;
    elements.runButton.disabled = false;
    elements.goal.focus();
  }
}

elements.chatsForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runAgent();
});

elements.goal.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatsForm.requestSubmit();
  }
});

elements.newConversationButton.addEventListener("click", resetConversation);

elements.startPlanningButton.addEventListener("click", () => {
  elements.chatsForm.scrollIntoView({ behavior: "smooth", block: "center" });
  window.setTimeout(() => elements.goal.focus(), 350);
});

elements.themeToggleButton.addEventListener("click", () => {
  const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
});

elements.copyButton.addEventListener("click", async () => {
  if (!state.lastAnswer) return;
  await navigator.clipboard.writeText(state.lastAnswer);
  elements.copyButton.classList.add("copied");
  window.setTimeout(() => elements.copyButton.classList.remove("copied"), 1000);
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab, { manual: true }));
});

document.querySelectorAll(".idea-card").forEach((button) => {
  button.addEventListener("click", () => {
    elements.goal.value = button.dataset.goal || "";
    elements.currentGoal.textContent = elements.goal.value;
    persistSession();
    elements.chatsForm.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => elements.goal.focus(), 350);
  });
});

elements.goal.addEventListener("input", () => {
  elements.currentGoal.textContent = elements.goal.value.trim() || "新的菜谱规划";
});

[
  [elements.maxMinutes, "max_minutes"],
  [elements.allergies, "allergies"],
].forEach(([element, key]) => {
  element.addEventListener("input", () => {
    state.dirtyConstraints.add(key);
    updateConditionsSummary();
  });
  element.addEventListener("change", () => {
    state.dirtyConstraints.add(key);
    persistSession();
  });
});

restoreSession();
applyForm(state.form);
updateConditionsSummary();
renderConversation();
renderStoredResult();
applyTheme(document.documentElement.dataset.theme);
if (state.history.length) {
  elements.currentGoal.textContent = [...state.history].reverse().find((item) => item.role === "user")?.content || "新的菜谱规划";
  elements.goal.placeholder = "继续调整，例如：把第二天换成素菜";
  setStatus("对话已恢复", "status-success");
} else {
  elements.goal.placeholder = "例如：规划晚餐，控制在 30 分钟内";
}
refreshIcons();
loadHealth();
startHeroRotation();
