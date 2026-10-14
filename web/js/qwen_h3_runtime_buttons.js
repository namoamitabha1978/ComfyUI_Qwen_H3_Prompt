import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("[QwenH3] runtime buttons extension loaded");

// ---- 推理进度条：监听后端 websocket 推送，渲染到所有 QwenH3 节点的 runtime_status 控件 ----
function collectQwenNodes() {
  try {
    const nodes = app.graph?._nodes || [];
    return nodes.filter(
      (n) => n.type === "QwenH3PromptLocal" || n.constructor?.type === "QwenH3PromptLocal"
    );
  } catch (e) {
    return [];
  }
}

function progressBar(percent, width) {
  const filled = Math.round(Math.max(0, Math.min(1, percent)) * width);
  return "█".repeat(filled) + "░".repeat(Math.max(0, width - filled));
}

const PROGRESS_STAGE_LABEL = {
  routing: "路由",
  inference: "推理",
  repair: "修复",
  done: "完成",
};

function setStatusWidgetValue(node, text) {
  try {
    const w = node.widgets?.find((w) => w.name === "runtime_status");
    if (w) {
      w.value = text;
      w.callback?.();
      node.setDirtyCanvas?.(true, true);
    }
  } catch (e) {}
}

function updateProgress(d) {
  if (!d || typeof d !== "object") return;
  const stage = d.stage || "inference";
  const pct = Math.round((d.percent || 0) * 100);
  const label = d.label || PROGRESS_STAGE_LABEL[stage] || stage;
  let text;
  if (stage === "done") {
    text = "✓ 推理完成";
  } else {
    text = `${label} ${pct}% ${progressBar(d.percent || 0, 16)} (${d.current ?? 0}/${d.total ?? 0} tok)`;
  }
  const nodes = collectQwenNodes();
  nodes.forEach((n) => setStatusWidgetValue(n, text));
  if (stage === "done") {
    setTimeout(() => {
      nodes.forEach((n) => setStatusWidgetValue(n, "就绪"));
    }, 2500);
  }
}

api.addEventListener("qwen_h3_progress", (event) => {
  try {
    const d = event?.detail?.data ?? event?.detail;
    updateProgress(d);
  } catch (e) {
    console.error("[QwenH3] progress handler error:", e);
  }
});


// 自建模态弹窗：不依赖 window.comfyAPI.dialog / alert（在 iframe 中 alert 常被拦截），
// 任何 ComfyUI 版本都能可靠弹出。
function showModal(title, html) {
  try {
    const overlay = document.createElement("div");
    overlay.style.cssText =
      "position:fixed;top:0;left:0;width:100%;height:100%;z-index:99999;background:rgba(0,0,0,0.55);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText =
      "background:#1e1e1e;color:#eee;border:1px solid #444;border-radius:8px;max-width:720px;max-height:85vh;display:flex;flex-direction:column;box-shadow:0 8px 30px rgba(0,0,0,0.6);";
    const head = document.createElement("div");
    head.style.cssText = "padding:10px 14px;font-weight:bold;border-bottom:1px solid #444;display:flex;justify-content:space-between;align-items:center;";
    head.innerHTML = `<span>${title}</span>`;
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    closeBtn.style.cssText = "background:none;border:none;color:#ccc;font-size:16px;cursor:pointer;";
    closeBtn.onclick = () => overlay.remove();
    head.appendChild(closeBtn);
    const body = document.createElement("div");
    body.style.cssText = "padding:12px 14px;overflow:auto;";
    body.innerHTML = html;
    const okBtn = document.createElement("button");
    okBtn.textContent = "确定";
    okBtn.style.cssText = "margin:10px 14px;padding:6px 16px;background:#3b82f6;color:#fff;border:none;border-radius:6px;cursor:pointer;align-self:flex-end;";
    okBtn.onclick = () => overlay.remove();
    box.appendChild(head);
    box.appendChild(body);
    box.appendChild(okBtn);
    overlay.appendChild(box);
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    document.body.appendChild(overlay);
  } catch (e) {
    console.error("[QwenH3] showModal failed:", e);
    alert("请查看控制台错误");
  }
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}

function formatStatusHtml(d) {
  if (!d) return "<p>（无响应）</p>";
  const h = (s) => esc(s);
  const sections = [];
  const line = (k, v) => `<div><b>${k}</b>: ${v}</div>`;

  // 总体状态
  let html = `<div style="font:13px/1.7 sans-serif">`;
  html += `<h4 style="margin:4px 0">运行状态</h4>`;
  html += line("就绪", d.ready ? "✅ 是" : "❌ 否");
  html += line("选中后端", `${h(d.selected_backend || "无")}（${h(d.selected_runtime || "无")}）`);
  html += line("GPU 卸载可用", d.gpu_offload_available ? "是" : "否");

  // 独立 runtime
  const s = d.standalone;
  html += `<h4 style="margin:10px 0 4px">① 独立 runtime（standalone）</h4>`;
  if (s) {
    html += line("后端", h(s.backend));
    html += line("平台", h(s.platform));
    html += line("可执行", `${h(s.executable)}`);
    if (s.executable_file) html += line("  ├ 文件大小", `${s.executable_file.size_mb} MB`);
    html += line("n_gpu_layers", h(s.n_gpu_layers));
    html += line("--fit", `${s.fit}（目标 ${s.fit_target_mib} MiB）`);
    html += line("flash_attention", h(s.flash_attention));
    if (s.library_files && s.library_files.length) {
      html += line("库文件", `${s.library_files.length} 个`);
      html += `<pre style="font:11px/1.4 monospace;background:#222;padding:6px;border-radius:4px;overflow:auto">${s.library_files.map((f) => `${f.path}  (${f.size_mb} MB)`).join("\n")}</pre>`;
    }
  } else {
    html += `<div>未安装（可点下方“安装 NV / Vulkan runtime”按钮）</div>`;
  }

  // llama-cpp-python
  const p = d.llama_cpp_python;
  html += `<h4 style="margin:10px 0 4px">② llama-cpp-python（进程内 CUDA 兜底）</h4>`;
  html += p
    ? line("版本", `${h(p.version)} | gpu_offload=${p.gpu_offload}`) +
      line("native_lib", h(p.native_lib || "无")) +
      (p.native_lib_file ? line("  ├ 文件大小", `${p.native_lib_file.size_mb} MB`) : "")
    : `<div>未安装（走 ComfyUI 内置 Python，若无则请先安装 llama-cpp-python）</div>`;

  // 关键路径
  const pa = d.paths || {};
  html += `<h4 style="margin:10px 0 4px">③ 关键路径</h4>`;
  html += line("runtime_config", h(pa.runtime_config || "无"));
  html += line("runtime_dir", h(pa.runtime_dir || "无"));
  html += line("install_script", h(pa.install_script || "无"));
  html += line("model_dir", h(pa.model_dir || "无"));

  // 安装目录文件清单
  const rf = d.runtime_files || [];
  html += `<h4 style="margin:10px 0 4px">④ runtime 安装目录文件（${rf.length} 个）</h4>`;
  html += `<pre style="font:11px/1.4 monospace;background:#222;padding:6px;border-radius:4px;overflow:auto;max-height:220px">${rf.map((f) => `${f.path}  (${f.size_mb} MB)`).join("\n")}</pre>`;

  // runtime_config 内容
  const rc = d.runtime_config;
  html += `<h4 style="margin:10px 0 4px">⑤ runtime_config.json 关键项</h4>`;
  if (rc) {
    html += line("版本", `${h(rc.llama_cpp_tag || "?")}（commit ${h((rc.llama_cpp_commit || "").slice(0, 8))}）`);
    html += line("backend", h(rc.backend_description || rc.backend));
    html += line("安装时间", h(rc.installed_at || "?"));
    html += line("检测 GPU", h((rc.detected_gpus || []).join("；")));
    html += line("CUDA", rc.detected_cuda ? `driver=${h(rc.detected_cuda.driver_compatibility)} toolkit=${h(rc.detected_cuda.toolkit)}` : "无");
    html += `<pre style="font:10px/1.4 monospace;background:#222;padding:6px;border-radius:4px;overflow:auto;max-height:140px">${h(rc.version_output || "")}\n${h(rc.devices_output || "")}</pre>`;
  } else {
    html += `<div>未找到 runtime_config.json</div>`;
  }

  // 模型
  const m = d.models || {};
  html += `<h4 style="margin:10px 0 4px">⑥ 模型（${m.count ?? 0} 个 gguf）</h4>`;
  if (m.gguf && m.gguf.length) {
    html += `<pre style="font:11px/1.4 monospace;background:#222;padding:6px;border-radius:4px;overflow:auto">${m.gguf.map((g) => `${g.name}  (${g.size_mb} MB)\n  ${h(g.path)}`).join("\n")}</pre>`;
  } else {
    html += `<div>models/LLM 下未扫描到 gguf</div>`;
  }

  // 警告
  (d.warnings || []).forEach((w) => (html += `<div style="color:#ffb3b3">⚠ ${h(w)}</div>`));
  html += `<div style="margin-top:8px;color:#aaa;font-size:11px">检测时间：${h(d.time)}</div>`;
  html += `</div>`;
  return html;
}

app.registerExtension({
  name: "QwenH3.RuntimeButtons",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "QwenH3PromptLocal") return;
    console.log("[QwenH3] wiring buttons to", nodeType.name);
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = onNodeCreated?.apply(this, arguments);
      try {
        addButtons(this);
      } catch (e) {
        console.error("[QwenH3] addButtons:", e);
      }
      return ret;
    };
  },
});

function addButtons(node) {
  if (node._qwenH3Buttons) return;
  node._qwenH3Buttons = true;

  const nonSerializing = (w) => {
    // 正确禁用序列化：serialize 必须是布尔 false，而不是返回 undefined 的函数（函数是真值，仍会被写入工作流 JSON）。
    try {
      w.serialize = false;
      w.options = w.options || {};
      w.options.serialize = false;
    } catch (e) {}
    return w;
  };

  const status = nonSerializing(node.addWidget("text", "runtime_status", "", () => {}));
  status.disabled = true;

  const setStatus = (v) => {
    try { status.value = v; status.callback?.(); } catch (e) {}
    try { node.setDirtyCanvas?.(true, true); } catch (e) {}
  };

  const detect = () => {
    console.log("[QwenH3] detect clicked");
    setStatus("检测中...");
    api.fetchApi("/qwen_h3/runtime/status", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        console.log("[QwenH3] status fetched:", d);
        setStatus(`就绪=${d.ready ? "是" : "否"} 后端=${d.selected_backend || "无"}`);
        showModal("Qwen H3 运行时检测（路径 / 文件 / 状态）", formatStatusHtml(d));
      })
      .catch((e) => {
        console.error("[QwenH3] status fetch error:", e);
        setStatus("检测失败");
        showModal("Qwen H3 运行时检测", esc("请求失败：" + e));
      });
  };

  const install = (backend, label) => {
    console.log("[QwenH3] install clicked:", backend);
    setStatus(`安装 ${label} runtime 中...`);
    api.fetchApi("/qwen_h3/runtime/install", {
      method: "POST",
      body: JSON.stringify({ backend }),
      headers: { "Content-Type": "application/json" },
    })
      .then(async (r) => {
        const j = await r.json().catch(() => ({}));
        console.log("[QwenH3] install resp:", j);
        if (!j.started) {
          setStatus(j.error || "无法启动安装");
          showModal("Qwen H3 安装失败", esc(j.error || "无法启动安装"));
          return;
        }
        pollInstall(label, setStatus);
      })
      .catch((e) => {
        console.error("[QwenH3] install request error:", e);
        setStatus("安装请求失败");
        showModal("Qwen H3 安装", esc("请求失败：" + e));
      });
  };

  const pollInstall = (label, setStatus) => {
    let n = 0;
    const timer = setInterval(async () => {
      n++;
      try {
        const j = await (await api.fetchApi("/qwen_h3/runtime/install/status", { cache: "no-store" })).json();
        setStatus(`${j.state} ${label}${j.message ? " — " + j.message : ""}`);
        if (j.state === "done" || j.state === "error") {
          clearInterval(timer);
          showModal(
            "Qwen H3 安装结果",
            `<pre style="font:12px/1.5 monospace">${esc(j.message)}</pre>` +
              `<details><summary>日志尾部</summary><pre style="font:11px/1.4 monospace;max-height:300px;overflow:auto">${esc(j.log_tail || "(无)")}</pre></details>`
          );
          setStatus(j.state === "done" ? `${label} runtime 安装完成` : `${label} runtime 安装失败`);
        } else if (n > 300) {
          clearInterval(timer);
          setStatus("安装超时，请查看 ComfyUI 日志");
        }
      } catch (e) {
        if (n > 5) { clearInterval(timer); setStatus("轮询失败"); }
      }
    }, 2000);
  };

  node.addWidget("button", "🔍 检测当前 runtime", null, detect);
  node.addWidget("button", "🖥️ 安装 NV (CUDA) runtime", null, () => install("cuda", "NV(CUDA)"));
  node.addWidget("button", "🖥️ 安装 Vulkan runtime", null, () => install("vulkan", "Vulkan"));
  for (const w of node.widgets) {
    if (w.type === "button" && ["🔍 检测当前 runtime", "🖥️ 安装 NV (CUDA) runtime", "🖥️ 安装 Vulkan runtime"].includes(w.name)) {
      nonSerializing(w);
    }
  }
}
