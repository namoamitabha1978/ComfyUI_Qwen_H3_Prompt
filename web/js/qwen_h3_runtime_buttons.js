import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const getDialog = () => window.comfyAPI?.dialog?.ComfyDialog || null;

function showDialog(title, html) {
  const C = getDialog();
  if (C) {
    new C().show(title, html);
  } else {
    alert(title + "\n\n" + String(html || "").replace(/<[^>]+>/g, ""));
  }
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}

function formatStatus(d) {
  if (!d) return "（无响应）";
  const lines = [];
  lines.push(`就绪: ${d.ready ? "是" : "否"}`);
  lines.push(`选中后端: ${d.selected_backend || "无"}（${d.selected_runtime || "无"}）`);
  lines.push(`GPU 卸载可用: ${d.gpu_offload_available ? "是" : "否"}`);
  const s = d.standalone;
  if (s) {
    lines.push(`独立 runtime: ${s.backend}`);
    lines.push(`  可执行: ${s.executable}`);
    lines.push(`  library: ${(s.library_dirs || []).join(";")}`);
    lines.push(`  --fit=${s.fit}（target ${s.fit_target_mib} MiB）| n_gpu_layers=${s.n_gpu_layers} | flash=${s.flash_attention}`);
  } else {
    lines.push("独立 runtime: 未安装");
  }
  const p = d.llama_cpp_python;
  lines.push(p ? `llama-cpp-python: v${p.version} | gpu_offload=${p.gpu_offload} | ${p.native_lib}` : "llama-cpp-python: 未安装");
  const pa = d.paths || {};
  lines.push(`runtime_config: ${pa.runtime_config || "无"}`);
  lines.push(`runtime_dir: ${pa.runtime_dir || "无"}`);
  lines.push(`install_script: ${pa.install_script || "无"}`);
  lines.push(`model_dir: ${pa.model_dir || "无"}（${d.models?.count ?? 0} 个 gguf）`);
  (d.models?.gguf || []).forEach((g) => lines.push(`  ${g}`));
  (d.warnings || []).forEach((w) => lines.push(`[警告] ${w}`));
  lines.push(`时间: ${d.time}`);
  return lines.join("\n");
}

app.registerExtension({
  name: "QwenH3.RuntimeButtons",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "QwenH3PromptLocal") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = onNodeCreated?.apply(this, arguments);
      try {
        addButtons(this);
      } catch (e) {
        console.error("QwenH3 runtime buttons:", e);
      }
      return ret;
    };
  },
});

function addButtons(node) {
  if (node._qwenH3Buttons) return;
  node._qwenH3Buttons = true;

  const status = node.addWidget("text", "runtime_status", "", () => {});
  status.disabled = true;
  status.serializeValue = () => undefined;

  const setStatus = (v) => {
    try {
      status.value = v;
      status.callback?.();
    } catch (e) {}
    try {
      node.setDirtyCanvas?.(true, true);
    } catch (e) {}
  };

  const detect = () => {
    setStatus("检测中...");
    api.fetchApi("/qwen_h3/runtime/status", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        setStatus(`就绪=${d.ready ? "是" : "否"} 后端=${d.selected_backend || "无"}`);
        showDialog("Qwen H3 运行时检测", `<pre style="font:12px/1.5 monospace">${esc(formatStatus(d))}</pre>`);
      })
      .catch((e) => {
        setStatus("检测失败");
        showDialog("Qwen H3 运行时检测", esc("请求失败：" + e));
      });
  };

  const install = (backend, label) => {
    setStatus(`安装 ${label} runtime 中...`);
    api.fetchApi("/qwen_h3/runtime/install", {
      method: "POST",
      body: JSON.stringify({ backend }),
      headers: { "Content-Type": "application/json" },
    })
      .then(async (r) => {
        const j = await r.json().catch(() => ({}));
        if (!j.started) {
          setStatus(j.error || "无法启动安装");
          showDialog("Qwen H3 安装失败", esc(j.error || "无法启动安装"));
          return;
        }
        pollInstall(label, setStatus);
      })
      .catch((e) => {
        setStatus("安装请求失败");
        showDialog("Qwen H3 安装", esc("请求失败：" + e));
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
          showDialog(
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
        if (n > 5) {
          clearInterval(timer);
          setStatus("轮询失败");
        }
      }
    }, 2000);
  };

  node.addWidget("button", "🔍 检测当前 runtime", null, detect);
  node.addWidget("button", "🖥️ 安装 NV (CUDA) runtime", null, () => install("cuda", "NV(CUDA)"));
  node.addWidget("button", "🖥️ 安装 Vulkan runtime", null, () => install("vulkan", "Vulkan"));
}
