/* global document, window, fetch */ 
(function () {
  const SF = (window.SF = window.SF || {});

  // ---------------- friendly error extraction ----------------
  SF.readFriendlyError = async function (resp) {
    try {
      const ct = resp.headers.get("content-type") || "";
      const body = await resp.text();

      // JSON payloads (FastAPI often returns {"detail": "..."} or {"detail":[{msg:...}, ...]})
      if (ct.includes("application/json")) {
        try {
          const j = JSON.parse(body);
          if (j && typeof j === "object") {
            // Prefer FastAPI/Pydantic structured errors and show field paths (loc)
            if (Array.isArray(j.detail)) {
              const parts = j.detail
                .map((e) => {
                  const where = Array.isArray(e.loc) ? e.loc.join(".") : "";
                  return where ? `${where}: ${e.msg}` : (e.msg || JSON.stringify(e));
                })
                .filter(Boolean);
              if (parts.length) return parts.join(" • ");
            }

            const d = j.detail ?? j.message ?? j.error ?? j.errors;
            if (typeof d === "string") return d;
            if (Array.isArray(d)) {
              const parts = d
                .map((x) => x?.msg || x?.detail || (typeof x === "string" ? x : JSON.stringify(x)))
                .filter(Boolean);
              if (parts.length) return parts.join(" • ");
            }
          }
        } catch {
          /* fall through to text handling */
        }
      }

      // Pydantic v2 plain-text validation error (e.g. "1 validation error for ForecastRequest ...")
      if (/validation error for/i.test(body)) {
        // strip “For further information …” tail
        const cleaned = body.replace(/For further information[\s\S]*$/i, "").trim();

        // try to capture the human part after "Value error,"
        const m = cleaned.match(/Value error,?\s*(.*?)(?:\n|$|\[)/i);
        if (m && m[1]) return m[1].trim();

        // otherwise use the last non-empty line
        const lines = cleaned.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
        if (lines.length > 1) return lines[lines.length - 1];
        return cleaned;
      }

      // If body starts with "Error: ..." use the part after colon
      const colon = body.match(/^Error:\s*(.*)$/i);
      if (colon && colon[1]) return colon[1].trim();

      // Last resort: plain text (shortened)
      const plain = body.replace(/<\/?[^>]+>/g, "").trim();
      if (plain) return plain.length > 500 ? plain.slice(0, 500) + "…" : plain;

      return `HTTP ${resp.status} ${resp.statusText}`;
    } catch {
      return `HTTP ${resp.status} ${resp.statusText}`;
    }
  };

  // ---------------- number formatters ----------------
  // integer stays fixed
  const intFmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

  // cache float formatters by decimals so we don't recreate them every time
  const _floatFmtCache = new Map();
  function getFloatFormatter(decimals) {
    const d = Math.max(0, Math.min(12, Number(decimals ?? 2)));
    if (_floatFmtCache.has(d)) return _floatFmtCache.get(d);
    const fmt = new Intl.NumberFormat(undefined, {
      minimumFractionDigits: d,
      maximumFractionDigits: d,
    });
    _floatFmtCache.set(d, fmt);
    return fmt;
  }

  // now accept an optional `decimals` arg
  SF.formatSales = function (value, unitType, decimals) {
    const v = typeof value === "number" ? value : parseFloat(value);
    if (!isFinite(v)) return "";
    if (unitType === "int" || unitType === "integer") return intFmt.format(Math.round(v));
    return getFloatFormatter(decimals).format(v);
  };

  // Show/hide decimal places input (forecast page)
  function toggleDecimalPlacesVisibility() {
    const sel = document.querySelector("#unit_type");
    const dp = document.querySelector("#decimal_places");
    const wrap = document.querySelector("#decimal_wrap");
    if (!sel || !dp || !wrap) return;

    const isFloat = (sel.value || "integer") === "float";
    wrap.style.display = isFloat ? "" : "none";
    dp.toggleAttribute("disabled", !isFloat);
  }

  SF.applyUnitTypeToPage = function () {
    const select = document.querySelector("#unit_type");
    const unitType = (select && select.value) || "integer";

    // read decimals only if float and the input is enabled
    let decimals;
    const dpEl = document.querySelector("#decimal_places");
    if (unitType === "float" && dpEl && !dpEl.disabled) {
      decimals = parseInt(dpEl.value || "2", 10);
    }

    document.querySelectorAll("[data-sales]").forEach((el) => {
      const raw = el.getAttribute("data-sales");
      el.textContent = SF.formatSales(raw, unitType, decimals);
    });

    toggleDecimalPlacesVisibility();
  };

  SF.renderPredictions = function (tbody, rows, unitType, decimals) {
    if (!tbody) return;
    tbody.innerHTML = "";
    if (!rows || rows.length === 0) {
      const tr = document.createElement("tr");
      tr.className = "msg-row";
      tr.innerHTML = `<td colspan="6" class="hint">No results.</td>`;
      tbody.appendChild(tr);
      return;
    }
    for (const r of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.store_id}</td>
        <td>${r.item_id}</td>
        <td>${r.base_date}</td>
        <td>${r.target_date}</td>
        <td class="text-right">${r.horizon}</td>
        <td class="text-right" data-sales="${r.sales}">${SF.formatSales(r.sales, unitType, decimals)}</td>
      `;
      tbody.appendChild(tr);
    }
  };

  // ---------------- Forecast UI helpers ----------------
  window.toggleUploadDisabled = function () {
    const useDemo = document.querySelector("#use_demo")?.checked;
    const file = document.querySelector("#upload_csv");
    if (file) file.disabled = !!useDemo;
  };

  // optional: models dir toggle if present
  window.toggleModelsDirDisabled = function () {
    const useDefault = document.querySelector("#use_default_models");
    const useCustom = document.querySelector("#use_custom_models");
    const input = document.querySelector("#models_dir");
    if (!input) return;
    // If "use_default_models" exists and is checked => disable text input
    if (useDefault) {
      input.disabled = !!useDefault.checked;
      return;
    }
    // Or if "use_custom_models" exists and is unchecked => disable text input
    if (useCustom) {
      input.disabled = !useCustom.checked;
    }
  };

  window.onScopeChange = function () {
    const scope = document.querySelector("#scope")?.value || "single";
    const show = (id, yes) => document.getElementById(id)?.classList.toggle("hide", !yes);

    show("single-target-fields", scope === "single");
    show("store-only-field", scope === "latest_per_store");
    show("item-only-field", scope === "latest_per_item");
    show("last-n-field", scope === "last_n_days");
    show("since-field", scope === "since_date");
    show("at-date-field", scope === "at_date");

    const note = document.getElementById("scope-note");
    if (note) {
      const msg = {
        single: "Predict for one specific store_id & item_id at their latest date.",
        latest_per_pair: "Predict at the latest date for each (store_id, item_id) pair.",
        latest_per_store: "For a store, predict latest date per item in that store.",
        latest_per_item: "For an item, predict latest date across all stores.",
        last_n_days: "Predict for all rows in the last N days.",
        since_date: "Predict for rows on/after a specific date.",
        at_date: "Predict for rows exactly at a specific date."
      }[scope] || "";
      note.textContent = msg;
    }
  };

  // Optional: helper for the “Try an example” button
  window.applyExample = function () {
    const sel = document.getElementById("example_scenario");
    const val = sel ? sel.value : "";

    // Always default to demo CSV for examples
    const useDemo = document.getElementById("use_demo");
    if (useDemo) {
      useDemo.checked = true;
      window.toggleUploadDisabled();
    }

    if (val === "single_promo_derived") {
      document.getElementById("scope").value = "single";
      window.onScopeChange();
      document.getElementById("target_store_id").value = "S01";
      document.getElementById("target_item_id").value = "I01";
      document.getElementById("horizons").value = "1-7";
      document.getElementById("promo_future").value = "0,0,0,0.5,0,0,0";
      document.getElementById("price_future").value = "";
    } else if (val === "latest_per_store_clean") {
      document.getElementById("scope").value = "latest_per_store";
      window.onScopeChange();
      document.getElementById("target_store_id_only").value = "S03";
      document.getElementById("horizons").value = "1-7";
      document.getElementById("promo_future").value = "";
      document.getElementById("price_future").value = "";
    } else if (val === "last7_markdown") {
      document.getElementById("scope").value = "last_n_days";
      window.onScopeChange();
      document.getElementById("days").value = "7";
      document.getElementById("horizons").value = "1-3";
      document.getElementById("price_future").value = "0.9";
      document.getElementById("promo_future").value = "0";
    }
  };

  window.resetForm = function () {
    document.getElementById("forecast-form")?.reset();
    const demo = document.querySelector("#use_demo");
    if (demo) demo.checked = true;
    toggleUploadDisabled();
    onScopeChange();
    SF.applyUnitTypeToPage();
    // optional: reset models dir toggle
    if (typeof toggleModelsDirDisabled === "function") toggleModelsDirDisabled();
  };

  // ----- paging & chips (forecast) -----
  let currentPage = 1;
  let totalPages = 1;

  function updatePageInfo(total, page, pageSize) {
    const info = document.getElementById("page-info");
    const pages = Math.max(1, Math.ceil(total / pageSize));
    currentPage = page;
    totalPages = pages;
    if (info) info.textContent = `Page ${page} / ${pages}`;
  }

  window.prevPage = function () {
    if (currentPage > 1) runForecast(currentPage - 1);
  };
  window.nextPage = function () {
    if (currentPage < totalPages) runForecast(currentPage + 1);
  };

  const pageSizeSelect = document.getElementById("page_size");
  if (pageSizeSelect) {
    pageSizeSelect.addEventListener("change", () => runForecast(1));
  }

  // ----- forecast request -----
  window.runForecast = async function (page) {
    const scope = document.querySelector("#scope")?.value || "single";
    const horizons = document.querySelector("#horizons")?.value || "";
    const useDemo = document.querySelector("#use_demo")?.checked;
    const pageSize = parseInt(document.querySelector("#page_size")?.value || "100", 10);

    // scope params
    const storeId = document.querySelector("#target_store_id")?.value || null;
    const itemId = document.querySelector("#target_item_id")?.value || null;
    const storeOnly = document.querySelector("#target_store_id_only")?.value || null;
    const itemOnly = document.querySelector("#target_item_id_only")?.value || null;
    const nDays = document.querySelector("#days")?.value || null;
    const sinceDate = document.querySelector("#since_date")?.value || null;
    const atDate = document.querySelector("#at_date")?.value || null;

    // future plans
    const priceFuture = (document.querySelector("#price_future")?.value || "").trim();
    const promoFuture = (document.querySelector("#promo_future")?.value || "").trim();

    // output formatting
    const unitType = document.querySelector("#unit_type")?.value || "integer";
    const decimalPlacesEl = document.querySelector("#decimal_places");
    const decimalPlaces = decimalPlacesEl && !decimalPlacesEl.disabled ? parseInt(decimalPlacesEl.value || "2", 10) : undefined;

    // models dir (optional override)
    const modelsDirInput = (document.querySelector("#models_dir")?.value || "").trim();
    const useDefault = document.querySelector("#use_default_models")?.checked;   // if present
    const useCustom = document.querySelector("#use_custom_models")?.checked;     // if present
    // Decide whether to send models_dir:
    // - If "use_default_models" exists and is checked => DON'T send override
    // - If "use_custom_models" exists and is unchecked => DON'T send override
    // - Otherwise, send if non-empty
    let modelsDir = "";
    if (typeof useDefault === "boolean") {
      modelsDir = useDefault ? "" : modelsDirInput;
    } else if (typeof useCustom === "boolean") {
      modelsDir = useCustom ? modelsDirInput : "";
    } else {
      modelsDir = modelsDirInput;
    }

    // resolve scope-specific ids
    let store_id = null, item_id = null;
    if (scope === "single") {
      store_id = storeId;
      item_id = itemId;
    } else if (scope === "latest_per_store") {
      store_id = storeOnly;
    } else if (scope === "latest_per_item") {
      item_id = itemOnly;
    }

    // ---------- quick client-side validations (friendly) ----------
    const showHint = (msg) => {
      const tbody = document.getElementById("results-body");
      if (tbody) {
        tbody.innerHTML = `<tr class="msg-row"><td colspan="6" class="hint">${msg}</td></tr>`;
      }
    };

    if (scope === "single" && (!store_id || !item_id)) {
      showHint("Please provide both Store ID and Item ID for scope “Single target”.");
      return;
    }
    if (scope === "latest_per_store" && !store_id) {
      showHint("Please provide a Store ID for scope “Latest per store”.");
      return;
    }
    if (scope === "latest_per_item" && !item_id) {
      showHint("Please provide an Item ID for scope “Latest per item”.");
      return;
    }
    if (scope === "last_n_days" && !nDays) {
      showHint("Please enter how many days for scope “Last N days”.");
      return;
    }
    if (scope === "since_date" && !sinceDate) {
      showHint("Please choose a date for scope “Since date”.");
      return;
    }
    if (scope === "at_date" && !atDate) {
      showHint("Please choose the exact date for scope “At date”.");
      return;
    }

    try {
      let rows = [];
      let total = 0;

      if (useDemo) {
        // JSON → /forecast (NOTE: embed under { req: {...}, models_dir })
        const body = {
          req: {
            scope,
            horizons,
            use_server_csv: true,
            page: page || 1,
            page_size: pageSize,
            // scope params
            store_id, item_id,
            n_days: nDays ? parseInt(nDays, 10) : undefined,
            since_date: sinceDate || undefined,
            at_date: atDate || undefined,
            // future plans
            price_future: priceFuture || undefined,
            promo_future: promoFuture || undefined,
            // output prefs (optional server-side formatting)
            unit_type: unitType,
            decimal_places: unitType === "float" ? decimalPlaces : undefined
          },
          models_dir: modelsDir || undefined
        };
        const resp = await fetch("/forecast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          const msg = await SF.readFriendlyError(resp);
          throw new Error(msg);
        }
        const data = await resp.json();
        rows = data.predictions || [];
        total = data.page?.total || rows.length;
      } else {
        // multipart → /forecast/csv
        const fd = new FormData();
        const f = document.querySelector("#upload_csv")?.files?.[0];
        if (f) fd.append("file", f);
        fd.append("scope", scope);
        if (store_id) fd.append("store_id", store_id);
        if (item_id) fd.append("item_id", item_id);
        if (horizons) fd.append("horizons", horizons);
        fd.append("use_demo_csv", "false");
        fd.append("page", String(page || 1));
        fd.append("page_size", String(pageSize));
        if (nDays) fd.append("n_days", String(parseInt(nDays, 10)));
        if (sinceDate) fd.append("since_date", sinceDate);
        if (atDate) fd.append("at_date", atDate);
        if (priceFuture) fd.append("price_future", priceFuture);
        if (promoFuture) fd.append("promo_future", promoFuture);
        fd.append("unit_type", unitType);
        if (unitType === "float" && typeof decimalPlaces === "number") {
          fd.append("decimal_places", String(decimalPlaces));
        }
        // add models_dir override if provided
        if (modelsDir) fd.append("models_dir", modelsDir);

        const resp = await fetch("/forecast/csv", { method: "POST", body: fd });
        if (!resp.ok) {
          const msg = await SF.readFriendlyError(resp);
          throw new Error(msg);
        }
        const data = await resp.json();
        rows = data.predictions || [];
        total = data.page?.total || rows.length;
      }

      // render
      SF.renderPredictions(
        document.getElementById("results-body"),
        rows,
        unitType,
        unitType === "float" ? decimalPlaces : undefined
      );
      updatePageInfo(total, page || 1, pageSize);
      SF.applyUnitTypeToPage();
    } catch (err) {
      const tbody = document.getElementById("results-body");
      const message = (err && err.message) ? err.message : String(err);
      if (tbody) {
        tbody.innerHTML = `<tr class="msg-row"><td colspan="6" class="hint">Error: ${message}</td></tr>`;
      }
      console.error(err);
    }
  };

  // ---------------- Training helpers & request ----------------
  window.trToggleUploadDisabled = function () {
    const useDemo = document.querySelector("#tr_use_demo")?.checked;
    const file = document.querySelector("#tr_upload_csv");
    if (file) file.disabled = !!useDemo;
  };

  window.trOnModeChange = function () {
    const mode = document.querySelector("#train_mode")?.value || "global";
    const row = document.getElementById("train-scope-row");
    if (row) row.style.display = mode === "per_group" ? "" : "none";
  };

  window.resetTrainingForm = function () {
    document.getElementById("training-form")?.reset();
    const demo = document.querySelector("#tr_use_demo");
    if (demo) demo.checked = true;
    trToggleUploadDisabled();
    trOnModeChange();
    const log = document.getElementById("train-log");
    if (log) log.textContent = "—";
    const status = document.getElementById("train-status");
    if (status) status.textContent = "";
  };

  // Presets: fill form fields with sensible bundles
  window.applyTrainingPreset = function () {
    const p = (document.getElementById("tr_preset") || {}).value || "balanced";

    const setVal = (id, v) => {
      const el = document.getElementById(id);
      if (el) el.value = v;
    };

    // common safe defaults we won't change unless needed
    setVal("tr_tree_method", "hist");

    if (p === "fast") {
      setVal("tr_n_estimators", 150);
      setVal("tr_max_depth", 4);
      setVal("tr_learning_rate", 0.08);
      setVal("tr_subsample", 0.9);
      setVal("tr_colsample_bytree", 0.9);
    } else if (p === "quality") {
      setVal("tr_n_estimators", 800);
      setVal("tr_max_depth", 8);
      setVal("tr_learning_rate", 0.03);
      setVal("tr_subsample", 0.8);
      setVal("tr_colsample_bytree", 0.8);
      setVal("tr_min_child_weight", 1);
      setVal("tr_gamma", 0);
      setVal("tr_reg_alpha", "");
      setVal("tr_reg_lambda", "");
    } else if (p === "gpu") {
      setVal("tr_tree_method", "gpu_hist");
      setVal("tr_n_estimators", 600);
      setVal("tr_max_depth", 7);
      setVal("tr_learning_rate", 0.05);
      setVal("tr_subsample", 0.85);
      setVal("tr_colsample_bytree", 0.85);
    } else {
      // balanced
      setVal("tr_n_estimators", 300);
      setVal("tr_max_depth", 6);
      setVal("tr_learning_rate", 0.05);
      setVal("tr_subsample", "");
      setVal("tr_colsample_bytree", "");
      setVal("tr_min_child_weight", "");
      setVal("tr_gamma", "");
      setVal("tr_reg_alpha", "");
      setVal("tr_reg_lambda", "");
    }

    trSuggestES();
  };

  // Server-side auto-suggest for early_stopping_rounds
  window.trSuggestES = async function () {
    const ne = parseInt((document.getElementById("tr_n_estimators") || {}).value || "0", 10);
    const tail = (document.getElementById("tr_valid_tail_days") || {}).value || "";
    const cutoff = (document.getElementById("tr_valid_cutoff_date") || {}).value || "";
    const out = document.getElementById("tr_es_rounds");
    const status = document.getElementById("train-status");

    if (!out) return;

    if (!Number.isFinite(ne) || ne <= 0) {
      out.value = "";
      if (status) status.textContent = "Enter n_estimators first to suggest early_stopping_rounds.";
      return;
    }

    try {
      const fd = new FormData();
      fd.append("n_estimators", String(ne));
      if (tail) fd.append("valid_tail_days", String(parseInt(tail, 10)));
      if (cutoff) fd.append("valid_cutoff_date", cutoff);
      // optional clamps; tweak if desired:
      fd.append("cap_min", "20");
      fd.append("cap_max", "120");

      const resp = await fetch("/train/suggest_es", { method: "POST", body: fd });
      if (!resp.ok) {
        const msg = await (window.SF?.readFriendlyError ? SF.readFriendlyError(resp) : resp.text());
        throw new Error(msg);
      }
      const data = await resp.json();
      out.value = String(data.suggestion);

      if (status) {
        const pct = Math.round((data.percent || 0) * 100);
        const days = data.used_tail_days ?? (cutoff ? 28 : null);
        const tailTxt = days ? `${days} days` : "no validation split";
        status.textContent = `Suggested ${data.suggestion} (~${pct}% of ${ne}, ${tailTxt}).`;
        setTimeout(() => { if (status.textContent?.includes("Suggested")) status.textContent = ""; }, 3500);
      }
    } catch (err) {
      if (status) status.textContent = `Suggest failed: ${(err && err.message) ? err.message : String(err)}`;
      console.error(err);
    }
  };

  window.runTraining = async function () {
    const btn = document.getElementById("train-btn");
    const status = document.getElementById("train-status");
    const log = document.getElementById("train-log");
    const setBusy = (b) => {
      if (btn) btn.disabled = b;
      if (status) status.textContent = b ? "Training… please wait" : "";
    };

    // collect fields
    const mode = document.querySelector("#train_mode")?.value || "global";
    const trainScope = document.querySelector("#train_scope")?.value || "pair";
    const useDemo = !!document.querySelector("#tr_use_demo")?.checked;
    const modelsDir = document.querySelector("#tr_models_dir")?.value || "models";
    const horizons = document.querySelector("#tr_horizons")?.value || "";
    const holCountry = document.querySelector("#tr_hol_country")?.value || "US";
    const holSubdiv = document.querySelector("#tr_hol_subdiv")?.value || "";

    const nthread = document.querySelector("#tr_nthread")?.value || "4";
    const nEstimators = document.querySelector("#tr_n_estimators")?.value || "300";
    const maxDepth = document.querySelector("#tr_max_depth")?.value || "6";
    const learningRate = document.querySelector("#tr_learning_rate")?.value || "0.05";
    const treeMethod = document.querySelector("#tr_tree_method")?.value || "hist";

    const subsample = document.querySelector("#tr_subsample")?.value || "";
    const colsample_bytree = document.querySelector("#tr_colsample_bytree")?.value || "";
    const min_child_weight = document.querySelector("#tr_min_child_weight")?.value || "";
    const gamma = document.querySelector("#tr_gamma")?.value || "";
    const reg_alpha = document.querySelector("#tr_reg_alpha")?.value || "";
    const reg_lambda = document.querySelector("#tr_reg_lambda")?.value || "";
    const max_bin = document.querySelector("#tr_max_bin")?.value || "";
    const random_state = document.querySelector("#tr_random_state")?.value || "";
    const req_notna = document.querySelector("#tr_req_notna")?.value || "";

    const cutoff = document.querySelector("#tr_valid_cutoff_date")?.value || "";
    const tailDays = document.querySelector("#tr_valid_tail_days")?.value || "";
    const esRounds = document.querySelector("#tr_es_rounds")?.value || "";
    const verboseEval = !!document.querySelector("#tr_verbose_eval")?.checked;
    const singleThreadEnv = !!document.querySelector("#tr_single_thread")?.checked;
    const wipe = !!document.querySelector("#tr_wipe")?.checked;

    try {
      setBusy(true);

      const fd = new FormData();
      fd.append("mode", mode);
      if (mode === "per_group") fd.append("train_scope", trainScope);

      // data
      fd.append("use_demo_csv", useDemo ? "true" : "false");
      const file = document.querySelector("#tr_upload_csv")?.files?.[0];
      if (!useDemo && file) fd.append("file", file);

      // paths & horizons & holidays
      fd.append("models_dir", modelsDir);
      if (horizons) fd.append("horizons", horizons);
      fd.append("hol_country", holCountry);
      if (holSubdiv) fd.append("hol_subdiv", holSubdiv);
      if (wipe) fd.append("wipe", "true"); // server may ignore if unsupported

      // xgb knobs
      fd.append("nthread", String(parseInt(nthread, 10)));
      fd.append("n_estimators", String(parseInt(nEstimators, 10)));
      fd.append("max_depth", String(parseInt(maxDepth, 10)));
      fd.append("learning_rate", String(parseFloat(learningRate)));
      fd.append("tree_method", treeMethod || "hist");

      if (subsample) fd.append("subsample", subsample);
      if (colsample_bytree) fd.append("colsample_bytree", colsample_bytree);
      if (min_child_weight) fd.append("min_child_weight", min_child_weight);
      if (gamma) fd.append("gamma", gamma);
      if (reg_alpha) fd.append("reg_alpha", reg_alpha);
      if (reg_lambda) fd.append("reg_lambda", reg_lambda);
      if (max_bin) fd.append("max_bin", max_bin);
      if (random_state) fd.append("random_state", random_state);
      if (req_notna) fd.append("required_feature_notna", req_notna);

      // validation
      if (cutoff) fd.append("valid_cutoff_date", cutoff);
      if (tailDays) fd.append("valid_tail_days", String(parseInt(tailDays, 10)));
      if (esRounds) fd.append("early_stopping_rounds", String(parseInt(esRounds, 10)));
      fd.append("verbose_eval", verboseEval ? "1" : "0");
      fd.append("enforce_single_thread_env", singleThreadEnv ? "true" : "false");

      const resp = await fetch("/train", { method: "POST", body: fd });

      if (!resp.ok) {
        const msg = await SF.readFriendlyError(resp);
        throw new Error(msg);
      }

      // Success: pretty-print JSON if available; otherwise show plain text
      let payloadText;
      const ct = resp.headers.get("content-type") || "";
      if (ct.includes("application/json")) {
        const data = await resp.json();
        payloadText = JSON.stringify(data, null, 2);
      } else {
        payloadText = await resp.text();
      }

      if (log) log.textContent = payloadText || "Training finished.";
      if (status) status.textContent = "✅ Training finished";
    } catch (err) {
      const msg = (err && err.message) ? err.message : String(err);
      if (log) log.textContent = `Error: ${msg}`;
      if (status) status.textContent = "❌ Training failed";
      console.error(err);
    } finally {
      setBusy(false);
    }
  };

  // ---------------- DOM init ----------------
  document.addEventListener("DOMContentLoaded", () => {
    // Forecast page init (if present)
    toggleUploadDisabled();
    onScopeChange();
    const unitSel = document.getElementById("unit_type");
    if (unitSel) unitSel.addEventListener("change", SF.applyUnitTypeToPage);
    const dpEl = document.getElementById("decimal_places");
    if (dpEl) dpEl.addEventListener("input", SF.applyUnitTypeToPage);
    SF.applyUnitTypeToPage();
    // optional: models dir toggle
    const useDefault = document.querySelector("#use_default_models");
    const useCustom = document.querySelector("#use_custom_models");
    if (useDefault) useDefault.addEventListener("change", toggleModelsDirDisabled);
    if (useCustom) useCustom.addEventListener("change", toggleModelsDirDisabled);
    toggleModelsDirDisabled();

    // Training page init (if present)
    trToggleUploadDisabled();
    trOnModeChange();
  });
})();
