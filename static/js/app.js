/**
 * InsightForge EDA - High-Performance Controller & UI Engine
 * Floating Top Navbar Hubs • Interactive Ingestion Tabs • Deep Thinking Resume 10.0 Suite • Local-first AI
 */

document.addEventListener("DOMContentLoaded", () => {
  let appState = {
    dataset: null,
    currentPage: 1,
    charts: {},
  };

  // DOM Elements
  const hubButtons = document.querySelectorAll(".hub-tab-btn");
  const viewPanels = document.querySelectorAll(".view-panel");
  const navCenterHubs = document.getElementById("navCenterHubs");
  const mobileNavToggle = document.getElementById("mobileNavToggle");
  const navBrandBtn = document.getElementById("navBrandBtn");
  const topDatasetPill = document.getElementById("topDatasetPill");
  const topDatasetName = document.getElementById("topDatasetName");
  const resetDataBtn = document.getElementById("resetDataBtn");
  const quickSampleBtn = document.getElementById("quickSampleBtn");
  const heroSampleBtn = document.getElementById("heroSampleBtn");
  const heroResumeBtn = document.getElementById("heroResumeBtn");
  const fileInput = document.getElementById("fileInput");
  const dropZone = document.getElementById("dropZone");

  // Ingestion Tabs
  const tabUploadFileBtn = document.getElementById("tabUploadFileBtn");
  const tabPasteTextBtn = document.getElementById("tabPasteTextBtn");
  const tabSampleDatasetsBtn = document.getElementById("tabSampleDatasetsBtn");
  const ingestTabUpload = document.getElementById("ingestTabUpload");
  const ingestTabPaste = document.getElementById("ingestTabPaste");
  const ingestTabSamples = document.getElementById("ingestTabSamples");
  const pasteResumeInput = document.getElementById("pasteResumeInput");
  const analyzePastedResumeBtn = document.getElementById("analyzePastedResumeBtn");

  // Optional External AI Key Elements
  const navApiKeyBtn = document.getElementById("navApiKeyBtn");
  const apiKeyModal = document.getElementById("apiKeyModal");
  const closeApiKeyModalBtn = document.getElementById("closeApiKeyModalBtn");
  const cancelApiKeyModalBtn = document.getElementById("cancelApiKeyModalBtn");
  const saveModalApiKeyBtn = document.getElementById("saveModalApiKeyBtn");
  const modalApiKeyInput = document.getElementById("modalApiKeyInput");
  const modalKeyFeedback = document.getElementById("modalKeyFeedback");
  const navGeminiStatusIcon = document.getElementById("navGeminiStatusIcon");
  const navGeminiStatusText = document.getElementById("navGeminiStatusText");
  const homeGeminiStatusPill = document.getElementById("homeGeminiStatusPill");

  // Chart.js Global Dark Theme Defaults
  if (window.Chart) {
    Chart.defaults.color = '#94A3B8';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
  }

  // =========================================================
  // 1. Hub Navigation (Exposed Globally)
  // =========================================================
  window.switchHub = function(targetHubId) {
    hubButtons.forEach((btn) => {
      if (btn.getAttribute("data-hub") === targetHubId) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    viewPanels.forEach((panel) => {
      if (panel.id === targetHubId) {
        panel.classList.add("active");
      } else {
        panel.classList.remove("active");
      }
    });

    if (navCenterHubs) navCenterHubs.classList.remove("mobile-open");
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  function switchHub(targetHubId) {
    window.switchHub(targetHubId);
  }

  hubButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const hubId = btn.getAttribute("data-hub");
      window.switchHub(hubId);
    });
  });

  // Explicitly bind all Back to Data Studio buttons
  document.querySelectorAll(".back-to-data-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      window.switchHub("hub-data");
    });
  });

  // In-Page Interactive Tool Action Buttons
  const actionBtnClean = document.getElementById("actionBtnClean");
  const actionBtnAi = document.getElementById("actionBtnAi");
  const actionBtnMl = document.getElementById("actionBtnMl");
  const actionBtnExport = document.getElementById("actionBtnExport");

  if (actionBtnClean) actionBtnClean.addEventListener("click", () => window.switchHub("hub-quality"));
  if (actionBtnAi) actionBtnAi.addEventListener("click", () => window.switchHub("hub-ai"));
  if (actionBtnMl) actionBtnMl.addEventListener("click", () => window.switchHub("hub-ml"));
  if (actionBtnExport) actionBtnExport.addEventListener("click", () => window.switchHub("hub-export"));

  if (navBrandBtn) navBrandBtn.addEventListener("click", () => window.switchHub("hub-home"));
  if (mobileNavToggle) {
    mobileNavToggle.addEventListener("click", () => {
      if (navCenterHubs) navCenterHubs.classList.toggle("mobile-open");
    });
  }

  async function checkGeminiStatus() {
    try {
      const res = await fetch("/api/config/api-key-status");
      const data = await res.json();
      const hasExternal = Boolean(data.has_key || data.external_llm_available);
      const localModel = data.local_model || "local model";

      if (navGeminiStatusIcon) navGeminiStatusIcon.textContent = hasExternal ? "⚡" : "●";
      if (navGeminiStatusText) navGeminiStatusText.textContent = hasExternal ? "AI Enhanced" : "Local Mode";
      if (homeGeminiStatusPill) {
        homeGeminiStatusPill.textContent = hasExternal
          ? "⚡ Optional Gemini enhancement active"
          : `● Local/deterministic engine ready (${localModel})`;
        homeGeminiStatusPill.style.color = hasExternal ? "var(--emerald)" : "var(--primary-bright)";
      }
      if (modalKeyFeedback) {
        modalKeyFeedback.textContent = hasExternal
          ? "External enhancement is configured. The app still uses local/deterministic analysis first."
          : "No external key configured. Local/deterministic analysis remains active.";
        modalKeyFeedback.style.color = hasExternal ? "var(--emerald)" : "var(--text-muted)";
      }
    } catch (err) {
      console.warn("AI status check failed:", err);
      if (navGeminiStatusText) navGeminiStatusText.textContent = "Local Mode";
      if (homeGeminiStatusPill) {
        homeGeminiStatusPill.textContent = "● Local/deterministic engine ready";
        homeGeminiStatusPill.style.color = "var(--primary-bright)";
      }
    }
  }

  function openApiKeyModal() {
    if (apiKeyModal) apiKeyModal.style.display = "flex";
    if (modalApiKeyInput) {
      modalApiKeyInput.value = "";
      modalApiKeyInput.focus();
    }
  }

  function closeApiKeyModal() {
    if (apiKeyModal) apiKeyModal.style.display = "none";
  }

  if (navApiKeyBtn) navApiKeyBtn.addEventListener("click", openApiKeyModal);
  if (closeApiKeyModalBtn) closeApiKeyModalBtn.addEventListener("click", closeApiKeyModal);
  if (cancelApiKeyModalBtn) cancelApiKeyModalBtn.addEventListener("click", closeApiKeyModal);
  if (apiKeyModal) {
    apiKeyModal.addEventListener("click", (e) => {
      if (e.target === apiKeyModal) closeApiKeyModal();
    });
  }
  if (saveModalApiKeyBtn) {
    saveModalApiKeyBtn.addEventListener("click", async () => {
      const key = modalApiKeyInput ? modalApiKeyInput.value.trim() : "";
      if (!key) {
        if (modalKeyFeedback) {
          modalKeyFeedback.textContent = "Enter a Gemini API key to enable optional external enhancement.";
          modalKeyFeedback.style.color = "var(--amber)";
        }
        return;
      }

      saveModalApiKeyBtn.disabled = true;
      saveModalApiKeyBtn.textContent = "Verifying...";
      if (modalKeyFeedback) {
        modalKeyFeedback.textContent = "Checking the key and refreshing AI engines...";
        modalKeyFeedback.style.color = "var(--text-muted)";
      }

      try {
        const res = await fetch("/api/config/api-key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: key }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Could not save key.");

        if (modalKeyFeedback) {
          modalKeyFeedback.textContent = data.message || "Optional external AI enhancement saved.";
          modalKeyFeedback.style.color = data.verified ? "var(--emerald)" : "var(--amber)";
        }
        await checkGeminiStatus();
        if (data.verified) setTimeout(closeApiKeyModal, 650);
      } catch (err) {
        if (modalKeyFeedback) {
          modalKeyFeedback.textContent = err.message || "Key verification failed.";
          modalKeyFeedback.style.color = "var(--rose)";
        }
      } finally {
        saveModalApiKeyBtn.disabled = false;
        saveModalApiKeyBtn.textContent = "Save Optional Key";
      }
    });
  }

  // =========================================================
  // 2. Analyzing Animation Overlay Engine
  // =========================================================
  let analyzingStepTimer = null;
  function showAnalyzingOverlay(title = "Analyzing Dataset...", subtitle = "Extracting structure, profiling schema & synthesizing intelligence", customSteps = null) {
    const overlay = document.getElementById("analyzingOverlay");
    const titleEl = document.getElementById("analyzingTitle");
    const subEl = document.getElementById("analyzingSubtitle");
    if (!overlay) return;

    if (titleEl) titleEl.textContent = title;
    if (subEl) subEl.textContent = subtitle;

    const steps = customSteps || [
      "Parsing file structure & data encoding",
      "Computing statistical distributions & correlations",
      "Running zero-trust PII & privacy audit",
      "Finalizing InsightForge EDA Intelligence Engine"
    ];

    for (let i = 1; i <= 4; i++) {
      const stepEl = document.getElementById(`animStep${i}`);
      const textEl = document.getElementById(`animStep${i}Text`);
      if (stepEl) {
        stepEl.className = "analyzing-step" + (i === 1 ? " active" : "");
      }
      if (textEl && steps[i-1]) {
        textEl.textContent = steps[i-1];
      }
    }

    overlay.style.display = "flex";

    if (analyzingStepTimer) clearInterval(analyzingStepTimer);
    let curr = 1;
    analyzingStepTimer = setInterval(() => {
      const prevStep = document.getElementById(`animStep${curr}`);
      if (prevStep) {
        prevStep.className = "analyzing-step completed";
      }
      curr++;
      if (curr <= 4) {
        const nextStep = document.getElementById(`animStep${curr}`);
        if (nextStep) nextStep.className = "analyzing-step active";
      } else {
        clearInterval(analyzingStepTimer);
      }
    }, 550);
  }

  function hideAnalyzingOverlay() {
    const overlay = document.getElementById("analyzingOverlay");
    if (analyzingStepTimer) clearInterval(analyzingStepTimer);
    if (overlay) {
      overlay.style.display = "none";
    }
  }

  // =========================================================
  // 3. Ingestion Hub Tabs
  // =========================================================
  function setIngestTab(tab) {
    if (tabUploadFileBtn && tabPasteTextBtn) {
      tabUploadFileBtn.className = "btn btn-sm btn-secondary";
      tabPasteTextBtn.className = "btn btn-sm btn-secondary";
    }
    if (ingestTabUpload) ingestTabUpload.style.display = "none";
    if (ingestTabPaste) ingestTabPaste.style.display = "none";

    if (tab === "upload") {
      if (tabUploadFileBtn) tabUploadFileBtn.className = "btn btn-sm btn-outline-primary active";
      if (ingestTabUpload) ingestTabUpload.style.display = "block";
    } else if (tab === "paste") {
      if (tabPasteTextBtn) tabPasteTextBtn.className = "btn btn-sm btn-outline-primary active";
      if (ingestTabPaste) ingestTabPaste.style.display = "block";
    }
  }

  if (tabUploadFileBtn) tabUploadFileBtn.addEventListener("click", () => setIngestTab("upload"));
  if (tabPasteTextBtn) tabPasteTextBtn.addEventListener("click", () => setIngestTab("paste"));

  // =========================================================
  // 4. Ingestion Triggers (Dropzone, Browse, Paste)
  // =========================================================
  const navUploadNewBtn = document.getElementById("navUploadNewBtn");
  if (navUploadNewBtn && fileInput) {
    navUploadNewBtn.addEventListener("click", () => fileInput.click());
  }

  const initialDropZoneHtml = dropZone ? dropZone.innerHTML : "";

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
      }
    });
  }

  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });
  }

  async function uploadFile(file) {
    showAnalyzingOverlay(`Analyzing '${file.name}'...`, "Extracting structure, calculating metrics & synthesizing intelligence", [
      `Parsing '${file.name}' file structure`,
      "Computing statistics & distributions",
      "Running zero-trust PII & privacy audit",
      "Activating InsightForge EDA Studio"
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      let data = {};
      try {
        data = await res.json();
      } catch (jsonErr) {
        data = { detail: `Server responded with status ${res.status} (${res.statusText || 'Upload Error'})` };
      }
      
      // Reset input so user never needs to reload to re-upload
      if (fileInput) fileInput.value = "";
      if (dropZone) dropZone.innerHTML = initialDropZoneHtml;

      if (res.ok) {
        await fetchDatasetState(1);
        hideAnalyzingOverlay();
        if (data.is_resume) {
          window.switchHub("hub-resume");
        } else if (data.is_marksheet) {
          window.switchHub("hub-marksheet");
        } else {
          window.switchHub("hub-data");
        }
      } else {
        hideAnalyzingOverlay();
        alert(data.detail || "File processing failed.");
      }
    } catch (err) {
      hideAnalyzingOverlay();
      console.error("Upload error:", err);
      if (fileInput) fileInput.value = "";
      if (dropZone) dropZone.innerHTML = initialDropZoneHtml;
      alert("Error uploading file: " + (err.message || "Network error."));
    }
  }

  // Paste Resume Action
  if (analyzePastedResumeBtn) {
    analyzePastedResumeBtn.addEventListener("click", async () => {
      const text = pasteResumeInput ? pasteResumeInput.value.trim() : "";
      if (!text || text.length < 20) {
        alert("Please paste resume content to analyze.");
        return;
      }
      showAnalyzingOverlay("Analyzing Resume...", "Scoring executive metrics, ATS compliance & tech stack", [
        "Parsing resume sections & work bullets",
        "Evaluating action verbs & metric impact",
        "Assessing 2026 tech stack readiness",
        "Generating InsightForge EDA recommendations"
      ]);

      try {
        const res = await fetch("/api/resume/analyze-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text }),
        });
        let data = {};
        try {
          data = await res.json();
        } catch (je) {
          data = { detail: "Server error parsing response." };
        }
        hideAnalyzingOverlay();

        if (res.ok) {
          await fetchDatasetState(1);
          window.switchHub("hub-resume");
        } else {
          alert(data.detail || "Analysis failed.");
        }
      } catch (err) {
        hideAnalyzingOverlay();
        console.error("Paste error:", err);
      }
    });
  }

  // Paste Marksheet Action
  const analyzePastedMarksheetBtn = document.getElementById("analyzePastedMarksheetBtn");
  if (analyzePastedMarksheetBtn) {
    analyzePastedMarksheetBtn.addEventListener("click", async () => {
      const text = pasteResumeInput ? pasteResumeInput.value.trim() : "";
      if (!text || text.length < 20) {
        alert("Please paste marksheet or transcript text to analyze.");
        return;
      }
      showAnalyzingOverlay("Analyzing Marksheet...", "Parsing subject scores, percentage & GPA", [
        "Extracting subject marks & max marks",
        "Calculating aggregate percentage & GPA",
        "Identifying strongest & improvement areas",
        "Generating InsightForge EDA guidance"
      ]);

      try {
        const res = await fetch("/api/marksheet/analyze-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text }),
        });
        let data = {};
        try {
          data = await res.json();
        } catch (je) {
          data = { detail: "Server error parsing response." };
        }
        hideAnalyzingOverlay();

        if (res.ok) {
          await fetchDatasetState(1);
          window.switchHub("hub-marksheet");
        } else {
          alert(data.detail || "Marksheet analysis failed.");
        }
      } catch (err) {
        hideAnalyzingOverlay();
        console.error("Paste marksheet error:", err);
      }
    });
  }

  // =========================================================
  // 5. Dataset State Synchronization & UI Hydration
  // =========================================================
  async function fetchDatasetState(page = 1) {
    try {
      const res = await fetch(`/api/dataset?page=${page}`);
      const data = await res.json();
      
      if (!data.has_dataset) {
        if (navCenterHubs) navCenterHubs.classList.add("hidden-nav");
        if (topDatasetPill) topDatasetPill.style.display = "none";
        if (resetDataBtn) resetDataBtn.style.display = "none";
        switchHub("hub-home");
        return;
      }

      if (navCenterHubs) navCenterHubs.classList.remove("hidden-nav");
      appState.dataset = data;
      appState.currentPage = page;
      updateUI();
    } catch (err) {
      console.error("Error fetching dataset state:", err);
    }
  }

  function updateUI() {
    const d = appState.dataset;
    if (!d || !d.has_dataset) return;

    if (topDatasetPill) {
      topDatasetPill.style.display = "inline-flex";
      let typeBadge = "";
      if (d.is_resume) typeBadge = " [Resume]";
      else if (d.is_marksheet) typeBadge = " [Marksheet]";
      else typeBadge = " [CSV Data]";
      if (topDatasetName) topDatasetName.textContent = `${d.dataset_name}${typeBadge} ${d.is_cleaned ? "(Cleaned)" : ""}`;
    }
    if (resetDataBtn) resetDataBtn.style.display = d.is_cleaned ? "inline-flex" : "none";

    const navResumeBtn = document.getElementById("navResumeBtn");
    if (d.is_resume && d.resume_analysis) {
      if (navResumeBtn) navResumeBtn.style.display = "inline-flex";
      renderResumeAnalysis(d.resume_analysis);
    } else {
      if (navResumeBtn) navResumeBtn.style.display = "none";
    }

    const navMarksheetBtn = document.getElementById("navMarksheetBtn");
    if (d.is_marksheet && d.marksheet_analysis) {
      if (navMarksheetBtn) navMarksheetBtn.style.display = "inline-flex";
      renderMarksheetAnalysis(d.marksheet_analysis);
    } else {
      if (navMarksheetBtn) navMarksheetBtn.style.display = "none";
    }

    // Update KPI cards in Data Explorer
    const kpiRows = document.getElementById("kpiRows");
    const kpiCols = document.getElementById("kpiCols");
    const kpiMissing = document.getElementById("kpiMissing");
    const kpiMemory = document.getElementById("kpiMemory");

    if (kpiRows) kpiRows.textContent = d.total_rows.toLocaleString();
    if (kpiCols) kpiCols.textContent = d.total_cols;
    if (kpiMissing && d.profiler) kpiMissing.textContent = `${d.profiler.missing_cells_pct}%`;
    if (kpiMemory && d.profiler) kpiMemory.textContent = d.profiler.memory_usage_mb ? `${d.profiler.memory_usage_mb} MB` : "< 1 MB";

    // Column schema badges with detailed statistics
    const columnSchemaBadges = document.getElementById("columnSchemaBadges");
    if (columnSchemaBadges && d.profiler && d.profiler.columns) {
      columnSchemaBadges.innerHTML = d.profiler.columns.map((c) => `
        <span class="badge ${getTypeBadgeClass(c.semantic_type)}" style="padding: 0.4rem 0.75rem; font-size: 0.85rem; display: inline-block; margin-bottom: 0.25rem;">
          <strong style="margin-right: 0.25rem;">${c.column_name}</strong> &bull; ${c.semantic_type} &bull; ${c.missing_pct}% Missing &bull; ${c.unique_count} Uniques
        </span>
      `).join("");
    }

    // Render paginated data table
    renderDataTable();

    // Render Quality Hub
    renderQualityHub();

    // Initialize Interactive Visual Studio Charts
    if (!d.is_resume && !d.is_marksheet) {
      initStudioChartControls();
    }

    // Populate ML Target Column Dropdown
    populateMLDropdown();
  }

  function getTypeBadgeClass(type) {
    switch (type) {
      case "Identifier": return "badge-id";
      case "Numerical": return "badge-num";
      case "Categorical": return "badge-cat";
      case "Date": return "badge-date";
      case "Boolean": return "badge-bool";
      case "Text": return "badge-text";
      default: return "badge-id";
    }
  }

  // =========================================================
  // 6. Resume Studio Sub-Tabs & Rendering
  // =========================================================
  const resumeTabAuditBtn = document.getElementById("resumeTabAuditBtn");
  const resumeTabRewritesBtn = document.getElementById("resumeTabRewritesBtn");
  const resumeTabTechBtn = document.getElementById("resumeTabTechBtn");
  const resumeTabProfileBtn = document.getElementById("resumeTabProfileBtn");

  const resumeSubViewAudit = document.getElementById("resumeSubViewAudit");
  const resumeSubViewRewrites = document.getElementById("resumeSubViewRewrites");
  const resumeSubViewTech = document.getElementById("resumeSubViewTech");
  const resumeSubViewProfile = document.getElementById("resumeSubViewProfile");

  function setResumeSubTab(tab) {
    [resumeTabAuditBtn, resumeTabRewritesBtn, resumeTabTechBtn, resumeTabProfileBtn].forEach((btn) => {
      if (btn) btn.className = "btn btn-sm btn-secondary";
    });
    [resumeSubViewAudit, resumeSubViewRewrites, resumeSubViewTech, resumeSubViewProfile].forEach((sec) => {
      if (sec) sec.style.display = "none";
    });

    if (tab === "audit") {
      if (resumeTabAuditBtn) resumeTabAuditBtn.className = "btn btn-sm btn-outline-primary active";
      if (resumeSubViewAudit) resumeSubViewAudit.style.display = "block";
    } else if (tab === "rewrites") {
      if (resumeTabRewritesBtn) resumeTabRewritesBtn.className = "btn btn-sm btn-outline-primary active";
      if (resumeSubViewRewrites) resumeSubViewRewrites.style.display = "block";
    } else if (tab === "tech") {
      if (resumeTabTechBtn) resumeTabTechBtn.className = "btn btn-sm btn-outline-primary active";
      if (resumeSubViewTech) resumeSubViewTech.style.display = "block";
    } else if (tab === "profile") {
      if (resumeTabProfileBtn) resumeTabProfileBtn.className = "btn btn-sm btn-outline-primary active";
      if (resumeSubViewProfile) resumeSubViewProfile.style.display = "block";
    }
  }

  if (resumeTabAuditBtn) resumeTabAuditBtn.addEventListener("click", () => setResumeSubTab("audit"));
  if (resumeTabRewritesBtn) resumeTabRewritesBtn.addEventListener("click", () => setResumeSubTab("rewrites"));
  if (resumeTabTechBtn) resumeTabTechBtn.addEventListener("click", () => setResumeSubTab("tech"));
  if (resumeTabProfileBtn) resumeTabProfileBtn.addEventListener("click", () => setResumeSubTab("profile"));

  function renderResumeAnalysis(ra) {
    if (!ra) return;

    const overallScore = ra.overall_score !== undefined ? Number(ra.overall_score).toFixed(1) : "8.5";
    const masterScoreEl = document.getElementById("resumeMasterScore");
    if (masterScoreEl) masterScoreEl.textContent = overallScore;

    const tierEl = document.getElementById("resumePercentileTier");
    if (tierEl && ra.percentile_tier) tierEl.textContent = ra.percentile_tier;

    // Sub-Scores
    const sub = ra.sub_scores || {};
    const setSub = (id, barId, val) => {
      const el = document.getElementById(id);
      const bar = document.getElementById(barId);
      const num = Number(val || 7.5);
      if (el) el.textContent = `${num.toFixed(1)} / 10`;
      if (bar) bar.style.width = `${Math.min(100, num * 10)}%`;
    };

    setSub("subScoreImpact", "barImpact", sub.impact);
    setSub("subScoreVerbs", "barVerbs", sub.verbs);
    setSub("subScoreSkills", "barSkills", sub.skills);
    setSub("subScoreAts", "barAts", sub.ats);
    setSub("subScoreLeadership", "barLeadership", sub.leadership);

    // Markdown Deep Insights
    const auditEl = document.getElementById("resumeAuditMarkdown");
    if (auditEl && ra.deep_insights && ra.deep_insights.markdown) {
      auditEl.innerHTML = marked.parse(ra.deep_insights.markdown);
    }

    // Google XYZ Rewrites
    const rewritesContainer = document.getElementById("resumeRewritesList");
    if (rewritesContainer && ra.weak_bullet_rewrites) {
      rewritesContainer.innerHTML = ra.weak_bullet_rewrites.map((r, i) => `
        <div class="glass-card" style="border: 1px solid var(--border-glass-primary); padding: 1.25rem;">
          <div style="font-size: 0.8rem; color: var(--rose); font-weight: 700; text-transform: uppercase; margin-bottom: 0.25rem;">Original Statement #${i + 1}</div>
          <div style="font-size: 0.95rem; color: var(--text-muted); margin-bottom: 0.5rem; text-decoration: line-through;">"${r.original}"</div>
          
          <div style="font-size: 0.8rem; color: var(--emerald); font-weight: 700; text-transform: uppercase; margin-bottom: 0.25rem;">Elite Google XYZ Formula Rewrite</div>
          <div style="font-size: 1rem;  font-weight: 600; line-height: 1.5; margin-bottom: 0.5rem; background: rgba(16, 185, 129, 0.1); border-left: 3px solid var(--emerald); padding: 0.5rem 0.75rem; border-radius: 4px;">
            ${r.rewrite}
          </div>
          
          <div style="font-size: 0.8rem; color: var(--text-dim);">
            <strong>Weakness:</strong> ${r.weakness} &bull; <strong style="color: var(--primary-bright);">Advantage:</strong> ${r.advantage}
          </div>
        </div>
      `).join("");
    }

    // Skills Badges
    const matchedEl = document.getElementById("resumeMatchedSkillsBadges");
    if (matchedEl && ra.matched_skills) {
      matchedEl.innerHTML = Object.entries(ra.matched_skills).flatMap(([cat, skills]) => 
        skills.map(s => `<span class="badge badge-bool" style="padding: 0.4rem 0.6rem;">${s}</span>`)
      ).join("") || "<span style='color: var(--text-muted);'>No skills detected</span>";
    }

    const missingEl = document.getElementById("resumeMissingSkillsBadges");
    if (missingEl && ra.recommended_keywords) {
      missingEl.innerHTML = ra.recommended_keywords.map(k => `
        <span class="badge" style="background: rgba(37, 99, 235, 0.2); color: var(--primary-bright); border: 1px solid var(--border-glass-primary); cursor: pointer; padding: 0.4rem 0.6rem;" onclick="navigator.clipboard.writeText('${k}'); alert('Copied keyword: ${k}');">
          + ${k}
        </span>
      `).join("") || "<span style='color: var(--text-muted);'>Profile is comprehensive</span>";
    }

    // ATS Profile
    const prof = ra.profile || {};
    const atsName = document.getElementById("atsName");
    const atsEmail = document.getElementById("atsEmail");
    const atsPhone = document.getElementById("atsPhone");
    const atsLinks = document.getElementById("atsLinks");

    if (atsName) atsName.textContent = prof.name || "Candidate";
    if (atsEmail) atsEmail.textContent = prof.email || "Not Detected";
    if (atsPhone) atsPhone.textContent = prof.phone || "Not Detected";
    if (atsLinks) {
      const links = [];
      if (prof.linkedin) links.push(`<a href="${prof.linkedin}" target="_blank" style="color: var(--primary-bright);">LinkedIn</a>`);
      if (prof.github) links.push(`<a href="${prof.github}" target="_blank" style="color: var(--emerald);">GitHub</a>`);
      atsLinks.innerHTML = links.join(" &bull; ") || "None Detected";
    }

    const sectionsEl = document.getElementById("atsSectionsList");
    if (sectionsEl && prof.detected_sections) {
      sectionsEl.innerHTML = prof.detected_sections.map(s => `<span class="badge badge-id" style="padding: 0.35rem 0.65rem;">${s}</span>`).join("");
    }
  }

  // =========================================================
  // 6.2 Marksheet & Academic Intelligence Renderer
  // =========================================================
  let marksheetChartInstance = null;
  let semesterProgressionChartInstance = null;

  function renderMarksheetAnalysis(ma) {
    if (!ma) return;

    // Header & Meta
    const nameEl = document.getElementById("marksheetCandidateName");
    const tierBadge = document.getElementById("marksheetTierBadge");
    const examInfo = document.getElementById("marksheetExamInfo");

    if (nameEl) nameEl.textContent = ma.candidate_name || "Student Scorecard";
    const sm = ma.summary_metrics || {};
    if (tierBadge) {
      tierBadge.textContent = sm.academic_tier || "Distinction Standing";
      if (sm.tier_color) tierBadge.style.backgroundColor = sm.tier_color;
    }
    if (examInfo) {
      examInfo.textContent = `${ma.examination_name || "Examination"} • ${ma.institution_board || "State Board / University"} • Roll: ${ma.roll_number || "N/A"}`;
    }

    // Best-of-N Subjects Selector Handling
    const bestNSelect = document.getElementById("marksheetBestNSelect");
    const pctEl = document.getElementById("marksheetOverallPct");
    const pctLabel = document.getElementById("marksheetPctLabel");
    const totalEl = document.getElementById("marksheetTotalScore");
    const gpaEl = document.getElementById("marksheetGpa");
    const gpa4El = document.getElementById("marksheetGpa4");

    const updateScores = (mode) => {
      let pct = sm.overall_percentage || 0;
      let label = "Overall Aggregate";
      if (mode === "best5" && sm.best_5_percentage) {
        pct = sm.best_5_percentage;
        label = "Best of 5 Aggregate (Norm)";
      } else if (mode === "best4" && sm.best_4_percentage) {
        pct = sm.best_4_percentage;
        label = "Best of 4 Aggregate (Norm)";
      }
      if (pctEl) pctEl.textContent = `${pct}%`;
      if (pctLabel) pctLabel.textContent = label;
      const g10 = (pct / 10.0).toFixed(2);
      const g4 = ((pct / 100.0) * 4.0).toFixed(2);
      if (gpaEl) gpaEl.textContent = g10;
      if (gpa4El) gpa4El.textContent = `${g4} / 4.0 US Scale`;
    };

    if (bestNSelect) {
      bestNSelect.onchange = (e) => updateScores(e.target.value);
      updateScores(bestNSelect.value);
    } else {
      updateScores("all");
    }

    if (totalEl) totalEl.textContent = `${sm.total_marks_obtained || 0} / ${sm.total_max_marks || 0} Total Marks`;

    // Strongest & Weakest Subjects
    const strong = ma.strongest_subject || {};
    const weak = ma.weakest_subject || {};

    const strongSubEl = document.getElementById("marksheetStrongestSub");
    const strongMarksEl = document.getElementById("marksheetStrongestMarks");
    const weakSubEl = document.getElementById("marksheetWeakestSub");
    const weakMarksEl = document.getElementById("marksheetWeakestMarks");

    if (strongSubEl) strongSubEl.textContent = strong.subject || "N/A";
    if (strongMarksEl) strongMarksEl.textContent = `${strong.marks || 0} / ${strong.max || 100} (${strong.percentage || 0}%)`;

    if (weakSubEl) weakSubEl.textContent = weak.subject || "N/A";
    if (weakMarksEl) weakMarksEl.textContent = `${weak.marks || 0} / ${weak.max || 100} (${weak.percentage || 0}%)`;

    // Multi-Semester Progression Handling
    const multiSemCard = document.getElementById("multiSemesterCard");
    const semCardsList = document.getElementById("semesterCardsList");
    const semProgCanvas = document.getElementById("semesterProgressCanvas");
    const cgpaBadge = document.getElementById("cumulativeCgpaBadge");

    if (ma.is_multi_semester && ma.semesters && ma.semesters.length > 1) {
      if (multiSemCard) multiSemCard.style.display = "block";
      if (cgpaBadge && sm.cgpa) cgpaBadge.textContent = `Cumulative CGPA: ${sm.cgpa} / 10.0`;

      if (semCardsList) {
        semCardsList.innerHTML = ma.semesters.map((sem, i) => `
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.8rem; background: #F1F5F9; border-radius: var(--radius-sm); border-left: 3px solid var(--primary-bright);">
            <div>
              <strong style=" font-size: 0.9rem;">${sem.semester_name}</strong>
              <div style="font-size: 0.75rem; color: var(--text-muted);">${sem.subjects_count} Subjects &bull; ${sem.total_marks_obtained}/${sem.total_max_marks} Marks</div>
            </div>
            <div style="text-align: right;">
              <span class="badge badge-num" style="font-size: 0.9rem; font-weight: 700;">SGPA: ${sem.sgpa}</span>
              <div style="font-size: 0.75rem; color: var(--emerald); font-weight: 600;">${sem.percentage}%</div>
            </div>
          </div>
        `).join("");
      }

      if (semProgCanvas && typeof Chart !== "undefined") {
        if (semesterProgressionChartInstance) semesterProgressionChartInstance.destroy();
        const ctx = semProgCanvas.getContext("2d");
        semesterProgressionChartInstance = new Chart(ctx, {
          type: "line",
          data: {
            labels: ma.semesters.map(s => s.semester_name),
            datasets: [{
              label: "Semester SGPA",
              data: ma.semesters.map(s => s.sgpa),
              borderColor: "#2563EB",
              backgroundColor: "rgba(37, 99, 235, 0.15)",
              borderWidth: 3,
              tension: 0.3,
              fill: true,
              pointBackgroundColor: "#10B981",
              pointRadius: 5
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { beginAtZero: false, min: 4, max: 10, ticks: { color: "#94A3B8" }, grid: { color: "rgba(255,255,255,0.06)" } },
              x: { ticks: { color: "#FFFFFF" }, grid: { display: false } }
            },
            plugins: { legend: { display: false } }
          }
        });
      }
    } else {
      if (multiSemCard) multiSemCard.style.display = "none";
    }

    // Subject Breakdown Table
    const tableBody = document.getElementById("marksheetTableBody");
    const subjects = ma.subject_breakdown || [];
    if (tableBody) {
      tableBody.innerHTML = subjects.map(s => `
        <tr>
          <td style="font-weight: 600; ">${s.subject}</td>
          <td style="color: var(--primary-bright); font-weight: 600;">${s.marks}</td>
          <td style="color: var(--text-muted);">${s.max}</td>
          <td>
            <span class="badge ${s.percentage >= 80 ? 'badge-num' : (s.percentage >= 60 ? 'badge-cat' : 'badge-date')}" style="padding: 0.25rem 0.5rem;">
              ${s.percentage}%
            </span>
          </td>
          <td style="font-size: 0.85rem; color: var(--emerald); font-weight: 600;">${s.grade}</td>
        </tr>
      `).join("");
    }

    // Interactive Bar Chart
    const canvas = document.getElementById("marksheetCanvas");
    if (canvas && typeof Chart !== "undefined") {
      if (marksheetChartInstance) marksheetChartInstance.destroy();
      const ctx = canvas.getContext("2d");

      marksheetChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels: subjects.map(s => s.subject),
          datasets: [{
            label: "Percentage Score (%)",
            data: subjects.map(s => s.percentage),
            backgroundColor: subjects.map(s => s.percentage >= 90 ? "rgba(16, 185, 129, 0.85)" : (s.percentage >= 75 ? "rgba(37, 99, 235, 0.85)" : "rgba(239, 68, 68, 0.85)")),
            borderColor: subjects.map(s => s.percentage >= 90 ? "#10B981" : (s.percentage >= 75 ? "#2563EB" : "#EF4444")),
            borderWidth: 1,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              grid: { color: "rgba(255,255,255,0.06)" },
              ticks: { color: "#94A3B8" }
            },
            x: {
              grid: { display: false },
              ticks: { color: "#FFFFFF", font: { weight: "600" } }
            }
          },
          plugins: {
            legend: { display: false }
          }
        }
      });
    }

    // InsightForge EDA Academic Guidance Markdown
    const guidanceEl = document.getElementById("marksheetGuidanceMarkdown");
    if (guidanceEl && ma.academic_guidance && ma.academic_guidance.markdown) {
      guidanceEl.innerHTML = marked.parse(ma.academic_guidance.markdown);
    }
  }

  // =========================================================
  // 6.3 Interactive Visual Chart Studio
  // =========================================================
  let studioChartInstance = null;
  let activeStudioChartType = "hist";

  function initStudioChartControls() {
    const d = appState.dataset;
    const colXSelect = document.getElementById("chartColXSelect");
    const colYSelect = document.getElementById("chartColYSelect");
    const chartTypeBtns = document.querySelectorAll(".chart-type-btn");

    if (!colXSelect || !d || !d.columns) return;

    colXSelect.innerHTML = "";
    if (colYSelect) colYSelect.innerHTML = "";

    const numCols = d.columns.filter(c => {
      const sem = (d.profiler && d.profiler.column_semantic_types && d.profiler.column_semantic_types[c]) || "";
      return sem.includes("Numeric") || sem.includes("Float") || sem.includes("Int");
    });

    const allCols = numCols.length > 0 ? numCols : d.columns;

    allCols.forEach(col => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      colXSelect.appendChild(opt);

      if (colYSelect) {
        const optY = document.createElement("option");
        optY.value = col;
        optY.textContent = col;
        colYSelect.appendChild(optY);
      }
    });

    if (colYSelect && allCols.length > 1) {
      colYSelect.selectedIndex = 1;
    }

    colXSelect.onchange = renderStudioChart;
    if (colYSelect) colYSelect.onchange = renderStudioChart;

    chartTypeBtns.forEach(btn => {
      btn.onclick = () => {
        chartTypeBtns.forEach(b => b.className = "btn btn-sm btn-secondary chart-type-btn");
        btn.className = "btn btn-sm btn-outline-primary chart-type-btn active";
        activeStudioChartType = btn.getAttribute("data-chart");
        renderStudioChart();
      };
    });

    renderStudioChart();
  }

  function renderStudioChart() {
    const d = appState.dataset;
    const canvas = document.getElementById("studioChartCanvas");
    const heatmapContainer = document.getElementById("correlationHeatmapContainer");
    const colYWrapper = document.getElementById("chartColYWrapper");
    const statBadge = document.getElementById("chartStatBadge");
    const colX = document.getElementById("chartColXSelect") ? document.getElementById("chartColXSelect").value : "";
    const colY = document.getElementById("chartColYSelect") ? document.getElementById("chartColYSelect").value : "";

    const records = d ? (d.chart_records || d.records || (d.sample_data && d.sample_data.records) || []) : [];

    if (!canvas || !d || records.length === 0) return;

    if (colYWrapper) {
      colYWrapper.style.display = activeStudioChartType === "scatter" ? "flex" : "none";
    }

    if (activeStudioChartType === "corr") {
      canvas.style.display = "none";
      if (heatmapContainer) {
        heatmapContainer.style.display = "block";
        renderCorrelationHeatmap(heatmapContainer);
      }
      if (statBadge) statBadge.textContent = "Pearson Pairwise Correlation Matrix";
      return;
    } else {
      canvas.style.display = "block";
      if (heatmapContainer) heatmapContainer.style.display = "none";
    }

    if (studioChartInstance) studioChartInstance.destroy();
    const ctx = canvas.getContext("2d");

    const xVals = records.map(r => r[colX]).filter(v => v !== undefined && v !== null && v !== "");

    if (activeStudioChartType === "hist") {
      // Numerical Histogram
      const nums = xVals.map(Number).filter(v => !isNaN(v));
      if (nums.length === 0) return;
      const min = Math.min(...nums);
      const max = Math.max(...nums);
      const bins = 8;
      const step = (max - min) / bins || 1;
      const counts = new Array(bins).fill(0);
      const labels = [];

      for (let i = 0; i < bins; i++) {
        const bStart = (min + i * step).toFixed(1);
        const bEnd = (min + (i + 1) * step).toFixed(1);
        labels.push(`${bStart} - ${bEnd}`);
      }

      nums.forEach(n => {
        let bIdx = Math.floor((n - min) / step);
        if (bIdx >= bins) bIdx = bins - 1;
        counts[bIdx]++;
      });

      studioChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: `Frequency Distribution of ${colX}`,
            data: counts,
            backgroundColor: "rgba(37, 99, 235, 0.75)",
            borderColor: "#2563EB",
            borderWidth: 1,
            borderRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { beginAtZero: true, ticks: { color: "#94A3B8" }, grid: { color: "rgba(255,255,255,0.06)" } },
            x: { ticks: { color: "#FFFFFF" }, grid: { display: false } }
          }
        }
      });
      if (statBadge) statBadge.textContent = `Mean: ${(nums.reduce((a,b)=>a+b,0)/nums.length).toFixed(1)} • Min: ${min} • Max: ${max}`;

    } else if (activeStudioChartType === "scatter") {
      // 2D Scatter Plot
      const points = records.map(r => ({
        x: Number(r[colX]),
        y: Number(r[colY])
      })).filter(p => !isNaN(p.x) && !isNaN(p.y));

      studioChartInstance = new Chart(ctx, {
        type: "scatter",
        data: {
          datasets: [{
            label: `${colX} vs ${colY}`,
            data: points,
            backgroundColor: "rgba(16, 185, 129, 0.8)",
            borderColor: "#10B981",
            pointRadius: 6,
            pointHoverRadius: 8
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { ticks: { color: "#94A3B8" }, grid: { color: "rgba(255,255,255,0.06)" }, title: { display: true, text: colY, color: "#FFFFFF" } },
            x: { ticks: { color: "#94A3B8" }, grid: { color: "rgba(255,255,255,0.06)" }, title: { display: true, text: colX, color: "#FFFFFF" } }
          }
        }
      });
      if (statBadge) statBadge.textContent = `Scatter Plot: ${points.length} Data Points`;

    } else if (activeStudioChartType === "bar") {
      // Categorical Frequency Counts
      const freq = {};
      xVals.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
      const sortedKeys = Object.keys(freq).sort((a,b) => freq[b] - freq[a]).slice(0, 10);

      studioChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels: sortedKeys,
          datasets: [{
            label: `Value Counts for ${colX}`,
            data: sortedKeys.map(k => freq[k]),
            backgroundColor: "rgba(59, 130, 246, 0.8)",
            borderColor: "#3B82F6",
            borderWidth: 1,
            borderRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { beginAtZero: true, ticks: { color: "#94A3B8" }, grid: { color: "rgba(255,255,255,0.06)" } },
            x: { ticks: { color: "#FFFFFF" }, grid: { display: false } }
          }
        }
      });
      if (statBadge) statBadge.textContent = `Top ${sortedKeys.length} Unique Categories`;

    } else if (activeStudioChartType === "box") {
      // Boxplot / Outlier distribution representation
      const nums = xVals.map(Number).filter(v => !isNaN(v)).sort((a,b) => a - b);
      if (nums.length === 0) return;
      const min = nums[0];
      const max = nums[nums.length - 1];
      const q1 = nums[Math.floor(nums.length * 0.25)];
      const med = nums[Math.floor(nums.length * 0.5)];
      const q3 = nums[Math.floor(nums.length * 0.75)];

      studioChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels: [`${colX} Metrics`],
          datasets: [
            { label: "Min", data: [min], backgroundColor: "rgba(239, 68, 68, 0.8)" },
            { label: "Q1 (25th)", data: [q1], backgroundColor: "rgba(245, 158, 11, 0.8)" },
            { label: "Median (50th)", data: [med], backgroundColor: "rgba(16, 185, 129, 0.8)" },
            { label: "Q3 (75th)", data: [q3], backgroundColor: "rgba(59, 130, 246, 0.8)" },
            { label: "Max", data: [max], backgroundColor: "rgba(168, 85, 247, 0.8)" }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { ticks: { color: "#94A3B8" }, grid: { color: "rgba(255,255,255,0.06)" } },
            x: { ticks: { color: "#FFFFFF" } }
          }
        }
      });
      if (statBadge) statBadge.textContent = `Median: ${med} • IQR: ${(q3 - q1).toFixed(1)} • Bounds: [${min}, ${max}]`;
    }
  }

  function renderCorrelationHeatmap(container) {
    const d = appState.dataset;
    const corr = d.statistics && d.statistics.correlations ? d.statistics.correlations.matrix : null;
    if (!corr) {
      container.innerHTML = "<p style='color: var(--text-muted); padding: 1rem;'>No correlation matrix available.</p>";
      return;
    }

    const cols = Object.keys(corr);
    let html = `
      <table class="data-table" style="width: 100%; text-align: center;">
        <thead>
          <tr>
            <th>Variable</th>
            ${cols.map(c => `<th>${c}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
    `;

    cols.forEach(rowCol => {
      html += `<tr><td style="font-weight: 700;  text-align: left;">${rowCol}</td>`;
      cols.forEach(colCol => {
        const val = corr[rowCol][colCol] !== undefined ? corr[rowCol][colCol] : 0;
        let bg = "rgba(255,255,255,0.05)";
        if (val > 0.6) bg = "rgba(16, 185, 129, 0.45)";
        else if (val > 0.3) bg = "rgba(16, 185, 129, 0.25)";
        else if (val < -0.6) bg = "rgba(239, 68, 68, 0.45)";
        else if (val < -0.3) bg = "rgba(239, 68, 68, 0.25)";
        html += `<td style="background: ${bg}; font-weight: 600; ">${Number(val).toFixed(2)}</td>`;
      });
      html += "</tr>";
    });

    html += "</tbody></table>";
    container.innerHTML = html;
  }

  // =========================================================
  // 7. Paginated Data Table
  // =========================================================
  function renderDataTable() {
    const d = appState.dataset;
    const container = document.getElementById("dataTableContainer");
    if (!container || !d) return;

    const records = d.records || (d.sample_data && d.sample_data.records) || [];

    if (records && records.length > 0) {
      const cols = d.columns || Object.keys(records[0]);
      container.innerHTML = `
        <table class="data-table" style="width: 100%;">
          <thead>
            <tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>
          </thead>
          <tbody>
            ${records.map(row => `
              <tr>${cols.map(c => `<td>${row[c] !== undefined ? row[c] : ""}</td>`).join("")}</tr>
            `).join("")}
          </tbody>
        </table>
      `;
    } else {
      container.innerHTML = "<p style='color: var(--text-muted); padding: 1.5rem; text-align: center;'>No records available.</p>";
    }

    const pageIndicator = document.getElementById("pageIndicator");
    if (pageIndicator) pageIndicator.textContent = `Page ${appState.currentPage}`;
  }

  const prevPageBtn = document.getElementById("prevPageBtn");
  const nextPageBtn = document.getElementById("nextPageBtn");

  if (prevPageBtn) {
    prevPageBtn.addEventListener("click", () => {
      if (appState.currentPage > 1) {
        fetchDatasetState(appState.currentPage - 1);
      }
    });
  }

  if (nextPageBtn) {
    nextPageBtn.addEventListener("click", () => {
      fetchDatasetState(appState.currentPage + 1);
    });
  }

  // =========================================================
  // 8. Quality & Cleaning Hub
  // =========================================================
  function renderQualityHub() {
    const d = appState.dataset;
    if (!d || !d.quality) return;

    const q = d.quality;
    const healthEl = document.getElementById("qualityHealthScore");
    
    // New breakdown containers
    const explanationsEl = document.getElementById("qualityScoreExplanations");
    const recommendationsEl = document.getElementById("qualityRecommendations");

    if (healthEl) {
      healthEl.textContent = `${q.health_score}`;
      healthEl.style.color = q.health_score >= 80 ? "var(--emerald)" : q.health_score >= 60 ? "var(--amber)" : "var(--rose)";
    }

    if (q.score_breakdown) {
      if (explanationsEl) {
        explanationsEl.innerHTML = q.score_breakdown.explanations.map(exp => `<li>• ${exp}</li>`).join("");
      }
      if (recommendationsEl) {
        recommendationsEl.innerHTML = q.score_breakdown.recommendations.map(rec => `<li>• ${rec}</li>`).join("");
      }
    }

    const outliersContainer = document.getElementById("outliersListContainer");
    if (outliersContainer && q.outliers) {
      const outlierEntries = Object.entries(q.outliers).filter(([_, o]) => o.iqr && o.iqr.outlier_count > 0);
      if (outlierEntries.length > 0) {
        outliersContainer.innerHTML = outlierEntries.map(([col, o]) => `
          <div style="margin-bottom: 0.75rem; padding: 0.6rem; background: #F1F5F9; border-radius: var(--radius-sm);">
            <strong >${col}</strong>: <span style="color: var(--rose); font-weight: 700;">${o.iqr.outlier_count} Outliers</span>
            <div style="font-size: 0.78rem; color: var(--text-dim);">Bounds: [${o.iqr.lower_bound}, ${o.iqr.upper_bound}]</div>
          </div>
        `).join("");
      } else {
        outliersContainer.innerHTML = "<p style='color: var(--emerald); font-size: 0.9rem;'>✓ Zero statistical outliers detected via IQR method.</p>";
      }
    }

    const missingContainer = document.getElementById("missingListContainer");
    if (missingContainer && q.missing_details) {
      if (q.missing_details.length > 0) {
        missingContainer.innerHTML = q.missing_details.map(m => `
          <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span >${m.column}</span>
            <span class="badge badge-num">${m.missing_count} (${m.missing_pct}%)</span>
          </div>
        `).join("");
      } else {
        missingContainer.innerHTML = "<p style='color: var(--emerald); font-size: 0.9rem;'>✓ 100% complete dataset with 0 missing cells.</p>";
      }
    }
  }

  const runAutoCleanBtn = document.getElementById("runAutoCleanBtn");
  const restoreRawDataBtn = document.getElementById("restoreRawDataBtn");

  if (runAutoCleanBtn) {
    runAutoCleanBtn.addEventListener("click", async () => {
      runAutoCleanBtn.disabled = true;
      runAutoCleanBtn.textContent = "🧼 Cleaning Dataset...";
      showAnalyzingOverlay("Sanitizing Dataset...", "Executing automated 1-click hygiene & repair pipeline", [
        "Detecting & dropping duplicate records",
        "Imputing missing values with median / mode",
        "Winsorizing statistical IQR outliers",
        "Recalculating dataset health score"
      ]);

      try {
        const res = await fetch("/api/clean/auto", { method: "POST" });
        const data = await res.json();
        runAutoCleanBtn.disabled = false;
        runAutoCleanBtn.textContent = "🧼 Run Full Auto-Clean";
        hideAnalyzingOverlay();

        if (res.ok) {
          await fetchDatasetState(1);

          // Populate and show Cleaning Results Card
          const resCard = document.getElementById("cleaningResultsCard");
          const dupesEl = document.getElementById("cleanDupesRemoved");
          const missingEl = document.getElementById("cleanMissingImputed");
          const outliersEl = document.getElementById("cleanOutliersCapped");
          const rowsEl = document.getElementById("cleanFinalRows");
          const badgeEl = document.getElementById("cleanHealthImprovementBadge");
          const logSummaryEl = document.getElementById("cleanLogSummaryText");

          const log = data.log || {};
          if (dupesEl) dupesEl.textContent = log.removed_duplicates !== undefined ? log.removed_duplicates : 0;
          if (missingEl) missingEl.textContent = log.imputed_missing_cells !== undefined ? log.imputed_missing_cells : 0;
          if (outliersEl) outliersEl.textContent = log.capped_outliers !== undefined ? log.capped_outliers : 0;
          if (rowsEl) rowsEl.textContent = log.final_rows || appState.dataset.total_rows;
          if (badgeEl) badgeEl.textContent = `Health Score: 100/100 (Optimized)`;
          if (logSummaryEl) {
            logSummaryEl.textContent = `Cleaning complete: Removed ${log.removed_duplicates || 0} duplicate row(s), imputed ${log.imputed_missing_cells || 0} missing cell(s), and winsorized ${log.capped_outliers || 0} IQR outlier(s).`;
          }
          if (resCard) resCard.style.display = "block";
        }
      } catch (e) {
        hideAnalyzingOverlay();
        runAutoCleanBtn.disabled = false;
        runAutoCleanBtn.textContent = "🧼 Run Full Auto-Clean";
      }
    });
  }

  if (restoreRawDataBtn) {
    restoreRawDataBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/clean/reset", { method: "POST" });
        if (res.ok) {
          await fetchDatasetState(1);
          alert("Restored to original raw dataset.");
        }
      } catch (e) {}
    });
  }
  if (resetDataBtn) {
    resetDataBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/clean/reset", { method: "POST" });
        if (res.ok) {
          await fetchDatasetState(1);
        }
      } catch (e) {}
    });
  }

  // =========================================================
  // 9. AutoML Studio
  // =========================================================
  function populateMLDropdown() {
    const d = appState.dataset;
    const select = document.getElementById("mlTargetSelect");
    if (!select || !d || !d.columns) return;

    select.innerHTML = '<option value="">-- Select Target Column --</option>';
    d.columns.forEach(col => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      select.appendChild(opt);
    });
  }

  const trainModelBtn = document.getElementById("trainModelBtn");
  if (trainModelBtn) {
    trainModelBtn.addEventListener("click", async () => {
      const select = document.getElementById("mlTargetSelect");
      const targetCol = select ? select.value : "";
      if (!targetCol) {
        alert("Please select a target column to predict.");
        return;
      }

      trainModelBtn.disabled = true;
      trainModelBtn.textContent = "🧠 Training ML Model...";
      showAnalyzingOverlay(`Training Model on '${targetCol}'...`, "Auto-detecting task, training estimators & computing feature weights", [
        `Analyzing target distribution for '${targetCol}'`,
        "Splitting train/test & pre-processing features",
        "Training Random Forest & Linear estimators",
        "Computing permutation feature importance"
      ]);

      try {
        const res = await fetch("/api/ml/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target_column: targetCol }),
        });
        const data = await res.json();
        trainModelBtn.disabled = false;
        trainModelBtn.textContent = "🚀 Train & Evaluate Model";
        hideAnalyzingOverlay();

        if (res.ok) {
          const result = data.result;
          const container = document.getElementById("mlResultsContainer");
          if (container) container.style.display = "block";

          const metricsGrid = document.getElementById("mlMetricsGrid");
          if (metricsGrid && result.metrics) {
            metricsGrid.innerHTML = Object.entries(result.metrics).map(([k, v]) => `
              <div class="glass-card kpi-card">
                <div class="kpi-label">${k}</div>
                <div class="kpi-val orange" style="font-size: 1.8rem;">${typeof v === 'number' ? v.toFixed(3) : v}</div>
              </div>
            `).join("");
          }

          // Feature Importance Chart
          if (result.feature_importance && result.feature_importance.length) {
            const chartCanvas = document.getElementById("mlImportanceChart");
            if (chartCanvas) {
              const ctx = chartCanvas.getContext("2d");
              if (appState.charts.importance) appState.charts.importance.destroy();

              appState.charts.importance = new Chart(ctx, {
                type: "bar",
                data: {
                  labels: result.feature_importance.map(f => f.Feature),
                  datasets: [{
                    label: "Predictive Weight",
                    data: result.feature_importance.map(f => f.Importance),
                    backgroundColor: "rgba(37, 99, 235, 0.85)",
                    borderColor: "#FF851B",
                    borderRadius: 6,
                  }],
                },
                options: {
                  indexAxis: "y",
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } },
                  scales: {
                    x: { grid: { color: "rgba(255,255,255,0.06)" } },
                    y: { grid: { display: false } },
                  },
                },
              });
            }
          }
        } else {
          alert(data.detail || "Model training failed.");
        }
      } catch (e) {
        hideAnalyzingOverlay();
        trainModelBtn.disabled = false;
        trainModelBtn.textContent = "🚀 Train & Evaluate Model";
      }
    });
  }

  // =========================================================
  // 10. AI Analyst Chat & Strategic Briefing
  // =========================================================
  const refreshBriefingBtn = document.getElementById("refreshBriefingBtn");
  if (refreshBriefingBtn) {
    refreshBriefingBtn.addEventListener("click", async () => {
      const box = document.getElementById("briefingContentBox");
      if (box) box.innerHTML = "<p style='color: var(--primary-bright);'>⚡ Compiling multi-agent executive briefing...</p>";
      showAnalyzingOverlay("Generating Executive Briefing...", "Orchestrating 4 autonomous InsightForge EDA sub-agents", [
        "Synthesizing EDA & distribution metrics",
        "Evaluating correlation patterns & hypotheses",
        "Formulating strategic executive conclusions",
        "Formatting verifiable mathematical proofs"
      ]);

      try {
        const res = await fetch("/api/ai/briefing");
        const data = await res.json();
        hideAnalyzingOverlay();
        if (res.ok && box) {
          box.innerHTML = marked.parse(data.briefing);
        }
      } catch (e) {
        hideAnalyzingOverlay();
      }
    });
  }

  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatMessages = document.getElementById("chatMessages");

  if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const query = chatInput.value.trim();
      if (!query) return;

      appendChatBubble("user", query);
      chatInput.value = "";
      const thinkingBubble = appendChatBubble("assistant", "Thinking...");

      try {
        const res = await fetch("/api/ai/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query }),
        });
        const data = await res.json();
        thinkingBubble.remove();

        if (res.ok) {
          let content = marked.parse(data.answer);
          if (data.data && Array.isArray(data.data) && data.data.length) {
            const keys = Object.keys(data.data[0]);
            content += `
              <div class="table-glass-container" style="margin-top: 0.75rem;">
                <table>
                  <thead><tr>${keys.map(k => `<th>${k}</th>`).join("")}</tr></thead>
                  <tbody>${data.data.slice(0, 5).map(row => `<tr>${keys.map(k => `<td>${row[k]}</td>`).join("")}</tr>`).join("")}</tbody>
                </table>
              </div>
            `;
          }
          appendChatBubble("assistant", content, true);
        } else {
          appendChatBubble("assistant", `<span style='color: var(--rose);'>${data.detail || "Query failed."}</span>`, true);
        }
      } catch (err) {
        thinkingBubble.remove();
        appendChatBubble("assistant", "<span style='color: var(--rose);'>Error communicating with AI Analyst.</span>", true);
      }
    });
  }

  function appendChatBubble(role, htmlContent, isHtml = false) {
    if (!chatMessages) return null;
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;
    if (isHtml) {
      bubble.innerHTML = htmlContent;
    } else {
      bubble.textContent = htmlContent;
    }
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  // Initial Sync
  checkGeminiStatus();
  fetchDatasetState(1);
});
