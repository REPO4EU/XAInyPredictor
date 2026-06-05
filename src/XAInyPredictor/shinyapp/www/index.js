$(() => {
  let currentTab = "data_input";  // default

  const setUseCaseLoading = (visible, title, subtitle) => {
    if (title) $("#loading-overlay-title").text(title);
    if (subtitle) $("#loading-overlay-subtitle").text(subtitle);
    $("#use-case-loading-overlay").toggleClass("is-visible", Boolean(visible));
  };

  const showStartupLoading = () => {
    const selectedUseCase = $("#startup_use_case option:selected").text() || "selected use case";
    $("body").addClass("startup-loading");
    setUseCaseLoading(
      true,
      "Loading selected use case...",
      `Preparing ${selectedUseCase}, model outputs, and reference context.`
    );
  };

  const normalizeManualDecimalInputs = (root = document) => {
    $(root)
      .find(".manual-decimal-input input")
      .addBack(".manual-decimal-input input")
      .each(function () {
        const normalized = String(this.value || "").replace(/,/g, ".");
        if (this.value !== normalized) {
          this.value = normalized;
          $(this).trigger("change");
        }
        this.setAttribute("inputmode", "decimal");
      });
  };

  normalizeManualDecimalInputs();

  $(document).on("input change blur", ".manual-decimal-input input", function () {
    const normalized = String(this.value || "").replace(/,/g, ".");
    if (this.value !== normalized) {
      this.value = normalized;
      $(this).trigger("change");
    }
  });

  new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) normalizeManualDecimalInputs(node);
      });
    });
  }).observe(document.body, { childList: true, subtree: true });

  $(document).on("click", "#confirm_switch", () => {
    setUseCaseLoading(true, "Loading use case...", "Preparing model outputs and reference context.");
  });

  $(document).on("click", "#confirm_startup_use_case", function () {
    window.setTimeout(() => {
      showStartupLoading();
      $(this).prop("disabled", true).text("Loading...");
    }, 0);
  });

  $(document).on("click", "#tab_prediction", () => {
    setUseCaseLoading(true, "Loading stratification support...", "Synchronizing inputs, scores, and reference context.");
  });

  $(document).on("click", "#data_input-btn_use_data", () => {
    setUseCaseLoading(true, "Confirming data...", "Preparing the confirmed set for downstream analysis.");
  });

  Shiny.addCustomMessageHandler("setUseCaseLoading", (payload) => {
    setUseCaseLoading(payload.visible, payload.title, payload.subtitle);
    if (!payload.visible) {
      $("body").removeClass("startup-loading");
      $("#confirm_startup_use_case").prop("disabled", false).text("Start");
    }
  });

  Shiny.addCustomMessageHandler("setStratificationTabLabels", (payload) => {
    const labels = {
      patient: payload.patient,
      cohort: payload.cohort,
      reference: payload.reference,
    };

    Object.entries(labels).forEach(([value, label]) => {
      if (!label) return;
      const currentText = {
        patient: ["Patient", "Candidate"],
        cohort: ["Cohort", "Candidate Set"],
        reference: ["Reference Patients", "Reference Candidates"],
      }[value] || [];

      $(".stratification-tabset a, .stratification-tabset button").each(function () {
        const $el = $(this);
        const dataValue = $el.attr("data-value") || $el.data("value");
        const text = $el.text().trim();
        if (dataValue === value || currentText.includes(text)) {
          $el.text(label);
        }
      });
    });
  });

  Shiny.addCustomMessageHandler("setAnalysisStepsLocked", (payload) => {
    const locked = Boolean(payload.locked);
    $("#tab_data_exploration, #tab_prediction")
      .toggleClass("workflow-step-locked", locked)
      .attr("aria-disabled", locked ? "true" : "false");
  });

  Shiny.addCustomMessageHandler("setDataInputMethodLabels", (payload) => {
    const fallbackText = {
      form: ["Manual Entry"],
      file: ["Upload File"],
      example: ["Example Cohort", "Example Candidate Set"],
    };

    Object.entries(payload || {}).forEach(([value, label]) => {
      if (!label) return;
      $(".data-input-method-tabs a, .data-input-method-tabs button").each(function () {
        const $el = $(this);
        const dataValue = $el.attr("data-value") || $el.data("value");
        const text = $el.text().trim();
        if (dataValue === value || (fallbackText[value] || []).includes(text)) {
          $el.text(label);
        }
      });
    });
  });

  Shiny.addCustomMessageHandler("toggleActiveTab", (payload) => {
    const activeTab = payload.activeTab;

    if (activeTab === currentTab) return;

    // Hide all
    $("#data-input-container").removeClass("main-visible");
    $("#inspect-input-container").removeClass("main-visible");
    $("#run-analysis-container").removeClass("main-visible");

    // Remove active-tab
    $("#tab_data_input").removeClass("active-tab");
    $("#tab_data_exploration").removeClass("active-tab");
    $("#tab_prediction").removeClass("active-tab");

    // Activate selected tab
    if (activeTab === "data_input") {
      $("#data-input-container").addClass("main-visible");
      $("#tab_data_input").addClass("active-tab");
    } else if (activeTab === "data_exploration") {
      $("#inspect-input-container").addClass("main-visible");
      $("#tab_data_exploration").addClass("active-tab");
    } else if (activeTab === "prediction") {
      $("#run-analysis-container").addClass("main-visible");
      $("#tab_prediction").addClass("active-tab");
    }

    currentTab = activeTab;
  });
});
