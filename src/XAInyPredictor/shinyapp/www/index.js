$(() => {
  let currentTab = "data_input";  // default

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
