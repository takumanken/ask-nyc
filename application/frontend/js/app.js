/**
 * Main application for NYC data explorer
 */
import apiService from "./apiService.js";
import visualization from "./visualization.js";
import initializeEventListeners from "./eventHandlers.js";
import { initializeLocationCheckbox } from "./locationService.js";
import { state } from "./state.js";

// Initialize app when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  initializeEventListeners();
  initializeLocationCheckbox();
  processInitialQuery();
});

// Process query passed from landing page via localStorage
function processInitialQuery() {
  const initialQuery = localStorage.getItem("initialQuery");

  if (initialQuery) {
    localStorage.removeItem("initialQuery");
    document.getElementById("promptInput").value = initialQuery;
    handleUserQuery(initialQuery);
  }
}

// Process user query and update visualization
export async function handleUserQuery(query) {
  const userQuery = query || document.getElementById("promptInput").value;
  if (!userQuery.trim()) return;

  try {
    const result = await processQueryWithLocation(userQuery);

    // Only visualize data if query was successful
    if (result !== false) {
      visualization();
    }
  } catch (error) {
    console.error("Error processing query:", error);
  }
}

// Make handleUserQuery available globally for the error handler
window.handleUserQuery = handleUserQuery;

// Add location data to query when enabled by user
async function processQueryWithLocation(query) {
  const locationData = await state.getOrFetchLocation();
  return apiService(query, locationData);
}
