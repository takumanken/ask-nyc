import { state } from "../state.js";
import { marked } from "https://cdn.jsdelivr.net/npm/marked@5.0.2/lib/marked.esm.min.js";

function renderTextResponse(container) {
  // Clear container
  container.innerHTML = "";

  // Create styled container for the text
  const textContainer = document.createElement("div");
  textContainer.id = "text-response";
  textContainer.classList.add("markdown-content");
  textContainer.style.lineHeight = "1.5";
  textContainer.style.maxWidth = "100%";
  textContainer.style.overflowWrap = "break-word";

  // Convert markdown to HTML and set it
  if (state.textResponse) {
    // Use marked to convert markdown to HTML
    const htmlContent = marked.parse(state.textResponse);
    textContainer.innerHTML = htmlContent;
  } else {
    textContainer.textContent = "";
  }

  // Append to the main container
  container.appendChild(textContainer);
}

// Add some basic styles for markdown elements
function addMarkdownStyles() {
  const styleEl = document.createElement("style");
  styleEl.textContent = `
    .markdown-content p {
      margin-bottom: 1em;
    }
    
    .markdown-content ul, .markdown-content ol {
      margin-bottom: 1em;
      padding-left: 2em;
    }
    
    .markdown-content li {
      margin-bottom: 0.5em;
    }
    
  `;
  document.head.appendChild(styleEl);
}

// Call this function once when the module is loaded
addMarkdownStyles();

export default renderTextResponse;
