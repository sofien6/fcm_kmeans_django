// Main JavaScript for the segmentation app

document.addEventListener("DOMContentLoaded", function () {
  // Get the segmentation form
  const segmentationForm = document.getElementById("segmentation-form");

  // Create and append the spinner container to the body
  const spinnerContainer = document.createElement("div");
  spinnerContainer.className = "spinner-container";
  spinnerContainer.innerHTML = `
        <div class="spinner-content">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Processing image... This may take a moment.</p>
        </div>
    `;
  document.body.appendChild(spinnerContainer);

  // Add submit event listener to the form
  if (segmentationForm) {
    segmentationForm.addEventListener("submit", function () {
      // Show the spinner
      spinnerContainer.style.display = "flex";

      // Disable the submit button to prevent multiple submissions
      const submitButton = segmentationForm.querySelector(
        'button[type="submit"]'
      );
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML =
          '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
      }
    });
  }
});
