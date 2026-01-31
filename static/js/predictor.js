// Simple helper (optional extra logic for UI)
function showSection(id) {
  document.querySelectorAll(".section").forEach(sec => sec.style.display = "none");
  const target = document.getElementById(id);
  if (target) target.style.display = "block";
}

document.addEventListener("DOMContentLoaded", () => {
  showSection("predictSection"); // default to prediction area
});
