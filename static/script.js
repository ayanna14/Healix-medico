// Optional: add click listeners
document.querySelectorAll(".card button").forEach((btn, i) => {
  btn.addEventListener("click", () => {
    alert(`Clicked on card ${i + 1}`);
  });
});
