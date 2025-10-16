document.addEventListener('DOMContentLoaded', () => {
  const interpolationSlider = document.getElementById('interpolation-slider');
  const interpolationLabel = document.getElementById('interpolation-label');
  if (interpolationSlider && interpolationLabel) {
    interpolationSlider.addEventListener('input', (event) => {
      interpolationLabel.textContent = `Frame ${event.target.value}`;
    });
  }
});
