const d40PaperVersions = {
  es: {
    label: 'Versión en español',
    title: 'Polinomios y volúmenes de Ehrhart exactos en alta dimensión mediante contracción tensorial cilíndrica y reconstrucción por el teorema chino del resto',
    path: 'BENCHMARK_D40/paper_tensor_crt_es.pdf',
    fileName: 'paper_tensor_crt_es.pdf',
    frameTitle: 'Paper en español, documento PDF',
  },
  en: {
    label: 'English version',
    title: 'Exact High-Dimensional Ehrhart Polynomials and Volumes via Cylindrical Tensor Contraction and Chinese Remainder Reconstruction',
    path: 'BENCHMARK_D40/paper_tensor_crt.pdf',
    fileName: 'paper_tensor_crt.pdf',
    frameTitle: 'Paper in English, PDF document',
  },
};

const d40Dialog = document.querySelector('#d40-paper-dialog');
const d40Frame = document.querySelector('#d40-paper-frame');
const d40CurrentVersion = document.querySelector('#d40-viewer-current');
const d40ViewerTitle = document.querySelector('#d40-viewer-title');
const d40OpenLink = document.querySelector('#d40-viewer-open');
const d40DownloadLink = document.querySelector('#d40-viewer-download');
const d40LanguageButtons = [
  ...document.querySelectorAll('[data-d40-viewer-language]'),
];

function selectD40Version(language) {
  const version = d40PaperVersions[language];
  if (!version) return;

  d40CurrentVersion.textContent = version.label;
  d40ViewerTitle.textContent = version.title;
  d40Frame.title = version.frameTitle;
  d40Frame.src = `${version.path}#view=FitH&toolbar=1&navpanes=0`;
  d40OpenLink.href = version.path;
  d40DownloadLink.href = version.path;
  d40DownloadLink.download = version.fileName;

  for (const button of d40LanguageButtons) {
    button.setAttribute(
      'aria-pressed',
      String(button.dataset.d40ViewerLanguage === language),
    );
  }
}

for (const button of document.querySelectorAll('[data-open-d40-viewer]')) {
  button.addEventListener('click', () => {
    selectD40Version(button.dataset.language || 'es');
    d40Dialog.showModal();
  });
}

for (const button of d40LanguageButtons) {
  button.addEventListener('click', () =>
    selectD40Version(button.dataset.d40ViewerLanguage),
  );
}

document
  .querySelector('[data-close-d40-viewer]')
  .addEventListener('click', () => d40Dialog.close());
d40Dialog.addEventListener('click', (event) => {
  if (event.target === d40Dialog) d40Dialog.close();
});
