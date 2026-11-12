const paperVersions = {
  es: {
    label: 'Versión en español',
    title: 'Verificación computacional y análisis de redes de tensores 2D',
    path: 'FORMATO/articulo_es.pdf',
    fileName: 'paper-coronas-multicapa-es.pdf',
    frameTitle: 'Paper en español, documento PDF',
  },
  en: {
    label: 'English version',
    title: 'Computational Verification and 2D Tensor Network Analysis',
    path: 'FORMATO/aticulo.pdf',
    fileName: 'paper-multilayer-crowns-en.pdf',
    frameTitle: 'Paper in English, PDF document',
  },
};

const dialog = document.querySelector('#paper-dialog');
const frame = document.querySelector('#paper-frame');
const currentVersion = document.querySelector('#viewer-current');
const viewerTitle = document.querySelector('#viewer-title');
const openLink = document.querySelector('#viewer-open');
const downloadLink = document.querySelector('#viewer-download');
const languageButtons = [...document.querySelectorAll('[data-viewer-language]')];

function selectVersion(language) {
  const version = paperVersions[language];
  if (!version) return;

  currentVersion.textContent = version.label;
  viewerTitle.textContent = version.title;
  frame.title = version.frameTitle;
  frame.src = `${version.path}#view=FitH&toolbar=1&navpanes=0`;
  openLink.href = version.path;
  downloadLink.href = version.path;
  downloadLink.download = version.fileName;

  for (const button of languageButtons) {
    button.setAttribute('aria-pressed', String(button.dataset.viewerLanguage === language));
  }
}

for (const button of document.querySelectorAll('[data-open-viewer]')) {
  button.addEventListener('click', () => {
    selectVersion(button.dataset.language || 'es');
    dialog.showModal();
  });
}

for (const button of languageButtons) {
  button.addEventListener('click', () => selectVersion(button.dataset.viewerLanguage));
}

document.querySelector('[data-close-viewer]').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => {
  if (event.target === dialog) dialog.close();
});

const searchInput = document.querySelector('#paper-search');
const paperItems = [...document.querySelectorAll('[data-paper]')];
const emptyState = document.querySelector('#empty-state');

function normalizeText(value) {
  return value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
}

searchInput.addEventListener('input', () => {
  const query = normalizeText(searchInput.value.trim());
  let visibleCount = 0;

  for (const item of paperItems) {
    const matches = normalizeText(item.dataset.search).includes(query);
    item.hidden = !matches;
    if (matches) visibleCount += 1;
  }

  emptyState.hidden = visibleCount > 0;
});
