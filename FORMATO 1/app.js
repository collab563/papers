const paperVersions = {
  es: {
    label: 'Versión en español',
    title: 'Numeradores de Ehrhart exactos y raíces reales negativas en politopos de orden de coronas multicapa: redes de tensores y teoremas exactos para n = 3, 4',
    summary: 'Para n = 3, una transferencia de seis estados prueba 2k raíces negativas simples para k ≥ 2. Para n = 4, una transferencia de 20 estados y una certificación intervalar prueban 3k raíces negativas simples para cada politopo C(k,4), k ≥ 2; la semilla auxiliar k = 1 también está certificada. Los casos uniformes n ≥ 5 permanecen abiertos.',
    path: 'FORMATO%201/articulo_es.pdf',
    fileName: 'numeradores-coronas-multicapa-es.pdf',
    frameTitle: 'Paper en español, documento PDF',
  },
  en: {
    label: 'English version',
    title: 'Exact Ehrhart Numerators and Negative Real-Rootedness for Multilayer Crown Order Polytopes: Tensor Networks and Exact Theorems for n = 3, 4',
    summary: 'For n = 3, a six-state transfer proves 2k simple negative roots for k ≥ 2. For n = 4, a 20-state transfer and interval certification prove 3k simple negative roots for each order polytope C(k,4), k ≥ 2; the auxiliary k = 1 seed is also certified. The uniform cases n ≥ 5 remain open.',
    path: 'FORMATO%201/articulo.pdf',
    fileName: 'exact-crown-ehrhart-numerators-en.pdf',
    frameTitle: 'Paper in English, PDF document',
  },
};

const dialog = document.querySelector('#paper-dialog');
const frame = document.querySelector('#paper-frame');
const currentVersion = document.querySelector('#viewer-current');
const viewerTitle = document.querySelector('#viewer-title');
const viewerDescription = document.querySelector('#viewer-description');
const openLink = document.querySelector('#viewer-open');
const downloadLink = document.querySelector('#viewer-download');
const languageButtons = [...document.querySelectorAll('[data-viewer-language]')];

function selectVersion(language) {
  const version = paperVersions[language];
  if (!version) return;

  currentVersion.textContent = version.label;
  viewerTitle.textContent = version.title;
  viewerDescription.textContent = version.summary;
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

function selectedSliderPage() {
  return Number(pageSlider.value);
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
