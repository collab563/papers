const treewidthPaperVersions = {
  es: {
    label: 'Ficha en español · PDF disponible en inglés',
    title: 'Polinomios y volúmenes de Ehrhart exactos de politopos de orden de alta dimensión mediante descomposiciones nice y CRT',
    description: 'Un algoritmo de tensores sobre descomposiciones de árbol reconstruye exactamente los polinomios de Ehrhart y los volúmenes racionales. Incluye 12 instancias de hasta dimensión 40.',
    repository: 'Repositorio ↗',
    open: 'Abrir aparte ↗',
    download: 'Descargar ↓',
    frameTitle: 'Quinto artículo, documento PDF en inglés',
  },
  en: {
    label: 'English metadata · English PDF',
    title: 'Exact High-Dimensional Ehrhart Polynomials and Volumes of Bounded-Treewidth Order Polytopes via Nice Tree Decompositions and CRT',
    description: 'A tensor dynamic program on nice tree decompositions reconstructs exact Ehrhart polynomials and rational volumes. The benchmark covers 12 instances through dimension 40.',
    repository: 'Source repository ↗',
    open: 'Open separately ↗',
    download: 'Download ↓',
    frameTitle: 'Fifth paper, PDF document in English',
  },
};

const treewidthPaperDialog = document.querySelector('#treewidth-paper-dialog');
const treewidthPaperFrame = document.querySelector('#treewidth-paper-frame');
const treewidthPaperTitle = document.querySelector('#treewidth-viewer-title');
const treewidthPaperCurrent = document.querySelector('#treewidth-viewer-current');
const treewidthPaperDescription = document.querySelector('#treewidth-viewer-description');
const treewidthPaperOpen = document.querySelector('#treewidth-viewer-open');
const treewidthPaperDownload = document.querySelector('#treewidth-viewer-download');
const treewidthPaperRepository = treewidthPaperDialog.querySelector('.repository-viewer-link');
const treewidthPaperLanguageButtons = [...document.querySelectorAll('[data-treewidth-viewer-language]')];

function selectTreewidthPaperLanguage(language) {
  const version = treewidthPaperVersions[language];
  if (!version) return;

  treewidthPaperTitle.textContent = version.title;
  treewidthPaperCurrent.textContent = version.label;
  treewidthPaperDescription.textContent = version.description;
  treewidthPaperFrame.title = version.frameTitle;
  treewidthPaperRepository.textContent = version.repository;
  treewidthPaperOpen.textContent = version.open;
  treewidthPaperDownload.textContent = version.download;

  for (const button of treewidthPaperLanguageButtons) {
    button.setAttribute('aria-pressed', String(button.dataset.treewidthViewerLanguage === language));
  }
}

for (const button of document.querySelectorAll('[data-open-treewidth-viewer]')) {
  button.addEventListener('click', () => {
    selectTreewidthPaperLanguage('es');
    treewidthPaperDialog.showModal();
  });
}

for (const button of treewidthPaperLanguageButtons) {
  button.addEventListener('click', () => {
    selectTreewidthPaperLanguage(button.dataset.treewidthViewerLanguage);
  });
}

document.querySelector('[data-close-treewidth-viewer]').addEventListener('click', () => {
  treewidthPaperDialog.close();
});
treewidthPaperDialog.addEventListener('click', (event) => {
  if (event.target === treewidthPaperDialog) treewidthPaperDialog.close();
});
