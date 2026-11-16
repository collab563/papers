const cyclicPipelineVersions = {
  es: {
    label: 'Ficha en español · PDF disponible en inglés',
    title: 'Conteo exacto de planificaciones en tiempo lineal, muestreo sin sesgo y garantías SLA de Pitman–Chernoff para pipelines paralelos cíclicos',
    description: 'Un algoritmo exacto cuenta planificaciones válidas en tiempo lineal y genera muestras uniformes. Las cotas de Pitman–Chernoff y Hoeffding cuantifican la congestión hasta 500 000 tareas.',
    repository: 'Repositorio ↗',
    open: 'Abrir aparte ↗',
    download: 'Descargar ↓',
    frameTitle: 'Cuarto artículo, documento PDF en inglés',
  },
  en: {
    label: 'English metadata · English PDF',
    title: 'Exact Linear-Time Schedule Counting, Unbiased Sampling, and Pitman–Chernoff SLA Guarantees for Cyclic Parallel Pipelines',
    description: 'An exact transfer algorithm counts valid schedules in linear time and generates uniform samples. Pitman–Chernoff and Hoeffding bounds quantify congestion risk up to 500,000 tasks.',
    repository: 'Source repository ↗',
    open: 'Open separately ↗',
    download: 'Download ↓',
    frameTitle: 'Fourth paper, PDF document in English',
  },
};

const cyclicPipelineDialog = document.querySelector('#sla-paper-dialog');
const cyclicPipelineFrame = document.querySelector('#sla-paper-frame');
const cyclicPipelineTitle = document.querySelector('#sla-viewer-title');
const cyclicPipelineCurrent = document.querySelector('#sla-viewer-current');
const cyclicPipelineDescription = document.querySelector('#sla-viewer-description');
const cyclicPipelineOpen = document.querySelector('#sla-viewer-open');
const cyclicPipelineDownload = document.querySelector('#sla-viewer-download');
const cyclicPipelineRepository = cyclicPipelineDialog.querySelector('.repository-viewer-link');
const cyclicPipelineLanguageButtons = [...document.querySelectorAll('[data-sla-viewer-language]')];

function selectCyclicPipelineLanguage(language) {
  const version = cyclicPipelineVersions[language];
  if (!version) return;

  cyclicPipelineTitle.textContent = version.title;
  cyclicPipelineCurrent.textContent = version.label;
  cyclicPipelineDescription.textContent = version.description;
  cyclicPipelineFrame.title = version.frameTitle;
  cyclicPipelineRepository.textContent = version.repository;
  cyclicPipelineOpen.textContent = version.open;
  cyclicPipelineDownload.textContent = version.download;

  for (const button of cyclicPipelineLanguageButtons) {
    button.setAttribute('aria-pressed', String(button.dataset.slaViewerLanguage === language));
  }
}

for (const button of document.querySelectorAll('[data-open-sla-viewer]')) {
  button.addEventListener('click', () => {
    selectCyclicPipelineLanguage('es');
    cyclicPipelineDialog.showModal();
  });
}

for (const button of cyclicPipelineLanguageButtons) {
  button.addEventListener('click', () => selectCyclicPipelineLanguage(button.dataset.slaViewerLanguage));
}

document.querySelector('[data-close-sla-viewer]').addEventListener('click', () => cyclicPipelineDialog.close());
cyclicPipelineDialog.addEventListener('click', (event) => {
  if (event.target === cyclicPipelineDialog) cyclicPipelineDialog.close();
});
