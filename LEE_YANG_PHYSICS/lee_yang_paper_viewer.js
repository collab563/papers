const leeYangVersions = {
  es: {
    label: 'Versión en español',
    title: 'Ceros exactos de Lee–Yang, susceptibilidad autodual y escalamiento de la brecha de Lee–Yang en modelos cilíndricos SOS multicapa',
    path: 'LEE_YANG_PHYSICS/paper_lee_yang_physics_es.pdf',
    fileName: 'paper_lee_yang_physics_es.pdf',
    frameTitle: 'Tercer artículo, documento PDF en español',
  },
  en: {
    label: 'English version',
    title: 'Exact Lee–Yang Zeros, Self-Dual Susceptibility, and Lee–Yang Gap Scaling in Cylindrical Multilayer Solid-on-Solid Models',
    path: 'LEE_YANG_PHYSICS/paper_lee_yang_physics.pdf',
    fileName: 'paper_lee_yang_physics.pdf',
    frameTitle: 'Third paper, PDF document in English',
  },
};

const leeYangDialog = document.querySelector('#lee-yang-paper-dialog');
const leeYangFrame = document.querySelector('#lee-yang-paper-frame');
const leeYangTitle = document.querySelector('#lee-yang-viewer-title');
const leeYangCurrent = document.querySelector('#lee-yang-viewer-current');
const leeYangOpen = document.querySelector('#lee-yang-viewer-open');
const leeYangDownload = document.querySelector('#lee-yang-viewer-download');
const leeYangLanguageButtons = [...document.querySelectorAll('[data-lee-yang-language]')];

function selectLeeYangVersion(language) {
  const version = leeYangVersions[language];
  if (!version) return;

  leeYangTitle.textContent = version.title;
  leeYangCurrent.textContent = version.label;
  leeYangFrame.title = version.frameTitle;
  leeYangFrame.src = `${version.path}#view=FitH&toolbar=1&navpanes=0`;
  leeYangOpen.href = version.path;
  leeYangDownload.href = version.path;
  leeYangDownload.download = version.fileName;

  for (const button of leeYangLanguageButtons) {
    button.setAttribute('aria-pressed', String(button.dataset.leeYangLanguage === language));
  }
}

for (const button of document.querySelectorAll('[data-open-lee-yang-viewer]')) {
  button.addEventListener('click', () => {
    selectLeeYangVersion('es');
    leeYangDialog.showModal();
  });
}

for (const button of leeYangLanguageButtons) {
  button.addEventListener('click', () => selectLeeYangVersion(button.dataset.leeYangLanguage));
}

document.querySelector('[data-close-lee-yang-viewer]').addEventListener('click', () => leeYangDialog.close());
leeYangDialog.addEventListener('click', (event) => {
  if (event.target === leeYangDialog) leeYangDialog.close();
});
