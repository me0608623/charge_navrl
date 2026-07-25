const pptxgen = require('pptxgenjs');
const html2pptx = require('./html2pptx');

(async () => {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'RL Navigation Lab';
  pptx.title = '讓 RL 使用 MOT 動態障礙運動資訊 — 實驗歷程';
  for (let i = 1; i <= 8; i++) {
    await html2pptx(`slides/slide${i}.html`, pptx);
    console.log(`slide ${i} ok`);
  }
  await pptx.writeFile({ fileName: '讓RL使用MOT_實驗歷程_進度報告.pptx' });
  console.log('DONE');
})().catch(e => { console.error(e); process.exit(1); });
