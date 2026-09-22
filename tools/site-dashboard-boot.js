(async()=>{
  'use strict';
  try{
    window.EWS_COMMON=await window.EWSWeb.read('common','전체 시각·정지 구간');
    window.EWSWeb.setShap(await window.EWSWeb.read('shap','저장된 실제 SHAP'));
    await import('./dashboard-main.js');
    await import('./dashboard-shap.js');
  }catch(error){window.EWSWeb.status(error.message,true);}
})();
