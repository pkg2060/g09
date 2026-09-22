(() => {
  'use strict';
  let loading=null;
  async function loadCatalog(){
    if(loading)return loading;
    loading=(async()=>{
      try{
        window.EWSWebCatalog=await window.EWSWeb.read('catalog','전체 실험·코드 기록');
        await import('./summary-catalog.js');
      }catch(error){loading=null;window.EWSWeb.status(error.message,true);}
    })();
    return loading;
  }
  for(const detail of document.querySelectorAll('#offline-catalog > details')){
    detail.addEventListener('toggle',()=>{if(detail.open)loadCatalog();});
    if(detail.open)loadCatalog();
  }
})();
