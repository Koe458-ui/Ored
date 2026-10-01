(function(){
  'use strict';

    // The hero was four tabs from a data table, now one static block — only the two buttons' navigation is left here

  function go(to){
    var path = typeof window.dzRoutePath === 'function' ? window.dzRoutePath(to) : null;
    if(path && typeof window.dzRouteGo === 'function' && window.dzRouteGo(path)) return;
    if(typeof openFG === 'function'){
      openFG();
      if(typeof fgSwitchSection === 'function') fgSwitchSection(to);
    }
  }

  function hpGo(){ go('artworks'); }

  function hpJoin(){
    if(typeof window.dzRouteTab === 'function') window.dzRouteTab('/community');
  }

  window.hpGo   = hpGo;
  window.hpJoin = hpJoin;
})();
