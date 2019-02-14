function s = dox_fix (s)
s = strrep (s, '<', '\<') ;
s = strrep (s, '>', '\>') ;
s = strrep (s, '#', '\#') ;
s = strrep (s, '&', '\&') ;
s = strrep (s, '"', '\"') ;
