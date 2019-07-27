
clear

optype = {
'logical'
'int8'
'int16'
'int32'
'int64'
'uint8'
'uint16'
'uint32'
'uint64'
'single'
'double' } ;

% TODO: 'complex' 

opnames = {
    '1st'
    'first'
    '2nd'
    'second'
    'min'
    'max'
    '+'
    'plus'
    '-'
    'minus'
    'rminus'
    '*'
    'times'
    '/'
    'div'
    '\'
    'rdiv'
    'iseq'
    'isne'
    'isgt'
    'islt'
    'isge'
    'isle'
    '=='
    '='
    'eq'
    '!='
    '~='
    'ne'
    '>'
    'gt'
    '<'
    'lt'
    '>='
    'ge'
    '<='
    'le'
    '||'
    '|'
    'or'
    'lor'
    '&&'
    '&'
    'and'
    'land' } ;


X = 1 ;

for k1 = 1:length(opnames)

    opname = opnames {k1} ;
    fprintf ('\n=================================== %s\n', opname) ;

    for k2 = 0:length(optype)

        op = opname ;
        if (k2 > 0)
            op = [op '.' optype{k2}] ;
        end
        fprintf ('\nop: [%s]\n', op) ;

        % usage: A = gbbuild (I, J, X, m, n, dup, type)
        try
            gbbuild ([ ], [ ], X, 1, 1, op) ;
        catch me
            me
            pause
        end
    end
    pause

end
