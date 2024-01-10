typedef struct
{
    double stuff [4][4] ;
    char whatstuff [64] ;
}
wildtype ;                      // C version of wildtype

void wildadd (wildtype *z, wildtype *x, wildtype *y)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            z->stuff [i][j] = x->stuff [i][j] + y->stuff [i][j] ;
        }
    }
    const char *psrc = "this was added" ;
    char *pdst = z->whatstuff ;
    while ((*pdst++ = *psrc++)) ;
}
