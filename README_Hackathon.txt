Instructions for the upcoming GraphBLAS Hackathon.

There is one parallel loop in the code in which each thread allocates memory
via malloc or realloc.  See the "reduction(&&:ok)" and comments in
GraphBLAS/Source/GB_AxB_saxpy_parallel.c.  For all other cases, memory is never
malloc/calloc/realloc/freed inside a parallel region.

Please follow my coding style for the Hackathon.  If you break these rules,
I'll need to go back over your modifications to the code to fix these stylistic
issues.

    * use lots of comments

    * follow my indentation style: indent each level by 4

    * use "//"-style comments (The code is not C++, but these are now valid in
        ANSI C11).  Exception: /* comments */ must be used inside long macros.

    * do not exceed 80 characters per line (hey, I started life on
        punch cards :-)

    * do not insert tabs; always use spaces.  Since the beginning of time,
        tabs are 8 spaces, but some IDEs insist on 4-space tabs.  Viewing code
        with an IDE that uses one rule with code written with an IDE that used
        another rule leads to ugly code.

    * do not put spurious blank space at the end of a line, like this line. 
        (with one exception).  I use vim, and the blank spaces causes my
        touch-typing for vim to break, particularly a blank line with explicit
        spaces.

        Exception to this rule:  you will find single spaces at the end of a
        line containing just a left curly bracket.  I use this for my test
        coverage mechanism in GraphBLAS/Tcov.  I have to do the coverage inside
        a MATLAB mexFunction, so I use my own code preprocessing mechanism, and
        "{ " with a space triggers the insertion of a code coverage statement.
        If you find such spaces, do not delete them!  They look like this:

            if (something)
            { 
                note the space after the curly bracket above
            }

    * always use curly brackets like this:

            if (something)
            {
                do stuff ;
            }
            else
            {
                something else
            }

        This is OK:

            if (something) do something ;

        but this is not ok

            if (something) do a very long thing that goes over a single line so
                that it is hard to read, use a pair of curly brackets instead ;

        but never do this

            if (something) do something ;
            else do something else ;

        Never do this

            if (something) {
                this is broken
            }

        Except for initializers, such as the following,
        curly brackets belong on their own line

            GrB_Matrix A_slice [2] = { NULL, NULL } ;

        I make a few exceptions to this rule: one line inline functions.

    * always use a space before a semicolon.  Only exception to this rule is
        if it allows an 81-character line to fit down to 80.

    * always use a space before and after a colon.  This includes case
        statements and ternaries.

    * for the "?" operator, always put a space before and after

    * I typically use ternaries like this:

            ((rule) ? a : b)

        or this, if a and b are expressions

            ((rule) ? (a*2) : (b+1))

        Never do

            rule?a:b

        this is ok as long as "rule" is single token:

            rule ? a : b

        Never do

            x == 0 ? a : b

        instead use

            (x == 0) ? a : b

        note the parantheses, and the space before and after the colon
        and question mark

    * always use a space after a comma

    * always use a space before a left paranthesis.

    * always follow a right paranthesis with a space, newline, or comma

    * use a space before and after an equal sign

        int likethis = 42 ;
        bool notlikethis=false ;        // do not to do this

