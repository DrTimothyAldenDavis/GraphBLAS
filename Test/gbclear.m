try
    GrB.finalize
    clear all
    GrB.init
catch me
    me
end
