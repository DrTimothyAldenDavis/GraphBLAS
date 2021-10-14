# SPDX-License-Identifier: Apache-2.0
# Generate test instances from a large tensor product set of options

def buildTest(ts="TestsuiteName",kern="vsvs", ds= "tiny-tiny", SR = "PLUS_TIMES",phase=3,
              typeC="int",typeM="int",typeA="int",typeB="int",type_x="int",type_y="int",type_z="int"):

    # build string interpolation from pieces
    Test_name = f"{ds}{SR}C{typeC}M{typeM}A{typeA}B{typeB}X{type_x}Y{type_y}Z{type_z}"

    Test_suite = ts
    #print(Test_suite)
    TEST_HEAD = f"""TEST( {Test_suite}, {Test_name})"""
    #print(TEST_HEAD)
    N = DataShapes[ds]['N']
    Anz = DataShapes[ds]['Anz']
    Bnz = DataShapes[ds]['Bnz']
    phase1_body= f""" test_AxB_phase1_factory< {typeC}, {typeM}, {typeA}, {typeB}>( 5, {N}, {Anz},{Bnz});"""
    phase2_body= f""" test_AxB_phase2_factory< {typeC} >( 5, {N}, {Anz},{Bnz});"""
    # phase2_end_body= f""" test_AxB_dot3_phase2end_factory< {typeC} >( 5, {N}, {Anz},{Bnz});"""
    # phase3_body = f""" test_AxB_dot3_{kern}_factory< {typeC},{typeM},{typeA},{typeB},{type_x},{type_y},{type_z} > (5, {N}, {Anz}, {Bnz}, SR);"""
    phasedict = { 1: phase1_body, 2: phase2_body }
    TEST_BODY= phasedict[phase]

    return TEST_HEAD,TEST_BODY

def load_types(argv):
    test_suite_name = argv[2]


    Monoids = argv[3].split(";")
    Binops  = argv[4].split(";")
    Semirings = argv[5]
    DataTypes = argv[6].split(";")

    # Hard-coding data shapes for now
    DataShapes ={
        "tinyxtiny": {'N':32, 'Anz':256, 'Bnz':128},
        "smallxsmall": {'N':1024, 'Anz': 65_536, 'Bnz':65_536}
        # "medxmed": {'N':4096, 'Anz': 2**20, 'Bnz':2**20}
        # "largexlarge": {'N':2**16, 'Anz': 64*2**20, 'Bnz':64*2**20}
    }

    Kernels= argv[7]

    return argv[1], test_suite_name, Monoids, Binops, Semirings, DataTypes, DataShapes, Kernels

def write_test_instances_header(test_suite_name, Monoids, Binops, Semirings, DataTypes, DataShapes, Kernels):
    outfile = f'{test_suite_name}_{Semirings}_{Kernels}_test_instances.hpp'
    with open(outfile, 'w') as fp:
        fp.write("#pragma once\n");
        Test_suite = f'{test_suite_name}_tests_{Semirings}_{Kernels}'
        for dtC in DataTypes:
            dtX = dtC
            dtY = dtC
            dtZ = dtC
            for dtM in ["bool", "int32_t"]:
                for dtA in DataTypes:
                    for dtB in DataTypes:
                        for ds in DataShapes:
                            for phase in [2]:

                                TEST_HEAD, TEST_BODY = buildTest( Test_suite, Kernels, ds, Semirings, phase,
                                                                  dtC, dtM, dtA, dtB, dtX, dtY, dtZ)
                                fp.write( TEST_HEAD)
                                fp.write( """{ std::string SR = "%s"; """%Semirings)
                                fp.write( TEST_BODY)
                                fp.write( "}\n")

def write_cuda_test(source_dir, test_suite_name, semiring, kernel):
    import shutil

    shutil.copy(f"{source_dir}/test/cuda_tests_template.cpp", f"{test_suite_name}_{semiring}_{kernel}_cuda_tests.cpp")

    with open(f"{test_suite_name}_{semiring}_{kernel}_cuda_tests.cpp", "a") as file_object:
        # Keeping this as a separate file for now to allow for further nesting
        # of test instances for each test_suite_name
        file_object.write(f"\n#include \"{test_suite_name}_{semiring}_{kernel}_test_instances.hpp\"")

if __name__ == "__main__":
    import sys

    if(len(sys.argv) != 8):
        raise ValueError("Expected 7 arguments but only got %s" % len(sys.argv))

    """
    First load values
    """
    source_dir, test_suite_name, Monoids, Binops, Semirings, DataTypes, DataShapes, Kernels = load_types(sys.argv)

    write_test_instances_header(test_suite_name, Monoids, Binops, Semirings, DataTypes, DataShapes, Kernels)

    write_cuda_test(source_dir, test_suite_name, Semirings, Kernels)
