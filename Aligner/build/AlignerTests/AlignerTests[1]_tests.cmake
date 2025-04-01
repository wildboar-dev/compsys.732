add_test([=[Example_Test.test_example]=]  /home/trevor/demo/Aligner/build/AlignerTests/AlignerTests [==[--gtest_filter=Example_Test.test_example]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Example_Test.test_example]=]  PROPERTIES WORKING_DIRECTORY /home/trevor/demo/Aligner/build/AlignerTests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  AlignerTests_TESTS Example_Test.test_example)
