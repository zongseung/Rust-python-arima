fn main() {
    cc::Build::new()
        .cpp(false)
        .include("lbfgsb_c")
        .file("lbfgsb_c/lbfgsb.c")
        .file("lbfgsb_c/linesearch.c")
        .file("lbfgsb_c/subalgorithms.c")
        .file("lbfgsb_c/print.c")
        .file("lbfgsb_c/linpack.c")
        .file("lbfgsb_c/miniCBLAS.c")
        .file("lbfgsb_c/timer.c")
        .flag_if_supported("-std=c11")
        .flag_if_supported("/std:c11")
        .compile("lbfgsb");
}
