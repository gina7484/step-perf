use prost_build::Config;
use std::io::Result;

fn main() -> Result<()> {
    // The protobuf definitions are a nested submodule, so Cargo otherwise misses
    // schema-only changes and can leave the Python and Rust vocabularies stale.
    for name in ["datatype.proto", "func.proto", "ops.proto", "graph.proto"] {
        println!("cargo:rerun-if-changed=step_perf_ir/proto/{name}");
    }
    Config::new()
        .out_dir(std::env::var("OUT_DIR").unwrap()) // Generated files go here
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "step_perf_ir/proto/datatype.proto",
                "step_perf_ir/proto/func.proto",
                "step_perf_ir/proto/ops.proto",
                "step_perf_ir/proto/graph.proto",
            ],
            &["step_perf_ir/proto/"],
        )
        .expect("Failed to compile Protobuf definitions");

    println!(
        "cargo:warning=OUT_DIR={}",
        std::env::var("OUT_DIR").unwrap()
    );

    Ok(())
}
