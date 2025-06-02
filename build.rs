use prost_build::Config;
use std::io::Result;

fn main() -> Result<()> {
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
