use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // ORT header: override with ORT_INCLUDE_DIR, otherwise third_party/ fetched
    // from scripts/fetch-deps.sh (per-target: linux-x64 or win-x64).
    let include_dir = match env::var("ORT_INCLUDE_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => {
            let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
            let platform = match target_os.as_str() {
                "windows" => "win-x64",
                "linux" => "linux-x64",
                other => panic!("unsupported platform: {other} (use ORT_INCLUDE_DIR)"),
            };
            manifest_dir
                .join("../../third_party/onnxruntime")
                .join(platform)
                .join("include")
        }
    };

    let c_api = include_dir.join("onnxruntime_c_api.h");
    let ep_api = include_dir.join("onnxruntime_ep_c_api.h");
    assert!(
        c_api.exists() && ep_api.exists(),
        "ORT headers not found in {} — run scripts/fetch-deps.sh or set ORT_INCLUDE_DIR",
        include_dir.display()
    );

    println!("cargo:rerun-if-env-changed=ORT_INCLUDE_DIR");
    println!("cargo:rerun-if-changed={}", c_api.display());
    println!("cargo:rerun-if-changed={}", ep_api.display());

    let bindings = bindgen::Builder::default()
        // onnxruntime_c_api.h in turn includes onnxruntime_ep_c_api.h (line ~8651);
        // the EP header is not standalone.
        .header(c_api.to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_item("Ort.*")
        .allowlist_item("ORT_.*")
        .allowlist_function("OrtGetApiBase")
        .allowlist_recursively(true)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .layout_tests(false)
        .generate()
        .expect("bindgen failed on the ORT headers");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("writing bindings.rs failed");
}
