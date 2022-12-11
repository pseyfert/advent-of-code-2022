extern crate cbindgen;

use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut config = cbindgen::Config {
        pragma_once: true,
        usize_is_size_t: true,
        namespace: Some("aoc_ffi_ocr".to_owned()),
        ..Default::default()
    };
    config.parse.parse_deps = true;

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("aoc_ffi_ocr.h");
}
