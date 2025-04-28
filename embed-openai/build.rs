fn main() {
    // Invalidate the build when these WIT files change
    println!("cargo:rerun-if-changed=wit/embed.wit");
    println!(
        "cargo:rerun-if-changed=../wit/deps/golem-durability/golem-durability.wit"
    );

    // 1) Generate the `embed` WIT bindings (async)
    wit_bindgen::generate!({
        path: "wit/embed.wit",
        world: "embed",
        async: true,
    });

    // 2) Import the host durability interface
    wit_bindgen::generate!({
        path: "../wit/deps/golem-durability/golem-durability-1.2.wit",
        world: "durability",
    });
}
